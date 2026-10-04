"""Calibrated mode through the real pipeline: the assessor, the bundle registry and its gate, scoring,
replay, bundle persistence and the server configuration."""

import json
from datetime import UTC, datetime, timedelta

import numpy as np
import pytest

from assay.api.main import load_registry
from assay.detection import load_bundle, save_bundle
from assay.ingestion import IngestionConfig, IngestionService, InMemoryRepository
from assay.scoring import BundleRegistry, ScoringConfig, ScoringError, ScoringService
from assay.synthetic import START, to_wire
from assay.trust import TrustState
from assay.trust.assessor import TrustAssessor
from assay.trust.calibrated import CalibratedConfig, CalibratedTrustModel
from calibrated_helpers import make_cases

T = "tenant-synth"


@pytest.fixture(scope="module")
def meta():
    """A meta-model fitted on synthetic cases, with the gate recorded as passed. This tests the plumbing,
    not the quality of the fit (that is test_calibrated.py and the validation experiments)."""
    fit, yf, _ = make_cases(4000, 1, exp_weight=0.5)
    cal, yc, _ = make_cases(2000, 2, exp_weight=0.5)
    m = CalibratedTrustModel.fit(fit, yf, cal, yc, CalibratedConfig(n_boot=10))
    m.gate = {"passed": True, "checked_at": datetime.now(UTC).isoformat()}
    return m


@pytest.fixture(scope="module")
def parts(trained, tmp_path_factory):
    d = tmp_path_factory.mktemp("calint") / "b"
    save_bundle(d, trained[3].scoring_bundle(), trained[3].manifest, b"k")
    return load_bundle(d, T, b"k")


def with_meta(artefact, m):
    return {**artefact, "trust_meta": m}


def cases_from(trained, n=60):
    _, txns, _, r = trained
    by_id, ids = {t["txn_id"]: t for t in txns}, r.test_table.txn_ids
    return r, by_id, ids


# -- the assessor ----------------------------------------------------------------------------------------
def assess_both(trained, meta, n=120):
    r, by_id, ids = cases_from(trained)
    idx = np.arange(0, n * 12, 12)
    X = r.test_table.X[idx]
    txns = [by_id[ids[i]] for i in idx]
    pred = r.model.predict(X, r.manifest.thresholds["t_high"])
    kw = {"model_version": r.manifest.bundle_id}
    prov = TrustAssessor(r.model, r.reference, r.store, r.manifest.thresholds, **kw).assess(
        X, txns, pred, drift_vector=np.zeros(X.shape[1]))
    cal = TrustAssessor(r.model, r.reference, r.store, r.manifest.thresholds, calibrated=meta, **kw).assess(
        X, txns, pred, drift_vector=np.zeros(X.shape[1]))
    return prov, cal


def test_calibrated_assessments_replace_the_score_and_keep_the_provisional_one(trained, meta):
    prov, cal = assess_both(trained, meta)
    scored = [(p, c) for p, c in zip(prov, cal, strict=True) if p.result.ti is not None]
    assert scored, "the sample must contain scoreable cases"
    for p, c in scored:
        assert c.result.mode == "calibrated" and p.result.mode == "provisional"
        assert c.provisional is not None and c.provisional.ti == p.result.ti      # kept for lineage
        assert 0 <= c.result.ti_low <= c.result.ti <= c.result.ti_high <= 100
        assert c.result.state in (TrustState.HIGH, TrustState.MODERATE, TrustState.LOW)
        assert c.result.config_version.endswith("+calibrated-0")


def test_cases_with_no_provisional_score_are_left_exactly_as_they_were(trained, meta):
    prov, cal = assess_both(trained, meta, n=200)
    insufficient = [(p, c) for p, c in zip(prov, cal, strict=True) if p.result.ti is None]
    assert insufficient, "the sample must contain Insufficient-evidence cases"
    for p, c in insufficient:
        assert c.result.state is TrustState.INSUFFICIENT and c.result.ti is None
        assert c.result.reason_codes == p.result.reason_codes and c.result.mode == "provisional"


def test_the_components_are_untouched_by_calibrated_mode(trained, meta):
    prov, cal = assess_both(trained, meta)
    for p, c in zip(prov, cal, strict=True):
        assert {k: v.score for k, v in p.components.items()} == {k: v.score for k, v in c.components.items()}


def test_the_assessor_refuses_a_model_that_has_not_passed_the_gate(trained, meta):
    r = trained[3]
    fresh = CalibratedTrustModel(meta.cfg, meta.fits, meta.boots, meta.n_fit, meta.n_calibration, meta.coefficients, gate=None)
    for gate in (None, {"passed": False}):
        fresh.gate = gate
        with pytest.raises(ValueError, match="section 6 gate"):
            TrustAssessor(r.model, r.reference, r.store, r.manifest.thresholds,
                          model_version="b", calibrated=fresh)


# -- the registry and its gate ---------------------------------------------------------------------------
def test_the_registry_enforces_the_gate_in_code(parts, meta):
    art, manifest = parts
    reg = BundleRegistry()
    reg.register(T, art, manifest)                                                   # provisional is unchanged
    with pytest.raises(ScoringError, match="has none"):
        BundleRegistry().register(T, art, manifest, trust_mode="calibrated")
    failed = CalibratedTrustModel(meta.cfg, meta.fits, meta.boots, meta.n_fit, meta.n_calibration,
                                  meta.coefficients, gate={"passed": False})
    with pytest.raises(ScoringError, match="not passed the section 6 gate"):
        BundleRegistry().register(T, with_meta(art, failed), manifest, trust_mode="calibrated")
    with pytest.raises(ScoringError, match="unknown trust mode"):
        BundleRegistry().register(T, art, manifest, trust_mode="magic")
    ok = BundleRegistry().register(T, with_meta(art, meta), manifest, trust_mode="calibrated")
    assert ok.assessor.calibrated is meta


def test_a_bundle_that_carries_a_meta_model_is_still_provisional_unless_asked(parts, meta):
    art, manifest = parts
    lb = BundleRegistry().register(T, with_meta(art, meta), manifest)               # no trust_mode given
    assert lb.assessor.calibrated is None


# -- scoring, lineage, replay --------------------------------------------------------------------------
@pytest.fixture
def scoring(trained, parts, meta):
    _, by_id, ids = cases_from(trained)
    art, manifest = parts
    repo, clock = InMemoryRepository(), {"t": START}
    reg = BundleRegistry()
    reg.register(T, with_meta(art, meta), manifest, trust_mode="calibrated")
    svc = ScoringService(repo, reg, ScoringConfig(clock=lambda: clock["t"]))

    def score(i):
        t = to_wire(by_id[ids[i]])
        clock["t"] = datetime.fromisoformat(t["event_time"]) + timedelta(seconds=2)
        IngestionService(repo, IngestionConfig(clock=lambda: clock["t"])).ingest_transaction(T, t, source_id="s", actor="t")
        return svc.score_transaction(T, t["txn_id"], actor="t")

    return svc, repo, score


def test_decisions_are_scored_in_calibrated_mode_and_the_lineage_records_both_scores(scoring):
    _, repo, score = scoring
    decisions = [score(i) for i in range(0, 240, 12)]
    scored = [d for d in decisions if d["trust"]["state"] != "insufficient_evidence"]
    assert scored and all(d["trust"]["mode"] == "calibrated" for d in scored)
    assert all(0 <= d["trust"]["ti_low"] <= d["trust"]["ti"] <= d["trust"]["ti_high"] <= 100 for d in scored)
    for d in decisions:
        ta = repo.get_by_id(T, "trust_assessments", repo.get_by_id(T, "policy_decisions", d["decision_id"])["trust_assessment_id"])
        if d["trust"]["state"] == "insufficient_evidence":
            assert ta["mode"] == "provisional" and ta["ti"] is None
        else:
            assert ta["mode"] == "calibrated" and ta["evidence"]["provisional_ti"] is not None
            assert "intercept" in ta["weights_used"] and ta["weights_version"].endswith("+calibrated-0")


def test_a_calibrated_decision_replays_to_the_same_result(scoring):
    svc, _, score = scoring
    d = next(x for x in (score(i) for i in range(0, 240, 12)) if x["trust"]["state"] != "insufficient_evidence")
    rep = svc.replay(T, d["decision_id"])
    assert rep["reproducible"] is True, rep


def test_the_policy_still_runs_on_the_calibrated_state(scoring):
    _, _, score = scoring
    decisions = [score(i) for i in range(0, 240, 12)]
    assert all(d["recommendation"] and d["automation_level"] == 0 for d in decisions)   # recommend-only, as ever


# -- persistence and configuration ----------------------------------------------------------------------
def test_a_saved_bundle_keeps_its_meta_model_and_its_gate(parts, meta, tmp_path):
    art, manifest = parts
    d = save_bundle(tmp_path / "b", with_meta(art, meta), manifest, b"k")
    loaded, _ = load_bundle(d, T, b"k")
    again = loaded["trust_meta"]
    assert again.enabled and again.gate["passed"] is True
    cases, _, _ = make_cases(50, 77, exp_weight=0.5)
    assert again.predict(cases) == meta.predict(cases)                  # the same answers after the round trip


def test_the_server_configuration_selects_the_mode_per_tenant(parts, meta, tmp_path):
    art, manifest = parts
    good = save_bundle(tmp_path / "good", with_meta(art, meta), manifest, b"k")
    plain = save_bundle(tmp_path / "plain", art, manifest, b"k")
    reg = load_registry([{"tenant_id": T, "path": str(good), "trust_mode": "calibrated"}], b"k")
    assert reg.champion(T).assessor.calibrated is not None
    assert load_registry([{"tenant_id": T, "path": str(good)}], b"k").champion(T).assessor.calibrated is None
    with pytest.raises(ScoringError, match="has none"):                  # asking for calibrated without a model stops start-up
        load_registry([{"tenant_id": T, "path": str(plain), "trust_mode": "calibrated"}], b"k")


def test_json_of_a_calibrated_result_survives_the_api_view(scoring):
    _, _, score = scoring
    d = next(x for x in (score(i) for i in range(0, 240, 12)) if x["trust"]["state"] != "insufficient_evidence")
    assert json.loads(json.dumps(d))["trust"]["mode"] == "calibrated"
