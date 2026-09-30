from collections import Counter

import numpy as np
import pytest

from assay.trust import ReasonCode, Status, TrustConfig, TrustState
from assay.trust.assessor import TrustAssessor

NOVEL = "novel_open_banking_scam"


@pytest.fixture(scope="module")
def setup(trained):
    ds, txns, _, r = trained
    by_id = {t["txn_id"]: t for t in txns}
    ids = r.test_table.txn_ids
    rng = np.random.default_rng(1)
    novel = [i for i, t in enumerate(ids) if ds.truth[t]["fraud_type"] == NOVEL]
    others = rng.choice([i for i in range(len(ids)) if i not in set(novel)], 220, replace=False)
    sample = np.sort(np.concatenate([novel, others]))
    X = r.test_table.X[sample]
    tx = [by_id[ids[i]] for i in sample]
    t_high = r.manifest.thresholds["t_high"]
    pred = r.model.predict(X, t_high)
    assessor = TrustAssessor(r.model, r.reference, r.store, r.manifest.thresholds,
                             model_version=r.manifest.bundle_id)
    return {"ds": ds, "r": r, "X": X, "tx": tx, "pred": pred, "assessor": assessor,
            "y": r.test_y[sample], "truth": [ds.truth[t["txn_id"]] for t in tx]}


@pytest.fixture(scope="module")
def assessed(setup):
    s = setup
    return s["assessor"].assess(s["X"], s["tx"], s["pred"])


def test_components_intervals_and_ordering_hold_for_every_case(assessed):
    for a in assessed:
        for name in ("conf", "rel", "fam", "dq", "exp", "drift"):
            c = a.components[name]
            if c.status is Status.ACTIVE:
                assert 0 <= c.lo <= c.score <= c.hi <= 1 or c.lo <= c.score <= c.hi, (name, c)
        assert a.components["hum"].status is Status.INACTIVE  # MVP: no verified analyst history
        r = a.result
        if r.ti is not None:
            assert r.ti_low <= r.ti <= r.ti_high and 0 <= r.ti_low and r.ti_high <= 100
            assert r.state in (TrustState.HIGH, TrustState.MODERATE, TrustState.LOW)
        else:
            assert r.state is TrustState.INSUFFICIENT and r.reason_codes


def test_weights_renormalise_over_the_active_components(assessed):
    w = assessed[0].result.weights_used
    assert "hum" not in w and sum(w.values()) == pytest.approx(1.0)


def test_novel_fraud_gets_insufficient_evidence_with_unfamiliar_pattern(setup, assessed):
    """PRD 6.5 held-out fraud type: familiarity falls and cases move to Insufficient evidence."""
    idx = [i for i, t in enumerate(setup["truth"]) if t["fraud_type"] == NOVEL]
    assert len(idx) >= 10
    states = Counter(assessed[i].result.state for i in idx)
    assert states[TrustState.INSUFFICIENT] >= 0.9 * len(idx), states
    assert all(ReasonCode.UNFAMILIAR_PATTERN in assessed[i].result.reason_codes
               for i in idx if assessed[i].result.state is TrustState.INSUFFICIENT)
    # ...while ordinary traffic mostly receives a score
    rest = [a for a, t in zip(assessed, setup["truth"], strict=True) if t["fraud_type"] != NOVEL]
    share_scored = np.mean([a.result.ti is not None for a in rest])
    assert share_scored > 0.8


def test_unfamiliar_cases_are_a_small_share_of_ordinary_traffic(setup, assessed):
    """Guards the non-stationary-feature bug: ordinary traffic must not look novel."""
    rest = [a for a, t in zip(assessed, setup["truth"], strict=True) if t["fraud_type"] != NOVEL]
    assert np.mean([a.novelty for a in rest]) < 0.7
    assert np.mean([ReasonCode.UNFAMILIAR_PATTERN in a.result.reason_codes for a in rest]) < 0.1


def test_assessment_is_deterministic(setup, assessed):
    again = setup["assessor"].assess(setup["X"][:30], setup["tx"][:30], _slice(setup["pred"], 30))
    for a, b in zip(assessed[:30], again, strict=True):
        assert a.result == b.result and a.components == b.components


def _slice(p, n):
    from assay.detection.ensemble import Predictions
    return Predictions(p.raw[:n], p.calibrated[:n], p.member_spread[:n],
                       None if p.distance_to_threshold is None else p.distance_to_threshold[:n])


def test_missing_explanation_caps_trust_below_high(setup):
    """PRD 5.4: a missing exp caps TI_low (about 48 with hum inactive), so no case can be High."""
    s = setup
    a = s["assessor"].assess(s["X"][:80], s["tx"][:80], _slice(s["pred"], 80),
                             explain_mask=np.zeros(80, bool))
    assert all(x.components["exp"].status is Status.MISSING for x in a)
    # drift needs only the (cheap) attributions, so it stays available without explanation testing
    assert all(x.components["drift"].status is Status.ACTIVE for x in a)
    scored = [x.result for x in a if x.result.ti is not None]
    assert scored and all(r.state is not TrustState.HIGH for r in scored)
    assert max(r.ti_low for r in scored) < 50


def test_drift_on_the_features_driving_a_case_lowers_its_drift_stability(setup):
    s = setup
    n = 60
    base = s["assessor"].assess(s["X"][:n], s["tx"][:n], _slice(s["pred"], n), drift_vector=np.zeros(s["X"].shape[1]))
    drifted = s["assessor"].assess(s["X"][:n], s["tx"][:n], _slice(s["pred"], n),
                                   drift_vector=np.full(s["X"].shape[1], 0.6))
    b = np.array([a.components["drift"].score for a in base])
    d = np.array([a.components["drift"].score for a in drifted])
    assert np.all(b == 1.0) and np.all(d < 1.0) and d.mean() < 0.5


def test_degraded_data_triggers_the_data_quality_floor(setup):
    s = setup
    broken = [{**t, "amount": -5.0, "device_hash": None, "ip_hash": None, "country": None,
               "merchant_id": None} for t in s["tx"][:20]]
    a = s["assessor"].assess(s["X"][:20], broken, _slice(s["pred"], 20))
    assert all(x.result.state is TrustState.INSUFFICIENT for x in a)
    assert all(ReasonCode.DATA_QUALITY_FLOOR in x.result.reason_codes for x in a)


def test_thin_cohorts_get_thin_cohort_not_a_confident_score(setup):
    s = setup
    a = s["assessor"].assess(s["X"][:60], s["tx"][:60], _slice(s["pred"], 60))
    thin = [x for x in a if ReasonCode.THIN_COHORT in x.result.reason_codes]
    for x in thin:
        assert x.result.ti is None and x.components["rel"].evidence_count < TrustConfig().n_min


def test_recorded_properties_of_trust_components(setup, assessed, capsys):
    """Measurements on this seed, printed for the record. Only robust facts are asserted."""
    s = setup
    call = s["pred"].calibrated >= s["r"].manifest.thresholds["t_high"]
    wrong = call != (s["y"] == 1)
    assert wrong.sum() >= 5
    out = {}
    for name in ("conf", "rel", "fam", "exp", "drift"):
        v = np.array([a.components[name].score if a.components[name].score is not None else np.nan
                      for a in assessed])
        out[name] = (np.nanmean(v[wrong]), np.nanmean(v[~wrong]))
    with capsys.disabled():
        print("\ncomponent mean score, wrong calls vs right calls:")
        for k, (w, r_) in out.items():
            print(f"  {k:6s} wrong {w:.3f}  right {r_:.3f}")
    # Model reliability and confidence are lower on the cases the model gets wrong.
    assert out["rel"][0] < out["rel"][1] and out["conf"][0] < out["conf"][1]
