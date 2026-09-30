"""Scoring service on both repositories: the same behaviour is required of each."""

from datetime import datetime, timedelta

import numpy as np
import pytest
from sklearn.isotonic import IsotonicRegression

from assay.detection import load_bundle, save_bundle
from assay.features import build_table, feature_names
from assay.ingestion import IngestionConfig, IngestionService, InMemoryRepository
from assay.ingestion.pg_repo import PostgresRepository
from assay.policy import Action
from assay.scoring import BundleRegistry, ScoringConfig, ScoringError, ScoringService
from assay.synthetic import START, to_wire

TENANT = "tenant-synth"
KEY = b"k"
NOVEL = "novel_open_banking_scam"
both_repos = pytest.mark.parametrize("env", ["memory", "postgres"], indirect=True)


@pytest.fixture(scope="module")
def artefact(trained, tmp_path_factory):
    """The real path: save a signed bundle, load it back, register what comes out."""
    _, _, _, r = trained
    d = tmp_path_factory.mktemp("bundle") / "b"
    save_bundle(d, r.scoring_bundle(), r.manifest, KEY)
    return load_bundle(d, TENANT, KEY)


@pytest.fixture
def env(request, trained, artefact):
    ds, txns, _, r = trained
    repo = (InMemoryRepository() if request.param == "memory"
            else PostgresRepository(request.getfixturevalue("pg_conn")))
    clock = {"t": START}
    ing = IngestionService(repo, IngestionConfig(clock=lambda: clock["t"]))
    reg = BundleRegistry()
    obj, manifest = artefact
    reg.register(TENANT, obj, manifest)
    svc = ScoringService(repo, reg, ScoringConfig(clock=lambda: clock["t"]))
    by_id = {t["txn_id"]: t for t in txns}

    def ingest(txn_id):
        t = to_wire(by_id[txn_id])
        clock["t"] = datetime.fromisoformat(t["event_time"]) + timedelta(seconds=2)
        assert ing.ingest_transaction(TENANT, t).status in ("accepted", "duplicate")

    def score(txn_id):
        ingest(txn_id)
        return svc.score_transaction(TENANT, txn_id)

    return {"repo": repo, "svc": svc, "ing": ing, "score": score, "ingest": ingest, "ds": ds,
            "txns": txns, "r": r, "by_id": by_id, "reg": reg}


def pick(env, n=12, *, novel=False):
    ds, ids = env["ds"], env["r"].test_table.txn_ids
    want = [t for t in ids if (ds.truth[t]["fraud_type"] == NOVEL) == novel]
    return want[:: max(1, len(want) // n)][:n]


def first_pending(env, n=40):
    for t in pick(env, n):
        d = env["score"](t)
        if d["explanation_status"] == "pending" and d["trust"]["state"] != "insufficient_evidence":
            return d
    raise AssertionError("expected a scored case without an explanation")


@both_repos
def test_a_scored_transaction_has_the_full_lineage_chain(env):
    d = env["score"](pick(env, 1)[0])
    assert d["automation_level"] == 0 and d["bundle_id"] == env["r"].manifest.bundle_id
    assert d["trust"]["components"]["hum"]["status"] == "inactive"
    lin = env["svc"].lineage(TENANT, d["decision_id"])
    for key in ("transaction", "ingestion_event", "feature_vector", "model_bundle", "prediction"):
        assert lin[key] is not None, key
    assert lin["trust_assessments"][0]["version_no"] == 1 and len(lin["policy_decisions"]) == 1
    assert lin["model_bundle"]["artifact_sha256"] != "unsigned"  # signed bundle recorded


@both_repos
def test_scoring_is_idempotent(env):
    txn_id = pick(env, 1)[0]
    a, b = env["score"](txn_id), env["score"](txn_id)
    assert a["decision_id"] == b["decision_id"]
    assert len(env["repo"].rows(TENANT, "predictions")) == 1


@both_repos
def test_unknown_transaction_and_missing_champion(env):
    with pytest.raises(ScoringError, match="unknown transaction"):
        env["svc"].score_transaction(TENANT, "nope")
    txn_id = pick(env, 1)[0]
    env["ingest"](txn_id)
    with pytest.raises(ScoringError, match="no champion"):
        ScoringService(env["repo"], BundleRegistry()).score_transaction(TENANT, txn_id)


@both_repos
def test_a_bundle_for_another_tenant_is_refused(env, artefact):
    obj, manifest = artefact
    with pytest.raises(ScoringError, match="different tenant"):
        BundleRegistry().register("tenant-other", obj, manifest)


@both_repos
def test_stored_history_drives_the_features_and_matches_the_independent_builder(env):
    """Features come from what is stored, as of the decision, and agree with the batch builder."""
    by_cust: dict[str, list[dict]] = {}
    for t in env["txns"]:
        by_cust.setdefault(t["customer_pid"], []).append(t)
    target = None
    for ids in (pick(env, 200),):
        for tid in ids:
            c = env["by_id"][tid]["customer_pid"]
            past = [t for t in by_cust[c] if t["event_time"] < env["by_id"][tid]["event_time"]]
            if 3 <= len(past) <= 25:
                target, history = tid, past
                break
    assert target, "no test transaction with a usable history"
    for h in sorted(history, key=lambda x: x["event_time"]):
        env["ingest"](h["txn_id"])
    d = env["score"](target)
    stored = env["repo"].find(TENANT, "feature_vectors", {"txn_id": target})[0]
    expected = build_table(history + [env["by_id"][target]])
    row = expected.X[expected.txn_ids.index(target)]
    assert stored["feature_names"] == list(feature_names())
    assert np.allclose(stored["feature_values"], row)
    assert stored["feature_values"][feature_names().index("velocity_30d")] >= 0
    assert d["txn_id"] == target


@both_repos
def test_insufficient_evidence_is_null_with_reasons_and_never_approved(env):
    novel = pick(env, 6, novel=True)
    assert novel
    decisions = [env["score"](t) for t in novel]
    insufficient = [d for d in decisions if d["trust"]["state"] == "insufficient_evidence"]
    assert insufficient
    for d in insufficient:
        assert d["trust"]["ti"] is None and d["trust"]["ti_low"] is None and d["trust"]["reason_codes"]
        assert d["recommendation"] != Action.APPROVE.value


@both_repos
def test_recommend_only_and_no_high_trust_without_explanation(env):
    decisions = [env["score"](t) for t in pick(env, 10)]
    assert all(d["automation_level"] == 0 for d in decisions)
    for d in decisions:
        if d["explanation_status"] == "pending":
            assert d["trust"]["components"]["exp"]["status"] == "missing"
            assert d["trust"]["state"] != "high"  # PRD 5.4 ceiling


@both_repos
def test_refine_adds_a_new_version_and_overwrites_nothing(env):
    pending = first_pending(env)
    refined = env["svc"].refine(TENANT, pending["decision_id"])
    assert refined["trust"]["version_no"] == 2 and refined["explanation_status"] == "computed"
    assert refined["trust"]["components"]["exp"]["status"] == "active"
    lin = env["svc"].lineage(TENANT, pending["decision_id"])
    assert [a["version_no"] for a in lin["trust_assessments"]] == [1, 2]
    assert len(lin["policy_decisions"]) == 2 and len(lin["explanations"]) == 1
    assert lin["trust_assessments"][0]["components"]["exp"]["status"] == "missing"  # v1 untouched
    assert env["svc"].refine(TENANT, refined["decision_id"])["decision_id"] == refined["decision_id"]


@both_repos
def test_replay_reproduces_every_decision(env):
    for t in pick(env, 10):
        rep = env["svc"].replay(TENANT, env["score"](t)["decision_id"])
        assert rep["reproducible"], rep
    refined = env["svc"].refine(TENANT, first_pending(env)["decision_id"])
    assert env["svc"].replay(TENANT, refined["decision_id"])["reproducible"]


@both_repos
def test_a_changed_model_is_flagged_non_reproducible(env):
    d = env["score"](pick(env, 1)[0])
    lb = env["reg"].champion(TENANT)
    original = lb.model.calibrator
    lb.model.calibrator = IsotonicRegression(out_of_bounds="clip").fit([0, 1], [0.5, 0.9])  # tampered
    try:
        rep = env["svc"].replay(TENANT, d["decision_id"])
    finally:
        lb.model.calibrator = original
    assert not rep["reproducible"] and not rep["checks"]["calibrated_risk"]
    flagged = [a for a in env["repo"].rows(TENANT, "audit_log")
               if a["action"] == "decision_non_reproducible"]
    assert flagged and flagged[-1]["object"] == d["decision_id"]


@both_repos
def test_chains_verify_after_scoring(env):
    for t in pick(env, 5):
        env["score"](t)
    env["svc"].refine(TENANT, first_pending(env)["decision_id"])
    for table in ("model_bundles", "feature_vectors", "predictions", "explanations",
                  "trust_assessments", "policy_decisions", "audit_log"):
        assert env["repo"].verify(TENANT, table) is None, table
