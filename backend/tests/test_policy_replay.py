"""Policy replay (PRD 10.4): preview a proposed policy against past cases."""

import json
from collections import Counter
from datetime import UTC, datetime, timedelta

import pytest
from fastapi.testclient import TestClient

from assay.api import Credential, Credentials, create_app, sign
from assay.detection import load_bundle, save_bundle
from assay.ingestion import IngestionConfig, IngestionService, InMemoryRepository
from assay.policy import PolicyError, PolicyService
from assay.scoring import BundleRegistry, ScoringConfig, ScoringService
from assay.synthetic import START, to_wire

T = "tenant-synth"
T0 = datetime(2026, 1, 1, tzinfo=UTC)
THRESHOLDS = {"t_low": 0.3, "t_high": 0.75}

# (txn, risk, trust state, reason codes, amount, gate it was decided at, the action stored)
CASES = [
    ("t1", 0.10, "high", [], 100, "matrix", "approve"),
    ("t2", 0.10, "high", [], 6000, "matrix", "approve"),
    ("t3", 0.10, "moderate", [], 200, "matrix", "approve_sampled_qa"),
    ("t4", 0.50, "high", [], 100, "matrix", "request_human_review"),
    ("t5", 0.50, "low", [], 100, "matrix", "escalate"),
    ("t6", 0.80, "high", [], 300, "matrix", "block"),
    ("t7", 0.80, "moderate", [], 300, "matrix", "request_human_review_priority"),
    ("t8", 0.10, "insufficient_evidence", ["THIN_COHORT"], 100, "matrix", "request_human_review"),
    ("t9", 0.10, "high", [], 50, "hard_rule", "block"),
    ("t10", 0.90, "insufficient_evidence", ["DATA_QUALITY_FLOOR"], 100, "data_quality", "request_human_review"),
]


def build(repo, tenant=T, cases=CASES, thresholds=THRESHOLDS, bundle="b-1"):
    manifest = {"thresholds": thresholds} if thresholds is not None else {}
    repo.insert(tenant, "model_bundles", {"bundle_id": bundle, "manifest": manifest}, "t")
    for i, (txn, risk, state, reasons, amount, gate, action) in enumerate(cases):
        repo.append_transaction(tenant, {"txn_id": txn, "amount": amount, "event_time": T0 + timedelta(hours=i)}, "t")
        pr = repo.insert(tenant, "predictions", {"txn_id": txn, "bundle_id": bundle, "calibrated_risk": risk}, "t")
        ta = repo.insert(tenant, "trust_assessments", {"prediction_id": pr["id"], "version_no": 1, "state": state,
                                                       "reason_codes": reasons}, "t")
        repo.insert(tenant, "policy_decisions", {"txn_id": txn, "trust_assessment_id": ta["id"], "gate": gate,
                                                 "recommended_action": action, "policy_version": "policy-0"}, "t")


@pytest.fixture
def svc():
    repo = InMemoryRepository()
    build(repo)
    return PolicyService(repo, lambda: T0 + timedelta(days=1)), repo


def run(svc, payload, **kw):
    """`svc` is the fixture pair (service, repo) or just the service."""
    service = svc[0] if isinstance(svc, tuple) else svc
    return service.preview(T, "u:admin", payload, **kw)


def test_the_baseline_reproduces_the_hand_computed_actions(svc):
    out = run(svc, {})
    assert out["n_decisions"] == 10 and out["baseline_version"] == "policy-0"
    assert out["actions"]["before"] == {"approve": 2, "approve_sampled_qa": 1, "block": 2, "escalate": 1,
                                        "request_human_review": 3, "request_human_review_priority": 1}
    assert out["review_volume"] == {"before": 5, "after": 5, "change": 0, "change_pct": 0.0}
    assert out["changed_decisions"] == 0 and out["changes"] == [] and out["examples"] == []


def test_an_amount_rule_sends_only_the_large_approval_to_a_human(svc):
    out = run(svc, {"always_review_above": 5000})
    assert out["changes"] == [{"from": "approve", "to": "request_human_review", "count": 1}]
    assert out["review_volume"] == {"before": 5, "after": 6, "change": 1, "change_pct": 20.0}
    (ex,) = out["examples"]
    assert (ex["txn_id"], ex["amount"], ex["was"], ex["now"], ex["because"]) == (
        "t2", 6000.0, "approve", "request_human_review", "segment_rule")


def test_wider_risk_thresholds_move_low_risk_cases_into_review(svc):
    out = run(svc, {"t_low": 0.05, "t_high": 0.6})
    assert sorted((c["from"], c["to"], c["count"]) for c in out["changes"]) == sorted([
        ("approve", "request_human_review", 2),             # t1, t2: now medium risk, high trust -> review
        ("approve_sampled_qa", "request_human_review", 1),  # t3
        ("request_human_review", "escalate", 1),            # t8: medium risk, insufficient evidence -> escalate
    ])
    assert out["review_volume"] == {"before": 5, "after": 8, "change": 3, "change_pct": 60.0}
    assert out["risk_bands"]["before"] == {"high": 3, "low": 5, "medium": 2}   # t6, t7, t10 high; t4, t5 medium
    assert out["risk_bands"]["after"] == {"high": 3, "medium": 7}


def test_a_hard_rule_decision_is_never_changed_by_any_policy(svc):
    for payload in ({"always_review_above": 1}, {"t_low": 0.05, "t_high": 0.06}, {"dq_gate_action": "hold"}):
        out = run(svc, payload)
        assert not any(e["txn_id"] == "t9" for e in out["examples"]), payload
        assert out["actions"]["after"].get("block", 0) >= 1


def test_changing_the_data_quality_action_changes_the_action_but_not_the_review_volume(svc):
    out = run(svc, {"dq_gate_action": "hold"})
    assert out["changes"] == [{"from": "request_human_review", "to": "hold", "count": 1}]
    assert out["review_volume"]["change"] == 0 and out["review_volume"]["change_pct"] == 0.0   # a hold is still a review
    assert out["examples"][0]["txn_id"] == "t10" and out["examples"][0]["because"] == "data_quality"


def test_the_comparison_is_against_the_policy_in_force_not_the_stored_actions(svc):
    s, _ = svc
    p = s.propose(T, "u:alice", {"always_review_above": 5000})
    s.approve(T, "u:bob", p["id"])
    out = run(s, {"always_review_above": 5000})
    assert out["baseline_version"] == "policy-1" and out["changed_decisions"] == 0     # same policy: no change
    # The stored decision for t2 is still "approve" (made before the policy), yet the baseline already reviews it.
    assert out["actions"]["before"]["request_human_review"] == 4
    looser = run(s, {})                                                                  # dropping the rule
    assert looser["changes"] == [{"from": "request_human_review", "to": "approve", "count": 1}]
    assert looser["review_volume"]["change"] == -1


def test_the_latest_decision_per_transaction_is_the_one_replayed(svc):
    s, repo = svc
    pr = repo.find(T, "predictions", {"txn_id": "t1"})[0]
    ta2 = repo.insert(T, "trust_assessments", {"prediction_id": pr["id"], "version_no": 2, "state": "low",
                                               "reason_codes": []}, "t")
    repo.insert(T, "policy_decisions", {"txn_id": "t1", "trust_assessment_id": ta2["id"], "gate": "matrix",
                                        "recommended_action": "request_human_review", "policy_version": "policy-0"}, "t")
    out = run(s, {})
    assert out["n_decisions"] == 10                                  # t1 counted once
    assert out["actions"]["before"]["approve"] == 1                  # its refined (Low trust) result replaced the first


def test_limit_looks_at_the_most_recent_cases_only(svc):
    s, _ = svc
    out = run(s, {}, limit=3)
    assert out["n_decisions"] == 3 and out["window"]["from"] < out["window"]["to"]
    assert out["actions"]["before"] == {"block": 1, "request_human_review": 2}   # t8, t9 (hard rule), t10


def test_an_empty_history_is_reported_not_divided_by_zero():
    out = PolicyService(InMemoryRepository(), lambda: T0).preview(T, "u:a", {"always_review_above": 10})
    assert out["n_decisions"] == 0 and out["review_volume"]["change_pct"] is None
    assert out["window"] == {"from": None, "to": None} and out["changes"] == []


def test_a_zero_baseline_has_no_percentage():
    only_approvals = InMemoryRepository()
    build(only_approvals, cases=[CASES[0], CASES[1]])
    out = PolicyService(only_approvals, lambda: T0).preview(T, "u:a", {"always_review_above": 5000})
    assert out["review_volume"] == {"before": 0, "after": 1, "change": 1, "change_pct": None}


def test_cases_scored_by_a_bundle_with_no_recorded_thresholds_are_skipped_and_counted():
    repo = InMemoryRepository()
    build(repo, thresholds=None)
    out = PolicyService(repo, lambda: T0).preview(T, "u:a", {})
    assert out["n_decisions"] == 0 and out["skipped"] == 10


def test_a_policy_the_server_would_refuse_is_refused_here_too(svc):
    s, _ = svc
    for bad in ({"automation_level": 1}, {"t_low": 0.9, "t_high": 0.1}, {"always_review_above": -1}, {"nope": 1}):
        with pytest.raises(PolicyError):
            run(s, bad)


def test_a_preview_changes_nothing_but_the_audit_log(svc):
    s, repo = svc
    tables = ("policy_decisions", "policy_versions", "policy_approvals", "trust_assessments", "predictions")
    before = {t: len(repo.rows(T, t)) for t in tables}
    run(s, {"always_review_above": 5000})
    assert {t: len(repo.rows(T, t)) for t in tables} == before
    log = [r for r in repo.rows(T, "audit_log") if r["action"] == "policy_preview"]
    assert len(log) == 1 and log[0]["actor"] == "u:admin" and log[0]["result"] == "10 cases"


def test_other_tenants_cases_are_not_replayed(svc):
    s, repo = svc
    build(repo, tenant="tenant-b", cases=[("x1", 0.1, "high", [], 9999, "matrix", "approve")], bundle="b-x")
    assert PolicyService(repo, lambda: T0).preview("tenant-b", "u:b", {"always_review_above": 1})["n_decisions"] == 1
    assert run(s, {})["n_decisions"] == 10


def test_a_replayed_policy_that_is_already_in_force_changes_nothing_on_real_scored_data(trained, tmp_path):
    """The oracle: replaying the policy in force must reproduce exactly what the real scoring pipeline stored."""
    _, txns, _, r = trained
    d = tmp_path / "b"
    save_bundle(d, r.scoring_bundle(), r.manifest, b"k")
    art = load_bundle(d, T, b"k")
    repo, clock = InMemoryRepository(), {"t": START}
    reg = BundleRegistry()
    reg.register(T, *art)
    ing = IngestionService(repo, IngestionConfig(clock=lambda: clock["t"]))
    scoring = ScoringService(repo, reg, ScoringConfig(clock=lambda: clock["t"]))
    by_id, ids = {t["txn_id"]: t for t in txns}, r.test_table.txn_ids
    for i in range(0, 360, 12):
        t = to_wire(by_id[ids[i]])
        clock["t"] = datetime.fromisoformat(t["event_time"]) + timedelta(seconds=2)
        ing.ingest_transaction(T, t, source_id="s", actor="t")
        scoring.score_transaction(T, t["txn_id"], actor="t")
    stored = Counter(pd["recommended_action"] for pd in
                     {p["txn_id"]: p for p in repo.find(T, "policy_decisions")}.values())
    s = PolicyService(repo, lambda: clock["t"])
    out = s.preview(T, "u:a", {})
    assert out["n_decisions"] == 30 and out["actions"]["before"] == dict(sorted(stored.items()))
    assert out["actions"]["after"] == out["actions"]["before"] and out["changed_decisions"] == 0
    stricter = s.preview(T, "u:a", {"always_review_above": 1})   # every amount is over 1: every approval needs a person
    assert stricter["actions"]["after"].get("approve", 0) == 0 and stricter["actions"]["after"].get("approve_sampled_qa", 0) == 0
    assert stricter["review_volume"]["after"] >= stricter["review_volume"]["before"]


# -- through the API ------------------------------------------------------------------------------------
@pytest.fixture
def api(svc):
    _, repo = svc
    creds = Credentials([Credential(k, T, k.encode() + b"-secret", frozenset({k})) for k in
                         ("admin", "approver", "auditor", "analyst")])
    from assay.scoring import BundleRegistry as BR
    app = create_app(IngestionService(repo), creds, scoring=ScoringService(repo, BR()), rate_limit_per_minute=None)
    c = TestClient(app)

    def call(key, payload, path="/v1/config/policies/preview"):
        body = json.dumps(payload).encode()
        return c.post(path, content=body, headers={"X-Assay-Key": key, "X-Assay-Signature": sign(key.encode() + b"-secret", body)})

    return call


def test_the_preview_route_returns_the_comparison_to_those_who_may_see_policies(api):
    for key in ("admin", "approver", "auditor"):
        r = api(key, {"always_review_above": 5000})
        assert r.status_code == 200, (key, r.text)
        assert r.json()["review_volume"]["change"] == 1
    assert api("analyst", {}).status_code == 403


def test_the_preview_route_refuses_a_bad_policy_with_a_reason_and_ignores_the_date(api):
    r = api("admin", {"t_low": 0.9, "t_high": 0.1})
    assert r.status_code == 422 and r.json()["detail"]["code"] == "bad_thresholds"
    assert api("admin", {"automation_level": 2}).status_code == 422
    assert api("admin", {"always_review_above": 5000, "effective_from": "2030-01-01T00:00:00+00:00"}).status_code == 200
    assert api("admin", [1]).status_code == 400
