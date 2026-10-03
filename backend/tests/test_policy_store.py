"""FR-21: versioned, effective-dated policies that need a second person's approval."""

import json
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


@pytest.fixture
def svc():
    clock = {"t": T0}
    return PolicyService(InMemoryRepository(), lambda: clock["t"]), clock


def test_proposal_is_pending_and_numbered(svc):
    s, _ = svc
    a, b = s.propose(T, "u:admin", {}), s.propose(T, "u:admin", {"dq_gate_action": "hold"})
    assert (a["version"], b["version"]) == ("policy-1", "policy-2")
    assert a["status"] == "pending" and a["approved_by"] is None
    assert a["payload"] == {"dq_gate_action": "request_human_review", "automation_level": 0}
    assert [p["version"] for p in s.list(T)] == ["policy-2", "policy-1"]  # newest first


def test_nobody_approves_their_own_policy(svc):
    s, _ = svc
    p = s.propose(T, "u:alice", {})
    with pytest.raises(PolicyError) as e:
        s.approve(T, "u:alice", p["id"])
    assert (e.value.code, e.value.http) == ("same_approver", 403)
    assert s.list(T)[0]["status"] == "pending"
    assert s.approve(T, "u:bob", p["id"])["approved_by"] == "u:bob"


def test_approval_is_once_only_and_unknown_ids_are_404(svc):
    s, _ = svc
    p = s.propose(T, "u:alice", {})
    s.approve(T, "u:bob", p["id"])
    with pytest.raises(PolicyError) as e:
        s.approve(T, "u:carol", p["id"])
    assert e.value.code == "already_approved"
    with pytest.raises(PolicyError) as e:
        s.approve(T, "u:bob", "no-such-id")
    assert e.value.http == 404


@pytest.mark.parametrize("payload,code", [
    ({"automation_level": 1}, "automation_not_supported"),   # FR-23: recommend-only
    ({"automation_level": True}, "automation_not_supported"),
    ({"dq_gate_action": "approve"}, "bad_dq_gate_action"),
    ({"t_high": 0.1}, "unknown_fields"),                     # thresholds belong to the model bundle
])
def test_invalid_policies_are_refused(svc, payload, code):
    s, _ = svc
    with pytest.raises(PolicyError) as e:
        s.propose(T, "u:alice", payload)
    assert e.value.code == code and s.list(T) == []


def test_only_an_approved_version_in_force_applies(svc):
    s, clock = svc
    assert s.active(T) is None and s.config(T, T0, t_low=0.3, t_high=0.7).version == "policy-0"
    p1 = s.propose(T, "u:alice", {"dq_gate_action": "hold"})
    assert s.active(T) is None  # pending is not in force
    clock["t"] = T0 + timedelta(hours=1)
    s.approve(T, "u:bob", p1["id"])
    cfg = s.config(T, clock["t"], t_low=0.3, t_high=0.7)
    assert (cfg.version, cfg.dq_gate_action.value, cfg.t_high) == ("policy-1", "hold", 0.7)
    # Decisions made before the approval still see the default: history is not rewritten.
    assert s.config(T, T0 + timedelta(minutes=30), t_low=0.3, t_high=0.7).version == "policy-0"


def test_a_future_effective_date_delays_the_policy(svc):
    s, _ = svc
    p = s.propose(T, "u:alice", {"dq_gate_action": "hold"}, effective_from=T0 + timedelta(days=2))
    s.approve(T, "u:bob", p["id"])
    assert s.active(T, T0 + timedelta(days=1)) is None
    assert s.active(T, T0 + timedelta(days=2))["version"] == "policy-1"


def test_the_latest_approved_version_wins_and_policies_are_per_tenant(svc):
    s, clock = svc
    p1, p2 = s.propose(T, "u:alice", {}), s.propose(T, "u:alice", {"dq_gate_action": "hold"})
    clock["t"] = T0 + timedelta(hours=1)
    s.approve(T, "u:bob", p1["id"])
    clock["t"] = T0 + timedelta(hours=2)
    s.approve(T, "u:bob", p2["id"])
    assert s.active(T)["version"] == "policy-2"
    assert s.active("other-tenant") is None and s.list("other-tenant") == []


def test_proposals_and_approvals_are_audited(svc):
    s, _ = svc
    p = s.propose(T, "u:alice", {})
    s.approve(T, "u:bob", p["id"])
    log = [(r["actor"], r["action"], r["object"]) for r in s.repo.rows(T, "audit_log")]
    assert log == [("u:alice", "policy_propose", "policy-1"), ("u:bob", "policy_approve", "policy-1")]


# -- through the API, and into scoring ------------------------------------------------------------
SEC = {k: k.encode() + b"-secret" for k in ("ing", "adm", "adm2", "apr", "apr2", "aud", "ana")}
ROLES = {"ing": {"ingest"}, "adm": {"admin"}, "adm2": {"admin"}, "apr": {"approver"},
         "apr2": {"approver"}, "aud": {"auditor"}, "ana": {"analyst"}}


@pytest.fixture(scope="module")
def artefact(trained, tmp_path_factory):
    d = tmp_path_factory.mktemp("polapi") / "b"
    save_bundle(d, trained[3].scoring_bundle(), trained[3].manifest, b"k")
    return load_bundle(d, T, b"k")


@pytest.fixture
def api(trained, artefact):
    _, txns, _, r = trained
    repo = InMemoryRepository()
    clock = {"t": START}
    reg = BundleRegistry()
    reg.register(T, *artefact)
    ing = IngestionService(repo, IngestionConfig(clock=lambda: clock["t"]))
    scoring = ScoringService(repo, reg, ScoringConfig(clock=lambda: clock["t"]))
    creds = Credentials([Credential(k, T, SEC[k], frozenset(v)) for k, v in ROLES.items()])
    client = TestClient(create_app(ing, creds, scoring=scoring))
    by_id = {t["txn_id"]: t for t in txns}
    ids = r.test_table.txn_ids

    def call(method, path, key, payload=None):
        body = b"" if payload is None else json.dumps(payload).encode()
        return client.request(method, path, content=body, headers={
            "X-Assay-Key": key, "X-Assay-Signature": sign(SEC[key], body)})

    def submit(i):
        t = to_wire(by_id[ids[i]])
        clock["t"] = datetime.fromisoformat(t["event_time"]) + timedelta(seconds=2)
        return call("POST", "/v1/transactions", "ing", payload=t).json()["decision"]

    return {"call": call, "submit": submit, "repo": repo}


def test_roles_gate_every_policy_route(api):
    call = api["call"]
    assert call("POST", "/v1/config/policies", "ana", {}).status_code == 403
    assert call("POST", "/v1/config/policies", "apr", {}).status_code == 403   # approvers do not propose
    p = call("POST", "/v1/config/policies", "adm", {})
    assert p.status_code == 201 and p.json()["status"] == "pending"
    pid = p.json()["id"]
    assert call("POST", f"/v1/config/policies/{pid}/approve", "adm").status_code == 403  # admins do not approve
    assert call("GET", "/v1/config/policies", "ana").status_code == 403
    for reader in ("adm", "apr", "aud"):
        assert call("GET", "/v1/config/policies", reader).status_code == 200


def test_the_proposer_cannot_approve_even_holding_both_roles(api):
    both = Credential("both", T, b"both-secret", frozenset({"admin", "approver"}))
    app = create_app(IngestionService(InMemoryRepository()), Credentials([both]),
                     scoring=ScoringService(InMemoryRepository(), BundleRegistry()))
    c = TestClient(app)

    def call(method, path, payload=None):
        body = b"" if payload is None else json.dumps(payload).encode()
        return c.request(method, path, content=body, headers={
            "X-Assay-Key": "both", "X-Assay-Signature": sign(b"both-secret", body)})

    pid = call("POST", "/v1/config/policies", {}).json()["id"]
    r = call("POST", f"/v1/config/policies/{pid}/approve")
    assert r.status_code == 403 and r.json()["detail"]["code"] == "same_approver"


def test_api_validation_and_error_codes(api):
    call = api["call"]
    assert call("POST", "/v1/config/policies", "adm", {"automation_level": 2}).status_code == 422
    assert call("POST", "/v1/config/policies", "adm", {"effective_from": "soon"}).json()["detail"]["code"] == "bad_date"
    assert call("POST", "/v1/config/policies", "adm", [1]).status_code == 400
    pid = call("POST", "/v1/config/policies", "adm", {}).json()["id"]
    assert call("POST", f"/v1/config/policies/{pid}/approve", "apr").status_code == 200
    assert call("POST", f"/v1/config/policies/{pid}/approve", "apr2").status_code == 409
    assert call("POST", "/v1/config/policies/nope/approve", "apr").status_code == 404


def test_decisions_record_the_policy_version_in_force(api):
    call = api["call"]
    first = api["submit"](0)
    assert first["policy_version"] == "policy-0"
    pid = call("POST", "/v1/config/policies", "adm", {"dq_gate_action": "hold"}).json()["id"]
    assert api["submit"](12)["policy_version"] == "policy-0"          # proposed, not yet approved
    call("POST", f"/v1/config/policies/{pid}/approve", "apr")
    assert api["submit"](24)["policy_version"] == "policy-1"          # approved: now in force
    # The decision made earlier keeps the version it was made under.
    again = call("GET", f"/v1/decisions/{first['decision_id']}", "aud").json()
    assert again["policy_version"] == "policy-0"
