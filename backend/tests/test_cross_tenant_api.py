"""FR-40: one tenant must not read or change another's data through ANY route.

Tenant A scores transactions, has a review case with an analyst decision, and a policy. Tenant B holds
every human role (and ingest), so no role check can be what stops it. B then goes at A's ids on every
route. Each attempt must find nothing, change nothing, and leak none of A's identifiers in what it
says back, including in error messages.
"""

import json
from datetime import datetime, timedelta

import pytest
from fastapi.testclient import TestClient

from assay.api import Credential, Credentials, create_app, sign
from assay.detection import load_bundle, save_bundle
from assay.ingestion import IngestionConfig, IngestionService, InMemoryRepository
from assay.review.service import REVIEW_ACTIONS, ReviewConfig, ReviewService
from assay.scoring import BundleRegistry, ScoringConfig, ScoringService
from assay.synthetic import START, to_wire

A, B = "tenant-synth", "tenant-b"  # the trained bundle belongs to tenant-synth
ALL_HUMAN = frozenset({"analyst", "senior_analyst", "manager", "auditor", "approver", "admin", "ingest"})
SECRETS = {"a-ing": b"s1", "a-ana": b"s2", "a-adm": b"s3", "a-apr": b"s4", "b-all": b"s5"}


@pytest.fixture(scope="module")
def artefact(trained, tmp_path_factory):
    d = tmp_path_factory.mktemp("xtenant") / "b"
    save_bundle(d, trained[3].scoring_bundle(), trained[3].manifest, b"k")
    return load_bundle(d, A, b"k")


@pytest.fixture
def world(trained, artefact):
    _, txns, _, r = trained
    repo, clock = InMemoryRepository(), {"t": START}
    reg = BundleRegistry()
    reg.register(A, *artefact)          # tenant B has no bundle at all
    ing = IngestionService(repo, IngestionConfig(clock=lambda: clock["t"]))
    scoring = ScoringService(repo, reg, ScoringConfig(clock=lambda: clock["t"]))
    review = ReviewService(repo, scoring, ReviewConfig(clock=lambda: clock["t"], blind_share=0.0))
    creds = Credentials([
        Credential("a-ing", A, SECRETS["a-ing"], frozenset({"ingest"})),
        Credential("a-ana", A, SECRETS["a-ana"], frozenset({"analyst", "senior_analyst"})),
        Credential("a-adm", A, SECRETS["a-adm"], frozenset({"admin"})),
        Credential("a-apr", A, SECRETS["a-apr"], frozenset({"approver"})),
        Credential("b-all", B, SECRETS["b-all"], ALL_HUMAN)])
    client = TestClient(create_app(ing, creds, scoring=scoring, review=review, rate_limit_per_minute=None))
    by_id, ids = {t["txn_id"]: t for t in txns}, r.test_table.txn_ids

    def call(method, path, key, payload=None):
        body = b"" if payload is None else json.dumps(payload).encode()
        return client.request(method, path, content=body, headers={
            "X-Assay-Key": key, "X-Assay-Signature": sign(SECRETS[key], body)})

    decisions, wires = [], {}
    for i in range(0, 240, 12):
        t = to_wire(by_id[ids[i]])
        wires[t["txn_id"]] = t
        clock["t"] = datetime.fromisoformat(t["event_time"]) + timedelta(seconds=2)
        decisions.append(call("POST", "/v1/transactions", "a-ing", t).json()["decision"])
    d = next(d for d in decisions if d["recommendation"] in REVIEW_ACTIONS)
    assert call("POST", f"/v1/review/cases/{d['decision_id']}/actions", "a-ana",
                {"action": "approve"}).status_code == 201
    pol = call("POST", "/v1/config/policies", "a-adm", {"dq_gate_action": "hold"}).json()
    assert call("POST", f"/v1/config/policies/{pol['id']}/approve", "a-apr").status_code == 200
    mine = {"decision": d["decision_id"], "txn": d["txn_id"], "policy": pol["id"],
            "all_txns": [x["txn_id"] for x in decisions], "all_decisions": [x["decision_id"] for x in decisions], "wires": wires}
    return {"call": call, "mine": mine, "repo": repo, "clock": clock}


def snapshot(repo):
    tables = ("transactions", "outcomes", "policy_decisions", "analyst_actions", "feedback_records",
              "policy_versions", "policy_approvals", "audit_log")
    return {t: len(repo.rows(A, t)) for t in tables}


def test_tenant_b_finds_nothing_of_tenant_a_on_any_route_and_changes_nothing(world):
    call, m = world["call"], world["mine"]
    d, t, p = m["decision"], m["txn"], m["policy"]
    before = snapshot(world["repo"])
    attempts = {
        "get decision": ("GET", f"/v1/decisions/{d}", None),
        "get explanation": ("GET", f"/v1/decisions/{d}/explanation", None),
        "get lineage": ("GET", f"/v1/decisions/{d}/lineage", None),
        "replay": ("POST", f"/v1/decisions/{d}/replay", None),
        "open review case": ("GET", f"/v1/review/cases/{d}", None),
        "record action on A's case": ("POST", f"/v1/review/cases/{d}/actions", {"action": "block"}),
        "override A's case": ("POST", f"/v1/review/cases/{d}/actions",
                              {"action": "override", "override_to": "block", "reason_code": "fraud_confirmed"}),
        "adjudicate A's case": ("POST", f"/v1/review/cases/{d}/adjudication",
                                {"final_decision": "block", "rationale": "x"}),
        "approve A's policy": ("POST", f"/v1/config/policies/{p}/approve", None),
    }
    leaks = []
    for name, (method, path, payload) in attempts.items():
        r = call(method, path, "b-all", payload)
        assert r.status_code == 404, (name, r.status_code, r.text)
        leaks.append((name, r.text))
    # Listings: B sees an empty world, never A's rows.
    lists = {
        "queue": call("GET", "/v1/review/queue?include_closed=true", "b-all"),
        "dashboard": call("GET", "/v1/dashboard/summary", "b-all"),
        "policies": call("GET", "/v1/config/policies", "b-all"),
        "bundles": call("GET", "/v1/models/bundles", "b-all"),
        "reports": call("GET", "/v1/validation/reports", "b-all"),
        "audit": call("GET", "/v1/audit/export?limit=5000", "b-all"),
    }
    for name, r in lists.items():
        assert r.status_code == 200, (name, r.status_code, r.text)
        leaks.append((name, r.text))
    assert lists["queue"].json()["items"] == [] and lists["queue"].json()["total_cases"] == 0
    assert lists["policies"].json()["items"] == [] and lists["bundles"].json() == []
    assert lists["dashboard"].json()["assessed_cases"] == 0
    # Nothing A owns appears anywhere in anything B was told.
    secrets = [d, t, p, "u:a-ana", "api:a-ing", "api:a-adm", "api:a-apr", *m["all_txns"], *m["all_decisions"]]
    for name, text in leaks:
        for s in secrets:
            assert s not in text, f"{name} leaked {s!r}"
    # And A is exactly as it was, apart from the audit log recording B's reads... which are B's own rows.
    assert snapshot(world["repo"]) == before


def test_tenant_bs_reads_and_refusals_are_filed_under_tenant_b_not_a(world):
    call = world["call"]
    before_a = len(world["repo"].rows(A, "audit_log"))
    call("GET", "/v1/audit/export", "b-all")
    call("GET", f"/v1/decisions/{world['mine']['decision']}", "b-all")
    assert len(world["repo"].rows(A, "audit_log")) == before_a
    assert any(r["action"] == "audit_read" for r in world["repo"].rows(B, "audit_log"))


def test_an_outcome_posted_by_b_for_a_transaction_id_does_not_touch_a(world):
    """B can post an outcome for any txn id it likes. It is B's own record (quarantined, since B has no such
    transaction) and A's transaction and outcomes are untouched."""
    call, t = world["call"], world["mine"]["txn"]
    before = snapshot(world["repo"])
    r = call("POST", "/v1/outcomes", "b-all", {
        "event_id": "o-1", "txn_id": t, "outcome_type": "confirmed_fraud", "source": "x",
        "event_time": "2026-01-01T00:00:00+00:00", "schema_version": "1"})
    assert r.status_code in (202, 422)
    assert snapshot(world["repo"]) == before
    assert not world["repo"].rows(A, "outcomes")
    assert world["repo"].rows(B, "outcomes") == [] or world["repo"].rows(B, "quarantine")


def test_b_cannot_see_a_through_the_audit_search_filters(world):
    call, m = world["call"], world["mine"]
    for q in (f"txn_id={m['txn']}", "actor=u:a-ana", "action=score", f"model_version={'x'}"):
        r = call("GET", f"/v1/audit/export?{q}", "b-all")
        assert r.status_code == 200 and r.json()["items"] == [], q
    # A's own auditor-equivalent sanity: the same search as A's auditor would find rows, so the empty
    # results above mean isolation, not a broken filter.
    mine = world["repo"].rows(A, "audit_log")
    assert any(r["object"] == m["txn"] for r in mine)


def test_the_two_tenants_can_use_the_same_ids_without_colliding(world):
    """Uniqueness is per tenant: B ingesting a transaction under A's txn_id creates B's own record, and A's is untouched."""
    call, t = world["call"], world["mine"]["txn"]
    a_txn = world["repo"].get_transaction(A, t)
    wire = {**world["mine"]["wires"][t], "event_id": "b-e1", "customer_pid": "c-b"}  # a valid transaction, B's own
    r = call("POST", "/v1/transactions", "b-all", wire)
    # B has no model bundle, so scoring is unavailable (503), but the transaction itself is stored for B.
    assert r.status_code == 503 and r.json()["detail"]["code"] == "scoring_unavailable", r.text
    assert world["repo"].get_transaction(B, t)["customer_pid"] == "c-b"
    assert world["repo"].get_transaction(A, t)["customer_pid"] == a_txn["customer_pid"]
    assert len(world["repo"].rows(A, "transactions")) == len(world["mine"]["all_txns"])
