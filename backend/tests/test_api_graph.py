"""GET /v1/decisions/{id}/graph and POST /v1/graph/edges/flag: access rules, what the view shows, and the as-of rule."""

import json
from datetime import datetime, timedelta

import pytest
from fastapi.testclient import TestClient

from assay.api import Credential, Credentials, create_app, sign
from assay.detection import load_bundle, save_bundle
from assay.ingestion import IngestionConfig, IngestionService, InMemoryRepository
from assay.review.service import ReviewConfig, ReviewService
from assay.scoring import BundleRegistry, ScoringConfig, ScoringService
from assay.synthetic import START

T = "tenant-synth"
ROLES = {"ing": {"ingest"}, "ana": {"analyst"}, "sen": {"senior_analyst"}, "aud": {"auditor"}, "mgr": {"manager"}}
SEC = {k: k.encode() + b"-secret" for k in (*ROLES, "oth")}
RING = ["c-r1", "c-r2", "c-r3"]


@pytest.fixture(scope="module")
def artefact(trained, tmp_path_factory):
    d = tmp_path_factory.mktemp("graphapi") / "b"
    save_bundle(d, trained[3].scoring_bundle(), trained[3].manifest, b"k")
    return load_bundle(d, T, b"k")


def wire(i, cust, day, *, ben=None, device=None, ip=None, hours=0):
    t = START + timedelta(days=day, hours=hours)
    return {"event_id": f"e-{i}", "txn_id": f"t-{i}", "event_time": t.isoformat().replace("+00:00", "Z"),
            "amount": "40.00", "currency": "GBP", "channel": "app", "customer_pid": cust, "account_pid": f"a-{cust}",
            "beneficiary_pid": ben, "device_hash": device, "ip_hash": ip, "country": "GB", "schema_version": "txn-1"}


@pytest.fixture
def api(artefact, request):
    repo, clock = InMemoryRepository(), {"t": START}
    reg = BundleRegistry()
    reg.register(T, *artefact)
    ing = IngestionService(repo, IngestionConfig(clock=lambda: clock["t"]))
    scoring = ScoringService(repo, reg, ScoringConfig(clock=lambda: clock["t"]))
    review = ReviewService(repo, scoring, ReviewConfig(clock=lambda: clock["t"], blind_share=getattr(request, "param", 0.0)))
    creds = Credentials([Credential(k, T, SEC[k], frozenset(v)) for k, v in ROLES.items()]
                        + [Credential("oth", "tenant-other", SEC["oth"], frozenset({"analyst", "auditor"}))])
    client = TestClient(create_app(ing, creds, scoring=scoring, review=review, rate_limit_per_minute=None))
    n = {"i": 0}

    def call(method, path, key, payload=None):
        body = b"" if payload is None else json.dumps(payload).encode()
        return client.request(method, path, content=body, headers={"X-Assay-Key": key, "X-Assay-Signature": sign(SEC[key], body)})

    def submit(cust, day, **kw):
        n["i"] += 1
        w = wire(n["i"], cust, day, **kw)
        clock["t"] = max(clock["t"], datetime.fromisoformat(w["event_time"]) + timedelta(seconds=2))
        r = call("POST", "/v1/transactions", "ing", w)
        assert r.status_code in (200, 201, 202), r.text
        return w, r.json()["decision"]

    def fraud_outcome(txn_id, day):
        n["i"] += 1
        ev = START + timedelta(days=day)
        r = call("POST", "/v1/outcomes", "ing", {"event_id": f"o-{n['i']}", "txn_id": txn_id,
                                                 "event_time": ev.isoformat().replace("+00:00", "Z"),
                                                 "outcome_type": "confirmed_fraud", "source": "bank", "schema_version": "out-1"})
        assert r.status_code in (200, 201, 202), r.text
        clock["t"] = max(clock["t"], ev + timedelta(seconds=2))

    def build_ring():
        """Three customers who each use one device and one beneficiary on several days. Returns {customer: first txn}."""
        first = {}
        for day in range(1, 5):
            for c in RING:
                w, _ = submit(c, day, ben="b-ring", device="d-ring", hours=RING.index(c))
                first.setdefault(c, w)
        return first

    return {"call": call, "submit": submit, "fraud_outcome": fraud_outcome, "build_ring": build_ring, "repo": repo, "clock": clock}


def test_a_new_customer_paying_the_rings_beneficiary_sees_the_ring_and_the_known_fraud(api):
    first = api["build_ring"]()
    api["fraud_outcome"](first["c-r2"]["txn_id"], 6)  # matured on day 6
    _, d = api["submit"]("c-new", 10, ben="b-ring")
    g = api["call"]("GET", f"/v1/decisions/{d['decision_id']}/graph", "ana").json()
    assert g["customer"] == "c-new" and g["graph_version"] == "graph-0"
    (ent,) = [e for e in g["entities"] if e["kind"] == "beneficiary"]
    assert ent["node"] == "b-ring" and ent["other_customers"] == 3 and ent["hub"] is False
    assert {lk["other_customer"] for lk in g["links"]} == set(RING)
    assert all(lk["relationship"] == "SHARES_BENEFICIARY" and 0 < lk["confidence"] <= 1 for lk in g["links"])
    assert {lk["other_customer"]: lk["fraud_known"] for lk in g["links"]} == {"c-r1": False, "c-r2": True, "c-r3": False}
    assert g["group"]["size"] == 4 and g["group"]["fraud_linked_members"] == 1
    assert set(g["features"]) >= {"g_ring_score", "g_fraud_prox_txn"} and g["features"]["g_fraud_prox_txn"] > 0
    assert "not of wrongdoing" in g["note"] and "must not be shared with a customer" in g["note"]
    assert all("probability" not in k for k in g)  # confidence is not a fraud probability


def test_the_view_is_the_graph_as_it_stood_when_the_decision_was_made(api):
    first = api["build_ring"]()
    _, d = api["submit"]("c-early", 10, ben="b-ring")  # decided BEFORE the fraud outcome matures
    api["fraud_outcome"](first["c-r2"]["txn_id"], 12)
    api["submit"]("c-late", 20, ben="b-ring")          # and a later customer joins the structure
    g = api["call"]("GET", f"/v1/decisions/{d['decision_id']}/graph", "ana").json()
    assert {lk["fraud_known"] for lk in g["links"]} == {False}   # the fraud was not known then
    assert all(lk["other_customer"] != "c-late" for lk in g["links"])  # nor was the later customer
    assert g["entities"][0]["other_customers"] == 3
    assert g["as_of"] >= (START + timedelta(days=10)).isoformat()


def test_a_case_with_no_shared_infrastructure_says_so(api):
    _, d = api["submit"]("c-solo", 3, ben="b-only-me", device="d-solo", ip="ip-solo")
    g = api["call"]("GET", f"/v1/decisions/{d['decision_id']}/graph", "ana").json()
    assert g["links"] == [] and g["links_total"] == 0 and g["group"]["size"] == 1
    assert g["reliability"] == "ok"


def test_a_public_hub_is_marked_and_does_not_link_everyone(api):
    for k in range(70):  # seventy strangers on one public IP address
        api["submit"](f"c-pub{k}", 1 + k // 30, ip="ip-cafe", hours=k % 20)
    _, d = api["submit"]("c-visitor", 6, ip="ip-cafe")
    g = api["call"]("GET", f"/v1/decisions/{d['decision_id']}/graph", "ana").json()
    (ent,) = [e for e in g["entities"] if e["kind"] == "IP address"]
    assert ent["hub"] is True and ent["other_customers"] == 70
    assert g["links"] == [] and g["group"]["size"] == 1  # a hub links no one to anyone


def test_reading_is_audited_and_roles_are_enforced(api):
    api["build_ring"]()
    _, d = api["submit"]("c-new", 10, ben="b-ring")
    path = f"/v1/decisions/{d['decision_id']}/graph"
    for key, code in (("ana", 200), ("aud", 200), ("sen", 403), ("mgr", 403), ("ing", 403)):
        assert api["call"]("GET", path, key).status_code == code, key
    audit = [a for a in api["repo"].find(T, "audit_log") if a["action"] == "graph_read"]
    assert len(audit) == 2 and all(a["object"] == d["txn_id"] for a in audit)
    assert api["call"]("GET", "/v1/decisions/nope/graph", "ana").status_code == 404
    assert api["call"]("GET", path, "oth").status_code == 404  # another tenant sees nothing


@pytest.mark.parametrize("api", [1.0], indirect=True)
def test_a_blind_case_does_not_leak_through_the_graph(api):
    api["build_ring"]()
    _, d = api["submit"]("c-new", 10, ben="b-ring")
    r = api["call"]("GET", f"/v1/decisions/{d['decision_id']}/graph", "ana")
    assert r.status_code == 403 and r.json()["detail"]["code"] == "blind_review"
    assert api["call"]("GET", f"/v1/decisions/{d['decision_id']}/graph", "aud").status_code == 200


def test_flagging_a_link_wrong_downweights_it_for_later_decisions_and_deletes_nothing(api):
    api["build_ring"]()
    _, d1 = api["submit"]("c-a", 10, ben="b-ring")
    before = {lk["other_customer"]: lk["confidence"]
              for lk in api["call"]("GET", f"/v1/decisions/{d1['decision_id']}/graph", "ana").json()["links"]}
    api["clock"]["t"] += timedelta(hours=1)  # the flag comes after the decision
    r = api["call"]("POST", "/v1/graph/edges/flag", "ana", {"relationship": "PAID", "src": "c-r1", "dst": "b-ring",
                                                          "reason": "a family account, not a ring"})
    assert r.status_code == 201 and r.json()["effect"] == "down-weighted, not deleted"
    api["clock"]["t"] += timedelta(hours=1)
    _, d2 = api["submit"]("c-b", 11, ben="b-ring")
    after = {lk["other_customer"]: lk for lk in api["call"]("GET", f"/v1/decisions/{d2['decision_id']}/graph", "ana").json()["links"]}
    assert after["c-r1"]["flagged_wrong"] is True and after["c-r2"]["flagged_wrong"] is False
    assert after["c-r1"]["confidence"] < before["c-r1"] * 0.5       # down-weighted
    assert abs(after["c-r2"]["confidence"] - before["c-r2"]) < 0.1  # others unaffected
    # the earlier decision still shows what was known then: the flag did not exist yet
    again = {lk["other_customer"]: lk["confidence"]
             for lk in api["call"]("GET", f"/v1/decisions/{d1['decision_id']}/graph", "ana").json()["links"]}
    assert again == before
    flags = api["repo"].find(T, "graph_edge_flags")
    assert len(flags) == 1 and flags[0]["reason"] == "a family account, not a ring" and flags[0]["created_by"] == "u:ana"
    assert any(a["action"] == "graph_edge_flagged" for a in api["repo"].find(T, "audit_log"))


def test_a_flag_needs_a_reason_a_real_edge_a_known_relationship_and_the_right_role(api):
    api["build_ring"]()
    body = {"relationship": "PAID", "src": "c-r1", "dst": "b-ring", "reason": "x"}
    assert api["call"]("POST", "/v1/graph/edges/flag", "aud", body).status_code == 403
    assert api["call"]("POST", "/v1/graph/edges/flag", "mgr", body).status_code == 403
    for bad, code in (({**body, "reason": ""}, "reason_required"), ({**body, "reason": "  "}, "reason_required"),
                      ({**body, "relationship": "OWNS"}, "bad_relationship"), ({**body, "dst": "b-nowhere"}, "unknown_edge")):
        r = api["call"]("POST", "/v1/graph/edges/flag", "ana", bad)
        assert r.json()["detail"]["code"] == code, bad
    assert api["call"]("POST", "/v1/graph/edges/flag", "ana", {**body, "extra": 1}).status_code == 422
    assert api["call"]("POST", "/v1/graph/edges/flag", "sen", body).status_code == 201  # a senior analyst may too
    assert api["call"]("POST", "/v1/graph/edges/flag", "oth", body).status_code == 404  # not another tenant's graph


def test_the_graph_grows_with_new_transactions_without_being_rebuilt(api):
    api["build_ring"]()
    _, d = api["submit"]("c-x", 10, ben="b-ring")
    api["call"]("GET", f"/v1/decisions/{d['decision_id']}/graph", "ana")
    _, d2 = api["submit"]("c-y", 11, ben="b-ring")
    g2 = api["call"]("GET", f"/v1/decisions/{d2['decision_id']}/graph", "ana").json()
    assert {lk["other_customer"] for lk in g2["links"]} >= set(RING) | {"c-x"}
