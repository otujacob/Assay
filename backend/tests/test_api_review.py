"""The review API end to end: roles, server-side blind review, actions, adjudication, dashboard."""

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
from assay.validation.report import store_report

T = "tenant-synth"
SEC = {k: k.encode() + b"-secret" for k in ("ing", "ana", "ana2", "sen", "mgr", "aud", "appr")}
ROLES = {"ing": {"ingest"}, "ana": {"analyst"}, "ana2": {"analyst"}, "sen": {"senior_analyst", "analyst"},
         "mgr": {"manager"}, "aud": {"auditor"}, "appr": {"approver"}}


@pytest.fixture(scope="module")
def artefact(trained, tmp_path_factory):
    d = tmp_path_factory.mktemp("rvapi") / "b"
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
    review = ReviewService(repo, scoring, ReviewConfig(clock=lambda: clock["t"], blind_share=0.5))
    creds = Credentials([Credential(k, T, SEC[k], frozenset(v)) for k, v in ROLES.items()])
    client = TestClient(create_app(ing, creds, scoring=scoring, review=review))
    by_id = {t["txn_id"]: t for t in txns}
    ids = r.test_table.txn_ids

    def call(method, path, key, payload=None):
        body = b"" if payload is None else json.dumps(payload).encode()
        return client.request(method, path, content=body, headers={
            "X-Assay-Key": key, "X-Assay-Signature": sign(SEC[key], body)})

    def submit(i):
        t = to_wire(by_id[ids[i]])
        clock["t"] = datetime.fromisoformat(t["event_time"]) + timedelta(seconds=2)
        return call("POST", "/v1/transactions", "ing", t).json()["decision"]

    def first(blind: bool):
        for i in range(0, 600, 12):
            d = submit(i)
            if d["recommendation"] in REVIEW_ACTIONS:
                c = repo.find(T, "review_cases", {"txn_id": d["txn_id"]})[0]
                if c["blind"] is blind:
                    return d
        raise AssertionError("no case")

    return {"call": call, "submit": submit, "first": first, "repo": repo, "r": r, "reg": reg}


def test_queue_and_roles(api):
    for i in range(0, 120, 12):
        api["submit"](i)
    q = api["call"]("GET", "/v1/review/queue", "mgr").json()
    assert q["items"] and q["total_cases"] >= q["matching"] > 0
    assert api["call"]("GET", "/v1/review/queue", "ana").status_code == 200
    assert api["call"]("GET", "/v1/review/queue", "ing").status_code == 403
    assert api["call"]("GET", "/v1/review/queue", "aud").status_code == 403  # auditors read lineage, not queues
    assert api["call"]("GET", "/v1/review/queue?risk_band=nonsense", "mgr").json()["items"] == []


def test_blind_case_is_hidden_on_every_route_until_the_analyst_decides(api):
    d = api["first"](blind=True)
    did = d["decision_id"]
    case = api["call"]("GET", f"/v1/review/cases/{did}", "ana").json()
    assert case["redacted"] is True and case["decision"] is None
    # the other routes must not leak what the case route hides
    assert api["call"]("GET", f"/v1/decisions/{did}", "ana").status_code == 403
    assert api["call"]("GET", f"/v1/decisions/{did}/explanation", "ana").status_code == 403
    assert api["call"]("GET", f"/v1/decisions/{did}", "ana").json()["detail"]["code"] == "blind_review"
    queue = {i["txn_id"]: i for i in api["call"]("GET", "/v1/review/queue", "ana").json()["items"]}
    assert queue[d["txn_id"]]["risk_band"] is None
    # staff see everything
    assert api["call"]("GET", f"/v1/decisions/{did}", "mgr").status_code == 403  # a manager has no decision-read role
    assert api["call"]("GET", f"/v1/review/cases/{did}", "mgr").json()["decision"] is not None
    assert api["call"]("GET", f"/v1/review/cases/{did}", "sen").json()["decision"] is not None
    # after deciding, the analyst may read it
    r = api["call"]("POST", f"/v1/review/cases/{did}/actions", "ana", {"action": "approve", "reason_code": "x"})
    assert r.status_code == 201
    assert api["call"]("GET", f"/v1/decisions/{did}", "ana").status_code == 200


def test_a_client_cannot_supply_the_display_state_or_other_fields(api):
    d = api["first"](blind=False)
    r = api["call"]("POST", f"/v1/review/cases/{d['decision_id']}/actions", "ana",
                    {"action": "approve", "reason_code": "x", "display_state": {"score_shown": False}})
    assert r.status_code == 422 and r.json()["detail"]["code"] == "unknown_fields"
    assert api["call"]("POST", f"/v1/review/cases/{d['decision_id']}/actions", "ana", {}).status_code == 422


def test_action_errors_map_to_http_codes(api):
    d = api["first"](blind=False)
    base = f"/v1/review/cases/{d['decision_id']}/actions"
    r = api["call"]("POST", base, "ana", {"action": "override", "override_to": "approve"})
    assert r.status_code == 400 and r.json()["detail"]["code"] == "reason_required"
    assert api["call"]("POST", base, "ana", {"action": "fly"}).status_code == 400
    assert api["call"]("POST", base, "mgr", {"action": "approve"}).status_code == 403
    assert api["call"]("POST", "/v1/review/cases/00000000-0000-0000-0000-000000000000/actions", "ana",
                       {"action": "approve", "reason_code": "x"}).status_code == 404
    ok = api["call"]("POST", base, "ana", {"action": "approve", "reason_code": "x", "confidence": 0.9,
                                           "checklist": {"a": True}})
    assert ok.status_code == 201 and ok.json()["feedback"]["formula_version"] == "fqs-0"
    again = api["call"]("POST", base, "ana", {"action": "block", "reason_code": "x"})
    assert again.status_code == 409 and again.json()["detail"]["code"] == "already_decided"


def test_conflict_and_adjudication_over_the_api(api):
    d = api["first"](blind=False)
    base = f"/v1/review/cases/{d['decision_id']}"
    assert api["call"]("POST", base + "/actions", "ana", {"action": "approve", "reason_code": "a"}).status_code == 201
    res = api["call"]("POST", base + "/actions", "ana2", {"action": "block", "reason_code": "b"}).json()
    assert res["status"] == "conflicted"
    assert api["call"]("POST", base + "/adjudication", "ana", {"final_decision": "block", "rationale": "r"}).status_code == 403
    assert api["call"]("POST", base + "/adjudication", "sen", {"final_decision": "block"}).status_code == 400  # no rationale
    done = api["call"]("POST", base + "/adjudication", "sen", {"final_decision": "block", "rationale": "evidence"})
    assert done.status_code == 201 and done.json()["status"] == "adjudicated"
    assert api["call"]("POST", base + "/actions", "ana2", {"action": "approve", "reason_code": "c"}).status_code == 409


def test_dashboard_and_governance_reads(api, trained):
    for i in range(0, 120, 12):
        api["submit"](i)
    dash = api["call"]("GET", "/v1/dashboard/summary", "mgr")
    assert dash.status_code == 200
    body = dash.json()
    assert {"pending_reviews", "high_trust_error_rate", "insufficient_evidence_rate", "feedback_acceptance",
            "trust_distribution", "model_drift", "data_quality"} <= set(body)
    assert api["call"]("GET", "/v1/dashboard/summary", "ana").status_code == 403
    bundles = api["call"]("GET", "/v1/models/bundles", "appr")
    assert bundles.status_code == 200 and bundles.json()[0]["champion"] is True
    assert bundles.json()[0]["artifact_sha256"] and "thresholds" in bundles.json()[0]
    assert api["call"]("GET", "/v1/models/bundles", "ana").status_code == 403
    assert api["call"]("GET", "/v1/validation/reports", "aud").json() == []  # none stored yet
    rep = {"meta": {"dataset_id": "d", "mode": "provisional", "n_cases": 5, "n_wrong": 1},
           "measures": {}, "baselines": {}, "ablations": {}, "pass_criteria": {}, "limitations": []}
    store_report(api["repo"], T, trained[3].manifest, rep)
    got = api["call"]("GET", "/v1/validation/reports", "aud").json()
    assert len(got) == 1 and got[0]["bundle_id"] == trained[3].manifest.bundle_id
