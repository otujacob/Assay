"""/v1/learning: who may read, who may act, and the full governed path through the API."""

import json
from dataclasses import replace
from datetime import datetime, timedelta

import pytest
from fastapi.testclient import TestClient

from assay.api import Credential, Credentials, create_app, sign
from assay.detection import load_bundle, save_bundle
from assay.ingestion import IngestionConfig, IngestionService, InMemoryRepository
from assay.learning.lifecycle import LifecycleConfig, LifecycleStore
from assay.learning.service import LearningConfig
from assay.scoring import BundleRegistry, ScoringConfig, ScoringService
from assay.synthetic import START, to_wire

T = "tenant-synth"
ROLES = {"ing": {"ingest"}, "adm": {"admin"}, "apr": {"approver"}, "apr2": {"approver"}, "aud": {"auditor"},
         "mgr": {"manager"}, "ana": {"analyst"}}
SEC = {k: k.encode() + b"-secret" for k in (*ROLES, "oth")}


@pytest.fixture(scope="module")
def artefact(trained, tmp_path_factory):
    d = tmp_path_factory.mktemp("lcapi") / "b"
    save_bundle(d, trained[3].scoring_bundle(), trained[3].manifest, b"k")
    return load_bundle(d, T, b"k")


@pytest.fixture
def api(trained, artefact):
    _, txns, _, r = trained
    repo, clock = InMemoryRepository(), {"t": START}
    reg = BundleRegistry()
    obj, manifest = artefact
    reg.register(T, obj, manifest)
    reg.register(T, obj, replace(manifest, bundle_id="cand-1"), champion=False)
    ing = IngestionService(repo, IngestionConfig(clock=lambda: clock["t"]))
    scoring = ScoringService(repo, reg, ScoringConfig(clock=lambda: clock["t"]))
    cfg = LearningConfig(lifecycle=LifecycleConfig(min_shadow_days=0.0, min_shadow_cases=2, min_canary_cases=2,
                                                   clock=lambda: clock["t"]))
    creds = Credentials([Credential(k, T, SEC[k], frozenset(v)) for k, v in ROLES.items()]
                        + [Credential("oth", "tenant-other", SEC["oth"], frozenset({"admin", "approver", "auditor"}))])
    client = TestClient(create_app(ing, creds, scoring=scoring, rate_limit_per_minute=None, learning=cfg))
    by_id, ids = {t["txn_id"]: t for t in txns}, r.test_table.txn_ids

    def call(method, path, key, payload=None):
        body = b"" if payload is None else json.dumps(payload).encode()
        return client.request(method, path, content=body, headers={
            "X-Assay-Key": key, "X-Assay-Signature": sign(SEC[key], body)})

    def submit(i):
        t = to_wire(by_id[ids[i]])
        clock["t"] = max(clock["t"], datetime.fromisoformat(t["event_time"]) + timedelta(seconds=2))
        return call("POST", "/v1/transactions", "ing", t).json()["decision"]

    def candidate(creator="u:adm"):
        gates = {"version": "gates-0", "failed": [], "unresolved": ["G4", "G6"], "gates": [
            {"id": "G1", "status": "pass", "detail": ""}, {"id": "G4", "status": "not_evaluated", "detail": ""},
            {"id": "G6", "status": "review_required", "detail": ""}]}
        LifecycleStore(repo, cfg.lifecycle).create(
            T, creator, candidate_id="cand-1", base_bundle_id=manifest.bundle_id, artefact_path=None,
            pool_summary={"accepted": 3}, training={"dataset_id": "d"}, gates=gates)

    return {"call": call, "submit": submit, "candidate": candidate, "clock": clock, "repo": repo,
            "champion": manifest.bundle_id, "start": lambda: clock.__setitem__(
                "t", datetime.fromisoformat(to_wire(by_id[ids[0]])["event_time"]) - timedelta(minutes=1))}


def test_reading_is_limited_to_the_roles_that_govern_models(api):
    api["candidate"]()
    for key, status_ok, cands_ok in (("adm", 200, 200), ("apr", 200, 200), ("aud", 200, 200),
                                     ("mgr", 200, 403), ("ana", 403, 403), ("ing", 403, 403)):
        assert api["call"]("GET", "/v1/learning/status", key).status_code == status_ok, key
        assert api["call"]("GET", "/v1/learning/candidates", key).status_code == cands_ok, key
    assert api["call"]("GET", "/v1/learning/feedback-pool", "ana").status_code == 403
    pool = api["call"]("GET", "/v1/learning/feedback-pool", "mgr").json()
    assert pool["cases"] == 0 and "unverified labels only count" in pool["note"]


def test_only_the_right_roles_may_take_each_step(api):
    api["candidate"]()
    cid = "/v1/learning/candidates/cand-1"
    for path, body in ((f"{cid}/approve", {}), (f"{cid}/canary", {"share": 0.1}), (f"{cid}/promote", None),
                       (f"{cid}/rollback", {"reason": "x"})):
        for key in ("adm", "aud", "mgr", "ana"):
            assert api["call"]("POST", path, key, body).status_code == 403, (path, key)
    assert api["call"]("POST", f"{cid}/shadow", "aud").status_code == 403
    assert api["call"]("POST", f"{cid}/reject", "aud", {"reason": "x"}).status_code == 403


def test_other_tenants_see_nothing_and_unsigned_requests_are_refused(api):
    api["candidate"]()
    assert api["call"]("GET", "/v1/learning/candidates", "oth").json() == {"items": []}
    r = api["call"]("GET", "/v1/learning/candidates/cand-1", "oth")
    assert r.status_code == 404
    assert api["call"]("POST", "/v1/learning/candidates/cand-1/shadow", "oth").status_code == 404
    bare = TestClient(create_app(IngestionService(InMemoryRepository()), Credentials([])))
    assert bare.get("/v1/learning/status").status_code in (401, 404)


def test_the_full_path_through_the_api_and_every_guard_on_the_way(api):
    api["candidate"]()
    cid = "/v1/learning/candidates/cand-1"
    got = api["call"]("GET", cid, "aud").json()
    assert got["state"] == "validated" and got["gates"]["unresolved"] == ["G4", "G6"]

    api["start"]()
    assert api["call"]("POST", f"{cid}/shadow", "adm").json()["state"] == "shadow"
    r = api["call"]("POST", f"{cid}/approve", "apr", {"waived": ["G4"], "reviewed_segments": True, "rationale": "x"})
    assert r.status_code == 409 and r.json()["detail"]["code"] == "shadow_not_passed"  # nothing scored yet
    for i in range(4):
        assert api["submit"](i)["bundle_id"] == api["champion"]
    assert api["call"]("GET", cid, "aud").json()["shadow"]["n"] == 4
    listed = api["call"]("GET", "/v1/learning/candidates", "aud").json()["items"][0]  # the web app reads the list
    assert listed["state"] == "shadow" and listed["shadow"]["n"] == 4 and listed["shadow"]["status"] in ("pass", "pending")

    assert api["call"]("POST", f"{cid}/approve", "apr", {"waived": ["G4"], "nonsense": 1}).status_code == 422
    r = api["call"]("POST", f"{cid}/approve", "apr", {"waived": [], "reviewed_segments": True, "rationale": "x"})
    assert r.status_code == 422 and r.json()["detail"]["code"] == "unresolved_gates"
    r = api["call"]("POST", f"{cid}/approve", "apr", {"waived": ["G4"], "reviewed_segments": True, "rationale": "ok"})
    assert r.status_code == 200 and r.json()["state"] == "approved"

    assert api["call"]("POST", f"{cid}/canary", "apr", {}).status_code == 422
    assert api["call"]("POST", f"{cid}/canary", "apr", {"share": 0.9}).status_code == 422
    assert api["call"]("POST", f"{cid}/canary", "apr", {"share": 0.5}).json()["state"] == "canary"
    assert api["call"]("GET", "/v1/learning/status", "aud").json()["canary"] == {"candidate": "cand-1", "share": 0.5}
    seen = {api["submit"](i)["bundle_id"] for i in range(4, 30)}
    assert seen == {api["champion"], "cand-1"}  # both decide while the canary runs

    assert api["call"]("POST", f"{cid}/promote", "apr").json()["state"] == "champion"
    assert api["call"]("GET", "/v1/learning/status", "aud").json()["champion"] == "cand-1"
    assert {api["submit"](i)["bundle_id"] for i in range(30, 34)} == {"cand-1"}

    assert api["call"]("POST", f"{cid}/rollback", "apr", {}).status_code == 422  # a reason is mandatory
    assert api["call"]("POST", f"{cid}/rollback", "apr2", {"reason": "calibration breach"}).json()["state"] == "rolled_back"
    assert api["call"]("GET", "/v1/learning/status", "aud").json()["champion"] == api["champion"]
    kinds = [e["kind"] for e in api["call"]("GET", cid, "aud").json()["events"]]
    assert kinds == ["validated", "shadow_started", "approved", "canary_started", "promoted", "rolled_back"]


def test_the_person_who_created_a_candidate_cannot_approve_it(api):
    api["candidate"](creator="u:apr")
    cid = "/v1/learning/candidates/cand-1"
    api["start"]()
    api["call"]("POST", f"{cid}/shadow", "adm")
    for i in range(3):
        api["submit"](i)
    r = api["call"]("POST", f"{cid}/approve", "apr", {"waived": ["G4"], "reviewed_segments": True, "rationale": "mine"})
    assert r.status_code == 403 and r.json()["detail"]["code"] == "separation_of_duties"
    assert api["call"]("POST", f"{cid}/approve", "apr2", {"waived": ["G4"], "reviewed_segments": True,
                                                         "rationale": "reviewed"}).status_code == 200
