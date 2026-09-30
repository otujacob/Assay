"""The scoring API end to end: signed requests, tenants, roles, lineage and replay."""

import json
from datetime import datetime, timedelta

import pytest
from fastapi.testclient import TestClient

from assay.api import Credential, Credentials, create_app, sign
from assay.detection import load_bundle, save_bundle
from assay.ingestion import IngestionConfig, IngestionService, InMemoryRepository
from assay.scoring import BundleRegistry, ScoringConfig, ScoringService
from assay.synthetic import START, to_wire

KEY = b"k"
SECRETS = {"ing": b"s1", "ana": b"s2", "aud": b"s3", "other": b"s4"}
ROLES = {"ing": {"ingest"}, "ana": {"analyst"}, "aud": {"auditor", "analyst"}, "other": {"auditor"}}


@pytest.fixture(scope="module")
def artefact(trained, tmp_path_factory):
    _, _, _, r = trained
    d = tmp_path_factory.mktemp("apibundle") / "b"
    save_bundle(d, r.scoring_bundle(), r.manifest, KEY)
    return load_bundle(d, "tenant-synth", KEY)


@pytest.fixture
def api(trained, artefact):
    _, txns, _, r = trained
    repo = InMemoryRepository()
    clock = {"t": START}
    reg = BundleRegistry()
    reg.register("tenant-synth", *artefact)
    ing = IngestionService(repo, IngestionConfig(clock=lambda: clock["t"]))
    scoring = ScoringService(repo, reg, ScoringConfig(clock=lambda: clock["t"]))
    creds = Credentials([
        Credential("ing", "tenant-synth", SECRETS["ing"], frozenset(ROLES["ing"])),
        Credential("ana", "tenant-synth", SECRETS["ana"], frozenset(ROLES["ana"])),
        Credential("aud", "tenant-synth", SECRETS["aud"], frozenset(ROLES["aud"])),
        Credential("other", "tenant-other", SECRETS["other"], frozenset(ROLES["other"])),
    ])
    client = TestClient(create_app(ing, creds, scoring=scoring))
    by_id = {t["txn_id"]: t for t in txns}
    ids = r.test_table.txn_ids

    def call(method, path, key="ing", payload=None):
        body = b"" if payload is None else json.dumps(payload).encode()
        return client.request(method, path, content=body, headers={
            "X-Assay-Key": key, "X-Assay-Signature": sign(SECRETS[key], body)})

    def submit(i=0, **kw):
        t = to_wire(by_id[ids[i]])
        clock["t"] = datetime.fromisoformat(t["event_time"]) + timedelta(seconds=2)
        return call("POST", "/v1/transactions", payload=t, **kw)

    return {"call": call, "submit": submit, "repo": repo, "ids": ids, "reg": reg, "creds": creds,
            "app": create_app(ing, creds)}


def test_posting_a_transaction_returns_a_decision_in_the_documented_shape(api):
    r = api["submit"]()
    assert r.status_code == 202 and r.json()["status"] == "accepted"
    d = r.json()["decision"]
    assert {"decision_id", "txn_id", "risk", "risk_band", "trust", "recommendation", "automation_level",
            "bundle_id", "policy_version", "explanation_status"} <= set(d)
    assert {"state", "mode", "ti", "ti_low", "ti_high", "reason_codes", "components"} <= set(d["trust"])
    assert d["automation_level"] == 0 and d["trust"]["mode"] == "provisional"
    assert 0 <= d["risk"] <= 1


def test_a_resend_returns_the_same_decision(api):
    a, b = api["submit"](), api["submit"]()
    assert b.json()["status"] == "duplicate"
    assert a.json()["decision"]["decision_id"] == b.json()["decision"]["decision_id"]
    assert len(api["repo"].rows("tenant-synth", "predictions")) == 1


def test_read_endpoints_enforce_roles(api):
    d = api["submit"]().json()["decision"]["decision_id"]
    assert api["call"]("GET", f"/v1/decisions/{d}", "ana").status_code == 200
    assert api["call"]("GET", f"/v1/decisions/{d}", "ing").status_code == 403  # ingest-only key
    assert api["call"]("GET", f"/v1/decisions/{d}/lineage", "ana").status_code == 403  # not an auditor
    assert api["call"]("POST", f"/v1/decisions/{d}/replay", "ana").status_code == 403
    assert api["call"]("POST", "/v1/transactions", "aud", payload={}).status_code == 403  # read-only key
    assert api["call"]("GET", f"/v1/decisions/{d}", "ana").json()["decision_id"] == d


def test_a_decision_is_invisible_to_another_tenant(api):
    d = api["submit"]().json()["decision"]["decision_id"]
    r = api["call"]("GET", f"/v1/decisions/{d}/lineage", "other")
    assert r.status_code == 404
    assert api["call"]("GET", "/v1/decisions/does-not-exist", "aud").status_code == 404


def test_lineage_and_replay_for_an_auditor(api):
    d = api["submit"]().json()["decision"]["decision_id"]
    lin = api["call"]("GET", f"/v1/decisions/{d}/lineage", "aud").json()
    assert lin["prediction"]["bundle_id"] and lin["model_bundle"]["artifact_sha256"]
    assert lin["feature_vector"]["feature_values"]
    rep = api["call"]("POST", f"/v1/decisions/{d}/replay", "aud").json()
    assert rep["reproducible"] is True and all(rep["checks"].values())


def test_explanation_endpoint_reports_pending_then_available(api):
    # a low-risk case outside the audit sample has no explanation until refined
    for i in range(60):
        r = api["submit"](i)
        d = r.json()["decision"]
        if d["explanation_status"] == "pending":
            assert api["call"]("GET", f"/v1/decisions/{d['decision_id']}/explanation", "ana").status_code == 404
            break
    else:
        pytest.fail("no pending case found")
    for i in range(60):
        r = api["submit"](i)
        d = r.json()["decision"]
        if d["explanation_status"] == "computed":
            e = api["call"]("GET", f"/v1/decisions/{d['decision_id']}/explanation", "ana")
            assert e.status_code == 200 and "attributions" in e.json() and e.json()["reproducible"] is True
            return
    pytest.fail("no explained case found")


def test_no_champion_means_503_but_the_transaction_is_kept(trained):
    _, txns, _, r = trained
    repo = InMemoryRepository()
    clock = {"t": START}
    ing = IngestionService(repo, IngestionConfig(clock=lambda: clock["t"]))
    scoring = ScoringService(repo, BundleRegistry(), ScoringConfig(clock=lambda: clock["t"]))
    creds = Credentials([Credential("ing", "tenant-synth", b"s", frozenset({"ingest"}))])
    client = TestClient(create_app(ing, creds, scoring=scoring))
    t = to_wire({t["txn_id"]: t for t in txns}[r.test_table.txn_ids[0]])
    clock["t"] = datetime.fromisoformat(t["event_time"]) + timedelta(seconds=2)
    body = json.dumps(t).encode()
    resp = client.post("/v1/transactions", content=body,
                       headers={"X-Assay-Key": "ing", "X-Assay-Signature": sign(b"s", body)})
    assert resp.status_code == 503 and resp.json()["detail"]["code"] == "scoring_unavailable"
    assert repo.get_transaction("tenant-synth", t["txn_id"]) is not None  # still stored for retry


def test_ingest_only_deployments_still_work_without_scoring(api):
    r = TestClient(api["app"]).post(
        "/v1/transactions", content=b"{}",
        headers={"X-Assay-Key": "ing", "X-Assay-Signature": sign(SECRETS["ing"], b"{}")})
    assert r.status_code == 422  # validation error, not a crash
