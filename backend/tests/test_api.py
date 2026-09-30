import json

import pytest
from fastapi.testclient import TestClient

from assay.api import Credential, Credentials, create_app, sign
from assay.ingestion import IngestionService, InMemoryRepository

SECRET_A, SECRET_B = b"secret-a", b"secret-b"


@pytest.fixture
def env():
    repo = InMemoryRepository()
    creds = Credentials([Credential("ka", "A", SECRET_A), Credential("kb", "B", SECRET_B)])
    return TestClient(create_app(IngestionService(repo), creds)), repo


def post(client, path, payload, key="ka", secret=SECRET_A, signature=None):
    body = json.dumps(payload).encode()
    return client.post(path, content=body, headers={
        "X-Assay-Key": key, "X-Assay-Signature": signature or sign(secret, body),
        "Content-Type": "application/json"})


TXN = {"event_id": "e1", "txn_id": "t1", "event_time": "2025-06-01T10:00:00Z", "amount": "9.99",
       "currency": "GBP", "channel": "web", "customer_pid": "c1", "account_pid": "a1",
       "schema_version": "txn-1"}


def test_signed_request_accepted_and_tenant_from_credential(env):
    c, repo = env
    r = post(c, "/v1/transactions", TXN)
    assert r.status_code == 202 and r.json()["status"] == "accepted"
    assert repo.get_transaction("A", "t1") and not repo.get_transaction("B", "t1")
    r = post(c, "/v1/transactions", {**TXN, "tenant_id": "B"})  # body cannot choose tenant
    assert r.status_code == 422 and repo.rows("B", "transactions") == []


@pytest.mark.parametrize("kw", [
    {"signature": "0" * 64}, {"key": "nope"}, {"secret": SECRET_B},
])
def test_bad_credentials_rejected_identically(env, kw):
    c, repo = env
    r = post(c, "/v1/transactions", TXN, **kw)
    assert r.status_code == 401 and r.json() == {"detail": {"code": "unauthorized"}}
    assert repo.rows("A", "transactions") == []


def test_missing_headers_rejected(env):
    c, _ = env
    assert c.post("/v1/transactions", json=TXN).status_code == 422


def test_idempotent_resend_over_api(env):
    c, repo = env
    post(c, "/v1/transactions", TXN)
    r = post(c, "/v1/transactions", TXN)
    assert r.status_code == 202 and r.json()["status"] == "duplicate"
    assert len(repo.rows("A", "transactions")) == 1


def test_validation_error_is_422_with_reasons_only(env):
    c, _ = env
    r = post(c, "/v1/transactions", {**TXN, "amount": "-1", "customer_pid": "PERSONAL"})
    assert r.status_code == 422
    assert "PERSONAL" not in r.text and r.json()["detail"]["code"] == "validation_failed"


def test_batch_and_outcomes(env):
    c, _repo = env
    r = post(c, "/v1/transactions:batch", [TXN, {**TXN, "event_id": "e2", "txn_id": "t2", "amount": "x"}])
    assert r.status_code == 207
    assert [x["status"] for x in r.json()["results"]] == ["accepted", "rejected"]
    o = {"event_id": "o1", "txn_id": "t1", "outcome_type": "confirmed_fraud", "source": "cb",
         "event_time": "2025-06-05T00:00:00Z", "schema_version": "out-1"}
    assert post(c, "/v1/outcomes", o).json()["status"] == "accepted"
    assert post(c, "/v1/outcomes", {**o, "event_id": "o2", "txn_id": "ghost"}).json()["status"] == "quarantined"


def test_malformed_body(env):
    c, _ = env
    body = b"{not json"
    r = c.post("/v1/transactions", content=body,
               headers={"X-Assay-Key": "ka", "X-Assay-Signature": sign(SECRET_A, body)})
    assert r.status_code == 400
