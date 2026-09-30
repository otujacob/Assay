"""The API on pooled Postgres connections. Needs ASSAY_TEST_DATABASE_URL."""

import json
import os
from concurrent.futures import ThreadPoolExecutor

import pytest
from fastapi.testclient import TestClient

from assay.api import Credential, Credentials, create_app, sign
from assay.api.main import make_provider, refuse_superuser
from assay.ingestion.pg_repo import PostgresRepository

psycopg = pytest.importorskip("psycopg")
pool_mod = pytest.importorskip("psycopg_pool")
pytestmark = pytest.mark.skipif(not os.environ.get("ASSAY_TEST_DATABASE_URL"),
                                reason="ASSAY_TEST_DATABASE_URL not set")

TXN = {"event_id": "e1", "txn_id": "t1", "event_time": "2025-06-01T10:00:00Z", "amount": "9.99",
       "currency": "GBP", "channel": "web", "customer_pid": "c1", "account_pid": "a1",
       "schema_version": "txn-1"}
SECRETS = {"ka": b"sa", "kb": b"sb"}


def post(client, payload, key="ka"):
    body = json.dumps(payload).encode()
    return client.post("/v1/transactions", content=body, headers={
        "X-Assay-Key": key, "X-Assay-Signature": sign(SECRETS[key], body)})


@pytest.fixture
def env(pg_conn):
    schema = pg_conn.schema
    pg_conn.commit()
    pg_conn.autocommit = True
    pg_conn.execute("DO $$ BEGIN IF NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'assay_login_test')"
                    " THEN CREATE ROLE assay_login_test LOGIN NOSUPERUSER NOBYPASSRLS; END IF; END $$")
    pg_conn.execute("GRANT assay_app TO assay_login_test")
    pg_conn.autocommit = False

    def configure(conn):  # behave like a plain login role, never a superuser
        conn.execute("SET SESSION AUTHORIZATION assay_login_test")
        conn.commit()

    pool = pool_mod.ConnectionPool(
        os.environ["ASSAY_TEST_DATABASE_URL"], min_size=1, max_size=8, configure=configure,
        kwargs={"options": f"-c search_path={schema}"}, open=True)
    refuse_superuser(pool)  # passes: current_user is the ordinary role
    creds = Credentials([Credential("ka", "A", SECRETS["ka"]), Credential("kb", "B", SECRETS["kb"])])
    client = TestClient(create_app(make_provider(pool), creds))
    yield client, PostgresRepository(pg_conn, role=None)
    pool.close()


def test_request_is_stored_under_the_credentials_tenant(env):
    client, reader = env
    r = post(client, TXN)
    assert r.status_code == 202 and r.json()["status"] == "accepted"
    assert len(reader.rows("A", "transactions")) == 1 and reader.rows("B", "transactions") == []
    assert reader.verify("A", "audit_log") is None


def test_healthz_needs_no_credentials(env):
    client, _ = env
    assert client.get("/healthz").json() == {"status": "ok"}


def test_same_ids_in_two_tenants(env):
    client, reader = env
    assert post(client, TXN, "ka").json()["status"] == "accepted"
    assert post(client, TXN, "kb").json()["status"] == "accepted"
    assert len(reader.rows("A", "transactions")) == len(reader.rows("B", "transactions")) == 1


def test_concurrent_identical_posts_store_exactly_one(env):
    client, reader = env
    with ThreadPoolExecutor(8) as ex:
        statuses = list(ex.map(lambda _: post(client, TXN).json()["status"], range(8)))
    assert statuses.count("accepted") == 1 and statuses.count("duplicate") == 7
    assert len(reader.rows("A", "transactions")) == 1
    assert len(reader.rows("A", "ingestion_events")) == 1
    assert reader.verify("A", "ingestion_events") is None


def test_service_refuses_to_run_as_superuser():
    pool = pool_mod.ConnectionPool(os.environ["ASSAY_TEST_DATABASE_URL"], min_size=1, open=True)
    try:
        with pytest.raises(RuntimeError, match="non-superuser"):
            refuse_superuser(pool)
    finally:
        pool.close()
