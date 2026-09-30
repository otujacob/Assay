"""The scoring API on pooled Postgres connections. Needs ASSAY_TEST_DATABASE_URL."""

import json
import os
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from datetime import datetime, timedelta

import pytest
from fastapi.testclient import TestClient

from assay.api import Credential, Credentials, create_app, sign
from assay.api.main import refuse_superuser
from assay.detection import load_bundle, save_bundle
from assay.ingestion import IngestionConfig, IngestionService
from assay.ingestion.pg_repo import PostgresRepository
from assay.scoring import BundleRegistry, ScoringConfig, ScoringService
from assay.synthetic import to_wire

psycopg = pytest.importorskip("psycopg")
pool_mod = pytest.importorskip("psycopg_pool")
pytestmark = pytest.mark.skipif(not os.environ.get("ASSAY_TEST_DATABASE_URL"),
                                reason="ASSAY_TEST_DATABASE_URL not set")

TENANT = "tenant-synth"
SECRETS = {"ing": b"s1", "aud": b"s2"}


@pytest.fixture(scope="module")
def artefact(trained, tmp_path_factory):
    d = tmp_path_factory.mktemp("pgbundle") / "b"
    save_bundle(d, trained[3].scoring_bundle(), trained[3].manifest, b"k")
    return load_bundle(d, TENANT, b"k")


@pytest.fixture
def env(pg_conn, trained, artefact):
    _, txns, _, r = trained
    schema = pg_conn.schema
    pg_conn.commit()
    pg_conn.autocommit = True
    pg_conn.execute("DO $$ BEGIN IF NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'assay_login_test')"
                    " THEN CREATE ROLE assay_login_test LOGIN NOSUPERUSER NOBYPASSRLS; END IF; END $$")
    pg_conn.execute("GRANT assay_app TO assay_login_test")
    pg_conn.autocommit = False

    def configure(conn):
        conn.execute("SET SESSION AUTHORIZATION assay_login_test")
        conn.commit()

    pool = pool_mod.ConnectionPool(os.environ["ASSAY_TEST_DATABASE_URL"], min_size=1, max_size=8,
                                   configure=configure, kwargs={"options": f"-c search_path={schema}"},
                                   open=True)
    refuse_superuser(pool)
    reg = BundleRegistry()
    reg.register(TENANT, *artefact)
    by_id = {t["txn_id"]: t for t in txns}
    txn = to_wire(by_id[r.test_table.txn_ids[0]])
    fixed = datetime.fromisoformat(txn["event_time"]) + timedelta(seconds=2)

    @contextmanager
    def ingest_provider():
        with pool.connection() as conn:
            yield IngestionService(PostgresRepository(conn), IngestionConfig(clock=lambda: fixed))

    @contextmanager
    def scoring_provider():
        with pool.connection() as conn:
            yield ScoringService(PostgresRepository(conn), reg, ScoringConfig(clock=lambda: fixed))

    creds = Credentials([Credential("ing", TENANT, SECRETS["ing"], frozenset({"ingest"})),
                         Credential("aud", TENANT, SECRETS["aud"], frozenset({"auditor", "analyst"}))])
    client = TestClient(create_app(ingest_provider, creds, scoring=scoring_provider))

    def call(method, path, key, payload=None):
        body = b"" if payload is None else json.dumps(payload).encode()
        return client.request(method, path, content=body, headers={
            "X-Assay-Key": key, "X-Assay-Signature": sign(SECRETS[key], body)})

    yield {"call": call, "txn": txn, "reader": PostgresRepository(pg_conn, role=None)}
    pool.close()


def test_scored_decision_through_pooled_postgres_then_lineage_and_replay(env):
    r = env["call"]("POST", "/v1/transactions", "ing", env["txn"])
    assert r.status_code == 202, r.text
    d = r.json()["decision"]
    assert d["automation_level"] == 0 and d["bundle_id"]
    assert env["call"]("GET", f"/v1/decisions/{d['decision_id']}", "aud").json()["decision_id"] == d["decision_id"]
    lin = env["call"]("GET", f"/v1/decisions/{d['decision_id']}/lineage", "aud").json()
    assert lin["model_bundle"]["artifact_sha256"] and lin["feature_vector"]["feature_values"]
    rep = env["call"]("POST", f"/v1/decisions/{d['decision_id']}/replay", "aud").json()
    assert rep["reproducible"] is True, rep


def test_concurrent_posts_of_one_transaction_produce_exactly_one_decision(env):
    def go(_):
        r = env["call"]("POST", "/v1/transactions", "ing", env["txn"])
        assert r.status_code == 202, r.text
        return r.json()["decision"]["decision_id"]

    with ThreadPoolExecutor(8) as ex:
        ids = set(ex.map(go, range(8)))
    assert len(ids) == 1
    reader = env["reader"]
    for table in ("feature_vectors", "predictions", "policy_decisions", "trust_assessments"):
        assert len(reader.rows(TENANT, table)) == 1, table
        assert reader.verify(TENANT, table) is None, table
    assert len(reader.rows(TENANT, "transactions")) == 1
