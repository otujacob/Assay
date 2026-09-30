"""Executes db/migrations/0001_init.sql against a real PostgreSQL and checks the guarantees.

Skipped unless ASSAY_TEST_DATABASE_URL points at a disposable database on which the
connecting user is a superuser (CI provides one). Each run uses its own schema.
"""

import os
from pathlib import Path

import pytest

from assay.lineage import row_hash

psycopg = pytest.importorskip("psycopg")
URL = os.environ.get("ASSAY_TEST_DATABASE_URL")
pytestmark = pytest.mark.skipif(not URL, reason="ASSAY_TEST_DATABASE_URL not set")

MIGRATIONS = sorted((Path(__file__).parents[2] / "db" / "migrations").glob("*.sql"))


@pytest.fixture(scope="module")
def conn():
    c = psycopg.connect(URL, autocommit=True)
    schema = f"live_{os.getpid()}"
    c.execute(f'CREATE SCHEMA "{schema}"')
    c.execute(f'SET search_path TO "{schema}"')
    for m in MIGRATIONS:
        c.execute(m.read_text(encoding="utf-8"))
    c.execute(f'GRANT USAGE ON SCHEMA "{schema}" TO assay_app')
    yield c
    c.close()


def as_tenant(conn, tenant):
    conn.execute("RESET ROLE")
    conn.execute("SET ROLE assay_app")
    if tenant is not None:
        conn.execute("SELECT set_config('app.tenant_id', %s, false)", (tenant,))
    else:
        conn.execute("RESET app.tenant_id")


def add_event(conn, tenant, event_id, payload_hash="p" * 64, result="accepted"):
    return conn.execute(
        """INSERT INTO ingestion_events (tenant_id, created_by, kind, source_id, event_id,
             recorded_at, event_payload_hash, validation_result, payload_hash, prev_hash, row_hash)
           VALUES (%s, 'test', 'transaction', 'src', %s, now(), %s, %s, %s, '', '')
           RETURNING id::text, prev_hash, row_hash""",
        (tenant, event_id, "h" * 64, result, payload_hash)).fetchone()


def test_chain_matches_python_and_verifies(conn):
    as_tenant(conn, "A")
    ids = [add_event(conn, "A", f"e{i}") for i in range(3)]
    assert ids[0][1] == "0" * 64 and ids[1][1] == ids[0][2]
    assert ids[1][2] == row_hash(ids[0][2], ids[1][0], "A", "p" * 64)  # SQL == Python formula
    assert conn.execute("SELECT assay_verify_chain('ingestion_events', 'A')").fetchone()[0] is None


def test_tenants_cannot_see_or_write_each_other(conn):
    as_tenant(conn, "A")
    add_event(conn, "A", "iso-a")
    as_tenant(conn, "B")
    assert conn.execute("SELECT count(*) FROM ingestion_events WHERE event_id = 'iso-a'").fetchone()[0] == 0
    with pytest.raises(psycopg.errors.InsufficientPrivilege):  # RLS WITH CHECK violation
        add_event(conn, "A", "cross-write")


def test_no_tenant_set_sees_nothing(conn):
    as_tenant(conn, None)
    assert conn.execute("SELECT count(*) FROM ingestion_events").fetchone()[0] == 0


@pytest.mark.parametrize("stmt", [
    "UPDATE ingestion_events SET source_id = 'x'",
    "DELETE FROM ingestion_events",
    "TRUNCATE ingestion_events",
])
def test_append_only(conn, stmt):
    as_tenant(conn, "A")
    with pytest.raises(psycopg.errors.InsufficientPrivilege):
        conn.execute(stmt)


def test_superuser_mutation_still_blocked_by_trigger(conn):
    conn.execute("RESET ROLE")
    with pytest.raises(psycopg.errors.InsufficientPrivilege):
        conn.execute("UPDATE ingestion_events SET source_id = 'x'")


def test_idempotency_index_and_tamper_detection(conn):
    as_tenant(conn, "T")
    add_event(conn, "T", "dup")
    with pytest.raises(psycopg.errors.UniqueViolation):
        add_event(conn, "T", "dup")
    add_event(conn, "T", "dup", result="rejected")  # rejected records may repeat
    # Tamper as superuser by disabling the guard trigger, then verify detects it.
    conn.execute("RESET ROLE")
    conn.execute("ALTER TABLE ingestion_events DISABLE TRIGGER ingestion_events_no_mutate")
    conn.execute("UPDATE ingestion_events SET payload_hash = %s WHERE event_id = 'dup' "
                 "AND validation_result = 'accepted' AND tenant_id = 'T'", ("z" * 64,))
    conn.execute("ALTER TABLE ingestion_events ENABLE TRIGGER ingestion_events_no_mutate")
    as_tenant(conn, "T")
    assert conn.execute("SELECT assay_verify_chain('ingestion_events', 'T')").fetchone()[0] is not None

