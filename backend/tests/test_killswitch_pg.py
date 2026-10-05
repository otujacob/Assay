"""The kill switch log on real PostgreSQL: append-only, tenant-isolated, reason required. Needs ASSAY_TEST_DATABASE_URL."""

import pytest

from assay.ingestion.pg_repo import PostgresRepository
from assay.learning.killswitch import KillSwitch

psycopg = pytest.importorskip("psycopg")
T = "tenant-synth"
RAW = "tenant_id, created_by, payload_hash, prev_hash, row_hash"
INS = f"INSERT INTO kill_switch_events ({RAW}, kind, reason) VALUES ('{T}', %s, '', '', '', %s, %s)"


@pytest.fixture
def db(pg_conn):
    pg_conn.autocommit = True
    return pg_conn


def test_engage_and_release_work_through_the_repository_and_the_chain_verifies(db):
    repo = PostgresRepository(db)
    ks = KillSwitch(repo)
    assert ks.engage(T, "u:a", "incident")["engaged"] is True
    assert ks.release(T, "u:b", "fixed")["engaged"] is False
    assert [h["kind"] for h in ks.history(T)] == ["engaged", "released"]
    assert repo.verify(T, "kill_switch_events") is None
    assert ks.state("other-tenant") == {"engaged": False}


def test_the_database_refuses_an_unknown_kind_and_a_blank_reason_and_any_change(db):
    with pytest.raises(psycopg.errors.CheckViolation):
        db.execute(INS, ("u:a", "paused", "x"))
    with pytest.raises(psycopg.errors.CheckViolation):
        db.execute(INS, ("u:a", "engaged", "   "))
    db.execute(INS, ("u:a", "engaged", "x"))
    for sql in ("UPDATE kill_switch_events SET reason = 'y'", "DELETE FROM kill_switch_events", "TRUNCATE kill_switch_events"):
        with pytest.raises(psycopg.Error):
            db.execute(sql)
