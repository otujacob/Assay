"""Analyst flags on entity-graph relationships (PRD 13.7) on real PostgreSQL: append-only, tenant-isolated, and refused
by the database when malformed, even if the service is bypassed. Needs ASSAY_TEST_DATABASE_URL."""

from datetime import UTC, datetime

import pytest

from assay.graph.service import GraphCache
from assay.ingestion.pg_repo import PostgresRepository

psycopg = pytest.importorskip("psycopg")
T = "tenant-synth"
RAW = "tenant_id, created_by, payload_hash, prev_hash, row_hash"  # the chain trigger fills the hashes
COLS = f"INSERT INTO graph_edge_flags ({RAW}, rel, src, dst, reason, flagged_at) VALUES ('{T}', %s, '', '', '', %s, %s, %s, %s, now())"


@pytest.fixture
def db(pg_conn):
    pg_conn.autocommit = True
    return pg_conn


def test_a_flag_is_stored_chained_and_read_back_through_the_repository(db):
    repo = PostgresRepository(db)
    with repo.atomic(T):
        repo.insert(T, "graph_edge_flags", {"schema_version": "gra-1", "rel": "PAID", "src": "c-1", "dst": "b-1",
                                            "reason": "family account", "flagged_at": datetime(2025, 1, 5, tzinfo=UTC)}, "u:ana")
    rows = repo.find(T, "graph_edge_flags")
    assert len(rows) == 1 and rows[0]["reason"] == "family account" and rows[0]["created_by"] == "u:ana"
    assert repo.verify(T, "graph_edge_flags") is None
    assert repo.find("other-tenant", "graph_edge_flags") == []


def test_the_database_refuses_an_unknown_relationship_and_an_empty_reason(db):
    with pytest.raises(psycopg.errors.CheckViolation):
        db.execute(COLS, ("u:ana", "OWNS", "c-1", "a-1", "why"))
    with pytest.raises(psycopg.errors.CheckViolation):
        db.execute(COLS, ("u:ana", "PAID", "c-1", "b-1", ""))
    db.execute(COLS, ("u:ana", "PAID", "c-1", "b-1", "why"))  # a well-formed one goes through


def test_flags_cannot_be_changed_or_removed(db):
    db.execute(COLS, ("u:ana", "PAID", "c-1", "b-1", "why"))
    with pytest.raises(psycopg.Error):
        db.execute("UPDATE graph_edge_flags SET reason = 'changed'")
    with pytest.raises(psycopg.Error):
        db.execute("DELETE FROM graph_edge_flags")
    with pytest.raises(psycopg.Error):
        db.execute("TRUNCATE graph_edge_flags")


def test_the_graph_cache_applies_stored_flags_to_the_graph(db):
    repo = PostgresRepository(db)
    with repo.atomic(T):
        repo.insert(T, "graph_edge_flags", {"schema_version": "gra-1", "rel": "PAID", "src": "c-1", "dst": "b-1",
                                            "reason": "x", "flagged_at": datetime(2025, 1, 5, tzinfo=UTC)}, "u:ana")
    with repo.atomic(T):
        _, txns, flags = GraphCache().get(repo, T)
    assert txns == [] and len(flags) == 1  # no such edge in an empty graph: the flag is stored, and harmless
