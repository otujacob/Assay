"""Guarantees that only the Postgres repository can provide. Needs ASSAY_TEST_DATABASE_URL."""

from datetime import UTC, datetime

import pytest

from assay.ingestion import IngestionConfig, IngestionService
from assay.ingestion.pg_repo import PostgresRepository
from assay.ingestion.repo import DuplicateError

psycopg = pytest.importorskip("psycopg")

T0 = datetime(2025, 6, 1, 12, 0, tzinfo=UTC)
TXN = {"event_id": "e1", "txn_id": "t1", "event_time": "2025-06-01T10:00:00Z", "amount": "12.50",
       "currency": "GBP", "channel": "app", "customer_pid": "c1", "account_pid": "a1",
       "schema_version": "txn-1"}


@pytest.fixture
def repo(pg_conn):
    return PostgresRepository(pg_conn)


@pytest.fixture
def svc(repo):
    return IngestionService(repo, IngestionConfig(clock=lambda: T0))


def test_repository_level_duplicates_raise(repo):
    repo.append_ingestion_event("A", event_id="e1", kind="transaction", source_id="s", event_time=T0,
                                recorded_at=T0, event_payload_hash="h", schema_version="txn-1",
                                validation_result="accepted", reasons=[], created_by="t")
    with pytest.raises(DuplicateError):
        repo.append_ingestion_event("A", event_id="e1", kind="transaction", source_id="s",
                                    event_time=T0, recorded_at=T0, event_payload_hash="h",
                                    schema_version="txn-1", validation_result="accepted",
                                    reasons=[], created_by="t")
    # The failed insert must not poison the connection or leave rows behind.
    assert len(repo.rows("A", "ingestion_events")) == 1


def test_concurrent_duplicate_is_answered_from_stored_data(svc, repo):
    assert svc.ingest_transaction("A", TXN).status == "accepted"

    class Blind(PostgresRepository):
        """Simulates the race: the pre-check does not see the concurrent writer's row."""
        def __init__(self, inner):
            self.__dict__ = inner.__dict__
            self.calls = 0

        def get_ingestion_event(self, tenant_id, event_id):
            self.calls += 1
            return None if self.calls == 1 else super().get_ingestion_event(tenant_id, event_id)

        def get_transaction(self, tenant_id, txn_id):
            return None if self.calls == 1 else super().get_transaction(tenant_id, txn_id)

    racer = IngestionService(Blind(repo), IngestionConfig(clock=lambda: T0))
    assert racer.ingest_transaction("A", TXN).status == "duplicate"
    assert len(repo.rows("A", "transactions")) == 1
    assert len(repo.rows("A", "ingestion_events")) == 1


def test_writes_are_all_or_nothing(svc, repo, monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("crash between writes")

    monkeypatch.setattr(repo, "append_transaction", boom)
    with pytest.raises(RuntimeError):
        svc.ingest_transaction("A", TXN)
    monkeypatch.undo()
    assert repo.rows("A", "ingestion_events") == []  # no orphan "accepted" event
    assert repo.rows("A", "audit_log") == []
    assert svc.ingest_transaction("A", TXN).status == "accepted"  # retry works normally


def test_database_enforces_isolation_even_without_a_tenant_filter(svc, repo):
    svc.ingest_transaction("A", TXN)
    with repo.atomic("B"):  # a buggy query with NO tenant filter, running as tenant B
        n = repo.conn.execute("SELECT count(*) AS n FROM transactions").fetchone()["n"]
        assert n == 0
    with repo.atomic("A"):
        assert repo.conn.execute("SELECT count(*) AS n FROM transactions").fetchone()["n"] == 1


def test_app_role_cannot_mutate_or_bypass(svc, repo):
    svc.ingest_transaction("A", TXN)
    with pytest.raises(psycopg.errors.InsufficientPrivilege), repo.atomic("A"):
        repo.conn.execute("UPDATE transactions SET amount = 1")
    with pytest.raises(psycopg.errors.InsufficientPrivilege), repo.atomic("A"):
        repo.conn.execute("DELETE FROM audit_log")


def test_ordinary_login_role_cannot_escalate(svc, repo):
    """The app must connect as a non-superuser (a superuser can always SET ROLE).

    Simulates that login: a plain role that is only a member of assay_app.
    """
    svc.ingest_transaction("A", TXN)
    conn = repo.conn
    conn.commit()
    conn.autocommit = True
    conn.execute("DO $$ BEGIN IF NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'assay_login_test')"
                 " THEN CREATE ROLE assay_login_test LOGIN NOSUPERUSER NOBYPASSRLS; END IF; END $$")
    conn.execute("GRANT assay_app TO assay_login_test")
    conn.execute("SET SESSION AUTHORIZATION assay_login_test")
    conn.autocommit = False
    try:
        assert len(repo.rows("A", "transactions")) == 1  # normal operation works
        for stmt in ("SET LOCAL ROLE postgres", "ALTER TABLE transactions DISABLE ROW LEVEL SECURITY",
                     "ALTER TABLE transactions DISABLE TRIGGER ALL"):
            with pytest.raises(psycopg.errors.InsufficientPrivilege), repo.atomic("A"):
                conn.execute(stmt)
    finally:
        conn.rollback()
        conn.autocommit = True
        conn.execute("RESET SESSION AUTHORIZATION")
        conn.autocommit = False


def test_chains_verify_after_a_mixed_workload(svc, repo):
    svc.ingest_transaction("A", TXN)
    svc.ingest_transaction("A", {**TXN, "event_id": "bad", "amount": "-1"})
    svc.ingest_outcome("A", {"event_id": "o1", "txn_id": "t1", "outcome_type": "confirmed_fraud",
                             "source": "cb", "event_time": "2025-06-05T00:00:00Z",
                             "schema_version": "out-1"})
    svc.ingest_outcome("A", {"event_id": "o2", "txn_id": "ghost", "outcome_type": "chargeback",
                             "source": "cb", "event_time": "2025-06-05T00:00:00Z",
                             "schema_version": "out-1"})
    for table in ("ingestion_events", "transactions", "outcomes", "quarantine", "audit_log"):
        assert repo.verify("A", table) is None, table
    assert repo.count_by_validation("A") == {"accepted": 2, "rejected": 1, "quarantined": 1}
