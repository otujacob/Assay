"""FR-21 and FR-34 on real PostgreSQL: the database itself must refuse what the service refuses,
and the audit search must work on the stored chain. Needs ASSAY_TEST_DATABASE_URL."""

import pytest

from assay import audit
from assay.ingestion.pg_repo import PostgresRepository
from assay.policy import PolicyError, PolicyService

psycopg = pytest.importorskip("psycopg")
T = "tenant-synth"
RAW = "tenant_id, created_by, payload_hash, prev_hash, row_hash"  # the chain trigger fills the hashes


@pytest.fixture
def db(pg_conn):
    pg_conn.autocommit = True
    return pg_conn


@pytest.fixture
def svc(db):
    return PolicyService(PostgresRepository(db))


def test_policy_lifecycle_on_postgres(svc):
    p = svc.propose(T, "u:alice", {"dq_gate_action": "hold"})
    assert p["version"] == "policy-1" and p["status"] == "pending" and svc.active(T) is None
    with pytest.raises(PolicyError) as e:
        svc.approve(T, "u:alice", p["id"])
    assert e.value.code == "same_approver"
    assert svc.approve(T, "u:bob", p["id"])["status"] == "approved"
    assert svc.active(T)["payload"]["dq_gate_action"] == "hold"
    with pytest.raises(PolicyError) as e:
        svc.approve(T, "u:carol", p["id"])
    assert e.value.code == "already_approved"
    assert svc.repo.verify(T, "policy_versions") is None and svc.repo.verify(T, "policy_approvals") is None


def test_other_tenants_see_none_of_it(svc):
    svc.propose(T, "u:alice", {})
    assert svc.list("other-tenant") == [] and svc.active("other-tenant") is None


def test_database_refuses_automation_above_level_zero(db):
    with pytest.raises(psycopg.errors.CheckViolation):
        db.execute(f"INSERT INTO policy_versions ({RAW}, version, payload, effective_from) "
                   f"VALUES ('{T}', 'x', '', '', '', 'policy-1', '{{\"automation_level\": 1}}', now())")


def test_database_refuses_self_approval_even_if_the_service_is_bypassed(svc, db):
    pid = svc.propose(T, "u:alice", {})["id"]
    ins = (f"INSERT INTO policy_approvals ({RAW}, policy_version_id, proposed_by, approved_by, approved_at) "
           f"VALUES ('{T}', %s, '', '', '', %s, %s, %s, now())")
    with pytest.raises(psycopg.errors.CheckViolation):
        db.execute(ins, ("u:alice", pid, "u:alice", "u:alice"))  # approver == proposer
    # Claiming a different proposer to dodge that check fails the key: proposed_by must be the real one.
    with pytest.raises(psycopg.errors.ForeignKeyViolation):
        db.execute(ins, ("u:bob", pid, "u:mallory", "u:bob"))
    db.execute(ins, ("u:bob", pid, "u:alice", "u:bob"))  # the honest approval goes through
    with pytest.raises(psycopg.errors.UniqueViolation):
        db.execute(ins, ("u:carol", pid, "u:alice", "u:carol"))  # approved once


def test_policy_tables_are_append_only(svc, db):
    svc.propose(T, "u:alice", {})
    with pytest.raises(psycopg.errors.InsufficientPrivilege):
        db.execute("UPDATE policy_versions SET version = 'policy-9'")
    with pytest.raises(psycopg.errors.InsufficientPrivilege):
        db.execute("DELETE FROM policy_versions")


def test_audit_search_and_chain_check_on_postgres(svc):
    p = svc.propose(T, "u:alice", {})
    svc.approve(T, "u:bob", p["id"])
    out = audit.query(svc.repo, T)
    assert out["chain_ok"] is True and out["matching"] == 2
    assert [i["action"] for i in out["items"]] == ["policy_approve", "policy_propose"]  # newest first
    assert audit.query(svc.repo, T, actor="u:alice")["matching"] == 1
    assert audit.query(svc.repo, T, action="policy_approve", limit=1)["items"][0]["object"] == "policy-1"
    assert audit.query(svc.repo, "other-tenant")["matching"] == 0
