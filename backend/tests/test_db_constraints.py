"""Rules the database itself enforces, independent of the application (needs a database).

Foreign keys are bypassed with session_replication_role so each CHECK constraint is exercised on
its own. These are the PRD rules that must hold even if the service has a bug.
"""

import uuid

import pytest

psycopg = pytest.importorskip("psycopg")


@pytest.fixture
def db(pg_conn):
    pg_conn.autocommit = True
    pg_conn.execute("SET session_replication_role = replica")  # skip FK triggers only
    return pg_conn


COMMON = "tenant_id, created_by, payload_hash, prev_hash, row_hash"


def base(conn, sql_cols: str, sql_vals: str):
    return conn.execute(f"INSERT INTO {sql_cols} VALUES {sql_vals}")


def policy(conn, automation=0):
    u = uuid.uuid4()
    return conn.execute(
        f"INSERT INTO policy_decisions ({COMMON}, txn_id, trust_assessment_id, policy_version, risk_band,"
        f" gate, recommended_action, automation_level) VALUES ('t','x','p','','', 'tx', %s, 'v', 'low', 'matrix',"
        f" 'approve', %s)", (u, automation))


def trust(conn, state, ti, reasons="[]", version=1):
    return conn.execute(
        f"INSERT INTO trust_assessments ({COMMON}, prediction_id, version_no, mode, state, ti, reason_codes,"
        f" components, weights_version, weights_used, evidence) VALUES ('t','x','p','','', %s, %s, 'provisional',"
        f" %s, %s, %s::jsonb, '{{}}', 'w', '{{}}', '{{}}')", (uuid.uuid4(), version, state, ti, reasons))


def action(conn, act, reason=None, final=None):
    return conn.execute(
        f"INSERT INTO analyst_actions ({COMMON}, txn_id, policy_decision_id, analyst_pid, role, action,"
        f" final_decision, reason_code, display_state, blind_flag, action_time) VALUES ('t','x','p','','',"
        f" 'tx', %s, 'a', 'analyst', %s, %s, %s, '{{}}', false, now())", (uuid.uuid4(), act, final, reason))


def test_automation_above_level_zero_cannot_be_stored(db):
    policy(db, 0)  # allowed
    with pytest.raises(psycopg.errors.CheckViolation):
        policy(db, 1)  # the MVP is recommend-only, enforced by the database (FR-23)


def test_a_score_cannot_be_manufactured_for_insufficient_evidence(db):
    trust(db, "moderate", 55.0)
    trust(db, "insufficient_evidence", None, '["THIN_COHORT"]', version=2)
    with pytest.raises(psycopg.errors.CheckViolation):
        trust(db, "insufficient_evidence", 55.0, '["THIN_COHORT"]', version=3)  # a number with no score state
    with pytest.raises(psycopg.errors.CheckViolation):
        trust(db, "moderate", None, version=4)  # a scored state with no number
    with pytest.raises(psycopg.errors.CheckViolation):
        trust(db, "insufficient_evidence", None, "[]", version=5)  # no reason given


def test_an_override_needs_a_reason_and_a_final_decision(db):
    action(db, "override", reason="customer confirmed", final="approve")
    with pytest.raises(psycopg.errors.CheckViolation):
        action(db, "override", reason=None, final="approve")  # FR-26, enforced by the database
    with pytest.raises(psycopg.errors.CheckViolation):
        action(db, "override", reason="r", final=None)
    with pytest.raises(psycopg.errors.CheckViolation):
        action(db, "adjudicate", reason="r", final=None)
    with pytest.raises(psycopg.errors.CheckViolation):
        action(db, "approve", final="maybe")  # only approve or block are decisions


def test_unknown_action_kinds_are_rejected(db):
    with pytest.raises(psycopg.errors.CheckViolation):
        action(db, "delete_everything")


def test_one_trust_version_per_prediction(db):
    pid = uuid.uuid4()
    sql = (f"INSERT INTO trust_assessments ({COMMON}, prediction_id, version_no, mode, state, ti, reason_codes,"
           f" components, weights_version, weights_used, evidence) VALUES ('t','x','p','','',%s,%s,'provisional',"
           f" 'moderate', 50, '[]', '{{}}', 'w', '{{}}', '{{}}')")
    db.execute(sql, (pid, 1))
    db.execute(sql, (pid, 2))  # a new version is a new row
    with pytest.raises(psycopg.errors.UniqueViolation):
        db.execute(sql, (pid, 1))  # the same version twice is not