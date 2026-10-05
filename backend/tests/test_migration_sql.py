"""Static checks on the Postgres migrations.

This parses the SQL with the real Postgres parser and checks that the security properties are
declared for EVERY table. It does not execute anything. Execution against a live database is
covered by test_db_live.py and every Postgres-backed test (skipped unless ASSAY_TEST_DATABASE_URL
is set).
"""

from pathlib import Path

import pytest

pglast = pytest.importorskip("pglast")

DIR = Path(__file__).parents[2] / "db" / "migrations"
FILES = sorted(DIR.glob("*.sql"))
SQL = {f.name: f.read_text(encoding="utf-8") for f in FILES}
ALL = "\n".join(SQL.values())


def tables_created(sql: str) -> dict[str, set[str]]:
    out = {}
    for s in pglast.parse_sql(sql):
        if type(s.stmt).__name__ == "CreateStmt":
            out[s.stmt.relation.relname] = {e.colname for e in s.stmt.tableElts if hasattr(e, "colname")}
    return out


def test_migrations_exist_in_order():
    assert [f.name for f in FILES][:4] == ["0001_init.sql", "0002_decisions.sql",
                                          "0003_validation_reports.sql", "0004_review.sql"]


@pytest.mark.parametrize("name", list(SQL))
def test_each_migration_parses(name):
    assert len(pglast.parse_sql(SQL[name])) >= 2  # parses, and is not an empty file (a table plus its security block)


def test_plpgsql_functions_parse():
    # pglast's PL/pgSQL parser fails on any trigger function, even `BEGIN RETURN NEW; END`
    # (a parser limitation, not a migration fault), so only non-trigger bodies are checked here.
    checked = total = 0
    for sql in SQL.values():
        for stmt in pglast.parse_sql(sql):
            if type(stmt.stmt).__name__ != "CreateFunctionStmt":
                continue
            total += 1
            end = stmt.stmt_location + stmt.stmt_len if stmt.stmt_len else len(sql)
            text = sql[stmt.stmt_location:end]
            if "RETURNS trigger" in text:
                continue
            pglast.parse_plpgsql(text)
            checked += 1
    assert total == 5 and checked == 3


def test_every_table_has_tenant_id_and_chain_columns():
    tables = tables_created(ALL)
    assert len(tables) == 22
    for name, cols in tables.items():
        assert {"tenant_id", "prev_hash", "row_hash", "payload_hash", "created_by"} <= cols, name


def test_every_table_is_covered_by_rls_chain_and_append_only_setup():
    """Each table must be named in the DO block that enables RLS, triggers and grants."""
    for name, sql in SQL.items():
        created = set(tables_created(sql))
        do_block = sql[sql.rindex("DO $$"):]
        assert "ENABLE ROW LEVEL SECURITY" in do_block and "assay_chain_row" in do_block, name
        covered = {t for t in created if f"'{t}'" in do_block}
        assert created == covered, (name, created - covered)


def test_chain_verifier_accepts_any_chained_table_but_only_chained_tables():
    """0003 replaced the hard-coded table list with a check for a row_hash column, so new chained
    tables need no edit to the verifier. The identifier is still quoted and existence-checked."""
    last = SQL["0003_validation_reports.sql"]
    fn = last[last.index("CREATE OR REPLACE FUNCTION assay_verify_chain"):]
    assert "pg_attribute" in fn and "'row_hash'" in fn and "%I" in fn and "RAISE EXCEPTION" in fn


def test_security_properties_declared():
    for token in ("ENABLE ROW LEVEL SECURITY", "FORCE ROW LEVEL SECURITY",
                  "current_setting('app.tenant_id', true)", "assay_block_mutation",
                  "BEFORE TRUNCATE", "NOBYPASSRLS", "GRANT SELECT, INSERT"):
        assert token in ALL
    assert "GRANT UPDATE" not in ALL and "GRANT DELETE" not in ALL and "GRANT ALL" not in ALL


def test_a_decision_table_cannot_express_an_automated_action_or_a_scoreless_state():
    sql = SQL["0002_decisions.sql"]
    assert "automation_level = 0" in sql  # MVP recommend-only, enforced by the database (FR-23)
    assert "(state = 'insufficient_evidence') = (ti IS NULL)" in sql  # no manufactured scores
