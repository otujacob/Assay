import itertools
import os
from pathlib import Path

import pytest

MIGRATIONS = sorted((Path(__file__).parents[2] / "db" / "migrations").glob("*.sql"))
DB_URL = os.environ.get("ASSAY_TEST_DATABASE_URL")
_schema_ids = itertools.count()


@pytest.fixture(scope="session")
def trained():
    """(dataset, txns, training config, TrainingResult), trained once for the whole session."""
    from assay.detection import train_bundle
    from conftest_detection import make_dataset

    ds, txns, cfg = make_dataset()
    return ds, txns, cfg, train_bundle(txns, ds.outcomes, cfg)


@pytest.fixture
def pg_conn():
    """A connection whose search_path is a fresh schema holding the migrated tables.

    Needs ASSAY_TEST_DATABASE_URL: a disposable database, superuser (append-only tables
    cannot be cleaned up, so each test gets its own schema).
    """
    psycopg = pytest.importorskip("psycopg")
    if not DB_URL:
        pytest.skip("ASSAY_TEST_DATABASE_URL not set")
    schema = f"t_{os.getpid()}_{next(_schema_ids)}"
    conn = psycopg.connect(DB_URL, autocommit=True)
    conn.execute(f'CREATE SCHEMA "{schema}"')
    conn.execute(f'SET search_path TO "{schema}"')
    for m in MIGRATIONS:
        conn.execute(m.read_text(encoding="utf-8"))
    conn.execute(f'GRANT USAGE ON SCHEMA "{schema}" TO assay_app')
    conn.autocommit = False
    conn.schema = schema
    yield conn
    conn.rollback()
    conn.close()
