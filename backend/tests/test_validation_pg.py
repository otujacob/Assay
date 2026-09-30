"""FR-39 on Postgres: a validation report is stored append-only, linked to its bundle. Needs a database."""

import pytest

from assay.ingestion.pg_repo import PostgresRepository
from assay.validation.report import store_report

psycopg = pytest.importorskip("psycopg")

REPORT = {
    "meta": {"dataset_id": "ds-1", "mode": "provisional", "n_cases": 10, "n_wrong": 1},
    "measures": {"auroc": {"value": 0.9, "lo": 0.8, "hi": 1.0}},
    "baselines": {"B1_distance_from_threshold": {}},
    "ablations": {"conf": {"auroc": 0.9}},
    "pass_criteria": {"atce_supported_as_specified": False},
    "limitations": ["synthetic"],
}


def test_report_stored_with_lineage_and_chain_verifies(pg_conn, trained):
    repo = PostgresRepository(pg_conn)
    m = trained[3].manifest
    row = store_report(repo, "tenant-synth", m, REPORT, stress={"x": {"passed": None}})
    assert row["bundle_id"] == m.bundle_id and row["dataset_id"] == "ds-1"
    got = repo.find("tenant-synth", "validation_reports", {"bundle_id": m.bundle_id})
    assert len(got) == 1 and got[0]["pass_criteria"] == {"atce_supported_as_specified": False}
    assert got[0]["measures"]["auroc"]["value"] == 0.9
    store_report(repo, "tenant-synth", m, REPORT)  # a second report appends; the bundle row is reused
    assert len(repo.rows("tenant-synth", "validation_reports")) == 2
    assert len(repo.find("tenant-synth", "model_bundles")) == 1
    for table in ("validation_reports", "model_bundles"):
        assert repo.verify("tenant-synth", table) is None


def test_generic_chain_verifier_rejects_non_chained_relations(pg_conn):
    repo = PostgresRepository(pg_conn)
    with pytest.raises(psycopg.errors.RaiseException, match="unknown chained table"), repo.atomic("t"):
        pg_conn.execute("SELECT assay_verify_chain('pg_class', 't')")
    with pytest.raises(psycopg.errors.RaiseException, match="unknown chained table"), repo.atomic("t"):
        pg_conn.execute("SELECT assay_verify_chain('users; drop table transactions', 't')")
