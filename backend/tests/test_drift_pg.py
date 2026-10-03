"""Drift runs on real PostgreSQL: the table's own rules, and the job end to end. Needs ASSAY_TEST_DATABASE_URL."""

from datetime import datetime, timedelta

import numpy as np
import pytest

from assay.detection import load_bundle, save_bundle
from assay.ingestion import IngestionConfig, IngestionService
from assay.ingestion.pg_repo import PostgresRepository
from assay.scoring import (
    BundleRegistry,
    DriftJobConfig,
    ScoringConfig,
    ScoringService,
    run_drift_job,
)
from assay.synthetic import to_wire
from assay.worker import run_once

psycopg = pytest.importorskip("psycopg")
T = "tenant-synth"
RAW = "tenant_id, created_by, payload_hash, prev_hash, row_hash"  # the chain trigger fills the hashes


def raw_run(conn, *, n_ref=10, n_win=10, mx=0.2, level=0.1, alarm=True):
    return conn.execute(
        f"INSERT INTO drift_runs ({RAW}, bundle_id, n_reference, n_window, window_start, window_end, "
        f"feature_names, drift, max_drift, alarm_level, alarm) VALUES ('{T}', 'job', '', '', '', 'b-1', %s, %s, "
        f"now(), now(), '[]', '[]', %s, %s, %s)", (n_ref, n_win, mx, level, alarm))


@pytest.fixture
def raw(pg_conn):
    pg_conn.autocommit = True
    pg_conn.execute("SET session_replication_role = replica")  # skip the bundle FK; each CHECK is tested alone
    return pg_conn


def test_the_alarm_flag_must_agree_with_the_numbers_it_summarises(raw):
    raw_run(raw, mx=0.2, level=0.1, alarm=True)
    raw_run(raw, mx=0.05, level=0.1, alarm=False)
    with pytest.raises(psycopg.errors.CheckViolation):
        raw_run(raw, mx=0.05, level=0.1, alarm=True)    # an alarm nothing justifies
    with pytest.raises(psycopg.errors.CheckViolation):
        raw_run(raw, mx=0.2, level=0.1, alarm=False)    # a breach that is hidden


def test_drift_must_be_a_real_proportion_over_real_windows(raw):
    for bad in ({"mx": 1.5}, {"mx": -0.1}, {"n_win": 0}, {"n_ref": 0}):
        with pytest.raises(psycopg.errors.CheckViolation):
            raw_run(raw, **bad)


def test_drift_runs_are_append_only(raw):
    raw_run(raw)
    raw.execute("SET session_replication_role = origin")  # replica mode would also switch the trigger off
    with pytest.raises(psycopg.errors.InsufficientPrivilege):
        raw.execute("UPDATE drift_runs SET alarm = false")
    with pytest.raises(psycopg.errors.InsufficientPrivilege):
        raw.execute("DELETE FROM drift_runs")


def test_a_run_must_reference_a_recorded_bundle(pg_conn):
    pg_conn.autocommit = True
    with pytest.raises(psycopg.errors.ForeignKeyViolation):
        raw_run(pg_conn)


# -- the job end to end ------------------------------------------------------------------------------
@pytest.fixture(scope="module")
def artefact(trained, tmp_path_factory):
    d = tmp_path_factory.mktemp("driftpg") / "b"
    save_bundle(d, trained[3].scoring_bundle(), trained[3].manifest, b"k")
    return load_bundle(d, T, b"k")


def test_job_stores_a_run_that_a_fresh_process_then_scores_with(pg_conn, trained, artefact):
    _, txns, _, r = trained
    reg = BundleRegistry()
    reg.register(T, *artefact)
    by_id, ids = {t["txn_id"]: t for t in txns}, r.test_table.txn_ids
    clock = {"t": None}
    repo = PostgresRepository(pg_conn)
    scoring = ScoringService(repo, reg, ScoringConfig(clock=lambda: clock["t"]))
    for i in range(0, 120, 12):
        t = to_wire(by_id[ids[i]])
        clock["t"] = datetime.fromisoformat(t["event_time"]) + timedelta(seconds=2)
        IngestionService(repo, IngestionConfig(clock=lambda: clock["t"])).ingest_transaction(
            T, t, source_id="s", actor="test")
        scoring.score_transaction(T, t["txn_id"], actor="test")

    cfg = DriftJobConfig(window=50, min_rows=5, min_span_hours=0, alarm_level=0.10)
    res = run_drift_job(scoring, T, cfg)
    assert res["status"] == "ok" and res["run"]["n_window"] == 10
    (stored,) = repo.find(T, "drift_runs")
    assert stored["bundle_id"] == artefact[1].bundle_id and stored["alarm"] == res["run"]["alarm"]
    assert repo.verify(T, "drift_runs") is None and repo.verify(T, "audit_log") is None

    other = ScoringService(PostgresRepository(pg_conn), reg)  # a fresh process: never ran the job
    assert other.drift == {}
    assert np.allclose(other.current_drift(T), stored["drift"])
    assert PostgresRepository(pg_conn).find("other-tenant", "drift_runs") == []  # row-level security

    out = run_once(scoring, T, refine_limit=3, drift_cfg=cfg)  # the worker pass on the same store
    assert out["refined"] <= 3 and out["drift"] == "ok"
