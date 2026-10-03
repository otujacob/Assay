"""The background worker: refinement and drift in one pass, with tenants isolated from each other's failures."""

from contextlib import contextmanager
from datetime import datetime, timedelta

import pytest

from assay.detection import load_bundle, save_bundle
from assay.ingestion import IngestionConfig, IngestionService, InMemoryRepository
from assay.scoring import BundleRegistry, DriftJobConfig, ScoringConfig, ScoringService
from assay.synthetic import START, to_wire
from assay.worker import run_cycle, run_once

T = "tenant-synth"


@pytest.fixture(scope="module")
def artefact(trained, tmp_path_factory):
    d = tmp_path_factory.mktemp("worker") / "b"
    save_bundle(d, trained[3].scoring_bundle(), trained[3].manifest, b"k")
    return load_bundle(d, T, b"k")


@pytest.fixture
def world(trained, artefact):
    _, txns, _, r = trained
    repo, clock = InMemoryRepository(), {"t": START}
    reg = BundleRegistry()
    reg.register(T, *artefact)
    scoring = ScoringService(repo, reg, ScoringConfig(clock=lambda: clock["t"], audit_explain_pct=0))
    by_id, ids = {t["txn_id"]: t for t in txns}, r.test_table.txn_ids

    def score_many(n):
        for i in range(0, n * 12, 12):
            t = to_wire(by_id[ids[i]])
            clock["t"] = datetime.fromisoformat(t["event_time"]) + timedelta(seconds=2)
            IngestionService(repo, IngestionConfig(clock=lambda: clock["t"])).ingest_transaction(
                T, t, source_id="s", actor="test")
            scoring.score_transaction(T, t["txn_id"], actor="test")

    return {"scoring": scoring, "repo": repo, "score": score_many}


def pending(world):
    latest = {pd["txn_id"]: pd for pd in world["repo"].find(T, "policy_decisions")}
    return sum(1 for pd in latest.values()
               if not world["repo"].get_by_id(T, "trust_assessments", pd["trust_assessment_id"])["evidence"]["explain"])


def test_one_pass_refines_pending_cases_and_reports_a_skipped_drift_run(world):
    world["score"](12)
    before = pending(world)
    assert before > 0  # cases scored without explanation testing, capped below High until refined
    res = run_once(world["scoring"], T, refine_limit=5, drift_cfg=DriftJobConfig(min_rows=500))
    assert res["refined"] == 5 and pending(world) == before - 5
    assert res["drift"] == "skipped" and res["drift_reason"] == "thin_window" and res["alarm"] is False


def test_repeating_a_pass_does_not_redo_finished_work(world):
    world["score"](6)
    cfg = DriftJobConfig(min_rows=500)
    first = run_once(world["scoring"], T, drift_cfg=cfg)
    again = run_once(world["scoring"], T, drift_cfg=cfg)
    assert first["refined"] > 0 and again["refined"] == 0 and pending(world) == 0


def test_a_failing_tenant_does_not_stop_the_others(world):
    world["score"](3)

    @contextmanager
    def open_scoring():
        yield world["scoring"]

    out = run_cycle(["no-such-tenant", T], open_scoring, drift_cfg=DriftJobConfig(min_rows=500))
    assert out["no-such-tenant"] == {"error": "ScoringError"}  # no champion bundle for it
    assert out[T]["refined"] >= 0 and "error" not in out[T]


def test_a_failure_opening_a_connection_is_contained_too():
    @contextmanager
    def broken():
        raise ConnectionError("db down")
        yield  # pragma: no cover

    assert run_cycle(["a", "b"], broken) == {"a": {"error": "ConnectionError"}, "b": {"error": "ConnectionError"}}
