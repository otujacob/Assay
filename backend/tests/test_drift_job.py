"""The population drift job (FR-17): drift measured on stored feature vectors, persisted, shared.

The feature store is seeded from the trained dataset's own feature rows, so a "clean" window has the
reference's distribution. Ingesting a few dozen transactions through the service would not: customers
would have almost no history, and history-based features would differ from the full-history reference
for reasons that have nothing to do with drift.
"""

from datetime import datetime, timedelta

import numpy as np
import pytest

from assay.detection import load_bundle, save_bundle
from assay.ingestion import IngestionConfig, IngestionService, InMemoryRepository
from assay.review.service import ReviewConfig, ReviewService
from assay.scoring import (
    BundleRegistry,
    DriftJobConfig,
    ScoringConfig,
    ScoringService,
    run_drift_job,
)
from assay.synthetic import START, to_wire

T = "tenant-synth"
CFG = DriftJobConfig(window=60, min_rows=40, alarm_level=0.10)  # min_span_hours stays at its 24 h default


@pytest.fixture(scope="module")
def artefact(trained, tmp_path_factory):
    d = tmp_path_factory.mktemp("driftjob") / "b"
    save_bundle(d, trained[3].scoring_bundle(), trained[3].manifest, b"k")
    return load_bundle(d, T, b"k")


@pytest.fixture
def env(trained, artefact):
    _, txns, _, r = trained
    repo, clock = InMemoryRepository(), {"t": START}
    reg = BundleRegistry()
    reg.register(T, *artefact)
    scoring = ScoringService(repo, reg, ScoringConfig(clock=lambda: clock["t"]))
    m = artefact[1]
    names = list(m.feature_names)
    X = np.asarray(r.test_table.X, float)
    by_id, ids = {t["txn_id"]: t for t in txns}, r.test_table.txn_ids

    def seed(n=60, *, shift_amount=1.0, hours_apart=1.0, offset=0):
        """Store n feature vectors drawn evenly from the test period, optionally with the amount
        features scaled, `hours_apart` hours apart."""
        rows = X[offset::max(1, (len(X) - offset) // n)][:n].copy()
        for j, name in enumerate(names):
            if "amount" in name:
                rows[:, j] *= shift_amount
        for k, row in enumerate(rows):
            repo.insert(T, "feature_vectors", {
                "schema_version": "x", "txn_id": f"seed-{offset}-{k}", "feature_set_version": m.feature_set_version,
                "definition_versions": {}, "as_of_time": START + timedelta(hours=hours_apart * k),
                "feature_names": names, "feature_values": [float(v) for v in row], "graph_snapshot_id": None}, "test")

    def score_real(i):
        t = to_wire(by_id[ids[i]])
        clock["t"] = datetime.fromisoformat(t["event_time"]) + timedelta(seconds=2)
        IngestionService(repo, IngestionConfig(clock=lambda: clock["t"])).ingest_transaction(
            T, t, source_id="s", actor="test")
        return scoring.score_transaction(T, t["txn_id"], actor="test")

    return {"seed": seed, "score": score_real, "scoring": scoring, "repo": repo, "reg": reg,
            "clock": clock, "names": names}


def drift_audit(env):
    return [r["action"] for r in env["repo"].rows(T, "audit_log") if r["action"].startswith("drift")]


def test_a_thin_window_is_skipped_and_stores_nothing(env):
    env["seed"](10)
    res = run_drift_job(env["scoring"], T, CFG)
    assert res == {"status": "skipped", "reason": "thin_window", "n_window": 10, "min_rows": 40}
    assert env["repo"].find(T, "drift_runs") == [] and drift_audit(env) == []
    assert not env["scoring"].current_drift(T).any()  # no information is treated as no drift


def test_a_window_covering_only_a_few_hours_is_skipped_not_alarmed(env):
    """60 consecutive transactions differ from the reference in time-of-day alone. Without the span
    rule, such a window raised an alarm on `hour` with nothing wrong (drift 0.30, measured)."""
    env["seed"](60, hours_apart=1 / 60)  # an hour of traffic
    res = run_drift_job(env["scoring"], T, CFG)
    assert res["status"] == "skipped" and res["reason"] == "short_span"
    assert res["span_hours"] < CFG.min_span_hours and env["repo"].find(T, "drift_runs") == []


def test_a_stable_window_stores_a_run_without_an_alarm(env):
    env["seed"](60)
    res = run_drift_job(env["scoring"], T, CFG)
    assert res["status"] == "ok" and res["run"]["alarm"] is False and res["run"]["n_window"] == 60
    assert res["run"]["max_drift"] < CFG.alarm_level
    assert drift_audit(env) == ["drift_run"]


def test_a_shifted_window_raises_the_alarm_and_names_the_feature(env):
    env["seed"](60, shift_amount=4.0)
    res = run_drift_job(env["scoring"], T, CFG)
    assert res["run"]["alarm"] is True and res["run"]["max_drift"] >= CFG.alarm_level
    assert drift_audit(env) == ["drift_run", "drift_alarm"]
    alarm = next(r for r in env["repo"].rows(T, "audit_log") if r["action"] == "drift_alarm")
    worst = env["names"][int(np.argmax(res["run"]["drift"]))]
    assert alarm["actor"] == "drift-job" and worst in alarm["result"] and "amount" in worst


def test_drift_on_a_feature_that_is_not_shifted_stays_near_zero(env):
    env["seed"](60, shift_amount=4.0)
    d = run_drift_job(env["scoring"], T, CFG)["run"]["drift"]
    untouched = [v for n, v in zip(env["names"], d, strict=True) if "amount" not in n and "velocity" not in n]
    assert max(untouched) < CFG.alarm_level  # the alarm is about the shifted features, not everything


def test_the_most_recent_window_is_the_one_compared(env):
    env["seed"](60, shift_amount=4.0, offset=0)       # old, shifted
    env["seed"](60, offset=1)                         # newer, clean: the job looks at the newest 60
    assert run_drift_job(env["scoring"], T, CFG)["run"]["alarm"] is False


def test_scoring_uses_the_stored_run_even_in_a_fresh_process(env):
    env["seed"](60, shift_amount=4.0)
    run_drift_job(env["scoring"], T, CFG)
    # A new ScoringService on the same store, as after a restart or in another worker: it never ran the job.
    other = ScoringService(env["repo"], env["reg"], ScoringConfig(clock=lambda: env["clock"]["t"]))
    assert other.drift == {} and other.current_drift(T).max() >= CFG.alarm_level
    d = env["score"](3)
    ta = env["repo"].find(T, "trust_assessments", {"prediction_id": env["repo"].find(
        T, "predictions", {"txn_id": d["txn_id"]})[0]["id"]})[0]
    assert max(ta["evidence"]["drift_vector"]) >= CFG.alarm_level  # the decision carries the drift it was scored under


def test_a_run_for_another_bundle_is_not_applied(env, artefact):
    env["seed"](60, shift_amount=4.0)
    run_drift_job(env["scoring"], T, CFG)
    reg2 = BundleRegistry()
    art, manifest = artefact
    reg2.register(T, art, type(manifest)(**{**manifest.__dict__, "bundle_id": "b-other"}))
    assert not ScoringService(env["repo"], reg2).current_drift(T).any()  # measured against a different model


def test_the_dashboard_reports_the_alarm(env):
    review = ReviewService(env["repo"], env["scoring"], ReviewConfig(clock=lambda: env["clock"]["t"]))
    env["seed"](60, shift_amount=4.0)
    assert review.dashboard(T)["model_drift"]["status"] == "stable"  # no run yet
    run_drift_job(env["scoring"], T, CFG)
    assert review.dashboard(T)["model_drift"]["status"] == "alarm"


def test_the_dashboard_survives_a_tenant_with_no_bundle(env):
    review = ReviewService(env["repo"], ScoringService(env["repo"], BundleRegistry()))
    assert review.dashboard("no-such-tenant")["model_drift"]["max_feature_drift"] == 0.0
