"""Automatic rollback of a live model that degrades (PRD 12.4, H7)."""

from dataclasses import replace

import numpy as np
import pytest

from assay.detection import load_bundle, save_bundle
from assay.ingestion import InMemoryRepository
from assay.learning.lifecycle import LifecycleConfig, LifecycleStore
from assay.learning.monitor import MonitorConfig, evaluate
from assay.learning.service import LearningConfig, LearningService
from assay.scoring import BundleRegistry, ScoringConfig, ScoringService
from assay.worker import run_once

T = "tenant-synth"
CFG = MonitorConfig()


def world(n, *, flag_rate=0.1, fraud_rate=0.1, skill=1.0, seed=0, high_wrong=0.0):
    """Matured live decisions with a known quality: `skill` is how often a fraud is flagged and a legit is not."""
    rng = np.random.default_rng(seed)
    fraud = (rng.random(n) < fraud_rate).astype(int)
    right = rng.random(n) < skill
    flagged = np.where(right, fraud == 1, fraud == 0)
    risk = np.where(flagged, 0.6, 0.02) + rng.normal(0, 0.005, n)
    risk = np.clip(risk, 0, 1)
    high = (~flagged) & (rng.random(n) < 0.5)
    if high_wrong:
        flip = high & (rng.random(n) < high_wrong)
        fraud = np.where(flip, 1, fraud)
    return risk, fraud, high


BASE = {"precision": 0.8, "recall": 0.7}


def test_too_few_matured_cases_gives_no_verdict():
    r = evaluate("b", *world(50), 0.3, BASE, CFG)
    assert r.status == "insufficient" and r.triggers == [] and "needs 150" in r.measures["note"]


def test_a_healthy_model_is_left_alone():
    risk, fraud, high = world(2000, skill=0.9)
    risk = np.where(fraud == 1, 0.8, 0.02) * 1.0
    r = evaluate("b", risk, fraud, high, 0.3, {"precision": 0.9, "recall": 0.9}, CFG)
    assert r.status == "ok" and r.triggers == []


def test_a_collapse_in_recall_and_precision_is_caught():
    risk, fraud, high = world(3000, skill=0.45, fraud_rate=0.15)
    r = evaluate("b", risk, fraud, high, 0.3, {"precision": 0.9, "recall": 0.9}, CFG)
    assert r.status == "breach"
    assert any("recall" in t for t in r.triggers) and any("precision" in t for t in r.triggers)


def test_miscalibration_is_caught():
    n = 1000
    fraud = (np.random.default_rng(1).random(n) < 0.05).astype(int)
    r = evaluate("b", np.full(n, 0.4), fraud, np.zeros(n, bool), 0.3, None, CFG)  # says 40%, is about 5%
    assert r.status == "breach" and any("calibration" in t for t in r.triggers)


def test_high_trust_cases_that_are_often_wrong_are_caught_only_with_enough_of_them():
    n = 2000
    risk = np.full(n, 0.02)
    fraud = np.zeros(n, int)
    high = np.zeros(n, bool)
    high[:200] = True
    fraud[:40] = 1  # 20% of the High-trust cases were fraud the model did not flag
    r = evaluate("b", risk, fraud, high, 0.3, None, CFG)
    assert any("high-trust" in t for t in r.triggers)
    high2 = np.zeros(n, bool)
    high2[:30] = True
    assert not any("high-trust" in t for t in evaluate("b", risk, fraud, high2, 0.3, None, CFG).triggers)


def test_a_small_dip_inside_the_tolerance_does_not_trigger():
    risk, fraud, high = world(3000, skill=0.9, fraud_rate=0.15)
    risk = np.where(fraud == 1, np.where(np.random.default_rng(2).random(3000) < 0.9, 0.8, 0.02), 0.02)
    r = evaluate("b", risk, fraud, high, 0.3, {"precision": 0.95, "recall": 0.95}, CFG)
    assert not any("recall" in t or "precision" in t for t in r.triggers)  # 0.90 against 0.95 is within 0.10


# ---- through the service, on stored decisions -----------------------------------------------------------------
@pytest.fixture
def live(trained, tmp_path):
    d = tmp_path / "b"
    save_bundle(d, trained[3].scoring_bundle(), trained[3].manifest, b"k")
    obj, manifest = load_bundle(d, T, b"k")
    repo = InMemoryRepository()
    reg = BundleRegistry()
    reg.register(T, obj, manifest)
    cand = replace(manifest, bundle_id="cand", metrics={**manifest.metrics, "at_t_high": {"precision": 0.9, "recall": 0.9}})
    reg.register(T, obj, cand, champion=False)
    scoring = ScoringService(repo, reg, ScoringConfig())
    lc = LifecycleConfig(min_canary_cases=1)
    svc = LearningService(repo, scoring, LearningConfig(lifecycle=lc, monitor=MonitorConfig(min_matured=100)))
    gates = {"failed": [], "unresolved": [], "gates": [{"id": "G1", "status": "pass"}]}
    store = LifecycleStore(repo, lc)
    store.create(T, "u:creator", candidate_id="cand", base_bundle_id=manifest.bundle_id, artefact_path=None,
                 pool_summary={}, training={}, gates=gates)
    return svc, store, repo, manifest


def into_canary(store, repo):
    store.start_shadow(T, "u:creator", "cand")
    store.cfg = replace(store.cfg, min_shadow_days=0.0, min_shadow_cases=0)
    store.approve(T, "u:approver", "cand", waived=[], reviewed_segments=True, rationale="ok")
    store.start_canary(T, "u:approver", "cand", 0.5)


def decide(repo, n, *, good):
    """n matured decisions by 'cand': good ones are flagged exactly when fraud; bad ones are the reverse."""
    rng = np.random.default_rng(3)
    for i in range(n):
        is_fraud = rng.random() < 0.2
        flagged = is_fraud if good else not is_fraud
        txn = f"t{i}"
        p = repo.insert(T, "predictions", {"txn_id": txn, "bundle_id": "cand", "calibrated_risk": 0.8 if flagged else 0.02}, "sys")
        repo.insert(T, "trust_assessments", {"prediction_id": p["id"], "version_no": 1, "state": "moderate"}, "sys")
        repo.insert(T, "outcomes", {"txn_id": txn, "outcome_type": "confirmed_fraud" if is_fraud else "confirmed_legitimate",
                                    "maturity_state": "matured"}, "sys")


def test_a_canary_that_degrades_is_rolled_back_by_the_monitor_with_the_evidence_stored(live):
    svc, store, repo, _ = live
    into_canary(store, repo)
    decide(repo, 400, good=False)
    seen = svc.check_health(T)  # reading changes nothing
    assert seen[0]["status"] == "breach" and not seen[0]["rolled_back"] and store.state_of(T, "cand") == "canary"
    out = svc.check_health(T, act=True)
    assert out[0]["rolled_back"] is True and store.state_of(T, "cand") == "rolled_back"
    ev = store.events(T, "cand")[-1]
    assert ev["kind"] == "rolled_back" and ev["created_by"] == "system:monitor" and ev["detail"]["reason"].startswith("automatic: ")
    svc.scoring.sync_routing(T)
    assert svc.scoring.registry.deciding_id(T) != "cand"
    assert svc.check_health(T, act=True) == []  # nothing live any more


def test_a_healthy_canary_is_not_touched_and_one_with_thin_evidence_is_not_judged(live):
    svc, store, repo, _ = live
    into_canary(store, repo)
    decide(repo, 40, good=True)
    assert svc.check_health(T, act=True)[0]["status"] == "insufficient"
    decide_more = 300
    for i in range(1000, 1000 + decide_more):
        txn = f"t{i}"
        p = repo.insert(T, "predictions", {"txn_id": txn, "bundle_id": "cand", "calibrated_risk": 0.02}, "sys")
        repo.insert(T, "trust_assessments", {"prediction_id": p["id"], "version_no": 1, "state": "moderate"}, "sys")
        repo.insert(T, "outcomes", {"txn_id": txn, "outcome_type": "confirmed_legitimate", "maturity_state": "matured"}, "sys")
    out = svc.check_health(T, act=True)
    assert out[0]["status"] != "breach" and store.state_of(T, "cand") == "canary"


def test_only_models_deciding_live_cases_are_monitored(live):
    svc, store, repo, _ = live
    assert svc.check_health(T, act=True) == []            # validated
    store.start_shadow(T, "u:creator", "cand")
    decide(repo, 400, good=False)
    assert svc.check_health(T, act=True) == []            # shadow decides nothing, so there is nothing to roll back


def test_the_worker_runs_the_check_and_reports_a_rollback(live, trained):
    svc, store, repo, _ = live
    into_canary(store, repo)
    decide(repo, 400, good=False)
    res = run_once(svc.scoring, T)
    assert res["rolled_back"] == ["cand"] and store.state_of(T, "cand") == "rolled_back"


def test_a_high_error_rate_on_thin_evidence_is_not_enough_because_the_whole_interval_must_clear_the_limit():
    n = 1000
    risk, fraud, high = np.full(n, 0.02), np.zeros(n, int), np.zeros(n, bool)
    high[:50] = True
    fraud[:4] = 1  # 8% of 50: above the 5% limit as a point estimate, but the interval still reaches below it
    r = evaluate("b", risk, fraud, high, 0.3, None, CFG)
    assert r.measures["high_trust_error"]["rate"] == 0.08 and r.measures["high_trust_error"]["lo"] < 0.05
    assert not any("high-trust" in t for t in r.triggers)
