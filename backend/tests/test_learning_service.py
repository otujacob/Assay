"""Creating a candidate end to end: accepted feedback in, signed and gated candidate out."""

from datetime import timedelta

import pytest

from assay.detection import load_bundle, save_bundle
from assay.detection.labels import labels_as_of
from assay.features.compute import parse_time
from assay.ingestion import InMemoryRepository
from assay.learning.service import LearningConfig, LearningService
from assay.scoring import BundleRegistry, ScoringConfig, ScoringService

T = "tenant-synth"
KEY = b"k"


@pytest.fixture(scope="module")
def champion(trained, tmp_path_factory):
    d = tmp_path_factory.mktemp("lsbundle") / "b"
    save_bundle(d, trained[3].scoring_bundle(), trained[3].manifest, KEY)
    return load_bundle(d, T, KEY)


@pytest.fixture(scope="module")
def made(trained, champion, tmp_path_factory):
    """One candidate, built once: it takes about a minute with the gates."""
    ds, txns, cfg, _ = trained
    repo = InMemoryRepository()
    reg = BundleRegistry()
    reg.register(T, *champion)
    scoring = ScoringService(repo, reg, ScoringConfig())
    verified = labels_as_of(txns, ds.matured_outcomes(cfg.as_of), cfg.as_of, cfg.horizon_days)
    when = {t["txn_id"]: parse_time(t["event_time"]) for t in txns}
    early = [t for t in verified if when[t] < cfg.train_end][:400]
    hidden = set(early)
    outcomes = [o for o in ds.matured_outcomes(cfg.as_of) if o["txn_id"] not in hidden]
    for tid in early:  # two analysts, both right: a corroborated (level 3) label
        for pid in ("an-a", "an-b"):
            final = "block" if verified[tid] else "approve"
            repo.insert(T, "analyst_actions", {
                "txn_id": tid, "analyst_pid": pid, "role": "analyst", "action": final, "final_decision": final,
                "reason_code": "reviewed", "confidence": 0.9, "evidence_checklist": {"a": True, "b": True},
                "blind_flag": True, "seconds_to_decision": 90.0, "action_time": when[tid] + timedelta(days=1),
                "display_state": {"recommendation_shown": False}}, pid)
    bundle_dir = tmp_path_factory.mktemp("candidates")
    svc = LearningService(repo, scoring, LearningConfig(bundle_dir=bundle_dir, signing_key=KEY))
    out = svc.create_candidate(T, "u:creator", txns=txns, outcomes=outcomes, training=cfg)
    return {"svc": svc, "out": out, "repo": repo, "reg": reg, "dir": bundle_dir, "n_hidden": len(hidden), "cfg": cfg}


def test_accepted_feedback_reaches_the_candidate_and_the_pool_is_recorded(made):
    out = made["out"]
    assert out["pool"]["accepted_by_level"].get("3", out["pool"]["accepted_by_level"].get(3)) == made["n_hidden"]
    assert out["training"]["extra_labels_used"] == made["n_hidden"]
    assert out["training"]["feedback_used"] is True
    assert out["base_bundle_id"] != out["candidate_id"] and out["created_by"] == "u:creator"


def test_the_candidate_is_signed_loadable_recorded_in_lineage_and_registered_but_not_champion(made):
    out, svc = made["out"], made["svc"]
    path = made["dir"] / T / out["candidate_id"]
    _, manifest = load_bundle(path, T, KEY)  # verifies the signature before loading
    assert manifest.bundle_id == out["candidate_id"] and manifest.extra["extra_labels_used"] == made["n_hidden"]
    assert made["repo"].find(T, "model_bundles", {"bundle_id": out["candidate_id"]})
    assert made["reg"].get(T, out["candidate_id"]) is not None
    assert svc.status(T)["champion"] != out["candidate_id"]  # nothing is promoted by creating it


def test_gates_were_run_and_unjudgeable_ones_are_not_silently_passed(made):
    gates = {g["id"]: g for g in made["out"]["gates"]["gates"]}
    assert set(gates) == {"G1", "G2", "G3", "G4", "G5", "G6", "G7", "G8", "G9"}
    assert gates["G1"]["status"] == "pass" and gates["G8"]["status"] == "pass"
    assert gates["G6"]["status"] == "review_required" and gates["G7"]["status"] == "pending"
    assert gates["G9"]["status"] == "pending"
    assert made["out"]["state"] in ("validated", "failed")
    assert (made["out"]["state"] == "failed") == bool(made["out"]["gates"]["failed"])


def test_the_holdout_used_by_the_gates_never_saw_analyst_labels(made, trained):
    _, _, cfg, base = trained
    g2 = next(g for g in made["out"]["gates"]["gates"] if g["id"] == "G2")
    # candidate and champion were scored on the same verified-only test window as the plain model
    assert g2["values"]["candidate"]["pr_auc"] == pytest.approx(made["out"]["training"]["metrics"]["pr_auc_ensemble"])
    assert made["out"]["training"]["test_end"] == cfg.test_end.isoformat()
    assert len(base.test_y) > 0


def test_a_candidate_without_feedback_is_possible_and_says_so(trained, champion, tmp_path):
    ds, txns, cfg, _ = trained
    repo = InMemoryRepository()
    reg = BundleRegistry()
    reg.register(T, *champion)
    svc = LearningService(repo, ScoringService(repo, reg, ScoringConfig()),
                          LearningConfig(bundle_dir=tmp_path, signing_key=KEY))
    out = svc.create_candidate(T, "u:creator", txns=txns, outcomes=ds.outcomes, training=cfg, use_feedback=False)
    assert out["training"]["feedback_used"] is False and out["training"]["extra_labels_used"] == 0


def test_without_a_bundle_directory_the_integrity_gate_fails(trained, champion):
    ds, txns, cfg, _ = trained
    repo = InMemoryRepository()
    reg = BundleRegistry()
    reg.register(T, *champion)
    svc = LearningService(repo, ScoringService(repo, reg, ScoringConfig()), LearningConfig())
    out = svc.create_candidate(T, "u:creator", txns=txns, outcomes=ds.outcomes, training=cfg, use_feedback=False)
    g8 = next(g for g in out["gates"]["gates"] if g["id"] == "G8")
    assert g8["status"] == "fail" and out["state"] == "failed"
