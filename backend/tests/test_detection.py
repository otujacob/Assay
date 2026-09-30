import json
import shutil

import numpy as np
import pytest

from assay.detection import (
    BundleError,
    expected_calibration_error,
    load_bundle,
    save_bundle,
    train_bundle,
)
from assay.detection.bundle import BundleManifest, assert_single_tenant
from assay.detection.metrics import cost_optimal_threshold
from conftest_detection import GEN, make_dataset

KEY = b"test-signing-key"


def test_training_uses_matured_labels_and_a_temporal_split(trained):
    _, _, cfg, r = trained
    m = r.manifest
    assert m.dataset_tenant_ids == [cfg.tenant_id] and m.feature_set_version == "fs-2"
    cb = m.extra["class_balance"]
    assert cb["train_pos"] > 20 and cb["cal_pos"] > 5
    # test rows are all at or after the calibration end: no overlap with training time
    assert r.test_table.t.min() >= (cfg.calibration_end.timestamp())


def test_ensemble_beats_chance_by_a_wide_margin_and_baseline_is_reported(trained):
    m = trained[3].manifest.metrics
    assert m["pr_auc_ensemble"] > 5 * m["test_base_rate"]
    assert "pr_auc_logistic_baseline" in m  # mandatory benchmark recorded, win or lose


def test_calibrated_score_is_monotone_in_raw_score(trained):  # FR-09
    p = trained[3].test_pred
    order = np.argsort(p.raw, kind="stable")
    assert np.all(np.diff(p.calibrated[order]) >= -1e-12)
    assert p.calibrated.min() >= 0 and p.calibrated.max() <= 1


def test_calibration_report_is_stored_and_reasonable(trained):
    cal = trained[3].manifest.calibration
    assert 0 <= cal["ece_test"] < 0.05 and len(cal["reliability_test"]) >= 5


def test_prediction_carries_uncertainty(trained):  # FR-10
    p = trained[3].test_pred
    assert p.member_spread.shape == p.calibrated.shape and (p.member_spread >= 0).all()
    t_high = trained[3].manifest.thresholds["t_high"]
    assert np.allclose(p.distance_to_threshold, np.abs(p.calibrated - t_high))


def test_scoring_is_deterministic(trained):
    model, r = trained[3].model, trained[3]
    a, b = model.predict(r.test_table.X), model.predict(r.test_table.X)
    assert np.array_equal(a.calibrated, b.calibrated) and np.array_equal(a.member_spread, b.member_spread)


def test_novel_fraud_type_is_the_blind_spot(trained):
    """Not a pass/fail claim about quality: records why the Trust Index needs Familiarity."""
    ds, _, _, r = trained
    truth = ds.truth
    known, novel = [], []
    for i, tid in enumerate(r.test_table.txn_ids):
        ft = truth[tid]["fraud_type"]
        if ft == "novel_open_banking_scam":
            novel.append(r.test_pred.calibrated[i])
        elif ft:
            known.append(r.test_pred.calibrated[i])
    assert novel and known
    t_high = trained[3].manifest.thresholds["t_high"]
    print(f"\nflagged at t_high: known fraud {np.mean(np.array(known) >= t_high):.2f}, "
          f"novel fraud {np.mean(np.array(novel) >= t_high):.2f} "
          f"(mean score {np.mean(known):.2f} vs {np.mean(novel):.2f})")


def test_ece_and_threshold_helpers():
    y = np.array([0, 0, 1, 1]); p = np.array([0.1, 0.2, 0.8, 0.9])
    assert expected_calibration_error(y, p, bins=2) == pytest.approx(0.15)
    # costly misses push the threshold down; costly false alarms push it up
    y2 = np.array([0, 0, 0, 1, 0, 1]); p2 = np.array([.1, .2, .3, .4, .5, .9])
    assert cost_optimal_threshold(y2, p2, 1, 50) <= cost_optimal_threshold(y2, p2, 50, 1)


def test_bundle_roundtrip_and_tenant_binding(trained, tmp_path):
    _, _, _, r = trained
    save_bundle(tmp_path / "b", r.model, r.manifest, KEY)
    model, m = load_bundle(tmp_path / "b", "tenant-synth", KEY)
    x = r.test_table.X[:50]
    assert np.array_equal(model.predict(x).calibrated, r.model.predict(x).calibrated)
    assert m.artifact_sha256 and m.signature
    with pytest.raises(BundleError, match="different tenant"):
        load_bundle(tmp_path / "b", "tenant-other", KEY)


def test_tampered_or_wrongly_signed_bundle_is_refused_before_loading(trained, tmp_path):
    _, _, _, r = trained
    d = save_bundle(tmp_path / "b", r.model, r.manifest, KEY)
    with pytest.raises(BundleError, match="bad signature"):
        load_bundle(d, "tenant-synth", b"wrong-key")
    (d / "model.joblib").write_bytes((d / "model.joblib").read_bytes() + b"x")
    with pytest.raises(BundleError, match="hash"):
        load_bundle(d, "tenant-synth", KEY)
    # manifest edited to claim another tenant's data
    d2 = save_bundle(tmp_path / "c", r.model, r.manifest, KEY)
    mf = json.loads((d2 / "manifest.json").read_text())
    mf["tenant_id"] = "tenant-other"
    (d2 / "manifest.json").write_text(json.dumps(mf))
    with pytest.raises(BundleError, match="bad signature"):
        load_bundle(d2, "tenant-other", KEY)
    shutil.rmtree(tmp_path)


def test_multi_tenant_dataset_manifest_is_rejected(trained):
    m = BundleManifest(**{**trained[3].manifest.__dict__, "dataset_tenant_ids": ["a", "b"]})
    with pytest.raises(BundleError):
        assert_single_tenant(m)


def test_too_little_data_fails_loudly():
    ds, txns, cfg = make_dataset()
    few = [t for t in txns if t["event_time"] < "2025-02-15"]
    with pytest.raises(ValueError, match="too few"):
        train_bundle(few, ds.outcomes, cfg)


def test_uses_generator_config_constants():
    assert GEN.dispute_window_days == 30
