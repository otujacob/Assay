"""FR-36: the measures must match known values on data with known truth."""

import numpy as np
import pytest

from assay.validation.measures import (
    auroc,
    auroc_with_ci,
    bootstrap_ci,
    coverage_curve,
    expected_calibration_error,
    paired_auroc_diff_ci,
    pr_auc,
    rate,
    spearman,
)


def test_auroc_known_values():
    wrong = np.array([0, 0, 1, 1])
    assert auroc(wrong, np.array([0.1, 0.2, 0.8, 0.9])) == 1.0  # perfect
    assert auroc(wrong, np.array([0.9, 0.8, 0.2, 0.1])) == 0.0  # perfectly backwards
    assert auroc(wrong, np.array([0.5, 0.5, 0.5, 0.5])) == 0.5  # uninformative
    assert np.isnan(auroc(np.zeros(4), np.arange(4)))  # undefined with one class
    assert pr_auc(np.array([0, 0, 1, 1]), np.array([0.1, 0.2, 0.8, 0.9])) == 1.0
    assert np.isnan(pr_auc(np.zeros(4), np.arange(4)))


def test_rate_with_wilson_interval_and_hand_values():
    hi_trust = np.array([1, 1, 1, 1, 0, 0], bool)
    wrong = np.array([1, 0, 0, 0, 1, 1], bool)
    r = rate(wrong, hi_trust)  # P(wrong | high trust) = 1/4
    assert (r["k"], r["n"], r["value"]) == (1, 4, 0.25) and r["lo"] < 0.25 < r["hi"]
    assert rate(wrong, np.zeros(6, bool))["value"] is None  # no cases: no estimate


def test_false_confidence_and_high_trust_error_are_different_quantities():
    high = np.array([1, 1, 1, 1, 0, 0, 0, 0], bool)
    wrong = np.array([1, 0, 0, 0, 1, 1, 1, 1], bool)
    assert rate(wrong, high)["value"] == 0.25  # P(wrong | High)   = 1/4
    assert rate(high, wrong)["value"] == 0.2  # P(High | wrong)   = 1/5


def test_ece_known_values():
    correct = np.array([0, 0, 1, 1])
    assert expected_calibration_error(correct, np.array([10, 20, 80, 90]), bins=2) == pytest.approx(0.15)
    assert expected_calibration_error(np.array([1, 1, 1, 1]), np.array([100] * 4)) == 0.0
    assert expected_calibration_error(np.array([0, 0, 0, 0]), np.array([100] * 4)) == 1.0


def test_spearman_with_ties():
    assert spearman(np.array([1, 2, 3, 4]), np.array([10, 20, 30, 40])) == pytest.approx(1.0)
    assert spearman(np.array([1, 2, 3, 4]), np.array([4, 3, 2, 1])) == pytest.approx(-1.0)
    assert np.isnan(spearman(np.ones(4), np.arange(4)))
    assert spearman(np.array([1, 1, 2, 2]), np.array([1, 1, 2, 2])) == pytest.approx(1.0)


def test_bootstrap_interval_brackets_the_truth_on_known_data():
    """Planted truth: a score with a known AUROC (~0.85). The 95% interval should contain it."""
    rng = np.random.default_rng(3)
    wrong = rng.random(3000) < 0.3
    score = np.where(wrong, rng.normal(1.5, 1, 3000), rng.normal(0, 1, 3000))  # AUROC = Phi(1.5/sqrt2)
    true_auc = 0.856
    r = auroc_with_ci(wrong, score, n_boot=200)
    assert r["lo"] < true_auc < r["hi"] and r["hi"] - r["lo"] < 0.06


def test_paired_difference_detects_a_real_gap_and_not_a_fake_one():
    rng = np.random.default_rng(4)
    wrong = rng.random(2500) < 0.3
    good = np.where(wrong, rng.normal(1.5, 1, 2500), rng.normal(0, 1, 2500))
    weak = np.where(wrong, rng.normal(0.4, 1, 2500), rng.normal(0, 1, 2500))
    clone = good + rng.normal(0, 0.01, 2500)
    better = paired_auroc_diff_ci(wrong, good, weak, n_boot=200)
    assert better["excludes_zero_above"] and better["diff"] > 0.15
    same = paired_auroc_diff_ci(wrong, good, clone, n_boot=200)
    assert not same["excludes_zero_above"] and abs(same["diff"]) < 0.01


def test_bootstrap_returns_nan_when_a_class_is_missing():
    lo, hi = bootstrap_ci(auroc, np.zeros(50, int), np.arange(50.0), n_boot=40)
    assert np.isnan(lo) and np.isnan(hi)


def test_coverage_curve_abstains_lowest_trust_and_unscored_first():
    wrong = np.array([1, 1, 0, 0, 0, 0, 0, 0, 0, 0], bool)
    ti = np.array([np.nan, 10, 90, 80, 70, 60, 95, 85, 75, 65])
    curve = coverage_curve(wrong, ti, points=5)
    assert curve[-1] == {"coverage": 1.0, "error_rate": 0.2, "errors_kept": 2}
    assert curve[0]["error_rate"] == 0.0  # the most trusted 20% contain no errors
    assert [c["coverage"] for c in curve] == pytest.approx([0.2, 0.4, 0.6, 0.8, 1.0])
    assert curve[3]["errors_kept"] == 0  # at 80% both the unscored and the TI=10 error are abstained
    fine = coverage_curve(wrong, ti, points=10)
    assert fine[8]["errors_kept"] == 1  # at 90% only the unscored case is abstained; TI=10 remains
