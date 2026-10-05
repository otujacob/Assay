"""Recalibrated Trust Index (PRD 5.6): a monotone map from the provisional index to P(correct), checked against
cases whose TRUE probability of a correct recommendation is known."""

import numpy as np
import pytest

from assay.trust import Component, TrustConfig, TrustState, compute_trust_index
from assay.trust.assessor import CaseAssessment
from assay.trust.calibrated import CalibratedFitError, scoreable
from assay.trust.recalibrated import RecalibratedConfig, RecalibratedTrustModel, check_monotone
from assay.trust.reliability import Cohort
from assay.validation import measures as M
from calibrated_helpers import make_cases

CFG = RecalibratedConfig(n_boot=40)


@pytest.fixture(scope="module")
def world():
    fit, yf, _ = make_cases(6000, 1, exp_weight=0.8)
    ev, ye, pe = make_cases(5000, 3, exp_weight=0.8)
    return fit, yf, ev, ye, pe


@pytest.fixture(scope="module")
def model(world):
    fit, yf, *_ = world
    return RecalibratedTrustModel.fit(fit, yf, CFG)


def prov(cases):
    return np.array([a.result.ti for a in cases])


def test_scores_are_calibrated_on_later_cases(model, world):
    *_, ev, ye, _ = world
    ti = np.array([r[0] for r in model.predict(ev)])
    assert M.expected_calibration_error(ye.astype(bool), ti) < 0.03
    # and better than reading the provisional index itself as a probability
    assert M.expected_calibration_error(ye.astype(bool), ti) < M.expected_calibration_error(ye.astype(bool), prov(ev))


def test_the_order_of_cases_is_exactly_the_provisional_index(model, world):
    *_, ev, ye, _ = world
    new = np.array([r[0] for r in model.predict(ev)])
    old = prov(ev)
    order = np.argsort(old, kind="stable")
    assert np.all(np.diff(new[order]) >= -1e-9)  # never lowers a case that ranked higher
    wrong = ye == 0
    d = M.auroc(wrong, 100 - new) - M.auroc(wrong, 100 - old)
    assert abs(d) < 0.002  # ranking is untouched (only the clipped ends can tie)


def test_isotonic_is_monotone_too_and_can_only_blur_the_order(world):
    fit, yf, ev, ye, _ = world
    m = RecalibratedTrustModel.fit(fit, yf, RecalibratedConfig(method="isotonic", n_boot=20))
    new = np.array([r[0] for r in m.predict(ev)])
    old = prov(ev)
    order = np.argsort(old, kind="stable")
    assert np.all(np.diff(new[order]) >= -1e-9)
    assert check_monotone(m)
    assert M.expected_calibration_error(ye.astype(bool), new) < 0.03
    wrong = ye == 0
    assert M.auroc(wrong, 100 - new) >= M.auroc(wrong, 100 - old) - 0.01


def test_check_monotone_sees_a_decreasing_map(model):
    assert check_monotone(model)
    bad = next(iter(model.maps.values()))
    bad.slope = -abs(bad.slope)
    try:
        assert not check_monotone(model)
    finally:
        bad.slope = abs(bad.slope)


def test_the_interval_contains_the_estimate_and_widens_where_evidence_is_thin(model, world):
    *_, ev, _, _ = world
    preds = model.predict(ev)
    assert all(lo <= ti <= hi for ti, lo, hi in preds)
    ti = np.array([p[0] for p in preds])
    width = np.array([p[2] - p[1] for p in preds])
    assert width[ti < np.quantile(ti, 0.1)].mean() > 0 and width.max() < 40


def test_bands_come_from_tolerated_error_rates_on_the_lower_bound(model):
    assert model.state(99.0) is TrustState.HIGH       # expected error at most 1%
    assert model.state(98.9) is TrustState.MODERATE
    assert model.state(95.0) is TrustState.LOW        # expected error at least 5%... 95 is exactly that
    assert model.state(96.0) is TrustState.MODERATE
    assert model.state(40.0) is TrustState.LOW


def test_a_case_without_a_provisional_score_stays_insufficient(model):
    c = {k: Component.active(0.8, n=1) for k in ("fam", "drift", "dq")}
    c |= {"conf": Component.missing(), "rel": Component.active(0.8, n=100), "exp": Component.missing(),
          "hum": Component.inactive()}
    r = compute_trust_index(c, TrustConfig(max_interval_width=100.0))
    a = CaseAssessment("x", r, c, 0.1, 0.0, {}, Cohort("default", "card", "mid", "low", "b-1"), {})
    assert not scoreable(a)
    assert model.predict([a]) == [None]
    assert model.apply(a) is r  # returned unchanged


def test_refuses_too_little_evidence(world):
    fit, yf, *_ = world
    with pytest.raises(CalibratedFitError, match="not enough matured evidence"):
        RecalibratedTrustModel.fit(fit[:100], yf[:100], CFG)
    allright = np.ones(len(fit), int)
    with pytest.raises(CalibratedFitError, match="errors"):
        RecalibratedTrustModel.fit(fit, allright, CFG)


def test_refuses_to_recalibrate_an_index_that_ranks_the_wrong_way_round(world):
    fit, yf, *_ = world
    with pytest.raises(CalibratedFitError, match="does not rank correct above wrong"):
        RecalibratedTrustModel.fit(fit, 1 - yf, CFG)


def test_cases_without_explanation_testing_use_their_own_map_and_a_thin_pattern_falls_back():
    fit, yf, _ = make_cases(6000, 5, exp_weight=0.8, exp_share=0.5)
    m = RecalibratedTrustModel.fit(fit, yf, CFG)
    assert set(m.maps) == {"exp", "noexp"}
    few, yfew, _ = make_cases(6000, 6, exp_weight=0.8, exp_share=0.01)
    m2 = RecalibratedTrustModel.fit(few, yfew, CFG)
    assert set(m2.maps) == {"noexp"}
    exp_case = next(a for a in fit if a.components["exp"].score is not None)
    assert m2.predict([exp_case])[0] is not None  # still scored, by the map that does not use exp


def test_apply_returns_a_calibrated_mode_result_with_the_version_recorded(model, world):
    *_, ev, _, _ = world
    r = model.apply(ev[0])
    assert r.mode == "calibrated" and r.config_version.endswith("+recalibrated-0")
    assert r.ti_low <= r.ti <= r.ti_high


def test_only_enabled_after_the_gate_passed(model):
    assert not model.enabled
    model.gate = {"passed": False}
    assert not model.enabled
    model.gate = {"passed": True}
    assert model.enabled
    model.gate = None


def test_same_seed_same_result(world):
    fit, yf, ev, *_ = world
    a = RecalibratedTrustModel.fit(fit, yf, CFG).predict(ev[:50])
    b = RecalibratedTrustModel.fit(fit, yf, CFG).predict(ev[:50])
    assert a == b


def test_bad_config_is_refused():
    with pytest.raises(ValueError):
        RecalibratedConfig(method="spline")
    with pytest.raises(ValueError):
        RecalibratedConfig(high_error=0.1, low_error=0.05)
