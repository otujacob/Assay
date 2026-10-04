"""Calibrated Trust Index (PRD 5.6): the meta-model, checked against cases whose TRUE probability of a
correct recommendation is known, so calibration and discrimination can be measured against the answer."""

import numpy as np
import pytest

from assay.trust import Component, TrustConfig, TrustResult, TrustState, compute_trust_index
from assay.trust.assessor import CaseAssessment
from assay.trust.calibrated import (
    CalibratedConfig,
    CalibratedFitError,
    CalibratedTrustModel,
    feature_names,
    features,
    has_exp,
    scoreable,
    temporal_split,
)
from assay.validation import measures as M
from calibrated_helpers import make_cases


@pytest.fixture(scope="module")
def world():
    fit, yf, _ = make_cases(4000, 1, exp_weight=0.8)
    cal, yc, _ = make_cases(2000, 2, exp_weight=0.8)
    ev, ye, pe = make_cases(5000, 3, exp_weight=0.8)
    return fit, yf, cal, yc, ev, ye, pe


@pytest.fixture(scope="module")
def model(world):
    fit, yf, cal, yc, *_ = world
    return CalibratedTrustModel.fit(fit, yf, cal, yc, CalibratedConfig(n_boot=25))


def test_the_scores_are_calibrated_against_the_true_probabilities(model, world):
    *_, ev, ye, pe = world
    ti = np.array([r[0] for r in model.predict(ev)])
    assert M.expected_calibration_error(ye.astype(bool), ti) < 0.03     # a score of 80 is right about 80% of the time
    assert np.abs(ti / 100 - pe).mean() < 0.04                          # and it tracks the true probability per case


def test_a_provisional_score_read_as_a_probability_is_not_calibrated_but_this_is(model, world):
    """The point of the mode: the fixed-weight index ranks cases but its number is not a rate."""
    *_, ev, ye, _ = world
    prov = np.array([a.result.ti for a in ev])
    cal = np.array([r[0] for r in model.predict(ev)])
    e_prov = M.expected_calibration_error(ye.astype(bool), prov)
    e_cal = M.expected_calibration_error(ye.astype(bool), cal)
    assert e_cal < e_prov / 3, (e_cal, e_prov)


def test_it_ranks_errors_about_as_well_as_knowing_the_true_probability(model, world):
    *_, ev, ye, pe = world
    wrong = ye == 0
    cal = np.array([r[0] for r in model.predict(ev)])
    best = M.auroc(wrong, 1 - pe)                                       # the best any score could do
    assert M.auroc(wrong, 100 - cal) > best - 0.02


def test_higher_reliability_means_higher_trust_holding_the_rest_fixed(model):
    base, _, _ = make_cases(1, 9)
    lo, hi = [], []
    for rel, store in ((0.2, lo), (0.8, hi)):
        c = dict(base[0].components)
        c["rel"] = Component.active(rel, n=120)
        a = CaseAssessment("x", compute_trust_index(c, TrustConfig(max_interval_width=100)), c, 0.1, 0.0, {"q": 1.0},
                           base[0].cohort, {})
        store.append(model.predict([a])[0][0])
    assert hi[0] > lo[0] + 3


def test_the_interval_brackets_the_estimate_and_narrows_with_more_data(world):
    fit, yf, cal, yc, ev, *_ = world
    small = CalibratedTrustModel.fit(fit[:600], yf[:600], cal[:300], yc[:300], CalibratedConfig(n_boot=25, min_fit_cases=300))
    big = CalibratedTrustModel.fit(fit, yf, cal, yc, CalibratedConfig(n_boot=25))
    widths = {}
    for name, m in (("small", small), ("big", big)):
        preds = m.predict(ev[:500])
        assert all(lo <= ti <= hi for ti, lo, hi in preds)
        widths[name] = np.mean([hi - lo for _, lo, hi in preds])
    assert widths["big"] < widths["small"]


def test_it_is_deterministic_for_a_seed(world):
    fit, yf, cal, yc, ev, *_ = world
    cfg = CalibratedConfig(n_boot=10)
    a = CalibratedTrustModel.fit(fit[:1500], yf[:1500], cal[:800], yc[:800], cfg).predict(ev[:50])
    b = CalibratedTrustModel.fit(fit[:1500], yf[:1500], cal[:800], yc[:800], cfg).predict(ev[:50])
    assert a == b


def test_a_case_without_explanation_testing_is_scored_by_a_model_that_never_saw_it(model, world):
    *_, ev, _, _ = world
    assert set(model.coefficients) == {"exp", "noexp"}
    assert "logit_exp" in model.coefficients["exp"] and "logit_exp" not in model.coefficients["noexp"]
    src = ev[0]
    c = dict(src.components)
    c["exp"] = Component.missing()
    stripped = CaseAssessment("y", compute_trust_index(c, TrustConfig(max_interval_width=100)), c, 0.1, 0.0,
                              {"q": 1.0}, src.cohort, {})
    assert has_exp(src) and not has_exp(stripped)
    assert model._which(src) == "exp" and model._which(stripped) == "noexp"
    assert model.predict([stripped])[0] is not None      # missing exp does not make it unscoreable


def test_explanation_reliability_is_used_when_it_carries_signal_and_ignored_when_not(world):
    """The exp model should lean on exp only if exp tells you something about correctness."""
    fit, yf, cal, yc, *_ = world                                    # these were generated with exp_weight 0.8
    informative = CalibratedTrustModel.fit(fit, yf, cal, yc, CalibratedConfig(n_boot=5))
    f2, y2, _ = make_cases(4000, 11, exp_weight=0.0)
    c2, yc2, _ = make_cases(2000, 12, exp_weight=0.0)
    useless = CalibratedTrustModel.fit(f2, y2, c2, yc2, CalibratedConfig(n_boot=5))
    assert informative.coefficients["exp"]["logit_exp"] > 0.3
    assert abs(useless.coefficients["exp"]["logit_exp"]) < abs(informative.coefficients["exp"]["logit_exp"]) / 2


def test_a_case_with_no_provisional_score_stays_insufficient_with_its_reasons(model, world):
    *_, ev, _, _ = world
    a = ev[0]
    insufficient = compute_trust_index({**a.components, "conf": Component.missing()}, TrustConfig())
    assert insufficient.state is TrustState.INSUFFICIENT
    b = CaseAssessment("z", insufficient, {**a.components, "conf": Component.missing()}, 0.1, 0.0, {"q": 1.0}, a.cohort, {})
    assert not scoreable(b) and model.predict([b]) == [None]
    assert model.apply(b) is insufficient                               # returned untouched, reasons and all


def test_the_result_says_it_is_calibrated_and_carries_the_coefficients(model, world):
    *_, ev, _, _ = world
    r = model.apply(ev[0])
    assert r.mode == "calibrated" and r.ti is not None and r.ti_low <= r.ti <= r.ti_high
    assert r.config_version.endswith("+calibrated-0") and "intercept" in r.weights_used
    assert r.state in (TrustState.HIGH, TrustState.MODERATE, TrustState.LOW)


@pytest.mark.parametrize("ti_low,state", [
    (99.0, TrustState.HIGH),        # expected error 1%: at the tolerance
    (98.9, TrustState.MODERATE),
    (95.1, TrustState.MODERATE),
    (95.0, TrustState.LOW),         # expected error 5%
    (60.0, TrustState.LOW),
])
def test_bands_come_from_tolerated_error_rates_not_arbitrary_cut_points(model, ti_low, state):
    assert model.state(ti_low) is state


def test_the_tolerated_error_rates_are_configurable(world):
    fit, yf, cal, yc, *_ = world
    strict = CalibratedTrustModel.fit(fit, yf, cal, yc, CalibratedConfig(n_boot=3, high_error=0.001, low_error=0.02))
    assert strict.state(99.0) is TrustState.MODERATE and strict.state(99.95) is TrustState.HIGH
    assert strict.state(97.9) is TrustState.LOW


def test_the_bands_are_ordered_by_the_interval_lower_bound(model, world):
    *_, ev, ye, _ = world
    results = [model.apply(a) for a in ev]
    err = {s: np.mean(1 - ye[[r.state is s for r in results]]) if any(r.state is s for r in results) else np.nan
           for s in (TrustState.HIGH, TrustState.MODERATE, TrustState.LOW)}
    assert err[TrustState.HIGH] < err[TrustState.MODERATE] < err[TrustState.LOW]
    assert err[TrustState.HIGH] <= 0.02    # the High band honours (roughly) the 1% tolerance it was set from


def test_isotonic_calibration_also_works_and_is_monotone(world):
    fit, yf, cal, yc, ev, ye, _ = world
    m = CalibratedTrustModel.fit(fit, yf, cal, yc, CalibratedConfig(n_boot=3, calibration="isotonic"))
    ti = np.array([r[0] for r in m.predict(ev)])
    assert M.expected_calibration_error(ye.astype(bool), ti) < 0.05
    raw = m.fits["noexp"]._raw(np.array([features(a, False) for a in ev[:200]]))
    order = np.argsort(raw)
    assert np.all(np.diff(m.fits["noexp"].predict(np.array([features(a, False) for a in ev[:200]]))[order]) >= -1e-9)


@pytest.mark.parametrize("n,msg", [(150, "fit cases"), (4000, None)])
def test_it_refuses_to_fit_on_too_little_evidence(world, n, msg):
    fit, yf, cal, yc, *_ = world
    if msg is None:
        CalibratedTrustModel.fit(fit[:n], yf[:n], cal, yc, CalibratedConfig(n_boot=2))
        return
    with pytest.raises(CalibratedFitError, match=msg):
        CalibratedTrustModel.fit(fit[:n], yf[:n], cal, yc)


def test_it_refuses_when_there_are_too_few_wrong_recommendations_to_learn_from():
    fit, yf, _ = make_cases(1500, 21, base=9.0)         # almost nothing is ever wrong
    cal, yc, _ = make_cases(500, 22, base=9.0)
    with pytest.raises(CalibratedFitError, match="errors"):
        CalibratedTrustModel.fit(fit, yf, cal, yc)


def test_it_is_not_enabled_until_the_section_6_checks_have_passed(model):
    assert model.enabled is False
    model.gate = {"passed": False}
    assert model.enabled is False
    model.gate = {"passed": True}
    assert model.enabled is True
    model.gate = None


def test_bad_configuration_is_refused():
    for kw in ({"calibration": "magic"}, {"high_error": 0.1, "low_error": 0.05}, {"high_error": 0}, {"low_error": 1}):
        with pytest.raises(ValueError):
            CalibratedConfig(**kw)


def test_features_line_up_with_their_names_and_exp_is_optional(world):
    fit, *_ = world
    a = fit[0]
    assert len(features(a, True)) == len(feature_names(True)) and len(features(a, False)) == len(feature_names(False))
    assert len(feature_names(True)) == len(feature_names(False)) + 1


def test_temporal_split_is_oldest_to_newest_with_no_overlap():
    t = np.array([5, 1, 9, 3, 7, 2, 8, 4, 6, 0])
    fit, cal, ev = temporal_split(t, 0.4, 0.2)
    assert len(fit) == 4 and len(cal) == 2 and len(ev) == 4
    assert t[fit].max() < t[cal].min() and t[cal].max() < t[ev].min()
    assert sorted(np.concatenate([fit, cal, ev]).tolist()) == list(range(10))
    for bad in ((0, 0.2), (0.5, 0.5), (0.6, 0.5), (-0.1, 0.3)):
        with pytest.raises(ValueError):
            temporal_split(t, *bad)


def test_a_provisional_result_is_what_the_type_says(world):
    fit, *_ = world
    assert isinstance(fit[0].result, TrustResult) and fit[0].result.mode == "provisional"


def test_the_fast_vectorised_path_matches_scikit_learns_own_predictions(model, world):
    from assay.trust.calibrated import predict_many

    *_, ev, _, _ = world
    X = np.array([features(a, False) for a in ev[:400]])
    fits = [model.fits["noexp"], *model.boots["noexp"][:5]]
    fast = predict_many(fits, X)
    slow = np.array([f.predict(X) for f in fits])
    assert np.allclose(fast, slow, atol=1e-9)


def test_scoring_one_case_at_a_time_is_fast_enough_for_the_synchronous_path(model, world):
    import time

    *_, ev, _, _ = world
    t0 = time.perf_counter()
    for a in ev[:200]:
        model.apply(a)
    per_case_ms = (time.perf_counter() - t0) / 200 * 1000
    assert per_case_ms < 5, f"{per_case_ms:.1f} ms per case"   # a parameter, not a measured requirement (OPD-1)
