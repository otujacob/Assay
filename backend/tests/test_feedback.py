import pytest

from assay.review.feedback import (
    FORMULA_VERSION,
    FeedbackConfig,
    analyst_accuracy,
    correction_reliability,
    disposition,
    feedback_confidence,
    feedback_quality,
    learning_value,
)

CFG = FeedbackConfig()


def hist(n, correct, age=0.0, blind=False):
    return [{"correct": c, "age_days": age, "blind": blind} for c in ([True] * correct + [False] * (n - correct))]


def test_aas_is_unknown_until_enough_matured_cases():
    assert analyst_accuracy(hist(9, 9), CFG) == (None, 9)
    aas, n = analyst_accuracy(hist(10, 10), CFG)
    assert n == 10 and aas == pytest.approx(11 / 12)  # Beta(1,1) posterior mean, no decay


def test_aas_hand_values_and_direction():
    assert analyst_accuracy(hist(20, 10), CFG)[0] == pytest.approx(11 / 22)
    assert analyst_accuracy(hist(20, 18), CFG)[0] > analyst_accuracy(hist(20, 12), CFG)[0]


def test_recency_decay_and_blind_weighting():
    old = analyst_accuracy(hist(10, 0, age=0) + hist(10, 10, age=360), CFG)[0]  # recent errors, old hits
    new = analyst_accuracy(hist(10, 10, age=0) + hist(10, 0, age=360), CFG)[0]  # recent hits, old errors
    assert new > 0.5 > old
    blind_hits = analyst_accuracy(hist(10, 10, blind=True) + hist(10, 0), CFG)[0]
    plain_hits = analyst_accuracy(hist(10, 10) + hist(10, 0, blind=True), CFG)[0]
    assert blind_hits > 0.5 > plain_hits  # blind cases weigh double


def test_crs_unknown_then_posterior():
    assert correction_reliability(4, 4, CFG) is None
    assert correction_reliability(8, 8, CFG) == pytest.approx(9 / 10)


def test_fcs_components_and_bounds():
    full = feedback_confidence(1.0, {"a": True, "b": True}, True, True, True, CFG)
    empty = feedback_confidence(0.0, {"a": False, "b": False}, False, False, False, CFG)
    assert full == pytest.approx(1.0) and 0 <= empty < 0.15
    assert feedback_confidence(None, {}, False, True, False, CFG) == pytest.approx(0.15 + 0.1 + 0.09)
    assert feedback_confidence(0.8, {"a": True}, False, True, True, CFG) > \
        feedback_confidence(0.8, {"a": True}, False, True, False, CFG)  # blind is more independent


def test_lvs_orders_cases_by_how_much_they_teach():
    boring = learning_value(0.95, 0.05, False, 0.1, CFG)
    rich = learning_value(0.2, 0.9, True, 0.9, CFG)
    assert rich > boring and 0 <= boring < rich <= 1


def test_fqs_formula_matches_prd_and_unknown_a_uses_half():
    assert feedback_quality(0.8, 1.0) == pytest.approx(0.8)
    assert feedback_quality(0.8, 0.0) == pytest.approx(0.4)
    assert feedback_quality(0.8, None) == pytest.approx(0.8 * 0.75)


def test_disposition_rules():
    kw = {"reason_present": True, "unsure": False, "conflicted": False, "outcome_known": False, "cfg": CFG}
    assert disposition(0.9, **{**kw, "reason_present": False})[0] == "reject"
    assert disposition(0.1, **kw)[0] == "reject"
    assert disposition(0.9, **{**kw, "unsure": True})[0] == "defer"
    assert disposition(0.05, **{**kw, "unsure": True})[0] == "defer"  # even a low-quality "unsure"
    assert disposition(0.05, **{**kw, "reason_present": False, "unsure": True})[0] == "defer"
    assert disposition(0.9, **{**kw, "conflicted": True})[0] == "defer"
    assert disposition(0.9, **kw)[0] == "defer"  # no verified outcome yet: never accepted on opinion alone
    assert disposition(0.9, **{**kw, "outcome_known": True})[0] == "accept"
    assert disposition(0.5, **{**kw, "outcome_known": True})[0] == "defer"


def test_formula_is_versioned():
    assert FORMULA_VERSION == "fqs-0"
