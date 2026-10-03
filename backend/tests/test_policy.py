import itertools

import pytest

from assay.policy import Action, PolicyConfig, PolicyInput, RiskBand, evaluate, risk_band
from assay.trust import ReasonCode, TrustState

RISK = {RiskBand.LOW: 0.1, RiskBand.MEDIUM: 0.5, RiskBand.HIGH: 0.9}

EXPECTED = {
    ("low", "high"): Action.APPROVE, ("low", "moderate"): Action.APPROVE_SAMPLED_QA,
    ("low", "low"): Action.REQUEST_HUMAN_REVIEW,
    ("low", "insufficient_evidence"): Action.REQUEST_HUMAN_REVIEW,
    ("medium", "high"): Action.REQUEST_HUMAN_REVIEW,
    ("medium", "moderate"): Action.REQUEST_HUMAN_REVIEW,
    ("medium", "low"): Action.ESCALATE, ("medium", "insufficient_evidence"): Action.ESCALATE,
    ("high", "high"): Action.BLOCK, ("high", "moderate"): Action.REQUEST_HUMAN_REVIEW_PRIORITY,
    ("high", "low"): Action.ESCALATE, ("high", "insufficient_evidence"): Action.ESCALATE,
}


@pytest.mark.parametrize("band,state", list(itertools.product(RiskBand, TrustState)))
def test_every_matrix_cell(band, state):
    r = evaluate(PolicyInput(RISK[band], state))
    assert r.action is EXPECTED[(band.value, state.value)] and r.gate == "matrix"


def test_band_boundaries():
    c = PolicyConfig(t_low=0.3, t_high=0.75)
    assert risk_band(0.2999, c) is RiskBand.LOW
    assert risk_band(0.30, c) is RiskBand.MEDIUM
    assert risk_band(0.75, c) is RiskBand.HIGH


def test_gate_precedence():
    hard = evaluate(PolicyInput(0.1, TrustState.HIGH, (ReasonCode.DATA_QUALITY_FLOOR,), Action.BLOCK))
    assert hard.gate == "hard_rule" and hard.action is Action.BLOCK
    dq = evaluate(PolicyInput(0.9, TrustState.INSUFFICIENT,
                              (ReasonCode.DATA_QUALITY_FLOOR, ReasonCode.UNFAMILIAR_PATTERN)))
    assert dq.gate == "data_quality"
    nov = evaluate(PolicyInput(0.9, TrustState.INSUFFICIENT, (ReasonCode.UNFAMILIAR_PATTERN,)))
    assert nov.gate == "novelty" and nov.queue == "novelty"


def test_automation_level_zero_only():
    with pytest.raises(NotImplementedError):
        evaluate(PolicyInput(0.1, TrustState.HIGH), PolicyConfig(automation_level=1))


def test_segment_rule_sends_large_approvals_to_a_human_but_never_loosens_anything():
    big = PolicyConfig(always_review_above=1000)
    # a transaction the matrix would approve: low risk, high trust
    assert evaluate(PolicyInput(0.1, TrustState.HIGH, amount=999.99), big).action == Action.APPROVE
    r = evaluate(PolicyInput(0.1, TrustState.HIGH, amount=1000), big)       # at the limit counts
    assert (r.action, r.gate) == (Action.REQUEST_HUMAN_REVIEW, "segment_rule")
    assert evaluate(PolicyInput(0.1, TrustState.MODERATE, amount=5000), big).action == Action.REQUEST_HUMAN_REVIEW
    # decisions that are already strict are left exactly as they were
    for risk, state, expected in ((0.9, TrustState.HIGH, Action.BLOCK), (0.9, TrustState.LOW, Action.ESCALATE),
                                  (0.5, TrustState.LOW, Action.ESCALATE)):
        r = evaluate(PolicyInput(risk, state, amount=99999), big)
        assert (r.action, r.gate) == (expected, "matrix")
    # no amount means the rule cannot apply, and with no rule configured nothing changes
    assert evaluate(PolicyInput(0.1, TrustState.HIGH), big).action == Action.APPROVE
    assert evaluate(PolicyInput(0.1, TrustState.HIGH, amount=1e9), PolicyConfig()).action == Action.APPROVE


def test_segment_rule_cannot_override_a_hard_rule_or_an_earlier_gate():
    big = PolicyConfig(always_review_above=1)
    r = evaluate(PolicyInput(0.1, TrustState.HIGH, hard_rule_action=Action.BLOCK, amount=500), big)
    assert (r.action, r.gate) == (Action.BLOCK, "hard_rule")
    r = evaluate(PolicyInput(0.1, TrustState.HIGH, (ReasonCode.UNFAMILIAR_PATTERN,), amount=500), big)
    assert r.gate == "novelty"
