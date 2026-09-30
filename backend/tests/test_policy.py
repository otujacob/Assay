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
