"""Decision policy engine (PRD 10.2, 10.3, FR-22, FR-23).

Fixed evaluation order: hard rules, data-quality gate, novelty gate, then the
trust x risk matrix. Automation level is 0 in the MVP, so every output is a
recommendation.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from assay.trust import ReasonCode, TrustState


class RiskBand(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class Action(str, Enum):
    APPROVE = "approve"
    APPROVE_SAMPLED_QA = "approve_sampled_qa"
    BLOCK = "block"
    ESCALATE = "escalate"
    REQUEST_HUMAN_REVIEW = "request_human_review"
    REQUEST_HUMAN_REVIEW_PRIORITY = "request_human_review_priority"
    HOLD = "hold"


@dataclass(frozen=True)
class PolicyConfig:
    version: str = "policy-0"
    t_low: float = 0.30
    t_high: float = 0.75  # operating threshold, also defines the fraud call (PRD 6.1)
    automation_level: int = 0
    dq_gate_action: Action = Action.REQUEST_HUMAN_REVIEW  # or HOLD, per tenant


@dataclass(frozen=True)
class PolicyInput:
    risk: float
    trust_state: TrustState
    reason_codes: tuple[ReasonCode, ...] = ()
    hard_rule_action: Action | None = None  # e.g. sanctions hit


@dataclass(frozen=True)
class PolicyResult:
    action: Action
    risk_band: RiskBand
    gate: str  # which step decided: hard_rule | data_quality | novelty | matrix
    queue: str | None
    policy_version: str
    automation_level: int


_M = {
    RiskBand.LOW: {
        TrustState.HIGH: Action.APPROVE,
        TrustState.MODERATE: Action.APPROVE_SAMPLED_QA,
        TrustState.LOW: Action.REQUEST_HUMAN_REVIEW,
        TrustState.INSUFFICIENT: Action.REQUEST_HUMAN_REVIEW,
    },
    RiskBand.MEDIUM: {
        TrustState.HIGH: Action.REQUEST_HUMAN_REVIEW,
        TrustState.MODERATE: Action.REQUEST_HUMAN_REVIEW,
        TrustState.LOW: Action.ESCALATE,
        TrustState.INSUFFICIENT: Action.ESCALATE,
    },
    RiskBand.HIGH: {
        TrustState.HIGH: Action.BLOCK,
        TrustState.MODERATE: Action.REQUEST_HUMAN_REVIEW_PRIORITY,
        TrustState.LOW: Action.ESCALATE,
        TrustState.INSUFFICIENT: Action.ESCALATE,
    },
}


def risk_band(risk: float, cfg: PolicyConfig) -> RiskBand:
    if risk >= cfg.t_high:
        return RiskBand.HIGH
    if risk < cfg.t_low:
        return RiskBand.LOW
    return RiskBand.MEDIUM


def evaluate(inp: PolicyInput, cfg: PolicyConfig | None = None) -> PolicyResult:
    cfg = cfg or PolicyConfig()
    if cfg.automation_level != 0:
        raise NotImplementedError("MVP runs at automation level 0 only (FR-23)")
    band = risk_band(inp.risk, cfg)

    def res(action: Action, gate: str, queue: str | None = None) -> PolicyResult:
        return PolicyResult(action, band, gate, queue, cfg.version, cfg.automation_level)

    if inp.hard_rule_action is not None:
        return res(inp.hard_rule_action, "hard_rule")
    if ReasonCode.DATA_QUALITY_FLOOR in inp.reason_codes:
        return res(cfg.dq_gate_action, "data_quality")
    if ReasonCode.UNFAMILIAR_PATTERN in inp.reason_codes:
        return res(Action.REQUEST_HUMAN_REVIEW, "novelty", queue="novelty")
    return res(_M[band][inp.trust_state], "matrix")
