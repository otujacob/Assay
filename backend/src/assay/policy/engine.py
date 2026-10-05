"""Decision policy engine (PRD 10.2, 10.3, FR-22, FR-23).

Fixed evaluation order: hard rules, data-quality gate, novelty gate, then the
trust x risk matrix, and last the institution's segment rule (which can only make a
decision stricter). Automation level is 0 in the MVP, so every output is a
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


# The actions that put a case in front of a person. The review queue and policy replay both count these.
REVIEW_ACTIONS = frozenset({Action.REQUEST_HUMAN_REVIEW.value, Action.REQUEST_HUMAN_REVIEW_PRIORITY.value,
                            Action.ESCALATE.value, Action.HOLD.value})


@dataclass(frozen=True)
class PolicyConfig:
    version: str = "policy-0"
    t_low: float = 0.30
    t_high: float = 0.75  # operating threshold, also defines the fraud call (PRD 6.1)
    automation_level: int = 0
    dq_gate_action: Action = Action.REQUEST_HUMAN_REVIEW  # or HOLD, per tenant
    # Segment rule (PRD 10.4): always send a transaction of at least this amount to a human, even if the
    # matrix would approve it. It never loosens anything: block, escalate and review stay as they are.
    always_review_above: float | None = None


@dataclass(frozen=True)
class PolicyInput:
    risk: float
    trust_state: TrustState
    reason_codes: tuple[ReasonCode, ...] = ()
    hard_rule_action: Action | None = None  # e.g. sanctions hit
    amount: float | None = None  # for the segment rule; None means the rule cannot apply
    kill_switch: bool = False  # the tenant's kill switch is engaged: every case goes to a person (PRD 12.4)


@dataclass(frozen=True)
class PolicyResult:
    action: Action
    risk_band: RiskBand
    gate: str  # which step decided: kill_switch | hard_rule | data_quality | novelty | matrix
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

    if inp.kill_switch:  # first of all: nothing the model or a rule says matters while the model is not to be relied on
        return res(Action.REQUEST_HUMAN_REVIEW, "kill_switch")
    if inp.hard_rule_action is not None:
        return res(inp.hard_rule_action, "hard_rule")
    if ReasonCode.DATA_QUALITY_FLOOR in inp.reason_codes:
        return res(cfg.dq_gate_action, "data_quality")
    if ReasonCode.UNFAMILIAR_PATTERN in inp.reason_codes:
        return res(Action.REQUEST_HUMAN_REVIEW, "novelty", queue="novelty")
    action = _M[band][inp.trust_state]
    if (cfg.always_review_above is not None and inp.amount is not None
            and inp.amount >= cfg.always_review_above
            and action in (Action.APPROVE, Action.APPROVE_SAMPLED_QA)):
        return res(Action.REQUEST_HUMAN_REVIEW, "segment_rule")
    return res(action, "matrix")
