"""Feedback quality scores (PRD 11.1, FR-30). SHADOW MODE: computed and stored, consumed by nothing.

Nothing in the MVP trains on feedback. These formulas are an initial design; the weights and
thresholds are parameters, not validated values (PRD 11.1, OPD-11), and are versioned so scores
can be recomputed when they change.

  AAS  Analyst Accuracy: Beta posterior of P(analyst decision matches the verified outcome) on
       matured cases, with recency decay; blind-review cases count more (no anchoring). Unknown
       (None) until the analyst has enough matured cases.
  FCS  Feedback Confidence: how far this one piece of feedback can be trusted.
  LVS  Learning Value: how much the case would teach the model. Sets sampling priority only; it
       never makes a label more true.
  CRS  Correction Reliability: for an override, how often past corrections of the same kind were
       right. Unknown (None) until enough history.
  FQS  = FCS x (0.5 + 0.5 x A), A = AAS for ordinary decisions, CRS for overrides, 0.5 if unknown.
"""

from __future__ import annotations

from dataclasses import dataclass

FORMULA_VERSION = "fqs-0"


@dataclass(frozen=True)
class FeedbackConfig:
    min_matured_for_aas: int = 10
    min_corrections_for_crs: int = 5
    decay_half_life_days: float = 90.0
    blind_weight: float = 2.0
    # FCS weights (sum to 1)
    w_confidence: float = 0.30
    w_checklist: float = 0.25
    w_corroboration: float = 0.20
    w_reason: float = 0.10
    w_independent: float = 0.15  # blind reviews are free of anchoring
    # LVS weights (sum to 1)
    w_uncertainty: float = 0.35
    w_novelty: float = 0.25
    w_disagreement: float = 0.25
    w_cost: float = 0.15
    # Placeholder disposition thresholds (OPD-11)
    accept_threshold: float = 0.7
    reject_threshold: float = 0.3


def analyst_accuracy(history: list[dict], cfg: FeedbackConfig) -> tuple[float | None, int]:
    """`history`: dicts with correct (bool), age_days (float), blind (bool), for MATURED cases only."""
    n = len(history)
    if n < cfg.min_matured_for_aas:
        return None, n
    num = den = 0.0
    for h in history:
        w = 0.5 ** (h["age_days"] / cfg.decay_half_life_days) * (cfg.blind_weight if h["blind"] else 1.0)
        num += w * h["correct"]
        den += w
    return (1.0 + num) / (2.0 + den), n  # Beta(1, 1) prior


def correction_reliability(correct: int, n: int, cfg: FeedbackConfig) -> float | None:
    return None if n < cfg.min_corrections_for_crs else (1.0 + correct) / (2.0 + n)


def feedback_confidence(confidence: float | None, checklist: dict, corroborated: bool,
                        reason_present: bool, blind: bool, cfg: FeedbackConfig) -> float:
    items = list(checklist.values())
    completeness = (sum(1 for v in items if v) / len(items)) if items else 0.0
    conf = 0.5 if confidence is None else confidence  # an unstated confidence is not assumed high
    return (cfg.w_confidence * conf + cfg.w_checklist * completeness
            + cfg.w_corroboration * float(corroborated) + cfg.w_reason * float(reason_present)
            + cfg.w_independent * (1.0 if blind else 0.6))


def learning_value(model_confidence: float, novelty: float, disagrees: bool, amount_norm: float,
                   cfg: FeedbackConfig) -> float:
    return (cfg.w_uncertainty * (1.0 - model_confidence) + cfg.w_novelty * novelty
            + cfg.w_disagreement * float(disagrees) + cfg.w_cost * amount_norm)


def feedback_quality(fcs: float, a: float | None) -> float:
    return fcs * (0.5 + 0.5 * (0.5 if a is None else a))


def disposition(fqs: float, *, reason_present: bool, unsure: bool, conflicted: bool,
                outcome_known: bool, cfg: FeedbackConfig) -> tuple[str, str]:
    """PRD 11.2 in shadow. With no verified outcome at action time, feedback is level 3 at best,
    so most records are deferred until an outcome arrives or a second review corroborates."""
    if unsure:  # deferred, never forced into a label, whatever its quality (PRD 11.3)
        return "defer", "analyst marked unsure"
    if not reason_present:
        return "reject", "missing reason code"
    if fqs < cfg.reject_threshold:
        return "reject", "feedback quality below reject threshold"
    if conflicted:
        return "defer", "conflicting analyst decisions, awaiting adjudication"
    if outcome_known and fqs >= cfg.accept_threshold:
        return "accept", "verified outcome available and quality above accept threshold"
    return "defer", "awaiting verified outcome or corroboration"
