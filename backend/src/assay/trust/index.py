"""Provisional AI Trust Index (PRD 5.3 to 5.7, FR-18, FR-19).

Weighted geometric mean over the seven reliability components. Weights and
thresholds are expert priors (OPD-3), not validated values, and are passed in
as versioned configuration rather than living in code paths.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from math import exp, log

EPSILON = 0.01

DEFAULT_WEIGHTS: dict[str, float] = {
    "conf": 0.15,
    "rel": 0.25,
    "exp": 0.15,
    "fam": 0.20,
    "drift": 0.10,
    "dq": 0.10,
    "hum": 0.05,
}
CRITICAL = frozenset({"conf", "rel", "fam", "dq"})


class Status(str, Enum):
    ACTIVE = "active"  # available for this case
    INACTIVE = "inactive"  # not enabled for tenant/phase: removed, weights renormalised
    MISSING = "missing"  # expected but unavailable: penalised, weight kept


class TrustState(str, Enum):
    HIGH = "high"
    MODERATE = "moderate"
    LOW = "low"
    INSUFFICIENT = "insufficient_evidence"


class ReasonCode(str, Enum):
    MISSING_CRITICAL = "MISSING_CRITICAL"
    THIN_COHORT = "THIN_COHORT"
    DATA_QUALITY_FLOOR = "DATA_QUALITY_FLOOR"
    UNFAMILIAR_PATTERN = "UNFAMILIAR_PATTERN"
    WIDE_INTERVAL = "WIDE_INTERVAL"
    MODEL_TOO_NEW = "MODEL_TOO_NEW"
    STALE_REFERENCE = "STALE_REFERENCE"
    ATCE_UNAVAILABLE = "ATCE_UNAVAILABLE"


@dataclass(frozen=True)
class Component:
    status: Status = Status.ACTIVE
    score: float | None = None
    lo: float | None = None
    hi: float | None = None
    evidence_count: int = 0

    @staticmethod
    def active(score: float, lo: float | None = None, hi: float | None = None, n: int = 0):
        return Component(Status.ACTIVE, score, score if lo is None else lo,
                         score if hi is None else hi, n)

    @staticmethod
    def missing() -> Component:
        return Component(Status.MISSING)

    @staticmethod
    def inactive() -> Component:
        return Component(Status.INACTIVE)


@dataclass(frozen=True)
class TrustConfig:
    version: str = "trust-config-0"
    weights: dict[str, float] = field(default_factory=lambda: dict(DEFAULT_WEIGHTS))
    high_threshold: float = 70.0
    low_threshold: float = 40.0
    n_min: int = 30  # OPD-5 placeholder
    dq_floor: float = 0.5
    novelty_ceiling: float = 0.95  # N_max, OPD-9 placeholder
    max_interval_width: float = 60.0
    min_model_outcomes: int = 0  # MODEL_TOO_NEW gate off until OPD-5 is decided
    max_reference_age_days: float = 90.0


@dataclass(frozen=True)
class TrustContext:
    """Case-level facts the components alone do not carry."""

    model_matured_outcomes: int = 10**9
    reference_age_days: float = 0.0


@dataclass(frozen=True)
class TrustResult:
    state: TrustState
    ti: float | None
    ti_low: float | None
    ti_high: float | None
    reason_codes: tuple[ReasonCode, ...]
    mode: str
    config_version: str
    weights_used: dict[str, float]


def _gm(scores: dict[str, float], weights: dict[str, float]) -> float:
    return 100.0 * exp(sum(w * log(max(scores[k], EPSILON)) for k, w in weights.items()))


def _clamp(x: float) -> float:
    return min(1.0, max(0.0, x))


def compute_trust_index(
    components: dict[str, Component],
    config: TrustConfig | None = None,
    context: TrustContext | None = None,
) -> TrustResult:
    cfg = config or TrustConfig()
    ctx = context or TrustContext()

    unknown = set(components) - set(cfg.weights)
    if unknown:
        raise ValueError(f"unknown components: {sorted(unknown)}")

    # Renormalise only over components that are not inactive.
    live = {k: w for k, w in cfg.weights.items()
            if components.get(k, Component.missing()).status is not Status.INACTIVE}
    total = sum(live.values())
    weights = {k: w / total for k, w in live.items()}

    reasons: list[ReasonCode] = []
    comps = {k: components.get(k, Component.missing()) for k in weights}

    rel = comps.get("rel")
    if rel is not None and rel.status is Status.ACTIVE and rel.evidence_count < cfg.n_min:
        reasons.append(ReasonCode.THIN_COHORT)
    missing_critical = [k for k, c in comps.items()
                        if k in CRITICAL and c.status is Status.MISSING]
    if missing_critical:
        reasons.append(ReasonCode.MISSING_CRITICAL)

    dq = comps.get("dq")
    if dq and dq.status is Status.ACTIVE and dq.score is not None and dq.score < cfg.dq_floor:
        reasons.append(ReasonCode.DATA_QUALITY_FLOOR)
    fam = comps.get("fam")
    if fam and fam.status is Status.ACTIVE and fam.score is not None \
            and (1.0 - fam.score) >= cfg.novelty_ceiling:
        reasons.append(ReasonCode.UNFAMILIAR_PATTERN)
    if ctx.model_matured_outcomes < cfg.min_model_outcomes:
        reasons.append(ReasonCode.MODEL_TOO_NEW)
    if ctx.reference_age_days > cfg.max_reference_age_days:
        reasons.append(ReasonCode.STALE_REFERENCE)

    def insufficient() -> TrustResult:
        return TrustResult(TrustState.INSUFFICIENT, None, None, None, tuple(reasons),
                           "provisional", cfg.version, weights)

    if reasons:
        return insufficient()

    point, lo, hi = {}, {}, {}
    for k, c in comps.items():
        if c.status is Status.MISSING:  # non-critical here
            point[k], lo[k], hi[k] = 0.5, 0.0, 1.0
        else:
            assert c.score is not None
            point[k] = _clamp(c.score)
            lo[k] = _clamp(c.lo if c.lo is not None else c.score)
            hi[k] = _clamp(c.hi if c.hi is not None else c.score)

    ti, ti_low, ti_high = _gm(point, weights), _gm(lo, weights), _gm(hi, weights)
    ti_low = min(ti_low, ti)
    ti_high = max(ti_high, ti)

    if ti_high - ti_low > cfg.max_interval_width:
        reasons.append(ReasonCode.WIDE_INTERVAL)
        return insufficient()

    if ti_low >= cfg.high_threshold:
        state = TrustState.HIGH
    elif ti_low >= cfg.low_threshold:
        state = TrustState.MODERATE
    else:
        state = TrustState.LOW
    return TrustResult(state, ti, ti_low, ti_high, (), "provisional", cfg.version, weights)
