"""Model Reliability (PRD 5.1, FR-15): how often has the model been right on similar cases,
judged against matured verified outcomes.

Cohort = product, channel, amount band, risk band, model version. rel is the LOWER bound of the
Wilson interval on cohort accuracy, so thin cohorts are penalised rather than trusted.
"Correct" means the model's fraud call at t_high agrees with the verified outcome (PRD 6.1).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from math import sqrt

import numpy as np

Z_95 = 1.959963984540054
Z_99 = 2.5758293035489004
DEFAULT_AMOUNT_EDGES = (25.0, 100.0, 500.0, 2000.0)  # parameters, per tenant


def wilson_interval(successes: int, n: int, z: float = Z_95) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion. Returns (lo, hi)."""
    if n < 0 or successes < 0 or successes > n:
        raise ValueError("require 0 <= successes <= n")
    if n == 0:
        return 0.0, 1.0
    p = successes / n
    z2 = z * z
    denom = 1 + z2 / n
    centre = (p + z2 / (2 * n)) / denom
    margin = z * sqrt(p * (1 - p) / n + z2 / (4 * n * n)) / denom
    return max(0.0, centre - margin), min(1.0, centre + margin)


def amount_band(amount: float, edges: tuple[float, ...] = DEFAULT_AMOUNT_EDGES) -> str:
    return f"b{int(np.searchsorted(edges, amount, side='right'))}"


def risk_band_name(risk: float, t_low: float, t_high: float) -> str:
    return "high" if risk >= t_high else ("low" if risk < t_low else "medium")


@dataclass(frozen=True)
class Cohort:
    product: str
    channel: str
    amount_band: str
    risk_band: str
    model_version: str

    def parent(self) -> Cohort:
        """A broader cohort (drops channel and amount band) for thin-cohort fallback (PRD 18)."""
        return Cohort(self.product, "*", "*", self.risk_band, self.model_version)


@dataclass
class CohortStore:
    edges: tuple[float, ...] = DEFAULT_AMOUNT_EDGES
    # A plain dict, not a defaultdict(lambda): the store is pickled into the signed bundle.
    _stats: dict[Cohort, list[int]] = field(default_factory=dict)

    def add(self, cohort: Cohort, correct: bool) -> None:
        for c in (cohort, cohort.parent()) if cohort.parent() != cohort else (cohort,):
            s = self._stats.setdefault(c, [0, 0])
            s[0] += int(correct)
            s[1] += 1

    def get(self, cohort: Cohort) -> tuple[int, int]:
        s = self._stats.get(cohort)
        return (s[0], s[1]) if s else (0, 0)

    def total(self) -> int:
        return sum(v[1] for c, v in self._stats.items() if c.channel != "*")

    def reliability(self, cohort: Cohort, *, fallback_to_parent_below: int | None = None):
        """(rel, lo, hi, n): Wilson 95% lower bound as the score, 99% lower and 95% upper as the
        interval, and the evidence count. With `fallback_to_parent_below`, a cohort thinner than
        that uses its broader parent cohort instead (PRD 18, optional)."""
        c, n = self.get(cohort)
        if fallback_to_parent_below is not None and n < fallback_to_parent_below:
            c, n = self.get(cohort.parent())
        lo95, hi95 = wilson_interval(c, n, Z_95)
        lo99, _ = wilson_interval(c, n, Z_99)
        return lo95, min(lo99, lo95), hi95, n
