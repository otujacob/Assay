"""Recalibrated Trust Index (PRD 5.6): the provisional index, mapped to a probability, with nothing else changed.

The calibrated meta-model (calibrated.py) turned out well calibrated but not better than the provisional index at
ranking the recommendations that will be wrong, and in the stress part of the pre-set test it ranked them worse
(docs/validation). The ranking is what makes a Trust Index useful. This variant therefore keeps the provisional index's
ORDER exactly and learns only a monotone map from it to P(the recommendation is correct), on matured cases, so
"80" can be read as "among cases like this the recommendation was right about 80% of the time" (PRD 4.3) without
giving up any ranking.

How it is built:
  * One map per pattern of availability of explanation testing (with `exp` / without), because the provisional index
    means something slightly different when `exp` is missing. A pattern with too little evidence falls back to the
    map without `exp`.
  * Map: a logistic curve of the log-odds of the provisional index (Platt), or isotonic regression. A fitted curve
    that is not increasing means the provisional index carries no usable information in that window, and the fit is
    refused rather than shipped.
  * Interval: `n_boot` resamples of the fit cases, each refitted; the 2.5th to 97.5th percentile. It reflects uncertainty
    in the map, not case-to-case randomness.
  * Bands: set from tolerated error rates on the interval's lower bound, as in calibrated mode.

Because the map is monotone, the ranking of cases is the provisional index's (isotonic regression can merge neighbouring
scores into one value, which can only blur the order, never reverse it). The Insufficient-evidence gates (PRD 5.7) are
applied first and untouched.

It has the same interface as `CalibratedTrustModel` and is used in the same place, and like it may only be enabled after
it passes the section 6 checks (validation/calibrated.py).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

from .assessor import CaseAssessment
from .calibrated import MODE, CalibratedFitError, has_exp, scoreable
from .index import TrustResult, TrustState

VERSION = "recalibrated-0"


@dataclass(frozen=True)
class RecalibratedConfig:
    version: str = VERSION
    method: str = "sigmoid"        # or "isotonic"
    n_boot: int = 200              # bootstrap refits for the interval (parameter)
    min_cases: int = 300           # parameters: below these, refuse to fit (OPD-5 is the real decision)
    min_errors: int = 20
    high_error: float = 0.01       # High trust: expected error at most this, at the interval's lower bound
    low_error: float = 0.05        # Low trust: expected error at least this
    seed: int = 0

    def __post_init__(self) -> None:
        if self.method not in ("sigmoid", "isotonic"):
            raise ValueError("method must be 'sigmoid' or 'isotonic'")
        if not 0 < self.high_error < self.low_error < 1:
            raise ValueError("need 0 < high_error < low_error < 1")


def _x(ti: np.ndarray) -> np.ndarray:
    """Log-odds of the provisional index on a 0 to 1 scale, clipped so 0 and 100 stay finite."""
    p = np.clip(np.asarray(ti, dtype=float) / 100.0, 0.01, 0.99)
    return np.log(p / (1.0 - p))


class _Map:
    """A monotone map from the provisional index to P(correct), fitted on one set of cases."""

    def __init__(self, ti: np.ndarray, y: np.ndarray, method: str):
        x = _x(ti)
        if method == "isotonic":
            self.iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0).fit(x, y)
            self.slope, self.intercept, self.lr = None, None, None
        else:
            self.lr = LogisticRegression(C=1e6, max_iter=2000).fit(x.reshape(-1, 1), y)
            self.slope, self.intercept = float(self.lr.coef_[0, 0]), float(self.lr.intercept_[0])
            self.iso = None

    def predict(self, ti: np.ndarray) -> np.ndarray:
        x = _x(ti)
        if self.iso is not None:
            return np.clip(self.iso.predict(x), 0.0, 1.0)
        return 1.0 / (1.0 + np.exp(-(self.slope * x + self.intercept)))


def _has_both(y: np.ndarray) -> bool:
    return 0 < y.sum() < len(y)


@dataclass
class RecalibratedTrustModel:
    cfg: RecalibratedConfig
    maps: dict[str, _Map]                       # "exp" / "noexp"
    boots: dict[str, list[_Map]]
    n_fit: int
    n_calibration: int = 0                      # kept so reports read the same as calibrated mode's
    coefficients: dict[str, dict[str, float]] = field(default_factory=dict)
    gate: dict | None = None

    # ---------------------------------------------------------------- fitting ----------------------
    @classmethod
    def fit(cls, cases: list[CaseAssessment], correct: np.ndarray,
            cfg: RecalibratedConfig | None = None) -> RecalibratedTrustModel:
        """`cases` must be later than everything the components were built from. The caller owns that split."""
        cfg = cfg or RecalibratedConfig()
        sel = [(a, int(c)) for a, c in zip(cases, correct, strict=True) if scoreable(a)]
        y_all = np.array([c for _, c in sel])
        short = [f"{n} {what} (need {need})" for n, need, what in
                 ((len(y_all), cfg.min_cases, "cases"), (int((y_all == 0).sum()), cfg.min_errors, "errors")) if n < need]
        if short:
            raise CalibratedFitError("not enough matured evidence: " + "; ".join(short))
        rng = np.random.default_rng(cfg.seed)
        maps: dict[str, _Map] = {}
        boots: dict[str, list[_Map]] = {}
        coefs: dict[str, dict[str, float]] = {}
        for name, only_exp in (("exp", True), ("noexp", False)):
            part = [(a, y) for a, y in sel if (has_exp(a) or not only_exp)]
            ti = np.array([a.result.ti for a, _ in part], dtype=float)
            y = np.array([c for _, c in part])
            if len(y) < cfg.min_cases or int((y == 0).sum()) < cfg.min_errors or not _has_both(y):
                continue
            main = _Map(ti, y, cfg.method)
            if main.slope is not None and main.slope <= 0:
                raise CalibratedFitError(f"the provisional index does not rank correct above wrong recommendations "
                                         f"in this window (slope {main.slope:.3f}); nothing sensible to recalibrate")
            maps[name] = main
            blist: list[_Map] = []
            tries = 0
            while len(blist) < cfg.n_boot and tries < cfg.n_boot * 5:
                tries += 1
                i = rng.integers(0, len(y), len(y))
                if _has_both(y[i]):
                    blist.append(_Map(ti[i], y[i], cfg.method))
            boots[name] = blist
            if main.slope is not None:
                coefs[name] = {"slope_on_logit_ti": round(main.slope, 4), "intercept": round(main.intercept, 4)}
        if "noexp" not in maps:
            raise CalibratedFitError("could not fit the map for cases without explanation testing")
        return cls(cfg, maps, boots, len(y_all), 0, coefs)

    # ---------------------------------------------------------------- prediction --------------------
    def _which(self, a: CaseAssessment) -> str:
        return "exp" if (has_exp(a) and "exp" in self.maps) else "noexp"

    def predict(self, assessments: list[CaseAssessment]) -> list[tuple[float, float, float] | None]:
        """(TI, TI_low, TI_high) on the 0 to 100 scale per case, or None where the case has no provisional score."""
        out: list[tuple[float, float, float] | None] = [None] * len(assessments)
        for name in ("exp", "noexp"):
            idx = [i for i, a in enumerate(assessments) if scoreable(a) and self._which(a) == name]
            if not idx or name not in self.maps:
                continue
            ti = np.array([assessments[i].result.ti for i in idx], dtype=float)
            p = self.maps[name].predict(ti)
            bp = np.array([m.predict(ti) for m in self.boots[name]]) if self.boots[name] else p[None, :]
            lo, hi = np.quantile(bp, 0.025, axis=0), np.quantile(bp, 0.975, axis=0)
            for k, i in enumerate(idx):
                out[i] = (100 * float(p[k]), 100 * float(min(lo[k], p[k])), 100 * float(max(hi[k], p[k])))
        return out

    def state(self, ti_low: float) -> TrustState:
        expected_error = 1.0 - ti_low / 100.0
        if expected_error <= self.cfg.high_error + 1e-9:  # tolerance: 1 - 0.99 is not exactly 0.01 in floating point
            return TrustState.HIGH
        if expected_error >= self.cfg.low_error - 1e-9:
            return TrustState.LOW
        return TrustState.MODERATE

    def apply(self, a: CaseAssessment, scored: tuple[float, float, float] | None = None) -> TrustResult:
        scored = scored if scored is not None else self.predict([a])[0]
        if scored is None:
            return a.result
        ti, lo, hi = scored
        return TrustResult(self.state(lo), ti, lo, hi, (), MODE, f"{a.result.config_version}+{self.cfg.version}",
                           dict(self.coefficients.get(self._which(a), {})))

    @property
    def enabled(self) -> bool:
        """True only after the section 6 checks were run and passed."""
        return bool(self.gate and self.gate.get("passed") is True)


def check_monotone(model: RecalibratedTrustModel, grid: np.ndarray | None = None) -> bool:
    """The map never lowers the probability as the provisional index rises (used by tests and the validation report)."""
    grid = np.linspace(0, 100, 201) if grid is None else grid
    return all(bool(np.all(np.diff(m.predict(grid)) >= -1e-12)) for m in model.maps.values()) and math.isfinite(float(grid[0]))
