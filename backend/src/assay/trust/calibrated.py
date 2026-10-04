"""Calibrated Trust Index (PRD 5.6, V1).

A meta-model predicts whether the model's recommendation will turn out to be CORRECT (the model's
fraud call at the operating threshold agrees with the matured verified outcome, PRD 6.1). The Trust
Index is then 100 x that probability, so "80" means that among similar cases the recommendation was
right about 80% of the time (PRD 4.3), with an interval from a bootstrap.

How it is built:
  * Features: the component scores (logit scale), how much evidence sits behind `rel`, the risk band,
    and the provisional Trust Index. Including the provisional score means the meta-model can only
    add to what the fixed-weight index already knows.
  * Learner: a regularised logistic regression, fitted on one time window.
  * Calibration: a second, LATER window turns its scores into probabilities (sigmoid by default,
    isotonic optional). Fitting both on the same cases would make the calibration look better than it is.
  * Two models, by whether explanation testing ran for the case. A case without `exp` is scored by a
    model that never saw `exp`, not by one fed a made-up value.
  * Interval: `n_boot` resamples of both windows, each refitted; the 2.5th to 97.5th percentile of the
    resulting probabilities. It reflects uncertainty in the fit, not case-to-case randomness.
  * Bands: because the score is a probability, bands are set from tolerated error rates directly
    (High: expected error at most `high_error`; Low: at least `low_error`), judged on the interval's
    lower bound as before. There is no 70/40 to defend.

What it does not change: the Insufficient evidence gates (PRD 5.7) are applied first and untouched,
so a case that cannot be scored provisionally cannot be scored here either.

It may only be enabled after it passes the section 6 checks (see `validation/calibrated.py`).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from .assessor import CaseAssessment
from .index import Status, TrustResult, TrustState

BASE = ("conf", "rel", "fam", "drift", "dq")
BANDS = ("low", "medium", "high")
MODE = "calibrated"


class CalibratedFitError(Exception):
    """There is not enough matured evidence to fit a calibrated Trust Index. Stay in Provisional mode."""


@dataclass(frozen=True)
class CalibratedConfig:
    version: str = "calibrated-0"
    c: float = 1.0                 # inverse L2 strength of the logistic regression (parameter)
    calibration: str = "sigmoid"   # or "isotonic"
    n_boot: int = 40               # bootstrap refits for the interval (parameter)
    min_fit_cases: int = 300       # parameters: below these, refuse to fit (OPD-5 is the real decision)
    min_fit_errors: int = 20
    min_calibration_cases: int = 100
    min_calibration_errors: int = 8
    high_error: float = 0.01       # High trust: expected error at most this, at the interval's lower bound
    low_error: float = 0.05        # Low trust: expected error at least this
    seed: int = 0

    def __post_init__(self) -> None:
        if self.calibration not in ("sigmoid", "isotonic"):
            raise ValueError("calibration must be 'sigmoid' or 'isotonic'")
        if not 0 < self.high_error < self.low_error < 1:
            raise ValueError("need 0 < high_error < low_error < 1")


def _logit(x: float) -> float:
    x = min(0.99, max(0.01, x))
    return math.log(x / (1.0 - x))


def has_exp(a: CaseAssessment) -> bool:
    return a.components["exp"].status is Status.ACTIVE and a.components["exp"].score is not None


def scoreable(a: CaseAssessment) -> bool:
    """Only cases that have a provisional Trust Index get a calibrated one."""
    return a.result.ti is not None and all(
        a.components[k].status is Status.ACTIVE and a.components[k].score is not None for k in BASE)


def features(a: CaseAssessment, with_exp: bool) -> np.ndarray:
    comps = a.components
    row = [_logit(comps[k].score) for k in BASE]
    if with_exp:
        row.append(_logit(comps["exp"].score))
    row.append(math.log1p(comps["rel"].evidence_count))
    band = a.cohort.risk_band
    row += [1.0 if band == b else 0.0 for b in BANDS]
    row.append(math.log(max(a.result.ti, 1.0) / 100.0))  # the provisional index, on a log scale
    return np.array(row, dtype=float)


def feature_names(with_exp: bool) -> list[str]:
    return ([f"logit_{k}" for k in BASE] + (["logit_exp"] if with_exp else []) + ["log_rel_evidence"]
            + [f"band_{b}" for b in BANDS] + ["log_provisional_ti"])


class _Fit:
    """A scaler, a logistic regression and a calibrator, fitted on two windows."""

    def __init__(self, Xf, yf, Xc, yc, cfg: CalibratedConfig):
        self.scaler = StandardScaler().fit(Xf)
        self.lr = LogisticRegression(C=cfg.c, max_iter=2000).fit(self.scaler.transform(Xf), yf)
        raw = self._raw(Xc)
        if cfg.calibration == "isotonic":
            self.cal = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0).fit(raw, yc)
            self._platt = None
        else:
            z = np.log(np.clip(raw, 1e-6, 1 - 1e-6) / (1 - np.clip(raw, 1e-6, 1 - 1e-6))).reshape(-1, 1)
            self._platt = LogisticRegression(C=1e6, max_iter=2000).fit(z, yc)
            self.cal = None

    def _raw(self, X) -> np.ndarray:
        return self.lr.predict_proba(self.scaler.transform(X))[:, 1]

    def linear_form(self):
        """With sigmoid calibration the whole chain is one logistic function of the inputs, so many fits
        can be evaluated in a single vectorised step. None for isotonic, which is not."""
        if self._platt is None:
            return None
        return (self.scaler.mean_, self.scaler.scale_, self.lr.coef_[0], float(self.lr.intercept_[0]),
                float(self._platt.coef_[0, 0]), float(self._platt.intercept_[0]))

    def predict(self, X) -> np.ndarray:
        raw = self._raw(X)
        if self.cal is not None:
            return np.clip(self.cal.predict(raw), 0.0, 1.0)
        r = np.clip(raw, 1e-6, 1 - 1e-6)
        z = np.log(r / (1 - r)).reshape(-1, 1)
        return self._platt.predict_proba(z)[:, 1]


_Z_CLIP = math.log((1 - 1e-6) / 1e-6)  # the same clipping the calibrator applies to the raw probability


def predict_many(fits: list[_Fit], X: np.ndarray) -> np.ndarray:
    """Probabilities from several fits at once: shape (len(fits), len(X))."""
    forms = [f.linear_form() for f in fits]
    if any(f is None for f in forms):
        return np.array([f.predict(X) for f in fits])
    mu = np.array([f[0] for f in forms])
    sd = np.array([f[1] for f in forms])
    beta = np.array([f[2] for f in forms])
    b0 = np.array([f[3] for f in forms])
    pw = np.array([f[4] for f in forms])
    pb = np.array([f[5] for f in forms])
    z = np.einsum("fnd,fd->fn", (X[None, :, :] - mu[:, None, :]) / sd[:, None, :], beta) + b0[:, None]
    z = np.clip(z, -_Z_CLIP, _Z_CLIP)
    return 1.0 / (1.0 + np.exp(-(pw[:, None] * z + pb[:, None])))


def _has_both(y: np.ndarray) -> bool:
    return 0 < y.sum() < len(y)


@dataclass
class CalibratedTrustModel:
    cfg: CalibratedConfig
    fits: dict[str, _Fit]                       # "exp" / "noexp" -> the main fit
    boots: dict[str, list[_Fit]]                # bootstrap refits, for the interval
    n_fit: int
    n_calibration: int
    coefficients: dict[str, dict[str, float]] = field(default_factory=dict)
    gate: dict | None = None                    # set once the section 6 checks have been run

    # ---------------------------------------------------------------- fitting ----------------------
    @classmethod
    def fit(cls, fit_cases: list[CaseAssessment], fit_correct: np.ndarray,
            cal_cases: list[CaseAssessment], cal_correct: np.ndarray,
            cfg: CalibratedConfig | None = None) -> CalibratedTrustModel:
        """`fit_cases` must be EARLIER than `cal_cases`, and both later than everything the components
        were built from. The caller owns that split (see `temporal_split`)."""
        cfg = cfg or CalibratedConfig()
        fc = [(a, int(c)) for a, c in zip(fit_cases, fit_correct, strict=True) if scoreable(a)]
        cc = [(a, int(c)) for a, c in zip(cal_cases, cal_correct, strict=True) if scoreable(a)]
        yf, yc = np.array([c for _, c in fc]), np.array([c for _, c in cc])
        checks = ((len(yf), cfg.min_fit_cases, "fit cases"), (int((yf == 0).sum()), cfg.min_fit_errors, "fit errors"),
                  (len(yc), cfg.min_calibration_cases, "calibration cases"),
                  (int((yc == 0).sum()), cfg.min_calibration_errors, "calibration errors"))
        short = [f"{n} {what} (need {need})" for n, need, what in checks if n < need]
        if short:
            raise CalibratedFitError("not enough matured evidence: " + "; ".join(short))
        rng = np.random.default_rng(cfg.seed)
        fits: dict[str, _Fit] = {}
        boots: dict[str, list[_Fit]] = {}
        coefs: dict[str, dict[str, float]] = {}
        for name, with_exp in (("exp", True), ("noexp", False)):
            # The "exp" model is trained only on cases that have exp; the "noexp" model on all of them,
            # without the column. Both are applied only to cases with the matching availability.
            fsel = [(a, y) for a, y in fc if (has_exp(a) or not with_exp)]
            csel = [(a, y) for a, y in cc if (has_exp(a) or not with_exp)]
            Xf = np.array([features(a, with_exp) for a, _ in fsel])
            Xc = np.array([features(a, with_exp) for a, _ in csel])
            yfs, ycs = np.array([y for _, y in fsel]), np.array([y for _, y in csel])
            if len(yfs) < cfg.min_fit_cases or not _has_both(yfs) or not _has_both(ycs):
                continue  # this pattern cannot be fitted: its cases fall back to the model without `exp`
            main = _Fit(Xf, yfs, Xc, ycs, cfg)
            fits[name] = main
            blist: list[_Fit] = []
            tries = 0
            while len(blist) < cfg.n_boot and tries < cfg.n_boot * 5:
                tries += 1
                fi = rng.integers(0, len(yfs), len(yfs))
                ci = rng.integers(0, len(ycs), len(ycs))
                if not (_has_both(yfs[fi]) and _has_both(ycs[ci])):
                    continue
                blist.append(_Fit(Xf[fi], yfs[fi], Xc[ci], ycs[ci], cfg))
            boots[name] = blist
            coefs[name] = dict(zip(feature_names(with_exp), main.lr.coef_[0].round(4).tolist(), strict=True))
            coefs[name]["intercept"] = round(float(main.lr.intercept_[0]), 4)
        if "noexp" not in fits:
            raise CalibratedFitError("could not fit the model for cases without explanation testing")
        return cls(cfg, fits, boots, len(yf), len(yc), coefs)

    # ---------------------------------------------------------------- prediction --------------------
    def _which(self, a: CaseAssessment) -> str:
        return "exp" if (has_exp(a) and "exp" in self.fits) else "noexp"

    def predict(self, assessments: list[CaseAssessment]) -> list[tuple[float, float, float] | None]:
        """(TI, TI_low, TI_high) on the 0 to 100 scale per case, or None where the case has no provisional
        score (those stay Insufficient evidence)."""
        out: list[tuple[float, float, float] | None] = [None] * len(assessments)
        for name in ("exp", "noexp"):
            idx = [i for i, a in enumerate(assessments) if scoreable(a) and self._which(a) == name]
            if not idx or name not in self.fits:
                continue
            X = np.array([features(assessments[i], name == "exp") for i in idx])
            p = predict_many([self.fits[name]], X)[0]
            bp = predict_many(self.boots[name], X) if self.boots[name] else p[None, :]
            lo, hi = np.quantile(bp, 0.025, axis=0), np.quantile(bp, 0.975, axis=0)
            for k, i in enumerate(idx):
                out[i] = (100 * float(p[k]), 100 * float(min(lo[k], p[k])), 100 * float(max(hi[k], p[k])))
        return out

    def state(self, ti_low: float) -> TrustState:
        # A tiny tolerance, so a score exactly at a tolerated error rate lands in the band it names:
        # 1 - 0.99 is 0.010000000000000009 in floating point, which is not "at most 1%".
        expected_error = 1.0 - ti_low / 100.0
        if expected_error <= self.cfg.high_error + 1e-9:
            return TrustState.HIGH
        if expected_error >= self.cfg.low_error - 1e-9:
            return TrustState.LOW
        return TrustState.MODERATE

    def apply(self, a: CaseAssessment, scored: tuple[float, float, float] | None = None) -> TrustResult:
        """The case's calibrated TrustResult. A case the provisional index could not score is returned
        unchanged (still Insufficient evidence, with its reason codes)."""
        scored = scored if scored is not None else self.predict([a])[0]
        if scored is None:
            return a.result
        ti, lo, hi = scored
        name = self._which(a)
        return TrustResult(self.state(lo), ti, lo, hi, (), MODE, f"{a.result.config_version}+{self.cfg.version}",
                           dict(self.coefficients.get(name, {})))

    # ---------------------------------------------------------------- gate --------------------------
    @property
    def enabled(self) -> bool:
        """True only after the section 6 checks were run and passed."""
        return bool(self.gate and self.gate.get("passed") is True)


def temporal_split(times: np.ndarray, fit_frac: float = 0.45, cal_frac: float = 0.20) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Index arrays (fit, calibrate, evaluate), oldest to newest. Everything evaluated is later than
    everything the meta-model and its calibration saw (PRD 6.3)."""
    if not (0 < fit_frac and 0 < cal_frac and fit_frac + cal_frac < 1):
        raise ValueError("fit and calibration fractions must be positive and leave room to evaluate")
    order = np.argsort(times, kind="stable")
    a = int(len(order) * fit_frac)
    b = a + int(len(order) * cal_frac)
    return order[:a], order[a:b], order[b:]
