"""Validate the calibrated Trust Index and apply the gate (PRD 5.6, 6.1 to 6.6).

The cases are split in time into three consecutive windows. The meta-model is FITTED on the oldest,
CALIBRATED on the next, and EVALUATED on the newest, so nothing it is scored on was seen by it or by
its calibration (PRD 6.3). The cases must already be later than everything the components were built
from, which is what `collect_cases` guarantees.

The gate is the PRD's: calibrated mode may replace Provisional mode only if it
  1. beats B1 to B3 at ranking wrong recommendations, with an interval that excludes no improvement,
  2. has a high-trust error rate materially below the overall error rate, and
  3. has a calibration error (ECE) within the tolerance agreed with the institution.
The tolerance is a parameter. Where there are too few errors to judge a criterion it fails: the gate
errs towards staying in Provisional mode.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime

import numpy as np

from assay.trust import TrustState
from assay.trust.calibrated import CalibratedConfig, CalibratedTrustModel, temporal_split

from . import measures as M
from .harness import ValidationCases, _err_score

DEFAULT_ECE_TOLERANCE = 0.03  # a parameter: the real value is agreed with each institution (PRD 6.6)


@dataclass(frozen=True)
class CalibratedValidationConfig:
    fit_frac: float = 0.45
    cal_frac: float = 0.20
    ece_tolerance: float = DEFAULT_ECE_TOLERANCE
    n_boot: int = 400
    seed: int = 0


def _reliability(correct: np.ndarray, ti: np.ndarray, bins: int = 10) -> list[dict]:
    order = np.argsort(ti, kind="stable")
    out = []
    for chunk in np.array_split(order, min(bins, len(ti))):
        if len(chunk):
            out.append({"n": len(chunk), "mean_predicted": float(ti[chunk].mean() / 100),
                        "observed_correct": float(correct[chunk].mean())})
    return out


def _slice_metrics(mask: np.ndarray, wrong, err_cal, err_prov, cal_ti, prov_ti, scored, state, b3, v) -> dict:
    """The headline numbers on one part of the evaluation window."""
    n = int(mask.sum())
    w = wrong[mask]
    out: dict = {"n": n, "wrong": int(w.sum())}
    if n < 30 or w.sum() < 3 or w.sum() == n:
        return out | {"status": "inconclusive: too few cases or errors"}
    sc = mask & scored
    out["auroc_calibrated"] = M.auroc(w, err_cal[mask])
    out["auroc_provisional"] = M.auroc(w, err_prov[mask])
    out["auroc_B3"] = M.auroc(w, b3[mask])
    out["cal_minus_prov"] = M.paired_auroc_diff_ci(w, err_cal[mask], err_prov[mask], v.n_boot, v.seed)
    out["cal_minus_B3"] = M.paired_auroc_diff_ci(w, err_cal[mask], b3[mask], v.n_boot, v.seed)
    if sc.sum() >= 20:
        out["ece_calibrated"] = M.expected_calibration_error(~wrong[sc], cal_ti[sc])
        out["ece_provisional_read_as_probability"] = M.expected_calibration_error(~wrong[sc], prov_ti[sc])
    hi = mask & (state == TrustState.HIGH.value)
    out["high_band"] = {"n": int(hi.sum()), "observed_error_rate": float(wrong[hi].mean()) if hi.any() else None}
    return out


def evaluate_calibrated(cases: ValidationCases, calibrated_cfg: CalibratedConfig | None = None,
                        vcfg: CalibratedValidationConfig | None = None,
                        slice_at: float | None = None, fit_model=None) -> tuple[CalibratedTrustModel, dict]:
    """Fit, calibrate and evaluate on consecutive windows, run the gate, and return (model, report).
    The model's `gate` is set from the result, so `model.enabled` is true only if the gate passed.
    Raises CalibratedFitError if there is too little matured evidence to fit at all.

    `fit_model(fit_cases, fit_correct, cal_cases, cal_correct)` replaces the meta-model with another that has the same
    interface (for example the recalibration-only map, trust/recalibrated.py). The windows and the gate are the same.

    `slice_at` (an event time, epoch seconds) additionally reports the evaluation window in two parts,
    before and after it: for example before and after a new fraud type appears."""
    cfg, v = calibrated_cfg or CalibratedConfig(), vcfg or CalibratedValidationConfig()
    fit_i, cal_i, ev_i = temporal_split(cases.times, v.fit_frac, v.cal_frac)
    correct = ~cases.wrong
    A = cases.assessments
    fit_args = ([A[i] for i in fit_i], correct[fit_i].astype(int), [A[i] for i in cal_i], correct[cal_i].astype(int))
    model = fit_model(*fit_args) if fit_model else CalibratedTrustModel.fit(*fit_args, cfg)

    ev = [A[i] for i in ev_i]
    wrong = cases.wrong[ev_i]
    preds = model.predict(ev)
    scored = np.array([p is not None for p in preds])
    cal_ti = np.array([p[0] if p else np.nan for p in preds])
    results = [model.apply(a, p) for a, p in zip(ev, preds, strict=True)]
    state = np.array([r.state.value for r in results])
    prov_ti = np.array([a.result.ti if a.result.ti is not None else np.nan for a in ev])

    err_cal, err_prov = _err_score(cal_ti), _err_score(prov_ti)   # no score counts as TI = 0: it goes to a human
    p, th = cases.calibrated[ev_i], cases.t_high
    baselines = {"B1_distance_from_threshold": -np.abs(p - th), "B2_ensemble_disagreement": cases.spread[ev_i],
                 "B3_max_class_probability": -np.maximum(p, 1 - p)}
    vs = {k: M.paired_auroc_diff_ci(wrong, err_cal, sc, v.n_boot, v.seed) for k, sc in baselines.items()}
    vs_prov = M.paired_auroc_diff_ci(wrong, err_cal, err_prov, v.n_boot, v.seed)

    overall = M.rate(wrong, np.ones(len(wrong), bool))
    high = state == TrustState.HIGH.value
    ht = M.rate(wrong, high)
    ece = M.expected_calibration_error(~wrong[scored], cal_ti[scored]) if scored.sum() >= 20 else float("nan")
    ece_prov_as_rate = (M.expected_calibration_error(~wrong[scored], prov_ti[scored])
                        if scored.sum() >= 20 else float("nan"))

    beats = {k: bool(d["excludes_zero_above"]) for k, d in vs.items()}
    ht_ok = None if ht["value"] is None or not np.isfinite(ht["hi"]) else bool(ht["hi"] < overall["value"])
    ece_ok = bool(np.isfinite(ece) and ece <= v.ece_tolerance)
    passed = bool(all(beats.values()) and ht_ok is True and ece_ok)
    gate = {"passed": passed, "evaluated_on": len(ev_i), "wrong_in_evaluation": int(wrong.sum()),
            "criteria": {"beats_B1_to_B3_with_ci_excluding_no_improvement": beats,
                         "high_trust_error_rate_materially_below_overall": ht_ok,
                         "ece_within_tolerance": ece_ok, "ece_tolerance": v.ece_tolerance},
            "checked_at": datetime.now(UTC).isoformat()}
    model.gate = gate

    band_error = {}
    for s in ("high", "moderate", "low", "insufficient_evidence"):
        m = state == s
        band_error[s] = {"n": int(m.sum()), "observed_error_rate": float(wrong[m].mean()) if m.any() else None}
    report = {
        "meta": {"mode": "calibrated", "n_fit": model.n_fit, "n_calibration": model.n_calibration,
                 "n_evaluation": len(ev_i), "wrong_in_evaluation": int(wrong.sum()),
                 "high_error_tolerance": model.cfg.high_error, "low_error_tolerance": model.cfg.low_error,
                 "variant": model.cfg.version},
        "auroc": {"calibrated": M.auroc(wrong, err_cal), "provisional": M.auroc(wrong, err_prov),
                  **{k: M.auroc(wrong, sc) for k, sc in baselines.items()}},
        "calibrated_minus_baseline": vs,
        "calibrated_minus_provisional": vs_prov,
        "ece": {"calibrated": ece, "provisional_read_as_probability": ece_prov_as_rate},
        "reliability_curve": _reliability(~wrong[scored], cal_ti[scored]) if scored.sum() >= 20 else [],
        "overall_error_rate": overall, "high_trust_error_rate": ht,
        "band_observed_error": band_error,
        "scored_share": float(scored.mean()),
        "mean_interval_width": float(np.mean([r.ti_high - r.ti_low for r in results if r.ti is not None])),
        "gate": gate,
        "coefficients": model.coefficients,
    }
    if slice_at is not None:
        t = cases.times[ev_i]
        args = (wrong, err_cal, err_prov, cal_ti, prov_ti, scored, state, baselines["B3_max_class_probability"], v)
        report["slices"] = {"before": _slice_metrics(t < slice_at, *args), "after": _slice_metrics(t >= slice_at, *args)}
    return model, report
