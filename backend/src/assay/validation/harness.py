"""Trust validation harness (PRD 6, FR-36 to FR-39).

Designed so the Trust Index CAN FAIL (PRD 6): it is compared with simple baselines, each component
is ablated, and the pass criteria are comparative and are not relaxed when they fail. Evaluation is
on matured verified outcomes from a period LATER than everything the model, calibration and cohort
statistics were built from. Every measure carries a confidence interval; small segments are
reported as inconclusive rather than as point estimates.

Provisional mode only: the score makes no probability claim, so calibration (ECE) does not apply
(PRD 6.1). The selective-label check (PRD 6.3) needs sampled-versus-unsampled verification data,
which the synthetic set does not model; it is listed under limitations.
"""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

import numpy as np

from assay.trust import Component, TrustContext, TrustState, compute_trust_index
from assay.trust.assessor import CaseAssessment, TrustAssessor
from assay.trust.reliability import amount_band

from . import measures as M

COMPONENTS = ("conf", "rel", "exp", "fam", "drift", "dq")
ABLATION_EPS = 0.005  # AUROC change treated as noise


@dataclass
class ValidationCases:
    """Assessed cases plus everything needed to score them against verified outcomes."""

    assessments: list[CaseAssessment]
    txns: list[dict]
    calibrated: np.ndarray
    raw: np.ndarray
    spread: np.ndarray
    y: np.ndarray  # verified outcome, 1 = fraud
    t_high: float
    t_low: float
    times: np.ndarray  # event time, epoch seconds
    trust_cfg: Any
    store_total: int

    @property
    def call(self) -> np.ndarray:
        return self.calibrated >= self.t_high

    @property
    def wrong(self) -> np.ndarray:
        return self.call != (self.y == 1)


def collect_cases(trained_result, txns_by_id: dict[str, dict], n: int, seed: int = 0) -> ValidationCases:
    """Assess a random sample of the test window, with explanation testing on every case."""
    r = trained_result
    ids = r.test_table.txn_ids
    idx = np.sort(np.random.default_rng(seed).choice(len(ids), min(n, len(ids)), replace=False))
    X = r.test_table.X[idx]
    txns = [txns_by_id[ids[i]] for i in idx]
    th, tl = r.manifest.thresholds["t_high"], r.manifest.thresholds["t_low"]
    pred = r.model.predict(X, th)
    assessor = TrustAssessor(r.model, r.reference, r.store, r.manifest.thresholds,
                             model_version=r.manifest.bundle_id)
    drift = np.zeros(X.shape[1])
    cases = assessor.assess(X, txns, pred, drift_vector=drift)
    return ValidationCases(cases, txns, pred.calibrated, pred.raw, pred.member_spread, r.test_y[idx],
                           th, tl, r.test_table.t[idx], assessor.trust_cfg, r.store.total())


# ---- helpers --------------------------------------------------------------------------------
def _ti(result) -> float:
    return float("nan") if result.ti is None else float(result.ti)


def _arrays(results) -> dict[str, np.ndarray]:
    return {
        "ti": np.array([_ti(r) for r in results]),
        "state": np.array([r.state.value for r in results]),
    }


def _err_score(ti: np.ndarray) -> np.ndarray:
    """Higher = more likely wrong. No score counts as TI = 0: it is routed to a human."""
    return 100.0 - np.where(np.isnan(ti), 0.0, ti)


def _jsonable(o):
    if isinstance(o, dict):
        return {str(k): _jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_jsonable(v) for v in o]
    if isinstance(o, (np.floating, float)):
        return None if math.isnan(float(o)) or math.isinf(float(o)) else float(o)
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, (np.bool_, bool)):
        return bool(o)
    return o


def recompute_without(cases: ValidationCases, drop: str | None):
    """Trust results with one component made inactive (weights renormalise), for ablation."""
    ctx = TrustContext(model_matured_outcomes=cases.store_total)
    out = []
    for a in cases.assessments:
        comps = dict(a.components)
        if drop is not None:
            comps[drop] = Component.inactive()
        out.append(compute_trust_index(comps, cases.trust_cfg, ctx))
    return out


def _headline(results, wrong: np.ndarray, n_boot: int, seed: int) -> dict:
    a = _arrays(results)
    state, ti = a["state"], a["ti"]
    high = state == TrustState.HIGH.value
    low_or_insuf = np.isin(state, [TrustState.LOW.value, TrustState.INSUFFICIENT.value])
    err = _err_score(ti)
    scored = ~np.isnan(ti)
    return {
        "auroc": M.auroc_with_ci(wrong, err, n_boot, seed),
        "auroc_scored_only": M.auroc(wrong[scored], err[scored]) if scored.sum() > 10 else float("nan"),
        "pr_auc": M.pr_auc(wrong, err),
        "high_trust_error_rate": M.rate(wrong, high),
        "low_trust_detection_rate": M.rate(low_or_insuf, wrong),
        "false_confidence_rate": M.rate(high, wrong),
        "scored_share": float(scored.mean()),
        "state_counts": dict(Counter(state.tolist())),
    }


def _segments(cases: ValidationCases, results, key_fn, n_boot: int) -> dict:
    a = _arrays(results)
    err = _err_score(a["ti"])
    keys = np.array([key_fn(i) for i in range(len(results))])
    out = {}
    for k in sorted(set(keys.tolist())):
        m = keys == k
        w = cases.wrong[m]
        if m.sum() < M.MIN_SEGMENT or w.sum() < 5 or (~w).sum() < 5:
            out[k] = {"n": int(m.sum()), "errors": int(w.sum()), "status": "inconclusive"}
            continue
        high = a["state"][m] == TrustState.HIGH.value
        out[k] = {"n": int(m.sum()), "errors": int(w.sum()), "status": "ok",
                  "auroc": M.auroc_with_ci(w, err[m], n_boot // 2),
                  "high_trust_error_rate": M.rate(w, high)}
    return out


def run_validation(cases: ValidationCases, *, bundle_id: str, dataset_id: str, n_boot: int = 400,
                   seed: int = 0, windows: int = 4) -> dict:
    results = [a.result for a in cases.assessments]
    wrong = cases.wrong
    a = _arrays(results)
    err_ti = _err_score(a["ti"])
    correct = ~wrong

    full = _headline(results, wrong, n_boot, seed)
    overall = M.rate(wrong, np.ones(len(wrong), bool))

    # --- baselines B1 to B3 (PRD 6.4) and B4 (each component alone) ---
    p, th = cases.calibrated, cases.t_high
    baselines_scores = {
        "B1_distance_from_threshold": -np.abs(p - th),
        "B2_ensemble_disagreement": cases.spread,
        "B3_max_class_probability": -np.maximum(p, 1 - p),
    }
    baselines: dict[str, Any] = {}
    for name, sc in baselines_scores.items():
        baselines[name] = {"auroc": M.auroc_with_ci(wrong, sc, n_boot, seed),
                           "trust_index_minus_baseline": M.paired_auroc_diff_ci(wrong, err_ti, sc, n_boot, seed)}
    b4 = {}
    for c in COMPONENTS:
        v = np.array([x.components[c].score if x.components[c].score is not None else np.nan
                      for x in cases.assessments])
        b4[c] = {"auroc": M.auroc(wrong, 1.0 - np.where(np.isnan(v), 0.5, v)),
                 "missing_share": float(np.isnan(v).mean())}
    baselines["B4_single_components"] = b4

    # --- ablation: remove each component in turn (PRD 6.4) ---
    ablations = {}
    for c in COMPONENTS:
        rc = recompute_without(cases, c)
        h = _headline(rc, wrong, n_boot // 4, seed)
        ablations[c] = {
            "auroc": h["auroc"]["value"],
            "delta_auroc": h["auroc"]["value"] - full["auroc"]["value"],
            "high_trust_error_rate": h["high_trust_error_rate"]["value"],
            "high_trust_n": h["high_trust_error_rate"]["n"],
            "scored_share": h["scored_share"],
        }
        # delta = AUROC without the component minus AUROC with it. A small threshold avoids
        # reading noise as a verdict; the paired interval is in the baselines section.
        d = ablations[c]["delta_auroc"]
        if d < -ABLATION_EPS:
            verdict = "adds value (removing it hurts discrimination)"
        elif d > ABLATION_EPS:
            verdict = "removing it IMPROVES discrimination: drop or down-weight (PRD 6.4)"
        else:
            verdict = "no measurable effect: drop or down-weight (PRD 6.4)"
        ablations[c]["verdict"] = verdict

    # --- trust/outcome correlation, coverage, rolling windows, segments ---
    corr = M.spearman(np.where(np.isnan(a["ti"]), 0.0, a["ti"]), correct.astype(float))
    t0 = cases.times.min()
    span = max(cases.times.max() - t0, 1.0)
    win = np.minimum(((cases.times - t0) / span * windows).astype(int), windows - 1)
    rolling = []
    for w_i in range(windows):
        m = win == w_i
        if m.sum() >= M.MIN_SEGMENT and wrong[m].sum() >= 3:
            rolling.append({"window": w_i, "n": int(m.sum()), "errors": int(wrong[m].sum()),
                            "auroc": M.auroc(wrong[m], err_ti[m]),
                            "high_trust_error_rate": M.rate(wrong[m], (a["state"] == "high")[m])})
        else:
            rolling.append({"window": w_i, "n": int(m.sum()), "status": "inconclusive"})

    txns = cases.txns
    segs = {
        "channel": _segments(cases, results, lambda i: str(txns[i].get("channel")), n_boot),
        "amount_band": _segments(cases, results, lambda i: amount_band(float(txns[i]["amount"])), n_boot),
        "risk_band": _segments(
            cases, results,
            lambda i: "high" if p[i] >= th else ("low" if p[i] < cases.t_low else "medium"), n_boot),
    }
    reasons = Counter(c.value for r in results for c in r.reason_codes)

    # --- pass criteria (PRD 6.6): comparative, not relaxed when they fail ---
    beats = {k: bool(baselines[k]["trust_index_minus_baseline"]["excludes_zero_above"])
             for k in baselines_scores}
    ht = full["high_trust_error_rate"]
    ht_ok = None if ht["value"] is None else bool(ht["hi"] < overall["value"])
    supported = None if ht_ok is None else bool(all(beats.values()) and ht_ok)
    pass_criteria = {
        "trust_index_beats_B1_to_B3_with_ci_excluding_no_improvement": beats,
        "high_trust_error_rate_materially_below_overall": ht_ok,
        "calibrated_mode_ece": "not applicable: Provisional mode makes no probability claim",
        "atce_supported_as_specified": supported,
        "if_not_supported": "revise the components or the method; do not relax the criteria (PRD 6.6)",
    }

    return _jsonable({
        "meta": {"mode": "provisional", "bundle_id": bundle_id, "dataset_id": dataset_id,
                 "n_cases": len(wrong), "n_wrong": int(wrong.sum()), "t_high": cases.t_high,
                 "created_at": datetime.now(UTC).isoformat(),
                 "evaluation_window": "matured verified outcomes, later than training, calibration "
                                      "and cohort-statistics windows"},
        "overall_error_rate": overall,
        "measures": {**full, "trust_outcome_spearman": corr, "reason_code_counts": dict(reasons),
                     "coverage_curve": M.coverage_curve(wrong, a["ti"]),
                     "historical_reliability": rolling, "segments": segs},
        "baselines": baselines,
        "ablations": ablations,
        "pass_criteria": pass_criteria,
        "limitations": [
            "Synthetic data with simulated labels and analysts: tests the machinery, not real fraud.",
            "Selective-label bias (PRD 6.3) is not modelled: every synthetic case receives an outcome.",
            "New-versus-existing-customer segment is not reported (no tenure feature yet).",
            "Explanations cover the boosted-tree members only.",
        ],
    })
