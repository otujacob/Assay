"""Trust validation measures (PRD 6.2). Pure functions over arrays, so they can be checked against
hand-computed values and against synthetic data with known truth (FR-36).

Conventions: `wrong` is True where the model's fraud call disagrees with the matured verified
outcome (PRD 6.1). A score for predicting errors is "higher = more likely wrong"; for the Trust
Index that is 100 - TI, with no score (Insufficient evidence) counted as TI = 0 because those cases
are routed to a human, which is the behaviour being measured.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

from assay.trust.reliability import wilson_interval

MIN_SEGMENT = 30  # below this a segment is "inconclusive", not a point estimate (PRD 6.3)


def auroc(wrong: np.ndarray, err_score: np.ndarray) -> float:
    wrong = np.asarray(wrong).astype(int)
    if wrong.min() == wrong.max():
        return float("nan")  # undefined without both classes
    return float(roc_auc_score(wrong, err_score))


def pr_auc(wrong: np.ndarray, err_score: np.ndarray) -> float:
    wrong = np.asarray(wrong).astype(int)
    return float("nan") if wrong.sum() == 0 else float(average_precision_score(wrong, err_score))


def rate(mask_num: np.ndarray, mask_den: np.ndarray) -> dict:
    """P(num | den) with a Wilson 95% interval. n is the evidence count."""
    n = int(np.sum(mask_den))
    k = int(np.sum(np.asarray(mask_num) & np.asarray(mask_den)))
    if n == 0:
        return {"value": None, "lo": None, "hi": None, "n": 0, "k": 0}
    lo, hi = wilson_interval(k, n)
    return {"value": k / n, "lo": lo, "hi": hi, "n": n, "k": k}


def bootstrap_ci(fn: Callable[..., float], *arrays: np.ndarray, n_boot: int = 400, seed: int = 0,
                 alpha: float = 0.05) -> tuple[float, float]:
    """Percentile bootstrap over cases. NaN resamples (one class missing) are skipped."""
    rng = np.random.default_rng(seed)
    n = len(arrays[0])
    vals = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        v = fn(*[a[idx] for a in arrays])
        if not np.isnan(v):
            vals.append(v)
    if len(vals) < max(20, n_boot // 4):
        return float("nan"), float("nan")
    return float(np.quantile(vals, alpha / 2)), float(np.quantile(vals, 1 - alpha / 2))


def paired_auroc_diff_ci(wrong: np.ndarray, score_a: np.ndarray, score_b: np.ndarray,
                         n_boot: int = 400, seed: int = 0) -> dict:
    """AUROC(a) - AUROC(b) on the SAME resampled cases, so the interval reflects the comparison."""
    def diff(w, a, b):
        x, y = auroc(w, a), auroc(w, b)
        return x - y
    d = diff(wrong, score_a, score_b)
    lo, hi = bootstrap_ci(diff, wrong, score_a, score_b, n_boot=n_boot, seed=seed)
    return {"diff": d, "lo": lo, "hi": hi, "excludes_zero_above": bool(lo > 0)}


def auroc_with_ci(wrong, err_score, n_boot: int = 400, seed: int = 0) -> dict:
    lo, hi = bootstrap_ci(auroc, wrong, err_score, n_boot=n_boot, seed=seed)
    return {"value": auroc(wrong, err_score), "lo": lo, "hi": hi}


def expected_calibration_error(correct: np.ndarray, ti: np.ndarray, bins: int = 10) -> float:
    """PRD 6.2 ECE for Calibrated mode: weighted |observed correctness - TI/100| over equal-count
    bins. Not meaningful for Provisional mode, which makes no probability claim."""
    correct, p = np.asarray(correct, float), np.asarray(ti, float) / 100.0
    order = np.argsort(p, kind="stable")
    total = 0.0
    for chunk in np.array_split(order, min(bins, len(p))):
        if len(chunk):
            total += len(chunk) / len(p) * abs(correct[chunk].mean() - p[chunk].mean())
    return float(total)


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    a, b = np.asarray(a, float), np.asarray(b, float)
    if np.std(a) == 0 or np.std(b) == 0:
        return float("nan")
    ra = np.argsort(np.argsort(a, kind="stable"), kind="stable").astype(float)
    rb = np.argsort(np.argsort(b, kind="stable"), kind="stable").astype(float)
    # average ranks for ties
    for arr, src in ((ra, a), (rb, b)):
        for v in np.unique(src):
            m = src == v
            if m.sum() > 1:
                arr[m] = arr[m].mean()
    return float(np.corrcoef(ra, rb)[0, 1])


def coverage_curve(wrong: np.ndarray, ti_or_none: np.ndarray, points: int = 10) -> list[dict]:
    """Error rate among the cases kept when the lowest-trust cases are abstained to a human.
    Cases without a score are the first to be abstained (PRD 6.2 coverage and abstention)."""
    wrong = np.asarray(wrong, bool)
    key = np.where(np.isnan(ti_or_none), -1.0, ti_or_none)
    order = np.argsort(-key, kind="stable")  # most trusted first
    out = []
    for frac in np.linspace(1.0 / points, 1.0, points):
        kept = order[: max(1, round(frac * len(wrong)))]
        out.append({"coverage": float(len(kept) / len(wrong)),
                    "error_rate": float(wrong[kept].mean()), "errors_kept": int(wrong[kept].sum())})
    return out
