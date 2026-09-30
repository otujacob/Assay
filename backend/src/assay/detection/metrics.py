"""Evaluation helpers used by detection and, later, the trust validation harness (PRD 6.2)."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score


def expected_calibration_error(y: np.ndarray, p: np.ndarray, bins: int = 10) -> float:
    """ECE with equal-count bins (equal-width bins collapse when fraud is rare)."""
    y, p = np.asarray(y, float), np.asarray(p, float)
    order = np.argsort(p, kind="stable")
    total = 0.0
    for chunk in np.array_split(order, min(bins, len(p))):
        if len(chunk):
            total += len(chunk) / len(p) * abs(y[chunk].mean() - p[chunk].mean())
    return float(total)


def reliability_curve(y: np.ndarray, p: np.ndarray, bins: int = 10) -> list[dict]:
    y, p = np.asarray(y, float), np.asarray(p, float)
    order = np.argsort(p, kind="stable")
    return [{"n": len(c), "mean_predicted": float(p[c].mean()), "observed_rate": float(y[c].mean())}
            for c in np.array_split(order, min(bins, len(p))) if len(c)]


def pr_auc(y: np.ndarray, score: np.ndarray) -> float:
    return float(average_precision_score(y, score))


def precision_recall_at(y: np.ndarray, p: np.ndarray, threshold: float) -> dict[str, float]:
    y, flagged = np.asarray(y).astype(bool), np.asarray(p) >= threshold
    tp = int((y & flagged).sum())
    return {"precision": tp / flagged.sum() if flagged.any() else float("nan"),
            "recall": tp / y.sum() if y.any() else float("nan"), "flagged": int(flagged.sum())}


def cost_optimal_threshold(y: np.ndarray, p: np.ndarray, cost_fp: float, cost_fn: float) -> float:
    """Pick the threshold minimising expected cost on held-out data (PRD 10.4). The costs are
    the institution's own cost matrix; Assay does not claim a universal optimum."""
    y, p = np.asarray(y).astype(bool), np.asarray(p)
    best_t, best_c = 1.0, float("inf")
    for t in np.unique(p):
        flagged = p >= t
        cost = cost_fp * (flagged & ~y).sum() + cost_fn * (~flagged & y).sum()
        if cost < best_c:
            best_t, best_c = float(t), float(cost)
    return best_t
