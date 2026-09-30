"""Drift: per-feature statistics and case-level Drift Exposure (PRD 9.1, 9.3, FR-17).

d_j is the two-sample Kolmogorov-Smirnov statistic between the reference window and a recent
window, scaled to [0, 1] after subtracting the 5% significance threshold, so sampling noise alone
gives 0. Case-level exposure weights each feature's drift by its share of the case's attribution:
D = sum_j phi_j * d_j, with phi_j = |a_j| / sum_k |a_k|. Drift stability is 1 - D.
"""

from __future__ import annotations

import numpy as np

KS_C_ALPHA_5PCT = 1.358


def ks_statistic(a: np.ndarray, b: np.ndarray) -> float:
    a, b = np.sort(np.asarray(a, float)), np.sort(np.asarray(b, float))
    allv = np.concatenate([a, b])
    cdf_a = np.searchsorted(a, allv, side="right") / len(a)
    cdf_b = np.searchsorted(b, allv, side="right") / len(b)
    return float(np.max(np.abs(cdf_a - cdf_b)))


def drift_vector(reference: np.ndarray, window: np.ndarray) -> np.ndarray:
    """Scaled drift statistic per feature, in [0, 1]. 0 means no significant shift."""
    n, m = len(reference), len(window)
    if n == 0 or m == 0:
        raise ValueError("need rows in both windows")
    critical = KS_C_ALPHA_5PCT * np.sqrt((n + m) / (n * m))
    out = np.empty(reference.shape[1])
    for j in range(reference.shape[1]):
        ks = ks_statistic(reference[:, j], window[:, j])
        out[j] = max(0.0, ks - critical) / (1.0 - critical) if critical < 1 else 0.0
    return np.clip(out, 0.0, 1.0)


def drift_exposure(attributions: np.ndarray, d: np.ndarray) -> np.ndarray:
    """Attribution-weighted drift per case. `d` is a vector (n_features,) or (n_cases, n_features)."""
    w = np.abs(attributions)
    total = w.sum(axis=1, keepdims=True)
    phi = np.divide(w, total, out=np.zeros_like(w), where=total > 0)
    d = np.broadcast_to(d, phi.shape)
    return np.clip((phi * d).sum(axis=1), 0.0, 1.0)
