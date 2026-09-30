"""Model Confidence (PRD 5.1, FR-14): 1 - normalised prediction uncertainty.

Uncertainty combines ensemble disagreement (spread across bootstrap members) with the margin from
the operating threshold. Both are unbounded, so each is converted to percentile rank against the
calibration-window reference (PRD 5.2). Equal weighting and the interval width are design
parameters, not validated values.
"""

from __future__ import annotations

import numpy as np

from .reference import pct_rank


def model_confidence(spread: np.ndarray, distance: np.ndarray, spread_ecdf: np.ndarray,
                     distance_ecdf: np.ndarray, *, interval_scale: float = 0.25):
    """Return (conf, lo, hi), each in [0, 1]. Deterministic."""
    u_spread = pct_rank(spread_ecdf, spread)  # high spread = high uncertainty
    margin = pct_rank(distance_ecdf, distance)  # large margin = low uncertainty
    u = 0.5 * u_spread + 0.5 * (1.0 - margin)
    conf = np.clip(1.0 - u, 0.0, 1.0)
    half = interval_scale * u_spread
    return conf, np.clip(conf - half, 0.0, 1.0), np.clip(conf + half, 0.0, 1.0)
