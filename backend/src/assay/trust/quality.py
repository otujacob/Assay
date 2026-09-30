"""Case-level Data Quality Q in [0, 1] (PRD 9.4, FR-17): the MINIMUM over five dimensions, because
one broken dimension can invalidate a prediction."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime

from assay.features.compute import parse_time


@dataclass(frozen=True)
class QualityConfig:
    # Importance of each optional field to completeness (parameters, not validated values).
    field_weights: dict[str, float] = field(default_factory=lambda: {
        "device_hash": 0.3, "ip_hash": 0.2, "country": 0.3, "merchant_id": 0.2})
    max_amount: float = 1_000_000.0
    allowed_ingestion_lag_s: float = 3600.0  # freshness budget
    allowed_channels: frozenset[str] = frozenset(
        {"card_present", "card_not_present", "app", "web", "open_banking"})


def data_quality(txn: dict, *, recorded_at: datetime | None = None, source_health: float = 1.0,
                 cfg: QualityConfig | None = None) -> dict[str, float]:
    """Return each dimension and the minimum under key "q"."""
    cfg = cfg or QualityConfig()
    total = sum(cfg.field_weights.values())
    completeness = 1.0 - sum(w for k, w in cfg.field_weights.items() if txn.get(k) is None) / total

    try:
        amount_ok = 0 < float(txn["amount"]) <= cfg.max_amount
    except (KeyError, TypeError, ValueError):
        amount_ok = False
    validity = 1.0 if amount_ok and len(str(txn.get("currency", ""))) == 3 else 0.0

    freshness = 1.0
    consistency = 1.0
    if recorded_at is not None:
        lag = (recorded_at - parse_time(txn["event_time"])).total_seconds()
        if lag < 0:
            consistency = 0.0  # recorded before it happened
        else:
            freshness = max(0.0, 1.0 - lag / cfg.allowed_ingestion_lag_s)
    if txn.get("channel") not in cfg.allowed_channels:
        consistency = 0.0

    dims = {"completeness": completeness, "validity": validity, "freshness": freshness,
            "consistency": consistency, "source_health": min(1.0, max(0.0, source_health))}
    dims["q"] = min(dims.values())
    return dims
