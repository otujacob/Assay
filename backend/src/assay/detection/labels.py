"""Labels from matured verified outcomes only (PRD 3.3).

A transaction is labelled only when its label horizon has passed, meaning it has had time to
accumulate its outcome. Without this, recent transactions would carry fraud labels (which arrive
early) but not legitimate labels (which arrive only when the dispute window closes), biasing the
data toward fraud.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any

from assay.features.compute import parse_time

FRAUD_TYPES = {"confirmed_fraud", "chargeback"}
LEGIT_TYPES = {"confirmed_legitimate", "dispute_closed"}


def labels_as_of(txns: list[dict], outcomes: list[dict[str, Any]], as_of: datetime,
                 horizon_days: float) -> dict[str, int]:
    """txn_id -> 1 (verified fraud) or 0 (verified legitimate), for matured, horizon-passed cases."""
    horizon = timedelta(days=horizon_days)
    outcome_by_txn: dict[str, int] = {}
    for o in outcomes:
        m = o.get("matured_at")
        if m is None or parse_time(m) > as_of:
            continue
        if o["outcome_type"] in FRAUD_TYPES:
            outcome_by_txn[o["txn_id"]] = 1  # fraud confirmation wins over any legit record
        elif o["outcome_type"] in LEGIT_TYPES:
            outcome_by_txn.setdefault(o["txn_id"], 0)
    return {t["txn_id"]: outcome_by_txn[t["txn_id"]] for t in txns
            if t["txn_id"] in outcome_by_txn and parse_time(t["event_time"]) + horizon <= as_of}
