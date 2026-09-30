"""Point-in-time feature computation (PRD 3.2, FR-06, FR-07).

Two implementations that must agree:
  compute_features  simple reference: recomputes from a history list. O(history), used in tests,
                    replay and online scoring of one case.
  build_table       fast streaming builder for training sets.

"Prior" always means event_time strictly before the transaction's event_time, and (if an as_of
is given) recorded_at <= as_of, so data that arrived after the decision never leaks in.
"""

from __future__ import annotations

import bisect
import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

import numpy as np

from .registry import (
    CHANNEL_ORDER,
    FEATURE_SET_VERSION,
    definition_versions,
    feature_names,
)

THIRTY_DAYS = 30 * 86400
OPTIONAL =("device_hash", "ip_hash", "country", "merchant_id")


def parse_time(v: Any) -> datetime:
    if isinstance(v, datetime):
        return v
    return datetime.fromisoformat(str(v))


def _epoch(txn: dict) -> float:
    return parse_time(txn["event_time"]).timestamp()


def _mode(counter: Counter) -> str | None:
    if not counter:
        return None
    return min(counter.items(), key=lambda kv: (-kv[1], kv[0]))[0]


@dataclass(frozen=True)
class FeatureVector:
    txn_id: str
    feature_set_version: str
    as_of_time: datetime
    values: dict[str, float]
    definition_versions: dict[str, int]


def _base(txn: dict, amount: float, t: float) -> dict[str, float]:
    ch = txn.get("channel")
    return {
        "amount": amount,
        "log_amount": math.log1p(amount),
        # UTC explicitly: a timestamp may arrive in any offset (Postgres returns the session zone).
        "hour": float(datetime.fromtimestamp(t, tz=UTC).hour),
        "channel_code": float(CHANNEL_ORDER.index(ch)) if ch in CHANNEL_ORDER else -1.0,
        "missing_fields": float(sum(1 for k in OPTIONAL if txn.get(k) is None)),
    }


def compute_features(txn: dict, history: list[dict], as_of: datetime | None = None) -> dict[str, float]:
    """Reference implementation. `history` may contain anything; only valid prior rows are used."""
    t = _epoch(txn)
    amount = float(txn["amount"])
    cust = txn["customer_pid"]

    def visible(h: dict) -> bool:
        if _epoch(h) >= t:
            return False
        ra = h.get("recorded_at")
        return not (as_of is not None and ra is not None and parse_time(ra) > as_of)

    prior = [h for h in history if visible(h)]
    mine = [h for h in prior if h["customer_pid"] == cust]
    f = _base(txn, amount, t)
    times = [_epoch(h) for h in mine]
    f["velocity_1h"] = float(sum(1 for x in times if x > t - 3600))
    f["velocity_24h"] = float(sum(1 for x in times if x > t - 86400))
    f["velocity_30d"] = float(sum(1 for x in times if x > t - THIRTY_DAYS))
    f["amount_ratio_baseline"] = (
        amount / (sum(float(h["amount"]) for h in mine) / len(mine)) if len(mine) >= 3 else 1.0)
    ben = txn.get("beneficiary_pid")
    f["new_beneficiary"] = float(ben is not None and not any(h.get("beneficiary_pid") == ben for h in mine))
    dev = txn.get("device_hash")
    f["new_device"] = float(dev is not None and len(mine) > 0
                            and not any(h.get("device_hash") == dev for h in mine))
    cmode = _mode(Counter(h["country"] for h in mine if h.get("country")))
    c = txn.get("country")
    f["country_mismatch"] = float(c is not None and cmode is not None and c != cmode)
    f["beneficiary_shared_customers"] = float(
        len({h["customer_pid"] for h in prior if ben is not None and h.get("beneficiary_pid") == ben
             and h["customer_pid"] != cust and _epoch(h) > t - THIRTY_DAYS}))
    return {k: f[k] for k in feature_names()}


def vector(txn: dict, history: list[dict], as_of: datetime | None = None) -> FeatureVector:
    """A stored feature vector carries its definition versions and as-of time (FR-06)."""
    as_of_time = as_of or parse_time(txn["event_time"])
    return FeatureVector(txn["txn_id"], FEATURE_SET_VERSION, as_of_time,
                         compute_features(txn, history, as_of), definition_versions())


@dataclass
class FeatureTable:
    names: tuple[str, ...]
    X: np.ndarray
    txn_ids: list[str]
    t: np.ndarray  # event time, epoch seconds

    def __len__(self) -> int:
        return len(self.txn_ids)

    def subset(self, mask: np.ndarray) -> FeatureTable:
        idx = np.flatnonzero(mask)
        return FeatureTable(self.names, self.X[idx], [self.txn_ids[i] for i in idx], self.t[idx])


class _State:
    __slots__ = ("amount_sum", "bens", "countries", "devs", "times")

    def __init__(self) -> None:
        self.times: list[float] = []
        self.amount_sum = 0.0
        self.bens: set[str] = set()
        self.devs: set[str] = set()
        self.countries: Counter = Counter()


def build_table(txns: list[dict]) -> FeatureTable:
    """Streaming builder. Transactions sharing a timestamp do not see each other."""
    rows = sorted(txns, key=lambda x: (_epoch(x), x["txn_id"]))
    states: dict[str, _State] = defaultdict(_State)
    ben_hist: dict[str, tuple[list[float], list[str]]] = defaultdict(lambda: ([], []))
    names = feature_names()
    X = np.empty((len(rows), len(names)), dtype=np.float64)
    ids: list[str] = []
    ts = np.empty(len(rows))

    i = 0
    while i < len(rows):
        j = i
        t = _epoch(rows[i])
        while j < len(rows) and _epoch(rows[j]) == t:
            j += 1
        for k in range(i, j):  # compute all features for the group first...
            txn = rows[k]
            amount = float(txn["amount"])
            s = states[txn["customer_pid"]]
            f = _base(txn, amount, t)
            n = len(s.times)
            f["velocity_1h"] = float(n - bisect.bisect_right(s.times, t - 3600))
            f["velocity_24h"] = float(n - bisect.bisect_right(s.times, t - 86400))
            f["velocity_30d"] = float(n - bisect.bisect_right(s.times, t - THIRTY_DAYS))
            f["amount_ratio_baseline"] = amount / (s.amount_sum / n) if n >= 3 else 1.0
            ben, dev, c = txn.get("beneficiary_pid"), txn.get("device_hash"), txn.get("country")
            f["new_beneficiary"] = float(ben is not None and ben not in s.bens)
            f["new_device"] = float(dev is not None and n > 0 and dev not in s.devs)
            cmode = _mode(s.countries)
            f["country_mismatch"] = float(c is not None and cmode is not None and c != cmode)
            shared = 0
            if ben is not None:
                bt, bc = ben_hist[ben]
                start = bisect.bisect_right(bt, t - THIRTY_DAYS)
                shared = len(set(bc[start:]) - {txn["customer_pid"]})
            f["beneficiary_shared_customers"] = float(shared)
            X[k] = [f[name] for name in names]
            ids.append(txn["txn_id"])
            ts[k] = t
        for k in range(i, j):  # ...then update state
            txn = rows[k]
            s = states[txn["customer_pid"]]
            s.times.append(t)
            s.amount_sum += float(txn["amount"])
            if txn.get("beneficiary_pid") is not None:
                s.bens.add(txn["beneficiary_pid"])
                ben_hist[txn["beneficiary_pid"]][0].append(t)
                ben_hist[txn["beneficiary_pid"]][1].append(txn["customer_pid"])
            if txn.get("device_hash") is not None:
                s.devs.add(txn["device_hash"])
            if txn.get("country"):
                s.countries[txn["country"]] += 1
        i = j
    return FeatureTable(names, X, ids, ts)
