"""Automated point-in-time check (FR-07): a feature that uses later data must fail the build.

For a sample of transactions, compute features from the full history and again from history
truncated at the decision time. An honest feature function gives identical values.
"""

from __future__ import annotations

import random
from collections.abc import Callable

from .compute import _epoch, compute_features

ComputeFn = Callable[[dict, list[dict]], dict[str, float]]


def find_leaks(txns: list[dict], compute: ComputeFn = compute_features, sample_size: int = 50,
               seed: int = 0) -> list[tuple[str, str]]:
    """Return (txn_id, feature) pairs whose value depends on data after the decision time."""
    rng = random.Random(seed)
    sample = rng.sample(txns, min(sample_size, len(txns)))
    leaks: list[tuple[str, str]] = []
    for txn in sample:
        t = _epoch(txn)
        truncated = [h for h in txns if _epoch(h) <= t]
        full, cut = compute(txn, txns), compute(txn, truncated)
        leaks.extend((txn["txn_id"], k) for k in full if full[k] != cut.get(k))
    return leaks


def assert_no_leakage(txns: list[dict], compute: ComputeFn = compute_features, **kw) -> None:
    leaks = find_leaks(txns, compute, **kw)
    if leaks:
        raise AssertionError(f"point-in-time leakage in {sorted({f for _, f in leaks})}: {leaks[:3]}")
