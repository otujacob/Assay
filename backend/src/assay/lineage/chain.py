"""Tamper-evident hash chain for append-only records (PRD 14.2, FR-32).

row_hash = sha256(prev_hash | row_id | tenant_id | payload_hash). The same
formula is implemented in SQL (db/migrations/0001_init.sql) so the database
and application agree.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Mapping
from typing import Any

GENESIS = "0" * 64


def canonical_json(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=str)


def payload_hash(obj: Any) -> str:
    return hashlib.sha256(canonical_json(obj).encode("utf-8")).hexdigest()


def row_hash(prev_hash: str, row_id: str, tenant_id: str, payload_hash_: str) -> str:
    data = f"{prev_hash}|{row_id}|{tenant_id}|{payload_hash_}"
    return hashlib.sha256(data.encode("utf-8")).hexdigest()


def verify_chain(rows: Iterable[Mapping[str, str]]) -> int | None:
    """Return None if the chain is intact, else the index of the first bad row.

    Rows must be in append order and carry id, tenant_id, payload_hash,
    prev_hash and row_hash.
    """
    prev = GENESIS
    for i, r in enumerate(rows):
        if r["prev_hash"] != prev:
            return i
        if r["row_hash"] != row_hash(prev, r["id"], r["tenant_id"], r["payload_hash"]):
            return i
        prev = r["row_hash"]
    return None
