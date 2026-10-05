"""Tenant-scoped storage for ingestion.

Every method takes a tenant_id and can only see that tenant's rows (FR-40). The
in-memory implementation mirrors the append-only, hash-chained Postgres tables in
db/migrations/0001_init.sql. Rows are never updated or deleted.
"""

from __future__ import annotations

import uuid
from collections import defaultdict
from contextlib import contextmanager
from datetime import datetime, timedelta
from typing import Any, ClassVar

from assay.lineage import GENESIS, payload_hash, row_hash, verify_chain

TABLES = ("ingestion_events", "transactions", "outcomes", "quarantine", "audit_log")


def _to_dt(v) -> datetime:
    return v if isinstance(v, datetime) else datetime.fromisoformat(str(v))


class DuplicateError(Exception):
    """A uniqueness rule fired (same event_id, or same txn_id, within a tenant)."""


class InMemoryRepository:
    def __init__(self) -> None:
        self._rows: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
        self._idx: dict[tuple[str, str, str], dict[str, Any]] = {}

    @contextmanager
    def atomic(self, tenant_id: str):
        """No-op here: the in-memory store has no rollback. Postgres wraps one transaction."""
        yield

    # -- append-only primitive -------------------------------------------------
    def _append(self, tenant_id: str, table: str, body: dict[str, Any], key: str | None = None,
                created_by: str = "system") -> dict[str, Any]:
        rows = self._rows[(tenant_id, table)]
        prev = rows[-1]["row_hash"] if rows else GENESIS
        rid = str(uuid.uuid4())
        ph = payload_hash(body)
        row = {"id": rid, "tenant_id": tenant_id, "created_by": created_by, "payload_hash": ph,
               "prev_hash": prev, "row_hash": row_hash(prev, rid, tenant_id, ph), **body}
        rows.append(row)
        if key is not None:
            self._idx[(tenant_id, table, key)] = row
        return dict(row)

    def _get(self, tenant_id: str, table: str, key: str) -> dict[str, Any] | None:
        row = self._idx.get((tenant_id, table, key))
        return dict(row) if row else None

    # -- generic lineage objects (PRD 14.1) ---------------------------------------------
    # Uniqueness rules mirrored from the Postgres schema.
    _UNIQUE: ClassVar[dict[str, tuple[str, ...]]] = {"model_bundles": ("bundle_id",), "trust_assessments": ("prediction_id", "version_no"),
                                                    "feature_vectors": ("txn_id",), "predictions": ("txn_id",),
                                                    "review_cases": ("txn_id",), "policy_versions": ("version",),
                                                    "policy_approvals": ("policy_version_id",),
                                                    "model_candidates": ("candidate_id",),
                                                    "shadow_scores": ("txn_id", "candidate_id")}

    def insert(self, tenant_id: str, table: str, body: dict[str, Any], created_by: str) -> dict:
        cols = self._UNIQUE.get(table)
        if cols and any(all(r.get(c) == body.get(c) for c in cols)
                        for r in self._rows[(tenant_id, table)]):
            raise DuplicateError(f"{table} unique {cols}")
        return self._append(tenant_id, table, body, created_by=created_by)

    def find(self, tenant_id: str, table: str, where: dict[str, Any] | None = None, *,
             newest_first: bool = False, limit: int | None = None) -> list[dict[str, Any]]:
        rows = [dict(r) for r in self._rows[(tenant_id, table)]
                if all(r.get(k) == v for k, v in (where or {}).items())]
        if newest_first:
            rows.reverse()
        return rows[:limit] if limit else rows

    def get_by_id(self, tenant_id: str, table: str, row_id: str) -> dict[str, Any] | None:
        rows = self.find(tenant_id, table, {"id": row_id})
        return rows[0] if rows else None

    def history_for_features(self, tenant_id: str, txn: dict[str, Any], window_days: int = 30):
        """Rows that can influence this transaction's features: the customer's earlier
        transactions, and payments to the same beneficiary in the window (PRD 13.8)."""
        t = _to_dt(txn["event_time"])
        lo = t - timedelta(days=window_days)
        out = []
        for r in self._rows[(tenant_id, "transactions")]:
            rt = _to_dt(r["event_time"])
            if rt >= t or r["txn_id"] == txn["txn_id"]:
                continue
            same_customer = r["customer_pid"] == txn["customer_pid"]
            same_beneficiary = (txn.get("beneficiary_pid") is not None
                                and r.get("beneficiary_pid") == txn["beneficiary_pid"] and rt > lo)
            if same_customer or same_beneficiary:
                out.append(dict(r))
        return out

    # -- ingestion events ------------------------------------------------------
    def get_ingestion_event(self, tenant_id: str, event_id: str):
        return self._get(tenant_id, "ingestion_events", event_id)

    def append_ingestion_event(self, tenant_id: str, *, event_id: str, kind: str, source_id: str,
                               event_time: datetime | None, recorded_at: datetime,
                               event_payload_hash: str, schema_version: str | None,
                               validation_result: str, reasons: list[str], created_by: str):
        return self._append(tenant_id, "ingestion_events", {
            "event_id": event_id, "kind": kind, "source_id": source_id, "event_time": event_time,
            "recorded_at": recorded_at, "event_payload_hash": event_payload_hash,
            "schema_version": schema_version, "validation_result": validation_result,
            "reasons": reasons,
        }, key=event_id if validation_result != "rejected" and event_id else None,
            created_by=created_by)

    # -- transactions / outcomes ----------------------------------------------
    def get_transaction(self, tenant_id: str, txn_id: str):
        return self._get(tenant_id, "transactions", txn_id)

    def append_transaction(self, tenant_id: str, body: dict[str, Any], created_by: str):
        return self._append(tenant_id, "transactions", body, key=body["txn_id"],
                            created_by=created_by)

    def append_outcome(self, tenant_id: str, body: dict[str, Any], created_by: str):
        return self._append(tenant_id, "outcomes", body, created_by=created_by)

    def append_quarantine(self, tenant_id: str, body: dict[str, Any], created_by: str):
        return self._append(tenant_id, "quarantine", body, created_by=created_by)

    def append_audit(self, tenant_id: str, actor: str, action: str, obj: str, result: str,
                     at: datetime):
        return self._append(tenant_id, "audit_log", {"actor": actor, "action": action,
                                                     "object": obj, "result": result, "time": at},
                            created_by=actor)

    # -- reads -----------------------------------------------------------------
    def rows(self, tenant_id: str, table: str) -> list[dict[str, Any]]:
        return [dict(r) for r in self._rows[(tenant_id, table)]]

    def transactions_recorded_before(self, tenant_id: str, as_of: datetime):
        """Point-in-time view by recorded time (FR-04): late events don't alter past views."""
        return [r for r in self.rows(tenant_id, "transactions") if r["recorded_at"] <= as_of]

    def count_by_validation(self, tenant_id: str) -> dict[str, int]:
        out: dict[str, int] = defaultdict(int)
        for r in self._rows[(tenant_id, "ingestion_events")]:
            out[r["validation_result"]] += 1
        return dict(out)

    def verify(self, tenant_id: str, table: str) -> int | None:
        return verify_chain(self._rows[(tenant_id, table)])
