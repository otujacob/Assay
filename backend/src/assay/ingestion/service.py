"""Ingestion service (FR-01 to FR-04).

Rules:
- valid record: stored with tenant, source, ingestion time, payload hash (FR-01)
- invalid record: rejected with field-level reasons, counted; the raw payload is not
  stored, because it may hold personal data (PRD 17, 27)
- identical resend (same event_id, same payload): no duplicate (FR-01)
- same event_id with a different payload, or a new event for an existing txn_id: rejected
- outcome for an unknown transaction: quarantined, not dropped (FR-03)
- event_time and recorded_at both kept (FR-04)
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import Any

from pydantic import ValidationError

from assay.lineage import payload_hash

from .repo import DuplicateError, InMemoryRepository
from .schema import DEFAULT_CHANNELS, OutcomeEvent, TransactionEvent

CONFIRMED_NOW = frozenset({"confirmed_fraud", "chargeback"})


@dataclass(frozen=True)
class IngestResult:
    status: str  # accepted | duplicate | rejected | quarantined
    event_id: str | None
    reasons: tuple[str, ...] = ()
    ingestion_event_id: str | None = None


@dataclass
class IngestionConfig:
    allowed_channels: frozenset[str] = DEFAULT_CHANNELS
    dispute_window_days: int = 120  # OPD-2 working default, per tenant/product
    clock: Callable[[], datetime] = field(default=lambda: datetime.now(UTC))


def _reasons(err: ValidationError) -> list[str]:
    """Field path plus error type only. Never echo input values (may be personal data)."""
    return sorted({f"{'.'.join(str(p) for p in e['loc']) or '_'}:{e['type']}" for e in err.errors()})


class IngestionService:
    def __init__(self, repo: InMemoryRepository, config: IngestionConfig | None = None):
        self.repo = repo
        self.cfg = config or IngestionConfig()

    def _record(self, tenant_id, source_id, actor, kind, raw: dict[str, Any], status: str,
                reasons: list[str], event_time=None, schema_version=None):
        now = self.cfg.clock()
        eid = str(raw.get("event_id") or "") if isinstance(raw, dict) else ""
        ie = self.repo.append_ingestion_event(
            tenant_id, event_id=eid, kind=kind, source_id=source_id, event_time=event_time,
            recorded_at=now, event_payload_hash=payload_hash(raw), schema_version=schema_version,
            validation_result=status, reasons=reasons, created_by=actor)
        self.repo.append_audit(tenant_id, actor, f"ingest_{kind}", eid or "-", status, now)
        return ie, now

    # -- transactions -----------------------------------------------------------
    def ingest_transaction(self, tenant_id: str, raw: dict[str, Any], *, source_id: str = "api",
                           actor: str = "system", _retry: bool = False) -> IngestResult:
        try:
            ev = TransactionEvent.model_validate(raw)
            if ev.channel not in self.cfg.allowed_channels:
                raise ValueError("channel not allowed")
        except ValidationError as e:
            return self._reject(tenant_id, source_id, actor, "transaction", raw, _reasons(e))
        except ValueError:
            return self._reject(tenant_id, source_id, actor, "transaction", raw, ["channel:not_allowed"])

        ph = payload_hash(raw)
        prior = self.repo.get_ingestion_event(tenant_id, ev.event_id)
        if prior is not None and prior["validation_result"] == "accepted":
            if prior["event_payload_hash"] == ph:
                return IngestResult("duplicate", ev.event_id, (), prior["id"])
            return self._reject(tenant_id, source_id, actor, "transaction", raw,
                                ["event_id:conflicting_payload"])
        if self.repo.get_transaction(tenant_id, ev.txn_id) is not None:
            # The event lookup above is a separate read, so the original writer may have committed
            # between the two reads. It commits event and transaction together, so the event is
            # visible now: re-read before calling this a conflict.
            prior = self.repo.get_ingestion_event(tenant_id, ev.event_id)
            if prior is not None and prior["validation_result"] == "accepted" \
                    and prior["event_payload_hash"] == ph:
                return IngestResult("duplicate", ev.event_id, (), prior["id"])
            return self._reject(tenant_id, source_id, actor, "transaction", raw,
                                ["txn_id:already_exists"])

        try:
            with self.repo.atomic(tenant_id):  # event record, transaction and audit: all or nothing
                ie, now = self._record(tenant_id, source_id, actor, "transaction", raw, "accepted",
                                       [], ev.event_time, ev.schema_version)
                body = ev.model_dump(mode="json")
                body.update(ingestion_event_id=ie["id"], recorded_at=now)
                self.repo.append_transaction(tenant_id, body, actor)
        except DuplicateError:
            if _retry:  # a concurrent writer won; re-read and answer from what is stored
                raise
            return self.ingest_transaction(tenant_id, raw, source_id=source_id, actor=actor,
                                           _retry=True)
        return IngestResult("accepted", ev.event_id, (), ie["id"])

    # -- outcomes ---------------------------------------------------------------
    def ingest_outcome(self, tenant_id: str, raw: dict[str, Any], *, source_id: str = "api",
                       actor: str = "system", _retry: bool = False) -> IngestResult:
        try:
            ev = OutcomeEvent.model_validate(raw)
        except ValidationError as e:
            return self._reject(tenant_id, source_id, actor, "outcome", raw, _reasons(e))

        ph = payload_hash(raw)
        prior = self.repo.get_ingestion_event(tenant_id, ev.event_id)
        if prior is not None and prior["validation_result"] in ("accepted", "quarantined"):
            if prior["event_payload_hash"] == ph:
                return IngestResult("duplicate", ev.event_id, (), prior["id"])
            return self._reject(tenant_id, source_id, actor, "outcome", raw,
                                ["event_id:conflicting_payload"])

        txn = self.repo.get_transaction(tenant_id, ev.txn_id)
        body = ev.model_dump(mode="json")
        try:
            with self.repo.atomic(tenant_id):
                if txn is None:
                    ie, now = self._record(tenant_id, source_id, actor, "outcome", raw,
                                           "quarantined", ["txn_id:unknown_transaction"],
                                           ev.event_time, ev.schema_version)
                    self.repo.append_quarantine(
                        tenant_id, {**body, "reason": "unknown_transaction", "recorded_at": now,
                                    "ingestion_event_id": ie["id"]}, actor)
                    return IngestResult("quarantined", ev.event_id,
                                        ("txn_id:unknown_transaction",), ie["id"])

                txn_time = txn["event_time"]
                if isinstance(txn_time, str):
                    txn_time = datetime.fromisoformat(txn_time)
                maturity, matured_at = self._maturity(ev, txn_time)
                ie, now = self._record(tenant_id, source_id, actor, "outcome", raw, "accepted", [],
                                       ev.event_time, ev.schema_version)
                self.repo.append_outcome(tenant_id, {**body, "maturity_state": maturity,
                                                     "matured_at": matured_at, "recorded_at": now,
                                                     "ingestion_event_id": ie["id"]}, actor)
        except DuplicateError:
            if _retry:
                raise
            return self.ingest_outcome(tenant_id, raw, source_id=source_id, actor=actor,
                                       _retry=True)
        return IngestResult("accepted", ev.event_id, (), ie["id"])

    def _maturity(self, ev: OutcomeEvent, txn_time: datetime):
        """PRD 3.3: only matured labels are used for evaluation."""
        if ev.outcome_type in CONFIRMED_NOW:
            return "matured", ev.event_time.isoformat()
        window_end = txn_time + timedelta(days=self.cfg.dispute_window_days)
        if ev.event_time >= window_end:
            return "matured", ev.event_time.isoformat()
        return "pending", None

    def _reject(self, tenant_id, source_id, actor, kind, raw, reasons) -> IngestResult:
        eid = raw.get("event_id") if isinstance(raw, dict) else None
        self._record(tenant_id, source_id, actor, kind, raw if isinstance(raw, dict) else {},
                     "rejected", reasons)
        return IngestResult("rejected", str(eid) if eid else None, tuple(reasons))
