"""Versioned event schemas (PRD 25.3, FR-02). Unknown fields are rejected, so a schema
change without a new version fails validation."""

from __future__ import annotations

from datetime import datetime
from decimal import Decimal
from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, field_validator

SUPPORTED_TXN_SCHEMAS = frozenset({"txn-1"})
SUPPORTED_OUTCOME_SCHEMAS = frozenset({"out-1"})
# Channel values are agreed per institution; this is the starting set.
DEFAULT_CHANNELS = frozenset({"card_present", "card_not_present", "app", "web", "open_banking"})
OUTCOME_TYPES = frozenset({"confirmed_fraud", "confirmed_legitimate", "chargeback", "dispute_closed"})

Id = Annotated[str, StringConstraints(min_length=1, max_length=128, strip_whitespace=True)]


class _Event(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    event_id: Id
    txn_id: Id
    event_time: datetime
    schema_version: Id

    @field_validator("event_time")
    @classmethod
    def _tz_aware(cls, v: datetime) -> datetime:
        if v.tzinfo is None:
            raise ValueError("event_time must be timezone-aware (UTC)")
        return v


class TransactionEvent(_Event):
    amount: Decimal = Field(gt=0, max_digits=18, decimal_places=2)
    currency: Annotated[str, StringConstraints(pattern=r"^[A-Z]{3}$")]
    channel: Id
    customer_pid: Id
    account_pid: Id
    beneficiary_pid: Id | None = None
    merchant_id: Id | None = None
    device_hash: Id | None = None
    ip_hash: Id | None = None
    country: Annotated[str, StringConstraints(pattern=r"^[A-Z]{2}$")] | None = None

    @field_validator("schema_version")
    @classmethod
    def _known_version(cls, v: str) -> str:
        if v not in SUPPORTED_TXN_SCHEMAS:
            raise ValueError("unsupported schema_version")
        return v


class OutcomeEvent(_Event):
    outcome_type: str
    source: Id

    @field_validator("schema_version")
    @classmethod
    def _known_version(cls, v: str) -> str:
        if v not in SUPPORTED_OUTCOME_SCHEMAS:
            raise ValueError("unsupported schema_version")
        return v

    @field_validator("outcome_type")
    @classmethod
    def _known_type(cls, v: str) -> str:
        if v not in OUTCOME_TYPES:
            raise ValueError("unknown outcome_type")
        return v
