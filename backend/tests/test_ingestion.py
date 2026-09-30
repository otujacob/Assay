from datetime import UTC, datetime, timedelta

import pytest

from assay.ingestion import IngestionConfig, IngestionService, InMemoryRepository
from assay.ingestion.pg_repo import PostgresRepository
from assay.synthetic import GeneratorConfig, generate, to_wire

T0 = datetime(2025, 6, 1, 12, 0, tzinfo=UTC)


def txn(**over):
    base = {"event_id": "e1", "txn_id": "t1", "event_time": "2025-06-01T10:00:00Z",
            "amount": "12.50", "currency": "GBP", "channel": "app", "customer_pid": "c1",
            "account_pid": "a1", "schema_version": "txn-1"}
    return {**base, **over}


def outcome(**over):
    base = {"event_id": "o1", "txn_id": "t1", "outcome_type": "confirmed_fraud",
            "source": "chargeback", "event_time": "2025-06-10T10:00:00Z", "schema_version": "out-1"}
    return {**base, **over}


@pytest.fixture(params=["memory", "postgres"])
def svc(request):
    """The same behaviour is required of both repositories."""
    if request.param == "memory":
        repo = InMemoryRepository()
    else:
        repo = PostgresRepository(request.getfixturevalue("pg_conn"), search_path=None)
    now = {"t": T0}
    s = IngestionService(repo, IngestionConfig(clock=lambda: now["t"]))
    s.now = now
    return s


def test_valid_record_stored_with_metadata(svc):
    r = svc.ingest_transaction("A", txn(), source_id="bank-api")
    assert r.status == "accepted"
    row = svc.repo.get_transaction("A", "t1")
    assert row["tenant_id"] == "A" and row["recorded_at"] == T0
    ie = svc.repo.get_ingestion_event("A", "e1")
    assert ie["source_id"] == "bank-api" and len(ie["event_payload_hash"]) == 64


def test_invalid_rejected_counted_and_no_values_echoed(svc):
    r = svc.ingest_transaction("A", txn(amount="-5", customer_pid="SECRET-NAME", extra_field=1))
    assert r.status == "rejected"
    assert any(x.startswith("amount:") for x in r.reasons)
    assert any(x.startswith("extra_field:") for x in r.reasons)
    assert "SECRET-NAME" not in repr(r) and "SECRET-NAME" not in repr(svc.repo.rows("A", "ingestion_events"))
    assert svc.repo.get_transaction("A", "t1") is None
    assert svc.repo.count_by_validation("A") == {"rejected": 1}


@pytest.mark.parametrize("over", [
    {"schema_version": "txn-9"}, {"currency": "gbp"}, {"event_time": "2025-06-01T10:00:00"},
    {"channel": "carrier_pigeon"}, {"country": "GBR"}, {"txn_id": ""},
])
def test_schema_and_field_rules(svc, over):
    assert svc.ingest_transaction("A", txn(**over)).status == "rejected"


def test_identical_resend_creates_no_duplicate(svc):
    assert svc.ingest_transaction("A", txn()).status == "accepted"
    assert svc.ingest_transaction("A", txn()).status == "duplicate"
    assert len(svc.repo.rows("A", "transactions")) == 1
    assert len(svc.repo.rows("A", "ingestion_events")) == 1


def test_conflicting_resend_and_reused_txn_id_rejected(svc):
    svc.ingest_transaction("A", txn())
    assert svc.ingest_transaction("A", txn(amount="99")).reasons == ("event_id:conflicting_payload",)
    assert svc.ingest_transaction("A", txn(event_id="e2")).reasons == ("txn_id:already_exists",)
    assert len(svc.repo.rows("A", "transactions")) == 1
    # The rejected conflict must not break idempotency of the original.
    assert svc.ingest_transaction("A", txn()).status == "duplicate"


def test_resend_racing_the_original_commit_is_a_duplicate_not_a_conflict(svc):
    """Regression: the event check and the transaction check are separate reads. If the original
    request commits between them, the resend sees no event but an existing transaction and used to
    be rejected (422) instead of answered as a duplicate. Found by the concurrent-POST test."""
    assert svc.ingest_transaction("A", txn()).status == "accepted"
    repo, real = svc.repo, svc.repo.get_ingestion_event
    calls = {"n": 0}

    def blind_first_time(tenant_id, event_id):
        calls["n"] += 1
        return None if calls["n"] == 1 else real(tenant_id, event_id)

    repo.get_ingestion_event = blind_first_time
    try:
        assert svc.ingest_transaction("A", txn()).status == "duplicate"
        # A different payload under the same txn_id is still a real conflict.
        assert svc.ingest_transaction("A", txn(event_id="e2")).reasons == ("txn_id:already_exists",)
    finally:
        repo.get_ingestion_event = real
    assert len(repo.rows("A", "transactions")) == 1


def test_outcome_for_unknown_txn_is_quarantined_not_dropped(svc):
    r = svc.ingest_outcome("A", outcome(txn_id="ghost"))
    assert r.status == "quarantined"
    assert len(svc.repo.rows("A", "quarantine")) == 1 and not svc.repo.rows("A", "outcomes")


def test_outcome_maturity_from_window(svc):
    svc.ingest_transaction("A", txn())
    svc.ingest_outcome("A", outcome())
    svc.ingest_outcome("A", outcome(event_id="o2", outcome_type="confirmed_legitimate",
                                    event_time="2025-06-20T10:00:00Z"))  # inside 120d window
    svc.ingest_outcome("A", outcome(event_id="o3", outcome_type="confirmed_legitimate",
                                    event_time="2026-01-01T10:00:00Z"))  # after window
    states = [r["maturity_state"] for r in svc.repo.rows("A", "outcomes")]
    assert states == ["matured", "pending", "matured"]


def test_late_event_does_not_change_past_view(svc):
    svc.ingest_transaction("A", txn())
    view_before = svc.repo.transactions_recorded_before("A", T0)
    svc.now["t"] = T0 + timedelta(days=3)
    # a late-arriving transaction with an old event_time
    svc.ingest_transaction("A", txn(event_id="e2", txn_id="t2", event_time="2025-05-01T00:00:00Z"))
    assert svc.repo.transactions_recorded_before("A", T0) == view_before
    t2 = svc.repo.get_transaction("A", "t2")
    assert t2["event_time"] != t2["recorded_at"]


def test_tenants_are_isolated(svc):
    svc.ingest_transaction("A", txn())
    assert svc.repo.get_transaction("B", "t1") is None
    assert svc.repo.rows("B", "transactions") == []
    assert svc.ingest_transaction("B", txn()).status == "accepted"  # same ids, other tenant
    assert svc.ingest_outcome("B", outcome()).status == "accepted"
    assert svc.repo.rows("A", "outcomes") == []


def test_every_write_is_audited_and_chains_verify(svc):
    svc.ingest_transaction("A", txn())
    svc.ingest_transaction("A", txn(amount="-1", event_id="bad"))
    svc.ingest_outcome("A", outcome())
    assert len(svc.repo.rows("A", "audit_log")) == 3
    for table in ("ingestion_events", "transactions", "outcomes", "audit_log"):
        assert svc.repo.verify("A", table) is None


def test_tampering_is_detected(svc):
    if not isinstance(svc.repo, InMemoryRepository):
        pytest.skip('Postgres tampering is covered in test_db_live.py')
    for i in range(3):
        svc.ingest_transaction("A", txn(event_id=f"e{i}", txn_id=f"t{i}"))
    svc.repo._rows[("A", "transactions")][1]["amount"] = "1.00"  # would not change hashes...
    assert svc.repo.verify("A", "transactions") is None  # hash covers payload_hash, not live body
    svc.repo._rows[("A", "transactions")][1]["payload_hash"] = "f" * 64
    assert svc.repo.verify("A", "transactions") == 1
    svc.repo._rows[("A", "audit_log")].pop(0)
    assert svc.repo.verify("A", "audit_log") == 0


def test_ingests_synthetic_dataset_end_to_end(svc):
    ds = generate(GeneratorConfig(n_days=4, txns_per_day=40, n_customers=100, novel_start_day=None))
    s = svc
    statuses = [s.ingest_transaction("A", to_wire(t)).status for t in ds.transactions]
    assert set(statuses) == {"accepted"}
    out = [s.ingest_outcome("A", to_wire(o)).status for o in ds.outcomes]
    assert set(out) == {"accepted"}
    assert len(s.repo.rows("A", "transactions")) == len(ds.transactions)
    assert s.repo.verify("A", "audit_log") is None
