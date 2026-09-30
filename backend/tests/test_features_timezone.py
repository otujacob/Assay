from datetime import UTC, datetime, timedelta, timezone

from assay.features import compute_features


def test_hour_is_utc_whatever_offset_the_timestamp_carries():
    """Regression: Postgres returns timestamps in the session zone (UTC+1 in a London summer)."""
    txn = {"txn_id": "t", "amount": 10, "customer_pid": "c", "channel": "app"}
    utc = datetime(2025, 6, 1, 12, 30, tzinfo=UTC)
    for tz in (UTC, timezone(timedelta(hours=1)), timezone(timedelta(hours=-5)), timezone(timedelta(hours=9))):
        t = {**txn, "event_time": utc.astimezone(tz)}
        assert compute_features(t, [])["hour"] == 12.0, tz
