from datetime import UTC, datetime, timedelta

import numpy as np
import pytest

from assay.synthetic import NOVEL_FRAUD_TYPE, GeneratorConfig, generate, write_jsonl

SMALL = {"n_days": 100, "txns_per_day": 150, "n_customers": 500}


def ts(s):
    return datetime.fromisoformat(s)


@pytest.fixture(scope="module")
def ds():
    return generate(GeneratorConfig(**SMALL))


def test_deterministic():
    a = generate(GeneratorConfig(**SMALL, seed=1))
    b = generate(GeneratorConfig(**SMALL, seed=1))
    c = generate(GeneratorConfig(**SMALL, seed=2))
    assert a.transactions == b.transactions and a.outcomes == b.outcomes
    assert a.transactions != c.transactions


def test_ids_unique_and_schema_fields(ds):
    ids = [t["txn_id"] for t in ds.transactions]
    assert len(ids) == len(set(ids))
    eids = [t["event_id"] for t in ds.transactions] + [o["event_id"] for o in ds.outcomes]
    assert len(eids) == len(set(eids))
    for k in ("event_id", "txn_id", "event_time", "amount", "currency", "channel",
              "customer_pid", "account_pid", "schema_version"):
        assert all(t[k] is not None for t in ds.transactions)


def test_fraud_rate_near_config(ds):
    rate = np.mean([v["is_fraud"] for v in ds.truth.values()])
    assert 0.01 < rate < 0.04


def test_novel_type_absent_before_start_and_present_after(ds):
    cut = ts(ds.transactions[0]["event_time"]).replace(hour=0, minute=0, second=0) \
        + timedelta(days=ds.config.novel_start_day)
    novel = [t for t in ds.transactions if ds.truth[t["txn_id"]]["fraud_type"] == NOVEL_FRAUD_TYPE]
    assert novel and all(ts(t["event_time"]) >= cut for t in novel)
    assert all(t["channel"] == "open_banking" for t in novel)
    assert not any(t["channel"] == "open_banking" for t in ds.transactions
                   if ds.truth[t["txn_id"]]["fraud_type"] != NOVEL_FRAUD_TYPE)


def test_outcomes_after_transactions_and_matured(ds):
    by = {t["txn_id"]: ts(t["event_time"]) for t in ds.transactions}
    assert all(ts(o["event_time"]) > by[o["txn_id"]] for o in ds.outcomes)
    early = min(by.values()) + timedelta(days=10)
    assert len(ds.matured_outcomes(early)) < len(ds.outcomes)
    late = datetime(2030, 1, 1, tzinfo=UTC)
    assert len(ds.matured_outcomes(late)) == len(ds.outcomes)


def test_outcome_types_match_truth(ds):
    for o in ds.outcomes:
        fraud = ds.truth[o["txn_id"]]["is_fraud"]
        assert (o["outcome_type"] == "confirmed_fraud") == fraud


def test_drift_shifts_amounts():
    base = generate(GeneratorConfig(**SMALL, seed=3))
    drifted = generate(GeneratorConfig(**SMALL, seed=3, drift_start_day=50, amount_scale=2.0))
    def mean_late(d):
        return np.mean([t["amount"] for t in d.transactions
                        if ts(t["event_time"]) > ts(d.transactions[0]["event_time"]) + timedelta(days=60)])
    assert mean_late(drifted) > 1.5 * mean_late(base)


def test_degraded_data_adds_nulls():
    d = generate(GeneratorConfig(**SMALL, null_rate=0.3))
    frac = np.mean([t["ip_hash"] is None for t in d.transactions])
    assert 0.2 < frac < 0.4


def test_bad_analysts_are_noisier():
    def acc(d, bad):
        ok = []
        for a in d.analyst_actions:
            if (d.analyst_quality[a["analyst_pid"]] < 0.6) == bad:
                ok.append((a["action"] == "block") == d.truth[a["txn_id"]]["is_fraud"])
        return np.mean(ok)
    d = generate(GeneratorConfig(**SMALL, bad_analyst_share=0.25, review_share=0.5))
    assert acc(d, bad=False) > acc(d, bad=True) + 0.2


def test_write_jsonl_keeps_truth_separate(tmp_path, ds):
    write_jsonl(ds, tmp_path)
    assert (tmp_path / "truth.json").exists()
    assert "is_fraud" not in (tmp_path / "transactions.jsonl").read_text()
