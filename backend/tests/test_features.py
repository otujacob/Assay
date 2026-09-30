import random
from datetime import UTC, datetime, timedelta

import numpy as np
import pytest

from assay.features import (
    FEATURE_SET_VERSION,
    REGISTRY,
    assert_no_leakage,
    build_table,
    compute_features,
    definition_versions,
    feature_names,
    find_leaks,
    vector,
)
from assay.synthetic import GeneratorConfig, generate, to_wire


@pytest.fixture(scope="module")
def txns():
    ds = generate(GeneratorConfig(n_days=30, txns_per_day=60, n_customers=80, seed=11,
                                  novel_start_day=None))
    return [to_wire(t) for t in ds.transactions]


def mk(txn_id, t, cust="c1", amount=10.0, **kw):
    return {"txn_id": txn_id, "event_time": t.isoformat(), "amount": amount, "customer_pid": cust,
            "account_pid": cust, "channel": "app", "device_hash": "d1", "ip_hash": "i", "country": "GB",
            "merchant_id": "m", "beneficiary_pid": None, **kw}


T = datetime(2025, 1, 1, 12, tzinfo=UTC)


def test_registry_is_complete_and_versioned():
    assert len(feature_names()) == len(REGISTRY) == len(set(feature_names()))
    assert all(v >= 1 for v in definition_versions().values())


def test_vector_records_versions_and_as_of():
    v = vector(mk("t", T), [], as_of=T)
    assert v.feature_set_version == FEATURE_SET_VERSION and v.as_of_time == T
    assert v.definition_versions == definition_versions() and set(v.values) == set(feature_names())


def test_velocity_and_history_features_by_hand():
    hist = [mk("a", T - timedelta(minutes=30)), mk("b", T - timedelta(minutes=10)),
            mk("c", T - timedelta(hours=5)), mk("other", T - timedelta(minutes=5), cust="c2")]
    f = compute_features(mk("now", T, amount=40.0), hist)
    assert f["velocity_1h"] == 2 and f["velocity_24h"] == 3 and f["velocity_30d"] == 3
    assert f["amount_ratio_baseline"] == pytest.approx(4.0)  # 40 / mean(10,10,10)
    assert f["new_device"] == 0 and f["country_mismatch"] == 0


def test_new_device_beneficiary_country_and_sharing():
    hist = [mk("a", T - timedelta(days=1), beneficiary_pid="b1"),
            mk("x", T - timedelta(days=1), cust="c2", beneficiary_pid="b9"),
            mk("y", T - timedelta(days=2), cust="c3", beneficiary_pid="b9")]
    f = compute_features(mk("n", T, beneficiary_pid="b9", device_hash="d-new", country="NG"), hist)
    assert f["new_beneficiary"] == 1 and f["new_device"] == 1 and f["country_mismatch"] == 1
    assert f["beneficiary_shared_customers"] == 2
    assert compute_features(mk("n", T, device_hash="d-new"), [])["new_device"] == 0  # no history


def test_future_and_late_arriving_data_is_ignored():
    future = mk("f", T + timedelta(minutes=1))
    same_time = mk("s", T)
    late = mk("late", T - timedelta(hours=1), recorded_at=(T + timedelta(days=2)).isoformat())
    base = compute_features(mk("now", T), [])
    assert compute_features(mk("now", T), [future, same_time]) == base
    # recorded after the decision: invisible when as_of is the decision time, visible otherwise
    assert compute_features(mk("now", T), [late], as_of=T) == base
    assert compute_features(mk("now", T), [late])["velocity_30d"] == 1


def test_streaming_builder_equals_reference(txns):
    table = build_table(txns)
    by_id = {t["txn_id"]: t for t in txns}
    rng = random.Random(3)
    for i in rng.sample(range(len(table)), 150):
        ref = compute_features(by_id[table.txn_ids[i]], txns)
        assert [ref[n] for n in table.names] == pytest.approx(list(table.X[i]), abs=1e-9), i


def test_builder_sorts_by_time_and_same_second_rows_do_not_see_each_other():
    a, b = mk("a", T), mk("b", T)
    tbl = build_table([b, a])
    assert tbl.txn_ids == ["a", "b"] and np.all(tbl.X[:, tbl.names.index("velocity_30d")] == 0)


def test_honest_features_pass_leak_check(txns):
    assert find_leaks(txns, sample_size=40) == []
    assert_no_leakage(txns, sample_size=20)


def test_leaky_feature_is_caught(txns):
    def leaky(txn, history):
        f = compute_features(txn, history)
        t = txn["event_time"]
        f["future_count"] = float(sum(1 for h in history if h["customer_pid"] == txn["customer_pid"]
                                      and h["event_time"] > t))  # peeks at later data
        return f
    leaks = find_leaks(txns, leaky, sample_size=80)
    assert leaks and {f for _, f in leaks} == {"future_count"}
    with pytest.raises(AssertionError, match="future_count"):
        assert_no_leakage(txns, leaky, sample_size=80)
