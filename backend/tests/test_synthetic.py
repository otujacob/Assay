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


# ---- the entity-graph scenario (PRD 13, H6) ---------------------------------------------------------------
GRAPH = {"n_days": 100, "txns_per_day": 120, "n_customers": 500, "graph_scenario": True}


def test_the_default_dataset_is_byte_for_byte_what_it_was_before_the_graph_scenario_existed():
    """Pinned: the scenario draws from its own random stream, so switching it off must change nothing else."""
    import hashlib
    import json

    d = generate(GeneratorConfig(seed=5, n_days=40, txns_per_day=60, n_customers=300, novel_start_day=30))
    h = hashlib.sha256(json.dumps([d.transactions, d.outcomes, d.analyst_actions, d.truth], sort_keys=True,
                                  default=str).encode()).hexdigest()
    assert h == "4144b8b807bc30748192af2b853b104c9f5554de0f2eb57fa61736eca8a633c5"
    assert d.structure == {} and not any("ring" in t for t in d.truth.values())


@pytest.fixture(scope="module")
def graph_ds():
    return generate(GeneratorConfig(**GRAPH, seed=11))


def test_rings_families_and_hubs_are_planted_and_do_not_overlap(graph_ds):
    s = graph_ds.structure
    assert len(s["rings"]) == 5 and len(s["families"]) == 60 and len(s["hubs"]) == 3
    ring_members = {m for ms in s["rings"].values() for m in ms}
    fam_members = {m for ms in s["families"].values() for m in ms}
    assert len(ring_members) >= 25 and not (ring_members & fam_members)
    assert all(5 <= len(ms) <= 8 for ms in s["rings"].values()) and all(2 <= len(ms) <= 3 for ms in s["families"].values())


def test_ring_fraud_goes_through_the_rings_shared_infrastructure_and_is_labelled(graph_ds):
    ring_txns = [t for t in graph_ds.transactions if graph_ds.truth[t["txn_id"]]["fraud_type"] == "ring_attack"]
    assert len(ring_txns) > 50
    for t in ring_txns[:200]:
        r = graph_ds.truth[t["txn_id"]]["ring"]
        assert t["customer_pid"] in graph_ds.structure["rings"][r]
        assert t["device_hash"].startswith(f"d-{r.replace('-', '')}-") and t["ip_hash"].startswith(f"ip-{r.replace('-', '')}-")
    assert any(o["txn_id"] in {t["txn_id"] for t in ring_txns} for o in graph_ds.outcomes if o["outcome_type"] == "confirmed_fraud")


def test_a_ring_transaction_looks_ordinary_on_its_own(graph_ds):
    ring = [t for t in graph_ds.transactions if graph_ds.truth[t["txn_id"]]["fraud_type"] == "ring_attack"]
    ratios = [t["amount"] / t["_ctx"]["baseline_amount"] for t in ring]
    assert np.median(ratios) < 2.5  # not the 3x-8x of account takeover
    assert all(t["country"] == t["_ctx"]["home_country"] for t in ring)


def test_ring_infrastructure_is_shared_across_distinct_customers(graph_ds):
    users: dict[str, set] = {}
    for t in graph_ds.transactions:
        if t["device_hash"] and t["device_hash"].startswith("d-ring"):
            users.setdefault(t["device_hash"], set()).add(t["customer_pid"])
    assert users and all(len(u) >= 3 for u in users.values())


def test_benign_sharing_exists_too_so_a_count_alone_cannot_tell_it_from_a_ring(graph_ds):
    fam = {}
    hub = {}
    for t in graph_ds.transactions:
        if t["device_hash"] and t["device_hash"].startswith("d-family"):
            fam.setdefault(t["device_hash"], set()).add(t["customer_pid"])
        if t["ip_hash"] and t["ip_hash"].startswith("ip-public"):
            hub.setdefault(t["ip_hash"], set()).add(t["customer_pid"])
    assert any(len(u) >= 2 for u in fam.values())            # a family device is used by more than one person
    assert hub and all(len(u) > 50 for u in hub.values())    # a public IP is used by many unrelated customers
    legit = {t["txn_id"] for t in graph_ds.transactions if not graph_ds.truth[t["txn_id"]]["is_fraud"]}
    assert all(any(t["txn_id"] in legit for t in graph_ds.transactions
                   if t["ip_hash"] == h) for h in hub)       # and it is legitimate traffic


def test_the_scenario_is_deterministic_for_a_seed():
    a = generate(GeneratorConfig(**GRAPH, seed=3))
    b = generate(GeneratorConfig(**GRAPH, seed=3))
    assert a.transactions == b.transactions and a.structure == b.structure
