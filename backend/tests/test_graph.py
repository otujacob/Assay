"""Temporal entity graph (PRD 13): confidence, expiry, bitemporal queries, and point-in-time features."""

import math
from datetime import UTC, datetime, timedelta

import numpy as np
import pytest

from assay.graph.features import (
    FLAT_FEATURES,
    GRAPH_FEATURES,
    FraudKnowledge,
    build_graph_table,
    graph_features,
)
from assay.graph.temporal import SECONDS_PER_DAY, GraphConfig, TemporalGraph

T0 = datetime(2025, 1, 1, tzinfo=UTC)
CFG = GraphConfig()


def ts(days: float) -> float:
    return (T0 + timedelta(days=days)).timestamp()


def txn(i, cust, day, *, device=None, ip=None, ben=None, merchant=None, account=None):
    return {"txn_id": f"t{i}", "customer_pid": cust, "account_pid": account or f"acc-{cust}",
            "event_time": (T0 + timedelta(days=day)).isoformat(), "device_hash": device, "ip_hash": ip,
            "beneficiary_pid": ben, "merchant_id": merchant, "amount": 10.0, "channel": "app"}


def outcome(txn_id, kind, matured_day):
    return {"txn_id": txn_id, "outcome_type": kind, "matured_at": (T0 + timedelta(days=matured_day)).isoformat()}


def graph(*txns, cfg=CFG, recorded=None):
    g = TemporalGraph(cfg)
    for k, t in enumerate(txns):
        g.observe(t, None if recorded is None else recorded[k])
    return g


# ---- confidence (PRD 13.3) -----------------------------------------------------------------------------
def test_confidence_follows_the_formula_by_hand():
    g = graph(*[txn(i, "c1", i, device="d1") for i in range(3)])  # three uses on days 0, 1, 2
    t = ts(2)
    expected = (1 - math.exp(-CFG.k * 3)) * 1.0 * (1 / (1 + math.log1p(1)))  # n=3, no age, deg=1 customer
    assert g.confidence("USED_DEVICE", "c1", "d1", t) == pytest.approx(expected)


def test_more_evidence_raises_confidence_and_age_lowers_it():
    g = graph(txn(0, "a", 0, device="d"), txn(1, "b", 0, device="e"), txn(2, "b", 1, device="e"), txn(3, "b", 2, device="e"))
    once, thrice = g.confidence("USED_DEVICE", "a", "d", ts(0)), g.confidence("USED_DEVICE", "b", "e", ts(2))
    assert thrice > once
    assert g.confidence("USED_DEVICE", "a", "d", ts(60)) < g.confidence("USED_DEVICE", "a", "d", ts(1))


def test_a_widely_shared_node_is_a_weak_link():
    quiet = graph(txn(0, "a", 0, ip="ip-home"), txn(1, "b", 0, ip="ip-home"))
    busy = graph(*[txn(i, f"c{i}", 0, ip="ip-cafe") for i in range(60)])
    assert busy.confidence("ACCESSED_FROM", "c0", "ip-cafe", ts(0)) < quiet.confidence("ACCESSED_FROM", "a", "ip-home", ts(0))


def test_confidence_is_zero_for_a_relationship_never_seen_and_is_not_a_fraud_probability():
    g = graph(txn(0, "a", 0, device="d"))
    assert g.confidence("USED_DEVICE", "a", "other", ts(0)) == 0.0
    assert 0.0 <= g.confidence("USED_DEVICE", "a", "d", ts(0)) <= 1.0


# ---- expiry: edges close and are never deleted ---------------------------------------------------------
def test_an_edge_expires_when_confidence_falls_below_the_floor_and_a_new_sighting_revives_it():
    g = graph(txn(0, "a", 0, ip="ip1"))
    assert g.active("ACCESSED_FROM", "a", "ip1", ts(1))
    assert not g.active("ACCESSED_FROM", "a", "ip1", ts(400))  # IPs decay fast
    assert ("ACCESSED_FROM", "a", "ip1") in g._edges  # closed, not deleted
    g.observe(txn(1, "a", 401, ip="ip1"))
    assert g.active("ACCESSED_FROM", "a", "ip1", ts(401))
    assert not g.active("ACCESSED_FROM", "a", "ip1", ts(100))  # history is preserved: day 100 is still expired


def test_ips_decay_faster_than_devices():
    g = graph(txn(0, "a", 0, ip="ip1", device="d1"))
    assert g.confidence("ACCESSED_FROM", "a", "ip1", ts(30)) < g.confidence("USED_DEVICE", "a", "d1", ts(30))


# ---- bitemporal queries (PRD 13.4) ---------------------------------------------------------------------
def test_a_query_sees_only_what_was_true_by_then_and_known_by_then():
    # the transaction happened on day 5 but was only recorded on day 20 (a late event)
    g = graph(txn(0, "a", 5, device="d1"), recorded=[ts(20)])
    assert g.confidence("USED_DEVICE", "a", "d1", ts(10), ts(10)) == 0.0   # at day 10 nobody knew yet
    assert g.confidence("USED_DEVICE", "a", "d1", ts(10), ts(25)) > 0.0    # known later, still true at day 10
    assert g.confidence("USED_DEVICE", "a", "d1", ts(3), ts(25)) == 0.0    # not yet true at day 3


def test_observations_arriving_out_of_order_keep_their_true_valid_time():
    g = graph(txn(0, "a", 10, device="d"), txn(1, "a", 2, device="d"))
    e = g._edges[("USED_DEVICE", "a", "d")]
    assert e.valid == sorted(e.valid) and e.valid[0] == ts(2)


def test_an_analyst_flag_downweights_from_the_time_it_was_recorded_and_deletes_nothing():
    g = graph(txn(0, "a", 0, device="d"), txn(1, "b", 0, device="d"))
    before = g.confidence("USED_DEVICE", "a", "d", ts(1))
    assert g.flag("USED_DEVICE", "a", "d", ts(2))
    assert g.confidence("USED_DEVICE", "a", "d", ts(1), ts(1.5)) == pytest.approx(before)  # not flagged yet then
    assert g.confidence("USED_DEVICE", "a", "d", ts(3), ts(3)) < before * CFG.flag_weight * 1.01
    assert not g.flag("USED_DEVICE", "a", "nope", ts(2))
    assert ("USED_DEVICE", "a", "d") in g._edges


# ---- derived links and communities ---------------------------------------------------------------------
def test_customers_sharing_a_device_are_linked_with_the_weaker_confidence():
    g = graph(txn(0, "a", 0, device="d"), txn(1, "a", 1, device="d"), txn(2, "b", 1, device="d"))
    (lk,) = g.links("a", ts(1))
    assert lk.other == "b" and lk.rel == "SHARES_DEVICE" and lk.via == "d"
    assert lk.confidence == pytest.approx(min(g.confidence("USED_DEVICE", "a", "d", ts(1)),
                                               g.confidence("USED_DEVICE", "b", "d", ts(1))))


def test_a_hub_node_does_not_link_everyone_to_everyone():
    g = graph(*[txn(i, f"c{i}", 0, ip="ip-cafe") for i in range(CFG.max_hub_degree + 5)])
    assert g.links("c0", ts(0)) == []


def test_a_community_is_the_group_reachable_through_strong_links():
    # r0,r1 share d0; r2,r3 share d1; all four share the beneficiary; each is seen three times
    ring = [txn(10 * d + i, f"r{i}", d, device=f"d{i // 2}", ben="b-mule") for d in range(3) for i in range(4)]
    stranger = [txn(900 + d, "s", d, device="d-solo") for d in range(3)]
    g = graph(*ring, *stranger)
    comm = g.community("r0", ts(2))
    assert comm["members"] == ["r0", "r1", "r2", "r3"]
    assert g.community("s", ts(2))["members"] == ["s"]
    # a customer about to use the ring's beneficiary for the first time already belongs to the group
    assert g.community("newbie", ts(3), extra=(("PAID", "b-mule"),))["members"] == ["newbie", "r0", "r1", "r2", "r3"]
    assert g.community("newbie", ts(3))["members"] == ["newbie"]
    # the same answer when the graph already holds the newcomer's own transaction (as the investigator view's does),
    # because it is asked as of a time before that transaction was seen
    g.observe(txn(500, "newbie", 3, ben="b-mule"))
    assert g.community("newbie", ts(3) - 1, extra=(("PAID", "b-mule"),))["members"] == ["newbie", "r0", "r1", "r2", "r3"]


# ---- features: point in time, no leakage ---------------------------------------------------------------
def rows_with_ring():
    """Day 0-9: a ring of four customers sharing a device and a beneficiary; customer r0 is confirmed fraud on day 5
    (matured day 6). Day 10: a new customer uses the ring's beneficiary."""
    txns = [txn(i, f"r{i % 4}", i * 0.5, device="d-ring", ben="b-ring") for i in range(20)]
    txns.append(txn(100, "newbie", 10, ben="b-ring"))
    out = [outcome("t10", "confirmed_fraud", 6)]  # t10 is customer r2 on day 5
    return txns, out


def test_graph_features_see_a_known_fraud_only_after_it_matured():
    txns, outs = rows_with_ring()
    ids, G, _ = build_graph_table(txns, outs)
    row = {t: G[k] for k, t in enumerate(ids)}
    prox = GRAPH_FEATURES.index("g_fraud_prox_txn")
    early = row["t8"][prox]   # day 4: the fraud has not matured
    late = row["t100"][prox]  # day 10: it has
    assert early == 0.0 and late > 0.0
    assert row["t100"][GRAPH_FEATURES.index("g_ben_shared")] > 0.0


def test_a_transaction_never_sees_itself_or_the_future():
    txns, outs = rows_with_ring()
    ids, G, F = build_graph_table(txns, outs)
    first = ids.index("t0")
    assert not G[first].any() or G[first][GRAPH_FEATURES.index("g_min_conf")] == 1.0  # nothing before it
    assert F[first].sum() == 0
    # dropping every later transaction changes nothing about an earlier one
    cut = [t for t in txns if t["txn_id"] in {f"t{i}" for i in range(8)}]
    ids2, G2, F2 = build_graph_table(cut, [])
    for k, tid in enumerate(ids2):
        assert np.allclose(G2[k], G[ids.index(tid)]) and np.allclose(F2[k], F[ids.index(tid)])


def test_an_outcome_that_matures_later_cannot_change_an_earlier_feature():
    txns, outs = rows_with_ring()
    a = build_graph_table(txns, outs)                                               # matures day 6
    b = build_graph_table(txns, [outcome("t10", "confirmed_fraud", 11)])            # matures day 11
    prox = GRAPH_FEATURES.index("g_fraud_prox_txn")
    assert np.allclose(a[1][a[0].index("t8")], b[1][b[0].index("t8")])              # day 4: neither knows
    assert a[1][a[0].index("t100")][prox] > 0.0                                      # day 10: known in the first
    assert b[1][b[0].index("t100")][prox] == 0.0                                     # not yet known in the second


def test_same_timestamp_transactions_do_not_see_each_other():
    twins = [txn(0, "a", 3, device="d"), txn(1, "b", 3, device="d")]
    _, G, F = build_graph_table(twins, [])
    assert not G.any(axis=0)[: GRAPH_FEATURES.index("g_ben_shared") + 1].any()
    assert F.sum() == 0


def test_the_flat_baseline_counts_the_same_entities_without_confidence_or_decay():
    txns, outs = rows_with_ring()
    ids, G, F = build_graph_table(txns, outs)
    k = ids.index("t100")
    assert F[k][FLAT_FEATURES.index("f_ben_customers")] == 4.0  # r0..r3, however old their links
    assert F[k][FLAT_FEATURES.index("f_ben_fraud")] == 1.0
    # the graph weights those same four by confidence, so it is not simply the count
    assert 0 < G[k][GRAPH_FEATURES.index("g_ben_shared")] < 4.0


def test_the_flat_window_forgets_old_links_and_the_graph_decays_them_too():
    txns = [txn(0, "old", 0, ben="b"), txn(1, "new", 200, ben="b")]
    ids, G, F = build_graph_table(txns, [])
    k = ids.index("t1")
    assert F[k][FLAT_FEATURES.index("f_ben_customers")] == 0.0
    assert G[k][GRAPH_FEATURES.index("g_ben_shared")] < 0.1


def test_a_new_link_into_an_existing_structure_is_noticed():
    txns = [txn(0, "a", 0, device="d"), txn(1, "b", 1, device="d"), txn(2, "b", 2, device="d")]
    ids, G, _ = build_graph_table(txns, [])
    nl = GRAPH_FEATURES.index("g_new_link")
    assert G[ids.index("t1")][nl] == 1.0   # b first uses a device a already used
    assert G[ids.index("t2")][nl] == 0.0   # not new any more


def test_ring_score_is_higher_for_a_dense_recent_fraud_linked_group_than_for_a_loose_old_one():
    txns, outs = rows_with_ring()
    ids, G, _ = build_graph_table(txns, outs)
    ring = G[ids.index("t100")][GRAPH_FEATURES.index("g_ring_score")]
    loose = [txn(10 * d + i, f"x{i}", i * 40 + d, device=f"d{i}", ben="b-wide") for d in range(3) for i in range(5)]
    ids2, G2, _ = build_graph_table(loose + [txn(99, "y", 250, ben="b-wide")], [])
    assert ring > G2[ids2.index("t99")][GRAPH_FEATURES.index("g_ring_score")]


def test_features_have_the_documented_width_and_are_finite():
    txns, outs = rows_with_ring()
    _, G, F = build_graph_table(txns, outs)
    assert G.shape == (len(txns), len(GRAPH_FEATURES)) and F.shape == (len(txns), len(FLAT_FEATURES))
    assert np.isfinite(G).all() and np.isfinite(F).all()


def test_fraud_knowledge_counts_only_matured_fraud_outcomes():
    txns = [txn(0, "a", 0), txn(1, "b", 0)]
    fk = FraudKnowledge([outcome("t0", "confirmed_fraud", 5), outcome("t1", "confirmed_legitimate", 5),
                         {"txn_id": "t1", "outcome_type": "chargeback", "matured_at": None}], txns)
    assert fk.known("a", ts(5)) and not fk.known("a", ts(4.9))
    assert not fk.known("b", ts(100))


def test_graph_features_function_matches_the_builder_on_a_live_graph():
    txns, outs = rows_with_ring()
    ids, G, _ = build_graph_table(txns, outs)
    g, fk = TemporalGraph(), FraudKnowledge(outs, txns)
    for t in txns[:-1]:
        g.observe(t)
    probe = txns[-1]
    t = datetime.fromisoformat(probe["event_time"]).timestamp()
    assert np.allclose(graph_features(g, fk, probe, t), G[ids.index("t100")])
    assert SECONDS_PER_DAY == 86400.0
