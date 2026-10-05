"""Graph features, point in time (PRD 13.5), and the flat baseline they must beat (PRD 20, H6).

Every feature for a transaction is computed from the graph as it was BEFORE that transaction: nothing the transaction
itself adds, nothing observed later, and no outcome that had not matured by then. `build_graph_table` replays
transactions in time order, computes the features of every transaction that shares a timestamp first and only then
observes them, the same rule the flat feature builder uses (features/compute.py), so a feature never sees its own row.

Graph features (all `g_`; a parameter set, not validated values):
  g_dev_shared, g_ip_shared, g_ben_shared   confidence-weighted number of OTHER customers linked through this
                                            transaction's device, IP address, beneficiary
  g_fraud_prox_txn      the strongest confidence-weighted link, within two hops, from this transaction's
                        entities to a customer with a confirmed fraud known at that time (second hop halved)
  g_fraud_prox_cust     the same, from the customer's whole history of entities, not only this transaction's
  g_comm_size           members of the customer's connected group (capped), counting the customer
  g_comm_density        links among the group's members as a share of all possible pairs
  g_comm_fraud_share    share of the group's other members with a confirmed fraud known at that time
  g_ring_score          density x size x (0.5 + 0.5 fraud share) x recency of the group's newest link: a ring is a
                        dense, fraud-linked group that formed recently (PRD 13.5)
  g_new_link            how many of this transaction's device, IP, beneficiary are NEW to the customer but already
                        used by other customers: a first appearance of a structure (feeds novelty, PRD 9)
  g_min_conf            the weakest confidence among the links that fed the features, so a case that leans on
                        weak links can be flagged as such (PRD 13.5, reliability of graph context)

Flat baseline (`f_`): the SAME entity data, as ordinary engineered features with no graph: how many distinct other
customers used this device, IP, beneficiary in the last 30 days, and how many of those have a confirmed fraud known at
that time. No confidence, no recency decay, no second hop, no communities. If the graph does not beat this, it earns
nothing over a join (PRD H6).
"""

from __future__ import annotations

import math
from collections import defaultdict

import numpy as np

from assay.features.compute import parse_time

from .temporal import SECONDS_PER_DAY, GraphConfig, TemporalGraph

GRAPH_FEATURES = ("g_dev_shared", "g_ip_shared", "g_ben_shared", "g_fraud_prox_txn", "g_fraud_prox_cust",
                  "g_comm_size", "g_comm_density", "g_comm_fraud_share", "g_ring_score", "g_new_link", "g_min_conf")
FLAT_FEATURES = ("f_dev_customers", "f_ip_customers", "f_ben_customers", "f_dev_fraud", "f_ip_fraud", "f_ben_fraud")
ENTITY = (("USED_DEVICE", "device_hash"), ("ACCESSED_FROM", "ip_hash"), ("PAID", "beneficiary_pid"))
FLAT_WINDOW_DAYS = 30.0


class FraudKnowledge:
    """When each customer's first confirmed fraud became known. A fraud counts only from the moment its outcome matured."""

    def __init__(self, outcomes: list[dict], txns: list[dict]):
        from assay.detection.labels import FRAUD_TYPES

        cust = {t["txn_id"]: t["customer_pid"] for t in txns}
        first: dict[str, float] = {}
        for o in outcomes:
            if o.get("matured_at") is None or o["outcome_type"] not in FRAUD_TYPES or o["txn_id"] not in cust:
                continue
            m = parse_time(o["matured_at"]).timestamp()
            c = cust[o["txn_id"]]
            first[c] = min(first.get(c, math.inf), m)
        self._first = first

    def known(self, customer: str, t: float) -> bool:
        return self._first.get(customer, math.inf) <= t


def graph_features(g: TemporalGraph, fk: FraudKnowledge, txn: dict, t: float, rec: float | None = None) -> np.ndarray:
    """The `GRAPH_FEATURES` vector for a transaction, from the graph as of `t` (and recorded time `rec`)."""
    rec = t if rec is None else rec
    cust = txn["customer_pid"]
    shared = dict.fromkeys(("USED_DEVICE", "ACCESSED_FROM", "PAID"), 0.0)
    prox_txn = 0.0
    confs: list[float] = []
    new_link = 0.0
    for rel, fld in ENTITY:
        node = txn.get(fld)
        if not node:
            continue
        node = str(node)
        users = g._by_dst.get((rel, node), set()) - {cust}
        hub = len(users) > g.cfg.max_hub_degree
        for o in sorted(users):
            w = g.confidence(rel, o, node, t, rec)
            if w < g.cfg.floor:
                continue
            shared[rel] += w
            confs.append(w)
            if not hub:
                if fk.known(o, t):
                    prox_txn = max(prox_txn, w)
                else:  # second hop: a customer linked to this one who has a confirmed fraud
                    for lk in g.links(o, t, rec):
                        if lk.confidence >= g.cfg.floor and fk.known(lk.other, t) and lk.other != cust:
                            prox_txn = max(prox_txn, 0.5 * w * lk.confidence)
        if users and g.confidence(rel, cust, node, t, rec) < g.cfg.floor:
            new_link += 1.0  # new to this customer, already used by others

    here = tuple((rel, str(txn[fld])) for rel, fld in ENTITY if txn.get(fld))
    own = g.links(cust, t, rec, extra=here)
    prox_cust = 0.0
    for lk in own:
        if fk.known(lk.other, t):
            prox_cust = max(prox_cust, lk.confidence)
        else:
            for l2 in g.links(lk.other, t, rec):
                if fk.known(l2.other, t) and l2.other != cust:
                    prox_cust = max(prox_cust, 0.5 * lk.confidence * l2.confidence)
    comm = g.community(cust, t, rec, extra=here)
    members, edges = comm["members"], comm["edges"]
    size = len(members)
    density = (len(edges) / (size * (size - 1) / 2)) if size > 1 else 0.0
    others = [m for m in members if m != cust]
    fraud_share = (sum(1 for m in others if fk.known(m, t)) / len(others)) if others else 0.0
    newest = max((lk.last_seen for lk in edges.values()), default=None)
    recency = math.exp(-max(0.0, (t - newest)) / SECONDS_PER_DAY / 30.0) if newest is not None else 0.0
    ring = density * min(size, 10) / 10.0 * (0.5 + 0.5 * fraud_share) * recency
    confs += [lk.confidence for lk in own]
    return np.array([shared["USED_DEVICE"], shared["ACCESSED_FROM"], shared["PAID"], prox_txn, prox_cust,
                     float(min(size, 50)), density, fraud_share, ring, new_link, min(confs) if confs else 1.0])


class _Flat:
    """The flat baseline: plain 30-day counts of distinct other customers per entity, with no graph."""

    def __init__(self):
        self.users: dict[tuple[str, str], dict[str, float]] = defaultdict(dict)  # (field, node) -> customer -> last seen

    def features(self, fk: FraudKnowledge, txn: dict, t: float) -> list[float]:
        cust, lo = txn["customer_pid"], t - FLAT_WINDOW_DAYS * SECONDS_PER_DAY
        counts, frauds = [], []
        for _, fld in ENTITY:
            node = txn.get(fld)
            users = {c: ts for c, ts in self.users.get((fld, str(node)), {}).items() if c != cust and ts >= lo} if node else {}
            counts.append(float(len(users)))
            frauds.append(float(sum(1 for c in users if fk.known(c, t))))
        return counts + frauds

    def observe(self, txn: dict, t: float) -> None:
        for _, fld in ENTITY:
            if txn.get(fld):
                self.users[(fld, str(txn[fld]))][txn["customer_pid"]] = t


def build_graph_table(txns: list[dict], outcomes: list[dict], cfg: GraphConfig | None = None
                      ) -> tuple[list[str], np.ndarray, np.ndarray]:
    """(txn_ids, graph feature matrix, flat feature matrix), one row per transaction in (time, id) order. Transactions
    that share a timestamp do not see each other."""
    keyed = sorted(((parse_time(x["event_time"]).timestamp(), x["txn_id"], x) for x in txns), key=lambda k: (k[0], k[1]))
    rows = [k[2] for k in keyed]
    times = [k[0] for k in keyed]
    g, flat, fk = TemporalGraph(cfg), _Flat(), FraudKnowledge(outcomes, txns)
    G = np.empty((len(rows), len(GRAPH_FEATURES)))
    F = np.empty((len(rows), len(FLAT_FEATURES)))
    ids: list[str] = []
    i = 0
    while i < len(rows):
        t = times[i]
        j = i
        while j < len(rows) and times[j] == t:
            j += 1
        for k in range(i, j):
            G[k] = graph_features(g, fk, rows[k], t)
            F[k] = flat.features(fk, rows[k], t)
            ids.append(rows[k]["txn_id"])
        for k in range(i, j):
            g.observe(rows[k])
            flat.observe(rows[k], t)
        i = j
    return ids, G, F
