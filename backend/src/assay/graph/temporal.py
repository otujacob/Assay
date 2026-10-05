"""Temporal, confidence-weighted entity graph (PRD 13, V1).

Nodes are the pseudonymised identifiers that already exist on a transaction: customer, account, device, IP address,
merchant and beneficiary. An edge is a relationship OBSERVED on transactions (customer USED_DEVICE device, customer
ACCESSED_FROM ip, customer PAID beneficiary, account TRANSACTED_AT merchant, customer OWNS account). Two customers are
linked, through a shared node, by a DERIVED edge (SHARES_DEVICE, SHARES_IP, SHARES_BENEFICIARY, SHARES_MERCHANT).

Every observation is stored with two times, so the graph is BITEMPORAL (PRD 13.4): when the relationship was true
(`valid`, the transaction's event time) and when Assay learned of it (`recorded`). A query names both, and sees only
observations that satisfy both: the graph as it was at a decision time, whatever arrived later. Nothing is deleted.

Confidence (PRD 13.3) is how sure the graph is that a relationship is real and meaningful. It is NOT a fraud probability:

    c = (1 - exp(-k n))  *  exp(-lambda * age_days)  *  1 / (1 + ln(1 + deg))
        evidence             recency                      specificity

`n` observations, `age` since the last one, and `deg` the number of distinct customers using the far node in the recent
window (a device shared by 300 customers, or a public Wi-Fi address, is a weak link). When confidence falls below a floor
the edge is treated as EXPIRED at that time (never deleted; a later observation revives it). An analyst can flag a
relationship as wrong, which down-weights it from the time of the flag (PRD 13.7).

k, the per-relationship decay rates, the floor and the specificity form are an initial design to be tuned, not
validated values (PRD 13.3). Communities are the connected groups of customers reachable through links above a confidence
threshold, a simpler stand-in for Louvain or Leiden that needs no dependency and is deterministic; it can merge two rings
joined by one strong link, and it is described that way wherever it is reported.
"""

from __future__ import annotations

import bisect
import math
from collections import defaultdict, deque
from dataclasses import dataclass, field

SECONDS_PER_DAY = 86400.0

# observed relationships: (relationship, the transaction field naming the far node)
OBSERVED = (("USED_DEVICE", "device_hash"), ("ACCESSED_FROM", "ip_hash"), ("PAID", "beneficiary_pid"),
            ("TRANSACTED_AT", "merchant_id"))
# Customers are linked through these by default. A merchant is recorded (OBSERVED) and can be asked for, but two customers
# who bought from the same large merchant are barely connected, so it does not link them unless a caller says so.
LINKING = ("USED_DEVICE", "ACCESSED_FROM", "PAID")
SHARES = {"USED_DEVICE": "SHARES_DEVICE", "ACCESSED_FROM": "SHARES_IP", "PAID": "SHARES_BENEFICIARY",
          "TRANSACTED_AT": "SHARES_MERCHANT"}
GRAPH_VERSION = "graph-0"


@dataclass(frozen=True)
class GraphConfig:
    k: float = 0.7  # evidence: 1 - exp(-k n)
    decay_per_day: dict[str, float] = field(default_factory=lambda: {
        "USED_DEVICE": 1 / 120, "ACCESSED_FROM": 1 / 20, "PAID": 1 / 90, "TRANSACTED_AT": 1 / 45, "OWNS": 1 / 365})
    floor: float = 0.05  # below this confidence an edge is expired
    degree_window_days: float = 60.0  # how recent a use of a node counts towards its degree
    link_threshold: float = 0.25  # a derived link must be at least this confident to join a community
    flag_weight: float = 0.2  # an analyst-flagged relationship keeps this share of its confidence
    max_hub_degree: int = 60  # a node used by more customers than this is not expanded (it links everyone to everyone)
    version: str = GRAPH_VERSION


@dataclass
class Edge:
    rel: str
    src: str
    dst: str
    valid: list[float] = field(default_factory=list)  # observation times (event time), sorted
    recorded: list[float] = field(default_factory=list)  # when each was learned, in the same order
    max_rec: float = -math.inf  # the latest recorded time, so the common case needs no scan
    flagged_at: float | None = None  # analyst flagged it as wrong, effective from this recorded time

    def seen(self, valid_to: float, recorded_to: float) -> tuple[int, float | None, float | None]:
        """(count, first_seen, last_seen) of observations true by `valid_to` AND known by `recorded_to`."""
        if self.max_rec <= recorded_to:  # everything was known in time: only the valid time cuts
            n = bisect.bisect_right(self.valid, valid_to)
            return (n, self.valid[0], self.valid[n - 1]) if n else (0, None, None)
        ok = [v for v, r in zip(self.valid, self.recorded, strict=True) if v <= valid_to and r <= recorded_to]
        return (len(ok), ok[0], ok[-1]) if ok else (0, None, None)


@dataclass(frozen=True)
class Link:
    """A derived customer-to-customer relationship through one shared node."""
    other: str
    rel: str  # SHARES_*
    via: str  # the shared node
    confidence: float
    observations: int
    first_seen: float
    last_seen: float


class TemporalGraph:
    def __init__(self, cfg: GraphConfig | None = None):
        self.cfg = cfg or GraphConfig()
        self._edges: dict[tuple[str, str, str], Edge] = {}
        self._by_dst: dict[tuple[str, str], set[str]] = defaultdict(set)  # (rel, far node) -> customers
        self._by_src: dict[str, dict[str, set[str]]] = defaultdict(lambda: defaultdict(set))  # customer -> rel -> nodes
        # Answers are cached for one (valid time, recorded time) and cleared when either changes or anything is written, so
        # the many lookups behind one transaction's features do not recount the same node.
        self._nwrites = 0
        self._ctx: tuple | None = None
        self._deg: dict[tuple[str, str], int] = {}
        self._conf: dict[tuple[str, str, str], float] = {}
        self._lnk: dict[tuple, list] = {}

    def _at(self, t: float, rec: float) -> None:
        key = (t, rec, self._nwrites)
        if self._ctx != key:
            self._ctx = key
            self._deg.clear()
            self._conf.clear()
            self._lnk.clear()

    # ------------------------------------------------------------------ writing -----------------
    def observe(self, txn: dict, recorded_at: float | None = None) -> None:
        """Record the relationships on one transaction. `txn` is a stored or wire transaction."""
        from assay.features.compute import parse_time

        t = parse_time(txn["event_time"]).timestamp()
        rec = t if recorded_at is None else recorded_at
        cust = txn["customer_pid"]
        if txn.get("account_pid"):
            self._add("OWNS", cust, txn["account_pid"], t, rec)
        for rel, fld in OBSERVED:
            dst = txn.get(fld)
            if dst:
                self._add(rel, cust, str(dst), t, rec)

    def _add(self, rel: str, src: str, dst: str, t: float, rec: float) -> None:
        e = self._edges.get((rel, src, dst))
        if e is None:
            e = self._edges[(rel, src, dst)] = Edge(rel, src, dst)
            self._by_dst[(rel, dst)].add(src)
            self._by_src[src][rel].add(dst)
        i = bisect.bisect_right(e.valid, t)
        e.valid.insert(i, t)
        e.recorded.insert(i, rec)
        e.max_rec = max(e.max_rec, rec)
        self._nwrites += 1

    def flag(self, rel: str, src: str, dst: str, recorded_at: float) -> bool:
        """An analyst says this relationship is wrong. It is down-weighted from `recorded_at`, never deleted."""
        e = self._edges.get((rel, src, dst))
        if e is None:
            return False
        e.flagged_at = recorded_at if e.flagged_at is None else min(e.flagged_at, recorded_at)
        self._nwrites += 1
        return True

    # ------------------------------------------------------------------ reading -----------------
    def degree(self, rel: str, dst: str, t: float, rec: float) -> int:
        """Distinct customers that used the far node in the degree window."""
        self._at(t, rec)
        key = (rel, dst)
        if key in self._deg:
            return self._deg[key]
        lo = t - self.cfg.degree_window_days * SECONDS_PER_DAY
        n = 0
        for src in self._by_dst.get(key, ()):
            _, _, last = self._edges[(rel, src, dst)].seen(t, rec)
            if last is not None and last >= lo:
                n += 1
        self._deg[key] = n
        return n

    def confidence(self, rel: str, src: str, dst: str, t: float, rec: float | None = None) -> float:
        """Confidence of the edge as of valid time `t` and recorded time `rec` (default: the same moment). 0 if not seen."""
        rec = t if rec is None else rec
        self._at(t, rec)
        key = (rel, src, dst)
        if key in self._conf:
            return self._conf[key]
        e = self._edges.get(key)
        c = 0.0
        if e is not None:
            n, _, last = e.seen(t, rec)
            if n:
                age_days = (t - last) / SECONDS_PER_DAY
                evidence = 1.0 - math.exp(-self.cfg.k * n)
                recency = math.exp(-self.cfg.decay_per_day.get(rel, 1 / 90) * age_days)
                specificity = 1.0 if rel == "OWNS" else 1.0 / (1.0 + math.log1p(self.degree(rel, dst, t, rec)))
                c = evidence * recency * specificity
                if e.flagged_at is not None and e.flagged_at <= rec:
                    c *= self.cfg.flag_weight
        self._conf[key] = c
        return c

    def active(self, rel: str, src: str, dst: str, t: float, rec: float | None = None) -> bool:
        """An edge whose confidence has decayed below the floor is expired (valid_to has passed), not deleted."""
        return self.confidence(rel, src, dst, t, rec) >= self.cfg.floor

    def entities(self, customer: str, rel: str) -> set[str]:
        return set(self._by_src.get(customer, {}).get(rel, ()))

    def links(self, customer: str, t: float, rec: float | None = None, *, rels: tuple[str, ...] | None = None,
              extra: tuple[tuple[str, str], ...] = ()) -> list[Link]:
        """The customer's derived links to other customers, as of the two times: customers who used a node this customer
        also used, with the weaker of the two edges' confidences. `extra` is (relationship, node) pairs the customer is
        about to use (the transaction being scored): a node new to the customer links it to that node's other users as if
        it already used it, at the other user's confidence, so a first use of a ring's beneficiary is not invisible."""
        rec = t if rec is None else rec
        self._at(t, rec)
        ckey = (customer, rels, extra)
        if ckey in self._lnk:
            return self._lnk[ckey]
        out: list[Link] = []
        for rel in rels or LINKING:
            nodes = self.entities(customer, rel) | {n for r, n in extra if r == rel}
            for node in sorted(nodes):
                mine = self.confidence(rel, customer, node, t, rec) if (rel, customer, node) in self._edges else None
                if mine == 0.0:
                    mine = None  # in the store, but not yet seen as of this time: a first use, whichever way it got here
                if mine is not None and mine < self.cfg.floor:
                    continue
                users = self._by_dst.get((rel, node), set())
                if len(users) - (customer in users) > self.cfg.max_hub_degree:
                    continue  # a hub links everyone to everyone; its edges still count, through specificity
                others = users - {customer}
                for o in sorted(others):
                    theirs = self.confidence(rel, o, node, t, rec)
                    if theirs < self.cfg.floor:
                        continue
                    n2, f2, l2 = self._edges[(rel, o, node)].seen(t, rec)
                    if mine is None:  # a first use: the link is as strong as the other user's, and starts now
                        out.append(Link(o, SHARES[rel], node, theirs, n2, t, t))
                        continue
                    _, f1, l1 = self._edges[(rel, customer, node)].seen(t, rec)
                    out.append(Link(o, SHARES[rel], node, min(mine, theirs), n2, min(f1, f2), max(l1, l2)))
        self._lnk[ckey] = out
        return out

    def community(self, customer: str, t: float, rec: float | None = None, *, depth: int = 2,
                  max_size: int = 200, extra: tuple[tuple[str, str], ...] = ()) -> dict:
        """Customers reachable from `customer` through links at or above the threshold (up to `depth` hops), with the edges
        among them. A connected group, not a statistically tested community: see the module note."""
        rec = t if rec is None else rec
        seen = {customer: 0}
        edges: dict[tuple[str, str], Link] = {}
        queue: deque[str] = deque([customer])
        while queue and len(seen) < max_size:
            c = queue.popleft()
            if seen[c] >= depth:
                continue
            for lk in self.links(c, t, rec, extra=extra if c == customer else ()):
                if lk.confidence < self.cfg.link_threshold:
                    continue
                key = tuple(sorted((c, lk.other)))
                if key not in edges or lk.confidence > edges[key].confidence:
                    edges[key] = lk
                if lk.other not in seen:
                    seen[lk.other] = seen[c] + 1
                    queue.append(lk.other)
        return {"members": sorted(seen), "edges": edges}
