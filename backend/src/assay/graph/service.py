"""The entity graph as investigator context (PRD 13, V1): what else is linked to this case, and how sure the graph is.

The graph is a deterministic function of the stored transactions, outcomes and analyst flags, so there is no second
copy of relationship data to fall out of step: it is rebuilt from them (incrementally, as new transactions arrive) and
can be rebuilt as it was at any past decision time. It supplies CONTEXT to a person. It is not a model input: the
graph did not clear the pre-set test for that (docs/validation), and the PRD keeps it as investigator context
otherwise (PRD 20, H6).

What it shows is evidence of shared infrastructure, not of guilt: families share devices, and public Wi-Fi addresses are
used by hundreds of strangers. Confidence says how sure the graph is that a relationship is real and meaningful, and
is never a fraud probability. Identifiers are the pseudonyms already on the transaction. Nothing here is for a customer.
"""

from __future__ import annotations

import threading
from datetime import UTC, datetime

from assay.features.compute import parse_time

from .features import ENTITY, GRAPH_FEATURES, FraudKnowledge, graph_features
from .temporal import SHARES, GraphConfig, TemporalGraph

MAX_LINKS = 20
WEAK = 0.2
NOTE = ("Shared infrastructure is evidence of a connection, not of wrongdoing: families share devices and public "
        "networks are used by strangers. Confidence says how sure the graph is that a link is real, not that anyone "
        "has committed fraud. This view is for investigators and must not be shared with a customer.")
REL_LABEL = {"USED_DEVICE": "device", "ACCESSED_FROM": "IP address", "PAID": "beneficiary", "TRANSACTED_AT": "merchant"}


class GraphError(Exception):
    def __init__(self, code: str, detail: str, http: int = 422):
        super().__init__(detail)
        self.code, self.http = code, http


class GraphCache:
    """One incrementally built graph per tenant, shared by every request in the process."""

    def __init__(self, cfg: GraphConfig | None = None):
        self.cfg = cfg or GraphConfig()
        self._lock = threading.Lock()
        self._state: dict[str, dict] = {}

    def get(self, repo, tenant_id: str) -> tuple[TemporalGraph, list[dict], list[dict]]:
        txns = repo.find(tenant_id, "transactions")
        flags = repo.find(tenant_id, "graph_edge_flags")
        with self._lock:
            st = self._state.get(tenant_id)
            if st is None or st["n_txns"] > len(txns):  # rows only ever grow, so fewer rows means a different store
                st = self._state[tenant_id] = {"g": TemporalGraph(self.cfg), "n_txns": 0, "n_flags": 0}
            g: TemporalGraph = st["g"]
            for row in txns[st["n_txns"]:]:
                rec = row.get("recorded_at")
                g.observe(row, parse_time(rec).timestamp() if rec else None)
            st["n_txns"] = len(txns)
            for f in flags[st["n_flags"]:]:
                g.flag(f["rel"], f["src"], f["dst"], _ts(f["flagged_at"]))
            st["n_flags"] = len(flags)
            return g, txns, flags


def _ts(v) -> float:
    return parse_time(v).timestamp() if not isinstance(v, datetime) else (v if v.tzinfo else v.replace(tzinfo=UTC)).timestamp()


def _iso(t: float) -> str:
    return datetime.fromtimestamp(t, UTC).isoformat()


class GraphService:
    def __init__(self, scoring, cache: GraphCache | None = None):
        self.scoring = scoring
        self.repo = scoring.repo
        self.cache = cache or GraphCache()

    def _known_fraud(self, tenant_id: str, txns: list[dict]) -> FraudKnowledge:
        return FraudKnowledge(self.repo.find(tenant_id, "outcomes", {"maturity_state": "matured"}), txns)

    def neighbourhood(self, tenant_id: str, decision_id: str, reader: str = "system") -> dict:
        """The graph around a decision's transaction, as it was when the decision was made."""
        pd = self.scoring._decision_row(tenant_id, decision_id)
        txn = self.repo.get_transaction(tenant_id, pd["txn_id"])
        ta = self.repo.get_by_id(tenant_id, "trust_assessments", pd["trust_assessment_id"])
        pr = self.repo.get_by_id(tenant_id, "predictions", ta["prediction_id"])
        fv = self.repo.get_by_id(tenant_id, "feature_vectors", pr["feature_vector_id"])
        t = parse_time(txn["event_time"]).timestamp()
        rec = _ts(fv["as_of_time"])  # what Assay knew when it decided: later arrivals do not appear (PRD 13.4)
        g, txns, flags = self.cache.get(self.repo, tenant_id)
        fk = self._known_fraud(tenant_id, txns)
        cust = txn["customer_pid"]
        # the graph strictly BEFORE this transaction: take its own observation away from the picture
        here = tuple((rel, str(txn[fld])) for rel, fld in ENTITY if txn.get(fld))
        t_before = t - 1e-3
        entities = []
        for rel, node in here:  # others who had used it by then (a later user is not part of what was known)
            users = [o for o in sorted(g._by_dst.get((rel, node), set()) - {cust})
                     if g.confidence(rel, o, node, t_before, rec) > 0]
            entities.append({"relationship": rel, "kind": REL_LABEL[rel], "node": node, "other_customers": len(users),
                             "degree": g.degree(rel, node, t_before, rec),
                             "hub": len(users) > g.cfg.max_hub_degree})
        links = []
        flagged = {(f["rel"], f["src"], f["dst"]) for f in flags}
        for lk in g.links(cust, t_before, rec, extra=here):
            rel = REL_OF[lk.rel]
            links.append({"other_customer": lk.other, "relationship": lk.rel, "via_kind": REL_LABEL[rel], "via": lk.via,
                          "confidence": round(lk.confidence, 3), "observations": lk.observations,
                          "first_seen": _iso(lk.first_seen), "last_seen": _iso(lk.last_seen),
                          "fraud_known": fk.known(lk.other, t_before),
                          "flagged_wrong": (rel, lk.other, lk.via) in flagged or (rel, cust, lk.via) in flagged})
        links.sort(key=lambda x: (-x["confidence"], x["other_customer"]))
        feats = dict(zip(GRAPH_FEATURES, (round(float(v), 4) for v in graph_features(g, fk, txn, t_before, rec)), strict=True))
        comm = g.community(cust, t_before, rec, extra=here)
        weakest = min([lk["confidence"] for lk in links], default=None)
        self.repo.append_audit(tenant_id, reader, "graph_read", pd["txn_id"], f"{len(links)} links", self.scoring.cfg.clock())
        return {"decision_id": decision_id, "customer": cust, "as_of": _iso(rec), "graph_version": g.cfg.version,
                "entities": entities, "links": links[:MAX_LINKS], "links_total": len(links),
                "group": {"size": len(comm["members"]), "links": len(comm["edges"]),
                          "fraud_linked_members": sum(1 for m in comm["members"] if m != cust and fk.known(m, t_before))},
                "features": feats,
                "reliability": "weak" if weakest is not None and weakest < WEAK else "ok",
                "reliability_note": ("This view leans on weak links (confidence below 0.2)." if weakest is not None and weakest < WEAK
                                     else None),
                "note": NOTE}

    def flag(self, tenant_id: str, actor: str, *, rel: str, src: str, dst: str, reason: str) -> dict:
        """An analyst says a relationship is wrong (PRD 13.7). It is down-weighted from now, never deleted."""
        if rel not in REL_LABEL:
            raise GraphError("bad_relationship", f"relationship must be one of {sorted(REL_LABEL)}")
        if not (src and dst):
            raise GraphError("bad_edge", "src and dst are required")
        if not reason or not reason.strip():
            raise GraphError("reason_required", "flagging a relationship needs a reason")
        g, _, _ = self.cache.get(self.repo, tenant_id)
        if (rel, src, dst) not in g._edges:
            raise GraphError("unknown_edge", "no such relationship in this tenant's graph", 404)
        now = self.scoring.cfg.clock()
        with self.repo.atomic(tenant_id):
            self.repo.insert(tenant_id, "graph_edge_flags", {
                "schema_version": "gra-1", "rel": rel, "src": src, "dst": dst, "reason": reason.strip(),
                "flagged_at": now}, actor)
            self.repo.append_audit(tenant_id, actor, "graph_edge_flagged", f"{rel}:{src}:{dst}", "flagged", now)
        self.cache.get(self.repo, tenant_id)  # apply it now
        return {"flagged": True, "relationship": rel, "src": src, "dst": dst, "effect": "down-weighted, not deleted"}


REL_OF = {v: k for k, v in SHARES.items()}  # SHARES_DEVICE -> USED_DEVICE, and so on
