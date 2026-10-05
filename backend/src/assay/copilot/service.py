"""Builds a case summary from the stored record and the entity graph. Reading it is audited (FR-43)."""

from __future__ import annotations

from assay.graph.service import GraphCache, GraphService

from .summary import summarise


class CaseSummaryService:
    def __init__(self, scoring, graph_cache: GraphCache | None = None):
        self.scoring = scoring
        self.graph = GraphService(scoring, graph_cache)

    def summary(self, tenant_id: str, decision_id: str, reader: str = "system", with_graph: bool = True) -> dict:
        sc = self.scoring
        decision = sc.get_decision(tenant_id, decision_id)
        t_high = float(sc.registry.get(tenant_id, decision["bundle_id"]).manifest.thresholds["t_high"])
        graph = self.graph.neighbourhood(tenant_id, decision_id, reader) if with_graph else None
        out = summarise(decision, t_high, sc.explanation(tenant_id, decision_id), graph)
        sc.repo.append_audit(tenant_id, reader, "summary_read", decision["txn_id"], f"{len(out['facts'])} facts", sc.cfg.clock())
        return out
