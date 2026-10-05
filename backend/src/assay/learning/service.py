"""Continuous learning, as a service (PRD 11, 12): feedback pool, candidate models, and their governed life.

This is the only place that joins the pieces: the stored analyst actions and outcomes feed the feedback pool
(pool.py); an accepted pool and the verified outcomes train a candidate (detection/train.py), which is signed,
recorded in lineage, scored against the champion on the verified-outcome-only holdout (gates.py) and then moves
through shadow, approval, canary, promotion and rollback (lifecycle.py).

Nothing runs by itself. A person starts every step, and promotion needs a second person (PRD 12.1, FR-23).
Training is a heavy job, so `create_candidate` is meant to be called from a worker or a script, not a request.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from assay.detection import TrainingConfig, train_bundle
from assay.detection.bundle import save_bundle
from assay.learning import gates as G
from assay.learning import monitor as mon
from assay.learning.lifecycle import LifecycleConfig, LifecycleError, LifecycleStore
from assay.learning.pool import POOL_VERSION, PoolConfig, PoolReport, assess
from assay.review.service import _family
from assay.scoring.service import ScoringError, ensure_bundle_row

FRAUD = ("confirmed_fraud", "chargeback")


@dataclass
class LearningConfig:
    pool: PoolConfig = field(default_factory=PoolConfig)
    gates: G.GateConfig = field(default_factory=G.GateConfig)
    lifecycle: LifecycleConfig = field(default_factory=LifecycleConfig)
    monitor: mon.MonitorConfig = field(default_factory=mon.MonitorConfig)
    bundle_dir: Path | None = None  # candidates are saved here, signed; without it G8 fails
    signing_key: bytes | None = None
    key_provider: Any = None  # seals the artefact with the tenant key (FR-41)


class LearningService:
    def __init__(self, repo, scoring, cfg: LearningConfig | None = None):
        self.repo, self.scoring, self.cfg = repo, scoring, cfg or LearningConfig()
        self.store = LifecycleStore(repo, self.cfg.lifecycle)

    # ---------------------------------------------------------------- feedback pool --------
    def _inputs(self, tenant_id: str) -> tuple[list[dict], dict[str, int], dict[str, str | None]]:
        actions = self.repo.find(tenant_id, "analyst_actions")
        outcomes: dict[str, int] = {}
        for o in self.repo.find(tenant_id, "outcomes", {"maturity_state": "matured"}):
            if o["outcome_type"] in FRAUD:
                outcomes[o["txn_id"]] = 1
            else:
                outcomes.setdefault(o["txn_id"], 0)
        shown = {a["txn_id"] for a in actions if (a.get("display_state") or {}).get("recommendation_shown")}
        rec = {pd["txn_id"]: _family(pd["recommended_action"])
               for pd in self.repo.find(tenant_id, "policy_decisions") if pd["txn_id"] in shown}
        return actions, outcomes, rec

    def pool(self, tenant_id: str, as_of: datetime | None = None) -> PoolReport:
        actions, outcomes, rec = self._inputs(tenant_id)
        return assess(actions, outcomes, rec, as_of or self.cfg.lifecycle.clock(), self.cfg.pool)

    def pool_summary(self, tenant_id: str) -> dict:
        rep = self.pool(tenant_id)
        return {**rep.counts(), "acceptance_rate": rep.acceptance_rate(), "pool_version": rep.version,
                "flagged": rep.flagged,
                "analyst_accuracy": {a: v for a, v in rep.analyst_accuracy.items()},
                "note": "unverified labels only count towards a candidate after it passes the gates and is approved"}

    # ---------------------------------------------------------------- candidates -----------
    def create_candidate(self, tenant_id: str, actor: str, *, txns: list[dict], outcomes: list[dict],
                         training: TrainingConfig, use_feedback: bool = True) -> dict:
        """Train a candidate on verified outcomes plus the accepted pool, record it, and run the gates."""
        champion = self.scoring.deciding_bundle(tenant_id)
        rep = self.pool(tenant_id, training.as_of) if use_feedback else None
        extra = rep.labels() if rep else {}
        res = train_bundle(txns, outcomes, training, extra_labels=extra)
        used = {t for t in extra if t in set(res.train_table.txn_ids)} if res.train_table is not None else set()

        artefact = res.scoring_bundle()
        path: str | None = None
        if self.cfg.bundle_dir and self.cfg.signing_key:
            d = Path(self.cfg.bundle_dir) / tenant_id / res.manifest.bundle_id
            save_bundle(d, artefact, res.manifest, self.cfg.signing_key, encrypt_with=self.cfg.key_provider)
            path = str(d)
        self.scoring.registry.register(tenant_id, artefact, res.manifest, champion=False)
        ensure_bundle_row(self.repo, tenant_id, res.manifest, actor)

        by_id = {t["txn_id"]: t for t in txns}
        report = G.evaluate(
            res, G.champion_on(res, champion), tenant_id=tenant_id, txns_by_id=by_id, cfg=self.cfg.gates,
            pool_labels=set(extra) if rep else None, used_labels=used if rep else None,
            segment_of={t["txn_id"]: str(t.get("channel")) for t in txns},
            artefact_signed=bool(res.manifest.signature), lineage_recorded=True)
        summary = {**rep.counts(), "acceptance_rate": rep.acceptance_rate(), "pool_version": POOL_VERSION,
                   "flagged": rep.flagged} if rep else {"feedback_used": False}
        info = {"dataset_id": res.manifest.dataset_id, "extra_labels_used": res.manifest.extra.get("extra_labels_used", 0),
                "feedback_used": bool(rep), "as_of": training.as_of.isoformat(),
                "horizon_days": training.horizon_days, "train_end": training.train_end.isoformat(),
                "test_end": training.test_end.isoformat(), "metrics": res.manifest.metrics,
                "thresholds": res.manifest.thresholds}
        return self.store.create(tenant_id, actor, candidate_id=res.manifest.bundle_id,
                                 base_bundle_id=champion.manifest.bundle_id, artefact_path=path,
                                 pool_summary=summary, training=info, gates=report.as_dict())

    def _with_shadow(self, tenant_id: str, out: dict) -> dict:
        """A candidate in shadow carries its running evidence, wherever it is read."""
        if out["state"] == "shadow":
            out["shadow"] = self.shadow_report(tenant_id, out["candidate_id"])
        return out

    def candidates(self, tenant_id: str) -> list[dict]:
        return [self._with_shadow(tenant_id, self.store.summary(tenant_id, c["candidate_id"]))
                for c in self.store.candidates(tenant_id)]

    def get(self, tenant_id: str, candidate_id: str) -> dict:
        return self._with_shadow(tenant_id, self.store.summary(tenant_id, candidate_id))

    # ---------------------------------------------------------------- lifecycle steps ------
    def _done(self, tenant_id: str, out: dict) -> dict:
        self.scoring.sync_routing(tenant_id)
        return out

    def start_shadow(self, tenant_id: str, actor: str, candidate_id: str) -> dict:
        self._ensure_loaded(tenant_id, candidate_id)
        return self._done(tenant_id, self.store.start_shadow(tenant_id, actor, candidate_id))

    def shadow_report(self, tenant_id: str, candidate_id: str) -> dict:
        _, outcomes, _ = self._inputs(tenant_id)
        return self.store.shadow_report(tenant_id, candidate_id, outcomes=outcomes)

    def approve(self, tenant_id: str, actor: str, candidate_id: str, **kw) -> dict:
        _, outcomes, _ = self._inputs(tenant_id)
        return self._done(tenant_id, self.store.approve(tenant_id, actor, candidate_id, outcomes=outcomes, **kw))

    def start_canary(self, tenant_id: str, actor: str, candidate_id: str, share: float) -> dict:
        self._ensure_loaded(tenant_id, candidate_id)
        return self._done(tenant_id, self.store.start_canary(tenant_id, actor, candidate_id, share))

    def promote(self, tenant_id: str, actor: str, candidate_id: str) -> dict:
        self.scoring.sync_routing(tenant_id)
        current = self.scoring.registry.deciding_id(tenant_id)
        return self._done(tenant_id, self.store.promote(tenant_id, actor, candidate_id, current_champion=current))

    def rollback(self, tenant_id: str, actor: str, candidate_id: str, reason: str) -> dict:
        return self._done(tenant_id, self.store.rollback(tenant_id, actor, candidate_id, reason))

    def reject(self, tenant_id: str, actor: str, candidate_id: str, reason: str) -> dict:
        return self._done(tenant_id, self.store.reject(tenant_id, actor, candidate_id, reason))

    def _ensure_loaded(self, tenant_id: str, candidate_id: str) -> None:
        self.store.candidate(tenant_id, candidate_id)  # an unknown or foreign candidate is a 404, not a 503
        try:
            self.scoring.registry.get(tenant_id, candidate_id)
        except ScoringError:
            raise LifecycleError("bundle_not_loaded", "the candidate's artefact is not available to this process", 503) from None

    def check_health(self, tenant_id: str, *, act: bool = False, actor: str = "system:monitor") -> list[dict]:
        """Judge every model now deciding live cases (a canary, or a promoted champion) on its matured outcomes. With
        `act`, a clear breach rolls it back to the previous champion, with the evidence stored as the reason. It never
        promotes, approves or starts anything."""
        out = []
        for c in self.store.candidates(tenant_id):
            cid = c["candidate_id"]
            if self.store.state_of(tenant_id, cid) not in ("canary", "champion"):
                continue
            lb = self.scoring.registry.get(tenant_id, cid)
            risk, fraud, high = mon.collect(self.repo, tenant_id, cid)
            rep = mon.evaluate(cid, risk, fraud, high, float(lb.manifest.thresholds["t_high"]),
                               (lb.manifest.metrics or {}).get("at_t_high"), self.cfg.monitor).as_dict()
            rep["rolled_back"] = False
            if act and rep["status"] == "breach":
                self.rollback(tenant_id, actor, cid, "automatic: " + "; ".join(rep["triggers"]))
                rep["rolled_back"] = True
            out.append(rep)
        return out

    def status(self, tenant_id: str) -> dict:
        self.scoring.sync_routing(tenant_id)
        r = self.scoring.registry._routing.get(tenant_id)
        return {"champion": self.scoring.registry.deciding_id(tenant_id),
                "shadow": r.shadow if r else None,
                "canary": {"candidate": r.canary, "share": r.canary_share} if r and r.canary else None,
                "candidates": len(self.store.candidates(tenant_id))}
