"""Scoring service: a stored transaction becomes a decision, and every step is lineage.

transaction -> point-in-time features -> prediction -> (explanation) -> trust assessment ->
policy decision. Each arrow is an append-only record (PRD 14.1, FR-20, FR-31). Nothing here
takes an action: automation level is 0 and the output is a recommendation (FR-23).

Explanation testing is the expensive part, so `should_explain` (OPD-7) decides which cases get it
synchronously: cases the policy might review, plus a deterministic audit sample. A case without
an explanation has exp missing, so its Trust Index is capped below High (PRD 5.4); `refine`
computes it later and stores a NEW trust-assessment version beside the first (PRD 5.8).
"""

from __future__ import annotations

import math
import zlib
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from typing import Any

import numpy as np

from assay.detection.bundle import BundleManifest
from assay.features import FEATURE_SET_VERSION, compute_features, definition_versions, feature_names
from assay.features.compute import parse_time
from assay.ingestion.repo import DuplicateError
from assay.policy import Action, PolicyConfig, PolicyInput, evaluate
from assay.trust import Component, TrustConfig
from assay.trust.assessor import CaseAssessment, TrustAssessor
from assay.trust.explain import ExplainConfig

SCHEMA = "dec-1"


class ScoringError(Exception):
    pass


@dataclass
class LoadedBundle:
    manifest: BundleManifest
    model: Any
    reference: Any
    store: Any
    assessor: TrustAssessor
    n_features: int


class BundleRegistry:
    """Per-tenant champion and historical bundles. Replay needs the exact bundle that scored."""

    def __init__(self, trust_cfg: TrustConfig | None = None, explain_cfg: ExplainConfig | None = None):
        self._bundles: dict[tuple[str, str], LoadedBundle] = {}
        self._champion: dict[str, str] = {}
        self.trust_cfg = trust_cfg or TrustConfig()
        self.explain_cfg = explain_cfg or ExplainConfig()

    def register(self, tenant_id: str, artefact: dict, manifest: BundleManifest, *,
                 champion: bool = True) -> LoadedBundle:
        if manifest.tenant_id != tenant_id:
            raise ScoringError("bundle belongs to a different tenant")  # PRD 15.3
        model, ref, store = artefact["model"], artefact["reference"], artefact["store"]
        assessor = TrustAssessor(model, ref, store, manifest.thresholds,
                                 model_version=manifest.bundle_id, trust_cfg=self.trust_cfg,
                                 explain_cfg=self.explain_cfg)
        lb = LoadedBundle(manifest, model, ref, store, assessor, len(manifest.feature_names))
        self._bundles[(tenant_id, manifest.bundle_id)] = lb
        if champion:
            self._champion[tenant_id] = manifest.bundle_id
        return lb

    def champion(self, tenant_id: str) -> LoadedBundle:
        bid = self._champion.get(tenant_id)
        if bid is None:
            raise ScoringError("no champion bundle for tenant")
        return self._bundles[(tenant_id, bid)]

    def get(self, tenant_id: str, bundle_id: str) -> LoadedBundle:
        try:
            return self._bundles[(tenant_id, bundle_id)]
        except KeyError:
            raise ScoringError(f"bundle {bundle_id} not loaded for tenant") from None


@dataclass
class ScoringConfig:
    policy_version: str = "policy-0"
    audit_explain_pct: int = 5  # deterministic audit sample of low-risk cases (OPD-7 default)
    source_health: float = 1.0
    hard_rule: Callable[[dict], Action | None] | None = None  # institution hard rules (PRD 10.2)
    repro_tol: float = 1e-9
    clock: Callable[[], datetime] = field(default=lambda: datetime.now(UTC))


def ensure_bundle_row(repo, tenant_id: str, m: BundleManifest, actor: str) -> None:
    """Record the model bundle in lineage once, so predictions and reports can reference it."""
    if not repo.find(tenant_id, "model_bundles", {"bundle_id": m.bundle_id}):
        repo.insert(tenant_id, "model_bundles", {
            "schema_version": SCHEMA, "bundle_id": m.bundle_id, "model_version": m.bundle_id,
            "dataset_id": m.dataset_id, "code_commit": m.code_commit, "status": m.status,
            "artifact_sha256": m.artifact_sha256 or "unsigned", "manifest": asdict(m)}, actor)


def _f(x) -> float:
    return float(x)


def _comp_json(c: Component) -> dict:
    return {"status": c.status.value, "score": None if c.score is None else _f(c.score),
            "lo": None if c.lo is None else _f(c.lo), "hi": None if c.hi is None else _f(c.hi),
            "n": int(c.evidence_count)}


class ScoringService:
    def __init__(self, repo, registry: BundleRegistry, cfg: ScoringConfig | None = None):
        self.repo, self.registry, self.cfg = repo, registry, cfg or ScoringConfig()
        self.drift: dict[str, np.ndarray] = {}  # latest per-feature drift per tenant (population job)
        # Called with (tenant_id, policy_decision_row, case_assessment) for every stored decision,
        # inside the same transaction. The review workflow uses it to enqueue cases.
        self.decision_hooks: list[Callable[[str, dict, Any], None]] = []

    def _after_decision(self, tenant_id: str, pd: dict, ca) -> None:
        for hook in self.decision_hooks:
            hook(tenant_id, pd, ca)

    # ------------------------------------------------------------------ scoring --------------
    def should_explain(self, txn_id: str, risk: float, t_low: float) -> bool:
        return risk >= t_low or zlib.crc32(txn_id.encode()) % 100 < self.cfg.audit_explain_pct

    def score_transaction(self, tenant_id: str, txn_id: str, actor: str = "system") -> dict:
        existing = self.repo.find(tenant_id, "policy_decisions", {"txn_id": txn_id},
                                  newest_first=True, limit=1)
        if existing:  # idempotent: a transaction is scored once
            return self._view(tenant_id, existing[0])
        txn = self.repo.get_transaction(tenant_id, txn_id)
        if txn is None:
            raise ScoringError("unknown transaction")
        try:
            with self.repo.atomic(tenant_id):
                return self._score(tenant_id, txn, actor)
        except DuplicateError:  # a concurrent scorer won; answer from what is stored
            existing = self.repo.find(tenant_id, "policy_decisions", {"txn_id": txn_id},
                                      newest_first=True, limit=1)
            if existing:
                return self._view(tenant_id, existing[0])
            raise

    def _ensure_bundle_row(self, tenant_id: str, lb: LoadedBundle, actor: str) -> None:
        ensure_bundle_row(self.repo, tenant_id, lb.manifest, actor)

    def _score(self, tenant_id: str, txn: dict, actor: str) -> dict:
        lb = self.registry.champion(tenant_id)
        now = self.cfg.clock()
        self._ensure_bundle_row(tenant_id, lb, actor)
        history = self.repo.history_for_features(tenant_id, txn)
        feats = compute_features(txn, history, as_of=now)
        names = feature_names()
        x = np.array([[feats[n] for n in names]])
        fv = self.repo.insert(tenant_id, "feature_vectors", {
            "schema_version": SCHEMA, "txn_id": txn["txn_id"],
            "feature_set_version": FEATURE_SET_VERSION,
            "definition_versions": definition_versions(), "as_of_time": now,
            "feature_names": list(names), "feature_values": [_f(v) for v in x[0]],
            "graph_snapshot_id": None}, actor)

        t_high, t_low = lb.manifest.thresholds["t_high"], lb.manifest.thresholds["t_low"]
        pred = lb.model.predict(x, t_high)
        pr = self.repo.insert(tenant_id, "predictions", {
            "schema_version": SCHEMA, "txn_id": txn["txn_id"], "feature_vector_id": fv["id"],
            "bundle_id": lb.manifest.bundle_id, "raw_score": _f(pred.raw[0]),
            "calibrated_risk": _f(pred.calibrated[0]), "member_spread": _f(pred.member_spread[0]),
            "distance_to_threshold": _f(pred.distance_to_threshold[0])}, actor)

        explain = self.should_explain(txn["txn_id"], _f(pred.calibrated[0]), t_low)
        drift = self.drift.get(tenant_id, np.zeros(x.shape[1]))
        ca = lb.assessor.assess(x, [txn], pred, drift_vector=drift, explain_mask=np.array([explain]),
                                source_health=self.cfg.source_health, recorded_at=[txn["recorded_at"]])[0]
        if explain:
            self._store_explanation(tenant_id, pr["id"], lb, ca, actor)
        ta = self._store_assessment(tenant_id, pr["id"], 1, ca, lb, drift, txn, explain, actor)
        pd = self._store_policy(tenant_id, txn, ta, pred.calibrated[0], ca, lb, actor)
        self._after_decision(tenant_id, pd, ca)
        self.repo.append_audit(tenant_id, actor, "score", txn["txn_id"], ca.result.state.value, now)
        return self._view(tenant_id, pd)

    def _store_explanation(self, tenant_id, prediction_id, lb, ca: CaseAssessment, actor):
        e = ca.explanation
        self.repo.insert(tenant_id, "explanations", {
            "schema_version": SCHEMA, "prediction_id": prediction_id, "method": "treeshap-groups",
            "params": asdict(lb.assessor.explain_cfg), "background_version": lb.manifest.bundle_id,
            "seed": int(lb.assessor.explain_cfg.seed), "attributions": e["group_attributions"],
            "stability": _nan_none(e["stability"]), "sensitivity": _nan_none(e["sensitivity"]),
            "faithfulness": _nan_none(e["faithfulness"]), "reproducible": e["reproducible"]}, actor)

    def _store_assessment(self, tenant_id, prediction_id, version, ca: CaseAssessment, lb, drift,
                          txn, explain: bool, actor):
        r = ca.result
        return self.repo.insert(tenant_id, "trust_assessments", {
            "schema_version": SCHEMA, "prediction_id": prediction_id, "version_no": version,
            "mode": r.mode, "state": r.state.value,
            "ti": None if r.ti is None else _f(r.ti),
            "ti_low": None if r.ti_low is None else _f(r.ti_low),
            "ti_high": None if r.ti_high is None else _f(r.ti_high),
            "reason_codes": [c.value for c in r.reason_codes],
            "components": {k: _comp_json(c) for k, c in ca.components.items()},
            "weights_version": r.config_version, "weights_used": {k: _f(v) for k, v in r.weights_used.items()},
            "evidence": {"drift_vector": [_f(v) for v in drift], "explain": explain,
                         "recorded_at": _iso(txn["recorded_at"]), "source_health": self.cfg.source_health,
                         "novelty": _f(ca.novelty), "data_quality": {k: _f(v) for k, v in ca.data_quality.items()},
                         "cohort": asdict(ca.cohort), "bundle_id": lb.manifest.bundle_id}}, actor)

    def _store_policy(self, tenant_id, txn, ta, risk, ca: CaseAssessment, lb, actor):
        pcfg = PolicyConfig(version=self.cfg.policy_version, t_low=lb.manifest.thresholds["t_low"],
                            t_high=lb.manifest.thresholds["t_high"])
        hard = self.cfg.hard_rule(txn) if self.cfg.hard_rule else None
        res = evaluate(PolicyInput(_f(risk), ca.result.state, ca.result.reason_codes, hard), pcfg)
        return self.repo.insert(tenant_id, "policy_decisions", {
            "schema_version": SCHEMA, "txn_id": txn["txn_id"], "trust_assessment_id": ta["id"],
            "policy_version": res.policy_version, "risk_band": res.risk_band.value, "gate": res.gate,
            "queue": res.queue, "recommended_action": res.action.value,
            "automation_level": res.automation_level}, actor)

    # ------------------------------------------------------------------ refinement ------------
    def refine(self, tenant_id: str, decision_id: str, actor: str = "system") -> dict:
        """Compute the explanation-dependent components and store trust assessment v+1 and a new
        policy decision beside the first. Nothing is overwritten (PRD 5.8)."""
        pd = self._decision_row(tenant_id, decision_id)
        ta = self.repo.get_by_id(tenant_id, "trust_assessments", pd["trust_assessment_id"])
        if ta["evidence"]["explain"]:
            return self._view(tenant_id, pd)  # already has an explanation
        pr = self.repo.get_by_id(tenant_id, "predictions", ta["prediction_id"])
        lb = self.registry.get(tenant_id, pr["bundle_id"])
        txn = self.repo.get_transaction(tenant_id, pr["txn_id"])
        fv = self.repo.get_by_id(tenant_id, "feature_vectors", pr["feature_vector_id"])
        x = np.array([fv["feature_values"]])
        pred = lb.model.predict(x, lb.manifest.thresholds["t_high"])
        drift = np.array(ta["evidence"]["drift_vector"])
        with self.repo.atomic(tenant_id):
            ca = lb.assessor.assess(x, [txn], pred, drift_vector=drift, explain_mask=np.array([True]),
                                    source_health=ta["evidence"]["source_health"],
                                    recorded_at=[parse_time(ta["evidence"]["recorded_at"])])[0]
            self._store_explanation(tenant_id, pr["id"], lb, ca, actor)
            latest = max(r["version_no"] for r in
                         self.repo.find(tenant_id, "trust_assessments", {"prediction_id": pr["id"]}))
            ta2 = self._store_assessment(tenant_id, pr["id"], latest + 1, ca, lb, drift, txn, True, actor)
            pd2 = self._store_policy(tenant_id, txn, ta2, pr["calibrated_risk"], ca, lb, actor)
            self._after_decision(tenant_id, pd2, ca)
        return self._view(tenant_id, pd2)

    def refine_pending(self, tenant_id: str, limit: int = 100, actor: str = "worker") -> int:
        """The asynchronous worker (PRD 5.8): complete explanation testing for decisions that were
        scored without it, storing a new trust-assessment version beside the first. Returns how many
        were refined. Oldest first; run it on a schedule or after scoring."""
        done = 0
        latest: dict[str, dict] = {}
        for pd in self.repo.find(tenant_id, "policy_decisions"):
            latest[pd["txn_id"]] = pd  # later rows overwrite earlier ones: newest per transaction
        for pd in latest.values():
            if done >= limit:
                break
            ta = self.repo.get_by_id(tenant_id, "trust_assessments", pd["trust_assessment_id"])
            if ta["evidence"]["explain"]:
                continue
            self.refine(tenant_id, pd["id"], actor)
            done += 1
        return done

    # ------------------------------------------------------------------ reading ---------------
    def _decision_row(self, tenant_id: str, decision_id: str) -> dict:
        pd = self.repo.get_by_id(tenant_id, "policy_decisions", decision_id)
        if pd is None:
            raise ScoringError("unknown decision")
        return pd

    def get_decision(self, tenant_id: str, decision_id: str) -> dict:
        return self._view(tenant_id, self._decision_row(tenant_id, decision_id))

    def _view(self, tenant_id: str, pd: dict) -> dict:
        ta = self.repo.get_by_id(tenant_id, "trust_assessments", pd["trust_assessment_id"])
        pr = self.repo.get_by_id(tenant_id, "predictions", ta["prediction_id"])
        has_expl = bool(self.repo.find(tenant_id, "explanations", {"prediction_id": pr["id"]}, limit=1))
        return {
            "decision_id": pd["id"], "txn_id": pd["txn_id"], "risk": pr["calibrated_risk"],
            "risk_band": pd["risk_band"],
            "trust": {"state": ta["state"], "mode": ta["mode"], "ti": ta["ti"], "ti_low": ta["ti_low"],
                      "ti_high": ta["ti_high"], "reason_codes": ta["reason_codes"],
                      "components": ta["components"], "version_no": ta["version_no"]},
            "recommendation": pd["recommended_action"], "gate": pd["gate"], "queue": pd["queue"],
            "automation_level": pd["automation_level"], "bundle_id": pr["bundle_id"],
            "policy_version": pd["policy_version"],
            "explanation_status": "computed" if has_expl else "pending",
        }

    def explanation(self, tenant_id: str, decision_id: str) -> dict | None:
        pd = self._decision_row(tenant_id, decision_id)
        ta = self.repo.get_by_id(tenant_id, "trust_assessments", pd["trust_assessment_id"])
        rows = self.repo.find(tenant_id, "explanations", {"prediction_id": ta["prediction_id"]},
                              newest_first=True, limit=1)
        if not rows:
            return None
        e = rows[0]
        return {k: e[k] for k in ("method", "params", "background_version", "seed", "attributions",
                                  "stability", "sensitivity", "faithfulness", "reproducible")}

    # ------------------------------------------------------------------ governance reads ------
    def validation_reports(self, tenant_id: str, limit: int = 5) -> list[dict]:
        """Newest first. Stored validation reports, each linked to its bundle and dataset (FR-39)."""
        return self.repo.find(tenant_id, "validation_reports", newest_first=True, limit=limit)

    def model_bundles(self, tenant_id: str) -> list[dict]:
        """Recorded bundles with their champion flag. The artefact itself is never exposed."""
        champion = self.registry._champion.get(tenant_id)
        rows = self.repo.find(tenant_id, "model_bundles", newest_first=True)
        return [{**{k: r[k] for k in ("bundle_id", "dataset_id", "code_commit", "status", "artifact_sha256")},
                 "champion": r["bundle_id"] == champion, "thresholds": r["manifest"].get("thresholds"),
                 "metrics": r["manifest"].get("metrics"), "calibration": r["manifest"].get("calibration"),
                 "created_at": r["manifest"].get("created_at"),
                 "feature_set_version": r["manifest"].get("feature_set_version")} for r in rows]

    def lineage(self, tenant_id: str, decision_id: str) -> dict:
        """Everything linked to a decision (PRD 14.2 backward impact query)."""
        pd = self._decision_row(tenant_id, decision_id)
        txn = self.repo.get_transaction(tenant_id, pd["txn_id"])
        ta = self.repo.get_by_id(tenant_id, "trust_assessments", pd["trust_assessment_id"])
        pr = self.repo.get_by_id(tenant_id, "predictions", ta["prediction_id"])
        return {
            "transaction": txn,
            "ingestion_event": self.repo.get_by_id(tenant_id, "ingestion_events", txn["ingestion_event_id"]),
            "feature_vector": self.repo.get_by_id(tenant_id, "feature_vectors", pr["feature_vector_id"]),
            "model_bundle": (self.repo.find(tenant_id, "model_bundles", {"bundle_id": pr["bundle_id"]}) or [None])[0],
            "prediction": pr,
            "explanations": self.repo.find(tenant_id, "explanations", {"prediction_id": pr["id"]}),
            "trust_assessments": self.repo.find(tenant_id, "trust_assessments", {"prediction_id": pr["id"]}),
            "policy_decisions": self.repo.find(tenant_id, "policy_decisions", {"txn_id": pr["txn_id"]}),
        }

    # ------------------------------------------------------------------ replay ----------------
    def replay(self, tenant_id: str, decision_id: str, actor: str = "system") -> dict:
        """Reproduce prediction and Trust Index from stored versions (FR-33). A decision that
        does not reproduce is flagged non-reproducible in the audit log."""
        pd = self._decision_row(tenant_id, decision_id)
        ta = self.repo.get_by_id(tenant_id, "trust_assessments", pd["trust_assessment_id"])
        pr = self.repo.get_by_id(tenant_id, "predictions", ta["prediction_id"])
        fv = self.repo.get_by_id(tenant_id, "feature_vectors", pr["feature_vector_id"])
        txn = self.repo.get_transaction(tenant_id, pr["txn_id"])
        tol = self.cfg.repro_tol
        checks: dict[str, bool] = {}
        try:
            lb = self.registry.get(tenant_id, pr["bundle_id"])
            x = np.array([fv["feature_values"]])
            pred = lb.model.predict(x, lb.manifest.thresholds["t_high"])
            checks["raw_score"] = abs(_f(pred.raw[0]) - pr["raw_score"]) <= tol
            checks["calibrated_risk"] = abs(_f(pred.calibrated[0]) - pr["calibrated_risk"]) <= tol
            ev = ta["evidence"]
            ca = lb.assessor.assess(x, [txn], pred, drift_vector=np.array(ev["drift_vector"]),
                                    explain_mask=np.array([ev["explain"]]),
                                    source_health=ev["source_health"],
                                    recorded_at=[parse_time(ev["recorded_at"])])[0]
            r = ca.result
            checks["state"] = r.state.value == ta["state"]
            checks["reason_codes"] = [c.value for c in r.reason_codes] == ta["reason_codes"]
            checks["ti"] = (r.ti is None) == (ta["ti"] is None) and (
                r.ti is None or abs(_f(r.ti) - ta["ti"]) <= 1e-6)
        except Exception as e:  # noqa: BLE001  any failure to replay means "not reproducible"
            checks["replay_error"] = False
            self.repo.append_audit(tenant_id, actor, "decision_non_reproducible", decision_id,
                                   type(e).__name__, self.cfg.clock())
            return {"decision_id": decision_id, "reproducible": False, "checks": checks}
        ok = all(checks.values())
        if not ok:
            self.repo.append_audit(tenant_id, actor, "decision_non_reproducible", decision_id,
                                   ",".join(k for k, v in checks.items() if not v), self.cfg.clock())
        return {"decision_id": decision_id, "reproducible": ok, "checks": checks}


def _nan_none(v):
    return None if v is None or math.isnan(float(v)) else float(v)


def _iso(v) -> str:
    return v.isoformat() if isinstance(v, datetime) else str(v)


__all__ = ["BundleRegistry", "LoadedBundle", "ScoringConfig", "ScoringError", "ScoringService"]
