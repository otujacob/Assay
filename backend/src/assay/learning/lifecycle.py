"""The life of a candidate model (PRD 12.2): validated -> shadow -> approved -> canary -> champion, with rollback.

Nothing here promotes a model automatically. Every step is a person's recorded act (PRD 12.1: "detection model
weights are never promoted automatically"), stored as an append-only row, so the history is the order of the
events and nothing is updated. The rules:

  * validated: the gate report (learning/gates.py) had no failed gate. A candidate with a failed gate stops.
  * shadow:    the candidate scores live traffic and takes no action. One candidate at a time per tenant.
  * approved:  needs shadow evidence (G7: long enough, enough cases, no alarm), a DIFFERENT person from the one
               who created the candidate (G9, also enforced by the database), and an explicit waiver for every gate
               that could not be judged. G6 (segment review) is never skipped: the approver confirms they reviewed it.
  * canary:    a share of live traffic is decided by the candidate (hash of the transaction id, so the same
               transaction always takes the same path). The share is capped.
  * champion:  promoted after the canary has decided enough cases. The previous champion stays deployable.
  * rollback:  restores the previous champion, at any point after canary starts, and records why.

The in-memory registry follows these records: `sync` points it at whatever the stored events say, so every worker
agrees, whichever one handled the request.
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from .status import FAIL, NOT_EVALUATED, REVIEW

SCHEMA = "lc-1"
KIND_STATE = {"validated": "validated", "validation_failed": "failed", "shadow_started": "shadow",
              "approved": "approved", "canary_started": "canary", "promoted": "champion",
              "rolled_back": "rolled_back", "rejected": "rejected"}
ALLOWED = {  # state -> event kinds that may follow
    "validated": {"shadow_started", "rejected"},
    "failed": {"rejected"},
    "shadow": {"approved", "rejected"},
    "approved": {"canary_started", "rejected"},
    "canary": {"promoted", "rolled_back", "rejected"},
    "champion": {"rolled_back"},
    "rolled_back": set(),
    "rejected": set(),
}
LIVE_STATES = frozenset({"shadow", "approved", "canary", "champion"})


class LifecycleError(Exception):
    def __init__(self, code: str, detail: str, http: int = 409):
        super().__init__(detail)
        self.code, self.http = code, http


@dataclass(frozen=True)
class LifecycleConfig:
    min_shadow_days: float = 7.0
    min_shadow_cases: int = 200
    max_call_rate_shift: float = 0.05  # |candidate flag rate - champion flag rate|
    max_mean_risk_shift: float = 0.05  # |mean candidate risk - mean champion risk|
    min_canary_cases: int = 100
    max_canary_share: float = 0.5
    canary_salt: str = "canary-v1"
    clock: Callable[[], datetime] = field(default=lambda: datetime.now(UTC))


@dataclass
class Routing:
    """Which bundle decides, and which one only watches."""
    champion: str | None = None  # None: keep the bundle the process was started with
    canary: str | None = None
    canary_share: float = 0.0
    canary_salt: str = "canary-v1"
    shadow: str | None = None
    previous: dict[str, str | None] = field(default_factory=dict)


def in_canary(txn_id: str, salt: str, share: float) -> bool:
    h = int(hashlib.sha256(f"{salt}:{txn_id}".encode()).hexdigest()[:12], 16) / 16**12
    return h < share


class LifecycleStore:
    def __init__(self, repo, cfg: LifecycleConfig | None = None):
        self.repo, self.cfg = repo, cfg or LifecycleConfig()

    # ------------------------------------------------------------------- reads --------------
    def candidates(self, tenant_id: str) -> list[dict]:
        return self.repo.find(tenant_id, "model_candidates")

    def events(self, tenant_id: str, candidate_id: str | None = None) -> list[dict]:
        where = {"candidate_id": candidate_id} if candidate_id else None
        return self.repo.find(tenant_id, "model_lifecycle_events", where)

    def candidate(self, tenant_id: str, candidate_id: str) -> dict:
        rows = self.repo.find(tenant_id, "model_candidates", {"candidate_id": candidate_id})
        if not rows:
            raise LifecycleError("not_found", "unknown candidate", 404)
        return rows[0]

    def state_of(self, tenant_id: str, candidate_id: str) -> str:
        evs = self.events(tenant_id, candidate_id)
        return KIND_STATE[evs[-1]["kind"]] if evs else "candidate"

    def summary(self, tenant_id: str, candidate_id: str) -> dict:
        c, evs = self.candidate(tenant_id, candidate_id), self.events(tenant_id, candidate_id)
        return {"candidate_id": candidate_id, "base_bundle_id": c["base_bundle_id"], "created_by": c["created_by"],
                "state": KIND_STATE[evs[-1]["kind"]] if evs else "candidate",
                "pool": c["pool_summary"], "training": c["training"], "gates": c["gates"],
                "events": [{"kind": e["kind"], "actor": e["created_by"], **e["detail"]} for e in evs]}

    def routing(self, tenant_id: str) -> Routing:
        r, stack = Routing(canary_salt=self.cfg.canary_salt), []
        evs = self.events(tenant_id)
        latest: dict[str, dict] = {}
        for e in evs:
            cid, k, d = e["candidate_id"], e["kind"], e["detail"]
            latest[cid] = e
            if k == "promoted":
                r.previous[cid] = d.get("previous")
                stack.append(cid)
            elif k == "rolled_back" and cid in stack:
                stack.remove(cid)
        r.champion = stack[-1] if stack else None
        for cid, e in latest.items():
            if e["kind"] == "canary_started":
                r.canary, r.canary_share = cid, float(e["detail"]["share"])
            elif e["kind"] == "shadow_started":
                r.shadow = cid
        return r

    # ------------------------------------------------------------------- writes -------------
    def create(self, tenant_id: str, actor: str, *, candidate_id: str, base_bundle_id: str,
               artefact_path: str | None, pool_summary: dict, training: dict, gates: dict) -> dict:
        row = self.repo.insert(tenant_id, "model_candidates", {
            "schema_version": SCHEMA, "candidate_id": candidate_id, "base_bundle_id": base_bundle_id,
            "artefact_path": artefact_path, "pool_summary": pool_summary, "training": training,
            "gates": gates}, actor)
        failed = gates.get("failed") or []
        self._event(tenant_id, actor, row, "validation_failed" if failed else "validated",
                    {"failed_gates": failed, "unresolved_gates": gates.get("unresolved") or []})
        return self.summary(tenant_id, candidate_id)

    def _event(self, tenant_id: str, actor: str, cand: dict, kind: str, detail: dict) -> dict:
        detail = {"at": self.cfg.clock().isoformat(), **detail}
        return self.repo.insert(tenant_id, "model_lifecycle_events", {
            "schema_version": SCHEMA, "candidate_id": cand["candidate_id"],
            "created_by_candidate": cand["created_by"], "kind": kind, "detail": detail}, actor)

    def _step(self, tenant_id: str, actor: str, candidate_id: str, kind: str) -> tuple[dict, str]:
        cand = self.candidate(tenant_id, candidate_id)
        st = self.state_of(tenant_id, candidate_id)
        if kind not in ALLOWED.get(st, set()):
            raise LifecycleError("bad_transition", f"a candidate that is {st} cannot go to {kind}")
        return cand, st

    def reject(self, tenant_id: str, actor: str, candidate_id: str, reason: str) -> dict:
        if not reason:
            raise LifecycleError("reason_required", "rejecting needs a stored reason", 422)
        cand, _ = self._step(tenant_id, actor, candidate_id, "rejected")
        self._event(tenant_id, actor, cand, "rejected", {"reason": reason})
        return self.summary(tenant_id, candidate_id)

    def start_shadow(self, tenant_id: str, actor: str, candidate_id: str) -> dict:
        cand, _ = self._step(tenant_id, actor, candidate_id, "shadow_started")
        other = self.routing(tenant_id).shadow
        if other and other != candidate_id:
            raise LifecycleError("shadow_busy", f"{other} is already in shadow; one at a time")
        self._event(tenant_id, actor, cand, "shadow_started", {})
        return self.summary(tenant_id, candidate_id)

    def shadow_report(self, tenant_id: str, candidate_id: str, *, outcomes: dict[str, int] | None = None) -> dict:
        cfg = self.cfg
        started = [e for e in self.events(tenant_id, candidate_id) if e["kind"] == "shadow_started"]
        if not started:
            return {"gate": "G7", "status": "pending", "detail": "not in shadow", "n": 0}
        since = datetime.fromisoformat(started[-1]["detail"]["at"])
        days = (cfg.clock() - since).total_seconds() / 86400
        rows = self.repo.find(tenant_id, "shadow_scores", {"candidate_id": candidate_id})
        n = len(rows)
        rep: dict[str, Any] = {"gate": "G7", "n": n, "days": days, "min_days": cfg.min_shadow_days,
                               "min_cases": cfg.min_shadow_cases}
        if n:
            cand_rate = sum(1 for r in rows if r["candidate_call"]) / n
            champ_rate = sum(1 for r in rows if r["champion_call"]) / n
            risk_shift = sum(r["candidate_risk"] - r["champion_risk"] for r in rows) / n
            rep |= {"flag_rate_candidate": cand_rate, "flag_rate_champion": champ_rate,
                    "agreement": sum(1 for r in rows if r["candidate_call"] == r["champion_call"]) / n,
                    "mean_risk_shift": risk_shift}
            alarms = []
            if abs(cand_rate - champ_rate) > cfg.max_call_rate_shift:
                alarms.append(f"flag rate differs from the champion's by {abs(cand_rate - champ_rate):.3f}")
            if abs(risk_shift) > cfg.max_mean_risk_shift:
                alarms.append(f"mean risk differs from the champion's by {abs(risk_shift):.3f}")
            rep["alarms"] = alarms
            if outcomes:
                judged = [(r, outcomes[r["txn_id"]]) for r in rows if r["txn_id"] in outcomes]
                pos = [(r, y) for r, y in judged if y == 1]
                rep["matured"] = {"n": len(judged), "frauds": len(pos),
                                  "recall_candidate": (sum(1 for r, _ in pos if r["candidate_call"]) / len(pos)) if pos else None,
                                  "recall_champion": (sum(1 for r, _ in pos if r["champion_call"]) / len(pos)) if pos else None,
                                  "note": "descriptive; matured shadow cases are few"}
        else:
            rep["alarms"] = []
        if days < cfg.min_shadow_days or n < cfg.min_shadow_cases:
            rep |= {"status": "pending", "detail": f"{n} cases over {days:.1f} days; needs "
                    f"{cfg.min_shadow_cases} over {cfg.min_shadow_days:g}"}
        elif rep["alarms"]:
            rep |= {"status": FAIL, "detail": "; ".join(rep["alarms"])}
        else:
            rep |= {"status": "pass", "detail": f"{n} cases over {days:.1f} days with no alarm"}
        return rep

    def approve(self, tenant_id: str, actor: str, candidate_id: str, *, waived: list[str] | None = None,
                reviewed_segments: bool = False, rationale: str = "",
                outcomes: dict[str, int] | None = None) -> dict:
        cand, _ = self._step(tenant_id, actor, candidate_id, "approved")
        if actor == cand["created_by"]:
            raise LifecycleError("separation_of_duties", "the person who created a candidate cannot approve it", 403)
        if not rationale:
            raise LifecycleError("reason_required", "an approval needs a stored rationale", 422)
        g7 = self.shadow_report(tenant_id, candidate_id, outcomes=outcomes)
        if g7["status"] != "pass":
            raise LifecycleError("shadow_not_passed", f"shadow evidence is not sufficient: {g7['detail']}")
        waived = sorted(set(waived or []))
        gates = {g["id"]: g for g in cand["gates"]["gates"]}
        need = {gid for gid, g in gates.items() if g["status"] == NOT_EVALUATED}
        if missing := sorted(need - set(waived)):
            raise LifecycleError("unresolved_gates", f"gates that could not be judged need an explicit waiver: {missing}", 422)
        if any(g["status"] == REVIEW for g in gates.values()) and not reviewed_segments:
            raise LifecycleError("segments_not_reviewed", "confirm that the results by segment were reviewed (G6)", 422)
        self._event(tenant_id, actor, cand, "approved", {
            "waived_gates": waived, "reviewed_segments": reviewed_segments, "rationale": rationale, "g7": g7})
        return self.summary(tenant_id, candidate_id)

    def start_canary(self, tenant_id: str, actor: str, candidate_id: str, share: float) -> dict:
        cand, _ = self._step(tenant_id, actor, candidate_id, "canary_started")
        if not (0 < share <= self.cfg.max_canary_share):
            raise LifecycleError("bad_share", f"canary share must be above 0 and at most {self.cfg.max_canary_share}", 422)
        if self.routing(tenant_id).canary:
            raise LifecycleError("canary_busy", "another candidate is already in canary")
        self._event(tenant_id, actor, cand, "canary_started", {"share": share})
        return self.summary(tenant_id, candidate_id)

    def canary_cases(self, tenant_id: str, candidate_id: str) -> int:
        return len(self.repo.find(tenant_id, "predictions", {"bundle_id": candidate_id}))

    def promote(self, tenant_id: str, actor: str, candidate_id: str, *, current_champion: str) -> dict:
        cand, _ = self._step(tenant_id, actor, candidate_id, "promoted")
        n = self.canary_cases(tenant_id, candidate_id)
        if n < self.cfg.min_canary_cases:
            raise LifecycleError("canary_too_small", f"the canary has decided {n} cases; needs {self.cfg.min_canary_cases}")
        approval = [e for e in self.events(tenant_id, candidate_id) if e["kind"] == "approved"][-1]
        self._event(tenant_id, actor, cand, "promoted", {
            "approval_event": approval["id"], "previous": current_champion, "canary_cases": n})
        return self.summary(tenant_id, candidate_id)

    def rollback(self, tenant_id: str, actor: str, candidate_id: str, reason: str) -> dict:
        if not reason:
            raise LifecycleError("reason_required", "a rollback needs a stored reason", 422)
        cand, _ = self._step(tenant_id, actor, candidate_id, "rolled_back")
        self._event(tenant_id, actor, cand, "rolled_back", {"reason": reason})
        return self.summary(tenant_id, candidate_id)
