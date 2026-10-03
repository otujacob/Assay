"""Human review workflow (PRD 8, 10.5, 11; FR-24 to FR-30, FR-35).

Everything here is append-only: a case's status is DERIVED from its analyst actions, never stored
and updated. Every analyst decision is kept (PRD 8.4). Recommend-only: no action is ever taken by
the system (FR-23); analysts decide.

Blind review (PRD 8.3, FR-27, FR-28): whether a case is blind is decided ONCE, deterministically,
when it enters the queue, and recorded. The server withholds the score and recommendation from a
blind case's analyst and records what was actually shown. The client cannot ask for more.
"""

from __future__ import annotations

import math
import zlib
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import Any

from assay.ingestion.repo import DuplicateError
from assay.scoring.service import ScoringError
from assay.trust.reliability import wilson_interval

from . import feedback as fb

SCHEMA = "rev-1"
REVIEW_ACTIONS = frozenset({"request_human_review", "request_human_review_priority", "escalate", "hold"})
ANALYST_ACTIONS = frozenset({"approve", "block", "escalate", "request_review", "override", "unsure"})
FINAL = frozenset({"approve", "block"})
TRUST_UNRELIABILITY = {"insufficient_evidence": 1.0, "low": 0.8, "moderate": 0.4, "high": 0.1}
OPEN_STATES = frozenset({"open", "escalated", "conflicted"})
# "superseded": a refined decision no longer recommends review and nobody has acted, so the case is moot.


class ReviewError(Exception):
    def __init__(self, code: str, message: str, http: int = 400):
        super().__init__(message)
        self.code, self.http = code, http


@dataclass
class ReviewConfig:
    blind_share: float = 0.10  # OPD-8 working default
    blind_salt: str = "blind-v1"
    # SLA minutes per queue (tenant-configurable, PRD 10.4)
    sla_minutes: dict[str, int] = field(default_factory=lambda: {
        "standard": 480, "priority": 90, "escalation": 120, "novelty": 240, "override": 480})
    # Priority = weighted mix of risk, amount, trust unreliability and age (PRD 10.5). Parameters.
    w_risk: float = 0.40
    w_amount: float = 0.25
    w_trust: float = 0.20
    w_age: float = 0.15
    amount_cap: float = 10_000.0
    p_thresholds: tuple[float, float, float] = (0.75, 0.55, 0.35)  # P1, P2, P3 lower bounds
    feedback: fb.FeedbackConfig = field(default_factory=fb.FeedbackConfig)
    drift_alarm: float = 0.10
    clock: Callable[[], datetime] = field(default=lambda: datetime.now(UTC))


def _dt(v) -> datetime:
    return v if isinstance(v, datetime) else datetime.fromisoformat(str(v))


def _public(view: dict) -> dict:
    """Drop internal fields before a view leaves the service."""
    return {k: v for k, v in view.items() if not k.startswith("_")}


def _mask(pid: str | None) -> str | None:
    return None if pid is None else (pid[:3] + "…" + pid[-2:] if len(pid) > 6 else "…")


def blind_assignment(tenant_id: str, txn_id: str, share: float, salt: str) -> bool:
    """Deterministic pseudo-random assignment: reproducible, independent of who looks, and
    uniformly distributed so the realised share matches the configured share (FR-28)."""
    h = zlib.crc32(f"{salt}:{tenant_id}:{txn_id}".encode()) / 2**32
    return h < share


def _family(action: str | None) -> str | None:
    """Recommendation -> approve / block / None (review and escalate are not agreeable)."""
    if action in ("approve", "approve_sampled_qa"):
        return "approve"
    return "block" if action == "block" else None


class ReviewService:
    def __init__(self, repo, scoring, cfg: ReviewConfig | None = None):
        self.repo, self.scoring, self.cfg = repo, scoring, cfg or ReviewConfig()
        scoring.decision_hooks.append(self.on_decision)

    # -------------------------------------------------------------------- enqueue -------------
    def on_decision(self, tenant_id: str, pd: dict, ca=None) -> None:
        """Hook called by the scoring service for every stored policy decision."""
        if pd["recommended_action"] not in REVIEW_ACTIONS:
            return
        if self.repo.find(tenant_id, "review_cases", {"txn_id": pd["txn_id"]}, limit=1):
            return  # already queued: the blind flag is decided once
        queue = ("novelty" if pd["gate"] == "novelty" else "escalation" if pd["recommended_action"] == "escalate"
                 else "priority" if pd["recommended_action"] == "request_human_review_priority" else "standard")
        try:
            self.repo.insert(tenant_id, "review_cases", {
                "schema_version": SCHEMA, "txn_id": pd["txn_id"], "first_decision_id": pd["id"],
                "queue": queue, "blind": blind_assignment(tenant_id, pd["txn_id"], self.cfg.blind_share,
                                                           self.cfg.blind_salt),
                "sla_minutes": self.cfg.sla_minutes[queue], "enqueued_at": self.cfg.clock()}, "review")
        except DuplicateError:
            pass  # a concurrent scorer queued it first

    # -------------------------------------------------------------------- derived state --------
    def _actions(self, tenant_id: str, txn_id: str) -> list[dict]:
        return self.repo.find(tenant_id, "analyst_actions", {"txn_id": txn_id})

    @staticmethod
    def status_of(actions: list[dict]) -> tuple[str, str | None]:
        """(status, final decision). open -> escalated/decided/conflicted -> adjudicated."""
        adj = [a for a in actions if a["action"] == "adjudicate"]
        if adj:
            return "adjudicated", adj[-1]["final_decision"]
        finals = [a for a in actions if a["final_decision"]]
        if len({a["final_decision"] for a in finals}) > 1:
            return "conflicted", None
        if finals:
            return "decided", finals[-1]["final_decision"]
        if any(a["action"] == "escalate" for a in actions):
            return "escalated", None
        return "open", None

    @staticmethod
    def _effective_status(status: str, ld: dict, actions: list[dict]) -> str:
        if status == "open" and not actions and ld["pd"]["recommended_action"] not in REVIEW_ACTIONS:
            return "superseded"
        return status

    def _outcome(self, tenant_id: str, txn_id: str) -> dict | None:
        """Matured verified outcome, if any. A verified outcome overrides any analyst decision as
        the label (PRD 8.4)."""
        rows = [o for o in self.repo.find(tenant_id, "outcomes", {"txn_id": txn_id})
                if o["maturity_state"] == "matured"]
        if not rows:
            return None
        fraud = any(o["outcome_type"] in ("confirmed_fraud", "chargeback") for o in rows)
        return {"verified": "fraud" if fraud else "legitimate", "type": rows[-1]["outcome_type"]}

    # -------------------------------------------------------------------- priority & SLA -------
    def _load(self, tenant_id: str, txn_id: str) -> dict:
        pd = self.repo.find(tenant_id, "policy_decisions", {"txn_id": txn_id}, newest_first=True, limit=1)[0]
        ta = self.repo.get_by_id(tenant_id, "trust_assessments", pd["trust_assessment_id"])
        pr = self.repo.get_by_id(tenant_id, "predictions", ta["prediction_id"])
        txn = self.repo.get_transaction(tenant_id, txn_id)
        return {"pd": pd, "ta": ta, "pr": pr, "txn": txn}

    def _sla(self, case: dict, now: datetime) -> dict:
        due = _dt(case["enqueued_at"]) + timedelta(minutes=case["sla_minutes"])
        left = (due - now).total_seconds()
        return {"minutes": case["sla_minutes"], "due_at": due.isoformat(), "seconds_left": left,
                "breached": left < 0}

    def _priority(self, loaded: dict, case: dict, now: datetime) -> dict:
        cfg = self.cfg
        t_high = self.scoring.registry.get(case["tenant_id"],
                                           loaded["pr"]["bundle_id"]).manifest.thresholds["t_high"]
        risk = min(1.0, loaded["pr"]["calibrated_risk"] / t_high) if t_high > 0 else 0.0
        amount = min(1.0, math.log1p(float(loaded["txn"]["amount"])) / math.log1p(cfg.amount_cap))
        trust = TRUST_UNRELIABILITY[loaded["ta"]["state"]]
        age = min(1.0, max(0.0, (now - _dt(case["enqueued_at"])).total_seconds() / 60.0 / case["sla_minutes"]))
        score = cfg.w_risk * risk + cfg.w_amount * amount + cfg.w_trust * trust + cfg.w_age * age
        p1, p2, p3 = cfg.p_thresholds
        return {"score": round(score, 4), "label": "P1" if score >= p1 else "P2" if score >= p2
                else "P3" if score >= p3 else "P4"}

    # -------------------------------------------------------------------- queue ---------------
    def queue(self, tenant_id: str, *, viewer_role: str = "analyst", risk_band: str | None = None,
              trust_state: str | None = None, reason_code: str | None = None, search: str | None = None,
              queue: str | None = None, include_closed: bool = False, limit: int = 200) -> dict:
        now = self.cfg.clock()
        rows, total = [], 0
        for case in self.repo.find(tenant_id, "review_cases"):
            total += 1
            actions = self._actions(tenant_id, case["txn_id"])
            status, _ = self.status_of(actions)
            ld = self._load(tenant_id, case["txn_id"])
            status = self._effective_status(status, ld, actions)
            if status not in OPEN_STATES and not include_closed:
                continue
            if queue and case["queue"] != queue:
                continue
            if search and search.lower() not in case["txn_id"].lower():
                continue
            if risk_band and ld["pd"]["risk_band"] != risk_band:
                continue
            if trust_state and ld["ta"]["state"] != trust_state:
                continue
            if reason_code and reason_code not in ld["ta"]["reason_codes"]:
                continue
            blind_hidden = case["blind"] and viewer_role == "analyst" and not actions
            pri = self._priority(ld, case, now)
            rows.append({
                "decision_id": ld["pd"]["id"], "txn_id": case["txn_id"], "status": status,
                "queue": case["queue"], "priority": pri["label"], "priority_score": pri["score"],
                "blind": case["blind"], "redacted": blind_hidden,
                "risk_band": None if blind_hidden else ld["pd"]["risk_band"],
                "trust_state": None if blind_hidden else ld["ta"]["state"],
                "reason_codes": [] if blind_hidden else ld["ta"]["reason_codes"],
                "amount": float(ld["txn"]["amount"]), "currency": ld["txn"]["currency"],
                "event_time": _dt(ld["txn"]["event_time"]).isoformat(),
                "sla": self._sla(case, now)})
        rows.sort(key=lambda r: (-r["priority_score"], r["sla"]["seconds_left"]))
        return {"items": rows[:limit], "matching": len(rows), "total_cases": total,
                "sla_breached": sum(1 for r in rows if r["sla"]["breached"])}

    # -------------------------------------------------------------------- case view -----------
    def _view(self, tenant_id: str, txn_id: str, viewer_role: str, viewer_pid: str | None) -> dict:
        ld = self._load(tenant_id, txn_id)
        found = self.repo.find(tenant_id, "review_cases", {"txn_id": txn_id}, limit=1)
        # An analyst may open or override a decision that was never queued (for example a high-trust
        # Block). Viewing must not write, so the case is transient until an action is recorded; the
        # blind assignment is deterministic, so it is the same when it is persisted.
        case = found[0] if found else {
            "tenant_id": tenant_id, "txn_id": txn_id, "first_decision_id": ld["pd"]["id"],
            "queue": "override", "sla_minutes": self.cfg.sla_minutes["override"],
            "enqueued_at": self.cfg.clock(), "_transient": True,
            "blind": blind_assignment(tenant_id, txn_id, self.cfg.blind_share, self.cfg.blind_salt)}
        actions = self._actions(tenant_id, txn_id)
        status, final = self.status_of(actions)
        status = self._effective_status(status, ld, actions)
        mine = [a for a in actions if a["analyst_pid"] == viewer_pid]
        blind_hidden = case["blind"] and viewer_role == "analyst" and not mine
        now = self.cfg.clock()
        txn = ld["txn"]
        view: dict[str, Any] = {
            "decision_id": ld["pd"]["id"], "txn_id": txn_id, "status": status, "final_decision": final,
            "queue": case["queue"], "blind": case["blind"], "redacted": blind_hidden,
            "priority": self._priority(ld, case, now), "sla": self._sla(case, now),
            "transaction": {"amount": float(txn["amount"]), "currency": txn["currency"],
                            "channel": txn["channel"], "event_time": _dt(txn["event_time"]).isoformat(),
                            "country": txn.get("country"), "merchant_id": txn.get("merchant_id"),
                            "customer": _mask(txn["customer_pid"]), "beneficiary": _mask(txn.get("beneficiary_pid"))},
            "outcome": self._outcome(tenant_id, txn_id),
            "decision": None, "explanation": None,
        }
        if not blind_hidden:
            view["decision"] = self.scoring._view(tenant_id, ld["pd"])
            view["explanation"] = self.scoring.explanation(tenant_id, ld["pd"]["id"])
        # Others' decisions stay hidden from a plain analyst on a blind case until they have decided.
        show_actions = viewer_role != "analyst" or not case["blind"] or bool(mine)
        view["actions"] = [{"analyst": a["analyst_pid"], "role": a["role"], "action": a["action"],
                            "final_decision": a["final_decision"], "reason_code": a["reason_code"],
                            "at": _dt(a["action_time"]).isoformat()} for a in actions] if show_actions else []
        view["needs_second_review"] = self._needs_second_review(ld, actions)
        view["queued"] = not case.get("_transient")
        view["_case"] = case
        return view

    def _needs_second_review(self, ld: dict, actions: list[dict]) -> bool:
        """An override of a HIGH-trust recommendation goes to second review (PRD 11.3)."""
        if ld["ta"]["state"] != "high":
            return False
        rec = _family(ld["pd"]["recommended_action"])
        overrides = [a for a in actions if a["action"] == "override" and a["final_decision"] != rec]
        others = [a for a in actions if a["final_decision"] and a not in overrides]
        return bool(overrides and not others)

    def case(self, tenant_id: str, decision_id: str, *, viewer_role: str, viewer_pid: str) -> dict:
        pd = self.repo.get_by_id(tenant_id, "policy_decisions", decision_id)
        if pd is None:
            raise ReviewError("not_found", "unknown decision", 404)
        return _public(self._view(tenant_id, pd["txn_id"], viewer_role, viewer_pid))

    # -------------------------------------------------------------------- actions --------------
    def record_action(self, tenant_id: str, decision_id: str, *, analyst_pid: str, role: str, action: str,
                      reason_code: str | None = None, override_to: str | None = None,
                      confidence: float | None = None, checklist: dict | None = None,
                      notes: str | None = None, seconds_to_decision: float | None = None) -> dict:
        if action not in ANALYST_ACTIONS:
            raise ReviewError("invalid_action", f"action must be one of {sorted(ANALYST_ACTIONS)}")
        pd = self.repo.get_by_id(tenant_id, "policy_decisions", decision_id)
        if pd is None:
            raise ReviewError("not_found", "unknown decision", 404)
        txn_id = pd["txn_id"]
        view = self._view(tenant_id, txn_id, role, analyst_pid)
        if view["status"] == "adjudicated":
            raise ReviewError("closed", "case already adjudicated", 409)
        if view["status"] == "escalated" and role not in ("senior_analyst",) and action in (*FINAL, "override"):
            raise ReviewError("escalated", "escalated cases are decided by a senior analyst", 403)
        actions = self._actions(tenant_id, txn_id)
        if any(a["analyst_pid"] == analyst_pid and a["final_decision"] for a in actions):
            raise ReviewError("already_decided", "this analyst has already decided this case", 409)
        if confidence is not None and not 0.0 <= confidence <= 1.0:
            raise ReviewError("invalid_confidence", "confidence must be in [0, 1]")

        final = None
        if action in FINAL:
            final = action
        elif action == "override":
            if override_to not in FINAL:
                raise ReviewError("invalid_override", "override_to must be approve or block")
            final = override_to
        if action == "override" and not reason_code:
            raise ReviewError("reason_required", "an override requires a reason code")  # FR-26
        rec = _family(pd["recommended_action"])
        if final and rec and final != rec and not reason_code:
            raise ReviewError("reason_required", "disagreeing with the recommendation requires a reason code")

        # What the analyst was shown is recorded by the SERVER, not claimed by the client (FR-26).
        display = {"blind": view["redacted"], "score_shown": not view["redacted"],
                   "recommendation_shown": not view["redacted"],
                   "risk": None if view["redacted"] else view["decision"]["risk"],
                   "trust_state": None if view["redacted"] else view["decision"]["trust"]["state"],
                   "ti": None if view["redacted"] else view["decision"]["trust"]["ti"],
                   "recommendation": None if view["redacted"] else view["decision"]["recommendation"],
                   "explanation_shown": not view["redacted"] and view["explanation"] is not None,
                   "trust_version": None if view["redacted"] else view["decision"]["trust"]["version_no"],
                   "policy_version": pd["policy_version"]}
        now = self.cfg.clock()
        case = view["_case"]
        with self.repo.atomic(tenant_id):
            if case.get("_transient"):  # first action on an unqueued decision: persist the case
                try:
                    self.repo.insert(tenant_id, "review_cases", {
                        "schema_version": SCHEMA, "txn_id": txn_id, "first_decision_id": pd["id"],
                        "queue": case["queue"], "blind": case["blind"], "sla_minutes": case["sla_minutes"],
                        "enqueued_at": now}, analyst_pid)
                except DuplicateError:
                    pass
            row = self.repo.insert(tenant_id, "analyst_actions", {
                "schema_version": SCHEMA, "txn_id": txn_id, "policy_decision_id": pd["id"],
                "analyst_pid": analyst_pid, "role": role, "action": action, "final_decision": final,
                "reason_code": reason_code, "confidence": confidence,
                "evidence_checklist": checklist or {}, "notes": notes, "display_state": display,
                "blind_flag": bool(view["blind"]), "seconds_to_decision": seconds_to_decision,
                "action_time": now}, analyst_pid)
            fr = self._score_feedback(tenant_id, row, view, pd, now)
            self.repo.append_audit(tenant_id, analyst_pid, f"review_{action}", txn_id,
                                   final or "none", now)
        status, final_decision = self.status_of(self._actions(tenant_id, txn_id))
        return {"action_id": row["id"], "status": status, "final_decision": final_decision,
                "feedback": fr, "needs_second_review": view["needs_second_review"] or (
                    action == "override" and self._needs_second_review(self._load(tenant_id, txn_id),
                                                                       self._actions(tenant_id, txn_id)))}

    def adjudicate(self, tenant_id: str, decision_id: str, *, analyst_pid: str, role: str,
                   final_decision: str, rationale: str, notes: str | None = None) -> dict:
        if role != "senior_analyst":
            raise ReviewError("forbidden", "only a senior analyst can adjudicate", 403)
        if final_decision not in FINAL:
            raise ReviewError("invalid_decision", "final_decision must be approve or block")
        if not rationale:
            raise ReviewError("reason_required", "an adjudication needs a stored rationale")  # PRD 8.4
        pd = self.repo.get_by_id(tenant_id, "policy_decisions", decision_id)
        if pd is None:
            raise ReviewError("not_found", "unknown decision", 404)
        status, _ = self.status_of(self._actions(tenant_id, pd["txn_id"]))
        if status not in ("conflicted", "escalated"):
            raise ReviewError("not_adjudicable", f"case is {status}; only conflicted or escalated cases are adjudicated", 409)
        view = self._view(tenant_id, pd["txn_id"], role, analyst_pid)
        now = self.cfg.clock()
        self.repo.insert(tenant_id, "analyst_actions", {
            "schema_version": SCHEMA, "txn_id": pd["txn_id"], "policy_decision_id": pd["id"],
            "analyst_pid": analyst_pid, "role": role, "action": "adjudicate",
            "final_decision": final_decision, "reason_code": rationale, "confidence": None,
            "evidence_checklist": {}, "notes": notes,
            "display_state": {"blind": False, "score_shown": True, "recommendation_shown": True,
                              "conflicting_decisions": [a["final_decision"] for a in view["actions"]
                                                        if a["final_decision"]]},
            "blind_flag": False, "seconds_to_decision": None, "action_time": now}, analyst_pid)
        self.repo.append_audit(tenant_id, analyst_pid, "review_adjudicate", pd["txn_id"], final_decision, now)
        return _public(self._view(tenant_id, pd["txn_id"], role, analyst_pid))

    # -------------------------------------------------------------------- feedback (shadow) -----
    def _analyst_history(self, tenant_id: str, analyst_pid: str, now: datetime) -> list[dict]:
        hist = []
        for a in self.repo.find(tenant_id, "analyst_actions", {"analyst_pid": analyst_pid}):
            if not a["final_decision"] or a["action"] == "adjudicate":
                continue
            out = self._outcome(tenant_id, a["txn_id"])
            if out is None:
                continue
            truth = "block" if out["verified"] == "fraud" else "approve"
            hist.append({"correct": a["final_decision"] == truth, "blind": a["blind_flag"],
                         "age_days": max(0.0, (now - _dt(a["action_time"])).total_seconds() / 86400)})
        return hist

    def _score_feedback(self, tenant_id: str, row: dict, view: dict, pd: dict, now: datetime) -> dict:
        cfg = self.cfg.feedback
        txn_id = row["txn_id"]
        actions = self._actions(tenant_id, txn_id)
        final = row["final_decision"]
        corroborated = bool(final) and any(
            a["final_decision"] == final and a["analyst_pid"] != row["analyst_pid"] for a in actions)
        aas, _ = fb.analyst_accuracy(self._analyst_history(tenant_id, row["analyst_pid"], now), cfg)
        crs = None
        if row["action"] == "override":
            past = [a for a in self.repo.find(tenant_id, "analyst_actions", {"action": "override"})
                    if a["final_decision"] == final and a["txn_id"] != txn_id]
            judged = [(a, self._outcome(tenant_id, a["txn_id"])) for a in past]
            judged = [(a, o) for a, o in judged if o]
            right = sum(1 for a, o in judged if (a["final_decision"] == "block") == (o["verified"] == "fraud"))
            crs = fb.correction_reliability(right, len(judged), cfg)
        fcs = fb.feedback_confidence(row["confidence"], row["evidence_checklist"], corroborated,
                                     bool(row["reason_code"]), bool(row["blind_flag"]), cfg)
        ld = self._load(tenant_id, txn_id)
        conf = (ld["ta"]["components"].get("conf") or {}).get("score") or 0.5
        nov = (ld["ta"]["evidence"] or {}).get("novelty", 0.0)
        rec = _family(pd["recommended_action"])
        amount_norm = min(1.0, math.log1p(float(ld["txn"]["amount"])) / math.log1p(self.cfg.amount_cap))
        lvs = fb.learning_value(conf, nov, bool(final and rec and final != rec), amount_norm, cfg)
        a_used = crs if row["action"] == "override" else aas
        fqs = fb.feedback_quality(fcs, a_used)
        status, _ = self.status_of(actions)
        disp, why = fb.disposition(
            fqs, reason_present=bool(row["reason_code"]) or not (final and rec and final != rec),
            unsure=row["action"] == "unsure", conflicted=status == "conflicted",
            outcome_known=self._outcome(tenant_id, txn_id) is not None, cfg=cfg)
        self.repo.insert(tenant_id, "feedback_records", {
            "schema_version": SCHEMA, "analyst_action_id": row["id"], "aas": aas, "fcs": fcs, "lvs": lvs,
            "crs": crs, "fqs": fqs, "formula_version": fb.FORMULA_VERSION, "disposition": disp,
            "disposition_reason": why}, "feedback")
        return {"aas": aas, "fcs": fcs, "lvs": lvs, "crs": crs, "fqs": fqs, "disposition": disp,
                "reason": why, "formula_version": fb.FORMULA_VERSION,
                "note": "shadow scores: stored, not used for learning (MVP)"}

    # -------------------------------------------------------------------- dashboard -------------
    def dashboard(self, tenant_id: str, *, window_days: int = 30) -> dict:
        now = self.cfg.clock()
        q = self.queue(tenant_id, viewer_role="manager", limit=10_000)
        soonest = min((r["sla"]["seconds_left"] for r in q["items"]), default=None)
        latest: dict[str, dict] = {}
        for ta in self.repo.find(tenant_id, "trust_assessments"):
            cur = latest.get(ta["prediction_id"])
            if cur is None or ta["version_no"] > cur["version_no"]:
                latest[ta["prediction_id"]] = ta
        preds = {p["id"]: p for p in self.repo.find(tenant_id, "predictions")}
        states = Counter(t["state"] for t in latest.values())
        n = sum(states.values())

        ht_k = ht_n = 0
        for pid, ta in latest.items():
            if ta["state"] != "high":
                continue
            pr = preds[pid]
            out = self._outcome(tenant_id, pr["txn_id"])
            if out is None:
                continue
            bundle = self.scoring.registry.get(tenant_id, pr["bundle_id"])
            call = pr["calibrated_risk"] >= bundle.manifest.thresholds["t_high"]
            ht_n += 1
            ht_k += int(call != (out["verified"] == "fraud"))
        if ht_n:
            lo, hi = wilson_interval(ht_k, ht_n)
            ht = {"value": ht_k / ht_n, "lo": lo, "hi": hi, "n": ht_n, "basis": "matured verified outcomes"}
        else:
            ht = {"value": None, "lo": None, "hi": None, "n": 0,
                  "basis": "no matured verified outcomes yet for High-trust cases"}

        def rate_in(lo_t: datetime, hi_t: datetime):
            items = []
            for ta in latest.values():
                fv = self.repo.find(tenant_id, "feature_vectors", {"txn_id": preds[ta["prediction_id"]]["txn_id"]}, limit=1)
                t = _dt(fv[0]["as_of_time"]) if fv else None
                if t and lo_t <= t < hi_t:
                    items.append(ta["state"] == "insufficient_evidence")
            return (sum(items) / len(items)) if items else None, len(items)

        cur, cur_n = rate_in(now - timedelta(days=window_days), now + timedelta(days=1))
        prev, prev_n = rate_in(now - timedelta(days=2 * window_days), now - timedelta(days=window_days))
        dispositions = Counter(r["disposition"] for r in self.repo.find(tenant_id, "feedback_records"))
        nd = sum(dispositions.values())
        dq_floor = sum(1 for t in latest.values() if "DATA_QUALITY_FLOOR" in t["reason_codes"])
        try:
            drift = self.scoring.current_drift(tenant_id)
        except ScoringError:  # no champion bundle: nothing to measure drift against
            drift = None
        max_drift = float(drift.max()) if drift is not None and len(drift) else 0.0
        return {
            "as_of": now.isoformat(),
            "pending_reviews": {"count": q["matching"], "total_cases": q["total_cases"],
                                "sla_breached": q["sla_breached"], "soonest_sla_seconds": soonest},
            "high_trust_error_rate": ht,
            "insufficient_evidence_rate": {"value": cur, "n": cur_n, "previous": prev, "previous_n": prev_n,
                                           "window_days": window_days},
            "feedback_acceptance": {"accepted": dispositions.get("accept", 0),
                                    "rejected": dispositions.get("reject", 0),
                                    "deferred": dispositions.get("defer", 0),
                                    "acceptance_rate": (dispositions.get("accept", 0) / nd) if nd else None,
                                    "note": "shadow scores; nothing trains on feedback in the MVP"},
            "trust_distribution": dict(states), "assessed_cases": n,
            "model_drift": {"status": "alarm" if max_drift >= self.cfg.drift_alarm else "stable",
                            "max_feature_drift": max_drift},
            "data_quality": {"status": "degraded" if n and dq_floor / n > 0.10 else "healthy",
                             "floor_share": (dq_floor / n) if n else 0.0},
        }
