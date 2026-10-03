"""Versioned, effective-dated tenant policies with second-person approval (PRD 10.4, FR-21).

A policy is proposed by one person and approved by a different one. Both are append-only rows, so
the history of who proposed and who approved what is never rewritten. A version takes effect at
the later of its effective date and its approval time; until a tenant has an approved version the
built-in default (`policy-0`) applies. Each decision records the version that produced it.

Automation above level 0 is refused here and again by the database (FR-23).
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, datetime
from typing import Any

from assay.ingestion.repo import DuplicateError, _to_dt

from .engine import Action, PolicyConfig

SCHEMA = "pol-1"
DQ_ACTIONS = (Action.REQUEST_HUMAN_REVIEW.value, Action.HOLD.value)
FIELDS = {"dq_gate_action", "automation_level"}


class PolicyError(Exception):
    def __init__(self, code: str, detail: str, http: int = 422):
        super().__init__(detail)
        self.code, self.http = code, http


def _utc(d: datetime) -> datetime:
    return d if d.tzinfo else d.replace(tzinfo=UTC)


class PolicyService:
    def __init__(self, repo, clock: Callable[[], datetime] = lambda: datetime.now(UTC)):
        self.repo, self.clock = repo, clock

    # ------------------------------------------------------------------ writes ---------------
    def propose(self, tenant_id: str, proposer: str, payload: dict[str, Any],
                effective_from: datetime | None = None) -> dict:
        payload = self._validate(payload)
        now = self.clock()
        with self.repo.atomic(tenant_id):
            n = len(self.repo.find(tenant_id, "policy_versions")) + 1
            try:
                row = self.repo.insert(tenant_id, "policy_versions", {
                    "schema_version": SCHEMA, "version": f"policy-{n}", "payload": payload,
                    "effective_from": _utc(effective_from) if effective_from else now}, proposer)
            except DuplicateError:  # two proposals raced for the same number
                raise PolicyError("conflict", "another policy was proposed at the same moment; retry",
                                  409) from None
            self.repo.append_audit(tenant_id, proposer, "policy_propose", row["version"], "pending", now)
        return self._view(row, None)

    def approve(self, tenant_id: str, approver: str, policy_id: str) -> dict:
        now = self.clock()
        with self.repo.atomic(tenant_id):
            pv = self.repo.get_by_id(tenant_id, "policy_versions", policy_id)
            if pv is None:
                raise PolicyError("not_found", "unknown policy version", 404)
            if pv["created_by"] == approver:  # also enforced by the database
                raise PolicyError("same_approver", "a policy cannot be approved by the person who proposed it",
                                  403)
            try:
                ap = self.repo.insert(tenant_id, "policy_approvals", {
                    "schema_version": SCHEMA, "policy_version_id": pv["id"], "proposed_by": pv["created_by"],
                    "approved_by": approver, "approved_at": now}, approver)
            except DuplicateError:
                raise PolicyError("already_approved", "this policy version is already approved", 409) from None
            self.repo.append_audit(tenant_id, approver, "policy_approve", pv["version"], "approved", now)
        return self._view(pv, ap)

    # ------------------------------------------------------------------ reads ----------------
    def list(self, tenant_id: str) -> list[dict]:
        """Newest first, each with its approval status."""
        approvals = {a["policy_version_id"]: a for a in self.repo.find(tenant_id, "policy_approvals")}
        rows = self.repo.find(tenant_id, "policy_versions", newest_first=True)
        return [self._view(r, approvals.get(r["id"])) for r in rows]

    def active(self, tenant_id: str, at: datetime | None = None) -> dict | None:
        """The approved version in force at `at`, or None (the built-in default applies)."""
        at = _utc(at or self.clock())
        approvals = {a["policy_version_id"]: a for a in self.repo.find(tenant_id, "policy_approvals")}
        best, best_t = None, None
        for r in self.repo.find(tenant_id, "policy_versions"):  # oldest first, so a later one wins a tie
            ap = approvals.get(r["id"])
            if ap is None:
                continue
            t = max(_utc(_to_dt(r["effective_from"])), _utc(_to_dt(ap["approved_at"])))
            if t <= at and (best_t is None or t >= best_t):
                best, best_t = self._view(r, ap), t
        return best

    def config(self, tenant_id: str, at: datetime, *, t_low: float, t_high: float,
               default_version: str = "policy-0") -> PolicyConfig:
        """The PolicyConfig to evaluate with: thresholds come from the model bundle, the rest from
        the active policy version."""
        pol = self.active(tenant_id, at)
        if pol is None:
            return PolicyConfig(version=default_version, t_low=t_low, t_high=t_high)
        return PolicyConfig(
            version=pol["version"], t_low=t_low, t_high=t_high,
            automation_level=pol["payload"].get("automation_level", 0),
            dq_gate_action=Action(pol["payload"].get("dq_gate_action", Action.REQUEST_HUMAN_REVIEW.value)))

    # ------------------------------------------------------------------ helpers --------------
    @staticmethod
    def _validate(payload: dict[str, Any]) -> dict[str, Any]:
        if not isinstance(payload, dict) or set(payload) - FIELDS:
            raise PolicyError("unknown_fields", f"allowed fields: {sorted(FIELDS)}")
        out = {"dq_gate_action": Action.REQUEST_HUMAN_REVIEW.value, "automation_level": 0, **payload}
        if out["automation_level"] != 0 or isinstance(out["automation_level"], bool):
            raise PolicyError("automation_not_supported", "the MVP runs at automation level 0 only (FR-23)")
        if out["dq_gate_action"] not in DQ_ACTIONS:
            raise PolicyError("bad_dq_gate_action", f"dq_gate_action must be one of {list(DQ_ACTIONS)}")
        return out

    @staticmethod
    def _view(pv: dict, ap: dict | None) -> dict:
        return {"id": pv["id"], "version": pv["version"], "payload": pv["payload"],
                "effective_from": _utc(_to_dt(pv["effective_from"])).isoformat(),
                "proposed_by": pv["created_by"], "status": "approved" if ap else "pending",
                "approved_by": ap["approved_by"] if ap else None,
                "approved_at": _utc(_to_dt(ap["approved_at"])).isoformat() if ap else None}
