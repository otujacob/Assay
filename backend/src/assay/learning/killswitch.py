"""The kill switch (PRD 12.4): every case to human review, whatever the model says.

For when no model can be trusted: a live model has degraded and there is nothing safe to fall back to, or someone has decided to
stop relying on the model now. While it is engaged for a tenant, the policy engine sends every new case to a person (gate
"kill_switch"). Scoring still runs and is still stored, so lineage and later analysis are complete; the recommendation just never
comes from the model. Existing decisions are not changed.

Anyone with authority can engage it (an administrator or approver, or the degradation monitor) because stopping relying on the model
is the safe direction. Only an approver can release it. Both need a stored reason. It is an append-only log: the latest row is the
state.
"""

from __future__ import annotations

from datetime import UTC, datetime

SCHEMA = "ks-1"


class KillSwitchError(Exception):
    def __init__(self, code: str, detail: str, http: int = 409):
        super().__init__(detail)
        self.code, self.http = code, http


class KillSwitch:
    def __init__(self, repo, clock=lambda: datetime.now(UTC)):
        self.repo, self.clock = repo, clock

    def state(self, tenant_id: str) -> dict:
        rows = self.repo.find(tenant_id, "kill_switch_events", newest_first=True, limit=1)
        if not rows:
            return {"engaged": False}
        r = rows[0]
        return {"engaged": r["kind"] == "engaged", "last": r["kind"], "reason": r["reason"], "by": r["created_by"]}

    def engaged(self, tenant_id: str) -> bool:
        return self.state(tenant_id)["engaged"]

    def history(self, tenant_id: str) -> list[dict]:
        return [{"kind": r["kind"], "reason": r["reason"], "by": r["created_by"]}
                for r in self.repo.find(tenant_id, "kill_switch_events")]

    def _write(self, tenant_id: str, actor: str, kind: str, reason: str) -> dict:
        if not (reason and reason.strip()):
            raise KillSwitchError("reason_required", f"the kill switch needs a stored reason to be {kind}", 422)
        with self.repo.atomic(tenant_id):
            self.repo.insert(tenant_id, "kill_switch_events", {"schema_version": SCHEMA, "kind": kind, "reason": reason.strip()}, actor)
            self.repo.append_audit(tenant_id, actor, f"kill_switch_{kind}", "-", reason.strip()[:200], self.clock())
        return self.state(tenant_id)

    def engage(self, tenant_id: str, actor: str, reason: str) -> dict:
        if self.engaged(tenant_id):
            raise KillSwitchError("already_engaged", "the kill switch is already engaged")
        return self._write(tenant_id, actor, "engaged", reason)

    def release(self, tenant_id: str, actor: str, reason: str) -> dict:
        if not self.engaged(tenant_id):
            raise KillSwitchError("not_engaged", "the kill switch is not engaged")
        return self._write(tenant_id, actor, "released", reason)
