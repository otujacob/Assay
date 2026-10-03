"""Read-only audit view and export (FR-34, PRD 14.3 Auditability).

Search is by transaction, actor, action, model version and date. Reads the append-only
`audit_log` plus the predictions that link a transaction to the model bundle that scored it.
Nothing here writes, except that the caller records the access itself (FR-43).
"""

from __future__ import annotations

import csv
import io
import json
from datetime import UTC, datetime

from assay.ingestion.repo import _to_dt

# Bound into the encryption of a sealed export, so it will not open as anything else (FR-41).
SEAL_CONTEXT = "audit-export"
FIELDS = ("seq", "time", "actor", "action", "object", "result", "row_hash")
MAX_LIMIT = 5000


def _utc(d: datetime | None) -> datetime | None:
    return d if d is None or d.tzinfo else d.replace(tzinfo=UTC)


def query(repo, tenant_id: str, *, txn_id: str | None = None, actor: str | None = None,
          action: str | None = None, model_version: str | None = None,
          since: datetime | None = None, until: datetime | None = None,
          limit: int = 500) -> dict:
    """Newest first. `matching` counts every hit; `items` is capped at `limit`."""
    since, until = (_utc(d) for d in (since, until))
    objects: set[str] | None = None
    if txn_id is not None:
        objects = {txn_id}
        txn = repo.get_transaction(tenant_id, txn_id)
        if txn and txn.get("ingestion_event_id"):
            ev = repo.get_by_id(tenant_id, "ingestion_events", txn["ingestion_event_id"])
            if ev:
                objects.add(ev["event_id"])  # ingest rows are filed under the event id
    if model_version is not None:
        scored = {p["txn_id"] for p in repo.rows(tenant_id, "predictions")
                  if p["bundle_id"] == model_version}
        objects = scored if objects is None else objects & scored

    rows = []
    for r in repo.rows(tenant_id, "audit_log"):
        t = _utc(_to_dt(r["time"]))
        if objects is not None and r["object"] not in objects:
            continue
        if actor is not None and r["actor"] != actor:
            continue
        if action is not None and r["action"] != action:
            continue
        if since is not None and t < since:
            continue
        if until is not None and t >= until:
            continue
        rows.append({"seq": r.get("seq"), "time": t.isoformat(), "actor": r["actor"],
                     "action": r["action"], "object": r["object"], "result": r["result"],
                     "row_hash": r["row_hash"]})
    rows.reverse()
    limit = max(1, min(limit, MAX_LIMIT))
    return {"items": rows[:limit], "matching": len(rows),
            "chain_ok": repo.verify(tenant_id, "audit_log") is None}


def _safe_cell(v):
    """A cell starting with = + - @ (or a tab or carriage return) is run as a formula by
    spreadsheets, so prefix it. A lone "-" is the audit log's "no object" placeholder, not a formula."""
    if isinstance(v, str) and v != "-" and v[:1] in ("=", "+", "-", "@", "\t", "\r"):
        return "'" + v
    return v


def to_csv(items: list[dict]) -> str:
    out = io.StringIO()
    w = csv.DictWriter(out, fieldnames=FIELDS, lineterminator="\n")
    w.writeheader()
    for i in items:
        w.writerow({f: _safe_cell(i.get(f)) for f in FIELDS})
    return out.getvalue()


def to_json(items: list[dict]) -> str:
    return json.dumps(items, default=str)
