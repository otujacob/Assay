"""PostgreSQL repository for ingestion, same interface as InMemoryRepository.

Isolation is enforced by the database (db/migrations/0001_init.sql): every transaction
runs as the low-privilege role and sets app.tenant_id, so row-level security applies even
if a query here forgot its tenant filter. Hashes for the chain are computed by the
database trigger, from the payload_hash supplied here.

One connection, used by one caller at a time. A pool comes with the service wiring.
"""

from __future__ import annotations

import uuid
from contextlib import contextmanager
from datetime import datetime
from typing import Any

import psycopg
from psycopg import errors, sql
from psycopg.rows import dict_row
from psycopg.types.json import Jsonb

from assay.lineage import payload_hash

from .repo import DuplicateError

# Columns each table accepts from the service. Anything else in a body (for example the
# event_id, which lives on ingestion_events) is dropped, and the hash covers what is stored.
COLUMNS: dict[str, tuple[str, ...]] = {
    "ingestion_events": ("schema_version", "event_id", "kind", "source_id", "event_time",
                         "recorded_at", "event_payload_hash", "validation_result", "reasons"),
    "transactions": ("schema_version", "txn_id", "customer_pid", "account_pid", "beneficiary_pid",
                     "merchant_id", "amount", "currency", "channel", "device_hash", "ip_hash",
                     "country", "event_time", "recorded_at", "ingestion_event_id"),
    "outcomes": ("schema_version", "txn_id", "outcome_type", "source", "event_time",
                 "maturity_state", "matured_at", "recorded_at", "ingestion_event_id"),
    "quarantine": ("schema_version", "txn_id", "outcome_type", "source", "event_time", "reason",
                   "recorded_at", "ingestion_event_id"),
    "audit_log": ("actor", "action", "object", "result", "time"),
    # decision lineage objects (db/migrations/0002_decisions.sql)
    "model_bundles": ("schema_version", "bundle_id", "model_version", "dataset_id", "code_commit",
                      "status", "artifact_sha256", "manifest"),
    "feature_vectors": ("schema_version", "txn_id", "feature_set_version", "definition_versions",
                        "as_of_time", "feature_names", "feature_values", "graph_snapshot_id"),
    "predictions": ("schema_version", "txn_id", "feature_vector_id", "bundle_id", "raw_score",
                    "calibrated_risk", "member_spread", "distance_to_threshold"),
    "explanations": ("schema_version", "prediction_id", "method", "params", "background_version",
                     "seed", "attributions", "stability", "sensitivity", "faithfulness",
                     "reproducible"),
    "trust_assessments": ("schema_version", "prediction_id", "version_no", "mode", "state", "ti",
                          "ti_low", "ti_high", "reason_codes", "components", "weights_version",
                          "weights_used", "evidence"),
    "policy_decisions": ("schema_version", "txn_id", "trust_assessment_id", "policy_version",
                         "risk_band", "gate", "queue", "recommended_action", "automation_level"),
    "review_cases": ("schema_version", "txn_id", "first_decision_id", "queue", "blind", "sla_minutes",
                     "enqueued_at"),
    "analyst_actions": ("schema_version", "txn_id", "policy_decision_id", "analyst_pid", "role", "action",
                        "final_decision", "reason_code", "confidence", "evidence_checklist", "notes",
                        "display_state", "blind_flag", "seconds_to_decision", "action_time"),
    "feedback_records": ("schema_version", "analyst_action_id", "aas", "fcs", "lvs", "crs", "fqs",
                         "formula_version", "disposition", "disposition_reason"),
    "validation_reports": ("schema_version", "bundle_id", "dataset_id", "mode", "n_cases", "measures",
                           "baselines", "ablations", "stress_results", "pass_criteria", "limitations"),
    "policy_versions": ("schema_version", "version", "payload", "effective_from"),
    "policy_approvals": ("schema_version", "policy_version_id", "proposed_by", "approved_by",
                         "approved_at"),
    "model_candidates": ("schema_version", "candidate_id", "base_bundle_id", "artefact_path", "pool_summary",
                         "training", "gates"),
    "model_lifecycle_events": ("schema_version", "candidate_id", "created_by_candidate", "kind", "detail"),
    "shadow_scores": ("schema_version", "txn_id", "candidate_id", "champion_id", "candidate_risk",
                      "champion_risk", "candidate_call", "champion_call"),
    "graph_edge_flags": ("schema_version", "rel", "src", "dst", "reason", "flagged_at"),
    "drift_runs": ("schema_version", "bundle_id", "n_reference", "n_window", "window_start",
                   "window_end", "feature_names", "drift", "max_drift", "alarm_level", "alarm"),
}
JSONB_COLUMNS = frozenset({"reasons", "manifest", "definition_versions", "feature_names",
                           "feature_values", "params", "attributions", "reason_codes", "components",
                           "weights_used", "evidence", "measures", "baselines", "ablations",
                           "stress_results", "pass_criteria", "limitations", "evidence_checklist",
                           "display_state", "payload", "drift", "pool_summary", "training", "gates",
                           "detail"})


def _norm(row: dict[str, Any]) -> dict[str, Any]:
    return {k: (str(v) if isinstance(v, uuid.UUID) else v) for k, v in row.items()}


class PostgresRepository:
    def __init__(self, conn: psycopg.Connection, *, role: str | None = "assay_app",
                 search_path: str | None = None):
        conn.row_factory = dict_row
        self.conn = conn
        self.role = role
        self._tenant: str | None = None
        if search_path:
            with conn.transaction():
                conn.execute(sql.SQL("SET search_path TO {}").format(sql.Identifier(search_path)))

    # -- transaction scope -------------------------------------------------------
    @contextmanager
    def atomic(self, tenant_id: str):
        """One DB transaction, running as the app role with the tenant set for RLS."""
        if self._tenant is not None:
            if self._tenant != tenant_id:
                raise RuntimeError("nested atomic() for a different tenant")
            yield
            return
        try:
            with self.conn.transaction():
                if self.role:
                    self.conn.execute(sql.SQL("SET LOCAL ROLE {}").format(sql.Identifier(self.role)))
                self.conn.execute("SELECT set_config('app.tenant_id', %s, true)", (tenant_id,))
                self.conn.execute("SET LOCAL TIME ZONE 'UTC'")  # timestamps are stored and read as UTC
                self._tenant = tenant_id
                yield
        finally:
            self._tenant = None

    # -- primitives ----------------------------------------------------------------
    def _insert(self, tenant_id: str, table: str, body: dict[str, Any], created_by: str) -> dict:
        cols = {c: body[c] for c in COLUMNS[table] if c in body}
        ph = payload_hash(cols)
        cols = {k: (Jsonb(v) if k in JSONB_COLUMNS else v) for k, v in cols.items()}
        names = ["tenant_id", "created_by", "payload_hash", "prev_hash", "row_hash", *cols]
        values = [tenant_id, created_by, ph, "", "", *cols.values()]  # hashes set by trigger
        q = sql.SQL("INSERT INTO {t} ({c}) VALUES ({v}) RETURNING *").format(
            t=sql.Identifier(table), c=sql.SQL(", ").join(map(sql.Identifier, names)),
            v=sql.SQL(", ").join(sql.Placeholder() * len(values)))
        with self.atomic(tenant_id):
            try:
                with self.conn.transaction():  # savepoint, so a duplicate leaves the tx usable
                    return _norm(self.conn.execute(q, values).fetchone())
            except errors.UniqueViolation as e:
                raise DuplicateError(str(e.diag.constraint_name)) from None

    def _one(self, tenant_id: str, query: sql.Composable, params: tuple) -> dict | None:
        with self.atomic(tenant_id):
            row = self.conn.execute(query, params).fetchone()
        return _norm(row) if row else None

    # -- generic lineage objects (PRD 14.1) ----------------------------------------------
    def insert(self, tenant_id: str, table: str, body: dict[str, Any], created_by: str) -> dict:
        return self._insert(tenant_id, table, body, created_by)

    def find(self, tenant_id: str, table: str, where: dict[str, Any] | None = None, *,
             newest_first: bool = False, limit: int | None = None) -> list[dict[str, Any]]:
        if table not in COLUMNS:
            raise ValueError(f"unknown table {table}")
        allowed = set(COLUMNS[table]) | {"id"}
        where = where or {}
        if not set(where) <= allowed:
            raise ValueError(f"cannot filter {table} by {sorted(set(where) - allowed)}")
        conds = [sql.SQL("tenant_id = %s")] + [sql.SQL("{} = %s").format(sql.Identifier(k)) for k in where]
        q = sql.SQL("SELECT * FROM {t} WHERE {w} ORDER BY seq {d}{lim}").format(
            t=sql.Identifier(table), w=sql.SQL(" AND ").join(conds),
            d=sql.SQL("DESC" if newest_first else "ASC"),
            lim=sql.SQL(f" LIMIT {int(limit)}") if limit else sql.SQL(""))
        with self.atomic(tenant_id):
            return [_norm(r) for r in self.conn.execute(q, [tenant_id, *where.values()]).fetchall()]

    def get_by_id(self, tenant_id: str, table: str, row_id: str) -> dict[str, Any] | None:
        rows = self.find(tenant_id, table, {"id": row_id})
        return rows[0] if rows else None

    def history_for_features(self, tenant_id: str, txn: dict[str, Any], window_days: int = 30):
        """Rows that can influence this transaction\x27s features: the customer\x27s earlier
        transactions, and payments to the same beneficiary in the window (PRD 13.8)."""
        with self.atomic(tenant_id):
            cur = self.conn.execute(
                "SELECT * FROM transactions WHERE tenant_id = %(t)s AND txn_id <> %(id)s "
                "AND event_time < %(et)s AND (customer_pid = %(c)s OR (%(b)s::text IS NOT NULL "
                "AND beneficiary_pid = %(b)s AND event_time > %(et)s - make_interval(days => %(w)s))) "
                "ORDER BY event_time",
                {"t": tenant_id, "id": txn["txn_id"], "et": txn["event_time"],
                 "c": txn["customer_pid"], "b": txn.get("beneficiary_pid"), "w": window_days})
            return [_norm(r) for r in cur.fetchall()]

    # -- ingestion events ------------------------------------------------------------
    def get_ingestion_event(self, tenant_id: str, event_id: str):
        return self._one(tenant_id, sql.SQL(
            "SELECT * FROM ingestion_events WHERE tenant_id = %s AND event_id = %s "
            "AND validation_result <> 'rejected' AND event_id <> '' ORDER BY seq LIMIT 1"),
            (tenant_id, event_id))

    def append_ingestion_event(self, tenant_id: str, *, event_id: str, kind: str, source_id: str,
                               event_time: datetime | None, recorded_at: datetime,
                               event_payload_hash: str, schema_version: str | None,
                               validation_result: str, reasons: list[str], created_by: str):
        return self._insert(tenant_id, "ingestion_events", {
            "event_id": event_id, "kind": kind, "source_id": source_id, "event_time": event_time,
            "recorded_at": recorded_at, "event_payload_hash": event_payload_hash,
            "schema_version": schema_version, "validation_result": validation_result,
            "reasons": reasons}, created_by)

    # -- transactions / outcomes ---------------------------------------------------------
    def get_transaction(self, tenant_id: str, txn_id: str):
        return self._one(tenant_id, sql.SQL(
            "SELECT * FROM transactions WHERE tenant_id = %s AND txn_id = %s"), (tenant_id, txn_id))

    def append_transaction(self, tenant_id: str, body: dict[str, Any], created_by: str):
        return self._insert(tenant_id, "transactions", body, created_by)

    def append_outcome(self, tenant_id: str, body: dict[str, Any], created_by: str):
        return self._insert(tenant_id, "outcomes", body, created_by)

    def append_quarantine(self, tenant_id: str, body: dict[str, Any], created_by: str):
        return self._insert(tenant_id, "quarantine", body, created_by)

    def append_audit(self, tenant_id: str, actor: str, action: str, obj: str, result: str,
                     at: datetime):
        return self._insert(tenant_id, "audit_log", {"actor": actor, "action": action,
                                                     "object": obj, "result": result, "time": at},
                            actor)

    # -- reads -------------------------------------------------------------------------
    def rows(self, tenant_id: str, table: str) -> list[dict[str, Any]]:
        if table not in COLUMNS:
            raise ValueError(f"unknown table {table}")
        with self.atomic(tenant_id):
            cur = self.conn.execute(
                sql.SQL("SELECT * FROM {} WHERE tenant_id = %s ORDER BY seq").format(
                    sql.Identifier(table)), (tenant_id,))
            return [_norm(r) for r in cur.fetchall()]

    def transactions_recorded_before(self, tenant_id: str, as_of: datetime):
        with self.atomic(tenant_id):
            cur = self.conn.execute(
                "SELECT * FROM transactions WHERE tenant_id = %s AND recorded_at <= %s ORDER BY seq",
                (tenant_id, as_of))
            return [_norm(r) for r in cur.fetchall()]

    def count_by_validation(self, tenant_id: str) -> dict[str, int]:
        with self.atomic(tenant_id):
            cur = self.conn.execute(
                "SELECT validation_result, count(*) AS n FROM ingestion_events "
                "WHERE tenant_id = %s GROUP BY 1", (tenant_id,))
            return {r["validation_result"]: r["n"] for r in cur.fetchall()}

    def verify(self, tenant_id: str, table: str) -> int | None:
        with self.atomic(tenant_id):
            row = self.conn.execute("SELECT assay_verify_chain(%s, %s) AS bad",
                                    (table, tenant_id)).fetchone()
        return row["bad"]
