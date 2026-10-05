"""Population drift job (PRD 9.1, 9.3, FR-17).

Compares the champion model's reference sample with the most recent scored feature vectors and
stores the per-feature drift as an append-only `drift_runs` row. Scoring reads the latest run, so
the signal survives a restart and is the same for every API worker. A run at or above the alarm
level is flagged and audited, which is the population-level alarm that opens a recalibration review.

Too few recent rows, or rows that span too little time, is not drift and not "no drift": the job
reports `skipped` and stores nothing, so scoring keeps using the previous run (or none) rather than
a figure from a thin or time-of-day-biased window.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from assay.features.compute import parse_time
from assay.trust.drift import drift_vector

SCHEMA = "drift-1"


@dataclass
class DriftJobConfig:
    window: int = 500            # most recent scored transactions to compare (parameter)
    min_rows: int = 100          # fewer than this and the run is skipped (parameter)
    # A window that covers only a few hours differs from the reference in time-of-day features
    # alone (measured on the synthetic data: 60 consecutive rows gave drift 0.30 on `hour` with
    # nothing wrong), so a narrow window is skipped rather than allowed to raise a false alarm.
    min_span_hours: float = 24.0  # (parameter)
    alarm_level: float = 0.10    # scaled KS statistic that raises the alarm (parameter, PRD 6.5)


def run_drift_job(scoring, tenant_id: str, cfg: DriftJobConfig | None = None,
                  actor: str = "drift-job") -> dict:
    """One run for a tenant. Returns {"status": "ok" | "skipped", ...}; "ok" carries the stored run."""
    cfg = cfg or DriftJobConfig()
    repo, lb = scoring.repo, scoring.deciding_bundle(tenant_id)
    rows = [r for r in repo.find(tenant_id, "feature_vectors", newest_first=True, limit=cfg.window)
            if r["feature_set_version"] == lb.manifest.feature_set_version
            and len(r["feature_values"]) == lb.n_features]
    if len(rows) < cfg.min_rows:
        return {"status": "skipped", "reason": "thin_window", "n_window": len(rows),
                "min_rows": cfg.min_rows}
    times = [parse_time(r["as_of_time"]) for r in rows]
    span_h = (max(times) - min(times)).total_seconds() / 3600
    if span_h < cfg.min_span_hours:
        return {"status": "skipped", "reason": "short_span", "n_window": len(rows),
                "span_hours": span_h, "min_span_hours": cfg.min_span_hours}
    window = np.array([r["feature_values"] for r in rows], dtype=float)
    d = drift_vector(np.asarray(lb.reference.ref_X, dtype=float), window)
    mx = float(d.max())
    run = repo.insert(tenant_id, "drift_runs", {
        "schema_version": SCHEMA, "bundle_id": lb.manifest.bundle_id, "n_reference": len(lb.reference.ref_X),
        "n_window": len(rows), "window_start": min(times), "window_end": max(times),
        "feature_names": list(lb.manifest.feature_names), "drift": [float(v) for v in d],
        "max_drift": mx, "alarm_level": cfg.alarm_level, "alarm": mx >= cfg.alarm_level}, actor)
    now = scoring.cfg.clock()
    repo.append_audit(tenant_id, actor, "drift_run", lb.manifest.bundle_id, f"max={mx:.3f}", now)
    if run["alarm"]:
        top = lb.manifest.feature_names[int(np.argmax(d))]
        repo.append_audit(tenant_id, actor, "drift_alarm", lb.manifest.bundle_id, f"{top}={mx:.3f}", now)
    scoring.drift[tenant_id] = d
    return {"status": "ok", "run": run}
