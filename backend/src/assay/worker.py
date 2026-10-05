"""Background worker: `python -m assay.worker` (add `--once` for a single pass).

Runs the two jobs the API does not:
  * explanation refinement (PRD 5.8): cases scored without explanation testing cannot reach High
    trust until it runs, so this stores the new trust-assessment version beside the first;
  * the population drift job (PRD 9, FR-17): stores the latest per-feature drift for scoring to use.

Configuration comes from the environment, the same variables as the API (see assay/api/main.py):
  ASSAY_DATABASE_URL, ASSAY_BUNDLES, ASSAY_BUNDLE_SIGNING_KEY, ASSAY_MASTER_KEY
and, optional:
  ASSAY_WORKER_INTERVAL_S   seconds between passes (default 300)
  ASSAY_WORKER_REFINE_LIMIT cases refined per tenant per pass (default 100)
One tenant failing never stops the others. A pass is safe to repeat: refinement skips cases that
already have an explanation, and a drift run is only stored when the window is wide enough.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import signal
import threading
from collections.abc import Callable
from contextlib import AbstractContextManager

from assay.scoring import DriftJobConfig, ScoringService, run_drift_job

log = logging.getLogger("assay.worker")


def run_once(scoring: ScoringService, tenant_id: str, *, refine_limit: int = 100,
             drift_cfg: DriftJobConfig | None = None) -> dict:
    refined = scoring.refine_pending(tenant_id, limit=refine_limit, actor="worker")
    drift = run_drift_job(scoring, tenant_id, drift_cfg)
    from assay.learning.service import LearningService

    with scoring.repo.atomic(tenant_id):  # a breach rolls the live model back to the previous champion, never forward
        health = LearningService(scoring.repo, scoring).check_health(tenant_id, act=True)
    return {"refined": refined, "drift": drift["status"], "drift_reason": drift.get("reason"),
            "alarm": bool(drift.get("run", {}).get("alarm")),
            "rolled_back": [h["bundle_id"] for h in health if h["rolled_back"]]}


def run_cycle(tenants: list[str], open_scoring: Callable[[], AbstractContextManager[ScoringService]], *,
              refine_limit: int = 100, drift_cfg: DriftJobConfig | None = None) -> dict[str, dict]:
    """One pass over every tenant. A failure is logged and reported for that tenant only."""
    out: dict[str, dict] = {}
    for tenant in tenants:
        try:
            with open_scoring() as scoring:
                out[tenant] = run_once(scoring, tenant, refine_limit=refine_limit, drift_cfg=drift_cfg)
        except Exception as e:
            log.exception("worker pass failed for tenant %s", tenant)
            out[tenant] = {"error": type(e).__name__}
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="assay.worker")
    ap.add_argument("--once", action="store_true", help="run one pass and exit")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")

    from psycopg_pool import ConnectionPool

    from assay.api.main import (
        key_provider_from_env,
        load_registry,
        make_scoring_provider,
        refuse_superuser,
    )

    specs = json.loads(os.environ["ASSAY_BUNDLES"])
    registry = load_registry(specs, os.environ["ASSAY_BUNDLE_SIGNING_KEY"].encode(), key_provider_from_env())
    pool = ConnectionPool(os.environ["ASSAY_DATABASE_URL"], min_size=1, max_size=2, open=True)
    refuse_superuser(pool)
    open_scoring = make_scoring_provider(pool, registry)  # also registers the review-queue hook
    tenants = sorted({s["tenant_id"] for s in specs})
    interval = float(os.environ.get("ASSAY_WORKER_INTERVAL_S", "300"))
    limit = int(os.environ.get("ASSAY_WORKER_REFINE_LIMIT", "100"))

    stop = threading.Event()
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, lambda *_: stop.set())
    while not stop.is_set():
        for tenant, result in run_cycle(tenants, open_scoring, refine_limit=limit).items():
            log.info("tenant=%s %s", tenant, result)
        if args.once:
            break
        stop.wait(interval)
    pool.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
