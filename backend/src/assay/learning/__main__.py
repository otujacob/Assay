"""Learning jobs: `python -m assay.learning create|pool ...`. Training is heavy, so it runs here and not in the API.

  python -m assay.learning pool   --tenant T
  python -m assay.learning create --tenant T --actor NAME --as-of 2026-06-01 --horizon-days 40 \\
      --train-end 2026-03-01 --calibration-end 2026-03-31 --reliability-end 2026-05-01 --test-end 2026-06-01
  (add --no-feedback to train on verified outcomes only)

`create` trains a candidate on verified outcomes plus the accepted feedback pool, signs and stores it, scores it
against the champion on the verified-outcome-only holdout, and records the gate report. It promotes nothing: the
candidate then needs shadow, a second person's approval, a canary and a promotion (PRD 12.2), each taken in the
web app or the API.

Environment: ASSAY_DATABASE_URL, ASSAY_BUNDLES, ASSAY_BUNDLE_SIGNING_KEY, ASSAY_MASTER_KEY (as for the API) and
ASSAY_CANDIDATE_DIR, where candidate bundles are written (without it the integrity gate G8 fails).
"""

from __future__ import annotations

import argparse
import json
import os
from datetime import UTC, datetime

from assay.detection import TrainingConfig
from assay.learning.service import LearningConfig, LearningService


def _when(s: str) -> datetime:
    d = datetime.fromisoformat(s)
    return d if d.tzinfo else d.replace(tzinfo=UTC)


def load_training_data(repo, tenant_id: str) -> tuple[list[dict], list[dict]]:
    """Every stored transaction and outcome for the tenant, in the shape the training code reads."""
    with repo.atomic(tenant_id):
        return repo.find(tenant_id, "transactions"), repo.find(tenant_id, "outcomes")


def create(scoring, tenant_id: str, actor: str, training: TrainingConfig, cfg: LearningConfig, *,
           use_feedback: bool = True) -> dict:
    txns, outcomes = load_training_data(scoring.repo, tenant_id)
    with scoring.repo.atomic(tenant_id):
        return LearningService(scoring.repo, scoring, cfg).create_candidate(
            tenant_id, actor, txns=txns, outcomes=outcomes, training=training, use_feedback=use_feedback)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="assay.learning")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("pool")
    p.add_argument("--tenant", required=True)
    c = sub.add_parser("create")
    c.add_argument("--tenant", required=True)
    c.add_argument("--actor", required=True, help="who is creating it; they cannot approve it later")
    c.add_argument("--no-feedback", action="store_true")
    c.add_argument("--horizon-days", type=float, required=True)
    for name in ("as-of", "train-end", "calibration-end", "test-end"):
        c.add_argument(f"--{name}", required=True)
    c.add_argument("--reliability-end")
    args = ap.parse_args(argv)

    from psycopg_pool import ConnectionPool

    from assay.api.main import (
        key_provider_from_env,
        load_registry,
        make_scoring_provider,
        refuse_superuser,
    )

    key = os.environ["ASSAY_BUNDLE_SIGNING_KEY"].encode()
    provider = key_provider_from_env()
    registry = load_registry(json.loads(os.environ["ASSAY_BUNDLES"]), key, provider)
    pool = ConnectionPool(os.environ["ASSAY_DATABASE_URL"], min_size=1, max_size=2, open=True)
    refuse_superuser(pool)
    cfg = LearningConfig(bundle_dir=os.environ.get("ASSAY_CANDIDATE_DIR") or None, signing_key=key,
                         key_provider=provider)
    try:
        with make_scoring_provider(pool, registry)() as scoring:
            if args.cmd == "pool":
                with scoring.repo.atomic(args.tenant):
                    out = LearningService(scoring.repo, scoring, cfg).pool_summary(args.tenant)
            else:
                tc = TrainingConfig(
                    tenant_id=args.tenant, as_of=_when(args.as_of), horizon_days=args.horizon_days,
                    train_end=_when(args.train_end), calibration_end=_when(args.calibration_end),
                    test_end=_when(args.test_end),
                    reliability_end=_when(args.reliability_end) if args.reliability_end else None)
                out = create(scoring, args.tenant, args.actor, tc, cfg, use_feedback=not args.no_feedback)
    finally:
        pool.close()
    print(json.dumps(out, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
