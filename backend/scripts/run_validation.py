"""Run the trust validation on a synthetic dataset and write a report.

    python scripts/run_validation.py --seed 99 --n 100000 --out ../docs/validation/seed-99 [--no-stress]

Everything is deterministic given the seed. Synthetic data proves the machinery works; it is not
evidence that Assay works on real fraud. Writes report.json, stress.json and report.md.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tests"))

from assay.detection import train_bundle
from assay.validation.harness import collect_cases, run_validation
from assay.validation.report import render_markdown
from assay.validation.stress import novel_type_summary, run_stress_tests
from conftest_detection import make_dataset


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=None, help="dataset seed (default: the dev seed)")
    ap.add_argument("--n", type=int, default=3500, help="test cases to assess (capped at the test window)")
    ap.add_argument("--out", type=Path, default=Path("validation-out"))
    ap.add_argument("--no-stress", action="store_true")
    args = ap.parse_args()

    t0 = time.time()
    ds, txns, cfg = make_dataset(args.seed)
    r = train_bundle(txns, ds.outcomes, cfg)
    by_id = {t["txn_id"]: t for t in txns}
    cases = collect_cases(r, by_id, args.n)
    report = run_validation(cases, bundle_id=r.manifest.bundle_id, dataset_id=r.manifest.dataset_id)
    stress = None
    if not args.no_stress:
        stress = run_stress_tests(r, ds, txns, cfg, by_id)
        stress["novel_fraud_type"] = novel_type_summary(cases, ds.truth) | {
            "expected": "never-seen fraud lands in Low or Insufficient evidence", "passed": None}
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "report.json").write_text(json.dumps(report, indent=1), encoding="utf-8")
    if stress is not None:
        (args.out / "stress.json").write_text(json.dumps(stress, indent=1, default=float), encoding="utf-8")
    md = render_markdown(report, stress)
    md = f"_Synthetic dataset seed {args.seed if args.seed is not None else 'dev'}; " \
         f"{len(cases.assessments)} test cases; run time {time.time() - t0:.0f}s._\n\n" + md
    (args.out / "report.md").write_text(md, encoding="utf-8")
    print(md)


if __name__ == "__main__":
    main()
