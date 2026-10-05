"""Fit a calibrated Trust Index for a model bundle, run the section 6 gate, and save the bundle.

    python scripts/build_calibrated_bundle.py --seed 300 --out ../scratch/cal-bundle
    python scripts/build_calibrated_bundle.py --seed 300 --out ../scratch/recal-bundle --method recalibrated

This builds a SYNTHETIC bundle from a generated dataset, to show the workflow. For a real tenant the
same steps run on that tenant's own matured verified outcomes (PRD 5.6), and never on another tenant's.

What it does:
  1. trains the detection model and the Provisional Trust Index components,
  2. assesses the later cases that have matured outcomes (explanation testing on a sample, as under OPD-7),
  3. fits the meta-model on the oldest part, calibrates it on the next, and evaluates on the newest (with
     --method recalibrated: fits only a monotone map from the provisional index to a probability on the oldest 65%),
  4. applies the PRD 6.6 gate and records the result INSIDE the meta-model,
  5. saves the signed bundle with the meta-model attached, and writes the validation report.

The bundle is written whether or not the gate passed, because the result is evidence either way. But a
server asked to run that tenant in calibrated mode REFUSES to start unless the gate passed, so a failed
gate cannot be switched on by accident:

    ASSAY_BUNDLES='[{"tenant_id": "...", "path": "...", "trust_mode": "calibrated"}]'
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tests"))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, required=True, help="synthetic dataset seed")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--method", choices=("meta", "recalibrated"), default="meta",
                    help="meta: the logistic meta-model (trust/calibrated.py). recalibrated: keep the provisional order and "
                         "only map it to a probability (trust/recalibrated.py)")
    ap.add_argument("--ece-tolerance", type=float, default=0.03, help="calibration error tolerance (PRD 6.6)")
    ap.add_argument("--signing-key", default=os.environ.get("ASSAY_BUNDLE_SIGNING_KEY", "dev-signing-key"))
    a = ap.parse_args()

    from assay.detection import save_bundle, train_bundle
    from assay.trust.calibrated import CalibratedFitError
    from assay.validation.calibrated import CalibratedValidationConfig, evaluate_calibrated
    from assay.validation.harness import collect_cases
    from conftest_detection import make_long_dataset

    ds, txns, cfg, novel_start = make_long_dataset(a.seed)
    r = train_bundle(txns, ds.outcomes, cfg)
    cases = collect_cases(r, {t["txn_id"]: t for t in txns}, 10**6, explain_frac=0.25)
    fit_model = None
    if a.method == "recalibrated":
        import numpy as np

        from assay.trust.recalibrated import RecalibratedConfig, RecalibratedTrustModel

        def fit_model(fc, fy, cc, cy):  # one map, fitted on both earlier windows together
            return RecalibratedTrustModel.fit(fc + cc, np.concatenate([fy, cy]), RecalibratedConfig())
    try:
        model, report = evaluate_calibrated(
            cases, vcfg=CalibratedValidationConfig(ece_tolerance=a.ece_tolerance), slice_at=novel_start.timestamp(),
            fit_model=fit_model)
    except CalibratedFitError as e:
        print(f"NOT FITTED: {e}\nStaying in Provisional mode; the bundle is not written.")
        return 2

    artefact = r.scoring_bundle()
    artefact["trust_meta"] = model
    manifest = r.manifest
    manifest.extra["trust_meta"] = {"version": model.cfg.version, "gate": model.gate, "n_fit": model.n_fit,
                                    "n_calibration": model.n_calibration}
    a.out.mkdir(parents=True, exist_ok=True)
    save_bundle(a.out / "bundle", artefact, manifest, a.signing_key.encode())
    (a.out / "report.json").write_text(json.dumps(report, indent=1, default=float), encoding="utf-8")

    g = model.gate
    print(f"Gate {'PASSED' if g['passed'] else 'FAILED'} on {g['evaluated_on']} cases "
          f"({g['wrong_in_evaluation']} wrong recommendations)")
    for k, v in g["criteria"].items():
        print(f"  {k}: {v}")
    print(f"ECE: calibrated {report['ece']['calibrated']:.4f} vs provisional read as a probability "
          f"{report['ece']['provisional_read_as_probability']:.4f}")
    print(f"Bundle written to {a.out / 'bundle'}")
    print("calibrated mode can be switched on for it." if g["passed"] else
          "calibrated mode is REFUSED for this bundle: it did not pass the gate.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
