"""Is the calibrated Trust Index (PRD 5.6) worth enabling? Run it on many independent synthetic datasets.

    python scripts/validate_calibrated_many_seeds.py run --seeds 301-312 --out ../scratch/cal
    python scripts/validate_calibrated_many_seeds.py summarise ../scratch/cal

Each seed uses a LONGER synthetic dataset (a 135-day test window, novel fraud starting at day 270). The
test window is split in time: the meta-model is FITTED on the oldest 45%, CALIBRATED on the next 20%,
and EVALUATED on the newest 35%. Explanation testing runs on a random 25% of cases (as OPD-7 sampling
would), so most cases are scored by the model without `exp`.

DESIGN NOTE. The first version used the standard dataset (35-day test window, novel fraud 5 days in).
A smoke run on seed 300 (not part of the set below) showed only 10 of its 145 errors fell in the oldest
45% and 1 in the calibration slice, because almost all errors come after the novel type appears. The
model rightly refused to fit and the experiment could not answer anything, so the dataset was changed
BEFORE the seeds below were run. Nothing about seeds 301-312 informed it.

DECISION RULE (written before seeds 301-312 were run, so it cannot be chosen after seeing them):
  The evaluation window has two parts: BEFORE the novel fraud type appears (the stable part, where a
  calibration is claimed to hold, PRD 4.4) and AFTER (the stress: something new after fitting).
  P1 Calibration, on the stable part. SUPPORTED if (a) the mean ECE of the calibrated Trust Index is at
     most 0.03 (a parameter), AND (b) it is lower than the ECE of the PROVISIONAL score read as a
     probability, with the 95% t-interval of the paired difference entirely above zero.
  P2 Discrimination, on the stable part. NON-INFERIOR if the lower end of the 95% t-interval of the
     mean AUROC(calibrated) - AUROC(provisional) is above -0.02. BETTER if it is above zero.
  P3 Novelty stress, after the novel type appears. Calibrated mode must not be materially worse than
     provisional at ranking errors: same non-inferiority rule on the AFTER part.
  VERDICT. "Worth enabling for a pilot" only if P1 is supported AND P2 and P3 are at least
     non-inferior. Otherwise: stay in Provisional mode.
  Reported, with no threshold: the mean gap to B3 on each part; seeds passing the PRD 6.6 gate on
  their own; seeds with too little evidence to fit; the High band's observed error against the 1% it
  was set from; calibration on the AFTER part.
Synthetic data only: nothing here says anything about real fraud (PRD 28.3).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tests"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from validate_many_seeds import boot_interval, parse_seeds, t_interval


def run(seeds: list[int], out: Path, n_boot: int) -> None:
    from assay.detection import train_bundle
    from assay.trust.calibrated import CalibratedFitError
    from assay.validation.calibrated import CalibratedValidationConfig, evaluate_calibrated
    from assay.validation.harness import collect_cases
    from conftest_detection import make_long_dataset

    out.mkdir(parents=True, exist_ok=True)
    for seed in seeds:
        f = out / f"seed-{seed}.json"
        if f.exists():
            print(f"seed {seed}: already done", flush=True)
            continue
        t0 = time.time()
        ds, txns, cfg, novel_start = make_long_dataset(seed)
        r = train_bundle(txns, ds.outcomes, cfg)
        cases = collect_cases(r, {t["txn_id"]: t for t in txns}, 10**6, explain_frac=0.25)
        try:
            _, report = evaluate_calibrated(cases, vcfg=CalibratedValidationConfig(n_boot=n_boot),
                                            slice_at=novel_start.timestamp())
            report["status"] = "fitted"
        except CalibratedFitError as e:
            report = {"status": "refused", "reason": str(e), "n_cases": len(cases.assessments),
                      "n_wrong": int(cases.wrong.sum())}
        f.write_text(json.dumps(report, default=float), encoding="utf-8")
        print(f"seed {seed}: {report['status']} in {time.time() - t0:.0f}s", flush=True)


def summarise(folder: Path) -> dict:
    reports = {int(p.stem.split("-")[1]): json.loads(p.read_text(encoding="utf-8"))
               for p in sorted(folder.glob("seed-*.json"))}
    if not reports:
        raise SystemExit("no seed reports found")
    fitted = {s: r for s, r in reports.items() if r["status"] == "fitted"}
    refused = {s: r["reason"] for s, r in reports.items() if r["status"] == "refused"}
    res: dict = {"seeds": sorted(reports), "n_seeds": len(reports), "n_fitted": len(fitted), "refused": refused}
    # Only seeds where the slice has enough to judge count towards that slice's question.
    usable = {part: {s: r["slices"][part] for s, r in fitted.items()
                     if "cal_minus_prov" in r.get("slices", {}).get(part, {}) and "ece_calibrated" in r["slices"][part]}
              for part in ("before", "after")}
    res["usable_seeds"] = {k: len(v) for k, v in usable.items()}
    if len(usable["before"]) < 3 or len(usable["after"]) < 3:
        res["verdict"] = "too few seeds could be judged to say anything"
        return res

    def col(part, *path):
        out = []
        for sl in usable[part].values():
            for k in path:
                sl = sl[k]
            out.append(sl)
        return np.array(out, dtype=float)

    def diff_block(part, a, b):
        d = col(part, *a) - col(part, *b) if b else col(part, *a)
        lo, hi = t_interval(d)
        return {"mean": float(d.mean()), "t_ci": [lo, hi], "per_seed": d.round(4).tolist(),
                "seeds_ahead": int((d > 0).sum()), "n": len(d)}

    ece_c = col("before", "ece_calibrated")
    ece_p = col("before", "ece_provisional_read_as_probability")
    d_ece = ece_p - ece_c
    elo, ehi = t_interval(d_ece)
    p1a, p1b = bool(ece_c.mean() <= 0.03), bool(elo > 0)
    before = diff_block("before", ("cal_minus_prov", "diff"), None)
    after = diff_block("after", ("cal_minus_prov", "diff"), None)

    def classify(blk):
        return "BETTER" if blk["t_ci"][0] > 0 else ("NON-INFERIOR" if blk["t_ci"][0] > -0.02 else "WORSE or unresolved")

    p2, p3 = classify(before), classify(after)
    ht = [s["high_band"]["observed_error_rate"] for s in usable["before"].values() if s["high_band"]["observed_error_rate"] is not None]
    ece_after = [(s["ece_calibrated"], s["ece_provisional_read_as_probability"]) for s in usable["after"].values()]
    res |= {
        "stable_part": {
            "ece_calibrated": {"mean": float(ece_c.mean()), "per_seed": ece_c.round(4).tolist()},
            "ece_provisional_as_probability": {"mean": float(ece_p.mean())},
            "ece_improvement": {"mean": float(d_ece.mean()), "t_ci": [elo, ehi], "boot_ci": list(boot_interval(d_ece))},
            "auroc_cal_minus_prov": before,
            "auroc_cal_minus_B3": diff_block("before", ("cal_minus_B3", "diff"), None),
            "auroc_calibrated": float(col("before", "auroc_calibrated").mean()),
            "auroc_provisional": float(col("before", "auroc_provisional").mean()),
            "high_band_observed_error": float(np.mean(ht)) if ht else None,
        },
        "novel_part": {
            "auroc_cal_minus_prov": after,
            "auroc_cal_minus_B3": diff_block("after", ("cal_minus_B3", "diff"), None),
            "ece_calibrated_mean": float(np.mean([a for a, _ in ece_after])),
            "ece_provisional_as_probability_mean": float(np.mean([b for _, b in ece_after])),
        },
        "gate_passed_seeds": sum(bool(r["gate"]["passed"]) for r in fitted.values()),
        "mean_interval_width": float(np.mean([r["mean_interval_width"] for r in fitted.values()])),
        "P1_calibration": "supported" if (p1a and p1b) else "not supported",
        "P1_detail": {"mean_ece_within_0.03": p1a, "better_than_provisional_as_probability": p1b},
        "P2_discrimination_stable": p2, "P3_discrimination_after_novelty": p3,
    }
    ok = (p1a and p1b) and p2 in ("BETTER", "NON-INFERIOR") and p3 in ("BETTER", "NON-INFERIOR")
    res["verdict"] = "WORTH ENABLING FOR A PILOT" if ok else "STAY IN PROVISIONAL MODE"
    return res


def render(r: dict) -> str:
    lines = [(f"Seeds: {r['n_seeds']} ({r['n_fitted']} fitted, {len(r['refused'])} refused for too little evidence); "
              f"judged on the stable part in {r['usable_seeds']['before']}, after novelty in {r['usable_seeds']['after']}")]
    for s, why in r["refused"].items():
        lines.append(f"  seed {s} refused: {why}")
    if "stable_part" not in r:
        return "\n".join(lines + [f"VERDICT: {r['verdict']}"])
    sp, npart = r["stable_part"], r["novel_part"]
    ei, bp, bb3 = sp["ece_improvement"], sp["auroc_cal_minus_prov"], sp["auroc_cal_minus_B3"]
    np_, nb3 = npart["auroc_cal_minus_prov"], npart["auroc_cal_minus_B3"]

    def ci(b):
        return f"{b['mean']:+.4f}  t-CI [{b['t_ci'][0]:+.4f}, {b['t_ci'][1]:+.4f}]  ahead in {b['seeds_ahead']}/{b['n']}"

    lines += [
        "", f"P1 CALIBRATION (stable part): {r['P1_calibration'].upper()}   {r['P1_detail']}",
        f"   mean ECE, calibrated:                        {sp['ece_calibrated']['mean']:.4f}  (tolerance 0.03)",
        f"   mean ECE, provisional read as a probability: {sp['ece_provisional_as_probability']['mean']:.4f}",
        f"   improvement {ei['mean']:+.4f}  t-CI [{ei['t_ci'][0]:+.4f}, {ei['t_ci'][1]:+.4f}]",
        "", f"P2 DISCRIMINATION (stable part): {r['P2_discrimination_stable']}",
        f"   AUROC calibrated {sp['auroc_calibrated']:.3f} vs provisional {sp['auroc_provisional']:.3f}: calibrated minus provisional {ci(bp)}",
        "", f"P3 AFTER THE NOVEL FRAUD TYPE APPEARS: {r['P3_discrimination_after_novelty']}",
        f"   calibrated minus provisional {ci(np_)}",
        "", "Reported, no threshold:",
        f"   calibrated minus B3, stable part:  {ci(bb3)}",
        f"   calibrated minus B3, after novelty: {ci(nb3)}",
        f"   calibration after novelty: ECE calibrated {npart['ece_calibrated_mean']:.4f} vs provisional-as-probability {npart['ece_provisional_as_probability_mean']:.4f}",
        f"   seeds passing the PRD 6.6 gate on their own: {r['gate_passed_seeds']}/{r['n_fitted']}",
        (f"   High band observed error (stable part): {sp['high_band_observed_error']:.4f} (set from a tolerated 1%)"
         if sp["high_band_observed_error"] is not None else "   High band: no cases"),
        f"   mean interval width: {r['mean_interval_width']:.2f} points",
        "", f"VERDICT: {r['verdict']}"]
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--seeds", required=True)
    r.add_argument("--out", type=Path, required=True)
    r.add_argument("--n-boot", type=int, default=200)
    s = sub.add_parser("summarise")
    s.add_argument("folder", type=Path)
    s.add_argument("--json", action="store_true")
    a = ap.parse_args()
    if a.cmd == "run":
        run(parse_seeds(a.seeds), a.out, a.n_boot)
    else:
        res = summarise(a.folder)
        print(json.dumps(res, indent=1) if a.json else render(res))


if __name__ == "__main__":
    main()
