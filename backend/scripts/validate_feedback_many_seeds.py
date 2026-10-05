"""H4: does scoring feedback before learning beat learning from every decision? (PRD 20, H4, section 11)

    python scripts/validate_feedback_many_seeds.py run --seeds 401-412 --out ../scratch/fb
    python scripts/validate_feedback_many_seeds.py summarise ../scratch/fb

SIMULATED ANALYSTS (assay/validation/analysts.py). This tests that the pipeline behaves as specified when
fed labels of known quality. It is NOT evidence that the feedback engine works on real analysts (PRD 19,
OPD-20), and the assumptions about how analysts behave are the experiment's, not the world's.

SETUP. Standard synthetic dataset, one per seed. 92% of the training-window cases that have a verified
outcome have it HIDDEN (as if the case was never confirmed), and simulated analysts decide those cases
instead. Calibration and test windows are untouched: they use verified outcomes only, so every model is
scored on the same verified-outcome-only holdout that analyst labels never reach (PRD 11.4).
Arms, all trained the same way except for which labels they get:
  V       verified outcomes only (the hidden half is simply missing)
  ALL     verified + every analyst decision on the hidden cases, no quality assessment
  ACC     verified + the labels the feedback pipeline ACCEPTED (learning/pool.py, default thresholds)
  ORACLE  verified + the true labels of the hidden cases (ceiling; reported, not used in the rule)
Noise: bad_share is the share of hidden cases handled by the weak, adversarial or lazy analysts, at
0.0 (clean), 0.3 and 0.6.

DESIGN NOTE. The first version hid half of the verified outcomes. A smoke run on seed 400 showed the
ORACLE arm (every hidden label, perfectly right) scored the same as V (0.7093 against 0.7095), so with that
much verified data extra labels were worth nothing and the experiment could not say whether the pipeline
helps. Hiding 92% (verified labels scarce, which is when feedback matters) made ORACLE clearly beat V on
smoke seeds 398 and 399 (0.76 and 0.65 against 0.58 and 0.60). The design was changed BEFORE the seeds
below were run; no result from seeds 401-412 informed it.

DECISION RULE (written before seeds 401-412 were run; a smoke run on seed 400, which is not in the set,
only checked that it runs and how long it takes). Metric: PR-AUC on the verified holdout. Differences are
paired by seed; intervals are 95% t-intervals across seeds.
  R1 Not worse when feedback is clean. At bad_share 0.0: lower end of the interval of (ACC - ALL) > -0.01.
  R2 More robust to noise. At bad_share 0.6: the interval of (ACC - ALL) lies entirely above 0.
  R3 Does not hurt versus ignoring feedback. At bad_share 0.6: lower end of (ACC - V) > -0.01.
  R4 Catches noise. At bad_share 0.6: the mean error rate of accepted labels (against the hidden truth)
     is at most half the mean error rate of all analyst labels.
  VERDICT. "Supported on simulated analysts" only if R1 to R4 all hold. Otherwise report which failed.
  Reported with no threshold: acceptance rate, share of wrong labels not accepted, which analysts were
  flagged, accuracy of accepted labels, and ACC - V at every noise level.
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

from validate_many_seeds import parse_seeds, t_interval

BAD_SHARES = (0.0, 0.3, 0.6)
HIDE = 0.92


def _score(r) -> dict:
    m = r.manifest.metrics
    return {"pr_auc": float(m["pr_auc_ensemble"]), "ece": float(r.manifest.calibration["ece_test"]),
            "extra_used": int(r.manifest.extra.get("extra_labels_used", 0)),
            "recall_at_t_high": float(m["at_t_high"].get("recall", float("nan"))),
            "precision_at_t_high": float(m["at_t_high"].get("precision", float("nan")))}


def run_seed(seed: int, hide: float = HIDE) -> dict:
    from assay.detection import train_bundle
    from assay.detection.labels import labels_as_of
    from assay.learning.pool import PoolConfig, assess
    from assay.validation.analysts import SimConfig, all_feedback_labels, simulate_actions
    from conftest_detection import make_dataset

    ds, txns, cfg = make_dataset(seed)
    rng = np.random.default_rng(seed + 7_000)
    matured = ds.matured_outcomes(cfg.as_of)
    verified = labels_as_of(txns, matured, cfg.as_of, cfg.horizon_days)
    in_train = [t["txn_id"] for t in txns if t["txn_id"] in verified
                and _before(t, cfg.train_end)]
    hidden = {t for t in in_train if rng.random() < hide}
    outcomes = [o for o in matured if o["txn_id"] not in hidden]
    truth = {t: int(ds.truth[t]["is_fraud"]) for t in hidden}

    res: dict = {"seed": seed, "n_hidden": len(hidden), "n_train_verified_kept": len(in_train) - len(hidden),
                 "hidden_fraud_rate": float(np.mean(list(truth.values()))), "arms": {}, "noise": {}}
    res["arms"]["V"] = _score(train_bundle(txns, outcomes, cfg))
    res["arms"]["ORACLE"] = _score(train_bundle(txns, outcomes, cfg, extra_labels=truth))
    kept_truth = {o["txn_id"]: (1 if o["outcome_type"] in ("confirmed_fraud", "chargeback") else 0)
                  for o in outcomes}
    for bs in BAD_SHARES:
        actions = simulate_actions(txns, ds.truth, hidden, SimConfig(bad_share=bs, seed=seed), cfg.as_of)
        pool_outcomes = {t: v for t, v in kept_truth.items() if t in verified}
        rep = assess(actions, pool_outcomes, {}, cfg.as_of, PoolConfig())
        acc = rep.labels()
        allf = all_feedback_labels(actions, hidden)
        wrong_all = {t for t, v in allf.items() if v != truth[t]}
        res["noise"][str(bs)] = {
            "n_all_labels": len(allf), "n_accepted": len(acc),
            "error_rate_all": len(wrong_all) / max(1, len(allf)),
            "error_rate_accepted": sum(1 for t, v in acc.items() if t in truth and v != truth[t]) / max(1, len(acc)),
            "wrong_not_accepted": (sum(1 for t in wrong_all if t not in acc) / len(wrong_all)) if wrong_all else None,
            "acceptance_rate": rep.acceptance_rate(), "counts": rep.counts(),
            "flagged": sorted(rep.flagged), "flagged_why": rep.flagged}
        res["arms"][f"ALL@{bs}"] = _score(train_bundle(txns, outcomes, cfg, extra_labels=allf))
        res["arms"][f"ACC@{bs}"] = _score(train_bundle(txns, outcomes, cfg, extra_labels=acc))
    return res


def _before(t: dict, end) -> bool:
    from assay.features.compute import parse_time
    return parse_time(t["event_time"]) < end


def run(seeds: list[int], out: Path, hide: float = HIDE) -> None:
    out.mkdir(parents=True, exist_ok=True)
    for seed in seeds:
        f = out / f"seed-{seed}.json"
        if f.exists():
            print(f"seed {seed}: already done", flush=True)
            continue
        t0 = time.time()
        f.write_text(json.dumps(run_seed(seed, hide), default=float), encoding="utf-8")
        print(f"seed {seed}: done in {time.time() - t0:.0f}s", flush=True)


def summarise(folder: Path) -> dict:
    reps = {int(p.stem.split("-")[1]): json.loads(p.read_text(encoding="utf-8"))
            for p in sorted(folder.glob("seed-*.json"))}
    if not reps:
        raise SystemExit("no seed reports found")
    seeds = sorted(reps)

    def arm(name, key="pr_auc"):
        return np.array([reps[s]["arms"][name][key] for s in seeds])

    def diff(a, b, key="pr_auc"):
        d = arm(a, key) - arm(b, key)
        lo, hi = t_interval(d)
        return {"mean": float(d.mean()), "t_ci": [lo, hi], "seeds_ahead": int((d > 0).sum()),
                "per_seed": d.round(4).tolist()}

    res: dict = {"seeds": seeds, "n_seeds": len(seeds),
                 "mean_pr_auc": {k: float(arm(k).mean()) for k in reps[seeds[0]]["arms"]},
                 "hidden_cases_mean": float(np.mean([reps[s]["n_hidden"] for s in seeds])), "noise": {}}
    for bs in BAD_SHARES:
        k = str(bs)
        nz = [reps[s]["noise"][k] for s in seeds]
        wna = [n["wrong_not_accepted"] for n in nz if n["wrong_not_accepted"] is not None]
        res["noise"][k] = {
            "acc_minus_all": diff(f"ACC@{bs}", f"ALL@{bs}"), "acc_minus_v": diff(f"ACC@{bs}", "V"),
            "all_minus_v": diff(f"ALL@{bs}", "V"), "oracle_minus_acc": diff("ORACLE", f"ACC@{bs}"),
            "error_rate_all": float(np.mean([n["error_rate_all"] for n in nz])),
            "error_rate_accepted": float(np.mean([n["error_rate_accepted"] for n in nz])),
            "wrong_labels_not_accepted": float(np.mean(wna)) if wna else None,
            "acceptance_rate": float(np.mean([n["acceptance_rate"] for n in nz])),
            "n_accepted_mean": float(np.mean([n["n_accepted"] for n in nz])),
            "n_all_labels_mean": float(np.mean([n["n_all_labels"] for n in nz])),
            "flagged_analysts": sorted({a for n in nz for a in n["flagged"]}),
            "seeds_flagging_each": {a: sum(1 for n in nz if a in n["flagged"]) for a in sorted({a for n in nz for a in n["flagged"]})}}
    n0, n6 = res["noise"]["0.0"], res["noise"]["0.6"]
    r1 = bool(n0["acc_minus_all"]["t_ci"][0] > -0.01)
    r2 = bool(n6["acc_minus_all"]["t_ci"][0] > 0)
    r3 = bool(n6["acc_minus_v"]["t_ci"][0] > -0.01)
    r4 = bool(n6["error_rate_accepted"] <= 0.5 * n6["error_rate_all"])
    res["rule"] = {"R1_not_worse_when_clean": r1, "R2_more_robust_to_noise": r2,
                   "R3_does_not_hurt_vs_ignoring_feedback": r3, "R4_catches_noise": r4}
    res["verdict"] = ("Supported on simulated analysts" if all((r1, r2, r3, r4))
                      else "NOT supported as specified: failed " + ", ".join(
                          k for k, v in res["rule"].items() if not v))
    return res


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("run")
    a.add_argument("--seeds", required=True)
    a.add_argument("--out", required=True)
    a.add_argument("--hide", type=float, default=HIDE)
    b = sub.add_parser("summarise")
    b.add_argument("folder")
    args = ap.parse_args()
    if args.cmd == "run":
        run(parse_seeds(args.seeds), Path(args.out), args.hide)
    else:
        print(json.dumps(summarise(Path(args.folder)), indent=2))


if __name__ == "__main__":
    main()
