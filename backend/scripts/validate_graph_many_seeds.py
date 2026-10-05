"""H6: does the entity graph add detection value over the same entity data used as flat features? (PRD 20, H6)

    python scripts/validate_graph_many_seeds.py run --seeds 601-612 --out ../scratch/graph
    python scripts/validate_graph_many_seeds.py summarise ../scratch/graph

SIMULATED RINGS. The standard synthetic dataset gets the entity-graph scenario (assay/synthetic/generator.py): five rings
of customers who commit fraud together through shared devices, IP addresses and beneficiaries, each transaction looking
ordinary on its own; 60 families who share a device legitimately; and three public IP addresses used by hundreds of
unrelated customers. The rings are the experiment's own construction, so this tests that the graph can find structure
that is there, and how it compares with a join on the same data. It is NOT evidence that real fraud has this shape
(PRD 13.6: these are hypotheses, not demonstrated capabilities).

ARMS, trained the same way and scored on the same verified-outcome-only test window (assay/validation/graph.py):
  F0 registry features;  F1 F0 + flat entity features (30-day counts of distinct other customers per device, IP and
  beneficiary, and how many of them have a confirmed fraud known at the time);  G  F0 + graph features (confidence
  weighted, decayed, specificity-discounted, two hops, connected groups).
Metric: PR-AUC on calibrated probabilities. "Ring cases" is PR-AUC on legitimate rows plus only ring frauds; "other fraud"
is legitimate rows plus only the other fraud types. Differences are paired by seed; intervals are 95% t-intervals.

DECISION RULE (written before seeds 601-612 were run; a smoke run on seed 600, not in the set, only checked that it runs
and how long it takes):
  R1 Better overall.            lower end of the interval of (G - F1) on all cases is above 0.
  R2 Better where the rings are. lower end of (G - F1) on ring cases is above 0.
  R3 No harm elsewhere.          lower end of (G - F1) on other fraud is above -0.01.
  VERDICT. "Supported on simulated rings" only if R1, R2 and R3 all hold. Otherwise the graph stays as investigator
  context only (PRD 20, H6), and which rule failed is reported.
  Reported with no threshold: G - F0 and F1 - F0 (what flat entity data alone adds), the share of derived links at the end
  of the test window that join unrelated customers (false links) and which relationships they come through, and the time
  to build the features.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tests"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from validate_many_seeds import parse_seeds, t_interval


def run_seed(seed: int) -> dict:
    from assay.detection.metrics import pr_auc
    from assay.synthetic import generate, to_wire
    from assay.validation import graph as V
    from conftest_detection import GEN, make_dataset

    ds, txns, cfg = make_dataset(seed)  # the standard config and windows; only the dataset differs below
    ds = generate(replace(GEN, seed=seed, graph_scenario=True))
    txns = [to_wire(t) for t in ds.transactions]
    t0 = time.time()
    prep = V.prepare(txns, ds.matured_outcomes(cfg.as_of), cfg)
    build_s = time.time() - t0
    out = V.arms(prep, cfg)
    ids, y = out["F0"]["ids"], out["F0"]["y"]
    assert all(out[a]["ids"] == ids for a in out), "all arms must be scored on the same rows"
    res: dict = {"seed": seed, "n_test": len(y), "n_test_fraud": int(y.sum()),
                 "n_test_ring_fraud": int(sum(1 for i, v in zip(ids, y, strict=True) if v and ds.truth[i]["fraud_type"] == "ring_attack")),
                 "feature_build_seconds": build_s, "arms": {}}
    for name, a in out.items():
        res["arms"][name] = {"all": pr_auc(y, a["p"]),
                             "ring": V.subset_pr_auc(y, a["p"], ids, ds.truth, ring=True),
                             "other": V.subset_pr_auc(y, a["p"], ids, ds.truth, ring=False)}
    res["links"] = V.false_links(txns, ds.structure, cfg.test_end.timestamp())
    return res


def run(seeds: list[int], out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    for seed in seeds:
        f = out / f"seed-{seed}.json"
        if f.exists():
            print(f"seed {seed}: already done", flush=True)
            continue
        t0 = time.time()
        f.write_text(json.dumps(run_seed(seed), default=float), encoding="utf-8")
        print(f"seed {seed}: done in {time.time() - t0:.0f}s", flush=True)


def summarise(folder: Path) -> dict:
    reps = {int(p.stem.split("-")[1]): json.loads(p.read_text(encoding="utf-8"))
            for p in sorted(folder.glob("seed-*.json"))}
    if not reps:
        raise SystemExit("no seed reports found")
    seeds = sorted(reps)

    def col(arm, part):
        return np.array([reps[s]["arms"][arm][part] for s in seeds], dtype=float)

    def diff(a, b, part):
        d = col(a, part) - col(b, part)
        d = d[np.isfinite(d)]
        lo, hi = t_interval(d)
        return {"mean": float(d.mean()), "t_ci": [lo, hi], "seeds_ahead": int((d > 0).sum()), "n": len(d),
                "per_seed": d.round(4).tolist()}

    res: dict = {"seeds": seeds, "n_seeds": len(seeds),
                 "mean_pr_auc": {a: {p: float(np.nanmean(col(a, p))) for p in ("all", "ring", "other")} for a in ("F0", "F1", "G")},
                 "g_minus_f1": {p: diff("G", "F1", p) for p in ("all", "ring", "other")},
                 "g_minus_f0": {p: diff("G", "F0", p) for p in ("all", "ring", "other")},
                 "f1_minus_f0": {p: diff("F1", "F0", p) for p in ("all", "ring", "other")},
                 "test_ring_fraud_mean": float(np.mean([reps[s]["n_test_ring_fraud"] for s in seeds])),
                 "test_fraud_mean": float(np.mean([reps[s]["n_test_fraud"] for s in seeds])),
                 "feature_build_seconds_mean": float(np.mean([reps[s]["feature_build_seconds"] for s in seeds]))}
    shares = [reps[s]["links"]["false_share"] for s in seeds if reps[s]["links"]["false_share"] is not None]
    rels: dict[str, list[int]] = {}
    for s in seeds:
        for rel, v in reps[s]["links"]["by_relationship"].items():
            acc = rels.setdefault(rel, [0, 0])
            acc[0] += v["true"]
            acc[1] += v["false"]
    res["false_links"] = {"mean_false_share": float(np.mean(shares)) if shares else None,
                          "by_relationship": {k: {"true": v[0], "false": v[1]} for k, v in rels.items()},
                          "mean_links": float(np.mean([reps[s]["links"]["links"] for s in seeds]))}
    r1 = bool(res["g_minus_f1"]["all"]["t_ci"][0] > 0)
    r2 = bool(res["g_minus_f1"]["ring"]["t_ci"][0] > 0)
    r3 = bool(res["g_minus_f1"]["other"]["t_ci"][0] > -0.01)
    res["rule"] = {"R1_better_overall": r1, "R2_better_on_ring_cases": r2, "R3_no_harm_on_other_fraud": r3}
    res["verdict"] = ("Supported on simulated rings" if all((r1, r2, r3)) else
                      "NOT supported as specified: failed " + ", ".join(k for k, v in res["rule"].items() if not v)
                      + ". The graph stays investigator context only.")
    return res


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("run")
    a.add_argument("--seeds", required=True)
    a.add_argument("--out", required=True)
    b = sub.add_parser("summarise")
    b.add_argument("folder")
    args = ap.parse_args()
    if args.cmd == "run":
        run(parse_seeds(args.seeds), Path(args.out))
    else:
        print(json.dumps(summarise(Path(args.folder)), indent=2))


if __name__ == "__main__":
    main()
