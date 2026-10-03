"""Run the Trust Index validation on many independent synthetic datasets and combine the results.

One synthetic dataset gives only about 50 wrong recommendations, so a single run cannot tell a real
advantage over the max-probability baseline (B3) from noise. This runs the same harness on many seeds,
each a separate synthetic "institution", and looks at the spread of the Trust Index minus B3 difference.

    python scripts/validate_many_seeds.py run --seeds 201-212 --out ../scratch/multi
    python scripts/validate_many_seeds.py summarise ../scratch/multi

DECISION RULE (fixed before the fresh seeds were run, so it cannot be chosen after seeing them):
  * Primary: the mean over seeds of AUROC(Trust Index) - AUROC(B3), for ranking wrong recommendations.
    SUPPORTED if the 95% interval of that mean excludes zero on the positive side; NOT SUPPORTED if it
    includes zero or the Trust Index is behind. The interval is a t interval over seeds (each seed is
    an independent draw), cross-checked with a bootstrap over seeds.
  * Secondary: the same for each component's ablation (AUROC without it minus AUROC with it).
    A component whose interval is entirely above zero hurts and should be dropped or down-weighted
    (PRD 6.4); entirely below zero it helps; including zero it is unresolved.
Synthetic data only. Whatever this finds says nothing about real fraud (PRD 28.3).
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tests"))


def parse_seeds(spec: str) -> list[int]:
    out: list[int] = []
    for part in spec.split(","):
        lo, _, hi = part.partition("-")
        out += list(range(int(lo), int(hi or lo) + 1))
    return out


def run(seeds: list[int], out: Path, n_boot: int) -> None:
    from assay.detection import train_bundle
    from assay.validation.harness import collect_cases, run_validation
    from conftest_detection import make_dataset

    out.mkdir(parents=True, exist_ok=True)
    for seed in seeds:
        f = out / f"seed-{seed}.json"
        if f.exists():
            print(f"seed {seed}: already done", flush=True)
            continue
        t0 = time.time()
        ds, txns, cfg = make_dataset(seed)
        r = train_bundle(txns, ds.outcomes, cfg)
        by_id = {t["txn_id"]: t for t in txns}
        cases = collect_cases(r, by_id, 10**6)  # the whole test window
        report = run_validation(cases, bundle_id=r.manifest.bundle_id, dataset_id=r.manifest.dataset_id, n_boot=n_boot)
        f.write_text(json.dumps(report), encoding="utf-8")
        print(f"seed {seed}: {report['meta']['n_cases']} cases, {report['meta']['n_wrong']} wrong, "
              f"done in {time.time() - t0:.0f}s", flush=True)


def t_interval(x: np.ndarray, level: float = 0.95) -> tuple[float, float]:
    from scipy import stats

    n = len(x)
    if n < 2:
        return float("nan"), float("nan")
    half = stats.t.ppf(0.5 + level / 2, n - 1) * x.std(ddof=1) / math.sqrt(n)
    return float(x.mean() - half), float(x.mean() + half)


def boot_interval(x: np.ndarray, n_boot: int = 20000, seed: int = 0) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    means = rng.choice(x, size=(n_boot, len(x)), replace=True).mean(axis=1)
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def summarise(folder: Path) -> dict:
    from scipy import stats

    reports = {int(p.stem.split("-")[1]): json.loads(p.read_text(encoding="utf-8"))
               for p in sorted(folder.glob("seed-*.json"))}
    if not reports:
        raise SystemExit("no seed reports found")
    seeds = sorted(reports)
    n_cases = np.array([reports[s]["meta"]["n_cases"] for s in seeds])
    n_wrong = np.array([reports[s]["meta"]["n_wrong"] for s in seeds])

    def col(path):
        out = []
        for s in seeds:
            v = reports[s]
            for k in path:
                v = v[k]
            out.append(v)
        return np.array(out, dtype=float)

    ti = col(["measures", "auroc", "value"])
    result = {"seeds": seeds, "n_seeds": len(seeds), "cases_total": int(n_cases.sum()),
              "wrong_total": int(n_wrong.sum()),
              "auroc_trust_index": {"mean": float(ti.mean()), "per_seed": ti.round(3).tolist()}}
    base = {}
    for name in ("B1_distance_from_threshold", "B2_ensemble_disagreement", "B3_max_class_probability"):
        d = col(["baselines", name, "trust_index_minus_baseline", "diff"])
        lo, hi = t_interval(d)
        blo, bhi = boot_interval(d)
        wins = int((d > 0).sum())
        base[name] = {"mean_diff": float(d.mean()), "t_ci": [lo, hi], "boot_ci": [blo, bhi], "seeds_ahead": wins,
                      "sign_test_p": float(stats.binomtest(wins, len(d), 0.5).pvalue),
                      "per_seed": d.round(3).tolist(),
                      "verdict": "supported" if lo > 0 else ("behind" if hi < 0 else "not supported (interval includes zero)")}
    result["vs_baselines"] = base
    abl = {}
    for c in ("conf", "rel", "exp", "fam", "drift", "dq"):
        d = col(["ablations", c, "delta_auroc"])
        lo, hi = t_interval(d)
        # The rule as written in the docstring, with no extra margin: entirely below zero it helps,
        # entirely above zero it hurts, otherwise it is unresolved. Effect size is reported separately.
        if not d.any():
            reading = "no effect in any seed (inactive on this data by construction)"
        elif hi < 0:
            reading = "helps (removing it lowers AUROC)"
        elif lo > 0:
            reading = "hurts (removing it RAISES AUROC): drop or down-weight, PRD 6.4"
        else:
            reading = "unresolved (interval includes zero)"
        size = "small" if abs(d.mean()) < 0.01 else "moderate" if abs(d.mean()) < 0.03 else "large"
        abl[c] = {"mean_delta": float(d.mean()), "t_ci": [lo, hi], "per_seed": d.round(3).tolist(),
                  "seeds_removal_raised_auroc": int((d > 0).sum()), "effect_size": size, "reading": reading}
    result["ablations"] = abl
    ht = col(["measures", "high_trust_error_rate", "value"])
    overall = col(["overall_error_rate", "value"])
    result["high_trust_error_rate"] = {"mean": float(np.nanmean(ht)), "overall_mean": float(np.nanmean(overall))}
    return result


def render(res: dict) -> str:
    b3 = res["vs_baselines"]["B3_max_class_probability"]
    lines = [f"Seeds: {res['n_seeds']} ({res['seeds'][0]} to {res['seeds'][-1]}); cases {res['cases_total']:,}"
             + (f"; wrong recommendations {res['wrong_total']:,}" if res["wrong_total"] else ""),
             f"Mean AUROC of the Trust Index at ranking errors: {res['auroc_trust_index']['mean']:.3f}", "",
             "Trust Index minus each baseline (AUROC), mean over seeds, 95% interval over seeds:"]
    for name, v in res["vs_baselines"].items():
        lines.append(f"  {name:28s} {v['mean_diff']:+.4f}  t-CI [{v['t_ci'][0]:+.4f}, {v['t_ci'][1]:+.4f}]"
                     f"  boot-CI [{v['boot_ci'][0]:+.3f}, {v['boot_ci'][1]:+.3f}]  ahead in {v['seeds_ahead']}/{res['n_seeds']}"
                     f"  sign-test p={v['sign_test_p']:.3f}  -> {v['verdict']}")
    lines += ["", "Ablation (AUROC without the component minus with it; positive = removing it helps):"]
    for c, v in res["ablations"].items():
        lines.append(f"  {c:6s} {v['mean_delta']:+.4f}  t-CI [{v['t_ci'][0]:+.4f}, {v['t_ci'][1]:+.4f}]  "
                     f"removal raised AUROC in {v['seeds_removal_raised_auroc']}/{res['n_seeds']} seeds  "
                     f"({v['effect_size']} effect)  -> {v['reading']}")
    lines += ["", f"PRIMARY (Trust Index vs B3): {b3['verdict'].upper()}"]
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--seeds", required=True, help="e.g. 201-212 or 201,205,210-212")
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
