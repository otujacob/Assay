# Trust Index validation: what the harness found

These reports come from the validation harness (`backend/src/assay/validation/`) run on
**synthetic data**. Synthetic data tests that the machinery works. It is **not evidence that Assay
works on real fraud** (PRD 28.3). Regenerate with:

```
python backend/scripts/run_validation.py --seed 99 --n 100000 --out docs/validation/seed-99-fresh
```

| Run | Cases | Wrong recommendations | Files |
|---|---|---|---|
| Seed 5 (development) | 3,891 | 57 (1.46%) | `seed-5-dev/` |
| Seed 99 (fresh, not used to make design decisions) | 3,911 | 47 (1.20%) | `seed-99-fresh/` |

Each folder has `report.md` (readable), `report.json` and `stress.json`.

## Result against the PRD 6.6 pass criteria: NOT SUPPORTED as specified (both seeds)

The criteria are comparative and are not relaxed when they fail (PRD 6.6).

| Criterion | Seed 5 | Seed 99 |
|---|---|---|
| Beats B1 (distance from threshold), CI excludes no improvement | yes (+0.133) | yes (+0.137) |
| Beats B2 (ensemble disagreement), CI excludes no improvement | yes (+0.084) | yes (+0.075) |
| Beats B3 (max class probability), CI excludes no improvement | **no** (+0.020, CI -0.010 to +0.050) | **no** (+0.042, CI -0.025 to +0.099) |
| High-trust error rate materially below overall | yes: 0.28% vs 1.46% | yes: 0.38% vs 1.20% |

The Trust Index is ahead of B3 on both seeds but not by enough to exclude "no improvement". With only
47 to 57 errors per run the test has little power, so this is **neither a confirmation nor a
refutation** against B3. More matured outcomes are needed, which is what the pilot is for.

What does hold on both seeds:
- Cases given High trust are wrong 3 to 5 times less often than average.
- 70% to 82% of errors reach a human (Low trust or Insufficient evidence).
- The Trust Index clearly beats the two simplest uncertainty baselines.

## Component ablations (PRD 6.4)

| Component | Seed 5 | Seed 99 | Reading |
|---|---|---|---|
| conf (model confidence) | -0.014 | -0.049 | adds value on both |
| rel (cohort reliability) | -0.018 | -0.037 | adds value on both |
| exp (explanation reliability) | +0.005 | -0.007 | inconsistent, near zero. H3 is **not supported** by this data. |
| fam (familiarity) | -0.000 | **+0.033** | removing it improved discrimination on seed 99. Value unresolved. |
| drift, dq | 0 | 0 | no effect on clean, stationary synthetic data, **by construction**. The stress tests show the mechanisms work. |

Per PRD 6.4, a component that adds nothing is dropped or down-weighted. That is a decision for the
product owner (OPD-3), and should be made on a window separate from the one used to evaluate it.

## Stress tests (PRD 6.5)

Drift injection, degraded data and held-out fraud type all pass on both seeds. The noisy-analyst-label
test is **not testable** until the feedback quality engine (E6) exists. It is reported as such, not
skipped silently.

## Method changes made after seeing results

These were made in response to stress-test failures and **then checked on the fresh seed 99**, which
was not used to choose them. They are listed so the result is not read as if the design was fixed in
advance.

1. **Exclude the feature warm-up period from training** (first 30 days). The 30-day history features
   are still filling up at the start of the data, which made the drift monitor raise a false alarm on
   a clean test window.
2. **Novelty reference over-represents fraud.** Built from ordinary traffic alone, known fraud
   patterns are rare and looked unfamiliar.
3. **Local distance ratio instead of a plain k-NN distance percentile.** A plain percentile treats
   every sparse region as novel, including known fraud such as account takeover.
4. **Isolation forest removed from Familiarity.** It measures rarity, so it flags known rare fraud as
   novel. This deviates from PRD 9.2, which names it as an MVP detector. The ensemble still uses it as a
   detection feature.
5. **Held-out fraud-type pass rule changed** from "more Low/Insufficient than known fraud" to
   "familiarity lower than known fraud and ordinary traffic, more unfamiliar-pattern reasons than known
   fraud, more Low/Insufficient than ordinary traffic". The first version compared with known fraud,
   which is routed to a human anyway for other reasons.

Earlier explanation-reliability fixes (group-level drivers, cosine similarity instead of rank
correlation, background-value deletion test) are documented in `explain.py`.

## Limitations

- Synthetic labels and analysts. No real fraud.
- Selective-label bias (PRD 6.3) is not modelled: every synthetic case gets an outcome.
- Few errors per run, so wide intervals. Segments under 30 cases are reported as inconclusive.
- No new-versus-existing-customer segment (no tenure feature).
- Explanations cover the boosted-tree members only.
- Provisional mode only. Calibrated mode (V1) is not built.
