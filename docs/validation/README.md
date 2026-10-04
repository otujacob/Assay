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
| Twelve fresh seeds, 201 to 212 (a pre-set analysis, see below) | 46,327 | 1,384 (2.99%) | `multi-seed-201-212/` |

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

## Twelve-seed check: is the gap to B3 real? (added after the two-seed result)

The two-seed result left one question open: the Trust Index was ahead of B3 (max class probability) on
both seeds, but 47 to 57 errors per run is too few to tell a real advantage from noise. So the same
harness was run on **twelve further seeds (201 to 212)**, none used for any design decision. Each
seed is a separate synthetic dataset; the unit of analysis is the seed.

**The decision rule was written before the runs** (it is in the docstring of
`backend/scripts/validate_many_seeds.py`): the mean over seeds of AUROC(Trust Index) minus AUROC(B3),
supported only if its 95% interval excludes zero on the positive side. Reproduce with:

```
python backend/scripts/validate_many_seeds.py run --seeds 201-212 --out multi
python backend/scripts/validate_many_seeds.py summarise multi
```

| Trust Index minus | Mean AUROC difference | 95% interval over seeds | Ahead in |
|---|---|---|---|
| B1, distance from threshold | +0.124 | +0.090 to +0.157 | 12 of 12 |
| B2, ensemble disagreement | +0.080 | +0.050 to +0.109 | 12 of 12 |
| **B3, max class probability** | **+0.030** | **+0.009 to +0.051** | **9 of 12** |

**By the pre-set rule, the Trust Index beats B3 on this synthetic data. That is a weak result, and it
should be read as one.**
- The advantage is small (+0.03 on an AUROC of 0.91), and its lower bound is only +0.009.
- It is ahead in 9 of 12 seeds and behind in 3 (by 0.013, 0.009 and 0.001). A sign test across seeds
  does not reach significance (p = 0.15). The interval over seeds does exclude zero, but that interval
  assumes the per-seed differences are roughly normal, and there are only twelve of them.
- Every seed shares one generator and one configuration, so these are not different worlds. They show
  stability over sampling, not over kinds of fraud.
- The Trust Index design was tuned on seeds 5 and 99, not on these. That makes this a fair test of the
  design on new data. It is still synthetic data (PRD 28.3).

What this changes: the earlier "inconclusive" reading was partly a power problem. With about 1,400
errors instead of about 50, the Trust Index is ahead of B3 on average. What it does not change: nothing
here says the Trust Index works on real fraud, and the PRD 6.6 criteria still have to be met on a pilot
institution's matured outcomes.

Component ablations over the twelve seeds (AUROC without the component minus with it; the rule is
the same: entirely below zero means it helps, entirely above zero means it hurts):

| Component | Mean change if removed | 95% interval | Reading |
|---|---|---|---|
| rel (cohort reliability) | -0.035 | -0.045 to -0.025 | **helps, a large effect.** Removing it hurt in 12 of 12 seeds |
| conf (model confidence) | -0.009 | -0.014 to -0.005 | helps, small effect, 12 of 12 seeds |
| exp (explanation reliability) | +0.002 | -0.001 to +0.005 | **no measurable value.** H3 stays unsupported, as on the first two seeds |
| fam (familiarity) | +0.009 | +0.000 to +0.017 | removing it RAISES discrimination in 10 of 12 seeds, but the lower bound is +0.0004. Down-weighting is indicated, weakly (PRD 6.4, OPD-3) |
| drift, dq | 0 | 0 | no effect in any seed: the data is clean and stationary, by construction |

Two things follow. The score's discriminating power comes mostly from `rel`, so it is only as good as
the matured outcomes behind it, which is exactly what a pilot would be short of at first. And `exp`,
which is the most expensive component to compute, has not shown value on any synthetic run so far.
That is a product decision for OPD-3 and OPD-7, and should be taken on data separate from this.

## Calibrated Trust Index: is it worth enabling? (twelve seeds, 301 to 312)

Calibrated mode (PRD 5.6) replaces the fixed-weight score with a meta-model's estimate of the chance
the recommendation is correct, so a score of 80 means "right about 80% of the time". It is built
(`backend/src/assay/trust/calibrated.py`) and was tested with the same discipline as above: a rule
written before the runs (docstring of `backend/scripts/validate_calibrated_many_seeds.py`), fresh seeds,
one analysis. Each seed's test window is split in time: fitted on the oldest 45%, calibrated on the
next 20%, evaluated on the newest 35%. Reproduce with that script (`run`, then `summarise`).

**One design change, made before the seeds were run.** The first design used the standard dataset. A
smoke run on seed 300, which is not in the set, showed that only 10 of its 145 errors fell in the oldest
45% of the window and 1 in the calibration slice, because almost all errors come after the novel fraud
type appears. The model correctly refused to fit, and the experiment could not have answered anything.
So the experiment uses a longer dataset (a 135-day test window, novel fraud from day 270). Nothing about
seeds 301 to 312 informed that choice.

| Question | Result | Rule | Verdict |
|---|---|---|---|
| P1 Calibration, stable part | ECE **0.005** against 0.268 for the provisional score read as a probability; improvement +0.264 (interval +0.256 to +0.271) | mean ECE at most 0.03 and better than provisional | **supported** |
| P2 Ranking errors, stable part | calibrated minus provisional **-0.010** AUROC (interval -0.037 to +0.016); ahead in 5 of 12 | lower bound above -0.02 | **not met** |
| P3 Ranking errors after novel fraud appears | **-0.003** (interval -0.013 to +0.006); ahead in 4 of 12 | lower bound above -0.02 | non-inferior |

**By the rule written in advance, the verdict is: stay in Provisional mode.** The rule is not
relaxed because calibration came out so well.

How to read it:
- **The calibration is real.** The scores are probabilities: 0.005 average error, held after the novel
  fraud type appeared (0.004), and the High band's observed error was 0.33% against the 1% it was set
  from. The provisional score read as a probability is off by 0.27, so the mode does what it is for.
- **The ranking test is unresolved, not lost.** P2 fails because its interval reaches below -0.02, but
  the same interval includes zero and +0.016. The stable part of each evaluation window is small (a
  median of about 45 wrong recommendations, and as few as 11 and 17 in two seeds), and those two seeds
  swing the result (-0.098 and +0.079). The honest summary is "not shown to be as good", not "shown to
  be worse".
- **Against B3** it is level on the stable part (+0.002, interval -0.049 to +0.053) and clearly ahead
  after novelty (+0.024, interval +0.010 to +0.039, ahead in 11 of 12). The rule did not use this.
- **The PRD 6.6 gate** passes on its own in only **2 of 12** seeds, mostly because beating B3 with an
  interval that excludes zero is hard on a single evaluation window. The gate is applied in code: a
  server asked to run a tenant in calibrated mode refuses to start unless the bundle's gate passed.
- A guess, not a finding: the meta-model learns its weights from only tens of wrong recommendations, so
  its estimation noise may cost a little ranking. More matured outcomes should shrink it.

What would change the verdict is more matured outcomes (a pilot), not a looser rule. One untested
alternative is a recalibration-only variant, a monotone mapping of the provisional score. It ranks
identically by construction, so it could not lose on P2, and it would give the calibration. It would
need its own pre-set test on fresh seeds. Synthetic data only, as always (PRD 28.3).

## Component ablations (PRD 6.4), the first two seeds

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
