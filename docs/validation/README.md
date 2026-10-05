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

What would change the verdict is more matured outcomes (a pilot), not a looser rule. Synthetic data only, as always (PRD 28.3).

### The recalibration-only variant (twelve fresh seeds, 501 to 512)

The untested alternative named above has now been tested. It keeps the provisional index's order and learns only a
monotone map from it to P(correct) (`backend/src/assay/trust/recalibrated.py`). The rule (same P1, P2, P3 as above) was
written before the runs in the docstring of `validate_calibrated_many_seeds.py`, and the seeds are fresh. The map is fitted
on the oldest 65% of each seed's window and everything is evaluated on the newest 35%, as before. Evidence:
[recalibrated-501-512/](recalibrated-501-512/).

| Question | Result | Rule | Verdict |
|---|---|---|---|
| P1 Calibration, stable part | ECE **0.0046** against 0.275 for the provisional score read as a probability; improvement +0.271 (interval +0.265 to +0.276) | mean ECE at most 0.03 and better than provisional | **supported** |
| P2 Ranking errors, stable part | recalibrated minus provisional **+0.0001** AUROC (interval -0.0001 to +0.0002) | lower bound above -0.02 | non-inferior (identical, as designed) |
| P3 Ranking errors after novel fraud appears | **-0.0000** (interval -0.0001 to +0.0000) | lower bound above -0.02 | non-inferior (identical) |

**By the rule written in advance, the verdict is: worth enabling for a pilot.** The calibration holds after the novel fraud type
appears too (ECE 0.0036), the High band's observed error was 0.37% against the 1% it was set from, and the interval is narrow
(0.49 points on the 0 to 100 scale on average).

What that verdict does **not** say:
- **It does not pass the system's own gate on most seeds.** The PRD 6.6 gate, which the server enforces, passes in only
  **2 of 12** seeds. The reason is the ranking comparison with B3 (the model's own maximum class probability), which the rule
  above did not use: on the stable part the provisional index ranks wrong recommendations **slightly worse** than B3 on these
  seeds (-0.019 AUROC, interval -0.027 to -0.011; ahead in 1 of 12), and **better** after the novel fraud type appears (+0.021, interval
  +0.005 to +0.038, ahead in 9 of 12). On the earlier twelve seeds (301 to 312) the meta-model was level with B3 on the stable part (+0.002). Both sets of seeds
  are synthetic and the stable part holds few errors, so the difference between them may be noise, and it is not tested here; but it is a reason not to
  claim the Trust Index beats the simplest baseline everywhere.
- **Calibrated mode stays off by default.** A server refuses to run a tenant in calibrated mode unless that tenant's bundle passed
  the gate on that tenant's own matured outcomes. This variant removes the calibration risk (the ranking is the provisional index's,
  so it cannot be worse than it) and leaves the ranking question for a pilot to answer.
- Build a bundle with it: `python scripts/build_calibrated_bundle.py --seed N --out DIR --method recalibrated`.

## Counterfactuals: how the engine behaves (a description, not a hypothesis test)

Counterfactuals (`backend/src/assay/trust/counterfactual.py`) answer "what small, realistic change
would have flipped the model's call?" for an investigator. They are **not** in the Trust Index: the
explanation component (`exp`) showed no measurable value on any validation run above, and adding more
to it without evidence would add cost, not trust. They are shown on the case screen on request.

What was measured, on synthetic data (seed 99, the standard dataset), and only to describe behaviour:

| Cases | Had any candidate | Had a counterfactual that passed every check and held up | Usually changed | Time per case |
|---|---|---|---|---|
| 100 random flagged cases | 100 | **82** | one thing in 75 of 82 (amount 59, beneficiary 22, time 7, channel 1) | 0.41 s mean, 0.64 s p90 |
| the 60 highest-risk unflagged cases | 60 | **57** | one thing in 56 of 57 (amount 41, beneficiary 17) | 0.39 s mean, 0.73 s p90 |

"Passed every check" means the independent validator found that re-scoring flips the call by a margin,
every value is inside the data's bounds, the customer's history and the data's completeness are
untouched, the amount group is internally consistent, at most two groups changed, the move is small,
and the flip survives small perturbations. In the other 18 flagged cases the call is firm: no realistic
change flips it, and the screen says so.

Limits to keep in mind:
- A counterfactual describes the **model**, not cause and effect (PRD 7.4). That amount is what the
  model leans on most says what it learned from synthetic data, not what fraud depends on.
- Pairs of changes are shortlisted by the boosted trees alone, so a flip the shortlist ranks low can be
  missed. A counterfactual that is reported was always scored by the full model.
- The full model is single-threaded, so replays are exact, which makes this about 0.4 s per case. That
  is fine for an investigator's click and far too slow to compute for every case.
- Cross-method consistency (SHAP against a permutation importance on the whole ensemble) is shown per
  case but has not been summarised across cases here. Some disagreement is expected by design, because
  the permutation view includes the random forest and isolation forest that SHAP here does not.

## Entity graph: does it add detection value over a join on the same data? (twelve seeds, 601 to 612)

The question (PRD H6): do temporal, confidence-weighted relationships add detection value over the same entity data used as
flat features? Script: `backend/scripts/validate_graph_many_seeds.py`; per-seed results and the summary are in
[graph-601-612/](graph-601-612/). The decision rule was written in the script before the seeds were run.

**These rings are simulated.** The standard dataset gets an optional scenario (`GeneratorConfig(graph_scenario=True)`, off by
default, and the default datasets are byte-for-byte unchanged, which a test pins): five rings of 5 to 8 customers who commit fraud
together through shared devices, IP addresses and beneficiaries, each transaction looking ordinary on its own, in four campaigns
spread over the timeline; 60 families who share a device legitimately; and three public IP addresses used by hundreds of
unrelated customers. The rings are the experiment's own construction. It tests that the graph can find structure that is there and
how it compares with a join; it is not evidence that real fraud has this shape (PRD 13.6).

Three arms, trained identically and scored on the same verified-outcome-only test window: **F0** the registry's flat features;
**F1** F0 plus the same entity data as flat features (30-day counts of distinct other customers per device, IP and beneficiary, and
how many of them have a confirmed fraud known at the time); **G** F0 plus graph features (confidence-weighted, decayed,
specificity-discounted, two hops, connected groups). Metric: PR-AUC on calibrated probabilities, "ring cases" being legitimate rows
plus only the ring frauds and "other fraud" legitimate rows plus only the other fraud types. About 186 frauds per test window, 60 of
them ring frauds.

| PR-AUC | F0 | F1 (flat entity data) | G (graph) | G minus F1 (95% interval) | ahead |
|---|---|---|---|---|---|
| All cases | 0.726 | 0.781 | **0.798** | **+0.018 (+0.006, +0.030)** | 10 of 12 |
| Ring cases | 0.689 | 0.893 | **0.963** | **+0.071 (+0.047, +0.094)** | 12 of 12 |
| Other fraud | 0.688 | 0.683 | 0.675 | -0.008 (-0.017, +0.001) | 4 of 12 |

**Verdict under the pre-set rule: NOT supported as specified.** R1 (better overall) and R2 (better on ring cases) held. **R3
failed**: on the other fraud types the graph is slightly worse than the flat baseline, and the lower end of the interval (-0.017)
is below the -0.01 limit. So, as the PRD says for this case (H6), **the graph stays investigator context only** and is not a model
input. It is built that way.

What this says, in plain terms:
- **Where the structure exists, the graph finds it, and finds it better than a join.** A plain join already helps a lot (ring
  cases 0.689 to 0.893), which is the main thing the entity data buys. The graph adds a further 0.07 on ring cases. Its features combine confidence weighting, recency, spreading known
  fraud over two hops, community size and density, and discounting busy nodes; this experiment did not separate which of them
  produces the gain.
- **Extra columns cost a little elsewhere.** Adding either set of features lowered PR-AUC on the non-ring fraud (flat -0.006, graph
  -0.014 against F0): more columns, and nothing in the other fraud types for them to find. A model that takes graph features would
  need them only where they help, which this experiment did not test.
- **False links were rare, and this is the easy case.** Of the derived links held at the end of the test window, 2.2% on average joined
  customers outside any planted ring or family (all of them through beneficiaries; no device link was false). The three public IP
  addresses never linked anyone, because the graph does not expand a node with more than 60 customers. A real tenant's shared
  infrastructure is messier than three clean hubs.
- **Cost.** Building the graph and flat features for a seed's roughly 27,000 transactions took about 44 seconds on one core
  (about 1.6 ms per transaction).
  No pilot-volume latency test has been run (OPD-14).

Not checked: whether graph features help detection of other patterns the PRD names (mule accounts, synthetic identities, which
need attributes this dataset does not have); stability of the communities (they are connected groups above a confidence
threshold, not Louvain or Leiden, and can merge two rings joined by one strong link); the PRD's false-relationship rate from
analyst flags (no real flags exist).

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
test needs the learning side of the feedback engine, which now exists: see "Feedback acceptance" below.

## Feedback acceptance: does scoring feedback before learning help? (twelve seeds, 401 to 412)

The question (PRD H4): is a model trained on feedback that passed a quality check better than one trained on
every analyst decision, judged on verified outcomes that analysts never touched? Script:
`backend/scripts/validate_feedback_many_seeds.py`; per-seed results and the summary are in
[feedback-401-412/](feedback-401-412/). The decision rule was written in the script before the seeds were run.

**These analysts are simulated** (`assay/validation/analysts.py`): four good (92% right), one weak (62%), one
adversarial (25%) and one who approves everything in seconds, plus a senior who rules on half of the conflicts.
The simulation tests that the pipeline does what it is specified to do when fed labels of known quality. It is
not evidence about real analysts (PRD 19, OPD-20), and the results below depend on the assumptions in that file.
Stated confidence and checklists were made uninformative about correctness, so the engine had to rely on track
record, corroboration and integrity checks.

Setup: 92% of the training-window verified outcomes were hidden, as if never confirmed, and the analysts decided
those cases instead (about 10,200 cases per seed). Calibration and test windows were untouched.

| Share of cases handled by weak, adversarial or lazy analysts | 0% | 30% | 60% |
|---|---|---|---|
| Wrong labels among ALL analyst decisions | 7.2% | 14.9% | 22.2% |
| Wrong labels among ACCEPTED labels | 1.0% | 1.7% | 2.5% |
| Share of cases accepted | 18% | 18% | 17% |
| PR-AUC, verified outcomes only (V) | 0.579 | 0.579 | 0.579 |
| PR-AUC, trained on all feedback (ALL) | 0.691 | 0.675 | 0.634 |
| PR-AUC, trained on accepted feedback (ACC) | 0.673 | 0.676 | 0.665 |
| PR-AUC, with the hidden labels true (ceiling) | 0.705 | 0.705 | 0.705 |
| ACC minus ALL (95% interval) | -0.018 (-0.038, +0.001) | +0.001 (-0.013, +0.015) | +0.031 (+0.010, +0.052) |
| ACC minus V (95% interval) | +0.094 (+0.055, +0.133) | +0.097 (+0.057, +0.136) | +0.085 (+0.059, +0.112) |

**Verdict under the pre-set rule: NOT supported as specified.** R2 (more robust than learning from everything
when noise is heavy), R3 (does not hurt compared with ignoring feedback) and R4 (accepted labels have at most half
the error of all labels) held. **R1 failed**: when analysts are clean, the accepted-feedback model is not shown to be at
least as good as the all-feedback model (ACC minus ALL -0.018, interval -0.038 to +0.001, lower end below the
-0.01 limit; ahead in only 4 of 12 seeds).

What this says, in plain terms:
- The check does what it is for. It removed 97% of the wrong labels at every noise level (the lazy and adversarial
  analysts were flagged in all 12 seeds), and the model built on what it kept beats the all-feedback model once
  noise is heavy.
- It is conservative. With the default thresholds it accepts about 18% of cases. Of the accepted unverified labels, 88% were
  ones a second analyst agreed on, 10% were adjudicated and 2% were single decisions (a lone decision cannot reach the accept
  threshold unless the analyst has a proven record and high confidence). When feedback is clean, throwing away
  four labels in five costs more than the noise it avoids. At 30% noise the two are level.
- Any feedback beats none: both feedback models beat the verified-only model by 0.05 to 0.11 PR-AUC, because
  verified labels were scarce by design. Where verified labels are plentiful, this gain would be much smaller
  (a smoke run with half the labels hidden showed none).
- Accept thresholds and the share needing corroboration are open product decisions (OPD-11). They were not tuned
  here, because tuning them on these seeds would invalidate the test. A pilot is where they get set.

Not checked: concentration of labels on particular merchants or beneficiaries (needs the graph store); sudden
shifts in an analyst's accuracy; whether real analysts' confidence says anything about being right.

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
