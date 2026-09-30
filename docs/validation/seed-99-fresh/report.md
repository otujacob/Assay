_Synthetic dataset seed 99; 3911 test cases; run time 108s._

# Trust Index validation report: bundle b-bb5f5920

Mode: **provisional**. Cases: 3911 (47 wrong recommendations, 1.20%). Dataset 39d55beb82af. Generated 2026-09-30T21:25:50Z.

**Verdict against the PRD 6.6 criteria: NOT SUPPORTED as specified.** revise the components or the method; do not relax the criteria (PRD 6.6).

## Headline measures (matured verified outcomes, later than all training data)
- Error discrimination, AUROC of (100 - TI): 0.808 (95% CI 0.736 to 0.877)
- High-trust error rate: 0.38% (95% CI 0.19% to 0.74%; 8/2124) against overall error rate 1.20% (95% CI 0.90% to 1.59%; 47/3911)
- Low-trust detection rate (share of errors sent to a human): 70.21% (95% CI 56.02% to 81.35%; 33/47)
- False-confidence rate (share of errors that had High trust): 17.02% (95% CI 8.89% to 30.14%; 8/47)
- Coverage (cases that received a score): 94.4%
- Trust states: {'moderate': 1203, 'high': 2124, 'low': 364, 'insufficient_evidence': 220}

## Against simple baselines (PRD 6.4)
| Baseline | AUROC | Trust Index minus baseline | Beats it? |
|---|---|---|---|
| B1_distance_from_threshold | 0.671 (95% CI 0.580 to 0.758) | +0.137 (CI +0.042 to +0.226) | yes |
| B2_ensemble_disagreement | 0.733 (95% CI 0.656 to 0.813) | +0.075 (CI +0.013 to +0.139) | yes |
| B3_max_class_probability | 0.767 (95% CI 0.703 to 0.828) | +0.042 (CI -0.025 to +0.099) | no (interval includes no improvement) |

Each component alone (B4), AUROC: conf 0.738, rel 0.784, exp 0.493, fam 0.680, drift 0.500, dq 0.500

## Ablation: remove each component in turn
| Component | AUROC without it | Change | Verdict |
|---|---|---|---|
| conf | 0.759 | -0.049 | adds value (removing it hurts discrimination) |
| rel | 0.772 | -0.037 | adds value (removing it hurts discrimination) |
| exp | 0.801 | -0.007 | adds value (removing it hurts discrimination) |
| fam | 0.841 | +0.033 | removing it IMPROVES discrimination: drop or down-weight (PRD 6.4) |
| drift | 0.808 | +0.000 | no measurable effect: drop or down-weight (PRD 6.4) |
| dq | 0.808 | +0.000 | no measurable effect: drop or down-weight (PRD 6.4) |

## Stress tests (PRD 6.5)
- **synthetic feature drift**: passed. Expected: drift stability and TI fall, alarm fires (and does not fire without drift).
- **degraded data completeness**: passed. Expected: data quality falls monotonically and the floor gate triggers.
- **held-out fraud type**: passed. Expected: familiarity falls below known fraud and ordinary traffic, the unfamiliar-pattern reason fires more often than for known fraud, and held-out cases land in Low or Insufficient evidence more often than ordinary traffic.
- **noisy or adversarial analyst labels**: not testable. Expected: feedback quality scores fall and labels are rejected (PRD 11).
- **novel fraud type (H8)**: reported, no pass rule. Expected: never-seen fraud lands in Low or Insufficient evidence.

## Limitations
- Synthetic data with simulated labels and analysts: tests the machinery, not real fraud.
- Selective-label bias (PRD 6.3) is not modelled: every synthetic case receives an outcome.
- New-versus-existing-customer segment is not reported (no tenure feature yet).
- Explanations cover the boosted-tree members only.
