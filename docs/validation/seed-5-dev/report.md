_Synthetic dataset seed 5; 3891 test cases; run time 110s._

# Trust Index validation report: bundle b-0ee7cc94

Mode: **provisional**. Cases: 3891 (57 wrong recommendations, 1.46%). Dataset 61450e08d746. Generated 2026-09-30T21:27:52Z.

**Verdict against the PRD 6.6 criteria: NOT SUPPORTED as specified.** revise the components or the method; do not relax the criteria (PRD 6.6).

## Headline measures (matured verified outcomes, later than all training data)
- Error discrimination, AUROC of (100 - TI): 0.874 (95% CI 0.814 to 0.922)
- High-trust error rate: 0.28% (95% CI 0.13% to 0.61%; 6/2125) against overall error rate 1.46% (95% CI 1.13% to 1.89%; 57/3891)
- Low-trust detection rate (share of errors sent to a human): 82.46% (95% CI 70.63% to 90.18%; 47/57)
- False-confidence rate (share of errors that had High trust): 10.53% (95% CI 4.91% to 21.12%; 6/57)
- Coverage (cases that received a score): 93.4%
- Trust states: {'moderate': 1110, 'low': 400, 'insufficient_evidence': 256, 'high': 2125}

## Against simple baselines (PRD 6.4)
| Baseline | AUROC | Trust Index minus baseline | Beats it? |
|---|---|---|---|
| B1_distance_from_threshold | 0.741 (95% CI 0.645 to 0.818) | +0.133 (CI +0.051 to +0.227) | yes |
| B2_ensemble_disagreement | 0.790 (95% CI 0.720 to 0.859) | +0.084 (CI +0.014 to +0.144) | yes |
| B3_max_class_probability | 0.854 (95% CI 0.803 to 0.906) | +0.020 (CI -0.010 to +0.050) | no (interval includes no improvement) |

Each component alone (B4), AUROC: conf 0.826, rel 0.773, exp 0.494, fam 0.843, drift 0.500, dq 0.500

## Ablation: remove each component in turn
| Component | AUROC without it | Change | Verdict |
|---|---|---|---|
| conf | 0.860 | -0.014 | adds value (removing it hurts discrimination) |
| rel | 0.855 | -0.018 | adds value (removing it hurts discrimination) |
| exp | 0.878 | +0.005 | no measurable effect: drop or down-weight (PRD 6.4) |
| fam | 0.874 | -0.000 | no measurable effect: drop or down-weight (PRD 6.4) |
| drift | 0.874 | +0.000 | no measurable effect: drop or down-weight (PRD 6.4) |
| dq | 0.874 | +0.000 | no measurable effect: drop or down-weight (PRD 6.4) |

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
