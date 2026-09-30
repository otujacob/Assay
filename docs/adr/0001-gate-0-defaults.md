# ADR 0001: Gate 0 working defaults

Status: proposed, awaiting product-owner confirmation. PRD v2.2 requires OPD-1, 2, 7, 16, 20, 21 and
22 to be decided before build. These are working defaults so the build can start. Change any of them
by superseding this ADR.

| OPD | Working default | Reason |
|---|---|---|
| OPD-1 latency/throughput | Design target only: p95 < 300 ms for risk + fast trust components, explanation testing async. Parameter, not a promise. | Real budget comes from the pilot's payment flows. |
| OPD-2 verified outcome | Verified fraud = confirmed chargeback or customer-confirmed fraud. Verified legitimate = no dispute after a 120-day window (configurable per product). | Card-fraud convention as a starting point; fraud-ops adviser to confirm. |
| OPD-7 explanation compute | Full reliability tests (M = 20 perturbations) for review-queue cases and a 5% audit sample; others async or `exp` missing. | Matches PRD 7.3. Effect on TI is exactly the tension noted in the review: missing `exp` caps TI_low near 50. |
| OPD-16 isolation tier | Pooled, logical isolation (Postgres RLS, per-tenant keys). Siloed later. | Lowest operational cost for a pre-seed team. |
| OPD-20 development data | Synthetic generator with planted fraud, drift, novelty and label noise (built in E0) plus public datasets for baseline modelling. Analyst behaviour simulated. | No customer data before Gate 1. Simulated results test machinery only. |
| OPD-21 stack | Python 3.12+/FastAPI, PostgreSQL 16 with RLS, XGBoost/scikit-learn/SHAP, MLflow, Prefect, React + TypeScript + Vite, OpenTelemetry, Terraform. Cloud: AWS `eu-west-2` (London), pending pilot hosting needs. | UK data residency default; PRD 24.3 proposals. |
| OPD-22 team | Solo founder plus AI-assisted build. Security review and independent reviewer must be external before Gate 1. | Gate 1 cannot be self-certified. |

## Parameters that are placeholders, not results
Trust weights, band thresholds (70/40), `n_min = 30`, `dq_floor = 0.5`, `novelty_ceiling = 0.95`,
`t_low = 0.30`, `t_high = 0.75` are versioned configuration (`TrustConfig`, `PolicyConfig`) and carry no empirical basis (PRD 5.5, OPD-3).

## Legal note
UK-specific data protection, FCA/PRA model-risk expectations and automated-decision rules need
professional advice before Gate 1 (PRD 17.1). Nothing in this repo asserts compliance.
