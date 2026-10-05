# Assay

Decision intelligence for fraud: a calibrated risk score plus an **AI Trust Index** that says how far
each prediction can be relied on, case by case. Built from the *Assay PRD v2.2*.

Scope: **recommend-only** (automation level 0), no copilot, no cross-tenant learning. Models are
never retrained or promoted automatically: a candidate trained from verified outcomes and accepted feedback goes
through validation gates, shadow, a second person's approval, a canary and a promotion, each taken by a person.
Assay recommends; people decide.

> **Status: pre-pilot.** Everything here has been built and tested on **synthetic data**. Nothing
> shows the Trust Index works on real fraud, and no security, legal or compliance review has been
> done (PRD Gate 1). See [Known gaps](#known-gaps) and [docs/validation](docs/validation/README.md).

## Layout
```
backend/src/assay/
  synthetic/   synthetic data generator with known ground truth (fraud types, drift, label delay)
  features/    versioned feature registry, point-in-time features, leakage checker
  detection/   XGBoost / random forest / isolation forest ensemble, calibration, signed bundles
  trust/       Trust Index components, explanation reliability, TrustAssessor, policy inputs, and the
               calibrated Trust Index (`calibrated.py`, off unless it passed its gate)
  policy/      decision policy engine (evaluation order, trust x risk matrix, recommend-only) and
               versioned tenant policies that need a second person's approval (FR-21)
  ingestion/   validation, idempotency, quarantine, in-memory and PostgreSQL repositories
  scoring/     scoring service: every step stored as append-only lineage, replay, async refinement
  review/      review queue, blind review, analyst actions, conflicts, shadow feedback scores
  graph/       temporal, confidence-weighted entity graph (bitemporal, expiry, analyst flags), point-in-time graph
               features, and the investigator view of what a case is linked to
  learning/    feedback acceptance pool, candidate models, validation gates, shadow / approval / canary /
               promotion / rollback, and the `python -m assay.learning` job that trains a candidate
  validation/  validation harness: baselines, ablations, stress tests, reports
  api/         FastAPI app (signed requests, roles) and server entry point
  lineage/     hash chain used for tamper evidence
  audit/       read-only audit search and CSV/JSON export (FR-34)
  crypto/      per-tenant envelope encryption and key rotation for model bundles and audit exports (FR-41)
  auth/        OpenID Connect bearer-token verification for single sign-on, with MFA (FR-42)
  worker.py    background worker: explanation refinement and the population drift job
backend/scripts/   run_validation.py, demo_server.py
db/migrations/     0001..0008: tables, row-level security, append-only triggers, hash chains
web/               React + TypeScript review app
docs/adr/          architecture decision records (0001: Gate 0 working defaults)
docs/validation/   recorded validation reports and what they found
```

## Quick start (Windows PowerShell)
```
# backend tests
cd backend
python -m venv .venv
.\.venv\Scripts\python -m pip install -e ".[dev]"
.\.venv\Scripts\python -m pytest -q          # database tests skip unless ASSAY_TEST_DATABASE_URL is set

# front-end tests, type check and build
cd ..\web
npm install
npm test; npm run typecheck; npm run build
```

### See it running (demo)
```
cd backend
.\.venv\Scripts\python scripts\demo_server.py      # ~1 minute to train and replay; then http://127.0.0.1:8000
```
It trains a model on synthetic data, replays a stream through the real services and serves the built
UI (run `npm run build` in `web/` first). The top-bar **Demo user** switches between analyst, senior
analyst, manager, auditor and approver. For live editing run `npm run dev` in `web/` (port 5173, proxies
to the demo server). `node web/scripts/e2e.mjs` drives a real browser (installed Edge) through the main
flows against the running demo. Start it with `--with-candidate` to also train a candidate model, which the
script then takes through shadow, approval, canary, promotion and rollback (once per demo run).

**The demo signs requests on the user's behalf through a dev-only proxy. That is not authentication.**
Production uses single sign-on through the institution's identity provider (PRD 16, FR-42).
`python backend/scripts/demo_sso.py` runs the real API with sign-in switched on against a **stand-in** provider
(app on port 8001, provider on 8100), and `node web/scripts/e2e-sso.mjs` drives the sign-in in a real browser.

### PostgreSQL tests
Tests needing a database are skipped unless `ASSAY_TEST_DATABASE_URL` is set. Point it at a
**disposable** database on which you are a superuser (each test makes its own schema and drops it afterwards;
a test that fails to may leave one behind):
```
$env:ASSAY_TEST_DATABASE_URL = "postgresql://postgres@localhost:5432/assay_test"
.\.venv\Scripts\python -m pytest -q
```
Requires PostgreSQL 13+. CI runs a `postgres:16` service and the web build.

### Running the real server
```
$env:ASSAY_DATABASE_URL = "postgresql://assay_login:...@host/assay"   # non-superuser, member of assay_app
$env:ASSAY_DEV_CREDENTIALS = '[{"key_id":"k1","tenant_id":"t1","secret":"...","roles":["ingest","analyst"]}]'
$env:ASSAY_BUNDLES = '[{"tenant_id":"t1","path":"C:/models/t1/b-1234"}]'   # signed bundles; omit = ingest only
$env:ASSAY_BUNDLE_SIGNING_KEY = "..."
# optional, FR-41: open bundles sealed with the tenant key, and refuse any that are not
$env:ASSAY_MASTER_KEY = "...32+ characters..."; $env:ASSAY_REQUIRE_ENCRYPTED_BUNDLES = "1"
uvicorn assay.api.main:app
```
Apply `db/migrations/*.sql` in order first. **The application must connect as an ordinary login role
that is a member of `assay_app`, never as a superuser or table owner**: a superuser can switch role,
disable row-level security and disable triggers, defeating the isolation and append-only guarantees.
The server refuses to start on a superuser, and verifies each bundle's hash, signature and tenant
before loading it. `ASSAY_DEV_CREDENTIALS` is for development only.

## Working rules (PRD 22.2)
- Every requirement has an FR id; reference it in commits.
- Thresholds, weights and policies are versioned configuration, never hard-coded in logic.
- No real customer data until Gate 1 (independent penetration test, isolation tests, professional legal review).
- Any new number in docs is labelled a parameter or an assumption, never a result.

## Build status (PRD epics)
| Epic | State |
|---|---|
| E0 Foundations | repo, CI, ADR 0001, synthetic data generator: done |
| E1 Ingestion and lineage | validation, idempotency, quarantine, maturity, late events, hash chain, tenant isolation (row-level security): done, tested on memory and real PostgreSQL 16 |
| E2 Features and detection | point-in-time features with leakage check, ensemble, calibration, signed tenant-bound bundles: done |
| E3 Explanation | TreeSHAP attributions with stability, sensitivity, faithfulness, reproducibility: done (boosted-tree members only). Counterfactuals ("what would change the decision?", PRD 7.1) and a SHAP-versus-permutation cross-method check are built and shown to investigators on request; they are **not** part of the Trust Index. On synthetic seed 99, 82 of 100 flagged cases had a counterfactual that passed every check, at about 0.4 s per case ([docs/validation/README.md](docs/validation/README.md)). They describe the model, not cause and effect, and must not be given to customers |
| E4 Trust Index | all six components plus the assessor, Insufficient-evidence state, async refinement: done. `hum` inactive in the MVP by design. Calibrated mode (PRD 5.6) is built and gated in code, but **not enabled**: the logistic meta-model was well calibrated but not shown to rank errors as well as the provisional score, and a recalibration-only variant (provisional order kept, mapped to a probability) passed its pre-set test on twelve more seeds (ECE 0.005, ranking identical) but still passes the PRD 6.6 gate on only 2 of 12, because the index ranks errors slightly worse than the simplest baseline on the stable part ([docs/validation/README.md](docs/validation/README.md)) |
| E5 Policy and review | policy engine, versioned policies with second-person approval (FR-21) that can set risk thresholds and an "always review amounts from" rule, policy replay (preview a proposed policy's effect on review volume against the policy in force; it does not predict fraud caught or analyst decisions). A policy that sets its own risk thresholds replaces the model's, so the validation figures, measured at the model's thresholds, no longer describe it, queue with priority and SLA, blind review, reason codes, conflicts and adjudication: done |
| E6 Feedback capture | decisions, blind flag, checklist, shadow scores AAS/FCS/LVS/CRS/FQS: done |
| E11 Entity graph (V1) | temporal graph of devices, IP addresses, beneficiaries and merchants: bitemporal, confidence with decay and expiry, hub handling, analyst flags (append-only, database-enforced), point-in-time graph features, and a "Linked entities" view for investigators: done. **Investigator context only, not a model input**: the pre-set test found graph features help where simulated rings exist but cost a little on other fraud, so by the PRD's own rule it does not feed scoring ([docs/validation/README.md](docs/validation/README.md), [docs/governance/entity-graph.md](docs/governance/entity-graph.md)). Communities are connected groups, not Louvain; no pilot-volume latency test (OPD-14); no identity attributes; ring and mule detection as products (V2) not built |
| E10 Continuous learning (V1) | feedback acceptance pool (label levels 1 to 4, integrity checks, per-analyst cap), candidate training that lets accepted analyst labels into the training window only, validation gates G1 to G9 against the champion on a verified-outcome-only holdout, and a governed lifecycle (shadow, second-person approval, canary, promotion, rollback) stored as append-only records and enforced by the database as well as the service: done. **Tested only with simulated analysts**, and the pre-set H4 test was **not supported as specified** ([docs/validation/README.md](docs/validation/README.md)). Accept thresholds, gate tolerances and the approval authority are untuned working defaults (OPD-11, OPD-12). Gates G4 and G5 can report "could not judge" and then need an explicit waiver. A monitor rolls a degrading live model back to the previous champion on clear evidence (calibration, High-trust error, recall, precision). Not built: automatic triggers or schedules for candidates (OPD-13), bounded auto-recalibration, a kill switch that falls back to human review for all cases |
| E7 Validation harness and dashboard | baselines B1 to B4, ablations, four accuracy measures, stress tests, stored reports, dashboard summary: done |
| E8 Lineage completion | every decision step is an append-only record; replay reproduces decisions; audit-log search and CSV/JSON export (FR-34) with an auditor UI panel: done (tested on the in-memory store; the PostgreSQL path is unexercised) |
| E9 Security and tenancy | tenant isolation, signed requests, roles, non-superuser enforcement, rate limiting, access events in the audit log (FR-43), CI secret and dependency scans (FR-44): done. Per-tenant encryption (FR-41): model bundles and audit exports, with a local key provider ([docs/security/key-management.md](docs/security/key-management.md)). Single sign-on and MFA (FR-42): token verification and the web app's browser sign-in (authorization code with PKCE) are built and tested end to end against a **stand-in** provider, **not against a real identity provider** ([docs/security/sso.md](docs/security/sso.md)). A managed key service, SAML, the independent penetration test: **not done** |
| Web app | review queue, case view, Trust Index gauge, component bars, drivers, actions, governance tabs, policy proposal and approval, auditor log, on-demand what-ifs, linked entities, Model Updates panel: done. 168 unit tests; the browser end-to-end check (`web/scripts/e2e.mjs`) passes against the demo server |

## What the validation found
On synthetic data the Trust Index beats the two simplest uncertainty baselines (B1, B2) clearly, and
cases rated High trust are wrong several times less often than average (3 to 5 times on the first two
seeds; about ten times on average over the twelve, 0.28% against 2.99%). Against the third baseline
(maximum class probability, B3) the first two seeds were inconclusive: ahead, but not by a margin that
excludes "no improvement". A pre-set analysis over twelve further seeds (about 1,400 errors) found a
**small advantage over B3 (+0.03 AUROC, 95% interval +0.009 to +0.051), ahead in 9 of 12 seeds**, so
by the PRD's criterion it holds on this data, weakly. Explanation reliability showed no measurable value
on any run, and drift and data quality have none on clean data by construction. None of this is evidence
about real fraud. Details, five method changes made along the way, and the limits:
[docs/validation/README.md](docs/validation/README.md).

## Known gaps
- **No real data, no pilot.** All evidence is synthetic. Real validation needs a pilot institution.
- **Security and compliance.** No real identity-provider test, managed key service, penetration test, or legal review
  (UK data protection, FCA/PRA expectations). Gate 1 cannot be passed yet.
- **CI has not run on GitHub** (written, unverified there).
- **Explanation cost.** About 27 ms per case, so real-time explanation testing of every case is
  expensive. Cases scored without it cannot reach High trust until the worker runs
  (`python -m assay.worker`, which loops on a timer; `--once` for a single pass). Nothing starts it
  for you: run it as its own process or scheduled task, or High trust is unreachable for those cases.
- **Population drift job** runs in that worker. It skips a window of fewer than 100 rows or under
  24 hours (a short window false-alarms on time-of-day features), so a quiet tenant has no drift
  signal and scoring treats drift as zero. Window and alarm level are parameters, not tuned values.
- **API rate limit** is per process, not shared across workers.
- **Institution hard rules** (sanctions holds) have a hook but no rule source.
- **Analyst reason codes** are working defaults, not agreed with a fraud-operations adviser.
- **Feedback learning is tested only on simulated analysts.** The pipeline caught about 97% of injected wrong labels, but the
  accepted-feedback model was not shown to be as good as learning from everything when analysts are clean (H4 not
  supported as specified; [docs/validation/README.md](docs/validation/README.md)). Thresholds are not tuned (OPD-11).
- **Not built (later phases):** automation above level 0,
  AI copilot, regulator export views.
- **Calibrated mode exists but should stay off.** It produces genuine probabilities (the recalibration-only variant,
  `--method recalibrated`, passed its pre-set test and cannot rank worse than the provisional index), but the PRD gate passes
  in only 2 of 12 synthetic seeds, because on the stable part the index does not reliably beat the model's own maximum class
  probability. A pilot's matured outcomes are what could change that. `python backend/scripts/build_calibrated_bundle.py` shows the
  workflow; a server refuses calibrated mode for any bundle whose gate failed.
- **Open product decisions** still on working defaults: see [docs/adr/0001-gate-0-defaults.md](docs/adr/0001-gate-0-defaults.md).
