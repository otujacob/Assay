# Assay

Decision intelligence for fraud: a calibrated risk score plus an **AI Trust Index** that says how far
each prediction can be relied on, case by case. Built from the *Assay PRD v2.2*.

MVP scope (PRD 19): **recommend-only**, no retraining from feedback, no graph store, no copilot, no
cross-tenant learning. Assay recommends; people decide.

> **Status: pre-pilot.** Everything here has been built and tested on **synthetic data**. Nothing
> shows the Trust Index works on real fraud, and no security, legal or compliance review has been
> done (PRD Gate 1). See [Known gaps](#known-gaps) and [docs/validation](docs/validation/README.md).

## Layout
```
backend/src/assay/
  synthetic/   synthetic data generator with known ground truth (fraud types, drift, label delay)
  features/    versioned feature registry, point-in-time features, leakage checker
  detection/   XGBoost / random forest / isolation forest ensemble, calibration, signed bundles
  trust/       Trust Index components, explanation reliability, TrustAssessor, policy inputs
  policy/      decision policy engine (evaluation order, trust x risk matrix, recommend-only) and
               versioned tenant policies that need a second person's approval (FR-21)
  ingestion/   validation, idempotency, quarantine, in-memory and PostgreSQL repositories
  scoring/     scoring service: every step stored as append-only lineage, replay, async refinement
  review/      review queue, blind review, analyst actions, conflicts, shadow feedback scores
  validation/  validation harness: baselines, ablations, stress tests, reports
  api/         FastAPI app (signed requests, roles) and server entry point
  lineage/     hash chain used for tamper evidence
  audit/       read-only audit search and CSV/JSON export (FR-34)
  crypto/      per-tenant envelope encryption and key rotation for model bundles and audit exports (FR-41)
  auth/        OpenID Connect bearer-token verification for single sign-on, with MFA (FR-42)
  worker.py    background worker: explanation refinement and the population drift job
backend/scripts/   run_validation.py, demo_server.py
db/migrations/     0001..0006: tables, row-level security, append-only triggers, hash chains
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
flows against the running demo.

**The demo signs requests on the user's behalf through a dev-only proxy. That is not authentication.**
Production needs single sign-on through the institution's identity provider (PRD 16, FR-42).

### PostgreSQL tests
Tests needing a database are skipped unless `ASSAY_TEST_DATABASE_URL` is set. Point it at a
**disposable** database on which you are a superuser (each test makes its own schema, and append-only
tables cannot be cleaned up):
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
| E3 Explanation | TreeSHAP attributions with stability, sensitivity, faithfulness, reproducibility: done (boosted-tree members only) |
| E4 Trust Index | all six components plus the assessor, Insufficient-evidence state, async refinement: done. `hum` inactive in the MVP by design |
| E5 Policy and review | policy engine, versioned policies with second-person approval (FR-21), queue with priority and SLA, blind review, reason codes, conflicts and adjudication: done |
| E6 Feedback capture | decisions, blind flag, checklist, shadow scores AAS/FCS/LVS/CRS/FQS: done. Nothing trains on them |
| E7 Validation harness and dashboard | baselines B1 to B4, ablations, four accuracy measures, stress tests, stored reports, dashboard summary: done |
| E8 Lineage completion | every decision step is an append-only record; replay reproduces decisions; audit-log search and CSV/JSON export (FR-34) with an auditor UI panel: done (tested on the in-memory store; the PostgreSQL path is unexercised) |
| E9 Security and tenancy | tenant isolation, signed requests, roles, non-superuser enforcement, rate limiting, access events in the audit log (FR-43), CI secret and dependency scans (FR-44): done. Per-tenant encryption (FR-41): model bundles and audit exports, with a local key provider ([docs/security/key-management.md](docs/security/key-management.md)). Single sign-on and MFA (FR-42): token verification built and tested with generated tokens, **not tested against a real identity provider**, and the web app has no sign-in flow yet ([docs/security/sso.md](docs/security/sso.md)). A managed key service, SAML, the independent penetration test: **not done** |
| Web app | review queue, case view, Trust Index gauge, component bars, drivers, actions, governance tabs, policy proposal and approval, auditor log: done. 73 unit tests; the browser end-to-end check (`web/scripts/e2e.mjs`) has not been rerun since the policy and audit panels were added |

## What the validation found
On synthetic data the Trust Index beats the two simplest uncertainty baselines, and cases rated High
trust are wrong 3 to 5 times less often than average. It does **not** beat the third baseline (maximum
class probability) by a margin that excludes "no improvement", so by the PRD's own criteria it is
**not supported as specified**. Explanation reliability, drift and data quality showed no measurable
value on this data. Details, five method changes made along the way, and the limits:
[docs/validation/README.md](docs/validation/README.md).

## Known gaps
- **No real data, no pilot.** All evidence is synthetic. Real validation needs a pilot institution.
- **Security and compliance.** No browser sign-in or real identity-provider test, managed key service, penetration test, or legal review
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
- **Noisy-analyst-label stress test** is reported as not testable (needs the feedback engine's learning side, V1).
- **Not built (later phases):** calibrated mode, counterfactuals, graph store, automation above level 0,
  policy replay, AI copilot, regulator export views.
- **Open product decisions** still on working defaults: see [docs/adr/0001-gate-0-defaults.md](docs/adr/0001-gate-0-defaults.md).
