# Gate 1 readiness: what stands between this and real customer data

PRD 22.3: no real customer data enters the system until the security baseline is in place, an
**independent** penetration test is done, tenant-isolation tests pass, and data-protection and
contract review is completed **by professionals**.

Status: **Gate 1 is not passed, and nothing in this repository can pass it on its own.** Two of the
four conditions need people outside this codebase. This page says what is built, what evidence
exists for it, and what is needed from others. It is a working document, not a claim of compliance.

## The four conditions

| Condition | State | Evidence in the repo | What is still needed |
|---|---|---|---|
| Security baseline (PRD 16) | **Partly built** | See the table below | The gaps below, then the independent test |
| Independent penetration test | **Not done** | The scoping brief below | A tester who is not the author, a scope agreed, findings tracked to closure |
| Tenant-isolation tests passing | **Done, on synthetic data** | Three layers. **Database:** row-level security on every table, tested as the restricted application role (`test_pg_repo.py`, `test_db_live.py`). **Every API route:** `test_cross_tenant_api.py` has a second tenant holding every role attack the first tenant's ids on all routes, and checks it finds nothing, changes nothing and leaks no identifier (it fails if tenant scoping is broken on any route I tried: decisions, review cases, policies, audit export, queue). **Real PostgreSQL:** the same attack through the pooled server (`test_api_scoring_pg.py`). All pass in CI on PostgreSQL 16 | Re-run against the real deployment. The route-level test of every route runs on the in-memory store; only the decision routes are repeated on PostgreSQL. Add the tester's own attempts |
| Data-protection and contract review | **Not started** | PRD 17.1 lists what needs advice | A solicitor or privacy adviser for each market (UK first, per the working defaults) |

## Security baseline (PRD 16), control by control

| Control | State | Notes |
|---|---|---|
| Encryption at rest | **Partly** | Model bundles and audit exports can be sealed per tenant. Database and backup encryption is a hosting setting, not verified here |
| Encryption in transit | **Not built here** | TLS and mutual TLS are deployment settings. The API speaks plain HTTP behind whatever terminates TLS |
| Authentication, SSO | **Built, not interoperability-tested** | Bearer-token verification; no browser sign-in flow; see `sso.md` |
| MFA | **Enforced through the token** | Relies on the provider reporting it truthfully in `amr` or `acr` |
| Role-based access | **Built** | Permission tests per route. Separation of duties for policies (proposer cannot approve) is enforced in the service and the database |
| Audit logs | **Built** | Append-only, hash-chained, row-level secured. Failed sign-ins and refused roles are recorded; reading the log is recorded |
| Tenant isolation | **Built** | As above |
| Secrets management | **Partly** | Secrets come from the environment. No secret manager integration. CI scans history for committed secrets and passed |
| API security | **Partly** | Signed requests, schema validation, rate limiting (per process), idempotency. Signing secrets are shared and have no rotation process |
| Privileged access | **Not built** | No just-in-time support access |
| Secure development | **Partly** | CI runs lint, tests, a dependency audit (passing) and a secret scan (passing). No signed builds, no SBOM, no container scanning |
| Vulnerability management | **Partly** | The dependency audit gates every push. No patch-by-severity policy or tracker |
| Incident response | **Not started** | No written plan |
| Backups, disaster recovery, ML-specific security | **Not started** | V1 in the PRD |

## Known weak points

Written down so a tester does not have to find them and so nobody assumes they are fine.

1. **The demo server is a hole by design.** `backend/scripts/demo_server.py` signs requests for any
   caller who names a demo user, with no authentication. It must never be deployed. Nothing stops
   it being deployed except this sentence and the banner it shows.
2. **`ASSAY_DEV_CREDENTIALS`** is a list of shared secrets in an environment variable. It is the
   only way service callers authenticate today.
3. **One master secret opens every tenant's sealed files** (`ASSAY_MASTER_KEY`, the local key
   provider). A managed key service per tenant is the intended design and is not built.
4. **Model bundles are pickles.** Loading one runs code. The defence is that the signature and hash
   are checked first, with a secret key; whoever holds `ASSAY_BUNDLE_SIGNING_KEY` can make a bundle
   that runs code on load. Protect it like a deployment credential.
5. **Tokens cannot be revoked.** A signed-in person stays signed in until their token ages out
   (default one hour).
6. **The tenant claim is trusted.** Whoever controls the identity provider's tenant claim can act as
   any tenant. One issuer per server only.
7. **Rate limits are per process,** so they weaken with the number of workers.
8. **The audit log can be read by every auditor of a tenant** and its CSV export is not signed, so
   a copy can be altered after it leaves. A sealed export protects confidentiality, not integrity;
   the hash chain verifies the stored log, not an exported copy.
9. **Nothing here has been tried by an adversary.** The tests are the author's.

## Penetration test: suggested scope

Give the tester this, and the source. The point is an independent view, so ask them to choose where
to spend their time.

**In scope**
- The HTTP API (`backend/src/assay/api/app.py`): every route listed there, both authentication
  modes (signed requests, bearer tokens), and the role checks on each.
- **Tenant isolation:** reading or writing another tenant's data by any route, by id guessing, by
  crafted tokens or crafted request bodies, and directly against PostgreSQL as the application role.
- The **append-only and hash-chain guarantees**: can a row be changed or removed without detection,
  by the application role or by a crafted request?
- Token verification (`backend/src/assay/auth/`): forged, replayed, confused-algorithm and
  wrong-audience tokens; key-set fetching.
- **Model bundle loading** (`detection/bundle.py`): can an unsigned, swapped or downgraded bundle be
  loaded? Can the signature check be bypassed?
- The **blind review** rule: can a plain analyst learn the score of a case under blind review by any
  route?
- Audit export and CSV handling (injection into spreadsheets, scope).
- Dependency and CI supply chain (the workflow, the lockfiles).

**Out of scope, or to be agreed:** the demo server (it is insecure on purpose and must not be
deployed), the hosting platform and its TLS, social engineering, denial-of-service load testing
(the rate limit is a convenience, not a defence).

**Give the tester:** a deployment with synthetic data only, two tenants, one credential of each
role, and a token issuer they can control for the sign-in tests. Agree how findings are reported
and tracked to closure, and retest after fixes.

## Legal and compliance: what to ask a professional

From PRD 17.1. None of this has been advised on, and nothing here is a legal claim.

- Lawful basis and controller or processor roles for fraud scoring and analyst review.
- The data protection law that applies in each market, and rules on automated decision-making and
  on explanations owed to customers (Assay recommends and does not automate at level 0; confirm
  that is enough).
- Cross-border transfer and data residency.
- Minimum retention for financial records against deletion requests (OPD-15), including how a
  deletion request interacts with an append-only audit log, and the open question of removing a
  person's influence from an already trained model.
- Supervisory expectations on model risk and AI governance.
- Contract terms with institutions, including who approves model promotions (OPD-12).
- Whether analyst performance data may be used as this design uses it (PRD 8.5: not for
  performance management, which is the institution's own policy).
- The PRD's own boundary: that Assay makes no certification or compliance claim until one is
  separately obtained and verified.

## Decisions only you can make

OPD-16 (isolation tier), OPD-18 (certification path and recovery objectives), OPD-21 (cloud and key
service), OPD-22 (who covers the security and independent-review roles before Gate 1). The open
decisions list is in the PRD, Closing E.
