# Single sign-on and MFA (FR-42)

Status: **pre-pilot, and not tested against a real identity provider.** The verification rules are
tested with keys and tokens generated in the test suite (`backend/tests/test_sso.py`). That shows
the rules work. It does not show that any particular provider's tokens satisfy them. No security
review has been done (PRD Gate 1).

## What exists

The API accepts `Authorization: Bearer <token>` from an OpenID Connect provider. Service callers
(ingestion) keep using signed requests. A request carrying a bearer token is verified, turned into
the same kind of identity a signed request has, and then goes through the same role checks, rate
limits and audit log.

A token is refused unless all of these hold:

| Check | Detail |
|---|---|
| Signature | Asymmetric algorithms only (RS/ES/PS 256-512), by a key in the provider's published key set. `none` and shared-secret algorithms are refused before any key is looked at, which closes the "sign with the public key as a secret" attack |
| Issuer and audience | `iss` equals the configured issuer, `aud` includes this API |
| Lifetime | `exp` not passed, `nbf` reached, `iat` not in the future, and `iat` no older than `ASSAY_OIDC_MAX_TOKEN_AGE_S` (default 3600): the person must have signed in recently, whatever `exp` says. Clock skew of 30 s is tolerated |
| MFA | `amr` contains `mfa`, or `acr` is one of `ASSAY_OIDC_MFA_ACR`. A single factor does not count |
| Tenant | A string in the tenant claim (default `assay_tenant`). The tenant comes from the token and nowhere else |
| Roles | The provider's groups (default claim `groups`) mapped through `ASSAY_OIDC_ROLE_MAP` to Assay roles. Unmapped groups grant nothing. A person with no mapped group authenticates but is refused everywhere |

Only human roles can be granted (`analyst`, `senior_analyst`, `manager`, `auditor`, `approver`,
`admin`). A token can never grant `ingest`, so a stolen user token cannot submit transactions, and
the server refuses to start if the role map tries.

A refused token is a 401 with `WWW-Authenticate: Bearer`. It says nothing about why, except
`mfa_required`, which the person can act on. The real reason goes to the application log.

## Configuration

```
ASSAY_OIDC_ISSUER=https://idp.example.org
ASSAY_OIDC_AUDIENCE=assay-api
ASSAY_OIDC_JWKS_URL=https://idp.example.org/.well-known/jwks.json   # must be https
ASSAY_OIDC_ROLE_MAP={"fraud-analysts":"analyst","fraud-auditors":"auditor","model-risk":"approver"}
# optional
ASSAY_OIDC_TENANT_CLAIM=assay_tenant
ASSAY_OIDC_ROLES_CLAIM=groups
ASSAY_OIDC_MFA_ACR=urn:example:mfa
ASSAY_OIDC_MAX_TOKEN_AGE_S=3600
```
Install with `pip install ".[sso]"`. Setting an issuer without the other required variables stops
the server at start-up rather than leaving a half-configured door. Turning MFA off
(`ASSAY_OIDC_REQUIRE_MFA=0`) exists for testing only.

The provider's key set is cached for an hour and refetched when a token names an unknown key (the
provider rotated), but not more than once a minute, so junk key ids cannot make this service
hammer the provider.

## What does not exist

- **The browser sign-in.** The web app still uses the demo proxy (`X-Demo-User`), which is not
  authentication. A real deployment needs the web app to sign people in with the provider
  (authorization code with PKCE) and send the token. That depends on the institution's provider,
  client registration and redirect addresses, so it is not built.
- **Interoperability.** Providers differ in where they put `amr`, groups and custom claims. Expect
  to adjust the claim names and role map, and test with the real provider before relying on it.
- **SAML.** The PRD allows SAML or OIDC; only OIDC is built.
- **One issuer per server.** A single issuer is configured for the whole process. If several
  institutions share one deployment, each with its own provider, this does not support that.
  Because the issuer asserts the tenant, **the provider must be trusted to set the tenant claim
  correctly**: whoever controls that claim can act as any tenant.
- **Revocation and logout.** A token is good until it expires or ages out. Disabling a person at the
  provider does not stop a token already issued, for up to the max token age. Keep that short.
- **Per-user rate limits and audit** are in place, keyed on the token's `sub`. The `sub` appears in
  the audit log as `sso-<sub>`, so use an opaque identifier, not an email address (PRD 17).
- **Service accounts** still use shared-secret signed requests, with no MFA and no expiry.
