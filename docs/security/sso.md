# Single sign-on and MFA (FR-42)

Status: **pre-pilot, and not tested against a real identity provider.** The verification rules are
tested with keys and tokens generated in the test suite (`backend/tests/test_sso.py`), and the browser sign-in is tested end to end
against a STAND-IN provider that issues real RS256 tokens (`backend/tests/mock_idp.py`, `web/scripts/e2e-sso.mjs`). That shows our
client and server agree with each other. It does not show that any particular provider's tokens or endpoints satisfy them. No
security review has been done (PRD Gate 1).

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

## The browser sign-in

When the server publishes sign-in settings, the web app makes a person sign in before it shows or requests anything. Otherwise it
is the demo, which uses its signing proxy (`X-Demo-User`, which is not authentication and exists only for the demo).

```
ASSAY_OIDC_CLIENT_ID=assay-web
ASSAY_OIDC_AUTHORIZATION_ENDPOINT=https://idp.example.org/authorize
ASSAY_OIDC_TOKEN_ENDPOINT=https://idp.example.org/token
# optional
ASSAY_OIDC_SCOPE="openid profile"                       # default: openid
ASSAY_OIDC_REDIRECT_URI=https://assay.example.org/      # default: where the app is served from
ASSAY_OIDC_END_SESSION_ENDPOINT=https://idp.example.org/logout
ASSAY_OIDC_AUTH_AUDIENCE=assay-api                      # for providers that mint access tokens per API (Auth0 style)
ASSAY_OIDC_TOKEN_USE=access_token                       # or id_token, if the API audience is the web client's id
```
These are public values, served unauthenticated at `GET /v1/auth/config` and only while single sign-on itself is on. Endpoints must
be `https` (a local `http://127.0.0.1` address is allowed, for testing against a stand-in provider).

**Flow.** Authorization code with PKCE (S256), `state` and `nonce`. The web app is a public client, so there is no client secret.
The browser asks the provider's token endpoint for the token directly, so **the provider must allow cross-origin requests from the
app's origin** (most do for single-page apps; it is a setting to check). The token goes to the API as `Authorization: Bearer`.
`GET /v1/me` then reports the person's subject, tenant and roles from the **verified token**; the web app shows the screens those
roles allow, and nothing in the browser grants access.

What the browser checks: `state` (a callback it did not start is refused; an attempt can be completed once) and the ID token's
`nonce`. What it does not check, because the API does on every request: the token's signature, issuer, audience, lifetime, MFA and roles.

Where the token is kept: in memory and in `sessionStorage`, so a reload does not sign the person out. `sessionStorage` is per-tab and
cleared when the tab closes, but any script on the page can read it. The app loads no third-party script. Keep token lifetimes short
at the provider; the API also refuses a token whose `iat` is older than `ASSAY_OIDC_MAX_TOKEN_AGE_S`.

Outcomes a person sees: no session means the sign-in screen and **no API calls**; a sign-in with a password only gives "multi-factor
authentication is required"; someone with no mapped group signs in and is told they have no access; a token the API stops accepting
ends the session; the provider cancelling or refusing returns to sign-in with its reason. Signing out clears the session and, if the
provider has a logout endpoint, ends its session too.

Try it: `python backend/scripts/demo_sso.py` serves the app on port 8001 with a stand-in provider on port 8100, and
`node web/scripts/e2e-sso.mjs` drives it in a real browser.

## What does not exist

- **Any test against a real provider.** The stand-in is ours: it uses the claim names we expect, allows any origin, and has no
  refresh, consent screen, session cookie, step-up or clock skew. A real provider needs the client registered (public client, PKCE,
  the redirect address), its CORS settings checked, its group and MFA claims mapped, and the sign-in tested with real accounts.
- **Silent renewal and refresh tokens.** When a token runs out the person signs in again. That is deliberate (short sessions, nothing
  long-lived in the browser), and it costs convenience.
- **A back-channel logout.** Signing out ends this tab's session and the provider's, but other tabs and devices keep theirs until
  their tokens expire.
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
