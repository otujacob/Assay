# Key management and rotation (FR-41)

Status: **pre-pilot.** This is a working design and runbook, not a security review. No independent
assessment has been done (PRD Gate 1).

## What is encrypted, and what is not

| Asset | Protection | Where it is verified |
|---|---|---|
| Model bundle artefacts | Sealed per tenant with AES-256-GCM envelope encryption (`assay.crypto`) | `backend/tests/test_crypto.py` |
| PostgreSQL data and backups | **Not done by this code.** A deployment setting on the managed database (storage encryption) | The hosting platform's configuration |
| Audit exports | Sealed per tenant on request (`?seal=true`), opened with the command-line tool below | `backend/tests/test_crypto.py` |
| Datasets and validation reports | **Not done yet.** They are not written to object storage by this code, so there is nothing to seal; if they are, they need the same treatment | Open item |
| Traffic | TLS, and mutual TLS between services: a deployment setting | Not built here |

The claim "all stores encrypted, per-tenant keys" (PRD 16) is therefore **only partly met**: model
artefacts and audit exports can be sealed per tenant, and the rest (database, backups, traffic)
depends on deployment settings.

## How it works

Each sealed object gets a random data key. That key is wrapped by the tenant's key-encryption key
(KEK). The tenant id and a context (for a bundle, `bundle|<bundle_id>`) are bound into both layers,
so a blob sealed for one tenant, or moved to another object, will not open. A KEK has a version;
new seals use the current version and old blobs name the version they were sealed under.

The signature and hash of a bundle cover the sealed file, so they are checked **before** anything is
decrypted or deserialised.

## Key providers

`assay.crypto.LocalKeyProvider` derives each tenant's KEK from one master secret
(`ASSAY_MASTER_KEY`, at least 32 characters). It is for development, tests and single-host installs.
**Whoever holds the master secret can open every tenant's artefacts**, so it is a weaker boundary
than the PRD intends (a separate key per tenant in a managed key service). Production should
implement the `KeyProvider` protocol over a managed key service (OPD-21 decides which), so a KEK
never leaves the service. Nothing else in the application changes.

## Sealed audit exports

`GET /v1/audit/export?seal=true` (auditors only, as for any export) returns the file encrypted with
the tenant's key, as `assay-audit.csv.sealed` or `assay-audit.json.sealed`. If no key provider is
configured the request is **refused with 409**, never answered in the clear. Reading the log is
audited either way. A sealed export opens only for the tenant it was made for, and only as an
audit export.

To open one, on a machine that holds the master secret:
```
python -m assay.crypto open assay-audit.csv.sealed --tenant tenant-a -o assay-audit.csv
```
It will not overwrite an existing file without `--force`. After a rotation, bring an old export to the
current key with `python -m assay.crypto rewrap FILE --tenant T --key-version N`, then revoke the old
version. Anyone who can run the tool with the master secret can open every tenant's exports, so
treat it like the secret itself.

## Starting up

```
ASSAY_MASTER_KEY=<32+ characters from the secret store>
ASSAY_REQUIRE_ENCRYPTED_BUNDLES=1      # refuse any bundle that is not sealed
```
With the second variable set, the server and the worker refuse to start if a bundle is plaintext
or no key provider is configured.

## Rotating a tenant's key

Rotate on a schedule the institution agrees, and immediately if a key may have been exposed.

1. `provider.rotate(tenant_id)`: new seals use the next version. Existing bundles still open.
2. For each of the tenant's bundles: `rewrap_bundle(path, tenant_id, signing_key, provider)`.
   This moves the data key under the new KEK and re-signs the manifest. The model is not
   decrypted or rewritten. It returns `False` if the bundle is already current.
3. Restart the server and the worker, and confirm the bundles load.
4. `provider.revoke(tenant_id, old_version)`: the old version can no longer open anything.
   The current version cannot be revoked, so rotate first.

If the **master secret** itself is exposed, rotation of a version is not enough: every KEK derives
from it. Change the master secret, and re-seal every bundle from its signed source (a rewrap
cannot help, because it needs the old KEK to unwrap).

## Known limits

- The local provider keeps the current version per tenant in memory. A restart forgets a rotation
  unless the deployment sets it again. A managed key service stores this itself, which is one more
  reason it is the production answer.
- `rewrap_bundle` keeps the original file as `model.joblib.sealed.prev` until the new manifest is
  written. If it is interrupted, the bundle fails to load (loudly, with a `BundleError`) rather than
  loading wrongly, and running `rewrap_bundle` again restores the original and finishes the job.
  This is tested with a simulated crash at each step.
- Nothing here protects against someone who can already run code as the application, because the
  application must hold the key to use the model.
