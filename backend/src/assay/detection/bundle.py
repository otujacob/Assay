"""Signed, tenant-bound model bundle (PRD 12.1, 12.4, 15.3, FR-08).

The manifest records what produced the model (dataset, code commit, parameters, metrics). The
artifact's SHA-256 is signed together with the tenant and bundle id, and the loader:
  1. refuses a bundle built for another tenant,
  2. checks the artifact hash and signature BEFORE deserialising (joblib uses pickle, so loading
     an unverified file would run arbitrary code).

A bundle can also be sealed with the tenant's key (assay.crypto, FR-41). The hash and signature then
cover the sealed file, so they are still checked before anything is decrypted, and the bundle id is
bound into the encryption so a sealed artifact cannot be swapped into another bundle.
"""

from __future__ import annotations

import hashlib
import hmac
import io
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import joblib

ARTIFACT = "model.joblib"
ARTIFACT_SEALED = "model.joblib.sealed"
MANIFEST = "manifest.json"


class BundleError(Exception):
    pass


@dataclass
class BundleManifest:
    bundle_id: str
    tenant_id: str
    created_at: str
    feature_set_version: str
    definition_versions: dict[str, int]
    feature_names: list[str]
    dataset_id: str
    dataset_tenant_ids: list[str]
    code_commit: str
    params: dict[str, Any]
    thresholds: dict[str, float]
    metrics: dict[str, Any]
    calibration: dict[str, Any]
    status: str = "candidate"  # candidate | shadow | champion | retired (PRD 12.2)
    artifact_sha256: str = ""
    signature: str = ""
    encrypted: bool = False  # artifact is sealed with the tenant key (FR-41)
    extra: dict[str, Any] = field(default_factory=dict)


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _sign(key: bytes, sha: str, tenant_id: str, bundle_id: str) -> str:
    return hmac.new(key, f"{sha}|{tenant_id}|{bundle_id}".encode(), hashlib.sha256).hexdigest()


def assert_single_tenant(manifest: BundleManifest) -> None:
    """Gate G1 / PRD 15.3: training data must come from exactly the bundle's own tenant."""
    if manifest.dataset_tenant_ids != [manifest.tenant_id]:
        raise BundleError(f"dataset tenants {manifest.dataset_tenant_ids} != [{manifest.tenant_id}]")


def _context(bundle_id: str) -> str:
    return f"bundle|{bundle_id}"


def _write_manifest(d: Path, manifest: BundleManifest, key: bytes) -> None:
    manifest.artifact_sha256 = _sha256(d / (ARTIFACT_SEALED if manifest.encrypted else ARTIFACT))
    manifest.signature = _sign(key, manifest.artifact_sha256, manifest.tenant_id, manifest.bundle_id)
    (d / MANIFEST).write_text(json.dumps(asdict(manifest), indent=2, default=str), encoding="utf-8")


def save_bundle(directory: str | Path, model: Any, manifest: BundleManifest, key: bytes, *,
                encrypt_with=None) -> Path:
    """`encrypt_with` is an assay.crypto KeyProvider: when given, the artifact is sealed with the
    tenant's key and no plaintext copy is written."""
    assert_single_tenant(manifest)
    d = Path(directory)
    d.mkdir(parents=True, exist_ok=True)
    manifest.encrypted = encrypt_with is not None
    if encrypt_with is None:
        joblib.dump(model, d / ARTIFACT)
    else:
        from assay.crypto import seal

        buf = io.BytesIO()
        joblib.dump(model, buf)
        (d / ARTIFACT_SEALED).write_bytes(
            seal(encrypt_with, manifest.tenant_id, buf.getvalue(), _context(manifest.bundle_id)))
    _write_manifest(d, manifest, key)
    return d


def load_bundle(directory: str | Path, tenant_id: str, key: bytes, *, decrypt_with=None,
                require_encryption: bool = False) -> tuple[Any, BundleManifest]:
    """`require_encryption` refuses a plaintext bundle, so a deployment that expects sealed
    artifacts cannot be handed an unsealed one."""
    d = Path(directory)
    m = BundleManifest(**json.loads((d / MANIFEST).read_text(encoding="utf-8")))
    if m.tenant_id != tenant_id:
        raise BundleError("bundle belongs to a different tenant")
    if require_encryption and not m.encrypted:
        raise BundleError("bundle is not encrypted but this deployment requires it")
    if m.encrypted and decrypt_with is None:
        raise BundleError("bundle is encrypted but no key provider was given")
    path = d / (ARTIFACT_SEALED if m.encrypted else ARTIFACT)
    if not path.exists():
        raise BundleError(f"artifact {path.name} is missing")
    sha = _sha256(path)
    if not hmac.compare_digest(sha, m.artifact_sha256):
        raise BundleError("artifact hash does not match manifest")
    if not hmac.compare_digest(_sign(key, sha, m.tenant_id, m.bundle_id), m.signature):
        raise BundleError("bad signature")
    assert_single_tenant(m)
    if not m.encrypted:
        return joblib.load(path), m  # only after every check above
    from assay.crypto import CryptoError, open_sealed

    try:
        plain = open_sealed(decrypt_with, m.tenant_id, path.read_bytes(), _context(m.bundle_id))
    except CryptoError as e:
        raise BundleError(f"cannot decrypt bundle: {e}") from None
    return joblib.load(io.BytesIO(plain)), m


def _recover_rewrap(d: Path, m: BundleManifest) -> None:
    """Finish or undo a rewrap that was interrupted. The old file is kept as `.prev` until the new
    manifest is written. If the manifest still describes `.prev`, the rewrap never completed, so put
    the old file back; otherwise the manifest was updated and `.prev` is stale."""
    path, prev = d / ARTIFACT_SEALED, d / (ARTIFACT_SEALED + ".prev")
    for leftover in d.glob("*.tmp"):
        leftover.unlink()
    if prev.exists():
        if hmac.compare_digest(_sha256(prev), m.artifact_sha256):
            prev.replace(path)
        else:
            prev.unlink()


def rewrap_bundle(directory: str | Path, tenant_id: str, key: bytes, provider) -> bool:
    """Move a sealed bundle to the tenant's current key version and re-sign it (key rotation).
    Returns False if it was already current. The model itself is not decrypted or rewritten.
    Safe to rerun after a crash: see `_recover_rewrap`."""
    from assay.crypto import CryptoError, rewrap

    d = Path(directory)
    m = BundleManifest(**json.loads((d / MANIFEST).read_text(encoding="utf-8")))
    if m.tenant_id != tenant_id:
        raise BundleError("bundle belongs to a different tenant")
    if not m.encrypted:
        raise BundleError("bundle is not encrypted")
    _recover_rewrap(d, m)
    path = d / ARTIFACT_SEALED
    if not hmac.compare_digest(_sign(key, _sha256(path), m.tenant_id, m.bundle_id), m.signature):
        raise BundleError("bad signature")  # never rewrap something that was not verified
    old = path.read_bytes()
    try:
        new = rewrap(provider, m.tenant_id, old, _context(m.bundle_id))
    except CryptoError as e:
        raise BundleError(f"cannot rewrap bundle: {e}") from None
    if new == old:
        return False
    tmp, prev = path.with_suffix(".tmp"), d / (ARTIFACT_SEALED + ".prev")
    tmp.write_bytes(new)
    path.replace(prev)       # keep the verified original until the new manifest is in place
    tmp.replace(path)
    _write_manifest(d, m, key)
    prev.unlink()
    return True
