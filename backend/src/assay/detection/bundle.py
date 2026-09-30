"""Signed, tenant-bound model bundle (PRD 12.1, 12.4, 15.3, FR-08).

The manifest records what produced the model (dataset, code commit, parameters, metrics). The
artifact's SHA-256 is signed together with the tenant and bundle id, and the loader:
  1. refuses a bundle built for another tenant,
  2. checks the artifact hash and signature BEFORE deserialising (joblib uses pickle, so loading
     an unverified file would run arbitrary code).
"""

from __future__ import annotations

import hashlib
import hmac
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import joblib

ARTIFACT = "model.joblib"
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


def save_bundle(directory: str | Path, model: Any, manifest: BundleManifest, key: bytes) -> Path:
    assert_single_tenant(manifest)
    d = Path(directory)
    d.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, d / ARTIFACT)
    manifest.artifact_sha256 = _sha256(d / ARTIFACT)
    manifest.signature = _sign(key, manifest.artifact_sha256, manifest.tenant_id, manifest.bundle_id)
    (d / MANIFEST).write_text(json.dumps(asdict(manifest), indent=2, default=str), encoding="utf-8")
    return d


def load_bundle(directory: str | Path, tenant_id: str, key: bytes) -> tuple[Any, BundleManifest]:
    d = Path(directory)
    m = BundleManifest(**json.loads((d / MANIFEST).read_text(encoding="utf-8")))
    if m.tenant_id != tenant_id:
        raise BundleError("bundle belongs to a different tenant")
    sha = _sha256(d / ARTIFACT)
    if not hmac.compare_digest(sha, m.artifact_sha256):
        raise BundleError("artifact hash does not match manifest")
    if not hmac.compare_digest(_sign(key, sha, m.tenant_id, m.bundle_id), m.signature):
        raise BundleError("bad signature")
    assert_single_tenant(m)
    return joblib.load(d / ARTIFACT), m  # only after every check above
