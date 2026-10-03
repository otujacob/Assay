"""Per-tenant envelope encryption (PRD 15.1, 16; FR-41).

Every sealed object gets a random data key (DEK). The DEK is wrapped by the tenant's key-encryption
key (KEK), and the payload is encrypted with the DEK, both with AES-256-GCM. The tenant id and a
caller-supplied context (for example the bundle id) are bound into both layers as authenticated
data, so a blob sealed for one tenant, or moved to another object, will not open: the cipher
refuses it rather than returning the wrong tenant's bytes.

KEKs come from a `KeyProvider`. `LocalKeyProvider` derives them from a master secret and is for
development, tests and single-host installs. Production should implement the same protocol over a
managed key service (the PRD's proposal, section 24.3), so a KEK never leaves the service and
rotation is the service's own. Nothing else in the application changes.

Rotation: a KEK has a version. New seals use the current version; old blobs name the version they
were sealed under, so they still open. `rewrap` moves a blob to the current KEK without touching its
payload, which is how rotation, and revoking an old version, are carried out.

What this does not do: it is not disk or database encryption. Encrypting PostgreSQL storage and
backups is a deployment setting on the managed database, and is verified there.

Blob layout: MAGIC(4) | kek_version(4, big-endian) | wrap_nonce(12) | wrapped_dek(48) |
data_nonce(12) | ciphertext+tag.
"""

from __future__ import annotations

import hashlib
import hmac
import os
import struct
from dataclasses import dataclass, field
from typing import Protocol

from cryptography.exceptions import InvalidTag
from cryptography.hazmat.primitives.ciphers.aead import AESGCM

MAGIC = b"ASY1"
_HEAD = struct.Struct(">4sI")
_WRAP_NONCE, _WRAPPED_DEK, _DATA_NONCE = 12, 48, 12
_FIXED = _HEAD.size + _WRAP_NONCE + _WRAPPED_DEK + _DATA_NONCE
_MIN_MASTER = 32


class CryptoError(Exception):
    """Wrong tenant, wrong context, a tampered or truncated blob, or an unknown key version."""


class KeyProvider(Protocol):
    def current_version(self, tenant_id: str) -> int: ...

    def kek(self, tenant_id: str, version: int) -> bytes:
        """The 32-byte key-encryption key. Raise CryptoError for an unknown or revoked version."""
        ...


def _hkdf(master: bytes, info: bytes) -> bytes:
    """HKDF-SHA256 (RFC 5869) with an empty salt, one 32-byte block."""
    prk = hmac.new(b"\x00" * 32, master, hashlib.sha256).digest()
    return hmac.new(prk, info + b"\x01", hashlib.sha256).digest()


@dataclass
class LocalKeyProvider:
    """KEKs derived from one master secret, a distinct key per tenant and version.

    `versions` maps tenant to its current version (default 1). `revoked` lists (tenant, version)
    pairs that must no longer open anything. Keep the master secret in a secret store, never in code.
    """

    master: bytes
    versions: dict[str, int] = field(default_factory=dict)
    revoked: set[tuple[str, int]] = field(default_factory=set)

    def __post_init__(self) -> None:
        if len(self.master) < _MIN_MASTER:
            raise CryptoError(f"master secret must be at least {_MIN_MASTER} bytes")

    def current_version(self, tenant_id: str) -> int:
        return self.versions.get(tenant_id, 1)

    def kek(self, tenant_id: str, version: int) -> bytes:
        if version < 1 or (tenant_id, version) in self.revoked:
            raise CryptoError(f"key version {version} is not available for this tenant")
        return _hkdf(self.master, f"assay-kek|{tenant_id}|v{version}".encode())

    def rotate(self, tenant_id: str) -> int:
        """Start using the next key version for new seals. Old blobs keep opening until revoked."""
        self.versions[tenant_id] = self.current_version(tenant_id) + 1
        return self.versions[tenant_id]

    def revoke(self, tenant_id: str, version: int) -> None:
        if version == self.current_version(tenant_id):
            raise CryptoError("cannot revoke the current key version; rotate first")
        self.revoked.add((tenant_id, version))


def _aad(tenant_id: str, context: str, layer: bytes) -> bytes:
    # Length-prefixed, so ("ab", "c") and ("a", "bc") can never produce the same bytes.
    t, c = tenant_id.encode(), context.encode()
    return layer + struct.pack(">II", len(t), len(c)) + t + c


def seal(provider: KeyProvider, tenant_id: str, plaintext: bytes, context: str = "") -> bytes:
    version = provider.current_version(tenant_id)
    dek = os.urandom(32)
    wrap_nonce, data_nonce = os.urandom(_WRAP_NONCE), os.urandom(_DATA_NONCE)
    wrapped = AESGCM(provider.kek(tenant_id, version)).encrypt(
        wrap_nonce, dek, _aad(tenant_id, context, b"wrap"))
    body = AESGCM(dek).encrypt(data_nonce, plaintext, _aad(tenant_id, context, b"data"))
    return _HEAD.pack(MAGIC, version) + wrap_nonce + wrapped + data_nonce + body


def _split(blob: bytes) -> tuple[int, bytes, bytes, bytes, bytes]:
    if len(blob) < _FIXED + 16 or blob[:4] != MAGIC:
        raise CryptoError("not a sealed blob")
    _, version = _HEAD.unpack_from(blob)
    o = _HEAD.size
    wn, o = blob[o:o + _WRAP_NONCE], o + _WRAP_NONCE
    wrapped, o = blob[o:o + _WRAPPED_DEK], o + _WRAPPED_DEK
    dn, o = blob[o:o + _DATA_NONCE], o + _DATA_NONCE
    return version, wn, wrapped, dn, blob[o:]


def key_version(blob: bytes) -> int:
    """Which KEK version sealed this blob (read from the header; it is not secret)."""
    return _split(blob)[0]


def open_sealed(provider: KeyProvider, tenant_id: str, blob: bytes, context: str = "") -> bytes:
    version, wn, wrapped, dn, body = _split(blob)
    try:
        dek = AESGCM(provider.kek(tenant_id, version)).decrypt(wn, wrapped, _aad(tenant_id, context, b"wrap"))
        return AESGCM(dek).decrypt(dn, body, _aad(tenant_id, context, b"data"))
    except InvalidTag:
        # One message for every cause, so the error does not tell a caller which tenant owns a blob.
        raise CryptoError("cannot open: wrong tenant or context, or the data was altered") from None


def rewrap(provider: KeyProvider, tenant_id: str, blob: bytes, context: str = "") -> bytes:
    """Move a blob to the tenant's current KEK without decrypting its payload. A no-op if it is
    already current. The payload ciphertext is unchanged, so this is cheap for large artefacts."""
    version, wn, wrapped, dn, body = _split(blob)
    current = provider.current_version(tenant_id)
    # Authenticate first, even when nothing needs to change, so a wrong tenant or context is an
    # error rather than a silent no-op.
    try:
        dek = AESGCM(provider.kek(tenant_id, version)).decrypt(wn, wrapped, _aad(tenant_id, context, b"wrap"))
    except InvalidTag:
        raise CryptoError("cannot rewrap: wrong tenant or context, or the data was altered") from None
    if version == current:
        return blob
    new_nonce = os.urandom(_WRAP_NONCE)
    new_wrapped = AESGCM(provider.kek(tenant_id, current)).encrypt(
        new_nonce, dek, _aad(tenant_id, context, b"wrap"))
    return _HEAD.pack(MAGIC, current) + new_nonce + new_wrapped + dn + body
