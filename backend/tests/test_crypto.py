"""FR-41: per-tenant envelope encryption, key rotation, and sealed model bundles."""

import json
import os

import pytest

pytest.importorskip("cryptography")

from assay.crypto import (
    CryptoError,
    LocalKeyProvider,
    key_version,
    open_sealed,
    rewrap,
    seal,
)
from assay.detection import BundleError, load_bundle, rewrap_bundle, save_bundle
from assay.detection.bundle import _write_manifest

MASTER = b"m" * 32
A, B = "tenant-a", "tenant-b"
SIGN = b"signing-key"


@pytest.fixture
def kp():
    return LocalKeyProvider(MASTER)


def test_round_trip_and_the_payload_is_not_readable_in_the_blob(kp):
    blob = seal(kp, A, b"patient zero", "ctx")
    assert b"patient zero" not in blob and open_sealed(kp, A, blob, "ctx") == b"patient zero"
    assert seal(kp, A, b"x", "ctx") != seal(kp, A, b"x", "ctx")  # fresh data key and nonces each time


def test_one_tenants_blob_does_not_open_for_another(kp):
    blob = seal(kp, A, b"secret", "ctx")
    with pytest.raises(CryptoError, match="wrong tenant or context"):
        open_sealed(kp, B, blob, "ctx")


def test_a_blob_moved_to_another_object_does_not_open(kp):
    blob = seal(kp, A, b"secret", "bundle|b-1")
    with pytest.raises(CryptoError):
        open_sealed(kp, A, blob, "bundle|b-2")


def test_tenant_and_context_cannot_be_shifted_across_the_boundary(kp):
    """('ab', 'c') and ('a', 'bc') are different bindings, not the same concatenation."""
    blob = seal(kp, "ab", b"x", "c")
    with pytest.raises(CryptoError):
        open_sealed(kp, "a", blob, "bc")


def test_every_flipped_byte_is_detected(kp):
    blob = bytearray(seal(kp, A, b"payload bytes", "ctx"))
    for i in range(len(blob)):
        bad = bytearray(blob)
        bad[i] ^= 0x01
        with pytest.raises(CryptoError):  # header, wrapped key, nonce or ciphertext: all refused
            open_sealed(kp, A, bytes(bad), "ctx")


def test_truncated_or_foreign_data_is_refused(kp):
    blob = seal(kp, A, b"x", "ctx")
    for bad in (blob[:-1], blob[:10], b"", b"not a blob at all" * 10):
        with pytest.raises(CryptoError):
            open_sealed(kp, A, bad, "ctx")


def test_different_tenants_get_different_keys_and_a_short_master_is_refused():
    p = LocalKeyProvider(MASTER)
    assert p.kek(A, 1) != p.kek(B, 1) != p.kek(A, 2) and len(p.kek(A, 1)) == 32
    assert LocalKeyProvider(MASTER).kek(A, 1) == p.kek(A, 1)  # deterministic: survives a restart
    assert LocalKeyProvider(b"n" * 32).kek(A, 1) != p.kek(A, 1)  # a different master gives different keys
    with pytest.raises(CryptoError):
        LocalKeyProvider(b"short")


def test_rotation_new_seals_use_the_new_version_and_old_blobs_still_open(kp):
    old = seal(kp, A, b"before", "c")
    assert key_version(old) == 1
    assert kp.rotate(A) == 2
    new = seal(kp, A, b"after", "c")
    assert key_version(new) == 2 and open_sealed(kp, A, old, "c") == b"before"
    assert key_version(seal(kp, B, b"x", "c")) == 1  # another tenant's version is untouched


def test_rewrap_moves_to_the_current_key_without_touching_the_payload(kp):
    old = seal(kp, A, b"big payload" * 1000, "c")
    kp.rotate(A)
    new = rewrap(kp, A, old, "c")
    assert key_version(new) == 2 and open_sealed(kp, A, new, "c") == b"big payload" * 1000
    assert new[-1000:] == old[-1000:]  # the payload ciphertext is byte-for-byte unchanged
    assert rewrap(kp, A, new, "c") == new  # already current: a no-op
    with pytest.raises(CryptoError):
        rewrap(kp, B, old, "c")  # not another tenant's to rewrap


def test_a_revoked_version_stops_opening_anything_but_a_rewrapped_copy_survives(kp):
    old = seal(kp, A, b"keep me", "c")
    with pytest.raises(CryptoError, match="current"):
        kp.revoke(A, 1)  # cannot revoke the version still in use
    kp.rotate(A)
    kept = rewrap(kp, A, old, "c")
    kp.revoke(A, 1)
    assert open_sealed(kp, A, kept, "c") == b"keep me"
    with pytest.raises(CryptoError, match="not available"):
        open_sealed(kp, A, old, "c")  # the copy under the revoked key is now unreadable


# -- sealed model bundles -------------------------------------------------------------------------------
@pytest.fixture
def trained_parts(trained):
    return trained[3].scoring_bundle(), trained[3].manifest


def manifest_for(manifest, tenant):
    return type(manifest)(**{**manifest.__dict__, "tenant_id": tenant, "dataset_tenant_ids": [tenant]})


def test_a_sealed_bundle_is_not_plaintext_on_disk_and_loads_with_the_key(tmp_path, kp, trained_parts):
    model, m = trained_parts
    d = save_bundle(tmp_path / "b", model, manifest_for(m, A), SIGN, encrypt_with=kp)
    assert (d / "model.joblib.sealed").exists() and not (d / "model.joblib").exists()
    assert json.loads((d / "manifest.json").read_text())["encrypted"] is True
    art, loaded = load_bundle(d, A, SIGN, decrypt_with=kp)
    assert loaded.encrypted and set(art) >= {"model", "reference", "store"}


def test_a_sealed_bundle_needs_a_key_provider_and_the_right_tenant(tmp_path, kp, trained_parts):
    model, m = trained_parts
    d = save_bundle(tmp_path / "b", model, manifest_for(m, A), SIGN, encrypt_with=kp)
    with pytest.raises(BundleError, match="no key provider"):
        load_bundle(d, A, SIGN)
    with pytest.raises(BundleError, match="different tenant"):
        load_bundle(d, B, SIGN, decrypt_with=kp)
    with pytest.raises(BundleError, match="bad signature"):
        load_bundle(d, A, b"wrong-signing-key", decrypt_with=kp)
    with pytest.raises(BundleError, match="cannot decrypt"):
        load_bundle(d, A, SIGN, decrypt_with=LocalKeyProvider(b"x" * 32))  # a different master secret


def test_tampering_with_a_sealed_artifact_is_caught_before_anything_is_decrypted(tmp_path, kp, trained_parts):
    model, m = trained_parts
    d = save_bundle(tmp_path / "b", model, manifest_for(m, A), SIGN, encrypt_with=kp)
    f = d / "model.joblib.sealed"
    raw = bytearray(f.read_bytes())
    raw[-5] ^= 0xFF
    f.write_bytes(bytes(raw))
    with pytest.raises(BundleError, match="hash does not match"):
        load_bundle(d, A, SIGN, decrypt_with=kp)


def test_a_sealed_artifact_cannot_be_swapped_into_another_bundle(tmp_path, kp, trained_parts):
    """Even with a correct hash and signature for the new bundle, the encryption is bound to the old id."""
    model, m = trained_parts
    d1 = save_bundle(tmp_path / "one", model, manifest_for(m, A), SIGN, encrypt_with=kp)
    m2 = manifest_for(m, A)
    m2.bundle_id = "b-other"
    d2 = save_bundle(tmp_path / "two", model, m2, SIGN, encrypt_with=kp)
    (d2 / "model.joblib.sealed").write_bytes((d1 / "model.joblib.sealed").read_bytes())
    _write_manifest(d2, m2, SIGN)  # a forger who can re-sign: the cipher still refuses
    with pytest.raises(BundleError, match="cannot decrypt"):
        load_bundle(d2, A, SIGN, decrypt_with=kp)


def test_require_encryption_refuses_a_plaintext_bundle(tmp_path, kp, trained_parts):
    model, m = trained_parts
    d = save_bundle(tmp_path / "plain", model, manifest_for(m, A), SIGN)
    assert load_bundle(d, A, SIGN)[1].encrypted is False  # plaintext still loads by default
    with pytest.raises(BundleError, match="requires it"):
        load_bundle(d, A, SIGN, require_encryption=True)


def test_rotating_the_key_rewraps_a_bundle_and_it_still_loads(tmp_path, kp, trained_parts):
    model, m = trained_parts
    d = save_bundle(tmp_path / "b", model, manifest_for(m, A), SIGN, encrypt_with=kp)
    assert rewrap_bundle(d, A, SIGN, kp) is False  # already on the current version
    kp.rotate(A)
    assert rewrap_bundle(d, A, SIGN, kp) is True
    assert key_version((d / "model.joblib.sealed").read_bytes()) == 2
    kp.revoke(A, 1)
    assert load_bundle(d, A, SIGN, decrypt_with=kp)[1].encrypted  # re-signed, and v1 is no longer needed
    assert not list(d.glob("*.tmp"))


def test_rewrap_refuses_what_it_cannot_trust(tmp_path, kp, trained_parts):
    model, m = trained_parts
    d = save_bundle(tmp_path / "b", model, manifest_for(m, A), SIGN, encrypt_with=kp)
    kp.rotate(A)
    with pytest.raises(BundleError, match="different tenant"):
        rewrap_bundle(d, B, SIGN, kp)
    with pytest.raises(BundleError, match="bad signature"):
        rewrap_bundle(d, A, b"wrong", kp)
    plain = save_bundle(tmp_path / "p", model, manifest_for(m, A), SIGN)
    with pytest.raises(BundleError, match="not encrypted"):
        rewrap_bundle(plain, A, SIGN, kp)


def test_server_config_requires_a_key_provider_when_encryption_is_required(monkeypatch):
    from assay.api.main import key_provider_from_env, load_registry

    monkeypatch.delenv("ASSAY_MASTER_KEY", raising=False)
    assert key_provider_from_env() is None
    with pytest.raises(RuntimeError, match="no key provider"):
        load_registry([], SIGN, None, require_encryption=True)
    monkeypatch.setenv("ASSAY_MASTER_KEY", "k" * 40)
    assert isinstance(key_provider_from_env(), LocalKeyProvider)
    monkeypatch.setenv("ASSAY_REQUIRE_ENCRYPTED_BUNDLES", "1")
    with pytest.raises(RuntimeError, match="no key provider"):
        load_registry([], SIGN, None)  # the env flag alone is enough to demand a provider


def test_os_random_is_not_reused_for_nonces(kp):
    blobs = {seal(kp, A, b"x")[8:20] for _ in range(200)}  # the wrap nonce region
    assert len(blobs) == 200 and os.urandom(1)


def test_the_server_loader_opens_a_sealed_bundle_and_enforces_the_requirement(tmp_path, kp, trained_parts):
    from assay.api.main import load_registry

    model, m = trained_parts
    sealed = save_bundle(tmp_path / "sealed", model, manifest_for(m, A), SIGN, encrypt_with=kp)
    plain = save_bundle(tmp_path / "plain", model, manifest_for(m, A), SIGN)
    reg = load_registry([{"tenant_id": A, "path": str(sealed)}], SIGN, kp, require_encryption=True)
    assert reg.champion(A).manifest.encrypted
    with pytest.raises(BundleError, match="requires it"):  # one unsealed bundle fails the whole start-up
        load_registry([{"tenant_id": A, "path": str(plain)}], SIGN, kp, require_encryption=True)


@pytest.mark.parametrize("crash_after", ["keep_original", "install_new", "write_manifest"])
def test_an_interrupted_rewrap_is_recovered_by_running_it_again(tmp_path, kp, trained_parts, monkeypatch, crash_after):
    """Crash at each step of the rewrap. The bundle is left in a state that fails loudly, never one
    that loads wrongly, and a rerun finishes the job."""
    import pathlib

    import assay.detection.bundle as bmod

    model, m = trained_parts
    d = save_bundle(tmp_path / "b", model, manifest_for(m, A), SIGN, encrypt_with=kp)
    kp.rotate(A)
    real_replace, real_manifest = pathlib.Path.replace, bmod._write_manifest
    calls = {"n": 0}

    def flaky_replace(self, target):
        calls["n"] += 1
        if (crash_after, calls["n"]) in {("keep_original", 1), ("install_new", 2)}:
            raise OSError("power cut")
        return real_replace(self, target)

    def flaky_manifest(*a, **k):
        raise OSError("power cut")

    monkeypatch.setattr(pathlib.Path, "replace", flaky_replace)
    if crash_after == "write_manifest":
        monkeypatch.setattr(bmod, "_write_manifest", flaky_manifest)
    with pytest.raises(OSError):
        rewrap_bundle(d, A, SIGN, kp)
    monkeypatch.undo()

    if crash_after != "keep_original":  # the on-disk state is inconsistent, and says so rather than loading
        with pytest.raises(BundleError):
            load_bundle(d, A, SIGN, decrypt_with=kp)
    assert rewrap_bundle(d, A, SIGN, kp) is True
    assert key_version((d / "model.joblib.sealed").read_bytes()) == 2
    assert load_bundle(d, A, SIGN, decrypt_with=kp)[1].encrypted
    assert [p.name for p in d.iterdir() if p.suffix in (".tmp", ".prev")] == []
    assert real_manifest is bmod._write_manifest
