"""A worker that has never seen a candidate loads it from the shared directory, verified first."""

import pytest

from assay.api.main import candidate_loader
from assay.detection import save_bundle
from assay.detection.bundle import BundleError
from assay.scoring import BundleRegistry, ScoringError

T, KEY = "tenant-synth", b"k"


@pytest.fixture
def saved(trained, tmp_path):
    r = trained[3]
    save_bundle(tmp_path / T / r.manifest.bundle_id, r.scoring_bundle(), r.manifest, KEY)
    return tmp_path, r.manifest.bundle_id


def test_a_signed_candidate_is_loaded_on_demand_and_is_not_made_champion(saved):
    root, bid = saved
    reg = BundleRegistry()
    reg.loader = candidate_loader(reg, str(root), KEY)
    assert reg.get(T, bid).manifest.bundle_id == bid
    with pytest.raises(ScoringError):
        reg.champion(T)  # loading a candidate never makes it the champion


def test_unknown_and_malicious_ids_load_nothing(saved):
    root, _ = saved
    reg = BundleRegistry()
    load = candidate_loader(reg, str(root), KEY)
    for tenant, bid in ((T, "nope"), (T, ".."), (T, "../x"), ("../" + T, "b"), (T, "a/b"), (T, "")):
        assert load(tenant, bid) is None, (tenant, bid)
    with pytest.raises(ScoringError):
        reg.get(T, "nope")


def test_a_tampered_artefact_is_refused_before_it_is_deserialised(saved):
    root, bid = saved
    art = next(p for p in (root / T / bid).iterdir() if p.suffix in (".joblib", ".pkl", ".bin", ".sealed")
               or p.name.startswith("artifact"))
    art.write_bytes(art.read_bytes() + b"tamper")
    reg = BundleRegistry()
    reg.loader = candidate_loader(reg, str(root), KEY)
    with pytest.raises(BundleError, match="hash"):
        reg.get(T, bid)


def test_the_wrong_signing_key_is_refused(saved):
    root, bid = saved
    reg = BundleRegistry()
    reg.loader = candidate_loader(reg, str(root), b"another-key")
    with pytest.raises(BundleError, match="signature"):
        reg.get(T, bid)
