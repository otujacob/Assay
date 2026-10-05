from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from assay.detection import train_bundle
from assay.detection.labels import labels_as_of
from assay.features.compute import parse_time
from assay.learning import gates as G
from assay.learning.gates import FAIL, NOT_EVALUATED, PASS, GateConfig

T = "tenant-synth"


def _manifest(**kw):
    base = {"tenant_id": T, "dataset_tenant_ids": [T], "dataset_id": "d", "feature_set_version": "f",
            "extra": {"class_balance": {"train_pos": 100, "train_n": 2000}}, "feature_names": ["a"],
            "bundle_id": "b-1", "thresholds": {"t_high": 0.5, "t_low": 0.2}}
    base.update(kw)
    return SimpleNamespace(**base)


def test_g1_passes_a_well_formed_candidate_and_fails_each_defect():
    cfg = GateConfig()
    ok = G.g1_data(SimpleNamespace(manifest=_manifest()), T, cfg, None, None)
    assert ok.status == PASS
    for kw, word in [({"dataset_tenant_ids": [T, "other"]}, "single-tenant"),
                     ({"extra": {}}, "class balance"),
                     ({"extra": {"class_balance": {"train_pos": 3, "train_n": 2000}}}, "too few"),
                     ({"dataset_id": ""}, "incomplete")]:
        g = G.g1_data(SimpleNamespace(manifest=_manifest(**kw)), T, cfg, None, None)
        assert g.status == FAIL and word in g.detail, kw


def test_g1_refuses_analyst_labels_the_pool_did_not_accept():
    c = SimpleNamespace(manifest=_manifest())
    assert G.g1_data(c, T, GateConfig(), {"a", "b"}, {"a"}).status == PASS
    g = G.g1_data(c, T, GateConfig(), {"a"}, {"a", "zzz"})
    assert g.status == FAIL and "1 analyst labels" in g.detail


def test_g8_needs_a_signed_artefact_and_recorded_lineage():
    c = SimpleNamespace(manifest=_manifest())
    assert G.g8_integrity(c, artefact_signed=True, lineage_recorded=True).status == PASS
    g = G.g8_integrity(c, artefact_signed=False, lineage_recorded=True)
    assert g.status == FAIL and "not signed" in g.detail
    assert "lineage" in G.g8_integrity(c, artefact_signed=True, lineage_recorded=False).detail


def _res(y, p, t_high=0.5):
    n = len(y)
    return SimpleNamespace(test_y=np.array(y), test_pred=SimpleNamespace(calibrated=np.array(p)),
                           test_table=SimpleNamespace(txn_ids=[f"t{i}" for i in range(n)]),
                           manifest=_manifest(thresholds={"t_high": t_high, "t_low": 0.1}))


def test_g3_fails_a_poorly_calibrated_candidate_and_one_worse_than_the_champion():
    rng = np.random.default_rng(0)
    p = rng.uniform(0, 1, 4000)
    y = (rng.uniform(0, 1, 4000) < p).astype(int)  # calibrated by construction
    good = _res(y, p)
    assert G.g3_calibration(good, good, GateConfig()).status == PASS
    assert G.g3_calibration(_res(y, np.clip(p + 0.3, 0, 1)), good, GateConfig()).status == FAIL
    assert G.g3_calibration(_res(y, np.clip(p + 0.08, 0, 1)), good, GateConfig(ece_max=0.5)).status == FAIL


def test_g2_per_segment_recall_is_judged_only_with_enough_frauds():
    n = 400
    y = np.array([1, 0] * (n // 2))
    champ = _res(y, np.where(y == 1, 0.9, 0.1))
    cand = _res(y, np.where(y == 1, 0.9, 0.1))
    ids = champ.test_table.txn_ids
    seg = {t: ("big" if i < 300 else "tiny") for i, t in enumerate(ids)}
    assert G.g2_performance(cand, champ, GateConfig(), seg).status == PASS
    p = np.where(y == 1, 0.9, 0.1).astype(float)
    for i in range(0, 300, 2):  # the candidate misses every fraud in segment "big"
        p[i] = 0.1
    g = G.g2_performance(_res(y, p), champ, GateConfig(perf_tol=1.0), seg)
    assert g.status == FAIL and "segment big" in g.detail
    rows = {r["segment"]: r for r in g.values["segments"]}
    assert rows["big"]["judged"] is True
    tiny_only = G.g2_performance(cand, champ, GateConfig(segment_min_positives=10_000), seg)
    assert all(r["judged"] is False for r in tiny_only.values["segments"])


def test_g6_never_passes_by_itself():
    g2 = G.Gate("G2", "Performance", PASS, "", {"segments": [{"segment": "x"}]})
    assert G.g6_segments(g2).status == G.REVIEW


# ---- with a real trained model -----------------------------------------------------------------------
def test_a_candidate_identical_to_the_champion_passes_and_a_scrambled_one_fails(trained):
    _, txns, _, r = trained
    by_id = {t["txn_id"]: t for t in txns}
    rep = G.evaluate(r, r, tenant_id=T, txns_by_id=by_id, artefact_signed=True, lineage_recorded=True,
                     cfg=GateConfig(trust_sample=600))
    assert not rep.failed, [(g.id, g.detail) for g in rep.failed]
    assert rep.get("G6").status == G.REVIEW and rep.get("G7").status == G.PENDING

    rng = np.random.default_rng(1)
    scrambled = replace(r, test_pred=replace(r.test_pred, calibrated=rng.uniform(0, 1, len(r.test_y))))
    bad = G.evaluate(scrambled, r, tenant_id=T, txns_by_id=by_id, artefact_signed=True, lineage_recorded=True,
                     cfg=GateConfig(trust_sample=600))
    assert "G2" in [g.id for g in bad.failed]


def test_different_holdouts_cannot_be_compared(trained):
    _, _, _, r = trained
    other = replace(r, test_table=r.test_table.subset(np.arange(len(r.test_y)) % 2 == 0))
    with pytest.raises(ValueError, match="same holdout"):
        G.evaluate(other, r, tenant_id=T, txns_by_id={}, cfg=GateConfig())


def test_unjudgeable_trust_gates_say_not_evaluated_instead_of_passing(trained):
    _, txns, _, r = trained
    by_id = {t["txn_id"]: t for t in txns}
    g4, g5 = G.g4_g5_trust(r, r, by_id, GateConfig(trust_sample=300, min_high_cases=10**6, min_explained=10**6))
    assert g4.status == NOT_EVALUATED and g5.status == NOT_EVALUATED


def test_extra_labels_reach_only_the_training_window_and_never_the_holdout(trained):
    ds, txns, cfg, _ = trained
    verified = labels_as_of(txns, ds.matured_outcomes(cfg.as_of), cfg.as_of, cfg.horizon_days)
    by_time = {t["txn_id"]: parse_time(t["event_time"]) for t in txns}
    early = [t for t in verified if by_time[t] < cfg.train_end][:300]
    late = [t for t in verified if by_time[t] >= cfg.calibration_end][:300]
    hidden = set(early) | set(late)
    outcomes = [o for o in ds.matured_outcomes(cfg.as_of) if o["txn_id"] not in hidden]
    plain = train_bundle(txns, outcomes, cfg)
    extra = {t: verified[t] for t in hidden}
    withx = train_bundle(txns, outcomes, cfg, extra_labels=extra)
    assert withx.manifest.extra["extra_labels_used"] == len(early)  # the late ones are never used
    assert withx.test_table.txn_ids == plain.test_table.txn_ids  # the holdout is untouched
    assert np.array_equal(withx.test_y, plain.test_y)
    # a label for a case that HAS a verified outcome never overrides it
    flipped = {t: 1 - v for t, v in list(verified.items())[:50]}
    same = train_bundle(txns, ds.matured_outcomes(cfg.as_of), cfg, extra_labels=flipped)
    assert same.manifest.extra["extra_labels_used"] == 0
