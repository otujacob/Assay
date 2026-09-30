import numpy as np
import pytest
import xgboost as xgb
from xgboost import XGBClassifier

from assay.features import feature_names
from assay.features.registry import group_index
from assay.trust.explain import ExplainConfig, _margin, explain_batch
from assay.trust.reference import build_reference

NAMES = feature_names()
GROUPS = group_index()


def make_model(kind: str, seed: int = 0):
    """Toy models with known behaviour, to check the explanation tests rank cases correctly."""
    rng = np.random.default_rng(seed)
    n, d = 4000, len(NAMES)
    X = rng.normal(size=(n, d))
    amt, vel, cp, dev = (GROUPS[g][0] for g in ("amount", "velocity", "counterparty", "device_location"))
    if kind == "stable":  # one dominant driver
        y = (X[:, amt] > 0.5).astype(int)
    else:  # many comparably weak drivers that trade places under small changes
        y = ((X[:, [amt, vel, cp, dev]].sum(axis=1) + rng.normal(scale=2.5, size=n)) > 0).astype(int)
    boosters = [XGBClassifier(n_estimators=40, max_depth=3, random_state=seed + i, n_jobs=1,
                              verbosity=0).fit(X, y) for i in range(3)]
    ref = build_reference(X, rng.random(300), rng.random(300), max_ref=1500)
    return boosters, ref, X[:150]


def test_tree_shap_is_additive():
    """Attributions plus the bias term must equal the model's log-odds exactly."""
    boosters, _, X = make_model("stable")
    d = xgb.DMatrix(X)
    contribs = np.mean([b.get_booster().predict(d, pred_contribs=True) for b in boosters], axis=0)
    assert np.allclose(contribs.sum(axis=1), _margin(boosters, X), atol=1e-4)


def test_subscores_are_bounded_reproducible_and_complete():
    boosters, ref, X = make_model("stable")
    ids = [f"t{i}" for i in range(len(X))]
    r = explain_batch(boosters, X, ids, ref)
    for arr in (r.stability, r.sensitivity, r.faithfulness, r.exp, r.exp_lo, r.exp_hi):
        ok = arr[~np.isnan(arr)]
        assert len(ok) > 0 and ok.min() >= 0 and ok.max() <= 1
    assert r.reproducible.all()
    assert np.all(r.exp_lo[~np.isnan(r.exp)] <= r.exp[~np.isnan(r.exp)] + 1e-12)
    assert np.all(r.exp_hi[~np.isnan(r.exp)] >= r.exp[~np.isnan(r.exp)] - 1e-12)
    assert r.attributions.shape == X.shape and r.group_attr.shape == (len(X), len(GROUPS))


def test_same_case_gives_the_same_result_alone_or_in_a_batch():
    """Per-case seeding: a case's explanation score must not depend on what else is in the batch."""
    boosters, ref, X = make_model("stable")
    ids = [f"t{i}" for i in range(len(X))]
    full = explain_batch(boosters, X, ids, ref)
    alone = explain_batch(boosters, X[7:8], ids[7:8], ref)
    again = explain_batch(boosters, X, ids, ref)
    assert np.array_equal(full.exp, again.exp, equal_nan=True)
    assert full.exp[7] == pytest.approx(alone.exp[0]) and full.stability[7] == pytest.approx(alone.stability[0])


def test_stable_model_scores_above_unstable_model():
    """FR-12: on synthetic stable and unstable cases the tests rank them correctly."""
    cfg = ExplainConfig(m=20)
    scores = {}
    for kind in ("stable", "unstable"):
        boosters, ref, X = make_model(kind)
        r = explain_batch(boosters, X, [f"t{i}" for i in range(len(X))], ref, cfg)
        scores[kind] = np.nanmean(r.stability)
    assert scores["stable"] > scores["unstable"] + 0.05, scores


def test_faithfulness_high_when_the_named_driver_really_drives_the_model():
    boosters, ref, X = make_model("stable")
    r = explain_batch(boosters, X, [f"t{i}" for i in range(len(X))], ref)
    assert np.nanmedian(r.faithfulness) > 0.6


def test_group_attribution_groups_cover_every_feature_once():
    flat = sorted(i for idx in GROUPS.values() for i in idx)
    assert flat == list(range(len(NAMES)))
