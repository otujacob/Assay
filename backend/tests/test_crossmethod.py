"""Cross-method consistency: SHAP against a local permutation importance."""

from types import SimpleNamespace

import numpy as np
import pytest

from assay.features.registry import feature_names, group_index
from assay.trust.crossmethod import (
    CrossMethodConfig,
    agreement,
    cross_method,
    local_permutation_importance,
)

NAMES = feature_names()
IDX = {n: i for i, n in enumerate(NAMES)}
GROUPS = list(group_index())


class Stub:
    """Raw score depends only on the amount and the beneficiary groups, so the right answer is known."""

    def predict(self, X):
        X = np.atleast_2d(X)
        raw = 0.001 * X[:, IDX["amount"]] + 0.5 * X[:, IDX["new_beneficiary"]]
        return SimpleNamespace(raw=raw, calibrated=raw)


def ref(n=200, seed=0):
    rng = np.random.default_rng(seed)
    X = np.zeros((n, len(NAMES)))
    X[:, IDX["amount"]] = rng.lognormal(5, 1, n)
    X[:, IDX["new_beneficiary"]] = (rng.random(n) < 0.3).astype(float)
    X[:, IDX["hour"]] = rng.integers(0, 24, n)
    X[:, IDX["velocity_1h"]] = rng.integers(0, 4, n)
    return SimpleNamespace(ref_X=X)


def case(**kw):
    x = np.zeros(len(NAMES))
    for k, v in kw.items():
        x[IDX[k]] = v
    return x[None, :]


# -- the agreement score ---------------------------------------------------------------------------------------------
def test_identical_importances_agree_completely_whatever_their_scale():
    v = np.array([[3.0, 1.0, 0.1, 0.0, 0.0, 0.0, 0.0]])
    assert agreement(v, v)[0] == pytest.approx(1.0)
    assert agreement(v, v * 1000)[0] == pytest.approx(1.0)            # scale differs between methods; direction is what counts
    assert agreement(-v, v)[0] == pytest.approx(1.0)                  # sign is irrelevant: importance is magnitude


def test_importances_on_different_groups_barely_agree():
    a = np.array([[5.0, 0, 0, 0, 0, 0, 0]])
    b = np.array([[0, 0, 0, 0, 5.0, 0, 0]])
    assert agreement(a, b)[0] < 0.05


def test_partial_overlap_scores_between_the_extremes():
    a = np.array([[5.0, 3.0, 0, 0, 0, 0, 0]])
    b = np.array([[5.0, 0, 0, 0, 3.0, 0, 0]])
    assert 0.2 < agreement(a, b)[0] < 0.8


def test_two_all_zero_vectors_count_as_agreeing_and_one_zero_one_not_does_not():
    z, v = np.zeros((1, 7)), np.array([[1.0, 0, 0, 0, 0, 0, 0]])
    assert agreement(z, z)[0] == 1.0
    assert agreement(z, v)[0] < 0.5


def test_agreement_is_always_between_zero_and_one():
    rng = np.random.default_rng(3)
    a, b = rng.normal(size=(500, 7)), rng.normal(size=(500, 7))
    s = agreement(a, b)
    assert s.shape == (500,) and s.min() >= 0.0 and s.max() <= 1.0


# -- the permutation method ------------------------------------------------------------------------------------------
def test_permutation_importance_is_largest_for_the_groups_the_model_actually_uses():
    X = case(amount=400.0, new_beneficiary=1.0, hour=12.0)
    imp = local_permutation_importance(Stub(), X, ref())[0]
    by = dict(zip(GROUPS, imp, strict=True))
    assert by["counterparty"] > 0 and by["amount"] > 0
    for unused in ("timing", "channel", "velocity", "device_location", "completeness"):
        assert by[unused] == 0.0, unused                              # the model ignores them, so replacing them changes nothing
    assert max(by, key=by.get) in ("amount", "counterparty")


def test_permutation_importance_is_deterministic_per_case_id_and_differs_between_ids():
    X = case(amount=400.0, new_beneficiary=1.0)
    a = local_permutation_importance(Stub(), X, ref(), txn_ids=["t1"])
    b = local_permutation_importance(Stub(), X, ref(), txn_ids=["t1"])
    c = local_permutation_importance(Stub(), X, ref(), txn_ids=["t2"])
    assert np.array_equal(a, b) and not np.array_equal(a, c)


def test_a_case_the_model_already_ignores_in_that_group_gets_zero_importance_for_it():
    X = case(amount=0.0, new_beneficiary=0.0)
    r = ref()
    r.ref_X[:, IDX["amount"]] = 0.0
    r.ref_X[:, IDX["new_beneficiary"]] = 0.0                          # background identical to the case: nothing can change
    assert np.all(local_permutation_importance(Stub(), X, r) == 0.0)


def test_more_background_rows_do_not_change_what_is_largest():
    X = case(amount=900.0, new_beneficiary=1.0)
    small = local_permutation_importance(Stub(), X, ref(), CrossMethodConfig(n_background=6))[0]
    large = local_permutation_importance(Stub(), X, ref(), CrossMethodConfig(n_background=40))[0]
    assert np.argmax(small) == np.argmax(large) or set(np.argsort(-small)[:2]) == set(np.argsort(-large)[:2])


# -- on the real model -----------------------------------------------------------------------------------------------
def test_on_the_real_model_both_views_and_their_agreement_are_returned_for_every_case(trained):
    r = trained[3]
    X = r.test_table.X[:6]
    out = cross_method(r.model, X, r.reference, [f"t{i}" for i in range(6)])
    assert out["groups"] == GROUPS and out["agreement"].shape == (6,)
    assert out["shap"].shape == out["permutation"].shape == (6, len(GROUPS))
    assert np.all((out["agreement"] >= 0) & (out["agreement"] <= 1))
    assert all(1 <= len(d) <= 3 for d in out["shap_drivers"] + out["permutation_drivers"])
    again = cross_method(r.model, X, r.reference, [f"t{i}" for i in range(6)])
    assert np.array_equal(out["agreement"], again["agreement"])
