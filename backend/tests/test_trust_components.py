from datetime import UTC, datetime, timedelta

import numpy as np
import pytest

from assay.features import feature_names
from assay.trust.confidence import model_confidence
from assay.trust.drift import drift_exposure, drift_vector, ks_statistic
from assay.trust.quality import data_quality
from assay.trust.reference import build_reference, pct_rank
from assay.trust.reliability import (
    Cohort,
    CohortStore,
    amount_band,
    risk_band_name,
    wilson_interval,
)

NAMES = feature_names()
RNG = np.random.default_rng(0)


# ---- reliability (FR-15) ----------------------------------------------------------------
def test_wilson_known_values():
    lo, hi = wilson_interval(8, 10)
    assert lo == pytest.approx(0.4902, abs=1e-3) and hi == pytest.approx(0.9433, abs=1e-3)
    assert wilson_interval(0, 0) == (0.0, 1.0)
    assert wilson_interval(10, 10)[1] == pytest.approx(1.0)
    with pytest.raises(ValueError):
        wilson_interval(5, 3)


def test_cohort_reliability_penalises_thin_cohorts():
    store = CohortStore()
    big, thin = Cohort("p", "app", "b1", "low", "m1"), Cohort("p", "web", "b1", "low", "m1")
    for _ in range(500):
        store.add(big, True)
    for _ in range(5):
        store.add(thin, True)
    rel_big, _, _, n_big = store.reliability(big)
    rel_thin, _, _, n_thin = store.reliability(thin)
    assert (n_big, n_thin) == (500, 5) and rel_big > 0.99 and rel_thin < 0.6  # 5/5 is not 100%
    assert store.reliability(Cohort("p", "x", "b9", "low", "m1"))[3] == 0  # unseen cohort: n = 0


def test_reliability_interval_ordering_and_parent_fallback():
    store = CohortStore()
    for ok in [True] * 40 + [False] * 10:
        store.add(Cohort("p", "app", "b1", "low", "m1"), ok)
    rel, lo, hi, n = store.reliability(Cohort("p", "app", "b1", "low", "m1"))
    assert lo <= rel <= hi and n == 50
    sibling = Cohort("p", "web", "b2", "low", "m1")  # never seen, same risk band and model
    assert store.reliability(sibling)[3] == 0
    assert store.reliability(sibling, fallback_to_parent_below=10)[3] == 50  # broader cohort


def test_cohorts_are_model_version_specific():
    store = CohortStore()
    store.add(Cohort("p", "app", "b1", "low", "m1"), True)
    assert store.reliability(Cohort("p", "app", "b1", "low", "m2"))[3] == 0  # history does not transfer


def test_bands():
    assert [amount_band(a) for a in (5, 25, 99, 100, 499, 500, 1999, 5000)] == \
        ["b0", "b1", "b1", "b2", "b2", "b3", "b3", "b4"]
    assert [risk_band_name(r, 0.1, 0.5) for r in (0.05, 0.1, 0.49, 0.5)] == ["low", "medium", "medium", "high"]


# ---- confidence (FR-14) -----------------------------------------------------------------
def test_confidence_bounds_determinism_and_direction():
    ref_s, ref_d = np.sort(RNG.random(1000) * 0.2), np.sort(RNG.random(1000))
    spread = np.array([0.0, 0.05, 0.2])
    dist = np.array([0.9, 0.5, 0.0])
    c1, lo, hi = model_confidence(spread, dist, ref_s, ref_d)
    c2, _, _ = model_confidence(spread, dist, ref_s, ref_d)
    assert np.array_equal(c1, c2) and np.all((0 <= lo) & (lo <= c1) & (c1 <= hi) & (hi <= 1))
    assert c1[0] > c1[1] > c1[2]  # more disagreement and less margin = less confident


# ---- data quality (FR-17) -----------------------------------------------------------------
BASE = {"txn_id": "t", "event_time": "2025-06-01T10:00:00Z", "amount": 10.0, "currency": "GBP",
        "channel": "app", "device_hash": "d", "ip_hash": "i", "country": "GB", "merchant_id": "m"}


def test_data_quality_is_the_minimum_dimension():
    assert data_quality(BASE)["q"] == 1.0
    some_null = data_quality({**BASE, "device_hash": None, "country": None})
    assert some_null["completeness"] == pytest.approx(0.4) and some_null["q"] == pytest.approx(0.4)
    assert data_quality({**BASE, "amount": -1})["q"] == 0.0  # one broken dimension sinks it
    assert data_quality({**BASE, "channel": "fax"})["consistency"] == 0.0
    rec = datetime(2025, 6, 1, 10, 30, tzinfo=UTC)
    assert data_quality(BASE, recorded_at=rec)["freshness"] == pytest.approx(0.5)
    assert data_quality(BASE, recorded_at=rec - timedelta(hours=2))["consistency"] == 0.0  # before it happened
    assert data_quality(BASE, source_health=0.2)["q"] == pytest.approx(0.2)


# ---- drift (FR-17) -------------------------------------------------------------------------
def test_ks_and_drift_vector():
    a = RNG.normal(size=3000)
    assert ks_statistic(a, a) == 0.0
    ref = RNG.normal(size=(3000, 3))
    same = RNG.normal(size=(1500, 3))
    assert drift_vector(ref, same).max() < 0.05  # sampling noise alone reads as no drift
    shifted = same.copy()
    shifted[:, 1] += 1.5
    d = drift_vector(ref, shifted)
    assert d[1] > 0.4 and d[0] < 0.05 and d[2] < 0.05


def test_drift_on_a_high_attribution_feature_hurts_more_than_on_a_low_one():
    """FR-17: injected drift on a high-attribution feature lowers drift stability more."""
    attr = np.array([[5.0, 0.1, 0.1]])  # the case is driven by feature 0
    drift_high = drift_exposure(attr, np.array([0.8, 0.0, 0.0]))
    drift_low = drift_exposure(attr, np.array([0.0, 0.0, 0.8]))
    assert drift_high[0] > 5 * drift_low[0]
    assert drift_exposure(np.zeros((1, 3)), np.ones(3))[0] == 0.0  # no attribution, no exposure


# ---- reference / novelty (FR-16) -------------------------------------------------------------
@pytest.fixture(scope="module")
def reference():
    X = RNG.normal(size=(2000, len(NAMES)))
    X[:, NAMES.index("channel_code")] = RNG.integers(0, 4, 2000)
    return build_reference(X, RNG.random(500), RNG.random(500), max_ref=1000), X


def test_pct_rank_is_monotone_and_bounded():
    ref = np.sort(RNG.random(200))
    p = pct_rank(ref, np.array([-1.0, 0.3, 0.6, 2.0]))
    assert p[0] == 0 and p[-1] == 1 and np.all(np.diff(p) >= 0)


def test_out_of_distribution_case_is_more_novel_than_typical_ones(reference):
    ref, X = reference
    typical = X[:200]
    far = typical.copy()
    far[:, :6] += 8.0  # far from everything seen
    n_typ, n_far = ref.novelty(typical), ref.novelty(far)
    assert n_far.min() > n_typ.mean() and n_far.mean() > 0.95
    assert np.all((0 <= n_typ) & (n_typ <= 1))


def test_unseen_category_is_novel_even_if_everything_else_is_typical(reference):
    ref, X = reference
    case = X[:5].copy()
    case[:, NAMES.index("channel_code")] = 4  # a channel never seen in the reference window
    assert np.all(ref.novelty(case) == 1.0)
