"""Counterfactuals (PRD 7.1). The logic is tested against a stub model whose behaviour is fully known, so
every rule can be checked by hand; the real model is used at the end."""

import math
from types import SimpleNamespace

import numpy as np
import pytest

from assay.features.registry import feature_names
from assay.trust.counterfactual import (
    IDX,
    Counterfactual,
    CounterfactualConfig,
    _distance,
    _set_amount,
    candidates,
    describe_change,
    generate,
    summarise,
    validate_counterfactual,
)

NAMES = feature_names()
T_HIGH = 0.5


class _Booster:
    def __init__(self, fn):
        self.fn = fn

    def predict_proba(self, X):
        p = np.array([self.fn(r) for r in X])
        return np.column_stack([1 - p, p])


class Stub:
    """A model with a known rule, so the expected counterfactuals can be worked out by hand."""

    def __init__(self, fn):
        self.fn, self.boosters, self.calls = fn, [_Booster(fn)], 0

    def predict(self, X):
        self.calls += 1
        return SimpleNamespace(calibrated=np.array([self.fn(r) for r in np.atleast_2d(X)]))


def vec(**kw):
    base = {"amount": 100.0, "log_amount": math.log1p(100.0), "hour": 14.0, "channel_code": 1.0, "velocity_1h": 0.0,
            "velocity_24h": 1.0, "velocity_30d": 8.0, "amount_ratio_baseline": 1.5, "new_beneficiary": 0.0,
            "new_device": 0.0, "country_mismatch": 0.0, "beneficiary_shared_customers": 0.0, "missing_fields": 0.0}
    base.update(kw)
    if "amount" in kw and "log_amount" not in kw:
        base["log_amount"] = math.log1p(base["amount"])
    return np.array([base[n] for n in NAMES], dtype=float)


def make_ref(seed=0, n=600):
    rng = np.random.default_rng(seed)
    X = np.array([vec(amount=float(a), hour=float(rng.integers(0, 24)), channel_code=float(rng.integers(0, 5)),
                      velocity_1h=float(rng.integers(0, 4)), new_beneficiary=float(rng.random() < 0.2),
                      new_device=float(rng.random() < 0.1), country_mismatch=float(rng.random() < 0.1),
                      beneficiary_shared_customers=float(rng.integers(0, 4)))
                  for a in rng.lognormal(4.5, 1.0, n)])
    return SimpleNamespace(ref_X=X, std=X.std(axis=0) + 1e-9, categorical_seen={"channel_code": {0.0, 1.0, 2.0, 3.0, 4.0}})


REF = make_ref()


def risk_new_beneficiary(r):
    """Flagged exactly when the beneficiary is new: nothing else matters."""
    return 0.9 if r[IDX["new_beneficiary"]] > 0.5 else 0.05


def risk_small_amount_and_night(r):
    """Flagged when the amount is small AND it is the small hours: it takes both to be flagged."""
    return 0.9 if (r[IDX["amount"]] < 50 and r[IDX["hour"]] < 5) else 0.05


def risk_velocity(r):
    return 0.9 if r[IDX["velocity_1h"]] >= 3 else 0.05


def risk_amount_cliff(r):
    """A sharp cliff at amount = 100: a flip just past it is a knife edge."""
    return 0.9 if r[IDX["amount"]] < 100 else 0.05


# -- finding counterfactuals ---------------------------------------------------------------------------------------
def test_the_single_change_that_flips_the_call_is_found_and_nothing_else_is_added():
    x = vec(new_beneficiary=1.0)
    cfs = generate(Stub(risk_new_beneficiary), REF, x, T_HIGH)
    assert cfs and cfs[0].valid
    assert cfs[0].groups == ("counterparty",)
    assert [c["feature"] for c in cfs[0].changes] == ["new_beneficiary"] or {c["feature"] for c in cfs[0].changes} <= {
        "new_beneficiary", "beneficiary_shared_customers"}
    assert cfs[0].risk_before == 0.9 and cfs[0].risk_after == 0.05


def test_counterfactuals_are_minimal_no_group_is_changed_that_does_not_matter():
    x = vec(new_beneficiary=1.0)
    for c in generate(Stub(risk_new_beneficiary), REF, x, T_HIGH, CounterfactualConfig(n_return=8)):
        assert c.groups == ("counterparty",), c.groups      # no pair that merely adds an irrelevant hour or amount


def test_when_two_changes_are_both_needed_both_are_reported():
    x = vec(amount=20.0, hour=2.0)                           # small amount AND small hours: flagged
    cfs = generate(Stub(risk_small_amount_and_night), REF, x, T_HIGH)
    assert cfs and {c.groups for c in cfs} == {("amount",), ("timing",)}   # undoing either one is enough here


def test_immutable_history_is_never_changed_even_if_it_is_what_drives_the_call():
    x = vec(velocity_1h=3.0)
    assert generate(Stub(risk_velocity), REF, x, T_HIGH) == []              # nothing realistic to change: that is information
    open_cfg = CounterfactualConfig(immutable_groups=())
    cfs = generate(Stub(risk_velocity), REF, x, T_HIGH, open_cfg)
    assert cfs and cfs[0].groups == ("velocity",)                           # only because we allowed it


def test_a_case_that_is_not_flagged_gets_counterfactuals_that_would_flag_it():
    x = vec(new_beneficiary=0.0)
    cfs = generate(Stub(risk_new_beneficiary), REF, x, T_HIGH)
    assert cfs and cfs[0].risk_before == 0.05 and cfs[0].risk_after == 0.9


def test_the_flip_must_clear_the_threshold_by_a_margin():
    def sliver(r):                                           # flagged at 0.52, "unflagged" only at 0.48: inside a 10% margin
        return 0.52 if r[IDX["new_beneficiary"]] > 0.5 else 0.48

    assert generate(Stub(sliver), REF, vec(new_beneficiary=1.0), T_HIGH) == []
    assert generate(Stub(sliver), REF, vec(new_beneficiary=1.0), T_HIGH, CounterfactualConfig(flip_margin=0.0))


def test_it_is_deterministic():
    x = vec(amount=20.0, hour=2.0)
    a = generate(Stub(risk_small_amount_and_night), REF, x, T_HIGH)
    b = generate(Stub(risk_small_amount_and_night), REF, x, T_HIGH)
    assert [(c.groups, c.distance, c.checks) for c in a] == [(c.groups, c.distance, c.checks) for c in b]


def test_pairs_are_shortlisted_by_the_boosted_trees_but_every_result_is_scored_by_the_full_model():
    stub = Stub(risk_small_amount_and_night)
    cfs = generate(stub, REF, vec(amount=20.0, hour=2.0), T_HIGH)
    for c in cfs:                                            # whatever was reported, the full model agrees with it
        assert stub.fn(c.x) == c.risk_after


def test_candidates_keep_the_amount_group_coherent_and_skip_immutable_groups():
    x = vec(amount=100.0, amount_ratio_baseline=2.0)
    cands = candidates(x, REF, CounterfactualConfig())
    assert "velocity" not in cands and "completeness" not in cands
    for c in cands["amount"]:
        assert abs(c[IDX["log_amount"]] - math.log1p(c[IDX["amount"]])) < 1e-9
        assert abs(c[IDX["amount_ratio_baseline"]] - 2.0 * c[IDX["amount"]] / 100.0) < 1e-9   # scales with the amount
    assert all(c[IDX["hour"]] != 14.0 for c in cands["timing"]) and len(cands["timing"]) == 23
    assert all(c[IDX["channel_code"]] != 1.0 for c in cands["channel"])
    # the 'fewer than 3 prior transactions' placeholder ratio stays 1.0 rather than becoming a made-up baseline
    assert all(c[IDX["amount_ratio_baseline"]] == 1.0 for c in candidates(vec(amount_ratio_baseline=1.0), REF, CounterfactualConfig())["amount"])


# -- the independent validator ---------------------------------------------------------------------------------------
def checks(x, cf, fn=risk_new_beneficiary, **cfg):
    return validate_counterfactual(Stub(fn), REF, x, cf, T_HIGH, CounterfactualConfig(**cfg))


def test_a_good_counterfactual_passes_every_check():
    x = vec(new_beneficiary=1.0)
    c = checks(x, vec(new_beneficiary=0.0))
    assert all(c[k] for k in ("flips", "within_bounds", "immutable_unchanged", "coherent", "few_changes", "small_change", "robust"))


@pytest.mark.parametrize("bad,failing", [
    ({"new_beneficiary": 0.0, "velocity_1h": 2.0}, "immutable_unchanged"),   # changed the customer's history
    ({"new_beneficiary": 0.0, "missing_fields": 2.0}, "immutable_unchanged"),
    ({"new_beneficiary": 1.0, "hour": 9.0}, "flips"),                        # does not flip the call
    ({"new_beneficiary": 0.0, "channel_code": 99.0}, "within_bounds"),       # a channel that does not exist
    ({"new_beneficiary": 0.0, "new_device": 0.5}, "within_bounds"),          # a half of a yes/no
    ({"new_beneficiary": 0.0, "hour": 25.0}, "coherent"),
    ({"new_beneficiary": 0.0, "hour": 3.5}, "coherent"),
])
def test_the_validator_rejects_what_it_should(bad, failing):
    c = checks(vec(new_beneficiary=1.0), vec(**bad))
    assert c[failing] is False, c


def test_an_incoherent_amount_group_is_rejected():
    cf = vec(new_beneficiary=0.0, amount=500.0)
    cf[IDX["log_amount"]] = 1.0                                              # the log no longer matches the amount
    assert checks(vec(new_beneficiary=1.0), cf)["coherent"] is False


def test_a_negative_or_infinite_amount_is_out_of_bounds():
    for amount in (-5.0, float("inf")):
        cf = vec(new_beneficiary=0.0)
        cf[IDX["amount"]] = amount
        assert checks(vec(new_beneficiary=1.0), cf)["within_bounds"] is False


def test_too_many_changed_groups_or_too_far_a_move_is_rejected():
    x = vec(new_beneficiary=1.0)
    three = vec(new_beneficiary=0.0, hour=9.0, channel_code=3.0)
    assert checks(x, three)["few_changes"] is False                          # three groups changed, two allowed
    assert checks(x, three, max_changes=3)["few_changes"] is True
    assert checks(x, vec(new_beneficiary=0.0, amount=5e5), max_distance=1.0)["small_change"] is False


def test_a_flip_on_a_knife_edge_is_not_robust():
    x = vec(amount=50.0)                                                     # flagged (below the cliff at 100)
    edge = vec(amount=100.5)                                                 # flips, but a 5% wobble puts it back
    solid = vec(amount=400.0)
    assert checks(x, edge, fn=risk_amount_cliff)["flips"] is True
    assert checks(x, edge, fn=risk_amount_cliff)["robust"] is False
    c = checks(x, solid, fn=risk_amount_cliff)
    assert c["robust"] is True and c["robust_share"] == 1.0


def test_the_validator_rescored_with_the_model_and_did_not_trust_the_caller():
    stub = Stub(risk_new_beneficiary)
    validate_counterfactual(stub, REF, vec(new_beneficiary=1.0), vec(new_beneficiary=0.0), T_HIGH)
    assert stub.calls == 1                                                   # one batched call re-scoring everything


# -- distance, text, summary ---------------------------------------------------------------------------------------
def test_hour_distance_wraps_around_midnight():
    a, b = vec(hour=23.0), vec(hour=1.0)
    assert _distance(a, b, REF, ("timing",)) == pytest.approx(2 / 6)         # two hours apart, not twenty-two
    assert _distance(a, vec(hour=11.0), REF, ("timing",)) == pytest.approx(12 / 6)


def test_a_category_change_costs_one_and_history_costs_nothing():
    assert _distance(vec(channel_code=1.0), vec(channel_code=3.0), REF, ("channel",)) == 1.0
    assert _distance(vec(), vec(velocity_1h=3.0), REF, ("velocity",)) == 0.0


def test_changes_are_described_in_words():
    assert describe_change("amount", 1250.0, 181.0) == "amount 1,250 → 181"      # whole units from 100 up
    assert describe_change("amount", 1.02, 17.75) == "amount 1.02 → 17.75"
    assert describe_change("hour", 3.0, 14.0) == "hour of day 03:00 → 14:00"
    assert describe_change("channel_code", 1.0, 3.0) == "channel card_not_present → web"
    assert describe_change("new_beneficiary", 1.0, 0.0) == "a new beneficiary: yes → no"
    assert describe_change("channel_code", 1.0, 9.0).endswith("other")


def test_the_summary_is_none_when_nothing_was_found_not_zero():
    assert summarise([]) == {"found": 0, "score": None}
    good = Counterfactual(vec(), ("timing",), [], 0.9, 0.1, 1.0, {"flips": True, "robust": True, "robust_share": 1.0})
    bad = Counterfactual(vec(), ("timing",), [], 0.9, 0.1, 1.0, {"flips": False, "robust": True, "robust_share": 0.5})
    assert summarise([good])["score"] == 1.0
    assert summarise([good, bad])["score"] == pytest.approx(0.5 * 0.75)


def test_set_amount_scales_the_ratio_but_not_the_placeholder():
    assert _set_amount(vec(amount=100.0, amount_ratio_baseline=2.0), 200.0)[IDX["amount_ratio_baseline"]] == 4.0
    assert _set_amount(vec(amount=100.0, amount_ratio_baseline=1.0), 200.0)[IDX["amount_ratio_baseline"]] == 1.0


# -- the real model --------------------------------------------------------------------------------------------------
def test_on_the_real_model_every_valid_counterfactual_really_flips_the_call_and_respects_the_rules(trained):
    r = trained[3]
    t_high = r.manifest.thresholds["t_high"]
    X = r.test_table.X
    risk = r.model.predict(X).calibrated
    flagged = np.flatnonzero(risk >= t_high)[:25:5]
    assert len(flagged) >= 3
    found = 0
    for i in flagged:
        for c in generate(r.model, r.reference, X[i], t_high):
            found += 1
            assert c.valid == all(v is True for k, v in c.checks.items() if k != "robust_share")
            if not c.valid:
                continue
            assert r.model.predict(c.x[None, :]).calibrated[0] < t_high          # re-scored here, not trusted
            for f in ("velocity_1h", "velocity_24h", "velocity_30d", "missing_fields"):
                assert c.x[IDX[f]] == X[i][IDX[f]]                                  # history and completeness untouched
    assert found >= 3


def test_on_the_real_model_an_unflagged_case_can_be_pushed_over_and_the_search_is_deterministic(trained):
    r = trained[3]
    t_high = r.manifest.thresholds["t_high"]
    X = r.test_table.X
    risk = r.model.predict(X[:300]).calibrated
    i = int(np.argmin(np.abs(risk - t_high * 0.6)))                              # a case near, but under, the threshold
    a = generate(r.model, r.reference, X[i], t_high)
    b = generate(r.model, r.reference, X[i], t_high)
    assert [(c.groups, round(c.distance, 9)) for c in a] == [(c.groups, round(c.distance, 9)) for c in b]
    for c in a:
        if c.valid:
            assert c.risk_before < t_high <= c.risk_after
