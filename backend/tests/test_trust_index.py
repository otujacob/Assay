import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from assay.trust import (
    CRITICAL,
    DEFAULT_WEIGHTS,
    Component,
    ReasonCode,
    TrustConfig,
    TrustContext,
    TrustState,
    compute_trust_index,
    wilson_interval,
)

SCORES = {"conf": 0.80, "rel": 0.70, "exp": 0.60, "fam": 0.90, "drift": 0.85, "dq": 0.95}


def example(hum=None, **over):
    s = {**SCORES, **over}
    comps = {k: Component.active(v, n=100) for k, v in s.items()}
    comps["hum"] = hum or Component.missing()
    return comps


def test_worked_example_5_4_point_and_high():
    r = compute_trust_index(example())
    assert r.ti == pytest.approx(76, abs=1)
    assert r.ti_high == pytest.approx(79, abs=1)  # missing hum has hi = 1
    assert r.ti_low <= r.ti <= r.ti_high


def test_inactive_hum_renormalises_and_differs_from_missing():
    inactive = compute_trust_index(example(hum=Component.inactive()))
    missing = compute_trust_index(example())
    assert sum(inactive.weights_used.values()) == pytest.approx(1.0)
    assert "hum" not in inactive.weights_used
    assert inactive.ti > missing.ti  # missing is penalised, inactive is not


def test_ceilings_from_5_4():
    all_one = {k: Component.active(1.0, lo=1.0, hi=1.0, n=100) for k in DEFAULT_WEIGHTS}
    no_exp = compute_trust_index({**all_one, "exp": Component.missing()})
    no_hum = compute_trust_index({**all_one, "hum": Component.missing()})
    assert no_exp.ti_low == pytest.approx(50, abs=1)
    assert no_hum.ti_low == pytest.approx(79, abs=1)
    assert no_exp.state is TrustState.LOW or no_exp.state is TrustState.MODERATE  # never HIGH


@pytest.mark.parametrize("k", sorted(CRITICAL - {"rel"}))
def test_missing_critical_gives_null(k):
    comps = example()
    comps[k] = Component.missing()
    r = compute_trust_index(comps)
    assert r.state is TrustState.INSUFFICIENT and r.ti is None
    assert ReasonCode.MISSING_CRITICAL in r.reason_codes


def test_each_reason_code_triggers():
    cfg = TrustConfig(min_model_outcomes=10)
    cases = {
        ReasonCode.THIN_COHORT: (example(rel=0.7) | {"rel": Component.active(0.7, n=5)}, None),
        ReasonCode.DATA_QUALITY_FLOOR: (example(dq=0.2), None),
        ReasonCode.UNFAMILIAR_PATTERN: (example(fam=0.02), None),
        ReasonCode.MODEL_TOO_NEW: (example(), TrustContext(model_matured_outcomes=1)),
        ReasonCode.STALE_REFERENCE: (example(), TrustContext(reference_age_days=999)),
    }
    for code, (comps, ctx) in cases.items():
        r = compute_trust_index(comps, cfg, ctx)
        assert r.ti is None and code in r.reason_codes, code


def test_wide_interval():
    comps = example()
    comps["conf"] = Component.active(0.8, lo=0.05, hi=1.0, n=100)
    comps["rel"] = Component.active(0.7, lo=0.05, hi=1.0, n=100)
    comps["fam"] = Component.active(0.9, lo=0.05, hi=1.0, n=100)
    r = compute_trust_index(comps, TrustConfig(max_interval_width=20))
    assert ReasonCode.WIDE_INTERVAL in r.reason_codes and r.ti is None


def test_config_version_recorded():
    r = compute_trust_index(example(), TrustConfig(version="v42"))
    assert r.config_version == "v42"


def test_wilson_known_values():
    lo, hi = wilson_interval(8, 10)
    assert lo == pytest.approx(0.4902, abs=1e-3) and hi == pytest.approx(0.9433, abs=1e-3)
    assert wilson_interval(0, 0) == (0.0, 1.0)
    assert wilson_interval(10, 10)[1] == pytest.approx(1.0)


# ---- property tests (PRD 28.2) ----
unit = st.floats(0.0, 1.0, allow_nan=False)


@st.composite
def component_sets(draw):
    comps = {}
    for k in DEFAULT_WEIGHTS:
        s = draw(unit)
        w = draw(st.floats(0.0, 0.3, allow_nan=False))
        comps[k] = Component.active(s, lo=max(0, s - w), hi=min(1, s + w), n=100)
    return comps


@given(component_sets(), st.sampled_from(sorted(DEFAULT_WEIGHTS)), unit)
@settings(max_examples=200)
def test_lowering_a_component_never_raises_ti(comps, key, new):
    cfg = TrustConfig(dq_floor=0.0, novelty_ceiling=2.0, max_interval_width=1000)
    base = compute_trust_index(comps, cfg)
    c = comps[key]
    new = min(new, c.score)
    lowered = {**comps, key: Component.active(new, lo=min(c.lo, new), hi=min(c.hi, new), n=100)}
    assert compute_trust_index(lowered, cfg).ti <= base.ti + 1e-9


@given(component_sets(), st.sampled_from(sorted(set(DEFAULT_WEIGHTS) - CRITICAL)))
@settings(max_examples=200)
def test_missing_never_raises_ti_low(comps, key):
    cfg = TrustConfig(dq_floor=0.0, novelty_ceiling=2.0, max_interval_width=1000)
    base = compute_trust_index(comps, cfg)
    miss = compute_trust_index({**comps, key: Component.missing()}, cfg)
    # Holds when the original lower bound is at or above the epsilon floor (0.01);
    # below the floor, a missing component's lo of 0 is clamped to the same value.
    if comps[key].lo >= 0.01:
        assert miss.ti_low <= base.ti_low + 1e-9


@given(component_sets())
def test_ordering_determinism_and_weights(comps):
    cfg = TrustConfig(dq_floor=0.0, novelty_ceiling=2.0, max_interval_width=1000)
    a, b = compute_trust_index(comps, cfg), compute_trust_index(comps, cfg)
    assert a == b
    assert a.ti_low <= a.ti + 1e-9 and a.ti <= a.ti_high + 1e-9
    assert sum(a.weights_used.values()) == pytest.approx(1.0)
