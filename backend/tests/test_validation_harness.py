"""The harness on the trained synthetic pipeline: structure, invariants, and the mechanical
stress-test behaviours. Quality numbers are printed, not asserted: the harness exists so the Trust
Index can fail, and a test that required it to pass would defeat that."""

import json

import numpy as np
import pytest

from assay.ingestion import InMemoryRepository
from assay.validation.harness import (
    COMPONENTS,
    collect_cases,
    recompute_without,
    run_validation,
)
from assay.validation.report import render_markdown, store_report
from assay.validation.stress import (
    noisy_analyst_labels,
    novel_type_summary,
    stress_data_degradation,
    stress_feature_drift,
    stress_held_out_fraud_type,
)


@pytest.fixture(scope="module")
def setup(trained):
    ds, txns, cfg, r = trained
    by_id = {t["txn_id"]: t for t in txns}
    cases = collect_cases(r, by_id, 500, seed=2)
    report = run_validation(cases, bundle_id=r.manifest.bundle_id, dataset_id=r.manifest.dataset_id,
                            n_boot=120)
    return ds, txns, cfg, r, by_id, cases, report


def test_report_is_json_serialisable_and_has_every_section(setup):
    report = setup[6]
    json.dumps(report)  # no NaN objects, numpy types or non-finite numbers left
    assert report["meta"]["mode"] == "provisional" and report["meta"]["n_cases"] == 500
    for key in ("measures", "baselines", "ablations", "pass_criteria", "limitations", "overall_error_rate"):
        assert key in report
    assert set(report["baselines"]) == {"B1_distance_from_threshold", "B2_ensemble_disagreement",
                                        "B3_max_class_probability", "B4_single_components"}
    assert set(report["ablations"]) == set(COMPONENTS)
    assert "calibrated_mode_ece" in report["pass_criteria"]  # explicitly not applicable in Provisional


def test_counts_are_consistent(setup):
    cases, report = setup[5], setup[6]
    m = report["measures"]
    assert sum(m["state_counts"].values()) == len(cases.assessments) == 500
    assert report["meta"]["n_wrong"] == int(cases.wrong.sum())
    ht = m["high_trust_error_rate"]
    assert ht["n"] == m["state_counts"].get("high", 0)
    assert m["false_confidence_rate"]["n"] == report["meta"]["n_wrong"]  # conditions on error
    assert m["low_trust_detection_rate"]["n"] == report["meta"]["n_wrong"]
    if ht["value"] is not None:
        assert ht["lo"] <= ht["value"] <= ht["hi"]
    assert 0 < m["scored_share"] <= 1


def test_high_trust_error_and_false_confidence_are_the_two_different_conditionals(setup):
    cases = setup[5]
    states = np.array([a.result.state.value for a in cases.assessments])
    wrong, high = cases.wrong, states == "high"
    m = setup[6]["measures"]
    assert m["high_trust_error_rate"]["k"] == int((wrong & high).sum()) == m["false_confidence_rate"]["k"]
    assert m["high_trust_error_rate"]["n"] == int(high.sum())
    assert m["false_confidence_rate"]["n"] == int(wrong.sum())


def test_ablating_nothing_reproduces_the_original_results(setup):
    cases = setup[5]
    again = recompute_without(cases, None)
    for a, b in zip(cases.assessments, again, strict=True):
        assert a.result.state == b.state and a.result.ti == b.ti and a.result.ti_low == b.ti_low


def test_ablating_a_component_renormalises_weights_and_drops_its_gate(setup):
    cases = setup[5]
    no_rel = recompute_without(cases, "rel")
    assert all("rel" not in r.weights_used and abs(sum(r.weights_used.values()) - 1) < 1e-9 for r in no_rel)
    assert not any("THIN_COHORT" in [c.value for c in r.reason_codes] for r in no_rel)  # gate goes too
    full_thin = sum("THIN_COHORT" in [c.value for c in a.result.reason_codes] for a in cases.assessments)
    assert full_thin > 0  # the gate was active before


def test_segments_report_small_cohorts_as_inconclusive_not_point_estimates(setup):
    segs = setup[6]["measures"]["segments"]
    statuses = {v["status"] for seg in segs.values() for v in seg.values()}
    assert statuses <= {"ok", "inconclusive"}
    for seg in segs.values():
        for v in seg.values():
            if v["status"] == "inconclusive":
                assert "auroc" not in v  # no point estimate on a thin cohort
            else:
                assert v["n"] >= 30


def test_pass_criteria_are_comparative_and_not_fixed_numbers(setup):
    pc = setup[6]["pass_criteria"]
    beats = pc["trust_index_beats_B1_to_B3_with_ci_excluding_no_improvement"]
    assert set(beats) == {"B1_distance_from_threshold", "B2_ensemble_disagreement",
                          "B3_max_class_probability"} and all(isinstance(v, bool) for v in beats.values())
    assert pc["atce_supported_as_specified"] in (True, False, None)
    assert "do not relax the criteria" in pc["if_not_supported"]


def test_markdown_report_renders_key_facts(setup):
    r, report = setup[3], setup[6]
    md = render_markdown(report)
    assert r.manifest.bundle_id in md and "High-trust error rate" in md and "Ablation" in md
    assert "NOT SUPPORTED" in md or "SUPPORTED" in md or "INCONCLUSIVE" in md
    assert "Limitations" in md


def test_report_is_stored_append_only_and_linked_to_the_bundle(setup):
    _, _, _, r, _, _, report = setup
    repo = InMemoryRepository()
    row = store_report(repo, "tenant-synth", r.manifest, report, stress={"x": {"test": "t", "expected": "e",
                                                                              "passed": None}})
    assert row["bundle_id"] == r.manifest.bundle_id and row["dataset_id"] == r.manifest.dataset_id
    assert repo.find("tenant-synth", "model_bundles", {"bundle_id": r.manifest.bundle_id})
    assert repo.verify("tenant-synth", "validation_reports") is None


# ---- stress tests (PRD 6.5) ----------------------------------------------------------------------
def test_stress_feature_drift(setup):
    _, _, _, r, by_id, _, _ = setup
    s = stress_feature_drift(r, by_id, n=120)
    assert s["baseline"]["alarm_fired"] is False and s["drifted"]["alarm_fired"] is True
    assert s["drifted"]["mean_drift_stability"] < s["baseline"]["mean_drift_stability"]
    assert s["drifted"]["mean_ti"] < s["baseline"]["mean_ti"] and s["passed"] is True
    assert "amount" in s["drifted"]["drifted_features"]


def test_stress_data_degradation(setup):
    _, _, _, r, by_id, _, _ = setup
    s = stress_data_degradation(r, by_id, n=120)
    dq = [row["mean_dq"] for row in s["rows"]]
    floor = [row["floor_share"] for row in s["rows"]]
    assert dq == sorted(dq, reverse=True) and floor == sorted(floor)
    assert floor[0] < 0.05 and floor[-1] > 0.95 and s["passed"] is True


def test_stress_held_out_fraud_type(setup, capsys):
    ds, txns, cfg, _, _, _, _ = setup
    s = stress_held_out_fraud_type(ds, txns, cfg, n_other=150)
    h, k, o = s["held_out_cases"], s["known_fraud_cases"], s["ordinary_cases"]
    assert h["n"] >= 10
    with capsys.disabled():
        print(f"\nheld-out mule_ring: familiarity {h['mean_familiarity']:.2f} vs known fraud "
              f"{k['mean_familiarity']:.2f} vs ordinary {o['mean_familiarity']:.2f}; "
              f"Low/Insufficient {h['share_low_or_insufficient']:.2f} vs {o['share_low_or_insufficient']:.2f}")
    # The mechanical claim: never-seen fraud is less familiar than both known fraud and normal traffic.
    assert h["mean_familiarity"] < k["mean_familiarity"] and h["mean_familiarity"] < o["mean_familiarity"]
    assert s["passed"] is True


def test_novel_type_summary_and_untestable_noisy_labels(setup):
    ds, _, _, _, _, cases, _ = setup
    n = novel_type_summary(cases, ds.truth)
    assert n["n_novel"] >= 0 and "share_low_or_insufficient_ordinary" in n
    nl = noisy_analyst_labels()
    assert nl["status"] == "not_testable" and nl["passed"] is None  # honest about what is not built
