"""The rule-written case summary: every sentence is a sourced fact, no number is invented, nothing overclaims, nothing leaks."""

import copy
import re

import pytest

from assay.copilot.summary import NOTE, REASON_TEXT, summarise
from test_api_graph import RING, api, artefact  # noqa: F401  (fixtures)

DEC = {"decision_id": "d1", "txn_id": "t1", "risk": 0.118, "risk_band": "high", "recommendation": "request_human_review_priority",
       "gate": "matrix", "policy_version": "policy-0",
       "trust": {"state": "moderate", "mode": "provisional", "ti": 61.4, "ti_low": 48.2, "ti_high": 70.9, "reason_codes": ["WIDE_INTERVAL"],
                 "components": {"conf": {"status": "active", "score": 0.31}, "rel": {"status": "active", "score": 0.91},
                                "exp": {"status": "missing", "score": None}, "fam": {"status": "active", "score": 0.07},
                                "drift": {"status": "active", "score": 1.0}, "dq": {"status": "active", "score": 1.0},
                                "hum": {"status": "inactive", "score": None}}}}
EXP = {"attributions": {"amount": 1.2, "counterparty": -0.4, "timing": 0.1, "channel": -0.05}, "stability": 0.93}
GRAPH = {"links": [{"x": 1}], "links_total": 3, "group": {"size": 4, "links": 3, "fraud_linked_members": 1}, "reliability": "ok", "note": "n"}


def numbers(text):
    return set(re.findall(r"[+-]?\d+(?:\.\d+)?", text.replace("policy-0", "policy")))  # a policy name is not a figure


def test_each_sentence_states_its_source_and_the_paragraph_is_just_the_sentences():
    s = summarise(DEC, 0.118, EXP, GRAPH)
    assert s["paragraph"] == " ".join(f["text"] for f in s["facts"])
    assert all(f["source"] in {"prediction", "trust_assessment", "explanation", "policy_decision", "graph"} for f in s["facts"])
    assert s["sources"] == ["explanation", "graph", "policy_decision", "prediction", "trust_assessment"]
    assert s["note"] == NOTE and "no language model was used" in s["note"]
    assert any("what-ifs" in n for n in s["not_covered"])


def test_every_number_in_the_summary_is_a_number_in_the_record():
    s = summarise(DEC, 0.118, EXP, GRAPH)
    allowed = {"11.8", "61", "48", "71", "100", "0.5", "0.31", "0.07", "1.20", "+1.20", "-0.40", "+0.10", "0.93", "3", "4", "1", "0.2"}
    allowed |= {"0", "-0.40"}
    stray = numbers(s["paragraph"]) - allowed
    assert not stray, stray


def test_the_risk_sentence_says_flagged_or_not_by_the_threshold_alone():
    assert "did not flag" in summarise(DEC, 0.5, EXP, None)["facts"][0]["text"]
    assert "so it flagged" in summarise(DEC, 0.118, EXP, None)["facts"][0]["text"]


def test_provisional_trust_is_not_described_as_a_probability():
    t = summarise(DEC, 0.1, EXP, None)["paragraph"]
    assert "not a probability" in t and "probability of" not in t.replace("not a probability", "")
    cal = copy.deepcopy(DEC)
    cal["trust"]["mode"] = "calibrated"
    assert "how often such recommendations are right" in summarise(cal, 0.1, EXP, None)["paragraph"]


def test_insufficient_evidence_gives_no_number_and_the_reasons():
    d = copy.deepcopy(DEC)
    d["trust"].update(state="insufficient_evidence", ti=None, ti_low=None, ti_high=None, reason_codes=["UNFAMILIAR_PATTERN", "THIN_COHORT"])
    t = summarise(d, 0.1, EXP, None)["facts"][1]["text"]
    assert t.startswith("No Trust Index was given (insufficient evidence)")
    assert REASON_TEXT["UNFAMILIAR_PATTERN"] in t and REASON_TEXT["THIN_COHORT"] in t and "/100" not in t and "out of 100" not in t


def test_weak_and_missing_components_are_named_and_a_clean_case_says_none_are_weak():
    t = summarise(DEC, 0.1, EXP, None)["paragraph"]
    assert "Familiarity 0.07, Model Confidence 0.31" in t           # weakest first
    assert "Not available for this case: Explanation Reliability." in t
    assert "Human Evidence" not in t                                   # inactive by design, not a gap
    clean = copy.deepcopy(DEC)
    for c in clean["trust"]["components"].values():
        if c["status"] == "active":
            c["score"] = 0.9
    assert "No active reliability component scored below 0.5." in summarise(clean, 0.1, EXP, None)["paragraph"]


def test_drivers_are_the_three_largest_with_direction_and_the_stability_caveat():
    t = summarise(DEC, 0.1, EXP, None)["paragraph"]
    assert "Amount vs baseline +1.20 (toward fraud), New or shared beneficiary -0.40 (toward legitimate), Time of day +0.10 (toward fraud)" in t
    assert "Channel" not in t and "can still explain a wrong prediction" in t
    assert "No explanation has been computed" in summarise(DEC, 0.1, None, None)["paragraph"]


def test_the_recommendation_is_reported_as_recommend_only_and_the_kill_switch_overrides_the_wording():
    t = summarise(DEC, 0.1, EXP, None)["paragraph"]
    assert "Policy policy-0 recommends: send to a person for priority review, decided by the risk-and-trust matrix." in t
    assert "nothing happens automatically" in t
    k = copy.deepcopy(DEC)
    k.update(gate="kill_switch", recommendation="request_human_review")
    assert "The kill switch is engaged" in summarise(k, 0.1, EXP, None)["paragraph"] and "recommends:" not in summarise(k, 0.1, EXP, None)["paragraph"]


def test_graph_sentences_never_claim_more_than_a_connection():
    t = summarise(DEC, 0.1, EXP, GRAPH)["paragraph"]
    assert "3 other customers are linked" in t and "connected group of 4" in t and "1 of them had a confirmed fraud known at the time" in t
    assert "evidence of a connection, not of wrongdoing" in t
    weak = {**GRAPH, "reliability": "weak"}
    assert "treat them with care" in summarise(DEC, 0.1, EXP, weak)["paragraph"]
    assert "No other customer shares" in summarise(DEC, 0.1, EXP, {**GRAPH, "links": []})["paragraph"]
    assert "Linked entities were not loaded" in summarise(DEC, 0.1, EXP, None)["paragraph"]
    one = {**GRAPH, "links_total": 1, "group": {"size": 2, "links": 1, "fraud_linked_members": 0}}
    t1 = summarise(DEC, 0.1, EXP, one)["paragraph"]
    assert "1 other customer is linked" in t1 and "none had a confirmed fraud" in t1


def test_the_wording_contains_no_advice_causes_or_guesses():
    t = summarise(DEC, 0.1, EXP, GRAPH)["paragraph"].lower()
    for banned in ("should", "because", "likely fraud", "suspicious", "guilty", "you must", "caused", "intent", "probably"):
        assert banned not in t, banned


# ---- through the API -------------------------------------------------------------------------------------------------
def test_the_api_summary_is_built_from_the_stored_record_and_reading_it_is_audited(api):  # noqa: F811
    api["build_ring"]()
    _, d = api["submit"]("c-new", 10, ben="b-ring")
    r = api["call"]("GET", f"/v1/decisions/{d['decision_id']}/summary", "ana")
    assert r.status_code == 200
    s = r.json()
    assert s["decision_id"] == d["decision_id"] and s["version"] == "summary-0"
    assert f"{d['risk'] * 100:.1f}%" in s["facts"][0]["text"]
    assert any(f["source"] == "graph" and f"{len(RING)} other customers are linked" in f["text"] for f in s["facts"])
    audit = [a for a in api["repo"].find("tenant-synth", "audit_log") if a["action"] == "summary_read"]
    assert len(audit) == 1 and audit[0]["object"] == d["txn_id"]


def test_the_api_enforces_roles_tenant_and_unknown_ids(api):  # noqa: F811
    _, d = api["submit"]("c-x", 3)
    path = f"/v1/decisions/{d['decision_id']}/summary"
    for key, code in (("ana", 200), ("aud", 200), ("sen", 403), ("mgr", 403), ("ing", 403), ("oth", 404)):
        assert api["call"]("GET", path, key).status_code == code, key
    assert api["call"]("GET", "/v1/decisions/nope/summary", "ana").status_code == 404


@pytest.mark.parametrize("api", [1.0], indirect=True)
def test_a_blind_case_gets_no_summary(api):  # noqa: F811
    _, d = api["submit"]("c-blind", 3)
    r = api["call"]("GET", f"/v1/decisions/{d['decision_id']}/summary", "ana")
    assert r.status_code == 403 and r.json()["detail"]["code"] == "blind_review"
    assert api["call"]("GET", f"/v1/decisions/{d['decision_id']}/summary", "aud").status_code == 200
