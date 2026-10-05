"""A case summary for an investigator, written by fixed rules from the stored record (PRD 19, V2 "AI Fraud Copilot", first step).

NO LANGUAGE MODEL IS USED and nothing leaves the system, which is the PRD's default answer to OPD-19 (no personal data to a
third-party model). The summary restates evidence that already exists on the case screen, in a few sentences. Every sentence is
one fact with the record it came from, so a reader can check it, and a test can check that every number in it is a number in the
record. It does not weigh the evidence, guess at intent, or advise: a rule-written sentence cannot be more confident than its
source, and the wording never says more than the source does.

What it leaves out on purpose: what-ifs (they take about half a second to compute, so they stay on request), analyst notes and
prior decisions (they could anchor the reader), and anything not in the stored record.
"""

from __future__ import annotations

SUMMARY_VERSION = "summary-0"

REASON_TEXT = {
    "MISSING_CRITICAL": "a critical component is unavailable",
    "THIN_COHORT": "too few matured outcomes for this cohort",
    "DATA_QUALITY_FLOOR": "input data quality is below the floor",
    "UNFAMILIAR_PATTERN": "the pattern is unfamiliar (unlike anything seen before)",
    "WIDE_INTERVAL": "the uncertainty interval is too wide to rely on",
    "MODEL_TOO_NEW": "the model version has too few matured outcomes",
    "STALE_REFERENCE": "reference data is out of date",
    "ATCE_UNAVAILABLE": "the trust engine was unavailable",
}
COMPONENT_NAME = {"conf": "Model Confidence", "rel": "Model Reliability", "exp": "Explanation Reliability", "fam": "Familiarity",
                  "drift": "Drift Stability", "dq": "Data Quality", "hum": "Human Evidence"}
GROUP_NAME = {"amount": "Amount vs baseline", "velocity": "Transaction velocity", "timing": "Time of day", "channel": "Channel",
              "counterparty": "New or shared beneficiary", "device_location": "New device or location",
              "completeness": "Missing fields"}
ACTION_TEXT = {"approve": "approve", "approve_sampled_qa": "approve, with a sample sent for quality assurance", "block": "block",
               "escalate": "escalate", "request_human_review": "send to a person for review",
               "request_human_review_priority": "send to a person for priority review", "hold": "hold"}
GATE_TEXT = {"matrix": "the risk-and-trust matrix", "novelty": "the novelty check", "data_quality": "the data-quality check",
             "hard_rule": "an institution rule", "segment_rule": "the always-review amount rule"}
NOTE = ("Written by fixed rules from the stored record: no language model was used and nothing left this system. It restates "
        "evidence already on the case screen. It does not weigh the evidence, say why anything happened, or advise, and it is not "
        "for a customer.")
NOT_COVERED = ["what-ifs (on request: they take a moment to compute)", "analyst notes and earlier decisions on this case",
               "anything that is not in the stored record"]


def _pct(x: float) -> str:
    return f"{x * 100:.1f}%"


def _fact(text: str, source: str, ref: list[str]) -> dict:
    return {"text": text, "source": source, "ref": ref}


def summarise(decision: dict, threshold: float, explanation: dict | None, graph: dict | None) -> dict:
    """`decision`: the scoring service's decision view. `threshold`: the model's flagging threshold. `explanation`: the stored
    explanation or None. `graph`: the graph neighbourhood or None (not loaded)."""
    d, trust = decision, decision["trust"]
    facts: list[dict] = []

    flagged = d["risk"] >= threshold
    facts.append(_fact(f"The model put the risk at {_pct(d['risk'])}; its flagging threshold is {_pct(threshold)}, so it "
                       f"{'flagged' if flagged else 'did not flag'} this case.", "prediction", ["predictions.calibrated_risk", "bundle.thresholds.t_high"]))

    if trust["ti"] is None:
        why = "; ".join(REASON_TEXT.get(c, c) for c in trust["reason_codes"]) or "no reason recorded"
        facts.append(_fact(f"No Trust Index was given (insufficient evidence): {why}.", "trust_assessment",
                           ["trust_assessments.state", "trust_assessments.reason_codes"]))
    else:
        mode = ("a ranking of how far to rely on the model, not a probability" if trust["mode"] == "provisional"
                else "an estimate of how often such recommendations are right")
        facts.append(_fact(f"Trust Index {trust['ti']:.0f} out of 100 (interval {trust['ti_low']:.0f} to {trust['ti_high']:.0f}), "
                           f"{trust['state']} trust, in {trust['mode']} mode: {mode}.", "trust_assessment",
                           ["trust_assessments.ti", "trust_assessments.ti_low", "trust_assessments.ti_high", "trust_assessments.mode"]))
        if trust["reason_codes"]:
            facts.append(_fact("Reasons recorded: " + "; ".join(REASON_TEXT.get(c, c) for c in trust["reason_codes"]) + ".",
                               "trust_assessment", ["trust_assessments.reason_codes"]))

    comps = trust["components"]
    weak = sorted(((k, c["score"]) for k, c in comps.items() if c["status"] == "active" and c["score"] is not None and c["score"] < 0.5),
                  key=lambda kv: kv[1])
    if weak:
        facts.append(_fact("Reliability components scoring below 0.5: " + ", ".join(f"{COMPONENT_NAME.get(k, k)} {s:.2f}" for k, s in weak) + ".",
                           "trust_assessment", ["trust_assessments.components"]))
    else:
        facts.append(_fact("No active reliability component scored below 0.5.", "trust_assessment", ["trust_assessments.components"]))
    missing = [COMPONENT_NAME.get(k, k) for k, c in comps.items() if c["status"] == "missing"]
    if missing:
        facts.append(_fact("Not available for this case: " + ", ".join(missing) + ".", "trust_assessment", ["trust_assessments.components"]))

    if explanation:
        rows = sorted(explanation["attributions"].items(), key=lambda kv: -abs(kv[1]))[:3]
        drivers = ", ".join(f"{GROUP_NAME.get(k, k)} {v:+.2f} ({'toward fraud' if v >= 0 else 'toward legitimate'})" for k, v in rows)
        facts.append(_fact(f"Largest drivers of the score (TreeSHAP, log-odds): {drivers}.", "explanation", ["explanations.attributions"]))
        if explanation.get("stability") is not None:
            facts.append(_fact(f"Attribution stability {explanation['stability']:.2f}. A stable explanation can still explain a wrong prediction.",
                               "explanation", ["explanations.stability"]))
    else:
        facts.append(_fact("No explanation has been computed for this case yet; it runs in the background.", "explanation", []))

    action = ACTION_TEXT.get(d["recommendation"], d["recommendation"])
    if d["gate"] == "kill_switch":
        facts.append(_fact("The kill switch is engaged, so every case goes to a person whatever the model says.", "policy_decision",
                           ["policy_decisions.gate"]))
    else:
        facts.append(_fact(f"Policy {d['policy_version']} recommends: {action}, decided by {GATE_TEXT.get(d['gate'], d['gate'])}. "
                           "Assay is recommend-only: nothing happens automatically.", "policy_decision",
                           ["policy_decisions.recommended_action", "policy_decisions.gate", "policy_decisions.policy_version"]))

    if graph is None:
        facts.append(_fact("Linked entities were not loaded.", "graph", []))
    elif not graph["links"]:
        facts.append(_fact("No other customer shares this case's device, IP address or beneficiary.", "graph", ["graph.links"]))
    else:
        fraud = graph["group"]["fraud_linked_members"]
        text = (f"{graph['links_total']} other customer{'s are' if graph['links_total'] != 1 else ' is'} linked through shared "
                f"devices, IP addresses or beneficiaries, in a connected group of {graph['group']['size']}")
        text += (f"; {fraud} of them had a confirmed fraud known at the time." if fraud else "; none had a confirmed fraud known at the time.")
        facts.append(_fact(text, "graph", ["graph.links", "graph.group"]))
        if graph.get("reliability") == "weak":
            facts.append(_fact("These links are weak (confidence below 0.2), so treat them with care.", "graph", ["graph.reliability"]))
        facts.append(_fact("A link is evidence of a connection, not of wrongdoing.", "graph", ["graph.note"]))

    return {"decision_id": d["decision_id"], "version": SUMMARY_VERSION, "paragraph": " ".join(f["text"] for f in facts),
            "facts": facts, "sources": sorted({f["source"] for f in facts}), "not_covered": NOT_COVERED, "note": NOTE}
