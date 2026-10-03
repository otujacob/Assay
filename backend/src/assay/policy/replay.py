"""Policy replay (PRD 10.4): preview what a proposed policy would do before anyone approves it.

The proposed policy is applied to the same historical cases as the policy now in force, and the two
are compared. Comparing against the stored actions instead would be misleading, because past
decisions may have been made under several different policies.

What this is, and is not:
  * It re-evaluates each transaction's latest stored risk score and trust assessment, so it shows the
    effect of the POLICY rules only. It does not re-score anything.
  * It says how review volume would change. It does not predict how analysts would decide, how much
    fraud would be caught, or what the false-positive rate would be. Those need verified outcomes.
  * A transaction a hard rule decided keeps that result (the rule source is not replayable here).
  * It is only as representative as the history it looks at, which in this build is synthetic data.
"""

from __future__ import annotations

from collections import Counter

from assay.trust import ReasonCode, TrustState

from .engine import REVIEW_ACTIONS, Action, PolicyInput, evaluate
from .store import config_from_payload, validate_payload

MAX_EXAMPLES = 10
CAVEATS = (
    ("Shows how the policy rules change review volume on past cases. It does not predict analyst decisions, "
     "fraud caught or false positives."),
    "Cases are re-evaluated from their latest stored risk score and trust assessment; nothing is re-scored.",
    "A case decided by an institution hard rule keeps that result.",
)


def _latest_per_transaction(repo, tenant_id: str) -> list[dict]:
    latest: dict[str, dict] = {}
    for pd in repo.find(tenant_id, "policy_decisions"):  # oldest first, so a refined decision replaces its first
        latest[pd["txn_id"]] = pd
    return list(latest.values())


def preview(repo, tenant_id: str, payload: dict, baseline: dict | None, *, limit: int = 5000) -> dict:
    """`baseline` is the active policy (as returned by PolicyService.active) or None for the built-in default.
    Looks at the most recent `limit` transactions."""
    payload = validate_payload(payload)
    base_payload = baseline["payload"] if baseline else {}
    base_version = baseline["version"] if baseline else "policy-0"
    pds = _latest_per_transaction(repo, tenant_id)[-max(1, limit):]
    thresholds: dict[str, dict | None] = {}
    changed: Counter = Counter()
    bands = {"before": Counter(), "after": Counter()}
    actions = {"before": Counter(), "after": Counter()}
    examples: list[dict] = []
    times = []
    skipped = 0
    for pd in pds:
        ta = repo.get_by_id(tenant_id, "trust_assessments", pd["trust_assessment_id"])
        pr = repo.get_by_id(tenant_id, "predictions", ta["prediction_id"])
        txn = repo.get_transaction(tenant_id, pd["txn_id"])
        if pr["bundle_id"] not in thresholds:
            found = repo.find(tenant_id, "model_bundles", {"bundle_id": pr["bundle_id"]})
            thresholds[pr["bundle_id"]] = found[0]["manifest"].get("thresholds") if found else None
        thr = thresholds[pr["bundle_id"]]
        if thr is None:
            skipped += 1  # no recorded thresholds for the model that scored it: it cannot be re-evaluated
            continue
        hard = Action(pd["recommended_action"]) if pd["gate"] == "hard_rule" else None
        inp = PolicyInput(float(pr["calibrated_risk"]), TrustState(ta["state"]),
                          tuple(ReasonCode(c) for c in ta["reason_codes"]), hard, amount=float(txn["amount"]))
        old = evaluate(inp, config_from_payload(base_payload, base_version, t_low=thr["t_low"], t_high=thr["t_high"]))
        new = evaluate(inp, config_from_payload(payload, "preview", t_low=thr["t_low"], t_high=thr["t_high"]))
        actions["before"][old.action.value] += 1
        actions["after"][new.action.value] += 1
        bands["before"][old.risk_band.value] += 1
        bands["after"][new.risk_band.value] += 1
        times.append(str(txn["event_time"]))
        if old.action != new.action:
            changed[(old.action.value, new.action.value)] += 1
            if len(examples) < MAX_EXAMPLES:
                examples.append({"txn_id": pd["txn_id"], "amount": float(txn["amount"]),
                                 "risk": round(float(pr["calibrated_risk"]), 4), "was": old.action.value,
                                 "now": new.action.value, "because": new.gate})
    n = sum(actions["before"].values())
    rev = {k: sum(c for a, c in actions[k].items() if a in REVIEW_ACTIONS) for k in ("before", "after")}
    pct = None if rev["before"] == 0 else round(100 * (rev["after"] - rev["before"]) / rev["before"], 1)
    return {
        "policy": payload, "baseline_version": base_version, "n_decisions": n, "skipped": skipped,
        "window": {"from": min(times) if times else None, "to": max(times) if times else None},
        "review_volume": {"before": rev["before"], "after": rev["after"], "change": rev["after"] - rev["before"],
                          "change_pct": pct},
        "actions": {k: dict(sorted(v.items())) for k, v in actions.items()},
        "risk_bands": {k: dict(sorted(v.items())) for k, v in bands.items()},
        "changed_decisions": sum(changed.values()),
        "changes": [{"from": a, "to": b, "count": c} for (a, b), c in sorted(changed.items(), key=lambda x: -x[1])],
        "examples": examples,
        "caveats": list(CAVEATS),
    }
