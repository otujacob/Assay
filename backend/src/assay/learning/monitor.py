"""Degradation monitor for a model that is deciding live cases (PRD 12.4, 12.5, H7).

A candidate in canary, or a promoted champion, is judged on the cases it actually decided once their outcomes mature.
Rollback triggers (PRD 12.4), each needing clear statistical evidence so one bad week of a few cases cannot undo a good model:

  calibration   the average gap between predicted risk and what happened (ECE) is above the limit;
  high trust    cases it called High trust were wrong more often than the limit, with the whole 95% interval above it;
  recall        the share of verified frauds it flagged is clearly below what it achieved on its validation holdout;
  precision     the share of its flags that were fraud is clearly below its validation holdout.

"Clearly" is the Wilson 95% interval lying entirely beyond the baseline minus a tolerance. With too few matured cases there is
no verdict, and nothing happens: the monitor never rolls back on thin evidence, and never promotes anything.

A rollback restores the previous champion, which is the known-good direction, so the monitor may take it without a person
(actor `system:monitor`, with the evidence stored as the reason). Promotion, approval and canary start stay people's acts.
Limits and tolerances are working defaults (OPD-12); no baselines have been measured on real traffic. The data-quality and
operational-incident triggers of PRD 12.4 are not monitored here: drift alarms are reported by the drift job, and an operational
incident is a person's call.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from assay.detection.metrics import expected_calibration_error
from assay.trust.reliability import wilson_interval

FRAUD = ("confirmed_fraud", "chargeback")


@dataclass(frozen=True)
class MonitorConfig:
    min_matured: int = 150  # matured live cases before any verdict
    ece_max: float = 0.05
    min_high_cases: int = 50
    high_error_limit: float = 0.05
    min_flagged: int = 30
    min_frauds: int = 30
    tolerance: float = 0.10  # recall and precision may fall this far below the validation holdout before it counts


@dataclass
class MonitorReport:
    bundle_id: str
    status: str  # ok | breach | insufficient
    n_matured: int
    triggers: list[str] = field(default_factory=list)
    measures: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {"bundle_id": self.bundle_id, "status": self.status, "n_matured": self.n_matured,
                "triggers": self.triggers, "measures": self.measures}


def evaluate(bundle_id: str, risk: np.ndarray, fraud: np.ndarray, high_trust: np.ndarray, t_high: float,
             baseline: dict | None, cfg: MonitorConfig | None = None) -> MonitorReport:
    """`risk`, `fraud` (1 = verified fraud), `high_trust` (bool): one entry per matured live decision of this bundle.
    `baseline`: {"precision", "recall"} at t_high on the validation holdout, if known."""
    cfg = cfg or MonitorConfig()
    n = len(risk)
    rep = MonitorReport(bundle_id, "insufficient", n)
    if n < cfg.min_matured:
        rep.measures["note"] = f"{n} matured cases; needs {cfg.min_matured}"
        return rep
    flagged = risk >= t_high
    wrong = flagged != (fraud == 1)
    ece = expected_calibration_error(fraud, risk)
    rep.measures["ece"] = ece
    if ece > cfg.ece_max:
        rep.triggers.append(f"calibration error {ece:.3f} above the limit {cfg.ece_max}")
    nh = int(high_trust.sum())
    if nh >= cfg.min_high_cases:
        k = int(wrong[high_trust].sum())
        lo, hi = wilson_interval(k, nh)
        rep.measures["high_trust_error"] = {"n": nh, "rate": k / nh, "lo": lo, "hi": hi}
        if lo > cfg.high_error_limit:
            rep.triggers.append(f"high-trust error rate {k / nh:.3f} (interval {lo:.3f} to {hi:.3f}) above {cfg.high_error_limit}")
    nf = int((fraud == 1).sum())
    if baseline and nf >= cfg.min_frauds and baseline.get("recall") is not None:
        k = int((flagged & (fraud == 1)).sum())
        lo, hi = wilson_interval(k, nf)
        rep.measures["recall"] = {"n": nf, "value": k / nf, "lo": lo, "hi": hi, "baseline": baseline["recall"]}
        if hi < baseline["recall"] - cfg.tolerance:
            rep.triggers.append(f"recall {k / nf:.2f} clearly below its validation {baseline['recall']:.2f}")
    nfl = int(flagged.sum())
    if baseline and nfl >= cfg.min_flagged and baseline.get("precision") is not None:
        k = int((flagged & (fraud == 1)).sum())
        lo, hi = wilson_interval(k, nfl)
        rep.measures["precision"] = {"n": nfl, "value": k / nfl, "lo": lo, "hi": hi, "baseline": baseline["precision"]}
        if hi < baseline["precision"] - cfg.tolerance:
            rep.triggers.append(f"precision {k / nfl:.2f} clearly below its validation {baseline['precision']:.2f}")
    rep.status = "breach" if rep.triggers else "ok"
    return rep


def collect(repo, tenant_id: str, bundle_id: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Risk, verified fraud and High-trust flag for every decision of `bundle_id` whose outcome has matured."""
    outcome: dict[str, int] = {}
    for o in repo.find(tenant_id, "outcomes", {"maturity_state": "matured"}):
        if o["outcome_type"] in FRAUD:
            outcome[o["txn_id"]] = 1
        else:
            outcome.setdefault(o["txn_id"], 0)
    state: dict[str, tuple[int, str]] = {}
    for ta in repo.find(tenant_id, "trust_assessments"):
        cur = state.get(ta["prediction_id"])
        if cur is None or ta["version_no"] > cur[0]:
            state[ta["prediction_id"]] = (ta["version_no"], ta["state"])
    risk, fraud, high = [], [], []
    for p in repo.find(tenant_id, "predictions", {"bundle_id": bundle_id}):
        if p["txn_id"] in outcome:
            risk.append(float(p["calibrated_risk"]))
            fraud.append(outcome[p["txn_id"]])
            high.append(state.get(p["id"], (0, ""))[1] == "high")
    return np.array(risk), np.array(fraud, dtype=int), np.array(high, dtype=bool)
