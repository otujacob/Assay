"""Simulated analysts, to exercise the feedback pipeline (PRD 11, H4).

THIS TESTS THE PIPELINE, NOT THE FEEDBACK ENGINE. Real analyst behaviour is not known, so every number here
comes from assumptions written below, and results from simulated analysts are not evidence that the engine
works on real ones (PRD 19, OPD-20). What the simulation can show is whether the machinery does what it is
specified to do when fed labels of known quality.

The population: four good analysts (accuracy 0.92), one weak (0.62), one adversarial (0.25) and one
"lazy" analyst who approves everything within seconds. `bad_share` is the share of the cases with no
verified outcome that the weak, adversarial and lazy analysts handle. A senior analyst (0.97) rules on
half of the conflicts. Each analyst also reviews a sample of cases that DO have verified outcomes, which
is where a track record (AAS) comes from.

By default, an analyst's stated confidence and evidence checklist say NOTHING about whether the decision is
right (`conf_signal=0`), because that is the harder and more honest case: the engine then has to rely on
track record, corroboration and integrity checks. Set `conf_signal` above zero to model analysts whose
confidence is informative.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta

import numpy as np

from assay.features.compute import parse_time

GOOD = ("an-g1", "an-g2", "an-g3", "an-g4")
WEAK, ADVERSARIAL, LAZY, SENIOR = "an-weak", "an-adv", "an-lazy", "an-senior"
BAD = (WEAK, ADVERSARIAL, LAZY)
ACCURACY = {**dict.fromkeys(GOOD, 0.92), WEAK: 0.62, ADVERSARIAL: 0.25, SENIOR: 0.97}


@dataclass(frozen=True)
class SimConfig:
    bad_share: float = 0.3
    second_review: float = 0.2  # share of unverified cases a second (good) analyst also reviews
    adjudicate: float = 0.5  # share of conflicts a senior analyst rules on
    record_share: float = 0.25  # share of verified cases each analyst reviews, for a track record
    conf_signal: float = 0.0
    seed: int = 0


def _decide(rng, pid: str, truth: int) -> str:
    if pid == LAZY:
        return "approve"
    right = rng.random() < ACCURACY[pid]
    fraud = truth if right else 1 - truth
    return "block" if fraud else "approve"


def _row(rng, cfg: SimConfig, pid: str, txn_id: str, final: str, truth: int, when, role="analyst"):
    correct = (final == "block") == bool(truth)
    conf = float(np.clip(0.7 + 0.2 * cfg.conf_signal * (1 if correct else -1) + rng.normal(0, 0.1), 0.05, 1.0))
    ticks = rng.random(4) < 0.7
    secs = float(rng.uniform(1.5, 4.0)) if pid == LAZY else float(rng.lognormal(4.4, 0.5))
    return {"txn_id": txn_id, "analyst_pid": pid, "role": role, "action": final, "final_decision": final,
            "reason_code": "reviewed", "confidence": conf,
            "evidence_checklist": {f"item{i}": bool(t) for i, t in enumerate(ticks)},
            "blind_flag": bool(rng.random() < 0.1), "seconds_to_decision": secs, "action_time": when}


def simulate_actions(txns: list[dict], truth: dict[str, dict], unverified: set[str], cfg: SimConfig,
                     as_of) -> list[dict]:
    """Analyst actions for the txns in `unverified` (no matured outcome) and for a sample of the rest."""
    rng = np.random.default_rng(cfg.seed)
    rows: list[dict] = []
    for tx in sorted(txns, key=lambda t: t["event_time"]):
        tid = tx["txn_id"]
        t_true = int(truth[tid]["is_fraud"])
        when = min(parse_time(tx["event_time"]) + timedelta(days=float(rng.uniform(1, 5))), as_of)
        if tid in unverified:
            bad = rng.random() < cfg.bad_share
            pid = str(rng.choice(BAD if bad else GOOD))
            first = _row(rng, cfg, pid, tid, _decide(rng, pid, t_true), t_true, when)
            rows.append(first)
            mine = [first]
            if rng.random() < cfg.second_review:
                other = str(rng.choice([g for g in GOOD if g != pid]))
                second = _row(rng, cfg, other, tid, _decide(rng, other, t_true), t_true, when)
                rows.append(second)
                mine.append(second)
                if len({m["final_decision"] for m in mine}) > 1 and rng.random() < cfg.adjudicate:
                    f = _decide(rng, SENIOR, t_true)
                    adj = _row(rng, cfg, SENIOR, tid, f, t_true, when, role="senior_analyst")
                    adj["action"] = "adjudicate"
                    rows.append(adj)
        else:
            for pid in (*GOOD, WEAK, ADVERSARIAL, LAZY):
                if rng.random() < cfg.record_share / 2:  # each analyst sees ~record_share/2 of verified cases
                    rows.append(_row(rng, cfg, pid, tid, _decide(rng, pid, t_true), t_true, when))
    return rows


def all_feedback_labels(actions: list[dict], unverified: set[str]) -> dict[str, int]:
    """What 'learn from every decision' would use: the last decision on each unverified case, with an
    adjudication taking precedence. No quality assessment at all."""
    out: dict[str, int] = {}
    adj: dict[str, int] = {}
    for a in actions:
        if a["txn_id"] not in unverified or not a.get("final_decision"):
            continue
        lab = 1 if a["final_decision"] == "block" else 0
        if a["action"] == "adjudicate":
            adj[a["txn_id"]] = lab
        else:
            out[a["txn_id"]] = lab
    return {**out, **adj}
