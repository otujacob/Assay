"""Feedback acceptance: which analyst decisions may reach a training set (PRD 11.2, 11.3, FR-30).

Nothing here trains anything. It reads analyst actions, the matured verified outcomes and the
recommendations that were shown, and decides for each case whether its label is accepted, rejected or
deferred, and why. Only accepted labels can enter a candidate's training pool, and then only through a
validated model update (learning/candidate.py, learning/gates.py).

Label hierarchy (PRD 11.2): 1 verified outcome, 2 adjudicated analyst decision, 3 single decision with
corroboration (a second analyst reached the same decision), 4 single uncorroborated decision.

Everything is RE-ASSESSED from raw actions each time it runs. The scores stored when an action was taken
(review/feedback.py) used the analyst's accuracy as known then, which is usually nothing; by the time a
pool is built there may be matured outcomes to judge each analyst against. The thresholds are working
defaults, not validated values (OPD-11).

Integrity checks (PRD 11.3), per analyst, over decisions that have no verified outcome:
  - implausibly fast: a large share of decisions faster than a person could have read the case;
  - bulk identical: a long run of decisions that are all the same;
  - a measured accuracy on matured cases well below chance.
A flagged analyst's unverified labels are quarantined (rejected, for manual audit), not silently trusted
and not silently dropped. Concentration on particular merchants or beneficiaries needs the graph store
and is NOT checked here. A cap limits any one analyst's share of the accepted unverified labels.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from assay.review import feedback as fb

POOL_VERSION = "pool-0"


@dataclass(frozen=True)
class PoolConfig:
    feedback: fb.FeedbackConfig = field(default_factory=fb.FeedbackConfig)
    min_age_days: float = 7.0  # an analyst label is not used until the case is this old (maturity)
    min_seconds: float = 8.0  # a decision faster than this is implausible for a real review
    fast_share: float = 0.5  # flagged if at least this share of an analyst's decisions are that fast...
    min_for_fast: int = 20  # ...over at least this many decisions
    bulk_share: float = 0.97  # flagged if this share of decisions are the same call...
    min_for_bulk: int = 40  # ...over at least this many
    min_accuracy: float = 0.5  # flagged if matured accuracy is below this with enough evidence
    min_for_accuracy: int = 15
    max_analyst_share: float = 0.35  # cap on one analyst's share of accepted unverified labels
    accept_level4_at_fqs: bool = True  # PRD 11.2: "or FQS at or above the accept threshold"


@dataclass
class PoolItem:
    txn_id: str
    label: int | None  # 1 fraud, 0 legitimate; None when there is no usable label
    level: int | None  # 1..4, None for deferred cases with no single label
    source: str  # verified | adjudicated | corroborated | single | none
    disposition: str  # accept | reject | defer
    reason: str
    analyst_pid: str | None = None
    fqs: float | None = None


@dataclass
class PoolReport:
    items: list[PoolItem]
    flagged: dict[str, list[str]]  # analyst -> why quarantined
    analyst_accuracy: dict[str, float | None]
    version: str = POOL_VERSION

    def counts(self) -> dict[str, Any]:
        by_disp = Counter(i.disposition for i in self.items)
        by_level = Counter(i.level for i in self.items if i.disposition == "accept")
        by_reason = Counter(i.reason for i in self.items if i.disposition != "accept")
        return {"cases": len(self.items), "accepted": by_disp.get("accept", 0),
                "rejected": by_disp.get("reject", 0), "deferred": by_disp.get("defer", 0),
                "accepted_by_level": {int(k): v for k, v in sorted(by_level.items()) if k is not None},
                "not_accepted_reasons": dict(by_reason.most_common()),
                "analysts_flagged": len(self.flagged)}

    def acceptance_rate(self) -> float | None:
        unverified = [i for i in self.items if i.source != "verified"]
        if not unverified:
            return None
        return sum(1 for i in unverified if i.disposition == "accept") / len(unverified)

    def labels(self, *, include_verified: bool = False) -> dict[str, int]:
        """Accepted labels for cases WITHOUT a verified outcome (those come from the outcome pipeline)."""
        return {i.txn_id: i.label for i in self.items
                if i.disposition == "accept" and i.label is not None
                and (include_verified or i.source != "verified")}


def _dt(v) -> datetime:
    if isinstance(v, datetime):
        return v if v.tzinfo else v.replace(tzinfo=UTC)
    d = datetime.fromisoformat(str(v))
    return d if d.tzinfo else d.replace(tzinfo=UTC)


def _truth(final: str) -> int:
    return 1 if final == "block" else 0


def analyst_histories(actions: list[dict], outcomes: dict[str, int], as_of: datetime) -> dict[str, list[dict]]:
    """Matured-case history per analyst, in the shape fb.analyst_accuracy expects."""
    out: dict[str, list[dict]] = defaultdict(list)
    for a in actions:
        if not a.get("final_decision") or a["action"] == "adjudicate" or a["txn_id"] not in outcomes:
            continue
        out[a["analyst_pid"]].append({
            "correct": _truth(a["final_decision"]) == outcomes[a["txn_id"]], "blind": bool(a.get("blind_flag")),
            "age_days": max(0.0, (as_of - _dt(a["action_time"])).total_seconds() / 86400)})
    return out


def integrity_flags(actions: list[dict], outcomes: dict[str, int], cfg: PoolConfig,
                    accuracy: dict[str, tuple[float | None, int]]) -> dict[str, list[str]]:
    by: dict[str, list[dict]] = defaultdict(list)
    for a in actions:
        if a.get("final_decision") and a["action"] != "adjudicate":
            by[a["analyst_pid"]].append(a)
    flagged: dict[str, list[str]] = {}
    for pid, acts in by.items():
        why = []
        timed = [a["seconds_to_decision"] for a in acts if a.get("seconds_to_decision") is not None]
        if len(timed) >= cfg.min_for_fast and sum(1 for s in timed if s < cfg.min_seconds) / len(timed) >= cfg.fast_share:
            why.append("implausibly fast decisions")
        calls = Counter(a["final_decision"] for a in acts)
        if len(acts) >= cfg.min_for_bulk and max(calls.values()) / len(acts) >= cfg.bulk_share:
            why.append("bulk identical decisions")
        aas, n = accuracy.get(pid, (None, 0))
        if aas is not None and n >= cfg.min_for_accuracy and aas < cfg.min_accuracy:
            why.append(f"accuracy on matured cases {aas:.2f}, below {cfg.min_accuracy:.2f}")
        if why:
            flagged[pid] = why
    return flagged


def assess(actions: list[dict], outcomes: dict[str, int], recommended: dict[str, str | None],
           as_of: datetime, cfg: PoolConfig | None = None) -> PoolReport:
    """`actions`: analyst_actions rows. `outcomes`: txn_id -> 1 fraud / 0 legitimate, MATURED by `as_of`.
    `recommended`: txn_id -> "approve" / "block" / None, what the model recommended (not shown if blind)."""
    cfg = cfg or PoolConfig()
    hist = analyst_histories(actions, outcomes, as_of)
    accuracy = {pid: fb.analyst_accuracy(h, cfg.feedback) for pid, h in hist.items()}
    flagged = integrity_flags(actions, outcomes, cfg, accuracy)

    by_txn: dict[str, list[dict]] = defaultdict(list)
    for a in actions:
        by_txn[a["txn_id"]].append(a)
    for txn in outcomes:  # a verified outcome is a label even if no analyst touched the case
        by_txn.setdefault(txn, [])

    items: list[PoolItem] = []
    for txn, acts in by_txn.items():
        if txn in outcomes:
            items.append(PoolItem(txn, outcomes[txn], 1, "verified", "accept", "verified outcome"))
            continue
        items.append(_assess_unverified(txn, acts, recommended.get(txn), as_of, cfg, accuracy, flagged))

    _cap_shares(items, cfg)
    return PoolReport(items, flagged, {pid: a for pid, (a, _) in accuracy.items()})


def _assess_unverified(txn: str, acts: list[dict], rec: str | None, as_of: datetime, cfg: PoolConfig,
                       accuracy: dict, flagged: dict) -> PoolItem:
    adj = [a for a in acts if a["action"] == "adjudicate" and a.get("final_decision")]
    finals = [a for a in acts if a.get("final_decision") and a["action"] != "adjudicate"]
    if not adj and not finals:
        reason = "analyst marked unsure" if any(a["action"] == "unsure" for a in acts) else "no decision yet"
        return PoolItem(txn, None, None, "none", "defer", reason)

    if adj:  # level 2: a senior person ruled on a conflict, with a stored rationale
        a = adj[-1]
        newest = max(_dt(x["action_time"]) for x in acts)
        if (as_of - newest).total_seconds() / 86400 < cfg.min_age_days:
            return PoolItem(txn, _truth(a["final_decision"]), 2, "adjudicated", "defer", "label not yet mature",
                            a["analyst_pid"])
        return PoolItem(txn, _truth(a["final_decision"]), 2, "adjudicated", "accept", "adjudicated decision",
                        a["analyst_pid"])

    if len({a["final_decision"] for a in finals}) > 1:
        return PoolItem(txn, None, None, "none", "defer", "conflicting decisions, awaiting adjudication")

    a = finals[-1]
    label, pid = _truth(a["final_decision"]), a["analyst_pid"]
    analysts = {x["analyst_pid"] for x in finals}
    corroborated = len(analysts) >= 2
    level = 3 if corroborated else 4
    source = "corroborated" if corroborated else "single"
    disagrees = bool(rec and a["final_decision"] != rec)
    reason_present = bool(a.get("reason_code")) or not disagrees
    aas, _ = accuracy.get(pid, (None, 0))
    fcs = fb.feedback_confidence(a.get("confidence"), a.get("evidence_checklist") or {}, corroborated,
                                 reason_present, bool(a.get("blind_flag")), cfg.feedback)
    fqs = fb.feedback_quality(fcs, aas)

    def item(disposition: str, why: str) -> PoolItem:
        return PoolItem(txn, label, level, source, disposition, why, pid, fqs)

    if pid in flagged and not corroborated:
        return item("reject", "analyst quarantined: " + "; ".join(flagged[pid]))
    if not reason_present:
        return item("reject", "missing reason code")
    if fqs < cfg.feedback.reject_threshold:
        return item("reject", "feedback quality below reject threshold")
    if (as_of - _dt(a["action_time"])).total_seconds() / 86400 < cfg.min_age_days:
        return item("defer", "label not yet mature")
    if level == 3:
        return item("accept", "corroborated by a second analyst")
    if cfg.accept_level4_at_fqs and fqs >= cfg.feedback.accept_threshold:
        return item("accept", "single decision with feedback quality above accept threshold")
    return item("defer", "awaiting verified outcome or corroboration")


def _cap_shares(items: list[PoolItem], cfg: PoolConfig) -> None:
    """No analyst may supply more than `max_analyst_share` of the accepted unverified labels. The excess
    (lowest quality first) is deferred, not rejected: those labels are not wrong, just over-represented."""
    acc = [i for i in items if i.disposition == "accept" and i.source != "verified" and i.analyst_pid]
    if len({i.analyst_pid for i in acc}) < 2:
        return  # a cap is meaningless with one contributor
    limit = max(1, int(cfg.max_analyst_share * len(acc)))
    by: dict[str, list[PoolItem]] = defaultdict(list)
    for i in acc:
        by[i.analyst_pid].append(i)
    for lst in by.values():
        lst.sort(key=lambda i: (i.level or 9, -(i.fqs or 0.0)))
        for i in lst[limit:]:
            i.disposition, i.reason = "defer", "analyst share cap"
