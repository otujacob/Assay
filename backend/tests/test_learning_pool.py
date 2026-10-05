from datetime import UTC, datetime, timedelta

import pytest

from assay.learning.pool import PoolConfig, analyst_histories, assess, integrity_flags

NOW = datetime(2025, 6, 1, tzinfo=UTC)
CFG = PoolConfig()


def act(txn, pid, final, *, action=None, reason="r", conf=0.9, checklist=None, blind=True, secs=60.0, age=30.0):
    return {"txn_id": txn, "analyst_pid": pid, "action": action or final or "unsure", "final_decision": final,
            "reason_code": reason, "confidence": conf,
            "evidence_checklist": {"a": True, "b": True} if checklist is None else checklist,
            "blind_flag": blind, "seconds_to_decision": secs, "action_time": NOW - timedelta(days=age)}


def proven(pid, n=14):
    """A track record: n matured cases, all called correctly, with both calls represented."""
    acts = [act(f"{pid}-p{i}", pid, "block" if i % 2 else "approve") for i in range(n)]
    return acts, {f"{pid}-p{i}": i % 2 for i in range(n)}


def one(actions, outcomes=None, rec=None, cfg=CFG, txn="t1", track=("a",)):
    outcomes = dict(outcomes or {})
    actions = list(actions)
    for pid in track:
        ha, ho = proven(pid)
        actions, outcomes = actions + ha, {**outcomes, **ho}
    rep = assess(actions, outcomes, rec or {}, NOW, cfg)
    return next(i for i in rep.items if i.txn_id == txn), rep


def test_verified_outcome_is_level_one_and_overrides_the_analyst():
    item, _ = one([act("t1", "a", "approve")], {"t1": 1})
    assert (item.level, item.label, item.source, item.disposition) == (1, 1, "verified", "accept")


def test_outcome_without_any_analyst_is_still_a_label():
    item, _ = one([], {"t1": 0})
    assert item.level == 1 and item.label == 0


def test_adjudicated_is_level_two():
    acts = [act("t1", "a", "approve"), act("t1", "b", "block"),
            act("t1", "s", "block", action="adjudicate")]
    item, _ = one(acts)
    assert (item.level, item.label, item.disposition) == (2, 1, "accept")


def test_unadjudicated_conflict_is_deferred_with_no_label():
    item, _ = one([act("t1", "a", "approve"), act("t1", "b", "block")])
    assert item.disposition == "defer" and item.label is None and "conflict" in item.reason


def test_two_analysts_agreeing_is_level_three():
    item, _ = one([act("t1", "a", "block"), act("t1", "b", "block")])
    assert (item.level, item.disposition, item.source) == (3, "accept", "corroborated")


def test_unsure_is_deferred_never_forced_into_a_label():
    item, _ = one([act("t1", "a", None, action="unsure")])
    assert item.disposition == "defer" and item.label is None


def test_single_good_decision_is_accepted_at_level_four_and_a_poor_one_is_not():
    good, _ = one([act("t1", "a", "block", conf=0.95)])
    assert good.level == 4 and good.disposition == "accept"
    meh, _ = one([act("t1", "a", "block", conf=0.1, checklist={"a": False, "b": False}, blind=False)])
    assert meh.disposition in ("defer", "reject")
    stranger, _ = one([act("t1", "z", "block", conf=0.95)], track=())
    assert stranger.disposition == "defer"  # no track record: a lone decision waits for corroboration
    assert meh.fqs < good.fqs


def test_missing_reason_when_disagreeing_with_the_model_is_rejected():
    item, _ = one([act("t1", "a", "approve", reason=None)], rec={"t1": "block"})
    assert item.disposition == "reject" and item.reason == "missing reason code"
    ok, _ = one([act("t1", "a", "approve", reason=None)], rec={"t1": "approve"})
    assert ok.disposition != "reject"  # agreeing needs no reason


def test_a_label_is_not_used_until_it_is_old_enough():
    item, _ = one([act("t1", "a", "block", age=2.0)])
    assert item.disposition == "defer" and "mature" in item.reason
    old, _ = one([act("t1", "a", "block", age=2.0)], cfg=PoolConfig(min_age_days=1.0))
    assert old.disposition == "accept"


def history_actions(pid, n_right, n_wrong, **kw):
    out = []
    for i in range(n_right + n_wrong):
        out.append(act(f"{pid}-v{i}", pid, "block", **kw))
    outcomes = {f"{pid}-v{i}": (1 if i < n_right else 0) for i in range(n_right + n_wrong)}
    return out, outcomes


def test_an_analyst_who_is_often_wrong_is_flagged_and_their_unverified_labels_quarantined():
    acts, outs = history_actions("bad", 3, 20)
    acts.append(act("u1", "bad", "block"))
    rep = assess(acts, outs, {}, NOW, CFG)
    assert "bad" in rep.flagged and any("accuracy" in w for w in rep.flagged["bad"])
    u1 = next(i for i in rep.items if i.txn_id == "u1")
    assert u1.disposition == "reject" and "quarantined" in u1.reason


def test_a_corroborated_label_survives_a_flagged_analyst():
    acts, outs = history_actions("bad", 3, 20)
    acts += [act("u1", "bad", "block"), act("u1", "good", "block")]
    rep = assess(acts, outs, {}, NOW, CFG)
    assert next(i for i in rep.items if i.txn_id == "u1").disposition == "accept"


def test_fast_and_bulk_patterns_are_flagged():
    fast = [act(f"f{i}", "f", "block" if i % 2 else "approve", secs=2.0) for i in range(25)]
    assert "implausibly fast decisions" in integrity_flags(fast, {}, CFG, {})["f"]
    bulk = [act(f"b{i}", "b", "approve") for i in range(45)]
    assert "bulk identical decisions" in integrity_flags(bulk, {}, CFG, {})["b"]
    mixed = [act(f"m{i}", "m", "block" if i % 2 else "approve", secs=60) for i in range(45)]
    assert integrity_flags(mixed, {}, CFG, {}) == {}


def test_too_little_history_never_flags():
    acts, outs = history_actions("new", 1, 8)
    assert assess(acts, outs, {}, NOW, CFG).flagged == {}


def test_accuracy_is_computed_from_matured_cases_only_and_ignores_adjudications():
    acts, outs = history_actions("p", 9, 1)
    acts.append(act("u", "p", "block"))
    acts.append(act("x", "s", "block", action="adjudicate"))
    hist = analyst_histories(acts, outs, NOW)
    assert len(hist["p"]) == 10 and "s" not in hist


def test_one_analyst_cannot_dominate_the_accepted_labels():
    acts = [act(f"a{i}", "big", "block" if i % 2 else "approve") for i in range(12)] + \
           [act(f"c{i}", "small", "approve" if i % 2 else "block") for i in range(3)]
    ha, ho = proven("big"); hb, hbo = proven("small")
    rep = assess(acts + ha + hb, {**ho, **hbo}, {}, NOW, PoolConfig(max_analyst_share=0.4))
    accepted_big = [i for i in rep.items if i.analyst_pid == "big" and i.disposition == "accept"]
    unver = [i for i in rep.items if i.source != "verified"]
    assert len(accepted_big) == int(0.4 * len(unver)) == 6
    capped = [i for i in rep.items if i.reason == "analyst share cap"]
    assert capped and all(i.analyst_pid == "big" for i in capped)


def test_cap_is_not_applied_to_a_single_contributor():
    acts = [act(f"a{i}", "only", "block" if i % 2 else "approve") for i in range(10)]
    ha, ho = proven("only")
    rep = assess(acts + ha, ho, {}, NOW, PoolConfig(max_analyst_share=0.1))
    assert sum(1 for i in rep.items if i.disposition == "accept" and i.source != "verified") == 10


def test_labels_exclude_verified_and_unaccepted_and_counts_add_up():
    ha, ho = proven("a")
    acts = [act("u1", "a", "block"), act("u2", "a", "approve", reason=None), act("v", "a", "approve")] + ha
    rep = assess(acts, {"v": 1, **ho}, {"u2": "block"}, NOW, CFG)
    assert rep.labels() == {"u1": 1}
    assert rep.labels(include_verified=True)["v"] == 1 and rep.labels(include_verified=True)["u1"] == 1
    c = rep.counts()
    assert c["cases"] == 3 + 14 and c["accepted"] + c["rejected"] + c["deferred"] == c["cases"]
    assert rep.acceptance_rate() == pytest.approx(0.5)


def test_iso_string_times_are_accepted():
    a = act("t1", "a", "block")
    a["action_time"] = (NOW - timedelta(days=30)).isoformat()
    item, _ = one([a])
    assert item.disposition == "accept"
