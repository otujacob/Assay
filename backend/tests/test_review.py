"""The review workflow on both repositories: queue, blind review, actions, conflicts, feedback."""

from datetime import datetime, timedelta

import numpy as np
import pytest

from assay.detection import load_bundle, save_bundle
from assay.ingestion import IngestionConfig, IngestionService, InMemoryRepository
from assay.ingestion.pg_repo import PostgresRepository
from assay.review.service import (
    REVIEW_ACTIONS,
    ReviewConfig,
    ReviewError,
    ReviewService,
    blind_assignment,
)
from assay.scoring import BundleRegistry, ScoringConfig, ScoringService
from assay.synthetic import START, to_wire

T = "tenant-synth"
both = pytest.mark.parametrize("env", ["memory", "postgres"], indirect=True)


@pytest.fixture(scope="module")
def artefact(trained, tmp_path_factory):
    d = tmp_path_factory.mktemp("rvbundle") / "b"
    save_bundle(d, trained[3].scoring_bundle(), trained[3].manifest, b"k")
    return load_bundle(d, T, b"k")


@pytest.fixture
def env(request, trained, artefact):
    ds, txns, _, r = trained
    repo = (InMemoryRepository() if request.param == "memory"
            else PostgresRepository(request.getfixturevalue("pg_conn")))
    clock = {"t": START}
    ing = IngestionService(repo, IngestionConfig(clock=lambda: clock["t"]))
    reg = BundleRegistry()
    reg.register(T, *artefact)
    scoring = ScoringService(repo, reg, ScoringConfig(clock=lambda: clock["t"]))
    review = ReviewService(repo, scoring, ReviewConfig(clock=lambda: clock["t"], blind_share=0.5))
    by_id = {t["txn_id"]: t for t in txns}
    ids = r.test_table.txn_ids

    def ingest(txn_id):
        t = to_wire(by_id[txn_id])
        clock["t"] = datetime.fromisoformat(t["event_time"]) + timedelta(seconds=2)
        ing.ingest_transaction(T, t)

    def score(txn_id):
        ingest(txn_id)
        return scoring.score_transaction(T, txn_id)

    def scored(n=14, offset=0):
        return [score(t) for t in ids[offset:offset + 3000:3000 // n][:n]]

    return {"repo": repo, "scoring": scoring, "review": review, "score": score, "scored": scored,
            "clock": clock, "ing": ing, "by_id": by_id, "ids": ids, "r": r, "ds": ds}


def queued(env, decisions):
    return [d for d in decisions if d["recommendation"] in REVIEW_ACTIONS]


def pick_case(env, *, blind: bool | None = None, n=24):
    for d in env["scored"](n):
        if d["recommendation"] in REVIEW_ACTIONS:
            c = env["repo"].find(T, "review_cases", {"txn_id": d["txn_id"]})[0]
            if blind is None or c["blind"] == blind:
                return d
    raise AssertionError("no matching case found")


# ---- blind assignment (FR-28) ------------------------------------------------------------------
def test_blind_assignment_is_deterministic_and_matches_the_configured_share():
    ids = [f"t-{i}" for i in range(20000)]
    flags = [blind_assignment(T, i, 0.10, "s") for i in ids]
    assert flags == [blind_assignment(T, i, 0.10, "s") for i in ids]
    assert abs(np.mean(flags) - 0.10) < 0.01
    assert blind_assignment(T, "x", 0.0, "s") is False and blind_assignment(T, "x", 1.0, "s") is True
    assert [blind_assignment("other", i, 0.1, "s") for i in ids[:200]] != flags[:200]  # tenant-specific


# ---- enqueue ----------------------------------------------------------------------------------------
@both
def test_only_review_and_escalate_decisions_are_queued_and_the_blind_flag_is_recorded_once(env):
    decisions = env["scored"](14)
    cases = env["repo"].rows(T, "review_cases")
    assert {c["txn_id"] for c in cases} == {d["txn_id"] for d in queued(env, decisions)}
    assert all(d["recommendation"] not in REVIEW_ACTIONS for d in decisions if d["txn_id"] not in
               {c["txn_id"] for c in cases})
    assert cases and all(isinstance(c["blind"], bool) and c["sla_minutes"] > 0 for c in cases)
    before = {c["txn_id"]: c["blind"] for c in cases}
    # refining a decision creates a new policy decision but must not re-queue or re-roll blind
    pend = [d for d in decisions if d["explanation_status"] == "pending" and d["recommendation"] in REVIEW_ACTIONS
            and d["trust"]["state"] != "insufficient_evidence"]
    if pend:
        env["scoring"].refine(T, pend[0]["decision_id"])
    after = {c["txn_id"]: c["blind"] for c in env["repo"].rows(T, "review_cases")}
    assert after == before


@both
def test_async_refinement_supersedes_cases_that_no_longer_need_review(env):
    """PRD 5.8: the explanation worker completes components and stores a new version. A case whose
    refined recommendation no longer asks for review, and that nobody has touched, is superseded."""
    env["scored"](60)
    before = env["review"].queue(T, viewer_role="manager")
    done = env["scoring"].refine_pending(T, limit=1000)
    assert done > 0
    assert env["scoring"].refine_pending(T, limit=1000) == 0  # idempotent: nothing left to refine
    after = env["review"].queue(T, viewer_role="manager")
    everything = env["review"].queue(T, viewer_role="manager", include_closed=True)
    superseded = [i for i in everything["items"] if i["status"] == "superseded"]
    assert superseded, "expected refinement to resolve some cases"
    assert after["matching"] == before["matching"] - len(superseded)
    assert all(i["status"] != "superseded" for i in after["items"])
    # drift is available on the first version too: it needs attributions, not explanation testing
    v1 = env["repo"].find(T, "trust_assessments", {"version_no": 1})
    assert all(a["components"]["drift"]["status"] == "active" for a in v1)
    for table in ("trust_assessments", "policy_decisions", "review_cases"):
        assert env["repo"].verify(T, table) is None


# ---- queue ------------------------------------------------------------------------------------------
@both
def test_queue_is_ordered_by_priority_with_sla_and_filters(env):
    env["scored"](20)
    q = env["review"].queue(T, viewer_role="manager")
    items = q["items"]
    assert items and q["matching"] == len(items) <= q["total_cases"]
    scores = [i["priority_score"] for i in items]
    assert scores == sorted(scores, reverse=True)
    assert {i["priority"] for i in items} <= {"P1", "P2", "P3", "P4"}
    assert all(i["sla"]["minutes"] > 0 and "due_at" in i["sla"] for i in items)
    band = items[0]["risk_band"]
    only = env["review"].queue(T, viewer_role="manager", risk_band=band)["items"]
    assert only and all(i["risk_band"] == band for i in only)
    one = items[0]["txn_id"]
    assert [i["txn_id"] for i in env["review"].queue(T, viewer_role="manager", search=one[-6:])["items"]
            if i["txn_id"] == one]
    assert env["review"].queue(T, viewer_role="manager", search="zzzz-none")["items"] == []


@both
def test_queue_redacts_blind_cases_for_analysts_but_not_for_managers(env):
    env["scored"](24)
    cases = {c["txn_id"]: c["blind"] for c in env["repo"].rows(T, "review_cases")}
    assert any(cases.values()) and not all(cases.values())  # the test needs both kinds
    analyst = {i["txn_id"]: i for i in env["review"].queue(T, viewer_role="analyst")["items"]}
    manager = {i["txn_id"]: i for i in env["review"].queue(T, viewer_role="manager")["items"]}
    for txn_id, blind in cases.items():
        assert analyst[txn_id]["redacted"] is blind
        if blind:
            assert analyst[txn_id]["risk_band"] is None and analyst[txn_id]["trust_state"] is None
            assert manager[txn_id]["risk_band"] is not None
        else:
            assert analyst[txn_id]["risk_band"] == manager[txn_id]["risk_band"]


# ---- case view & blind review (FR-27) -----------------------------------------------------------------
@both
def test_blind_case_hides_score_and_recommendation_until_the_analyst_decides(env):
    d = pick_case(env, blind=True)
    v = env["review"].case(T, d["decision_id"], viewer_role="analyst", viewer_pid="u:ana")
    assert v["blind"] and v["redacted"] and v["decision"] is None and v["explanation"] is None
    assert v["transaction"]["amount"] > 0  # transaction facts are still shown
    res = env["review"].record_action(T, d["decision_id"], analyst_pid="u:ana", role="analyst",
                                      action="block" if d["risk_band"] == "high" else "approve",
                                      reason_code="blind_call")
    row = env["repo"].find(T, "analyst_actions", {"analyst_pid": "u:ana"})[0]
    assert row["blind_flag"] is True and row["display_state"]["score_shown"] is False
    assert row["display_state"]["risk"] is None and row["display_state"]["recommendation"] is None
    assert res["feedback"]["fcs"] > 0
    after = env["review"].case(T, d["decision_id"], viewer_role="analyst", viewer_pid="u:ana")
    assert after["decision"] is not None  # once decided, the analyst may see what the model said
    mgr = env["review"].case(T, d["decision_id"], viewer_role="manager", viewer_pid="u:mgr")
    assert mgr["decision"] is not None and mgr["redacted"] is False


@both
def test_other_analysts_decisions_are_hidden_on_blind_cases_until_you_decide(env):
    d = pick_case(env, blind=True)
    env["review"].record_action(T, d["decision_id"], analyst_pid="u:a1", role="analyst", action="approve",
                                reason_code="r")
    second = env["review"].case(T, d["decision_id"], viewer_role="analyst", viewer_pid="u:a2")
    assert second["actions"] == []  # no anchoring on a colleague's call
    assert env["review"].case(T, d["decision_id"], viewer_role="manager", viewer_pid="m")["actions"]


@both
def test_non_blind_case_shows_everything_and_records_what_was_shown(env):
    d = pick_case(env, blind=False)
    v = env["review"].case(T, d["decision_id"], viewer_role="analyst", viewer_pid="u:ana")
    assert not v["redacted"] and v["decision"]["decision_id"] == d["decision_id"]
    assert v["transaction"]["customer"] != env["by_id"][d["txn_id"]]["customer_pid"]  # masked by default
    env["review"].record_action(T, d["decision_id"], analyst_pid="u:ana", role="analyst",
                                action="request_review", confidence=0.6, checklist={"a": True, "b": False})
    ds = env["repo"].find(T, "analyst_actions", {"analyst_pid": "u:ana"})[0]["display_state"]
    assert ds["score_shown"] is True and ds["risk"] == d["risk"] and ds["recommendation"] == d["recommendation"]
    assert ds["policy_version"] == d["policy_version"]


# ---- actions & validation (FR-26) ------------------------------------------------------------------------
@both
def test_action_validation(env):
    d = pick_case(env, blind=False)
    rv = env["review"]
    kw = {"analyst_pid": "u:a", "role": "analyst"}
    with pytest.raises(ReviewError, match="one of"):
        rv.record_action(T, d["decision_id"], action="delete", **kw)
    with pytest.raises(ReviewError) as e:
        rv.record_action(T, d["decision_id"], action="override", override_to="approve", **kw)
    assert e.value.code == "reason_required"  # an override without a reason is rejected (FR-26)
    with pytest.raises(ReviewError, match="approve or block"):
        rv.record_action(T, d["decision_id"], action="override", reason_code="x", override_to="maybe", **kw)
    with pytest.raises(ReviewError, match="confidence"):
        rv.record_action(T, d["decision_id"], action="approve", reason_code="x", confidence=1.5, **kw)
    with pytest.raises(ReviewError) as e:
        rv.record_action(T, "00000000-0000-0000-0000-000000000000", action="approve", **kw)
    assert e.value.http == 404
    assert env["repo"].find(T, "analyst_actions") == []  # nothing rejected was stored


@both
def test_disagreeing_with_a_recommendation_needs_a_reason_but_agreeing_does_not(env):
    decisions = env["scored"](40)
    approve_like = [d for d in decisions if d["recommendation"] in ("approve", "approve_sampled_qa")]
    if not approve_like:
        pytest.skip("no approve recommendation in this sample")
    d = approve_like[0]
    with pytest.raises(ReviewError) as e:
        env["review"].record_action(T, d["decision_id"], analyst_pid="u:a", role="analyst", action="block")
    assert e.value.code == "reason_required"
    ok = env["review"].record_action(T, d["decision_id"], analyst_pid="u:a", role="analyst", action="approve")
    assert ok["status"] == "decided" and ok["final_decision"] == "approve"


@both
def test_a_case_can_be_opened_and_overridden_even_if_never_queued(env):
    decisions = env["scored"](40)
    unqueued = [d for d in decisions if d["recommendation"] not in REVIEW_ACTIONS]
    assert unqueued, "expected at least one approve or block recommendation"
    d = unqueued[0]
    v = env["review"].case(T, d["decision_id"], viewer_role="manager", viewer_pid="m")
    assert v["queued"] is False
    assert env["repo"].find(T, "review_cases", {"txn_id": d["txn_id"]}) == []  # viewing wrote nothing
    flip = "block" if d["recommendation"].startswith("approve") else "approve"
    env["review"].record_action(T, d["decision_id"], analyst_pid="u:a", role="analyst", action="override",
                                override_to=flip, reason_code="customer_called")
    assert env["repo"].find(T, "review_cases", {"txn_id": d["txn_id"]})[0]["queue"] == "override"


def test_override_of_a_high_trust_recommendation_needs_second_review():
    svc = object.__new__(ReviewService)
    ld = {"ta": {"state": "high"}, "pd": {"recommended_action": "block"}}
    override = {"action": "override", "final_decision": "approve"}
    assert svc._needs_second_review(ld, [override]) is True
    assert svc._needs_second_review(ld, [override, {"action": "approve", "final_decision": "approve"}]) is False
    assert svc._needs_second_review({"ta": {"state": "moderate"}, "pd": ld["pd"]}, [override]) is False
    assert svc._needs_second_review(ld, [{"action": "block", "final_decision": "block"}]) is False


# ---- conflicts & adjudication (FR-29, PRD 8.4) --------------------------------------------------------------
@both
def test_conflicting_decisions_are_kept_then_adjudicated(env):
    d = pick_case(env, blind=False)
    rv = env["review"]
    rv.record_action(T, d["decision_id"], analyst_pid="u:a1", role="analyst", action="approve", reason_code="r1")
    res = rv.record_action(T, d["decision_id"], analyst_pid="u:a2", role="analyst", action="block", reason_code="r2")
    assert res["status"] == "conflicted"
    assert len(env["repo"].find(T, "analyst_actions", {"txn_id": d["txn_id"]})) == 2  # none overwritten
    assert [i["txn_id"] for i in rv.queue(T, viewer_role="manager")["items"] if i["status"] == "conflicted"]
    with pytest.raises(ReviewError) as e:
        rv.adjudicate(T, d["decision_id"], analyst_pid="u:a3", role="analyst", final_decision="block", rationale="x")
    assert e.value.http == 403
    with pytest.raises(ReviewError) as e:
        rv.adjudicate(T, d["decision_id"], analyst_pid="u:s", role="senior_analyst", final_decision="block", rationale="")
    assert e.value.code == "reason_required"
    v = rv.adjudicate(T, d["decision_id"], analyst_pid="u:s", role="senior_analyst", final_decision="block",
                      rationale="second evidence", notes="called customer")
    assert v["status"] == "adjudicated" and v["final_decision"] == "block"
    assert len(env["repo"].find(T, "analyst_actions", {"txn_id": d["txn_id"]})) == 3
    with pytest.raises(ReviewError) as e:
        rv.record_action(T, d["decision_id"], analyst_pid="u:a4", role="analyst", action="approve", reason_code="r")
    assert e.value.http == 409
    assert d["txn_id"] not in [i["txn_id"] for i in rv.queue(T, viewer_role="manager")["items"]]


@both
def test_agreeing_decisions_are_decided_not_conflicted_and_a_review_cannot_decide_twice(env):
    d = pick_case(env, blind=False)
    rv = env["review"]
    a = rv.record_action(T, d["decision_id"], analyst_pid="u:a1", role="analyst", action="approve", reason_code="r")
    b = rv.record_action(T, d["decision_id"], analyst_pid="u:a2", role="analyst", action="approve", reason_code="r")
    assert a["status"] == b["status"] == "decided"
    assert b["feedback"]["fcs"] > a["feedback"]["fcs"]  # the second call is corroborated by the first
    with pytest.raises(ReviewError) as e:
        rv.record_action(T, d["decision_id"], analyst_pid="u:a1", role="analyst", action="block", reason_code="r")
    assert e.value.code == "already_decided"


@both
def test_escalated_cases_are_decided_by_a_senior_analyst(env):
    d = pick_case(env, blind=False)
    rv = env["review"]
    assert rv.record_action(T, d["decision_id"], analyst_pid="u:a1", role="analyst", action="escalate")["status"] == "escalated"
    with pytest.raises(ReviewError) as e:
        rv.record_action(T, d["decision_id"], analyst_pid="u:a2", role="analyst", action="approve", reason_code="r")
    assert e.value.http == 403
    done = rv.record_action(T, d["decision_id"], analyst_pid="u:s", role="senior_analyst", action="block", reason_code="r")
    assert done["status"] == "decided"


@both
def test_a_verified_outcome_overrides_the_analyst_as_the_label(env):
    d = pick_case(env, blind=False)
    env["review"].record_action(T, d["decision_id"], analyst_pid="u:a1", role="analyst", action="approve", reason_code="r")
    txn = env["by_id"][d["txn_id"]]
    env["ing"].ingest_outcome(T, {"event_id": "o1", "txn_id": d["txn_id"], "outcome_type": "confirmed_fraud",
                                  "source": "chargeback", "event_time": "2027-01-01T00:00:00Z",
                                  "schema_version": "out-1"})
    v = env["review"].case(T, d["decision_id"], viewer_role="manager", viewer_pid="m")
    assert v["outcome"] == {"verified": "fraud", "type": "confirmed_fraud"}  # stands against "approve"
    assert txn["txn_id"] == d["txn_id"]


# ---- feedback (FR-30, shadow) ---------------------------------------------------------------------------------
@both
def test_feedback_scores_are_stored_in_shadow_with_the_formula_version(env):
    d = pick_case(env, blind=False)
    res = env["review"].record_action(T, d["decision_id"], analyst_pid="u:a1", role="analyst", action="unsure",
                                      confidence=0.4, checklist={"kyc": True, "contact": False})
    f = res["feedback"]
    assert f["formula_version"] == "fqs-0" and f["aas"] is None and f["crs"] is None  # no history yet
    assert f["disposition"] == "defer" and "unsure" in f["reason"]
    assert 0 <= f["fqs"] <= f["fcs"] <= 1 and "not used for learning" in f["note"]
    stored = env["repo"].find(T, "feedback_records")[0]
    assert stored["disposition"] == "defer" and stored["formula_version"] == "fqs-0"
    assert env["repo"].verify(T, "feedback_records") is None


# ---- dashboard (FR-35) ---------------------------------------------------------------------------------------------
@both
def test_dashboard_numbers_equal_direct_queries(env):
    decisions = env["scored"](20)
    rv = env["review"]
    d = pick_case(env, blind=False)
    rv.record_action(T, d["decision_id"], analyst_pid="u:a1", role="analyst", action="approve", reason_code="r",
                     confidence=0.9, checklist={"kyc": True, "contact": True})
    dash = rv.dashboard(T)
    q = rv.queue(T, viewer_role="manager", limit=10_000)
    assert dash["pending_reviews"]["count"] == q["matching"] and dash["pending_reviews"]["sla_breached"] == q["sla_breached"]
    states = {}
    for dd in decisions:
        states[dd["trust"]["state"]] = states.get(dd["trust"]["state"], 0) + 1
    assert sum(dash["trust_distribution"].values()) == dash["assessed_cases"]
    assert dash["high_trust_error_rate"]["value"] is None and dash["high_trust_error_rate"]["n"] == 0
    assert dash["feedback_acceptance"]["deferred"] == 1 and dash["feedback_acceptance"]["accepted"] == 0
    assert dash["model_drift"]["status"] == "stable" and dash["data_quality"]["status"] == "healthy"


@both
def test_dashboard_high_trust_error_rate_uses_only_matured_outcomes(env):
    decisions = env["scored"](100)
    # Explanation testing is what makes High trust reachable (PRD 5.4), so complete it first.
    refined = [env["scoring"].refine(T, d["decision_id"]) if d["explanation_status"] == "pending" else d
               for d in decisions]
    high = [d for d in refined if d["trust"]["state"] == "high"]
    assert len(high) >= 2, f"only {len(high)} high-trust cases in the sample"
    lb = env["scoring"].registry.champion(T)
    t_high = lb.manifest.thresholds["t_high"]
    wrong = 0
    for i, d in enumerate(high):
        fraud = (d["risk"] < t_high) if i == 0 else (d["risk"] >= t_high)  # make the first one wrong
        wrong += int((d["risk"] >= t_high) != fraud)
        env["ing"].ingest_outcome(T, {"event_id": f"o{i}", "txn_id": d["txn_id"],
                                      "outcome_type": "confirmed_fraud" if fraud else "confirmed_legitimate",
                                      "source": "chargeback" if fraud else "dispute_window_closed",
                                      "event_time": "2030-01-01T00:00:00Z", "schema_version": "out-1"})
    ht = env["review"].dashboard(T)["high_trust_error_rate"]
    assert ht["n"] == len(high) and ht["value"] == pytest.approx(wrong / len(high))
    assert ht["lo"] <= ht["value"] <= ht["hi"] and ht["basis"] == "matured verified outcomes"


@both
def test_review_chains_verify(env):
    d = pick_case(env, blind=False)
    env["review"].record_action(T, d["decision_id"], analyst_pid="u:a1", role="analyst", action="approve", reason_code="r")
    for table in ("review_cases", "analyst_actions", "feedback_records", "audit_log"):
        assert env["repo"].verify(T, table) is None, table
