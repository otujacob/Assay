"""Candidate lifecycle on both repositories: shadow, approval, canary, promotion, rollback (PRD 12)."""

from dataclasses import replace
from datetime import datetime, timedelta

import pytest

from assay.detection import load_bundle, save_bundle
from assay.ingestion import IngestionConfig, IngestionService, InMemoryRepository
from assay.ingestion.pg_repo import PostgresRepository
from assay.learning.lifecycle import LifecycleConfig, LifecycleError, LifecycleStore, in_canary
from assay.scoring import BundleRegistry, ScoringConfig, ScoringService
from assay.synthetic import START, to_wire

T = "tenant-synth"
both = pytest.mark.parametrize("env", ["memory", "postgres"], indirect=True)
CREATOR, APPROVER, OTHER = "u:creator", "u:approver", "u:other"


@pytest.fixture(scope="module")
def artefact(trained, tmp_path_factory):
    d = tmp_path_factory.mktemp("lcbundle") / "b"
    save_bundle(d, trained[3].scoring_bundle(), trained[3].manifest, b"k")
    return load_bundle(d, T, b"k")


def gate_report(*, failed=(), unresolved=("G4", "G6")):
    status = {"G1": "pass", "G2": "pass", "G3": "pass", "G4": "not_evaluated", "G5": "pass",
              "G6": "review_required", "G8": "pass"}
    for g in failed:
        status[g] = "fail"
    return {"version": "gates-0", "gates": [{"id": k, "status": v, "detail": ""} for k, v in status.items()],
            "failed": list(failed), "unresolved": list(unresolved)}


@pytest.fixture
def env(request, trained, artefact):
    _, txns, _, r = trained
    repo = (InMemoryRepository() if request.param == "memory"
            else PostgresRepository(request.getfixturevalue("pg_conn")))
    clock = {"t": START}
    ing = IngestionService(repo, IngestionConfig(clock=lambda: clock["t"]))
    reg = BundleRegistry()
    obj, manifest = artefact
    reg.register(T, obj, manifest)
    for cid in ("cand-1", "cand-2"):  # the same model under other ids: enough to exercise routing
        reg.register(T, obj, replace(manifest, bundle_id=cid), champion=False)
    scoring = ScoringService(repo, reg, ScoringConfig(clock=lambda: clock["t"]))
    cfg = LifecycleConfig(min_shadow_days=1.0, min_shadow_cases=5, min_canary_cases=3, clock=lambda: clock["t"])
    store = LifecycleStore(repo, cfg)
    by_id = {t["txn_id"]: t for t in txns}
    ids = r.test_table.txn_ids

    def score(txn_id):
        t = to_wire(by_id[txn_id])
        clock["t"] = max(clock["t"], datetime.fromisoformat(t["event_time"]) + timedelta(seconds=2))
        ing.ingest_transaction(T, t)
        return scoring.score_transaction(T, txn_id)

    def at_start():
        """Put the clock just before the first test transaction, so shadow time is measured from there."""
        clock["t"] = datetime.fromisoformat(to_wire(by_id[ids[0]])["event_time"]) - timedelta(minutes=1)

    return {"repo": repo, "scoring": scoring, "at_start": at_start, "store": store, "reg": reg, "score": score, "clock": clock,
            "ids": ids, "manifest": manifest}


def make(env, cid="cand-1", actor=CREATOR, **kw):
    return env["store"].create(T, actor, candidate_id=cid, base_bundle_id=env["manifest"].bundle_id,
                               artefact_path=None, pool_summary={}, training={}, gates=gate_report(**kw))


def to_shadow_ready(env, cid="cand-1", n=6):
    """In shadow with enough evidence and enough elapsed time."""
    make(env, cid)
    env["at_start"]()
    env["store"].start_shadow(T, CREATOR, cid)
    for t in env["ids"][:n]:
        env["score"](t)
    env["clock"]["t"] += timedelta(days=2)


def approve(env, cid="cand-1", **kw):
    return env["store"].approve(T, APPROVER, cid, waived=["G4"], reviewed_segments=True,
                                rationale="reviewed the gate report", **kw)


@both
def test_a_candidate_with_a_failed_gate_is_stopped_and_one_without_is_validated(env):
    assert make(env, "cand-1", failed=("G2",))["state"] == "failed"
    with pytest.raises(LifecycleError) as e:
        env["store"].start_shadow(T, CREATOR, "cand-1")
    assert e.value.code == "bad_transition"
    assert make(env, "cand-2")["state"] == "validated"


@both
def test_steps_cannot_be_skipped(env):
    make(env)
    for step in (lambda: approve(env), lambda: env["store"].start_canary(T, APPROVER, "cand-1", 0.1),
                 lambda: env["store"].promote(T, APPROVER, "cand-1", current_champion="x"),
                 lambda: env["store"].rollback(T, APPROVER, "cand-1", "no")):
        with pytest.raises(LifecycleError) as e:
            step()
        assert e.value.code == "bad_transition"


@both
def test_shadow_scores_live_traffic_and_changes_no_decision(env):
    make(env)
    env["store"].start_shadow(T, CREATOR, "cand-1")
    d = env["score"](env["ids"][0])
    assert d["bundle_id"] == env["manifest"].bundle_id  # the champion still decides
    rows = env["repo"].find(T, "shadow_scores", {"candidate_id": "cand-1"})
    assert len(rows) == 1 and rows[0]["champion_id"] == env["manifest"].bundle_id
    assert rows[0]["candidate_call"] == rows[0]["champion_call"]  # same model, same call
    assert len(env["repo"].find(T, "predictions")) == 1  # shadow made no prediction of its own


@both
def test_only_one_candidate_in_shadow_at_a_time(env):
    make(env, "cand-1")
    make(env, "cand-2")
    env["store"].start_shadow(T, CREATOR, "cand-1")
    with pytest.raises(LifecycleError) as e:
        env["store"].start_shadow(T, CREATOR, "cand-2")
    assert e.value.code == "shadow_busy"


@both
def test_approval_needs_shadow_evidence_a_second_person_a_rationale_and_explicit_waivers(env):
    make(env)
    env["at_start"]()
    env["store"].start_shadow(T, CREATOR, "cand-1")
    with pytest.raises(LifecycleError) as e:  # nothing scored yet
        approve(env)
    assert e.value.code == "shadow_not_passed"
    for t in env["ids"][:6]:
        env["score"](t)
    with pytest.raises(LifecycleError) as e:  # cases but not enough time
        approve(env)
    assert e.value.code == "shadow_not_passed" and "days" in str(e.value)
    env["clock"]["t"] += timedelta(days=2)

    with pytest.raises(LifecycleError) as e:
        env["store"].approve(T, CREATOR, "cand-1", waived=["G4"], reviewed_segments=True, rationale="mine")
    assert e.value.code == "separation_of_duties" and e.value.http == 403
    with pytest.raises(LifecycleError) as e:
        env["store"].approve(T, APPROVER, "cand-1", waived=["G4"], reviewed_segments=True, rationale="")
    assert e.value.code == "reason_required"
    with pytest.raises(LifecycleError) as e:  # G4 could not be judged and nobody waived it
        env["store"].approve(T, APPROVER, "cand-1", waived=[], reviewed_segments=True, rationale="ok")
    assert e.value.code == "unresolved_gates" and "G4" in str(e.value)
    with pytest.raises(LifecycleError) as e:  # G6 is never skipped
        env["store"].approve(T, APPROVER, "cand-1", waived=["G4"], reviewed_segments=False, rationale="ok")
    assert e.value.code == "segments_not_reviewed"
    out = approve(env)
    assert out["state"] == "approved"
    ev = out["events"][-1]
    assert ev["kind"] == "approved" and ev["actor"] == APPROVER and ev["waived_gates"] == ["G4"]


@both
def test_shadow_alarm_blocks_approval(env):
    make(env)
    env["at_start"]()
    env["store"].start_shadow(T, CREATOR, "cand-1")
    for t in env["ids"][:6]:
        env["score"](t)
    env["clock"]["t"] += timedelta(days=2)
    store = LifecycleStore(env["repo"], LifecycleConfig(min_shadow_days=1.0, min_shadow_cases=5,
                                                       max_call_rate_shift=-1.0, clock=lambda: env["clock"]["t"]))
    rep = store.shadow_report(T, "cand-1")
    assert rep["status"] == "fail" and rep["alarms"]  # any difference at all is an alarm with a negative limit
    with pytest.raises(LifecycleError) as e:
        store.approve(T, APPROVER, "cand-1", waived=["G4"], reviewed_segments=True, rationale="x")
    assert e.value.code == "shadow_not_passed"


@both
def test_canary_routes_a_share_of_traffic_deterministically(env):
    to_shadow_ready(env)
    approve(env)
    with pytest.raises(LifecycleError) as e:
        env["store"].start_canary(T, APPROVER, "cand-1", 0.9)
    assert e.value.code == "bad_share"
    env["store"].start_canary(T, APPROVER, "cand-1", 0.5)
    ids = env["ids"][6:30]
    got = {t: env["score"](t)["bundle_id"] for t in ids}
    expect = {t: ("cand-1" if in_canary(t, "canary-v1", 0.5) else env["manifest"].bundle_id) for t in ids}
    assert got == expect
    assert 0 < sum(1 for b in got.values() if b == "cand-1") < len(ids)


@both
def test_promotion_needs_a_canary_that_decided_enough_cases(env):
    to_shadow_ready(env)
    approve(env)
    env["store"].start_canary(T, APPROVER, "cand-1", 0.01)  # almost nothing goes to the candidate
    for t in env["ids"][6:12]:
        env["score"](t)
    with pytest.raises(LifecycleError) as e:
        env["store"].promote(T, APPROVER, "cand-1", current_champion=env["manifest"].bundle_id)
    assert e.value.code == "canary_too_small"


@both
def test_promote_then_roll_back_restores_the_previous_champion(env):
    to_shadow_ready(env)
    approve(env)
    env["store"].start_canary(T, APPROVER, "cand-1", 0.5)
    for t in env["ids"][6:40]:
        env["score"](t)
    out = env["store"].promote(T, APPROVER, "cand-1", current_champion=env["manifest"].bundle_id)
    assert out["state"] == "champion"
    env["scoring"].sync_routing(T)
    assert env["reg"].deciding_id(T) == "cand-1"
    assert {env["score"](t)["bundle_id"] for t in env["ids"][40:46]} == {"cand-1"}

    with pytest.raises(LifecycleError) as e:
        env["store"].rollback(T, APPROVER, "cand-1", "")
    assert e.value.code == "reason_required"
    assert env["store"].rollback(T, APPROVER, "cand-1", "calibration breach")["state"] == "rolled_back"
    env["scoring"].sync_routing(T)
    assert env["reg"].deciding_id(T) == env["manifest"].bundle_id
    assert {env["score"](t)["bundle_id"] for t in env["ids"][46:50]} == {env["manifest"].bundle_id}
    with pytest.raises(LifecycleError):  # a rolled-back candidate cannot be revived
        env["store"].start_shadow(T, CREATOR, "cand-1")


@both
def test_routing_is_rebuilt_from_the_stored_events_so_a_fresh_worker_agrees(env):
    to_shadow_ready(env)
    approve(env)
    env["store"].start_canary(T, APPROVER, "cand-1", 0.5)
    for t in env["ids"][6:40]:
        env["score"](t)
    env["store"].promote(T, APPROVER, "cand-1", current_champion=env["manifest"].bundle_id)
    fresh = LifecycleStore(env["repo"], env["store"].cfg).routing(T)
    assert fresh.champion == "cand-1" and fresh.canary is None and fresh.shadow is None
    assert fresh.previous["cand-1"] == env["manifest"].bundle_id


@both
def test_reject_needs_a_reason_and_is_final(env):
    make(env)
    with pytest.raises(LifecycleError) as e:
        env["store"].reject(T, APPROVER, "cand-1", "")
    assert e.value.code == "reason_required"
    assert env["store"].reject(T, APPROVER, "cand-1", "not needed")["state"] == "rejected"
    with pytest.raises(LifecycleError):
        env["store"].start_shadow(T, CREATOR, "cand-1")


@both
def test_every_step_is_an_append_only_record_in_order(env):
    to_shadow_ready(env)
    approve(env)
    kinds = [e["kind"] for e in env["store"].events(T, "cand-1")]
    assert kinds == ["validated", "shadow_started", "approved"]
    if isinstance(env["repo"], PostgresRepository):
        assert env["repo"].verify(T, "model_lifecycle_events") is None
        assert env["repo"].verify(T, "model_candidates") is None


def test_the_database_refuses_what_the_service_also_refuses(trained, artefact, pg_conn):
    """Separation of duties and 'no promotion without an approval' are enforced by Postgres itself."""
    import psycopg

    repo = PostgresRepository(pg_conn)
    store = LifecycleStore(repo)
    store.create(T, CREATOR, candidate_id="cand-db", base_bundle_id="b", artefact_path=None,
                 pool_summary={}, training={}, gates=gate_report())
    cand = store.candidate(T, "cand-db")

    def raw(actor, kind, detail):
        with repo.atomic(T):
            repo.insert(T, "model_lifecycle_events", {
                "schema_version": "lc-1", "candidate_id": "cand-db",
                "created_by_candidate": cand["created_by"], "kind": kind, "detail": detail}, actor)

    with pytest.raises(psycopg.errors.CheckViolation):
        raw(CREATOR, "approved", {})  # the creator approving their own candidate
    with pytest.raises(psycopg.errors.CheckViolation):
        raw(APPROVER, "promoted", {"previous": "b"})  # promotion that cites no approval
    # lying about who created the candidate, to dodge the check
    with pytest.raises(psycopg.errors.ForeignKeyViolation), repo.atomic(T):
        repo.insert(T, "model_lifecycle_events", {
            "schema_version": "lc-1", "candidate_id": "cand-db", "created_by_candidate": OTHER,
            "kind": "approved", "detail": {}}, CREATOR)
    raw(APPROVER, "approved", {})  # a different person may
