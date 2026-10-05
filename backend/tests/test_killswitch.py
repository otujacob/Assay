"""The kill switch (PRD 12.4): every case to a person while it is engaged."""

import json
from datetime import datetime, timedelta

import numpy as np
import pytest
from fastapi.testclient import TestClient

from assay.api import Credential, Credentials, create_app, sign
from assay.detection import load_bundle, save_bundle
from assay.ingestion import IngestionConfig, IngestionService, InMemoryRepository
from assay.learning.killswitch import KillSwitch, KillSwitchError
from assay.learning.monitor import MonitorConfig
from assay.learning.service import LearningConfig, LearningService
from assay.policy import Action, PolicyInput, evaluate
from assay.scoring import BundleRegistry, ScoringConfig, ScoringService
from assay.synthetic import START, to_wire
from assay.trust import TrustState

T = "tenant-synth"
ROLES = {"ing": {"ingest"}, "adm": {"admin"}, "apr": {"approver"}, "ana": {"analyst"}, "aud": {"auditor"}}
SEC = {k: k.encode() + b"-secret" for k in ROLES}


def test_the_engine_sends_everything_to_a_person_before_anything_else_is_considered():
    calm = PolicyInput(0.001, TrustState.HIGH)
    assert evaluate(calm).action is Action.APPROVE
    killed = evaluate(PolicyInput(0.001, TrustState.HIGH, kill_switch=True))
    assert killed.action is Action.REQUEST_HUMAN_REVIEW and killed.gate == "kill_switch"
    # even a hard rule that would block, and a high-risk high-trust case that would be blocked, go to a person
    assert evaluate(PolicyInput(0.9, TrustState.HIGH, kill_switch=True)).action is Action.REQUEST_HUMAN_REVIEW
    assert evaluate(PolicyInput(0.1, TrustState.HIGH, hard_rule_action=Action.BLOCK, kill_switch=True)).gate == "kill_switch"


def test_state_changes_need_a_reason_and_the_right_order_and_leave_a_history():
    ks = KillSwitch(InMemoryRepository())
    assert ks.state(T) == {"engaged": False}
    with pytest.raises(KillSwitchError) as e:
        ks.engage(T, "u:a", "  ")
    assert e.value.code == "reason_required"
    with pytest.raises(KillSwitchError) as e:
        ks.release(T, "u:a", "x")
    assert e.value.code == "not_engaged"
    assert ks.engage(T, "u:a", "model misbehaving")["engaged"] is True
    with pytest.raises(KillSwitchError) as e:
        ks.engage(T, "u:b", "again")
    assert e.value.code == "already_engaged"
    st = ks.release(T, "u:c", "fixed")
    assert st["engaged"] is False and st["by"] == "u:c"
    assert ks.history(T) == [{"kind": "engaged", "reason": "model misbehaving", "by": "u:a"},
                             {"kind": "released", "reason": "fixed", "by": "u:c"}]
    assert ks.state("other") == {"engaged": False}
    assert [a["action"] for a in ks.repo.find(T, "audit_log")] == ["kill_switch_engaged", "kill_switch_released"]


@pytest.fixture(scope="module")
def artefact(trained, tmp_path_factory):
    d = tmp_path_factory.mktemp("ksbundle") / "b"
    save_bundle(d, trained[3].scoring_bundle(), trained[3].manifest, b"k")
    return load_bundle(d, T, b"k")


@pytest.fixture
def env(trained, artefact):
    _, txns, _, r = trained
    repo, clock = InMemoryRepository(), {"t": START}
    reg = BundleRegistry()
    reg.register(T, *artefact)
    ing = IngestionService(repo, IngestionConfig(clock=lambda: clock["t"]))
    scoring = ScoringService(repo, reg, ScoringConfig(clock=lambda: clock["t"]))
    cfg = LearningConfig(monitor=MonitorConfig(min_matured=100))
    creds = Credentials([Credential(k, T, SEC[k], frozenset(v)) for k, v in ROLES.items()])
    client = TestClient(create_app(ing, creds, scoring=scoring, rate_limit_per_minute=None, learning=cfg))
    by_id, ids = {t["txn_id"]: t for t in txns}, r.test_table.txn_ids

    def call(method, path, key, payload=None):
        body = b"" if payload is None else json.dumps(payload).encode()
        return client.request(method, path, content=body, headers={"X-Assay-Key": key, "X-Assay-Signature": sign(SEC[key], body)})

    def submit(i):
        t = to_wire(by_id[ids[i]])
        clock["t"] = max(clock["t"], datetime.fromisoformat(t["event_time"]) + timedelta(seconds=2))
        return call("POST", "/v1/transactions", "ing", t).json()["decision"]

    return {"call": call, "submit": submit, "repo": repo, "svc": LearningService(repo, scoring, cfg), "ids": ids, "scoring": scoring}


def low_risk_ids(env, n=3):
    out, i = [], 0
    while len(out) < n:
        d = env["submit"](i)
        i += 1
        if d["gate"] != "kill_switch" and d["recommendation"] not in ("request_human_review", "escalate", "hold", "request_human_review_priority", "block"):
            out.append(i - 1)
    return out


def test_while_engaged_every_new_case_goes_to_a_person_and_the_model_is_still_recorded(env):
    approvable = low_risk_ids(env)  # cases the model would normally approve
    assert env["call"]("POST", "/v1/learning/kill-switch/engage", "adm", {"reason": "drift incident"}).json()["engaged"] is True
    n_pred = len(env["repo"].find(T, "predictions"))
    for i in range(600, 606):
        d = env["submit"](i)
        assert d["gate"] == "kill_switch" and d["recommendation"] == "request_human_review"
    assert len(env["repo"].find(T, "predictions")) == n_pred + 6      # scoring still ran and was stored: lineage is complete
    assert all(isinstance(env["submit"](i)["risk"], float) for i in approvable[:1])
    env["call"]("POST", "/v1/learning/kill-switch/release", "apr", {"reason": "root cause fixed"})
    assert env["submit"](700)["gate"] != "kill_switch"


def test_decisions_made_before_it_was_engaged_are_not_rewritten(env):
    low_risk_ids(env, 1)
    before = env["repo"].find(T, "policy_decisions")
    env["call"]("POST", "/v1/learning/kill-switch/engage", "apr", {"reason": "stop"})
    assert env["repo"].find(T, "policy_decisions") == before


def test_who_may_engage_and_release_and_what_the_status_shows(env):
    p = "/v1/learning/kill-switch/"
    for key in ("ana", "aud", "ing"):
        assert env["call"]("POST", p + "engage", key, {"reason": "x"}).status_code == 403, key
    assert env["call"]("POST", p + "engage", "adm", {}).status_code == 422                   # a reason is required
    assert env["call"]("POST", p + "engage", "adm", {"reason": "x"}).status_code == 200
    assert env["call"]("POST", p + "engage", "apr", {"reason": "y"}).status_code == 409      # already engaged
    assert env["call"]("POST", p + "release", "adm", {"reason": "z"}).status_code == 403     # an administrator cannot release
    assert env["call"]("POST", p + "release", "apr", {}).status_code == 422
    st = env["call"]("GET", "/v1/learning/status", "aud").json()["kill_switch"]
    assert st == {"engaged": True, "last": "engaged", "reason": "x", "by": "u:adm"}
    assert env["call"]("POST", p + "release", "apr", {"reason": "z"}).status_code == 200
    assert env["call"]("GET", "/v1/learning/status", "aud").json()["kill_switch"]["engaged"] is False


def live_decisions(repo, bundle_id, n, *, good):
    rng = np.random.default_rng(5)
    for i in range(n):
        fraud = rng.random() < 0.2
        flagged = fraud if good else not fraud
        p = repo.insert(T, "predictions", {"txn_id": f"m{i}", "bundle_id": bundle_id, "calibrated_risk": 0.9 if flagged else 0.01}, "s")
        repo.insert(T, "trust_assessments", {"prediction_id": p["id"], "version_no": 1, "state": "moderate"}, "s")
        repo.insert(T, "outcomes", {"txn_id": f"m{i}", "outcome_type": "confirmed_fraud" if fraud else "confirmed_legitimate",
                                    "maturity_state": "matured"}, "s")


def test_when_the_only_model_degrades_the_monitor_engages_the_kill_switch_because_there_is_nothing_to_fall_back_to(env):
    bid = env["scoring"].registry.deciding_id(T)
    live_decisions(env["repo"], bid, 400, good=False)
    seen = env["svc"].check_health(T)
    assert seen[0]["status"] == "breach" and seen[0]["kill_switch_engaged"] is False and not env["svc"].kill_switch().engaged(T)
    out = env["svc"].check_health(T, act=True)
    assert out[0]["kill_switch_engaged"] is True and env["svc"].kill_switch().engaged(T)
    st = env["svc"].kill_switch().state(T)
    assert st["by"] == "system:monitor" and st["reason"].startswith("automatic: ") and "no safe model" in st["reason"]
    assert env["svc"].check_health(T, act=True)[0]["kill_switch_engaged"] is False  # not engaged twice


def test_a_healthy_only_model_never_trips_it(env):
    live_decisions(env["repo"], env["scoring"].registry.deciding_id(T), 400, good=True)
    assert env["svc"].check_health(T, act=True)[0]["status"] != "breach"
    assert not env["svc"].kill_switch().engaged(T)
