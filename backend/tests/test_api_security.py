"""FR-43 access events in the audit log, and API rate limiting (PRD 16)."""

import json

import pytest
from fastapi.testclient import TestClient

from assay.api import Credential, Credentials, create_app, sign
from assay.api.limits import RateLimiter
from assay.ingestion import IngestionService, InMemoryRepository

T = "tenant-a"
SEC = {"ing": b"ing-secret", "ana": b"ana-secret"}


class Clock:
    t = 1000.0

    def __call__(self):
        return self.t


@pytest.fixture
def env():
    repo, clock = InMemoryRepository(), Clock()
    creds = Credentials([Credential("ing", T, SEC["ing"], frozenset({"ingest"})),
                         Credential("ana", T, SEC["ana"], frozenset({"analyst"}))])

    def make(limit=600):
        return TestClient(create_app(IngestionService(repo), creds, rate_limit_per_minute=limit, clock=clock))

    def call(client, key, path="/v1/transactions", method="POST", body=b"{}", sig=None):
        return client.request(method, path, content=body, headers={
            "X-Assay-Key": key, "X-Assay-Signature": sig if sig is not None else sign(SEC.get(key, b"x"), body)})

    def audit_rows(*actions):
        return [r for r in repo.rows(T, "audit_log") if r["action"] in actions]

    return {"make": make, "call": call, "rows": audit_rows, "clock": clock, "repo": repo}


def test_a_bad_signature_is_audited_against_the_key_owner(env):
    c = env["make"]()
    assert env["call"](c, "ing", sig="0" * 64).status_code == 401
    (row,) = env["rows"]("auth_failed")
    assert (row["actor"], row["object"], row["result"]) == ("api:ing", "/v1/transactions", "bad_signature")


def test_an_unknown_key_cannot_be_attributed_so_it_writes_nothing(env):
    c = env["make"]()
    assert env["call"](c, "nobody").status_code == 401
    assert env["rows"]("auth_failed") == []  # no tenant to file it under, and no way to flood anyone's trail


def test_failed_sign_ins_cannot_flood_the_audit_log(env):
    c = env["make"]()
    for _ in range(50):
        assert env["call"](c, "ing", sig="0" * 64).status_code == 401
    assert len(env["rows"]("auth_failed")) == 5          # the cap; the rest went to the application log
    env["clock"].t += 61
    env["call"](c, "ing", sig="0" * 64)
    assert len(env["rows"]("auth_failed")) == 6          # a new minute, a new allowance


def test_a_refused_role_is_audited_with_the_route_and_roles_needed(env):
    c = env["make"]()
    assert env["call"](c, "ana").status_code == 403       # an analyst cannot ingest
    (row,) = env["rows"]("access_denied")
    assert (row["actor"], row["object"], row["result"]) == ("api:ana", "/v1/transactions", "ingest")


def test_a_successful_call_is_not_logged_as_a_denial(env):
    c = env["make"]()
    env["call"](c, "ing", body=json.dumps({"txn_id": "x"}).encode())
    assert env["rows"]("access_denied", "auth_failed") == []


def test_the_rate_limit_returns_429_with_retry_after_then_recovers(env):
    c = env["make"](limit=3)
    assert [env["call"](c, "ing").status_code for _ in range(3)] == [422, 422, 422]  # reached the handler
    r = env["call"](c, "ing")
    assert r.status_code == 429 and r.json()["detail"]["code"] == "rate_limited"
    assert 1 <= int(r.headers["Retry-After"]) <= 60
    env["clock"].t += 61
    assert env["call"](c, "ing").status_code == 422


def test_limits_are_per_key_and_a_bad_signature_does_not_spend_the_owners_allowance(env):
    c = env["make"](limit=2)
    for _ in range(5):
        env["call"](c, "ing", sig="0" * 64)               # an attacker who knows the key id
    assert env["call"](c, "ing").status_code == 422        # the real client is not locked out
    assert env["call"](c, "ana").status_code == 403        # another key has its own window


def test_rate_limiting_can_be_turned_off(env):
    c = env["make"](limit=None)
    assert all(env["call"](c, "ing").status_code == 422 for _ in range(50))


def test_an_audit_write_failure_never_becomes_a_500(env):
    c = env["make"]()
    env["repo"].append_audit = lambda *a, **k: (_ for _ in ()).throw(RuntimeError("db down"))
    assert env["call"](c, "ing", sig="0" * 64).status_code == 401
    assert env["call"](c, "ana").status_code == 403


def test_the_limiter_itself():
    t = [0.0]
    lim = RateLimiter(2, window_s=10, clock=lambda: t[0])
    assert lim.check("k") is None and lim.check("k") is None
    assert lim.check("k") == 10                            # blocked until the oldest hit ages out
    t[0] = 4
    assert lim.check("k") == 6                             # retrying does not extend the wait
    t[0] = 10
    assert lim.check("k") is None
    with pytest.raises(ValueError):
        RateLimiter(0)
