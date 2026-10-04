"""GET /v1/decisions/{id}/counterfactuals: access rules, blind review, determinism, and what it returns."""

import json
from datetime import datetime, timedelta

import numpy as np
import pytest
from fastapi.testclient import TestClient

from assay.api import Credential, Credentials, create_app, sign
from assay.detection import load_bundle, save_bundle
from assay.ingestion import IngestionConfig, IngestionService, InMemoryRepository
from assay.policy import REVIEW_ACTIONS
from assay.review.service import ReviewConfig, ReviewService
from assay.scoring import BundleRegistry, ScoringConfig, ScoringService
from assay.synthetic import START, to_wire

T = "tenant-synth"
SEC = {k: k.encode() + b"-secret" for k in ("ing", "ana", "sen", "mgr", "aud", "oth")}
ROLES = {"ing": {"ingest"}, "ana": {"analyst"}, "sen": {"senior_analyst", "analyst"}, "mgr": {"manager"},
         "aud": {"auditor"}}
IMMUTABLE = ("velocity_1h", "velocity_24h", "velocity_30d", "missing_fields")


@pytest.fixture(scope="module")
def artefact(trained, tmp_path_factory):
    d = tmp_path_factory.mktemp("cfapi") / "b"
    save_bundle(d, trained[3].scoring_bundle(), trained[3].manifest, b"k")
    return load_bundle(d, T, b"k")


def build(trained, artefact, blind_share):
    _, txns, _, r = trained
    repo, clock = InMemoryRepository(), {"t": START}
    reg = BundleRegistry()
    reg.register(T, *artefact)
    ing = IngestionService(repo, IngestionConfig(clock=lambda: clock["t"]))
    scoring = ScoringService(repo, reg, ScoringConfig(clock=lambda: clock["t"]))
    review = ReviewService(repo, scoring, ReviewConfig(clock=lambda: clock["t"], blind_share=blind_share))
    creds = Credentials([Credential(k, T, SEC[k], frozenset(v)) for k, v in ROLES.items()]
                        + [Credential("oth", "tenant-other", SEC["oth"], frozenset({"analyst", "auditor"}))])
    client = TestClient(create_app(ing, creds, scoring=scoring, review=review, rate_limit_per_minute=None))
    by_id, ids = {t["txn_id"]: t for t in txns}, r.test_table.txn_ids
    risk = r.model.predict(r.test_table.X).calibrated
    t_high = r.manifest.thresholds["t_high"]

    def call(method, path, key, payload=None):
        body = b"" if payload is None else json.dumps(payload).encode()
        return client.request(method, path, content=body, headers={
            "X-Assay-Key": key, "X-Assay-Signature": sign(SEC[key], body)})

    def submit(i):
        t = to_wire(by_id[ids[i]])
        clock["t"] = datetime.fromisoformat(t["event_time"]) + timedelta(seconds=2)
        return call("POST", "/v1/transactions", "ing", t).json()["decision"]

    def flagged():
        """A decision the model actually flagged WHEN SCORED HERE. Scoring one ingested transaction recomputes its
        features from the history in this store (sparse), so it can score lower than the same row in the
        training table, and the candidate is chosen by what scoring returned."""
        for i in np.flatnonzero(risk >= t_high)[:60]:
            d = submit(int(i))
            if d["risk"] >= t_high:
                return d
        raise AssertionError("no flagged case when scored through the API")

    def blind():
        for i in range(0, 600, 12):
            d = submit(i)
            if d["recommendation"] in REVIEW_ACTIONS and repo.find(T, "review_cases", {"txn_id": d["txn_id"]})[0]["blind"]:
                return d
        raise AssertionError("no blind case")

    return {"call": call, "submit": submit, "flagged": flagged, "blind": blind, "repo": repo, "t_high": t_high}


@pytest.fixture
def api(trained, artefact):
    """No case is under blind review, so access rules other than blindness can be tested alone."""
    return build(trained, artefact, blind_share=0.0)


@pytest.fixture
def blind_api(trained, artefact):
    """Every case that goes to review is blind."""
    return build(trained, artefact, blind_share=1.0)


def test_an_analyst_and_an_auditor_can_read_but_other_roles_cannot(api):
    d = api["flagged"]()
    path = f"/v1/decisions/{d['decision_id']}/counterfactuals"
    assert api["call"]("GET", path, "ana").status_code == 200
    assert api["call"]("GET", path, "aud").status_code == 200
    assert api["call"]("GET", path, "ing").status_code == 403
    assert api["call"]("GET", path, "mgr").status_code == 403


def test_it_returns_valid_changes_in_words_with_the_caveat_and_both_explanation_views(api):
    d = api["flagged"]()
    out = api["call"]("GET", f"/v1/decisions/{d['decision_id']}/counterfactuals", "ana").json()
    assert out["flagged"] is True and out["risk"] >= out["threshold"] == api["t_high"]
    assert "not cause and effect" in out["note"] and "must not be given to a customer" in out["note"]
    cfs = out["counterfactuals"]
    assert cfs and out["summary"]["found"] == len(cfs)
    for c in cfs:
        assert c["changes"] and all(ch["text"] for ch in c["changes"])
        assert c["risk_before"] >= api["t_high"] > c["risk_after"] or not c["valid"]
        if c["valid"]:
            assert not c["failed_checks"] and c["robust_share"] >= 0.8
    cm = out["cross_method"]
    assert 0 <= cm["agreement"] <= 1 and cm["shap_drivers"] and cm["permutation_drivers"]


def test_history_and_completeness_are_never_among_the_changes(api):
    d = api["flagged"]()
    out = api["call"]("GET", f"/v1/decisions/{d['decision_id']}/counterfactuals", "aud").json()
    for c in out["counterfactuals"]:
        assert not {ch["feature"] for ch in c["changes"]} & set(IMMUTABLE)


def test_the_amounts_derived_features_are_not_listed_as_changes_of_their_own(api):
    d = api["flagged"]()
    out = api["call"]("GET", f"/v1/decisions/{d['decision_id']}/counterfactuals", "aud").json()
    for c in out["counterfactuals"]:
        feats = {ch["feature"] for ch in c["changes"]}
        if "amount" in feats:
            assert not feats & {"log_amount", "amount_ratio_baseline"}


def test_the_same_decision_gives_the_same_answer_every_time(api):
    d = api["flagged"]()
    path = f"/v1/decisions/{d['decision_id']}/counterfactuals"
    assert api["call"]("GET", path, "ana").json() == api["call"]("GET", path, "ana").json()


def test_a_case_that_is_not_flagged_is_explained_in_the_other_direction(api):
    d = api["submit"](1)                                       # an ordinary, unflagged case
    out = api["call"]("GET", f"/v1/decisions/{d['decision_id']}/counterfactuals", "ana").json()
    assert out["flagged"] is False
    for c in out["counterfactuals"]:
        if c["valid"]:
            assert c["risk_before"] < api["t_high"] <= c["risk_after"]


def test_a_blind_case_stays_hidden_from_a_plain_analyst_and_open_to_staff(blind_api):
    d = blind_api["blind"]()
    path = f"/v1/decisions/{d['decision_id']}/counterfactuals"
    r = blind_api["call"]("GET", path, "ana")
    assert r.status_code == 403 and r.json()["detail"]["code"] == "blind_review"
    assert blind_api["call"]("GET", path, "sen").status_code == 200
    assert blind_api["call"]("GET", path, "aud").status_code == 200


def test_an_unknown_decision_and_another_tenants_decision_are_both_not_found(api):
    assert api["call"]("GET", "/v1/decisions/no-such-id/counterfactuals", "ana").status_code == 404
    d = api["flagged"]()
    r = api["call"]("GET", f"/v1/decisions/{d['decision_id']}/counterfactuals", "oth")
    assert r.status_code == 404 and d["decision_id"] not in r.text


def test_reading_counterfactuals_is_audited(api):
    d = api["flagged"]()
    api["call"]("GET", f"/v1/decisions/{d['decision_id']}/counterfactuals", "ana")
    rows = [r for r in api["repo"].rows(T, "audit_log") if r["action"] == "counterfactuals_read"]
    assert len(rows) == 1 and rows[0]["actor"] == "api:ana" and rows[0]["object"] == d["txn_id"]
