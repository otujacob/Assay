"""FR-34 audit view and export, and FR-43 (reading the audit log is itself logged)."""

import csv
import io
import json
from datetime import datetime, timedelta
from urllib.parse import quote

import pytest
from fastapi.testclient import TestClient

from assay import audit
from assay.api import Credential, Credentials, create_app, sign
from assay.detection import load_bundle, save_bundle
from assay.ingestion import IngestionConfig, IngestionService, InMemoryRepository
from assay.scoring import BundleRegistry, ScoringConfig, ScoringService
from assay.synthetic import START, to_wire

T = "tenant-synth"
SEC = {k: k.encode() + b"-secret" for k in ("ing", "ana", "aud")}
ROLES = {"ing": {"ingest"}, "ana": {"analyst"}, "aud": {"auditor"}}


@pytest.fixture(scope="module")
def artefact(trained, tmp_path_factory):
    d = tmp_path_factory.mktemp("auditapi") / "b"
    save_bundle(d, trained[3].scoring_bundle(), trained[3].manifest, b"k")
    return load_bundle(d, T, b"k")


@pytest.fixture
def api(trained, artefact):
    _, txns, _, r = trained
    repo = InMemoryRepository()
    clock = {"t": START}
    reg = BundleRegistry()
    reg.register(T, *artefact)
    ing = IngestionService(repo, IngestionConfig(clock=lambda: clock["t"]))
    scoring = ScoringService(repo, reg, ScoringConfig(clock=lambda: clock["t"]))
    creds = Credentials([Credential(k, T, SEC[k], frozenset(v)) for k, v in ROLES.items()])
    client = TestClient(create_app(ing, creds, scoring=scoring))
    by_id = {t["txn_id"]: t for t in txns}
    ids = r.test_table.txn_ids

    def call(path, key, method="GET", payload=None):
        body = b"" if payload is None else json.dumps(payload).encode()
        return client.request(method, path, content=body, headers={
            "X-Assay-Key": key, "X-Assay-Signature": sign(SEC[key], body)})

    def submit(i):
        t = to_wire(by_id[ids[i]])
        clock["t"] = datetime.fromisoformat(t["event_time"]) + timedelta(seconds=2)
        return call("/v1/transactions", "ing", "POST", t).json()["decision"]

    return {"call": call, "submit": submit, "repo": repo, "reg": reg, "clock": clock}


def test_only_auditors_can_read_the_audit_log(api):
    api["submit"](0)
    assert api["call"]("/v1/audit/export", "aud").status_code == 200
    assert api["call"]("/v1/audit/export", "ana").status_code == 403
    assert api["call"]("/v1/audit/export", "ing").status_code == 403


def test_filters_by_transaction_actor_action_and_date(api):
    d0, d1 = api["submit"](0), api["submit"](12)
    out = api["call"]("/v1/audit/export", "aud").json()
    assert out["matching"] >= 4 and out["chain_ok"] is True

    by_txn = api["call"](f"/v1/audit/export?txn_id={d0['txn_id']}", "aud").json()["items"]
    assert by_txn and {i["action"] for i in by_txn} >= {"ingest_transaction", "score"}
    assert all(i["object"] != d1["txn_id"] for i in by_txn)

    assert {i["action"] for i in api["call"]("/v1/audit/export?action=score", "aud").json()["items"]} == {"score"}
    assert api["call"]("/v1/audit/export?actor=nobody", "aud").json()["items"] == []
    future = (api["clock"]["t"] + timedelta(days=1)).isoformat()
    assert api["call"](f"/v1/audit/export?since={quote(future)}", "aud").json()["items"] == []
    assert api["call"]("/v1/audit/export?since=not-a-date", "aud").status_code == 422


def test_filters_by_model_version(api):
    d = api["submit"](0)
    items = api["call"](f"/v1/audit/export?model_version={d['bundle_id']}", "aud").json()["items"]
    assert any(i["action"] == "score" and i["object"] == d["txn_id"] for i in items)
    assert api["call"]("/v1/audit/export?model_version=b-other", "aud").json()["items"] == []


def test_csv_export_and_formula_injection_guard(api):
    api["submit"](0)
    r = api["call"]("/v1/audit/export?format=csv", "aud")
    assert r.status_code == 200 and r.headers["content-type"].startswith("text/csv")
    rows = list(csv.DictReader(io.StringIO(r.text)))
    assert rows and set(rows[0]) == set(audit.FIELDS)
    assert api["call"]("/v1/audit/export?format=xml", "aud").status_code == 422
    # A hostile object id must not become a spreadsheet formula.
    cell = lambda v: audit.to_csv([{"object": v, "seq": 1}]).splitlines()[1].split(",")[4]
    assert cell("=HYPERLINK(1)") == "'=HYPERLINK(1)" and cell("-1+1") == "'-1+1"
    assert cell("-") == "-" and cell("t-1") == "t-1"  # the no-object placeholder and ordinary ids are untouched


def test_reading_the_audit_log_is_audited(api):
    api["submit"](0)
    api["call"]("/v1/audit/export", "aud")
    items = api["call"]("/v1/audit/export?action=audit_read", "aud").json()["items"]
    assert items and items[0]["actor"] == "api:aud"


def test_limit_caps_items_but_not_the_match_count(api):
    for i in (0, 12, 24):
        api["submit"](i)
    out = api["call"]("/v1/audit/export?limit=2", "aud").json()
    assert len(out["items"]) == 2 and out["matching"] > 2
