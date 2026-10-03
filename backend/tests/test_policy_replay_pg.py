"""Policy replay on real PostgreSQL. Needs ASSAY_TEST_DATABASE_URL."""

from collections import Counter
from datetime import datetime, timedelta

import pytest

from assay.detection import load_bundle, save_bundle
from assay.ingestion import IngestionConfig, IngestionService
from assay.ingestion.pg_repo import PostgresRepository
from assay.policy import PolicyService
from assay.scoring import BundleRegistry, ScoringConfig, ScoringService
from assay.synthetic import to_wire

pytest.importorskip("psycopg")
T = "tenant-synth"


def test_replaying_the_policy_in_force_reproduces_what_scoring_stored_on_postgres(pg_conn, trained, tmp_path):
    _, txns, _, r = trained
    d = tmp_path / "b"
    save_bundle(d, r.scoring_bundle(), r.manifest, b"k")
    reg = BundleRegistry()
    reg.register(T, *load_bundle(d, T, b"k"))
    by_id, ids = {t["txn_id"]: t for t in txns}, r.test_table.txn_ids
    clock = {"t": None}
    repo = PostgresRepository(pg_conn)
    scoring = ScoringService(repo, reg, ScoringConfig(clock=lambda: clock["t"]))
    for i in range(0, 120, 12):
        t = to_wire(by_id[ids[i]])
        clock["t"] = datetime.fromisoformat(t["event_time"]) + timedelta(seconds=2)
        IngestionService(repo, IngestionConfig(clock=lambda: clock["t"])).ingest_transaction(
            T, t, source_id="s", actor="test")
        scoring.score_transaction(T, t["txn_id"], actor="test")

    stored = Counter(pd["recommended_action"] for pd in
                     {p["txn_id"]: p for p in repo.find(T, "policy_decisions")}.values())
    svc = PolicyService(repo, lambda: clock["t"])
    out = svc.preview(T, "u:admin", {})
    assert out["n_decisions"] == 10 and out["actions"]["before"] == dict(sorted(stored.items()))
    assert out["changed_decisions"] == 0

    stricter = svc.preview(T, "u:admin", {"always_review_above": 1})   # every amount is over 1
    assert stricter["actions"]["after"].get("approve", 0) == 0
    assert stricter["review_volume"]["after"] >= stricter["review_volume"]["before"]

    # Nothing a preview does is stored, except one audit entry each, and the chain still verifies.
    assert len(repo.find(T, "policy_versions")) == 0
    assert [a["action"] for a in repo.find(T, "audit_log") if a["action"] == "policy_preview"] == ["policy_preview"] * 2
    assert repo.verify(T, "audit_log") is None
    assert PolicyService(PostgresRepository(pg_conn), lambda: clock["t"]).preview("other-tenant", "u:x", {})["n_decisions"] == 0
