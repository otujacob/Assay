"""DEMO server: the real Assay services on synthetic data, plus a dev-only signing proxy.

    python scripts/demo_server.py            # http://127.0.0.1:8000  (serves web/dist if built)

What it does: trains a bundle on synthetic data, replays a stream of synthetic transactions through
the real ingestion, scoring and review services, records a few analyst actions, and loads the
stored validation reports from docs/validation. Everything is in memory and is lost on restart.

!! DEV ONLY. Real callers sign requests with an HMAC secret, which a browser must never hold.
!! This server adds a proxy at /api/* that signs requests on behalf of a demo user chosen by the
!! X-Demo-User header. It has no authentication and exists only so the UI can be shown. Production
!! needs single sign-on through the institution's identity provider (PRD 16, FR-42).
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path

import httpx
import uvicorn
from fastapi import FastAPI, Request, Response
from fastapi.staticfiles import StaticFiles

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tests"))

from assay.api import Credential, Credentials, create_app, sign
from assay.detection import train_bundle
from assay.ingestion import IngestionConfig, IngestionService, InMemoryRepository
from assay.review.service import REVIEW_ACTIONS, ReviewConfig, ReviewError, ReviewService
from assay.scoring import BundleRegistry, ScoringConfig, ScoringService
from assay.synthetic import to_wire
from assay.validation.report import store_report
from conftest_detection import make_dataset

TENANT = "tenant-synth"
DEMO_USERS = {
    "analyst": {"roles": {"analyst"}, "label": "Analyst"},
    "analyst2": {"roles": {"analyst"}, "label": "Analyst (2nd)"},
    "senior": {"roles": {"senior_analyst", "analyst"}, "label": "Senior Analyst"},
    "manager": {"roles": {"manager"}, "label": "Fraud Manager"},
    "auditor": {"roles": {"auditor"}, "label": "Auditor (read-only)"},
    "approver": {"roles": {"approver"}, "label": "Model Risk Approver"},
    "ingest": {"roles": {"ingest"}, "label": "Ingestion system"},
}
SECRETS = {k: f"demo-{k}-secret".encode() for k in DEMO_USERS}


def build_world(hours: int = 36, log=print):
    log("generating synthetic data and training the bundle (about 15 s)...")
    ds, txns, cfg = make_dataset()
    r = train_bundle(txns, ds.outcomes, cfg)
    by_id = {t["txn_id"]: t for t in txns}

    ids = r.test_table.txn_ids
    last = max(datetime.fromisoformat(by_id[t]["event_time"]) for t in ids)
    now = datetime.now(UTC)
    shift = now - (last + timedelta(minutes=10))  # move the whole stream to "now"; features are unchanged
    score_from = last - timedelta(hours=hours)
    ingest_from = score_from - timedelta(days=31)  # the 30-day history the features need

    clock = {"fn": lambda: START_CLOCK[0]}
    START_CLOCK = [now]
    repo = InMemoryRepository()
    reg = BundleRegistry()
    reg.register(TENANT, r.scoring_bundle(), r.manifest)
    ing = IngestionService(repo, IngestionConfig(clock=lambda: clock["fn"]()))
    scoring = ScoringService(repo, reg, ScoringConfig(clock=lambda: clock["fn"]()))
    review = ReviewService(repo, scoring, ReviewConfig(clock=lambda: clock["fn"](), blind_share=0.15))

    stream = sorted((t for t in txns if ingest_from <= datetime.fromisoformat(t["event_time"]) <= last),
                    key=lambda t: t["event_time"])
    log(f"replaying {len(stream)} transactions ({hours}h scored) through the real services...")
    scored = 0
    for t in stream:
        w = to_wire(t)
        ev = datetime.fromisoformat(w["event_time"])
        w["event_time"] = (ev + shift).isoformat().replace("+00:00", "Z")
        START_CLOCK[0] = ev + shift + timedelta(seconds=2)
        ing.ingest_transaction(TENANT, w)
        if ev >= score_from:
            scoring.score_transaction(TENANT, w["txn_id"])
            scored += 1
    START_CLOCK[0] = now
    log(f"scored {scored}; running the asynchronous explanation worker (PRD 5.8)...")
    refined = scoring.refine_pending(TENANT, limit=10_000)
    log(f"refined {refined} decisions; {len(repo.rows(TENANT, 'review_cases'))} cases were ever queued")

    # A little analyst activity so the dashboard and feedback panels have content.
    acted = 0
    for c in repo.rows(TENANT, "review_cases"):
        if acted >= 8 or c["blind"]:
            continue
        pd = repo.find(TENANT, "policy_decisions", {"txn_id": c["txn_id"]}, newest_first=True, limit=1)[0]
        if pd["recommended_action"] not in REVIEW_ACTIONS:
            continue
        try:
            review.record_action(TENANT, pd["id"], analyst_pid=f"u:demo-analyst{acted % 2}", role="analyst",
                                 action="unsure" if acted % 3 == 0 else "request_review",
                                 confidence=0.6, checklist={"kyc_checked": True, "customer_contacted": acted % 2 == 0})
            acted += 1
        except ReviewError as e:  # a case that cannot take this demo action is simply skipped
            log(f"skipped demo action on {c['txn_id']}: {e.code}")

    # Stored validation evidence from the recorded runs (synthetic).
    for name in ("seed-99-fresh",):
        p = ROOT.parent / "docs" / "validation" / name
        if (p / "report.json").exists():
            stress = json.loads((p / "stress.json").read_text()) if (p / "stress.json").exists() else None
            store_report(repo, TENANT, r.manifest, json.loads((p / "report.json").read_text()), stress)
    return repo, scoring, review, ing, r, clock


def build_app(log=print) -> FastAPI:
    _repo, scoring, review, ing, _r, _clock = build_world(log=log)
    creds = Credentials([Credential(k, TENANT, SECRETS[k], frozenset(v["roles"])) for k, v in DEMO_USERS.items()])
    inner = create_app(ing, creds, scoring=scoring, review=review)
    transport = httpx.ASGITransport(app=inner)

    app = FastAPI(title="Assay DEMO (dev-only signing proxy)")

    @app.get("/demo/users")
    def users():
        return [{"key": k, "label": v["label"], "roles": sorted(v["roles"])} for k, v in DEMO_USERS.items()
                if k != "ingest"]

    @app.api_route("/api/{path:path}", methods=["GET", "POST"])
    async def proxy(path: str, request: Request):
        user = request.headers.get("x-demo-user", "analyst")
        if user not in DEMO_USERS:
            return Response(json.dumps({"detail": {"code": "unknown_demo_user"}}), 400, media_type="application/json")
        body = await request.body()
        async with httpx.AsyncClient(transport=transport, base_url="http://inner") as c:
            resp = await c.request(request.method, f"/v1/{path}", params=dict(request.query_params),
                                   content=body, headers={"X-Assay-Key": user,
                                                          "X-Assay-Signature": sign(SECRETS[user], body),
                                                          "Content-Type": "application/json"})
        return Response(resp.content, resp.status_code, media_type="application/json")

    dist = ROOT.parent / "web" / "dist"
    if dist.exists():
        app.mount("/", StaticFiles(directory=dist, html=True), name="web")
    return app


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    a = ap.parse_args()
    print("DEMO server: synthetic data, dev-only signing proxy, NOT for production.")
    uvicorn.run(build_app(), host=a.host, port=a.port, log_level="warning")


if __name__ == "__main__":
    main()
