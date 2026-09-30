"""Assay API (PRD 26). The tenant comes from the caller's credential, never from a URL or body
parameter. Requests are signed: X-Assay-Signature = hex HMAC-SHA256(secret, body); GET requests
sign the empty body.

Roles (a first step toward FR-42): "ingest" submits transactions and outcomes; "analyst" reads
decisions and explanations; "auditor" reads lineage and runs replay. A credential may hold several.

If scoring cannot run (no champion bundle), the transaction is still stored and the caller gets 503
so its own fallback controls apply (PRD 18, 26.5). Retrying is safe: ingestion and scoring are
idempotent.
"""

from __future__ import annotations

import hashlib
import hmac
import json
from collections.abc import Callable
from contextlib import AbstractContextManager, contextmanager
from dataclasses import dataclass

from fastapi import Depends, FastAPI, Header, HTTPException, Request

from assay.ingestion import IngestionService
from assay.review.service import ReviewError, ReviewService
from assay.scoring import ScoringError, ScoringService

MAX_BATCH = 1000
DEFAULT_ROLES = frozenset({"ingest"})
# Roles that see blind cases in full (PRD 8.3): everyone except a plain analyst.
STAFF_ROLES = frozenset({"senior_analyst", "manager", "auditor", "approver"})


@dataclass(frozen=True)
class Credential:
    key_id: str
    tenant_id: str
    secret: bytes
    roles: frozenset[str] = DEFAULT_ROLES


class Credentials:
    """Dev/test credential store. Production keys come from the managed secret store."""

    def __init__(self, creds: list[Credential]):
        self._by_id = {c.key_id: c for c in creds}

    def get(self, key_id: str) -> Credential | None:
        return self._by_id.get(key_id)


def sign(secret: bytes, body: bytes) -> str:
    return hmac.new(secret, body, hashlib.sha256).hexdigest()


ServiceProvider = Callable[[], AbstractContextManager[IngestionService]]
ScoringProvider = Callable[[], AbstractContextManager[ScoringService]]
ReviewProvider = Callable[[], AbstractContextManager[ReviewService]]


def _as_provider(obj, cls):
    """Accept one service (tests, in-memory) or a provider that yields one per request."""
    if obj is None:
        return None
    if isinstance(obj, cls):
        @contextmanager
        def fixed():
            yield obj

        return fixed
    return obj


def create_app(service: IngestionService | ServiceProvider, credentials: Credentials, *,
               scoring: ScoringService | ScoringProvider | None = None,
               review: ReviewService | ReviewProvider | None = None) -> FastAPI:
    provider = _as_provider(service, IngestionService)
    scorer = _as_provider(scoring, ScoringService)
    reviewer = _as_provider(review, ReviewService)
    app = FastAPI(title="Assay API", version="0.1.0")

    @app.get("/healthz")
    def healthz():
        return {"status": "ok"}

    async def authed(request: Request, x_assay_key: str = Header(...),
                     x_assay_signature: str = Header(...)) -> tuple[Credential, bytes]:
        body = await request.body()
        cred = credentials.get(x_assay_key)
        # Same response for unknown key and bad signature: don't reveal which keys exist.
        if cred is None or not hmac.compare_digest(sign(cred.secret, body), x_assay_signature):
            raise HTTPException(401, {"code": "unauthorized"})
        return cred, body

    def need(cred: Credential, *roles: str) -> None:
        if not (cred.roles & set(roles)):
            raise HTTPException(403, {"code": "forbidden"})

    def parse(body: bytes):
        try:
            return json.loads(body)
        except ValueError:
            raise HTTPException(400, {"code": "invalid_json"}) from None

    def score(cred: Credential, txn_id: str) -> dict | None:
        if scorer is None:
            return None
        try:
            with scorer() as sc:
                return sc.score_transaction(cred.tenant_id, txn_id, actor=f"api:{cred.key_id}")
        except ScoringError as e:
            raise HTTPException(503, {"code": "scoring_unavailable", "detail": str(e)}) from None

    # -- ingestion + scoring ------------------------------------------------------------------
    @app.post("/v1/transactions", status_code=202)
    def post_transaction(auth=Depends(authed)):
        cred, body = auth
        need(cred, "ingest")
        payload = parse(body)
        if not isinstance(payload, dict):
            raise HTTPException(400, {"code": "expected_object"})
        with provider() as svc:
            r = svc.ingest_transaction(cred.tenant_id, payload, source_id=cred.key_id,
                                       actor=f"api:{cred.key_id}")
        if r.status == "rejected":
            raise HTTPException(422, {"code": "validation_failed", "reasons": list(r.reasons)})
        out = {"status": r.status, "event_id": r.event_id}
        decision = score(cred, str(payload["txn_id"]))
        if decision is not None:
            out["decision"] = decision
        return out

    @app.post("/v1/transactions:batch", status_code=207)
    def post_batch(auth=Depends(authed)):
        cred, body = auth
        need(cred, "ingest")
        payload = parse(body)
        if not isinstance(payload, list) or len(payload) > MAX_BATCH:
            raise HTTPException(400, {"code": "expected_list", "max": MAX_BATCH})
        results = []
        with provider() as svc:
            for item in payload:
                if not isinstance(item, dict):
                    results.append({"status": "rejected", "event_id": None,
                                    "reasons": ["_:not_object"]})
                    continue
                r = svc.ingest_transaction(cred.tenant_id, item, source_id=cred.key_id,
                                           actor=f"api:{cred.key_id}")
                results.append({"status": r.status, "event_id": r.event_id,
                                "reasons": list(r.reasons)})
        return {"results": results}

    @app.post("/v1/outcomes", status_code=202)
    def post_outcome(auth=Depends(authed)):
        cred, body = auth
        need(cred, "ingest")
        payload = parse(body)
        if not isinstance(payload, dict):
            raise HTTPException(400, {"code": "expected_object"})
        with provider() as svc:
            r = svc.ingest_outcome(cred.tenant_id, payload, source_id=cred.key_id,
                                   actor=f"api:{cred.key_id}")
        if r.status == "rejected":
            raise HTTPException(422, {"code": "validation_failed", "reasons": list(r.reasons)})
        return {"status": r.status, "event_id": r.event_id}

    # -- decisions ----------------------------------------------------------------------------
    def with_scoring(cred: Credential, fn):
        if scorer is None:
            raise HTTPException(404, {"code": "not_found"})
        try:
            with scorer() as sc:
                return fn(sc)
        except ScoringError as e:
            # Unknown ids, and ids belonging to another tenant, look identical (RLS hides them).
            raise HTTPException(404, {"code": "not_found", "detail": str(e)}) from None

    def blind_guard(cred: Credential, decision_id: str) -> None:
        """A plain analyst must not read a blind case's score by any route (PRD 8.3, FR-27)."""
        if reviewer is None or cred.roles & STAFF_ROLES or "analyst" not in cred.roles:
            return
        try:
            with reviewer() as rv:
                v = rv.case(cred.tenant_id, decision_id, viewer_role="analyst", viewer_pid=f"u:{cred.key_id}")
        except (ReviewError, ScoringError):
            return  # unknown decision: the handler below answers 404
        if v["redacted"]:
            raise HTTPException(403, {"code": "blind_review",
                                      "detail": "this case is under blind review until you record a decision"})

    @app.get("/v1/decisions/{decision_id}")
    def get_decision(decision_id: str, auth=Depends(authed)):
        cred, _ = auth
        need(cred, "analyst", "auditor")
        blind_guard(cred, decision_id)
        return with_scoring(cred, lambda sc: sc.get_decision(cred.tenant_id, decision_id))

    @app.get("/v1/decisions/{decision_id}/explanation")
    def get_explanation(decision_id: str, auth=Depends(authed)):
        cred, _ = auth
        need(cred, "analyst", "auditor")
        blind_guard(cred, decision_id)
        e = with_scoring(cred, lambda sc: sc.explanation(cred.tenant_id, decision_id))
        if e is None:
            raise HTTPException(404, {"code": "explanation_pending"})
        return e

    @app.get("/v1/decisions/{decision_id}/lineage")
    def get_lineage(decision_id: str, auth=Depends(authed)):
        cred, _ = auth
        need(cred, "auditor")
        return json.loads(json.dumps(
            with_scoring(cred, lambda sc: sc.lineage(cred.tenant_id, decision_id)), default=str))

    @app.post("/v1/decisions/{decision_id}/replay")
    def replay(decision_id: str, auth=Depends(authed)):
        cred, _ = auth
        need(cred, "auditor")
        return with_scoring(cred, lambda sc: sc.replay(cred.tenant_id, decision_id,
                                                        actor=f"api:{cred.key_id}"))

    # -- review workflow (PRD 10.5, FR-24 to FR-30) ---------------------------------------------------
    def viewer(cred: Credential) -> tuple[str, str]:
        """(role used for redaction and recording, pseudonymous analyst id)."""
        for r in ("senior_analyst", "manager", "auditor", "analyst"):
            if r in cred.roles:
                return r, f"u:{cred.key_id}"
        raise HTTPException(403, {"code": "forbidden"})

    def with_review(fn):
        if reviewer is None:
            raise HTTPException(404, {"code": "not_found"})
        try:
            with reviewer() as rv:
                return fn(rv)
        except ReviewError as e:
            raise HTTPException(e.http, {"code": e.code, "detail": str(e)}) from None
        except ScoringError as e:
            raise HTTPException(404, {"code": "not_found", "detail": str(e)}) from None

    @app.get("/v1/review/queue")
    def review_queue(risk_band: str | None = None, trust_state: str | None = None,
                     reason_code: str | None = None, search: str | None = None, queue: str | None = None,
                     include_closed: bool = False, limit: int = 200, auth=Depends(authed)):
        cred, _ = auth
        need(cred, "analyst", "senior_analyst", "manager")
        role, _pid = viewer(cred)
        return with_review(lambda rv: rv.queue(
            cred.tenant_id, viewer_role=role, risk_band=risk_band, trust_state=trust_state,
            reason_code=reason_code, search=search, queue=queue, include_closed=include_closed,
            limit=min(limit, 1000)))

    @app.get("/v1/review/cases/{decision_id}")
    def review_case(decision_id: str, auth=Depends(authed)):
        cred, _ = auth
        need(cred, "analyst", "senior_analyst", "manager", "auditor")
        role, pid = viewer(cred)
        return with_review(lambda rv: rv.case(cred.tenant_id, decision_id, viewer_role=role, viewer_pid=pid))

    @app.post("/v1/review/cases/{decision_id}/actions", status_code=201)
    def review_action(decision_id: str, auth=Depends(authed)):
        cred, body = auth
        need(cred, "analyst", "senior_analyst")
        p = parse(body)
        if not isinstance(p, dict):
            raise HTTPException(400, {"code": "expected_object"})
        role, pid = viewer(cred)
        allowed = {"action", "reason_code", "override_to", "confidence", "checklist", "notes",
                   "seconds_to_decision"}
        if set(p) - allowed:  # the display state is recorded by the server, never taken from the client
            raise HTTPException(422, {"code": "unknown_fields", "fields": sorted(set(p) - allowed)})
        if "action" not in p:
            raise HTTPException(422, {"code": "action_required"})
        return with_review(lambda rv: rv.record_action(
            cred.tenant_id, decision_id, analyst_pid=pid, role=role, **p))

    @app.post("/v1/review/cases/{decision_id}/adjudication", status_code=201)
    def review_adjudication(decision_id: str, auth=Depends(authed)):
        cred, body = auth
        need(cred, "senior_analyst")
        p = parse(body)
        if not isinstance(p, dict) or "final_decision" not in p:
            raise HTTPException(422, {"code": "final_decision_required"})
        role, pid = viewer(cred)
        return with_review(lambda rv: rv.adjudicate(
            cred.tenant_id, decision_id, analyst_pid=pid, role=role, final_decision=p["final_decision"],
            rationale=p.get("rationale", ""), notes=p.get("notes")))

    @app.get("/v1/dashboard/summary")
    def dashboard(window_days: int = 30, auth=Depends(authed)):
        cred, _ = auth
        need(cred, "manager")
        return with_review(lambda rv: rv.dashboard(cred.tenant_id, window_days=window_days))

    # -- governance reads (PRD 12, 26.2) -----------------------------------------------------------------
    @app.get("/v1/validation/reports")
    def validation_reports(limit: int = 5, auth=Depends(authed)):
        cred, _ = auth
        need(cred, "auditor", "approver", "manager")
        rows = with_scoring(cred, lambda sc: sc.validation_reports(cred.tenant_id, limit=min(limit, 50)))
        return json.loads(json.dumps(rows, default=str))

    @app.get("/v1/models/bundles")
    def model_bundles(auth=Depends(authed)):
        cred, _ = auth
        need(cred, "approver", "auditor", "manager")
        rows = with_scoring(cred, lambda sc: sc.model_bundles(cred.tenant_id))
        return json.loads(json.dumps(rows, default=str))
    return app
