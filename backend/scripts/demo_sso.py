"""DEMO of browser single sign-on, against a STAND-IN identity provider.

    python scripts/demo_sso.py        # the app at http://127.0.0.1:8001, the stand-in provider at http://127.0.0.1:8100

The real Assay API, with single sign-on switched on, serves the web app. People sign in at a stand-in provider (tests/mock_idp.py)
that issues genuine RS256 tokens, and the API verifies them exactly as it would a real provider's: signature, issuer, audience,
lifetime, MFA, tenant and mapped roles. Choose who to sign in as on the provider's page:
    ada      an analyst                       audrey   an auditor
    morgan   a model-risk approver            nomfa    an analyst who signed in with a password only (refused: MFA)
    ghost    signed in, but no group maps to an Assay role (no access)

THIS PROVES OUR OWN CLIENT AND SERVER AGREE. It does not prove anything about a real provider, which will differ in claim names,
CORS rules, token lifetimes and how it treats `audience` (docs/security/sso.md). The data is synthetic and in memory.
"""

from __future__ import annotations

import argparse
import sys
import threading
import time
from pathlib import Path

import httpx
import uvicorn
from fastapi import FastAPI, Request, Response
from fastapi.staticfiles import StaticFiles

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tests"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from demo_server import TENANT, build_world

from assay.api import Credential, Credentials, create_app
from assay.auth import OidcClientConfig, OidcConfig, OidcVerifier, StaticJwks
from mock_idp import ROLE_MAP, make_idp


def build_app(idp, *, log=print) -> FastAPI:
    _repo, scoring, review, ing, _r, _clock, _extras = build_world(log=log)
    verifier = OidcVerifier(OidcConfig(issuer=idp.issuer, audience=idp.api_audience, role_map=ROLE_MAP), StaticJwks(idp.jwks))
    client = OidcClientConfig(client_id=idp.client_id, authorization_endpoint=f"{idp.issuer}/authorize", token_endpoint=f"{idp.issuer}/token",
                              end_session_endpoint=f"{idp.issuer}/logout", audience=idp.api_audience)
    inner = create_app(ing, Credentials([Credential("ingest", TENANT, b"demo-ingest-secret", frozenset({"ingest"}))]),
                       scoring=scoring, review=review, oidc=verifier, oidc_client=client)
    transport = httpx.ASGITransport(app=inner)
    app = FastAPI(title="Assay DEMO with single sign-on (stand-in provider)")

    @app.api_route("/api/{path:path}", methods=["GET", "POST"])
    async def forward(path: str, request: Request):
        """Maps /api/* to /v1/*, as a reverse proxy would. It signs nothing: the person's own bearer token goes through."""
        headers = {k: v for k, v in request.headers.items() if k.lower() in ("authorization", "content-type")}
        async with httpx.AsyncClient(transport=transport, base_url="http://inner") as c:
            resp = await c.request(request.method, f"/v1/{path}", params=dict(request.query_params), content=await request.body(), headers=headers)
        keep = {k: v for k, v in resp.headers.items() if k.lower() in ("content-disposition", "www-authenticate")}
        return Response(resp.content, resp.status_code, headers=keep, media_type=resp.headers.get("content-type", "application/json"))

    dist = ROOT.parent / "web" / "dist"
    if dist.exists():
        app.mount("/", StaticFiles(directory=dist, html=True), name="web")
    return app


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8001)
    ap.add_argument("--idp-port", type=int, default=8100)
    a = ap.parse_args()
    idp = make_idp(a.idp_port)
    threading.Thread(target=lambda: uvicorn.run(idp.app(), host="127.0.0.1", port=a.idp_port, log_level="warning"), daemon=True).start()
    time.sleep(0.5)
    print("DEMO with single sign-on: a STAND-IN provider, synthetic data, NOT for production.")
    app = build_app(idp)
    print(f"Open http://127.0.0.1:{a.port} (the stand-in provider is at http://127.0.0.1:{a.idp_port}).")
    uvicorn.run(app, host="127.0.0.1", port=a.port, log_level="warning")


if __name__ == "__main__":
    main()
