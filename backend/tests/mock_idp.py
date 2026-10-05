"""A STAND-IN OpenID Connect provider, for testing the browser sign-in end to end. It is not a real provider and proves nothing
about any real one: real providers differ in where they put `amr`, groups and custom claims, in CORS rules, in how they treat
`audience`, and in many smaller ways (docs/security/sso.md).

What it does, faithfully enough to catch mistakes in our own client and server:
  * /authorize  shows a page listing the test people; choosing one returns a one-time code to the redirect address with `state`;
  * /token      exchanges the code, checking the redirect address and the PKCE verifier (S256) and refusing a code used twice,
                and returns an access token and an ID token signed RS256, with the claims Assay requires;
  * /jwks       the public key;
  * /logout     sends the browser back to the address it asks for.
Run it directly (`python tests/mock_idp.py`) or import `make_idp`.
"""

from __future__ import annotations

import base64
import hashlib
import secrets
import time
from html import escape
from urllib.parse import parse_qs, urlencode

import jwt
from cryptography.hazmat.primitives.asymmetric import rsa
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from jwt.algorithms import RSAAlgorithm

KID = "mock-idp-key-1"
# name -> (groups, authentication methods)
PEOPLE = {
    "ada": (["fraud-analysts"], ["pwd", "mfa"]),
    "morgan": (["model-risk"], ["pwd", "mfa"]),
    "audrey": (["fraud-auditors"], ["pwd", "mfa"]),
    "nomfa": (["fraud-analysts"], ["pwd"]),          # signed in with a password only
    "ghost": (["unrelated-group"], ["pwd", "mfa"]),  # no group Assay maps to a role
}
ROLE_MAP = {"fraud-analysts": "analyst", "fraud-auditors": "auditor", "model-risk": "approver"}


class MockIdp:
    def __init__(self, issuer: str, *, client_id: str = "assay-web", api_audience: str = "assay-api", tenant: str = "tenant-synth",
                 lifetime_s: int = 600, clock=time.time):
        self.issuer, self.client_id, self.api_audience, self.tenant = issuer.rstrip("/"), client_id, api_audience, tenant
        self.lifetime_s, self.clock = lifetime_s, clock
        self.key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
        self.jwks = {"keys": [{**__import__("json").loads(RSAAlgorithm.to_jwk(self.key.public_key())), "kid": KID, "alg": "RS256", "use": "sig"}]}
        self.codes: dict[str, dict] = {}
        self.issued: list[dict] = []

    def mint(self, *, sub: str, aud: str, groups: list[str], amr: list[str], nonce: str | None = None, name: str | None = None,
             iat: float | None = None) -> str:
        now = self.clock() if iat is None else iat
        claims = {"iss": self.issuer, "aud": aud, "sub": sub, "iat": int(now), "exp": int(now) + self.lifetime_s, "amr": amr,
                  "assay_tenant": self.tenant, "groups": groups}
        if nonce:
            claims["nonce"] = nonce
        if name:
            claims["name"] = name
        return jwt.encode(claims, self.key, algorithm="RS256", headers={"kid": KID})

    def app(self) -> FastAPI:
        app = FastAPI(title="Mock identity provider (testing only)")

        @app.get("/authorize", response_class=HTMLResponse)
        def authorize(request: Request, response_type: str = "", client_id: str = "", redirect_uri: str = "", state: str = "",
                      nonce: str = "", code_challenge: str = "", code_challenge_method: str = "", scope: str = ""):
            if response_type != "code" or client_id != self.client_id or not redirect_uri or code_challenge_method != "S256" or not code_challenge:
                return HTMLResponse("bad authorization request", 400)
            q = urlencode({"redirect_uri": redirect_uri, "state": state, "nonce": nonce, "code_challenge": code_challenge})
            links = "".join(f'<li><a data-person="{n}" href="/consent?person={n}&{escape(q)}">Sign in as {n}</a></li>' for n in PEOPLE)
            deny = f'<a data-person="deny" href="{escape(redirect_uri)}?{urlencode({"error": "access_denied", "error_description": "User cancelled", "state": state})}">Cancel</a>'
            return HTMLResponse(f"<!doctype html><title>Mock IdP</title><h1>Mock identity provider</h1><p>Testing only.</p><ul>{links}</ul>{deny}")

        @app.get("/consent")
        def consent(person: str, redirect_uri: str, state: str = "", nonce: str = "", code_challenge: str = ""):
            if person not in PEOPLE:
                return HTMLResponse("unknown person", 400)
            code = secrets.token_urlsafe(24)
            self.codes[code] = {"person": person, "redirect_uri": redirect_uri, "nonce": nonce, "challenge": code_challenge,
                                "issued": self.clock(), "used": False}
            return RedirectResponse(f"{redirect_uri}?{urlencode({'code': code, 'state': state})}", 302)

        @app.post("/token")
        async def token(request: Request):
            # parsed by hand: FastAPI's Form needs python-multipart, which Assay does not otherwise depend on
            form = {k: v[0] for k, v in parse_qs((await request.body()).decode()).items()}
            grant_type, code, redirect_uri = form.get("grant_type", ""), form.get("code", ""), form.get("redirect_uri", "")
            client_id, code_verifier = form.get("client_id", ""), form.get("code_verifier", "")
            cors = {"Access-Control-Allow-Origin": "*"}

            def refuse(err: str, desc: str):
                return JSONResponse({"error": err, "error_description": desc}, 400, headers=cors)

            rec = self.codes.get(code)
            if grant_type != "authorization_code" or rec is None or client_id != self.client_id:
                return refuse("invalid_grant", "unknown code")
            if rec["used"]:
                return refuse("invalid_grant", "code already used")
            rec["used"] = True  # a code is single-use, whatever happens next
            if rec["redirect_uri"] != redirect_uri:
                return refuse("invalid_grant", "redirect_uri does not match")
            digest = base64.urlsafe_b64encode(hashlib.sha256(code_verifier.encode()).digest()).rstrip(b"=").decode()
            if not code_verifier or digest != rec["challenge"]:
                return refuse("invalid_grant", "PKCE verification failed")
            groups, amr = PEOPLE[rec["person"]]
            out = {"access_token": self.mint(sub=f"user-{rec['person']}", aud=self.api_audience, groups=groups, amr=amr),
                   "id_token": self.mint(sub=f"user-{rec['person']}", aud=self.client_id, groups=groups, amr=amr, nonce=rec["nonce"],
                                         name=rec["person"].capitalize()),
                   "token_type": "Bearer", "expires_in": self.lifetime_s}
            self.issued.append({"person": rec["person"]})
            return JSONResponse(out, headers=cors)

        @app.get("/jwks")
        def jwks():
            return self.jwks

        @app.get("/logout")
        def logout(post_logout_redirect_uri: str = "/", client_id: str = ""):
            return RedirectResponse(post_logout_redirect_uri, 302)

        return app


def make_idp(port: int = 8100, **kw) -> MockIdp:
    return MockIdp(f"http://127.0.0.1:{port}", **kw)


if __name__ == "__main__":
    import uvicorn

    idp = make_idp()
    uvicorn.run(idp.app(), host="127.0.0.1", port=8100, log_level="warning")
