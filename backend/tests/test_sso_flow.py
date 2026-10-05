"""The browser sign-in's whole path without a browser: a stand-in provider issues real RS256 tokens, and the real Assay API
verifies them. This tests our client's assumptions and our server's checks against each other. It does not test any real
provider (see tests/mock_idp.py and docs/security/sso.md)."""

import base64
import hashlib
import re
from urllib.parse import parse_qs, urlparse

import pytest

pytest.importorskip("jwt")
from fastapi.testclient import TestClient

from assay.api import Credentials, create_app
from assay.auth import OidcClientConfig, OidcConfig, OidcVerifier, StaticJwks
from assay.ingestion import IngestionService, InMemoryRepository
from mock_idp import ROLE_MAP, make_idp

REDIRECT = "http://127.0.0.1:8001/"


def s256(verifier):
    return base64.urlsafe_b64encode(hashlib.sha256(verifier.encode()).digest()).rstrip(b"=").decode()


@pytest.fixture
def world():
    idp = make_idp()
    verifier = OidcVerifier(OidcConfig(issuer=idp.issuer, audience=idp.api_audience, role_map=ROLE_MAP), StaticJwks(idp.jwks))
    client_cfg = OidcClientConfig(client_id=idp.client_id, authorization_endpoint=f"{idp.issuer}/authorize", token_endpoint=f"{idp.issuer}/token",
                                  audience=idp.api_audience)
    assay = TestClient(create_app(IngestionService(InMemoryRepository()), Credentials([]), oidc=verifier, oidc_client=client_cfg,
                                  rate_limit_per_minute=None))
    return idp, TestClient(idp.app(), follow_redirects=False), assay


def sign_in(idp_client, idp, person, verifier="v" * 64, state="st", nonce="no"):
    """What the browser does: the authorize page, a choice, the redirect with a code."""
    page = idp_client.get("/authorize", params={"response_type": "code", "client_id": idp.client_id, "redirect_uri": REDIRECT, "state": state,
                                                "nonce": nonce, "code_challenge": s256(verifier), "code_challenge_method": "S256", "scope": "openid"})
    assert page.status_code == 200 and f'data-person="{person}"' in page.text
    href = re.search(rf'href="(/consent\?person={person}&[^"]+)"', page.text).group(1).replace("&amp;", "&")
    back = idp_client.get(href)
    assert back.status_code == 302 and back.headers["location"].startswith(REDIRECT)
    q = parse_qs(urlparse(back.headers["location"]).query)
    assert q["state"] == [state]
    return q["code"][0]


def exchange(idp_client, idp, code, verifier="v" * 64, redirect=REDIRECT):
    return idp_client.post("/token", data={"grant_type": "authorization_code", "code": code, "redirect_uri": redirect, "client_id": idp.client_id,
                                           "code_verifier": verifier})


def test_a_person_signs_in_and_the_api_knows_who_they_are_and_what_they_may_do(world):
    idp, idp_client, assay = world
    r = exchange(idp_client, idp, sign_in(idp_client, idp, "ada"))
    assert r.status_code == 200 and r.headers["access-control-allow-origin"] == "*"
    body = r.json()
    me = assay.get("/v1/me", headers={"Authorization": f"Bearer {body['access_token']}"}).json()
    assert me == {"subject": "sso-user-ada", "tenant": "tenant-synth", "roles": ["analyst"], "method": "sso"}
    assert assay.get("/v1/auth/config").json()["client_id"] == "assay-web"


def test_the_id_token_has_the_nonce_and_is_for_the_web_app_not_the_api(world):
    idp, idp_client, assay = world
    body = exchange(idp_client, idp, sign_in(idp_client, idp, "ada", nonce="my-nonce")).json()
    import jwt as pyjwt
    idc = pyjwt.decode(body["id_token"], options={"verify_signature": False})
    assert idc["nonce"] == "my-nonce" and idc["aud"] == "assay-web" and idc["name"] == "Ada"
    # sent to the API as a bearer it is refused: its audience is the web app, and the API checks the audience
    assert assay.get("/v1/me", headers={"Authorization": f"Bearer {body['id_token']}"}).status_code == 401


def test_each_person_gets_the_roles_their_groups_map_to(world):
    idp, idp_client, assay = world
    got = {}
    for who in ("ada", "morgan", "audrey", "ghost"):
        tok = exchange(idp_client, idp, sign_in(idp_client, idp, who)).json()["access_token"]
        got[who] = assay.get("/v1/me", headers={"Authorization": f"Bearer {tok}"}).json()["roles"]
    assert got == {"ada": ["analyst"], "morgan": ["approver"], "audrey": ["auditor"], "ghost": []}


def test_a_password_only_sign_in_is_refused_by_the_api_with_the_mfa_message(world):
    idp, idp_client, assay = world
    tok = exchange(idp_client, idp, sign_in(idp_client, idp, "nomfa")).json()["access_token"]
    r = assay.get("/v1/me", headers={"Authorization": f"Bearer {tok}"})
    assert r.status_code == 401 and r.json()["detail"]["code"] == "mfa_required"


def test_the_provider_refuses_a_wrong_verifier_a_reused_code_and_a_different_redirect(world):
    idp, idp_client, _ = world
    code = sign_in(idp_client, idp, "ada")
    bad = exchange(idp_client, idp, code, verifier="w" * 64)
    assert bad.status_code == 400 and "PKCE" in bad.json()["error_description"]
    assert "already used" in exchange(idp_client, idp, code).json()["error_description"]  # burnt by the failed attempt
    other = exchange(idp_client, idp, sign_in(idp_client, idp, "ada"), redirect="http://evil.example/")
    assert other.status_code == 400 and "redirect_uri" in other.json()["error_description"]
    ok = exchange(idp_client, idp, sign_in(idp_client, idp, "ada"))
    assert ok.status_code == 200
    assert exchange(idp_client, idp, sign_in(idp_client, idp, "ada"), verifier="").status_code == 400


def test_the_provider_refuses_a_request_that_is_not_the_code_flow_with_s256(world):
    idp, idp_client, _ = world
    base = {"response_type": "code", "client_id": idp.client_id, "redirect_uri": REDIRECT, "code_challenge": "x", "code_challenge_method": "S256"}
    assert idp_client.get("/authorize", params=base).status_code == 200
    for bad in ({"response_type": "token"}, {"client_id": "other"}, {"code_challenge_method": "plain"}, {"code_challenge": ""}):
        assert idp_client.get("/authorize", params={**base, **bad}).status_code == 400, bad


def test_a_token_from_another_provider_key_is_refused_by_the_api(world):
    _, _, assay = world
    other = make_idp()  # a different key, same issuer and audience
    forged = other.mint(sub="attacker", aud=other.api_audience, groups=["model-risk"], amr=["pwd", "mfa"])
    r = assay.get("/v1/me", headers={"Authorization": f"Bearer {forged}"})
    assert r.status_code == 401 and r.json()["detail"] == {"code": "unauthorized"}


def test_an_old_sign_in_is_refused_even_though_the_token_has_not_expired(world):
    idp, _, assay = world
    import time

    idp.lifetime_s = 86400  # exp is a day away, but the sign-in itself is older than the API allows
    stale = idp.mint(sub="user-ada", aud=idp.api_audience, groups=["fraud-analysts"], amr=["pwd", "mfa"], iat=time.time() - 7200)
    assert assay.get("/v1/me", headers={"Authorization": f"Bearer {stale}"}).status_code == 401
