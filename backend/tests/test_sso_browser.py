"""The server side of the web app's browser sign-in (FR-42): the public settings it needs, and /v1/me. Tokens are generated in
the test suite (see test_sso.py); no identity provider is contacted."""

import json

import pytest

pytest.importorskip("jwt")
from fastapi.testclient import TestClient

from assay.api import Credential, Credentials, create_app, sign
from assay.api.main import oidc_client_from_env
from assay.auth import OidcClientConfig, OidcConfig, OidcVerifier, StaticJwks
from assay.ingestion import IngestionService, InMemoryRepository
from test_sso import AUD, ISS, JWKS, NOW, ROLE_MAP, claims, token

CLIENT = OidcClientConfig(client_id="assay-web", authorization_endpoint="https://idp.example/authorize",
                          token_endpoint="https://idp.example/token", scope="openid profile",
                          end_session_endpoint="https://idp.example/logout", audience="assay-api")
SECRET = b"svc-secret"


def app(*, client=CLIENT, with_oidc=True):
    oidc = OidcVerifier(OidcConfig(issuer=ISS, audience=AUD, role_map=ROLE_MAP), StaticJwks(JWKS), clock=lambda: NOW) if with_oidc else None
    creds = Credentials([Credential("svc", "tenant-a", SECRET, frozenset({"auditor"}))])
    return TestClient(create_app(IngestionService(InMemoryRepository()), creds, oidc=oidc, oidc_client=client, rate_limit_per_minute=None))


def bearer(tok):
    return {"Authorization": f"Bearer {tok}"}


def test_the_sign_in_settings_are_public_and_hold_nothing_secret():
    r = app().get("/v1/auth/config")  # no credentials at all
    assert r.status_code == 200
    body = r.json()
    assert body == {"enabled": True, "client_id": "assay-web", "authorization_endpoint": "https://idp.example/authorize",
                    "token_endpoint": "https://idp.example/token", "scope": "openid profile",
                    "end_session_endpoint": "https://idp.example/logout", "audience": "assay-api", "token_use": "access_token"}
    text = json.dumps(body)
    assert "secret" not in text and "jwks" not in text.lower() and "role" not in text.lower()


def test_without_single_sign_on_the_web_app_is_told_it_is_off_even_if_settings_exist():
    assert app(with_oidc=False).get("/v1/auth/config").json() == {"enabled": False}
    assert app(client=None).get("/v1/auth/config").json() == {"enabled": False}


def test_me_reports_the_roles_the_token_grants_and_nothing_the_browser_claims():
    c = app()
    r = c.get("/v1/me", headers=bearer(token(claims(groups=["fraud-analysts", "auditors", "stranger"]))))
    assert r.status_code == 200
    assert r.json() == {"subject": "sso-user-123", "tenant": "tenant-a", "roles": ["analyst", "auditor"], "method": "sso"}
    # roles named in a header or query are ignored: they come from the verified token only
    r = c.get("/v1/me?roles=admin", headers={**bearer(token()), "X-Roles": "admin"})
    assert r.json()["roles"] == ["analyst"]


def test_me_works_for_a_signed_service_caller_too_and_says_which_method_was_used():
    r = app().get("/v1/me", headers={"X-Assay-Key": "svc", "X-Assay-Signature": sign(SECRET, b"")})
    assert r.json() == {"subject": "svc", "tenant": "tenant-a", "roles": ["auditor"], "method": "signed"}


def test_me_refuses_the_unauthenticated_and_a_bad_token_without_saying_why():
    c = app()
    assert c.get("/v1/me").status_code == 401
    bad = c.get("/v1/me", headers=bearer("not.a.token"))
    assert bad.status_code == 401 and bad.json()["detail"] == {"code": "unauthorized"}
    assert bad.headers["WWW-Authenticate"] == "Bearer"


def test_a_sign_in_without_mfa_says_so_so_the_person_can_act_on_it():
    r = app().get("/v1/me", headers=bearer(token(claims(amr=["pwd"]))))
    assert r.status_code == 401 and r.json()["detail"]["code"] == "mfa_required"


def test_a_person_with_no_mapped_group_signs_in_with_no_roles_and_is_refused_everywhere_else():
    c = app()
    h = bearer(token(claims(groups=["unmapped"])))
    assert c.get("/v1/me", headers=h).json()["roles"] == []  # the web app can say "you have no access" instead of failing
    assert c.get("/v1/review/queue", headers=h).status_code in (403, 404)
    assert c.get("/v1/audit/export", headers=h).status_code == 403


def test_settings_that_would_be_unsafe_are_refused_at_construction():
    ok = {"client_id": "x", "authorization_endpoint": "https://idp/a", "token_endpoint": "https://idp/t"}
    OidcClientConfig(**ok)
    OidcClientConfig(**{**ok, "authorization_endpoint": "http://127.0.0.1:9000/a", "token_endpoint": "http://localhost:9000/t"})
    with pytest.raises(ValueError, match="https"):
        OidcClientConfig(**{**ok, "token_endpoint": "http://idp.example/t"})
    with pytest.raises(ValueError, match="https"):
        OidcClientConfig(**{**ok, "end_session_endpoint": "ftp://idp/logout"})
    with pytest.raises(ValueError, match="token_use"):
        OidcClientConfig(**ok, token_use="refresh_token")
    with pytest.raises(ValueError, match="client_id"):
        OidcClientConfig(**{**ok, "client_id": ""})


def test_the_environment_turns_browser_sign_in_on_only_when_everything_it_needs_is_there(monkeypatch):
    for k in ("ASSAY_OIDC_CLIENT_ID", "ASSAY_OIDC_ISSUER", "ASSAY_OIDC_AUTHORIZATION_ENDPOINT", "ASSAY_OIDC_TOKEN_ENDPOINT",
              "ASSAY_OIDC_TOKEN_USE", "ASSAY_OIDC_SCOPE"):
        monkeypatch.delenv(k, raising=False)
    assert oidc_client_from_env() is None
    monkeypatch.setenv("ASSAY_OIDC_CLIENT_ID", "assay-web")
    with pytest.raises(RuntimeError, match="single sign-on is not on"):
        oidc_client_from_env()
    monkeypatch.setenv("ASSAY_OIDC_ISSUER", ISS)
    with pytest.raises(RuntimeError, match="AUTHORIZATION_ENDPOINT"):
        oidc_client_from_env()
    monkeypatch.setenv("ASSAY_OIDC_AUTHORIZATION_ENDPOINT", "https://idp.example/authorize")
    monkeypatch.setenv("ASSAY_OIDC_TOKEN_ENDPOINT", "https://idp.example/token")
    monkeypatch.setenv("ASSAY_OIDC_TOKEN_USE", "id_token")
    c = oidc_client_from_env()
    assert c.client_id == "assay-web" and c.token_use == "id_token" and c.scope == "openid"
