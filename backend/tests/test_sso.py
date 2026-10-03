"""FR-42: single sign-on with OpenID Connect bearer tokens. Keys and tokens are generated here; no
identity provider is contacted, so this proves the verification rules, not interoperability with
any particular provider."""

import base64
import hashlib
import hmac
import json

import pytest

jwt = pytest.importorskip("jwt")
from cryptography.hazmat.primitives.asymmetric import ec, rsa
from cryptography.hazmat.primitives.serialization import Encoding, PublicFormat
from fastapi.testclient import TestClient
from jwt.algorithms import ECAlgorithm, RSAAlgorithm

from assay.api import Credential, Credentials, create_app, sign
from assay.auth import AuthError, HttpJwks, OidcConfig, OidcVerifier, StaticJwks
from assay.ingestion import IngestionService, InMemoryRepository
from assay.scoring import BundleRegistry, ScoringService

ISS, AUD, NOW = "https://idp.example", "assay-api", 1_800_000_000.0
RSA_KEY = rsa.generate_private_key(public_exponent=65537, key_size=2048)
OTHER_KEY = rsa.generate_private_key(public_exponent=65537, key_size=2048)
EC_KEY = ec.generate_private_key(ec.SECP256R1())
ROLE_MAP = {"fraud-analysts": "analyst", "auditors": "auditor", "approvers": "approver", "admins": "admin"}


def jwk(public_key, kid, alg):
    algo = RSAAlgorithm if alg.startswith(("RS", "PS")) else ECAlgorithm
    return {**json.loads(algo.to_jwk(public_key)), "kid": kid, "alg": alg, "use": "sig"}


JWKS = {"keys": [jwk(RSA_KEY.public_key(), "k-rsa", "RS256"), jwk(EC_KEY.public_key(), "k-ec", "ES256")]}


def claims(**over):
    base = {"iss": ISS, "aud": AUD, "sub": "user-123", "iat": NOW - 60, "exp": NOW + 600,
            "amr": ["pwd", "mfa"], "assay_tenant": "tenant-a", "groups": ["fraud-analysts"]}
    base.update(over)
    return {k: v for k, v in base.items() if v is not None}


def token(payload=None, key=RSA_KEY, alg="RS256", kid="k-rsa", headers=None):
    return jwt.encode(payload if payload is not None else claims(), key, algorithm=alg,
                      headers={"kid": kid, **(headers or {})})


def verifier(**cfg):
    config = OidcConfig(issuer=ISS, audience=AUD, role_map=ROLE_MAP, **cfg)
    return OidcVerifier(config, StaticJwks(JWKS), clock=lambda: NOW)


def refused(v, tok, code):
    with pytest.raises(AuthError) as e:
        v.verify(tok)
    assert e.value.code == code, e.value.code


def test_a_valid_token_becomes_a_principal_with_mapped_roles():
    p = verifier().verify(token(claims(groups=["fraud-analysts", "auditors", "unrelated-group"])))
    assert (p.subject, p.tenant_id) == ("user-123", "tenant-a")
    assert p.roles == {"analyst", "auditor"}                     # unmapped groups grant nothing
    assert "mfa" in p.amr


def test_an_elliptic_curve_signature_is_accepted_too():
    assert verifier().verify(token(key=EC_KEY, alg="ES256", kid="k-ec")).tenant_id == "tenant-a"


def test_a_token_signed_by_a_different_key_is_refused():
    refused(verifier(), token(key=OTHER_KEY), "invalid_token")   # right kid, wrong private key


def test_the_none_algorithm_is_refused():
    header = base64.urlsafe_b64encode(json.dumps({"alg": "none", "kid": "k-rsa"}).encode()).rstrip(b"=")
    body = base64.urlsafe_b64encode(json.dumps(claims()).encode()).rstrip(b"=")
    refused(verifier(), (header + b"." + body + b".").decode(), "bad_algorithm")


def test_algorithm_confusion_is_refused():
    """The classic attack: sign with HS256 using the provider's PUBLIC key as the shared secret. PyJWT
    refuses to build such a token, so it is assembled by hand, as an attacker would."""
    pub_pem = RSA_KEY.public_key().public_bytes(Encoding.PEM, PublicFormat.SubjectPublicKeyInfo)

    def b64(b):
        return base64.urlsafe_b64encode(b).rstrip(b"=")

    signing_input = (b64(json.dumps({"alg": "HS256", "kid": "k-rsa"}).encode()) + b"."
                     + b64(json.dumps(claims(groups=["admins"])).encode()))
    forged = signing_input + b"." + b64(hmac.new(pub_pem, signing_input, hashlib.sha256).digest())
    refused(verifier(), forged.decode(), "bad_algorithm")


@pytest.mark.parametrize("bad", ["not.a.jwt", "", "a.b", "x" * 50])
def test_garbage_is_refused_not_crashed_on(bad):
    refused(verifier(), bad, "malformed")


def test_a_token_without_a_key_id_or_with_an_unknown_one_is_refused():
    no_kid = jwt.encode(claims(), RSA_KEY, algorithm="RS256")
    refused(verifier(), no_kid, "no_kid")
    refused(verifier(), token(kid="made-up"), "unknown_kid")


@pytest.mark.parametrize("over,code", [
    ({"iss": "https://evil.example"}, "invalid_token"),
    ({"aud": "some-other-api"}, "invalid_token"),
    ({"aud": ["other", "another"]}, "invalid_token"),
    ({"exp": NOW - 31}, "expired"),
    ({"nbf": NOW + 600}, "not_yet_valid"),
    ({"iat": NOW + 600}, "issued_in_the_future"),
    ({"iat": NOW - 7200}, "too_old"),                              # valid exp, but signed in too long ago
    ({"exp": None}, "invalid_token"),
    ({"sub": None}, "invalid_token"),
    ({"exp": "tomorrow"}, "bad_claims"),
    ({"exp": True}, "bad_claims"),
])
def test_claims_are_checked(over, code):
    refused(verifier(), token(claims(**over)), code)


def test_the_audience_may_be_a_list_that_includes_this_api():
    assert verifier().verify(token(claims(aud=["other", AUD]))).subject == "user-123"


def test_small_clock_skew_is_tolerated_but_not_a_lot():
    assert verifier().verify(token(claims(exp=NOW - 10))).subject == "user-123"   # inside the 30 s leeway
    refused(verifier(), token(claims(exp=NOW - 40)), "expired")


@pytest.mark.parametrize("amr", [["pwd"], [], None, ["otp"], "mfa"])
def test_mfa_is_required_and_a_single_factor_does_not_count(amr):
    refused(verifier(), token(claims(amr=amr)), "mfa_required")


def test_mfa_can_be_proved_by_an_acr_value_the_institution_names():
    v = verifier(mfa_acr_values=frozenset({"urn:inst:mfa"}))
    assert v.verify(token(claims(amr=None, acr="urn:inst:mfa"))).subject == "user-123"
    refused(v, token(claims(amr=None, acr="urn:inst:password")), "mfa_required")


def test_a_tenant_is_required_and_must_be_a_string():
    for bad in (None, "", 7, ["tenant-a"]):
        refused(verifier(), token(claims(assay_tenant=bad)), "no_tenant")


def test_a_token_can_never_grant_ingest_or_any_non_human_role():
    with pytest.raises(ValueError, match="human roles"):
        OidcVerifier(OidcConfig(issuer=ISS, audience=AUD, role_map={"svc": "ingest"}), StaticJwks(JWKS))
    with pytest.raises(ValueError):
        OidcVerifier(OidcConfig(issuer=ISS, audience=AUD, role_map={"x": "superuser"}), StaticJwks(JWKS))


def test_a_single_group_given_as_a_string_is_handled():
    assert verifier().verify(token(claims(groups="auditors"))).roles == {"auditor"}


def test_a_user_with_no_mapped_group_authenticates_but_holds_no_roles():
    assert verifier().verify(token(claims(groups=["nobody-cares"]))).roles == frozenset()


# -- fetching the provider's keys ----------------------------------------------------------------------------
def test_the_jwks_url_must_be_https():
    with pytest.raises(ValueError, match="https"):
        HttpJwks("http://idp.example/keys")


def test_keys_are_cached_and_refetched_on_rotation_but_not_hammered():
    t = {"now": 0.0}
    served = {"doc": {"keys": [JWKS["keys"][0]]}, "n": 0}

    def fetch(_):
        served["n"] += 1
        return json.dumps(served["doc"]).encode()

    src = HttpJwks("https://idp.example/keys", ttl_s=3600, min_refetch_s=60, fetch=fetch, clock=lambda: t["now"])
    assert src.key_for("k-rsa") and src.key_for("k-rsa") and served["n"] == 1     # cached
    served["doc"] = JWKS                                                          # the provider adds a key
    with pytest.raises(AuthError):
        src.key_for("k-ec")                                                       # too soon to refetch
    assert served["n"] == 1
    t["now"] = 61
    assert src.key_for("k-ec") and served["n"] == 2                               # rotation picked up
    for i in range(20):
        with pytest.raises(AuthError):
            src.key_for(f"made-up-{i}")                                           # junk key ids
    assert served["n"] == 2                                                       # cause no extra fetches
    t["now"] = 4000
    assert src.key_for("k-rsa") and served["n"] == 3                              # the ttl expired


def test_an_unreachable_provider_is_a_refusal_not_a_crash():
    def broken(_):
        raise OSError("network down")

    refused(OidcVerifier(OidcConfig(issuer=ISS, audience=AUD, role_map=ROLE_MAP),
                         HttpJwks("https://idp.example/keys", fetch=broken), clock=lambda: NOW),
            token(), "jwks_unavailable")


# -- through the API --------------------------------------------------------------------------------------------
@pytest.fixture
def api():
    repo = InMemoryRepository()
    creds = Credentials([Credential("svc", "tenant-a", b"svc-secret", frozenset({"ingest"}))])
    app = create_app(IngestionService(repo), creds, scoring=ScoringService(repo, BundleRegistry()),
                     oidc=verifier(), rate_limit_per_minute=5)
    c = TestClient(app)
    return {"c": c, "repo": repo}


def bearer(c, path, tok, method="GET", body=b""):
    return c.request(method, path, content=body, headers={"Authorization": f"Bearer {tok}"})


def test_a_signed_in_user_reaches_the_routes_their_role_allows_and_no_others(api):
    c = api["c"]
    auditor = token(claims(groups=["auditors"]))
    assert bearer(c, "/v1/audit/export", auditor).status_code == 200
    assert bearer(c, "/v1/audit/export", token(claims(groups=["fraud-analysts"]))).status_code == 403
    assert bearer(c, "/v1/config/policies", token(claims(groups=["admins"]))).status_code == 200
    assert bearer(c, "/v1/audit/export", token(claims(groups=["nobody-cares"]))).status_code == 403


def test_the_tenant_comes_from_the_token_and_nothing_else(api):
    c = api["c"]
    for tenant in ("tenant-a", "tenant-b"):
        bearer(c, "/v1/audit/export", token(claims(groups=["auditors"], assay_tenant=tenant)))
    # each read was filed under the tenant of the token that made it
    assert [r["actor"] for r in api["repo"].rows("tenant-a", "audit_log")] == ["api:sso-user-123"]
    assert [r["actor"] for r in api["repo"].rows("tenant-b", "audit_log")] == ["api:sso-user-123"]


def test_a_user_token_cannot_submit_transactions(api):
    """Ingestion is for services. No token can carry the ingest role, so a stolen user token cannot post data."""
    r = bearer(api["c"], "/v1/transactions", token(claims(groups=["admins", "auditors", "approvers"])), "POST", b"{}")
    assert r.status_code == 403


@pytest.mark.parametrize("tok,code", [
    ("garbage", "unauthorized"),
    (token(claims(exp=NOW - 999)), "unauthorized"),
    (token(key=OTHER_KEY), "unauthorized"),
    (token(claims(amr=["pwd"])), "mfa_required"),                 # the one reason a person can act on
])
def test_a_bad_token_is_a_401_that_says_no_more_than_it_must(api, tok, code):
    r = bearer(api["c"], "/v1/audit/export", tok)
    assert r.status_code == 401 and r.json()["detail"]["code"] == code
    assert r.headers["www-authenticate"] == "Bearer"
    assert "idp" not in r.text and "expired" not in r.text and "signature" not in r.text


def test_signed_requests_still_work_and_a_request_with_no_credentials_is_a_401(api):
    c = api["c"]
    assert c.post("/v1/transactions", content=b"{}", headers={
        "X-Assay-Key": "svc", "X-Assay-Signature": sign(b"svc-secret", b"{}")}).status_code == 422  # reaches validation
    assert c.get("/v1/audit/export").status_code == 401
    assert c.get("/v1/audit/export", headers={"X-Assay-Key": "svc"}).status_code == 401


def test_bearer_tokens_are_refused_when_sso_is_not_configured():
    repo = InMemoryRepository()
    app = create_app(IngestionService(repo), Credentials([]), scoring=ScoringService(repo, BundleRegistry()))
    assert bearer(TestClient(app), "/v1/audit/export", token(claims(groups=["auditors"]))).status_code == 401


def test_a_wrong_authorization_scheme_is_refused(api):
    r = api["c"].get("/v1/audit/export", headers={"Authorization": "Basic dXNlcjpwYXNz"})
    assert r.status_code == 401


def test_signed_in_users_are_rate_limited_per_person(api):
    c, tok = api["c"], token(claims(groups=["auditors"]))
    assert [bearer(c, "/v1/audit/export", tok).status_code for _ in range(5)] == [200] * 5
    assert bearer(c, "/v1/audit/export", tok).status_code == 429
    other = token(claims(groups=["auditors"], sub="someone-else"))
    assert bearer(c, "/v1/audit/export", other).status_code == 200      # a different person has their own allowance


def test_a_refused_role_is_recorded_against_the_signed_in_person(api):
    bearer(api["c"], "/v1/audit/export", token(claims(groups=["fraud-analysts"])))
    denied = [r for r in api["repo"].rows("tenant-a", "audit_log") if r["action"] == "access_denied"]
    assert denied and denied[0]["actor"] == "api:sso-user-123" and denied[0]["result"] == "auditor"


def test_server_configuration_for_sso(monkeypatch):
    from assay.api.main import oidc_from_env

    for k in ("ASSAY_OIDC_ISSUER", "ASSAY_OIDC_AUDIENCE", "ASSAY_OIDC_JWKS_URL", "ASSAY_OIDC_ROLE_MAP"):
        monkeypatch.delenv(k, raising=False)
    assert oidc_from_env() is None                                  # off unless an issuer is named
    monkeypatch.setenv("ASSAY_OIDC_ISSUER", ISS)
    with pytest.raises(RuntimeError, match="ASSAY_OIDC_AUDIENCE"):
        oidc_from_env()                                             # half-configured is an error, not a silent open door
    monkeypatch.setenv("ASSAY_OIDC_AUDIENCE", AUD)
    monkeypatch.setenv("ASSAY_OIDC_JWKS_URL", "https://idp.example/keys")
    monkeypatch.setenv("ASSAY_OIDC_ROLE_MAP", json.dumps(ROLE_MAP))
    v = oidc_from_env()
    assert v.cfg.require_mfa is True and v.cfg.max_token_age_s == 3600
    monkeypatch.setenv("ASSAY_OIDC_JWKS_URL", "http://idp.example/keys")
    with pytest.raises(ValueError, match="https"):
        oidc_from_env()
