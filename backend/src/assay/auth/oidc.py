"""Single sign-on: verify OpenID Connect bearer tokens from the institution's identity provider
(PRD 16, FR-42). This is the resource-server side only: the API checks a token it is handed. The
browser sign-in flow (authorization code with PKCE) and the provider itself are the institution's.

What a token must satisfy, or the request is refused:
  * signed with an allowed asymmetric algorithm by a key in the provider's JWKS (never `none`, and
    never a shared-secret algorithm, which would let a public key be used as a secret);
  * `iss` is the configured issuer and `aud` includes this API;
  * `exp` has not passed, `nbf` has, and `iat` is recent: credentials are short-lived;
  * it names a tenant, and at least one role this API knows after mapping the provider's groups;
  * MFA was used (PRD 16): `amr` contains "mfa", or `acr` is one the institution configured.

Roles are only ever the human ones. A token can never grant "ingest", so service callers stay on
signed requests and a stolen user token cannot submit transactions.
"""

from __future__ import annotations

import json
import threading
import time
import urllib.request
from collections.abc import Callable
from typing import Any, Protocol

import jwt
from jwt import PyJWK
from jwt.exceptions import InvalidTokenError

from .base import HUMAN_ROLES, AuthError, OidcConfig, Principal

ALLOWED_ALGORITHMS = ("RS256", "RS384", "RS512", "ES256", "ES384", "ES512", "PS256", "PS384", "PS512")


class JwksSource(Protocol):
    def key_for(self, kid: str) -> Any: ...


class StaticJwks:
    """A fixed key set, for tests and for installs that pin the provider's keys."""

    def __init__(self, jwks: dict):
        self._keys = {k["kid"]: PyJWK(k) for k in jwks.get("keys", []) if "kid" in k}

    def key_for(self, kid: str):
        try:
            return self._keys[kid].key
        except KeyError:
            raise AuthError("unknown_kid") from None


class HttpJwks:
    """The provider's published key set, cached. An unknown key id triggers one refetch (the provider
    rotated its keys), but not more often than `min_refetch_s`, so a stream of tokens with made-up key
    ids cannot make this service hammer the provider."""

    def __init__(self, url: str, ttl_s: int = 3600, min_refetch_s: int = 60,
                 fetch: Callable[[str], bytes] | None = None, clock: Callable[[], float] = time.monotonic):
        if not url.lower().startswith("https://"):
            raise ValueError("the JWKS URL must be https")
        self.url, self.ttl, self.min_refetch, self.clock = url, ttl_s, min_refetch_s, clock
        self._fetch = fetch or self._http
        self._keys: dict[str, Any] = {}
        self._loaded_at: float | None = None
        self._lock = threading.Lock()

    @staticmethod
    def _http(url: str) -> bytes:
        with urllib.request.urlopen(url, timeout=5) as r:
            return r.read(1 << 20)

    def _load(self) -> None:
        try:
            doc = json.loads(self._fetch(self.url))
        except Exception as e:  # noqa: BLE001
            raise AuthError("jwks_unavailable", f"could not fetch the key set: {type(e).__name__}") from None
        self._keys = {k["kid"]: PyJWK(k) for k in doc.get("keys", []) if "kid" in k}
        self._loaded_at = self.clock()

    def key_for(self, kid: str):
        with self._lock:
            now = self.clock()
            if self._loaded_at is None or now - self._loaded_at > self.ttl or kid not in self._keys and now - self._loaded_at >= self.min_refetch:
                self._load()
            try:
                return self._keys[kid].key
            except KeyError:
                raise AuthError("unknown_kid") from None


class OidcVerifier:
    def __init__(self, cfg: OidcConfig, jwks: JwksSource, clock: Callable[[], float] = time.time):
        bad = set(cfg.role_map.values()) - HUMAN_ROLES
        if bad:
            raise ValueError(f"role_map may only grant human roles, not {sorted(bad)}")
        self.cfg, self.jwks, self.clock = cfg, jwks, clock

    def verify(self, token: str) -> Principal:
        cfg = self.cfg
        try:
            header = jwt.get_unverified_header(token)
        except InvalidTokenError:
            raise AuthError("malformed") from None
        alg = header.get("alg")
        if alg not in ALLOWED_ALGORITHMS:  # refuses "none" and HS*, before any key is looked at
            raise AuthError("bad_algorithm", f"algorithm {alg!r} is not allowed")
        kid = header.get("kid")
        if not kid:
            raise AuthError("no_kid")
        key = self.jwks.key_for(kid)
        now = self.clock()
        try:
            claims = jwt.decode(
                token, key, algorithms=[alg], audience=cfg.audience, issuer=cfg.issuer,
                # Every time check is done below against one injectable clock, not PyJWT's own.
                options={"require": ["exp", "iat", "iss", "aud", "sub"],
                         "verify_exp": False, "verify_nbf": False, "verify_iat": False},
            )
        except InvalidTokenError as e:
            raise AuthError("invalid_token", type(e).__name__) from None
        for name in ("exp", "iat", "nbf"):
            v = claims.get(name, 0)
            if isinstance(v, bool) or not isinstance(v, (int, float)):
                raise AuthError("bad_claims", f"{name} must be a number")
        if claims["exp"] + cfg.leeway_s < now:
            raise AuthError("expired")
        if claims.get("nbf", 0) - cfg.leeway_s > now:
            raise AuthError("not_yet_valid")
        if claims["iat"] - cfg.leeway_s > now:
            raise AuthError("issued_in_the_future")
        if now - claims["iat"] > cfg.max_token_age_s + cfg.leeway_s:
            raise AuthError("too_old", f"token is older than {cfg.max_token_age_s}s: sign in again")
        amr = tuple(str(a) for a in claims.get("amr", ()) if isinstance(a, str))
        if cfg.require_mfa and not ("mfa" in amr or claims.get("acr") in cfg.mfa_acr_values):
            raise AuthError("mfa_required", "multi-factor authentication is required")
        tenant = claims.get(cfg.tenant_claim)
        if not isinstance(tenant, str) or not tenant:
            raise AuthError("no_tenant")
        groups = claims.get(cfg.roles_claim, ())
        groups = [groups] if isinstance(groups, str) else [g for g in groups if isinstance(g, str)]
        roles = frozenset(cfg.role_map[g] for g in groups if g in cfg.role_map)
        return Principal(subject=str(claims["sub"]), tenant_id=tenant, roles=roles, amr=amr)
