"""The parts of single sign-on that need no third-party library, so the API can refer to them
whether or not the optional `sso` extra (PyJWT) is installed."""

from __future__ import annotations

from dataclasses import dataclass, field

HUMAN_ROLES = frozenset({"analyst", "senior_analyst", "manager", "auditor", "approver", "admin"})


class AuthError(Exception):
    """A token was refused. `code` is for logs and tests, never sent to the caller except `mfa_required`."""

    def __init__(self, code: str, detail: str = ""):
        super().__init__(detail or code)
        self.code = code


@dataclass(frozen=True)
class Principal:
    subject: str
    tenant_id: str
    roles: frozenset[str]
    amr: tuple[str, ...] = ()


@dataclass(frozen=True)
class OidcConfig:
    issuer: str
    audience: str
    tenant_claim: str = "assay_tenant"
    roles_claim: str = "groups"
    role_map: dict[str, str] = field(default_factory=dict)  # provider group -> Assay role
    require_mfa: bool = True
    mfa_acr_values: frozenset[str] = frozenset()
    max_token_age_s: int = 3600  # how old `iat` may be (a parameter)
    leeway_s: int = 30           # clock skew tolerated (a parameter)


@dataclass(frozen=True)
class OidcClientConfig:
    """What the web app needs to sign a person in (authorization code with PKCE). All of it is PUBLIC: it is served to
    anyone who asks, so it holds no secret. The web app is a public client, so there is no client secret to hold."""

    client_id: str
    authorization_endpoint: str
    token_endpoint: str
    scope: str = "openid"
    redirect_uri: str | None = None  # default: where the app is served from
    end_session_endpoint: str | None = None
    audience: str | None = None  # sent to the provider when its access tokens are minted per API (Auth0 style)
    token_use: str = "access_token"  # which token the API receives as the bearer: access_token or id_token

    def __post_init__(self) -> None:
        if self.token_use not in ("access_token", "id_token"):
            raise ValueError("token_use must be access_token or id_token")
        for name in ("authorization_endpoint", "token_endpoint", "end_session_endpoint"):
            url = getattr(self, name)
            if url is None:
                continue
            u = url.lower()
            local = u.startswith(("http://localhost", "http://127.0.0.1"))  # only for local testing against a stand-in provider
            if not (u.startswith("https://") or local):
                raise ValueError(f"{name} must be https")
        if not self.client_id:
            raise ValueError("client_id is required")

    def public(self) -> dict:
        return {"enabled": True, **{k: v for k, v in self.__dict__.items() if v is not None}}
