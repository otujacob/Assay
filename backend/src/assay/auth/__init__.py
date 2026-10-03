"""Single sign-on (PRD 16, FR-42). See `oidc.py` for what a token must satisfy.

`AuthError`, `Principal` and `OidcConfig` are always available. `OidcVerifier`, `HttpJwks` and
`StaticJwks` need the optional `sso` extra (PyJWT) and are loaded on first use.
"""

from .base import HUMAN_ROLES, AuthError, OidcConfig, Principal

__all__ = ["HUMAN_ROLES", "AuthError", "HttpJwks", "OidcConfig", "OidcVerifier", "Principal", "StaticJwks"]


def __getattr__(name: str):
    if name in ("OidcVerifier", "HttpJwks", "StaticJwks", "JwksSource"):
        from . import oidc

        return getattr(oidc, name)
    raise AttributeError(name)
