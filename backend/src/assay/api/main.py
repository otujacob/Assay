"""Server entry point: `uvicorn assay.api.main:app`.

Configuration comes from the environment:
  ASSAY_DATABASE_URL        connection string for a NON-SUPERUSER login role that is a member of
                            assay_app (see README). The service refuses to start on a superuser.
  ASSAY_DEV_CREDENTIALS     JSON list of {"key_id", "tenant_id", "secret", "roles"}. Development
                            only: production credentials belong in the managed secret store.
  ASSAY_BUNDLES             JSON list of {"tenant_id", "path"}: signed model bundles to load.
  ASSAY_BUNDLE_SIGNING_KEY  key the bundles were signed with (from the secret store in production).
                            Bundles are verified before they are deserialised.
  ASSAY_MASTER_KEY          master secret (at least 32 characters) for the LOCAL key provider that
                            opens sealed bundles (FR-41). Production should use a managed key service
                            behind the same assay.crypto.KeyProvider interface instead.
  ASSAY_REQUIRE_ENCRYPTED_BUNDLES   "1" refuses any bundle that is not sealed with the tenant key.
Without ASSAY_BUNDLES the server runs ingestion only.
"""

from __future__ import annotations

import json
import os
from contextlib import contextmanager

from psycopg.rows import tuple_row
from psycopg_pool import ConnectionPool

from assay.api.app import Credential, Credentials, create_app
from assay.detection import load_bundle
from assay.ingestion import IngestionService
from assay.ingestion.pg_repo import PostgresRepository
from assay.review.service import ReviewService
from assay.scoring import BundleRegistry, ScoringService


def make_provider(pool: ConnectionPool):
    @contextmanager
    def provider():
        with pool.connection() as conn:  # one connection per request, returned to the pool
            yield IngestionService(PostgresRepository(conn))

    return provider


def make_scoring_provider(pool: ConnectionPool, registry: BundleRegistry):
    @contextmanager
    def provider():
        with pool.connection() as conn:
            repo = PostgresRepository(conn)
            scoring = ScoringService(repo, registry)
            ReviewService(repo, scoring)  # registers the hook that queues cases for human review
            yield scoring

    return provider


def make_review_provider(pool: ConnectionPool, registry: BundleRegistry):
    @contextmanager
    def provider():
        with pool.connection() as conn:
            repo = PostgresRepository(conn)
            yield ReviewService(repo, ScoringService(repo, registry))

    return provider


def key_provider_from_env():
    """The local key provider if ASSAY_MASTER_KEY is set, else None (plaintext bundles only)."""
    master = os.environ.get("ASSAY_MASTER_KEY")
    if not master:
        return None
    from assay.crypto import LocalKeyProvider

    return LocalKeyProvider(master.encode())


def load_registry(specs: list[dict], key: bytes, provider=None,
                  require_encryption: bool | None = None) -> BundleRegistry:
    if require_encryption is None:
        require_encryption = os.environ.get("ASSAY_REQUIRE_ENCRYPTED_BUNDLES") == "1"
    if require_encryption and provider is None:
        raise RuntimeError("ASSAY_REQUIRE_ENCRYPTED_BUNDLES is set but there is no key provider "
                           "(set ASSAY_MASTER_KEY)")
    registry = BundleRegistry()
    for s in specs:
        # verifies the signature, decrypts if sealed, then loads
        artefact, manifest = load_bundle(s["path"], s["tenant_id"], key, decrypt_with=provider,
                                         require_encryption=require_encryption)
        registry.register(s["tenant_id"], artefact, manifest)
    return registry


def refuse_superuser(pool: ConnectionPool) -> None:
    with pool.connection() as conn:
        row = conn.cursor(row_factory=tuple_row).execute(
            "SELECT rolsuper OR rolbypassrls FROM pg_roles WHERE rolname = current_user").fetchone()
    if row and row[0]:
        raise RuntimeError("ASSAY_DATABASE_URL must use a non-superuser role without BYPASSRLS; "
                           "a superuser defeats tenant isolation and append-only guarantees")


def build_app():
    url = os.environ["ASSAY_DATABASE_URL"]
    pool = ConnectionPool(url, min_size=1, max_size=int(os.environ.get("ASSAY_POOL_MAX", "10")),
                          open=True)
    refuse_superuser(pool)
    creds = [Credential(c["key_id"], c["tenant_id"], c["secret"].encode(),
                        frozenset(c.get("roles", ["ingest"])))
             for c in json.loads(os.environ["ASSAY_DEV_CREDENTIALS"])]
    scoring = review = None
    if os.environ.get("ASSAY_BUNDLES"):
        registry = load_registry(json.loads(os.environ["ASSAY_BUNDLES"]),
                                 os.environ["ASSAY_BUNDLE_SIGNING_KEY"].encode(), key_provider_from_env())
        scoring = make_scoring_provider(pool, registry)
        review = make_review_provider(pool, registry)
    return create_app(make_provider(pool), Credentials(creds), scoring=scoring, review=review)


def __getattr__(name: str):
    # Built lazily so importing this module (tests, tooling) does not need a database.
    if name == "app":
        return build_app()
    raise AttributeError(name)
