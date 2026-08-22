"""Fixtures for the live shared-memory integration suite.

These tests exercise the shared / team / org memory surface end to end against
**live** Azure Cosmos DB and AI Foundry (no stubs, no fakes). They are gated by
``AGENT_MEMORY_RUN_INTEGRATION=true`` and reuse the session-scoped Cosmos / Foundry
env fixtures from ``tests/conftest.py``.

Isolation strategy
------------------
Every test runs under a unique ``tenant_id`` (``it-<uuid>``). Because the record
partition key is ``[/tenant_id, /scope_key, /thread_id]``, a whole test's data is a
disjoint slice of the shared containers; teardown deletes every document carrying the
test's ``tenant_id`` across the memories / turns / summaries containers (shared-state
records and pins live in the memories container, so they are covered too).

Determinism
-----------
``MEMORY_PROCESSOR_OWNER`` is forced to ``durable`` for the session so the SDK's
write-through auto-trigger becomes a no-op; tests drive the pipeline explicitly
(``extract_memories`` / ``generate_thread_summary`` / ``generate_user_summary`` /
``reconcile``).
"""

from __future__ import annotations

import os
import time
import uuid
from collections.abc import Callable, Iterator
from typing import Any

import pytest

from azure.cosmos.agent_memory import CosmosMemoryClient
from azure.cosmos.agent_memory._security import SecurityContext
from tests.conftest import INTEGRATION_ENABLED


def _skip_if_disabled() -> None:
    if not INTEGRATION_ENABLED:
        pytest.skip("Set AGENT_MEMORY_RUN_INTEGRATION=true to run the shared-memory integration suite")


# ---------------------------------------------------------------------------
# Live client
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def shared_client(
    cosmos_endpoint: str,
    cosmos_key: str,
    cosmos_database: str,
    ai_foundry_endpoint: str,
    ai_foundry_api_key: str,
    embedding_deployment_name: str,
    embedding_dimensions: int,
    chat_deployment_name: str,
) -> Iterator[CosmosMemoryClient]:
    """A single live :class:`CosmosMemoryClient` shared by the whole module session."""
    _skip_if_disabled()
    if not cosmos_endpoint or not ai_foundry_endpoint:
        pytest.skip("COSMOS_DB_ENDPOINT / AI_FOUNDRY_ENDPOINT not set")

    # Silence the SDK write-through auto-trigger so tests drive the pipeline explicitly.
    os.environ["MEMORY_PROCESSOR_OWNER"] = "durable"

    client = CosmosMemoryClient(
        cosmos_endpoint=cosmos_endpoint,
        cosmos_key=cosmos_key or None,
        cosmos_database=cosmos_database,
        cosmos_container=os.environ.get("COSMOS_DB_MEMORIES_CONTAINER", "memories"),
        ai_foundry_endpoint=ai_foundry_endpoint,
        ai_foundry_api_key=ai_foundry_api_key or None,
        embedding_deployment_name=embedding_deployment_name,
        embedding_dimensions=embedding_dimensions,
        chat_deployment_name=chat_deployment_name,
    )
    try:
        yield client
    finally:
        client.close()


@pytest.fixture(scope="session")
def foundry_live(shared_client: CosmosMemoryClient) -> bool:
    """Skip a test when embeddings + chat are not actually reachable with these creds."""
    try:
        shared_client._embeddings_client.generate("ping")
        shared_client._chat_client.generate([{"role": "user", "content": "reply with the word ok"}])
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"AI Foundry embeddings/chat not reachable: {exc}")
    return True


# ---------------------------------------------------------------------------
# Tenant isolation + cleanup
# ---------------------------------------------------------------------------


def _cleanup_tenant(client: CosmosMemoryClient, tenant_id: str) -> None:
    """Best-effort delete of every document written under ``tenant_id``."""
    query = "SELECT c.id, c.tenant_id, c.scope_key, c.thread_id FROM c WHERE c.tenant_id = @t"
    params = [{"name": "@t", "value": tenant_id}]
    for container in (
        client._memories_container_client,
        client._turns_container_client,
        client._summaries_container_client,
    ):
        if container is None:
            continue
        try:
            docs = list(container.query_items(query=query, parameters=params, enable_cross_partition_query=True))
        except Exception:  # noqa: BLE001
            continue
        for doc in docs:
            try:
                container.delete_item(
                    item=doc["id"],
                    partition_key=[doc.get("tenant_id"), doc.get("scope_key", ""), doc.get("thread_id", "")],
                )
            except Exception:  # noqa: BLE001
                pass


@pytest.fixture
def tenant(shared_client: CosmosMemoryClient) -> Iterator[str]:
    """A unique tenant id; all documents written under it are deleted on teardown."""
    tenant_id = "it-" + uuid.uuid4().hex[:12]
    try:
        yield tenant_id
    finally:
        _cleanup_tenant(shared_client, tenant_id)


# ---------------------------------------------------------------------------
# SecurityContext factory + read/write helpers
# ---------------------------------------------------------------------------


@pytest.fixture
def make_ctx(tenant: str) -> Callable[..., SecurityContext]:
    """Build a :class:`SecurityContext` bound to the test's isolated tenant."""

    def _make(
        principal: str = "user:alice",
        *,
        groups: list[str] | None = None,
        roles: list[str] | None = None,
        agent_id: str | None = None,
        tenant_id: str | None = None,
    ) -> SecurityContext:
        return SecurityContext(
            tenant_id=tenant_id or tenant,
            principal=principal,
            groups=groups or [],
            roles=roles or [],
            agent_id=agent_id,
        )

    return _make


@pytest.fixture
def write_fact(shared_client: CosmosMemoryClient) -> Callable[..., str]:
    """Write a memory in a given placement scope and return its id (written through)."""

    def _write(
        ctx: SecurityContext,
        scope: str,
        content: str,
        *,
        thread_id: str = "t1",
        memory_type: str = "fact",
        metadata: dict[str, Any] | None = None,
    ) -> str:
        # Seeding a record into a shared/foreign scope now requires write authorization
        # (group membership grants read only). Author the seed as an authorized writer for
        # that scope - modelling the producer that legitimately placed the record - while
        # the read/enforcement path under test still uses the original ``ctx``.
        write_ctx = ctx
        if scope not in (ctx.principal, ctx.agent_id):
            write_ctx = ctx.model_copy(update={"roles": [*ctx.roles, f"{scope}:writer"]})
        return shared_client.session(write_ctx, write_scope=scope).upsert_memory(
            role="user",
            content=content,
            memory_type=memory_type,
            thread_id=thread_id,
            metadata=metadata,
        )

    return _write


@pytest.fixture
def search_as(shared_client: CosmosMemoryClient) -> Callable[..., list[dict[str, Any]]]:
    """Search as a caller, auto-resolving read scopes from the context unless overridden."""

    def _search(
        ctx: SecurityContext,
        terms: str,
        *,
        read_scopes: list[str] | None = None,
        project_scope: str | None = None,
        top_k: int = 10,
        **kwargs: Any,
    ) -> list[dict[str, Any]]:
        session = shared_client.session(ctx, read_scopes=read_scopes, project_scope=project_scope)
        return session.search_cosmos(terms, top_k=top_k, **kwargs)

    return _search


@pytest.fixture
def read_record(shared_client: CosmosMemoryClient) -> Callable[..., dict[str, Any]]:
    """Point-read a persisted memories-container document by scope + id."""

    def _read(*, tenant_id: str, scope_key: str, thread_id: str, memory_id: str) -> dict[str, Any]:
        return shared_client._memories_container_client.read_item(
            item=memory_id, partition_key=[tenant_id, scope_key, thread_id]
        )

    return _read


@pytest.fixture
def write_with_acl(
    shared_client: CosmosMemoryClient, write_fact: Callable[..., str], read_record: Callable[..., dict[str, Any]]
) -> Callable[..., str]:
    """Write a memory (real embedding) then overwrite its inline ACL for enforcement tests.

    Authoring an arbitrary ACL is not a public write parameter yet, so we persist a
    normal record and replace its ``acl`` block - the read/write enforcement path is
    identical regardless of how the ACL got there.
    """

    def _write(
        ctx: SecurityContext,
        scope: str,
        content: str,
        acl: dict[str, list[str]],
        *,
        thread_id: str = "t1",
    ) -> str:
        memory_id = write_fact(ctx, scope, content, thread_id=thread_id)
        doc = read_record(tenant_id=ctx.tenant_id, scope_key=scope, thread_id=thread_id, memory_id=memory_id)
        doc["acl"] = acl
        shared_client._memories_container_client.replace_item(item=memory_id, body=doc)
        return memory_id

    return _write


@pytest.fixture
def search_until(search_as: Callable[..., list[dict[str, Any]]]) -> Callable[..., list[dict[str, Any]]]:
    """Poll ``search_as`` until ``predicate`` holds, tolerating vector/FTS index lag.

    Presence assertions use this because Cosmos updates the vector / full-text index
    asynchronously after the write commits. Absence (isolation) assertions do *not*
    need it - the ACL pre-filter excludes unauthorized rows regardless of indexing.
    """

    def _run(
        ctx: SecurityContext,
        terms: str,
        predicate: Callable[[list[dict[str, Any]]], bool],
        *,
        attempts: int = 8,
        delay: float = 1.5,
        **kwargs: Any,
    ) -> list[dict[str, Any]]:
        results: list[dict[str, Any]] = []
        for _ in range(attempts):
            results = search_as(ctx, terms, **kwargs)
            if predicate(results):
                return results
            time.sleep(delay)
        return results

    return _run


def contents(results: list[dict[str, Any]]) -> list[str]:
    """Extract the ``content`` field from a list of search / query results."""
    return [str(r.get("content", "")) for r in results]
