"""Async-mirror coverage for the shared-memory surface.

The async client mirrors the sync client line-for-line; this proves the shared
scope / ACL / shared-record / promote paths behave identically through
``AsyncCosmosMemoryClient``. Reuses the sync ``tenant`` fixture for isolation and
teardown (both clients target the same physical Cosmos containers).
"""

from __future__ import annotations

import asyncio
import os
import uuid
from collections.abc import AsyncIterator, Callable

import pytest

from azure.cosmos.agent_memory._security import SecurityContext
from azure.cosmos.agent_memory.aio.cosmos_memory_client import AsyncCosmosMemoryClient
from azure.cosmos.agent_memory.exceptions import MemoryConflictError
from tests.conftest import INTEGRATION_ENABLED

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(not INTEGRATION_ENABLED, reason="Set AGENT_MEMORY_RUN_INTEGRATION=true"),
]


@pytest.fixture
async def aio_client(
    cosmos_endpoint: str,
    cosmos_key: str,
    cosmos_database: str,
    ai_foundry_endpoint: str,
    ai_foundry_api_key: str,
    embedding_deployment_name: str,
    embedding_dimensions: int,
    chat_deployment_name: str,
) -> AsyncIterator[AsyncCosmosMemoryClient]:
    if not cosmos_endpoint or not ai_foundry_endpoint:
        pytest.skip("COSMOS_DB_ENDPOINT / AI_FOUNDRY_ENDPOINT not set")
    os.environ["MEMORY_PROCESSOR_OWNER"] = "durable"
    client = AsyncCosmosMemoryClient(
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
    await client.create_memory_store()
    try:
        yield client
    finally:
        await client.close()


async def _search_until(session, terms: str, needle: str, *, attempts: int = 8, delay: float = 1.5, **kw) -> bool:
    for _ in range(attempts):
        results = await session.search_cosmos(terms, top_k=10, **kw)
        if any(needle in str(r.get("content", "")) for r in results):
            return True
        await asyncio.sleep(delay)
    return False


async def test_async_scoped_write_read_and_acl_deny(aio_client, make_ctx: Callable[..., SecurityContext], foundry_live):
    gid = "eng-" + uuid.uuid4().hex[:8]
    team = f"team:{gid}"
    marker = f"ASYNCTEAM-{uuid.uuid4().hex[:8]} the oncall rotation handoff is every monday"

    author = make_ctx("user:alice", groups=[gid], roles=[f"{team}:member"])
    teammate = make_ctx("user:carol", groups=[gid])
    outsider = make_ctx("user:bob")

    await aio_client.session(author, write_scope=team).upsert_memory(
        role="user", content=marker, memory_type="fact", thread_id="t1"
    )

    assert await _search_until(aio_client.session(teammate), "oncall rotation handoff every monday", marker)
    outsider_hits = await aio_client.session(outsider).search_cosmos("oncall rotation handoff every monday", top_k=10)
    assert not any(marker in str(r.get("content", "")) for r in outsider_hits)


async def test_async_shared_record_compare_and_swap(aio_client, make_ctx: Callable[..., SecurityContext]):
    gid = "eng-" + uuid.uuid4().hex[:8]
    team = f"team:{gid}"
    author = make_ctx("user:alice", groups=[gid], roles=[f"{team}:member"])
    key = "aioplan-" + uuid.uuid4().hex[:6]

    first = await aio_client.put_shared_record(ctx=author, scope_key=team, key=key, data={"n": 0})
    stale = first["_etag"]
    await aio_client.compare_and_swap_shared_record(
        ctx=author, scope_key=team, key=key, record={"data": {"n": 1}}, etag=stale
    )
    with pytest.raises(MemoryConflictError):
        await aio_client.compare_and_swap_shared_record(
            ctx=author, scope_key=team, key=key, record={"data": {"n": 2}}, etag=stale
        )


async def test_async_promote_then_cross_scope_read(
    aio_client, make_ctx: Callable[..., SecurityContext], tenant: str, foundry_live
):
    gid = "eng-" + uuid.uuid4().hex[:8]
    team = f"team:{gid}"
    # The promoter is an authorized writer on the source team and the target org scope.
    author = make_ctx("user:alice", groups=[gid], roles=[f"{team}:member", f"org:{tenant}:member"])
    marker = f"ASYNCPROMO-{uuid.uuid4().hex[:8]} we block merges without two approvals"

    memory_id = await aio_client.session(author, write_scope=team).upsert_memory(
        role="user", content=marker, memory_type="fact", thread_id="t1"
    )
    org = f"org:{author.tenant_id}"
    await aio_client.promote(memory_id, team, org, author)

    # Any tenant member (no group) reads the org-promoted copy.
    peer = make_ctx("user:zoe")
    assert await _search_until(aio_client.session(peer), "block merges without two approvals", marker)
