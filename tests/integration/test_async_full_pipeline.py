"""Async live integration smoke for ``AsyncCosmosMemoryClient``.

The async pipeline mirrors every sync dedup/reconcile code path line-for-line
(watermark, euclidean guard, ``exclude_ids`` parity, reconcile cadence, the
vector-floor dedup ladder). The sync suite covers all of that against a live
backend, but the async client had **no** live coverage - its processor test is
fully mocked. This module exercises the real async end-to-end flow (write turns
→ extract → reconcile → search) so the async mirror can't silently diverge from
sync without a test failing.

The Azure Function host is **not** required: the same ``AsyncPipelineService``
the change-feed trigger drives is exposed directly on the client.

Enable by setting::

    AGENT_MEMORY_RUN_INTEGRATION=true

Auth: ``COSMOS_DB_KEY`` is used when present; otherwise ``DefaultAzureCredential``.
"""

from __future__ import annotations

import asyncio
import time

import pytest

from azure.cosmos.agent_memory._partitioning import OWNER_FILTER, owner_parameters, partition_key_for
from azure.cosmos.agent_memory.aio.cosmos_memory_client import AsyncCosmosMemoryClient
from tests.conftest import INTEGRATION_ENABLED

# What ``reconcile`` returns: the facts checked that stay active (``kept``), the duplicates folded (``merged``),
# the conflicting pairs linked (``conflicts``), and the pass's other counts, including the facts combined into one.
RECONCILE_KEYS = {
    "kept",
    "merged",
    "conflicts",
    "checked",
    "compared",
    "vetoed",
    "combined",
    "synthesized",
    "combine_groups",
    "combine_failed",
}

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        not INTEGRATION_ENABLED,
        reason="Set AGENT_MEMORY_RUN_INTEGRATION=true",
    ),
]


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------


@pytest.fixture
async def async_agent_memory(
    cosmos_endpoint,
    cosmos_key,
    cosmos_database,
    cosmos_container,
    ai_foundry_endpoint,
    ai_foundry_api_key,
    embedding_deployment_name,
    embedding_dimensions,
    chat_deployment_name,
    integration_tenant_id,
):
    """A live AsyncCosmosMemoryClient with its containers provisioned/connected."""
    if not cosmos_endpoint or not ai_foundry_endpoint:
        pytest.skip("COSMOS_DB_ENDPOINT / AI_FOUNDRY_ENDPOINT not set")

    client = AsyncCosmosMemoryClient(
        cosmos_endpoint=cosmos_endpoint,
        cosmos_key=cosmos_key or None,
        cosmos_database=cosmos_database,
        cosmos_container=cosmos_container,
        ai_foundry_endpoint=ai_foundry_endpoint,
        ai_foundry_api_key=ai_foundry_api_key or None,
        embedding_deployment_name=embedding_deployment_name,
        embedding_dimensions=embedding_dimensions,
        chat_deployment_name=chat_deployment_name,
        tenant_id=integration_tenant_id,
    )
    # The async client cannot auto-connect in __init__; do it explicitly.
    await client.create_memory_store()
    try:
        yield client
    finally:
        await client.close()


async def _async_add_turns(
    mem: AsyncCosmosMemoryClient,
    user_id: str,
    thread_id: str,
    turns: list[tuple[str, str]],
) -> None:
    for role, content in turns:
        await mem.upsert_memory(
            user_id=user_id,
            role=role,
            content=content,
            memory_type="turn",
            thread_id=thread_id,
        )


async def _async_cleanup(mem: AsyncCosmosMemoryClient, user_id: str) -> None:
    """Best-effort delete of every document for *user_id* across all containers."""
    partition_key = partition_key_for(mem.tenant_id, user_id)
    sql = f"SELECT c.id FROM c WHERE {OWNER_FILTER}"
    params = owner_parameters(mem.tenant_id, user_id)
    for container in (
        mem._turns_container_client,
        mem._memories_container_client,
        mem._summaries_container_client,
    ):
        if container is None:
            continue
        try:
            docs = [
                doc async for doc in container.query_items(query=sql, parameters=params, partition_key=partition_key)
            ]
        except Exception:
            continue
        for doc in docs:
            try:
                await container.delete_item(item=doc["id"], partition_key=partition_key)
            except Exception:
                pass


async def _async_seed_fact_with_embedding(
    mem: AsyncCosmosMemoryClient,
    user_id: str,
    thread_id: str,
    content: str,
    *,
    retries: int = 4,
) -> None:
    """Seed a fact and confirm it stored *with* an embedding (async mirror of the
    sync helper). Retries through transient embedding-service blips so the
    extract-time vector floor always has a neighbour; skips honestly if the
    embedding service is genuinely unavailable."""
    check = f"SELECT c.id FROM c WHERE {OWNER_FILTER} AND c.content = @content AND IS_DEFINED(c.embedding)"
    params = [*owner_parameters(mem.tenant_id, user_id), {"name": "@content", "value": content}]
    partition_key = partition_key_for(mem.tenant_id, user_id)
    for _ in range(retries):
        await mem.upsert_memory(
            user_id=user_id,
            role="user",
            content=content,
            memory_type="fact",
            thread_id=thread_id,
            salience=0.7,
        )
        embedded = [
            doc
            async for doc in mem._memories_container_client.query_items(
                query=check, parameters=params, partition_key=partition_key
            )
        ]
        if embedded:
            return
        await asyncio.sleep(1)
    pytest.skip(f"embedding service unavailable - could not seed an embedded fact for {content!r}")


async def _async_wait_vector_searchable(
    mem: AsyncCosmosMemoryClient,
    user_id: str,
    search_terms: str,
    *,
    timeout: float = 20.0,
) -> None:
    """Poll vector search until the user's seeded fact is retrievable (DiskANN
    index caught up), so the subsequent retrieval is deterministic rather than
    racing the async index."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            if await mem.search_cosmos(search_terms=search_terms, user_id=user_id, top_k=5):
                return
        except Exception:
            pass
        await asyncio.sleep(1)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestAsyncEndToEnd:
    async def test_extract_reconcile_search(
        self,
        async_agent_memory,
        unique_user_id,
        unique_thread_id,
    ):
        """Full async round-trip: turns → extract → reconcile → search."""
        mem = async_agent_memory
        try:
            await _async_add_turns(
                mem,
                unique_user_id,
                unique_thread_id,
                [
                    ("user", "I just adopted a golden retriever puppy named Cosmo."),
                    ("agent", "Congrats on Cosmo! How old is the puppy?"),
                    ("user", "Cosmo is 10 weeks old and already loves swimming in the lake."),
                ],
            )
            await asyncio.sleep(1)

            extract_stats = await mem.extract_memories(
                user_id=unique_user_id,
                thread_id=unique_thread_id,
            )
            assert isinstance(extract_stats, dict)

            facts = await mem.get_memories(user_id=unique_user_id, memory_types=["fact"])
            assert len(facts) >= 1, "Async extraction should produce at least one fact"

            reconcile_stats = await mem.reconcile(user_id=unique_user_id)
            assert isinstance(reconcile_stats, dict)
            assert RECONCILE_KEYS <= set(reconcile_stats), (
                f"reconcile should return the duplicate, combine and conflict counts, got {reconcile_stats}"
            )

            results = await mem.search_cosmos(
                search_terms="golden retriever puppy",
                user_id=unique_user_id,
                top_k=5,
            )
            assert len(results) >= 1, "Async search should return at least one result"
        finally:
            await _async_cleanup(mem, unique_user_id)
