"""Async tests for procedural synthesis and procedural prompt retrieval.

The procedural-synthesis business logic is covered exhaustively by sync
tests in ``tests/unit/test_procedural_synthesis.py`` against
``PipelineService``; ``AsyncPipelineService`` is a 1:1 async mirror.
These tests verify async wiring — that the client awaits the pipeline
correctly, that the durable-processor branch short-circuits, and that
the store-backed procedural reads work over async iterators.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from agent_memory_toolkit.aio.cosmos_memory_client import AsyncCosmosMemoryClient
from agent_memory_toolkit.aio.processors import AsyncDurableFunctionProcessor


class AsyncIterator:
    def __init__(self, items):
        self._items = iter(items)

    def __aiter__(self):
        return self

    async def __anext__(self):
        try:
            return next(self._items)
        except StopIteration:
            raise StopAsyncIteration


def _procedural_doc(
    doc_id: str,
    *,
    version: int,
    content: str,
    source_fact_ids: list[str],
    source_episodic_ids: list[str],
    superseded_by: str | None = None,
    ts: int = 0,
    etag: str = "etag-1",
) -> dict:
    doc = {
        "id": doc_id,
        "user_id": "u1",
        "thread_id": "__procedural__",
        "type": "procedural",
        "version": version,
        "content": content,
        "source_fact_ids": list(source_fact_ids),
        "source_episodic_ids": list(source_episodic_ids),
        "supersedes_ids": [],
        "created_at": f"2025-01-0{version}T00:00:00+00:00",
        "role": "system",
        "tags": ["sys:procedural", "sys:synthesized"],
        "_etag": etag,
        "_ts": ts,
    }
    if superseded_by is not None:
        doc["superseded_by"] = superseded_by
    return doc


def _make_client(*, processor=None) -> AsyncCosmosMemoryClient:
    client = AsyncCosmosMemoryClient(use_default_credential=False, processor=processor)
    client._memories_container_client = MagicMock()
    client._turns_container_client = client._memories_container_client
    client._summaries_container_client = client._memories_container_client
    return client


@pytest.mark.asyncio
async def test_async_synthesize_procedural_awaits_async_pipeline():
    """The client must ``await`` ``AsyncPipelineService.synthesize_procedural``
    directly (no ``asyncio.to_thread`` indirection) and forward force=True."""
    client = _make_client()
    pipeline = AsyncMock()
    expected = {"status": "synthesized", "procedural": {"id": "proc_u1_1", "version": 1}}
    pipeline.synthesize_procedural.return_value = expected
    pipeline._store = client._get_store()
    pipeline._containers = dict(client._containers)
    client._pipeline = pipeline

    result = await client.synthesize_procedural("u1", force=True)

    assert result == expected
    pipeline.synthesize_procedural.assert_awaited_once_with("u1", force=True)


@pytest.mark.asyncio
async def test_async_get_procedural_prompt_returns_none_when_missing():
    client = _make_client()
    client._memories_container_client.query_items = MagicMock(return_value=AsyncIterator([]))

    assert await client.get_procedural_prompt("u1") is None


@pytest.mark.asyncio
async def test_async_get_procedural_prompt_returns_active_content():
    active_doc = _procedural_doc(
        "proc_u1_2",
        version=2,
        content="Active prompt",
        source_fact_ids=["f1"],
        source_episodic_ids=["e1"],
        ts=2,
    )
    superseded_doc = _procedural_doc(
        "proc_u1_1",
        version=1,
        content="Old prompt",
        source_fact_ids=["f1"],
        source_episodic_ids=["e1"],
        superseded_by="proc_u1_2",
        ts=1,
    )
    docs = [superseded_doc, active_doc]
    client = _make_client()

    def _query_items(**kwargs):
        query = kwargs["query"]
        if "superseded_by" in query:
            return AsyncIterator([doc for doc in docs if not doc.get("superseded_by")])
        return AsyncIterator(docs)

    client._memories_container_client.query_items = MagicMock(side_effect=_query_items)

    assert await client.get_procedural_prompt("u1") == "Active prompt"


@pytest.mark.asyncio
async def test_async_get_procedural_history_orders_active_first_then_newest_versions():
    v1 = _procedural_doc(
        "proc_u1_1",
        version=1,
        content="v1",
        source_fact_ids=["f1"],
        source_episodic_ids=["e1"],
        superseded_by="proc_u1_2",
        ts=1,
    )
    v2 = _procedural_doc(
        "proc_u1_2",
        version=2,
        content="v2",
        source_fact_ids=["f1", "f2"],
        source_episodic_ids=["e1"],
        superseded_by="proc_u1_3",
        ts=2,
    )
    v3 = _procedural_doc(
        "proc_u1_3",
        version=3,
        content="v3",
        source_fact_ids=["f1", "f2", "f3"],
        source_episodic_ids=["e1"],
        ts=3,
    )
    client = _make_client()
    client._memories_container_client.query_items = MagicMock(return_value=AsyncIterator([v1, v3, v2]))

    history = await client.get_procedural_history("u1", limit=10)

    assert [doc["id"] for doc in history] == ["proc_u1_3", "proc_u1_2", "proc_u1_1"]


@pytest.mark.asyncio
async def test_async_client_synthesize_procedural_raises_for_remote_processors():
    client = _make_client(processor=AsyncDurableFunctionProcessor())
    client._pipeline = AsyncMock()

    with pytest.raises(NotImplementedError, match="durable mode"):
        await client.synthesize_procedural("u1")

    client._pipeline.synthesize_procedural.assert_not_called()
