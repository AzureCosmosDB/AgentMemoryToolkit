"""Async tests for procedural synthesis and procedural prompt retrieval."""

from __future__ import annotations

import json
from datetime import datetime
from unittest.mock import MagicMock, patch

import pytest

from agent_memory_toolkit.aio.cosmos_memory_client import AsyncCosmosMemoryClient
from agent_memory_toolkit.aio.processors import AsyncDurableFunctionProcessor
from agent_memory_toolkit.pipeline import ProcessingPipeline


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


def _assert_iso8601(text: str) -> None:
    assert text
    datetime.fromisoformat(text)


def _capture_upserts():
    upserted: list[dict] = []

    def _capture(*, body):
        upserted.append(body)
        return body

    return upserted, _capture


def _fact_doc(
    doc_id: str,
    content: str,
    *,
    category: str = "preference",
    salience: float = 0.9,
    created_at: str = "2025-01-01T00:00:00+00:00",
) -> dict:
    return {
        "id": doc_id,
        "user_id": "u1",
        "thread_id": "t-source",
        "role": "system",
        "type": "fact",
        "content": content,
        "metadata": {"category": category},
        "salience": salience,
        "created_at": created_at,
    }


def _episodic_doc(
    doc_id: str,
    *,
    lesson: str,
    salience: float = 0.7,
    created_at: str = "2025-01-02T00:00:00+00:00",
) -> dict:
    return {
        "id": doc_id,
        "user_id": "u1",
        "thread_id": "t-source",
        "role": "system",
        "type": "episodic",
        "content": f"Episode {doc_id}",
        "metadata": {"lesson": lesson},
        "salience": salience,
        "created_at": created_at,
    }


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


def _make_synthesis_pipeline(
    *,
    prior_docs: list[dict] | None = None,
    fact_docs: list[dict] | None = None,
    episodic_docs: list[dict] | None = None,
    name_docs: list[dict] | None = None,
    llm_output: str = "Follow the user's preferences.",
):
    container = MagicMock()
    container.query_items.side_effect = [
        list(prior_docs or []),
        list(fact_docs or []),
        list(episodic_docs or []),
        list(name_docs or []),
    ]
    upserted, capture = _capture_upserts()
    container.upsert_item.side_effect = capture

    pipeline = ProcessingPipeline(
        cosmos_container=container,
        chat_client=MagicMock(),
        embeddings_client=MagicMock(),
    )
    pipeline._run_prompty = MagicMock(return_value=json.dumps({"system_prompt": llm_output}))
    return pipeline, container, upserted


def _make_client(*, processor=None) -> AsyncCosmosMemoryClient:
    client = AsyncCosmosMemoryClient(use_default_credential=False, processor=processor)
    client._container_client = MagicMock()
    return client


@pytest.fixture
def inline_to_thread():
    async def _inline(func, /, *args, **kwargs):
        return func(*args, **kwargs)

    with patch("agent_memory_toolkit.aio.cosmos_memory_client.asyncio.to_thread", new=_inline):
        yield


@pytest.mark.asyncio
async def test_async_synthesize_procedural_first_synthesis(inline_to_thread):
    fact_docs = [
        _fact_doc("f1", "Always use bullet points.", category="preference"),
        _fact_doc("f2", "Never use var in TypeScript.", category="requirement"),
    ]
    episodic_docs = [_episodic_doc("e1", lesson="Keep examples small.")]
    pipeline, _, upserted = _make_synthesis_pipeline(
        fact_docs=fact_docs,
        episodic_docs=episodic_docs,
        llm_output="Async prompt",
    )
    client = _make_client()
    client._pipeline = pipeline

    result = await client.synthesize_procedural("u1", force=False)

    assert result["status"] == "synthesized"
    doc = result["procedural"]
    assert doc["version"] == 1
    assert doc["content"] == "Async prompt"
    assert set(doc["source_fact_ids"]) == {"f1", "f2"}
    assert set(doc["source_episodic_ids"]) == {"e1"}
    assert upserted == [doc]


@pytest.mark.asyncio
async def test_async_synthesize_procedural_resynthesis_supersedes_prior(inline_to_thread):
    prior_doc = _procedural_doc(
        "proc_u1_1",
        version=1,
        content="Old prompt",
        source_fact_ids=["f1", "f2"],
        source_episodic_ids=["e1"],
        ts=1,
    )
    fact_docs = [
        _fact_doc("f1", "Always use bullet points.", category="preference"),
        _fact_doc("f2", "Never use var in TypeScript.", category="requirement"),
        _fact_doc("f3", "Lead with the answer.", category="preference"),
    ]
    episodic_docs = [_episodic_doc("e1", lesson="Keep examples small.")]
    pipeline, container, upserted = _make_synthesis_pipeline(
        prior_docs=[prior_doc],
        fact_docs=fact_docs,
        episodic_docs=episodic_docs,
        llm_output="Updated prompt",
    )
    client = _make_client()
    client._pipeline = pipeline

    result = await client.synthesize_procedural("u1")

    assert result["status"] == "synthesized"
    new_doc = result["procedural"]
    assert new_doc["version"] == 2
    assert new_doc["supersedes_ids"] == [prior_doc["id"]]
    assert upserted == [new_doc]
    body = container.replace_item.call_args.kwargs["body"]
    assert body["superseded_by"] == new_doc["id"]
    _assert_iso8601(body["superseded_at"])
    assert body["supersede_reason"] == "update"


@pytest.mark.asyncio
async def test_async_synthesize_procedural_noop_when_source_ids_are_unchanged(inline_to_thread):
    prior_doc = _procedural_doc(
        "proc_u1_1",
        version=1,
        content="Existing prompt",
        source_fact_ids=["f1", "f2"],
        source_episodic_ids=["e1"],
        ts=1,
    )
    fact_docs = [
        _fact_doc("f2", "Never use var in TypeScript.", category="requirement"),
        _fact_doc("f1", "Always use bullet points.", category="preference"),
    ]
    episodic_docs = [_episodic_doc("e1", lesson="Keep examples small.")]
    pipeline, container, _ = _make_synthesis_pipeline(
        prior_docs=[prior_doc],
        fact_docs=fact_docs,
        episodic_docs=episodic_docs,
    )
    client = _make_client()
    client._pipeline = pipeline

    result = await client.synthesize_procedural("u1", force=False)

    assert result == {"status": "unchanged", "procedural": prior_doc}
    pipeline._run_prompty.assert_not_called()
    container.upsert_item.assert_not_called()
    container.replace_item.assert_not_called()


@pytest.mark.asyncio
async def test_async_synthesize_procedural_force_true_reruns(inline_to_thread):
    prior_doc = _procedural_doc(
        "proc_u1_1",
        version=1,
        content="Existing prompt",
        source_fact_ids=["f1", "f2"],
        source_episodic_ids=["e1"],
        ts=1,
    )
    fact_docs = [
        _fact_doc("f1", "Always use bullet points.", category="preference"),
        _fact_doc("f2", "Never use var in TypeScript.", category="requirement"),
    ]
    episodic_docs = [_episodic_doc("e1", lesson="Keep examples small.")]
    pipeline, container, upserted = _make_synthesis_pipeline(
        prior_docs=[prior_doc],
        fact_docs=fact_docs,
        episodic_docs=episodic_docs,
        llm_output="Refreshed async prompt",
    )
    client = _make_client()
    client._pipeline = pipeline

    result = await client.synthesize_procedural("u1", force=True)

    assert result["status"] == "synthesized"
    new_doc = result["procedural"]
    assert new_doc["version"] == 2
    assert upserted == [new_doc]
    body = container.replace_item.call_args.kwargs["body"]
    assert body["superseded_by"] == new_doc["id"]
    assert body["supersede_reason"] == "update"


@pytest.mark.asyncio
async def test_async_get_procedural_prompt_returns_none_when_missing():
    client = _make_client()
    client._container_client.query_items = MagicMock(return_value=AsyncIterator([]))

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

    client._container_client.query_items = MagicMock(side_effect=_query_items)

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
    client._container_client.query_items = MagicMock(return_value=AsyncIterator([v1, v3, v2]))

    history = await client.get_procedural_history("u1", limit=10)

    assert [doc["id"] for doc in history] == ["proc_u1_3", "proc_u1_2", "proc_u1_1"]


@pytest.mark.asyncio
async def test_async_client_synthesize_procedural_defers_remote_processors():
    client = _make_client(processor=AsyncDurableFunctionProcessor())
    client._pipeline = MagicMock()

    result = await client.synthesize_procedural("u1")

    assert result["status"] == "deferred"
    assert result["reason"] == "durable_auto_trigger"
    assert isinstance(result["message"], str)
    assert result["message"]
    client._pipeline.synthesize_procedural.assert_not_called()
