from __future__ import annotations

import json
from typing import Any

import pytest

from agent_memory_toolkit._container_routing import ContainerKey
from agent_memory_toolkit.aio.services.pipeline import AsyncPipelineService, _AsyncStoreContainerAdapter
from agent_memory_toolkit.services.pipeline import PipelineService, _StoreContainerAdapter


class _SyncChat:
    def __init__(self, responses: list[dict[str, Any]]):
        self.responses = list(responses)
        self.calls = 0

    def generate(self, messages: list[dict[str, Any]], **opts: Any) -> str:
        del messages, opts
        self.calls += 1
        return json.dumps(self.responses.pop(0))


class _SyncEmbeddings:
    def __init__(self):
        self.calls: list[list[str]] = []

    def generate_batch(self, texts: list[str]) -> list[list[float]]:
        self.calls.append(list(texts))
        return [[1.0] for _ in texts]

    def generate(self, text: str) -> list[float]:
        self.calls.append([text])
        return [1.0]


class _AsyncChat:
    def __init__(self, responses: list[dict[str, Any]]):
        self.responses = list(responses)
        self.calls = 0

    async def generate(self, messages: list[dict[str, Any]], **opts: Any) -> str:
        del messages, opts
        self.calls += 1
        return json.dumps(self.responses.pop(0))


class _AsyncEmbeddings(_SyncEmbeddings):
    async def generate_batch(self, texts: list[str]) -> list[list[float]]:
        self.calls.append(list(texts))
        return [[1.0] for _ in texts]

    async def generate(self, text: str) -> list[float]:
        self.calls.append([text])
        return [1.0]


class _Store:
    def __init__(self, docs: list[dict[str, Any]]):
        self.docs = [dict(doc) for doc in docs]

    def query(self, sql: str, parameters=None, partition_key=None, cross_partition: bool = False):
        del partition_key, cross_partition
        params = {p["name"]: p["value"] for p in (parameters or [])}
        docs = [dict(doc) for doc in self.docs]
        if "@user_id" in params:
            docs = [doc for doc in docs if doc.get("user_id") == params["@user_id"]]
        if "@thread_id" in params:
            docs = [doc for doc in docs if doc.get("thread_id") == params["@thread_id"]]
        if "c.type IN" in sql:
            types = {value for name, value in params.items() if name.startswith("@mtype")}
            docs = [doc for doc in docs if doc.get("type") in types]
        if "superseded_by" in sql:
            docs = [doc for doc in docs if not doc.get("superseded_by")]
        return docs

    def read_item(self, item_id: str, partition_key: Any):
        del partition_key
        for doc in self.docs:
            if doc.get("id") == item_id:
                return dict(doc)
        raise KeyError(item_id)

    def add_cosmos(self, record: dict[str, Any]) -> dict[str, Any]:
        self.docs.append(dict(record))
        return record

    def mark_superseded(self, old_doc: dict[str, Any], superseder_id: str, *, reason: str) -> bool:
        del old_doc, superseder_id, reason
        return True


class _AsyncStore(_Store):
    async def query(self, sql: str, parameters=None, partition_key=None, cross_partition: bool = False):
        return super().query(sql, parameters=parameters, partition_key=partition_key, cross_partition=cross_partition)

    async def read_item(self, item_id: str, partition_key: Any):
        return super().read_item(item_id, partition_key)

    async def add_cosmos(self, record: dict[str, Any]) -> dict[str, Any]:
        return super().add_cosmos(record)

    async def mark_superseded(self, old_doc: dict[str, Any], superseder_id: str, *, reason: str) -> bool:
        return super().mark_superseded(old_doc, superseder_id, reason=reason)


def _containers_for_store(
    memories_store: _Store,
    *,
    turns_store: _Store | None = None,
    summaries_store: _Store | None = None,
) -> dict[ContainerKey, _StoreContainerAdapter]:
    turns_store = turns_store or _Store([])
    summaries_store = summaries_store or _Store([])
    return {
        ContainerKey.TURNS: _StoreContainerAdapter(turns_store, ContainerKey.TURNS),
        ContainerKey.MEMORIES: _StoreContainerAdapter(memories_store, ContainerKey.MEMORIES),
        ContainerKey.SUMMARIES: _StoreContainerAdapter(summaries_store, ContainerKey.SUMMARIES),
    }


def _async_containers_for_store(
    memories_store: _AsyncStore,
    *,
    turns_store: _AsyncStore | None = None,
    summaries_store: _AsyncStore | None = None,
) -> dict[ContainerKey, _AsyncStoreContainerAdapter]:
    turns_store = turns_store or _AsyncStore([])
    summaries_store = summaries_store or _AsyncStore([])
    return {
        ContainerKey.TURNS: _AsyncStoreContainerAdapter(turns_store, ContainerKey.TURNS),
        ContainerKey.MEMORIES: _AsyncStoreContainerAdapter(memories_store, ContainerKey.MEMORIES),
        ContainerKey.SUMMARIES: _AsyncStoreContainerAdapter(summaries_store, ContainerKey.SUMMARIES),
    }


def _turn(i: int) -> dict[str, Any]:
    return {
        "id": f"turn-{i}",
        "user_id": "u1",
        "thread_id": "t1",
        "role": "user",
        "type": "turn",
        "content": f"Turn {i}: I prefer dark mode and stable retries.",
        "created_at": f"2025-01-01T00:{i:02d}:00+00:00",
    }


def _response() -> dict[str, Any]:
    return {
        "facts": [
            {
                "text": "The user prefers dark mode.",
                "action": "ADD",
                "category": "preference",
                "confidence": 0.9,
                "salience": 0.8,
                "tags": ["ui"],
            }
        ],
        "episodic": [
            {
                "scope_type": "project",
                "scope_value": "CI",
                "summary": "CI retries resolved flaky tests.",
                "lesson": "Use retries for flaky CI tests.",
                "confidence": 0.8,
            }
        ],
    }


def test_extract_memories_dry_shape_is_small_and_has_no_embeddings() -> None:
    chat = _SyncChat([_response()])
    embeddings = _SyncEmbeddings()
    memories_store = _Store([])
    turns_store = _Store([_turn(i) for i in range(50)])
    service = PipelineService(
        memories_store,
        chat,
        embeddings,
        containers=_containers_for_store(memories_store, turns_store=turns_store),
    )

    output = service.extract_memories_dry("u1", "t1")

    assert set(output) == {"facts", "episodic", "updates"}
    assert len(json.dumps(output)) < 32 * 1024
    assert output["facts"] and output["episodic"]
    assert all("embedding" not in doc for docs in (output["facts"], output["episodic"]) for doc in docs)
    assert embeddings.calls == []


def test_extract_memories_dry_is_byte_deterministic_for_same_llm_response() -> None:
    store = _Store([])
    turns_store = _Store([_turn(1)])
    service = PipelineService(
        store,
        _SyncChat([_response(), _response()]),
        _SyncEmbeddings(),
        containers=_containers_for_store(store, turns_store=turns_store),
    )

    first = service.extract_memories_dry("u1", "t1")
    second = service.extract_memories_dry("u1", "t1")

    assert json.dumps(first, sort_keys=True, separators=(",", ":")) == json.dumps(
        second, sort_keys=True, separators=(",", ":")
    )


@pytest.mark.asyncio
async def test_async_extract_memories_dry_shape_is_small_and_has_no_embeddings() -> None:
    chat = _AsyncChat([_response()])
    embeddings = _AsyncEmbeddings()
    memories_store = _AsyncStore([])
    turns_store = _AsyncStore([_turn(i) for i in range(50)])
    service = AsyncPipelineService(
        memories_store,
        chat,
        embeddings,
        containers=_async_containers_for_store(memories_store, turns_store=turns_store),
    )

    output = await service.extract_memories_dry("u1", "t1")

    assert set(output) == {"facts", "episodic", "updates"}
    assert len(json.dumps(output)) < 32 * 1024
    assert all("embedding" not in doc for docs in (output["facts"], output["episodic"]) for doc in docs)
    assert embeddings.calls == []


@pytest.mark.asyncio
async def test_async_extract_memories_dry_is_byte_deterministic_for_same_llm_response() -> None:
    store = _AsyncStore([])
    turns_store = _AsyncStore([_turn(1)])
    service = AsyncPipelineService(
        store,
        _AsyncChat([_response(), _response()]),
        _AsyncEmbeddings(),
        containers=_async_containers_for_store(store, turns_store=turns_store),
    )

    first = await service.extract_memories_dry("u1", "t1")
    second = await service.extract_memories_dry("u1", "t1")

    assert json.dumps(first, sort_keys=True, separators=(",", ":")) == json.dumps(
        second, sort_keys=True, separators=(",", ":")
    )
