from __future__ import annotations

import hashlib
from typing import Any
from unittest.mock import MagicMock

import pytest
from azure.cosmos.exceptions import CosmosResourceExistsError

from agent_memory_toolkit._container_routing import ContainerKey
from agent_memory_toolkit._utils import compute_content_hash
from agent_memory_toolkit.aio.services.pipeline import AsyncPipelineService, _AsyncStoreContainerAdapter
from agent_memory_toolkit.services._pipeline_helpers import ID_SEED_SEP
from agent_memory_toolkit.services.pipeline import PipelineService, _StoreContainerAdapter


class _Container:
    def __init__(self, *, raise_for_ids: set[str] | None = None):
        self.docs: dict[str, dict[str, Any]] = {}
        self.created_ids: list[str] = []
        self.create_attempts = 0
        self.raise_for_ids = set(raise_for_ids or set())

    def create_item(self, *, body: dict[str, Any]) -> dict[str, Any]:
        self.create_attempts += 1
        item_id = body["id"]
        if item_id in self.raise_for_ids or item_id in self.docs:
            raise CosmosResourceExistsError(message="exists")
        self.docs[item_id] = dict(body)
        self.created_ids.append(item_id)
        return dict(body)


class _AsyncContainer(_Container):
    async def create_item(self, *, body: dict[str, Any]) -> dict[str, Any]:
        return super().create_item(body=body)


class _Store:
    def __init__(self, container: _Container):
        self.container = container
        self.upserts: list[dict[str, Any]] = []

    def query(self, *args, **kwargs):
        return []

    def read_item(self, item_id: str, partition_key: Any):
        del partition_key
        return self.container.docs[item_id]

    def add_cosmos(self, record: dict[str, Any]) -> dict[str, Any]:
        self.upserts.append(dict(record))
        self.container.docs[record["id"]] = dict(record)
        return record

    def mark_superseded(self, old_doc: dict[str, Any], superseder_id: str, *, reason: str) -> bool:
        old_doc["superseded_by"] = superseder_id
        old_doc["supersede_reason"] = reason
        return True


class _AsyncStore(_Store):
    async def query(self, *args, **kwargs):
        return []

    async def read_item(self, item_id: str, partition_key: Any):
        return super().read_item(item_id, partition_key)

    async def add_cosmos(self, record: dict[str, Any]) -> dict[str, Any]:
        return super().add_cosmos(record)

    async def mark_superseded(self, old_doc: dict[str, Any], superseder_id: str, *, reason: str) -> bool:
        return super().mark_superseded(old_doc, superseder_id, reason=reason)


def _containers_for_store(store: _Store) -> dict[ContainerKey, _StoreContainerAdapter]:
    return {
        ContainerKey.TURNS: _StoreContainerAdapter(_Store(_Container()), ContainerKey.TURNS),
        ContainerKey.MEMORIES: _StoreContainerAdapter(store, ContainerKey.MEMORIES),
        ContainerKey.SUMMARIES: _StoreContainerAdapter(_Store(_Container()), ContainerKey.SUMMARIES),
    }


def _async_containers_for_store(store: _AsyncStore) -> dict[ContainerKey, _AsyncStoreContainerAdapter]:
    return {
        ContainerKey.TURNS: _AsyncStoreContainerAdapter(_AsyncStore(_AsyncContainer()), ContainerKey.TURNS),
        ContainerKey.MEMORIES: _AsyncStoreContainerAdapter(store, ContainerKey.MEMORIES),
        ContainerKey.SUMMARIES: _AsyncStoreContainerAdapter(_AsyncStore(_AsyncContainer()), ContainerKey.SUMMARIES),
    }


class _Embeddings:
    def __init__(self):
        self.batch_calls: list[list[str]] = []

    def generate_batch(self, texts: list[str]) -> list[list[float]]:
        self.batch_calls.append(list(texts))
        return [[float(i)] for i, _ in enumerate(texts, start=1)]

    def generate(self, text: str) -> list[float]:
        return [0.0]


class _AsyncEmbeddings(_Embeddings):
    async def generate_batch(self, texts: list[str]) -> list[list[float]]:
        return super().generate_batch(texts)

    async def generate(self, text: str) -> list[float]:
        return [0.0]


def _fact_doc(content: str = "The user prefers dark mode.") -> dict[str, Any]:
    content_hash = compute_content_hash(content)
    seed = ID_SEED_SEP.join(("u1", "t1", content_hash))
    return {
        "id": f"fact_{hashlib.sha256(seed.encode()).hexdigest()[:32]}",
        "user_id": "u1",
        "thread_id": "t1",
        "role": "system",
        "type": "fact",
        "content": content,
        "content_hash": content_hash,
        "confidence": 0.9,
        "salience": 0.8,
        "tags": ["sys:fact", "sys:auto-extracted"],
        "prompt_id": "extract_memories.prompty",
        "prompt_version": "v1",
        "metadata": {"category": "preference"},
        "created_at": "2025-01-01T00:00:00+00:00",
        "updated_at": "2025-01-01T00:00:00+00:00",
    }


def test_persist_extracted_memories_uses_deterministic_ids_and_skips_replay() -> None:
    container = _Container()
    store = _Store(container)
    service = PipelineService(
        store,
        chat_client=object(),
        embeddings_client=_Embeddings(),
        containers=_containers_for_store(store),
    )
    doc = _fact_doc()

    first = service.persist_extracted_memories("u1", {"facts": [doc], "episodic": [], "updates": []})
    second = service.persist_extracted_memories("u1", {"facts": [doc], "episodic": [], "updates": []})

    assert doc["id"] in container.docs
    assert first["fact_count"] == 1
    assert second["fact_count"] == 0
    assert container.created_ids == [doc["id"]]


def test_persist_extracted_memories_routes_facts_to_memories_container() -> None:
    turns_container = MagicMock()
    memories_container = MagicMock()
    summaries_container = MagicMock()
    memories_container.create_item.side_effect = lambda body: body
    containers = {
        ContainerKey.TURNS: turns_container,
        ContainerKey.MEMORIES: memories_container,
        ContainerKey.SUMMARIES: summaries_container,
    }
    service = PipelineService(
        _Store(_Container()),
        chat_client=object(),
        embeddings_client=_Embeddings(),
        containers=containers,
    )
    doc = _fact_doc()

    result = service.persist_extracted_memories("u1", {"facts": [doc], "episodic": [], "updates": []})

    assert result["fact_count"] == 1
    memories_container.create_item.assert_called_once()
    assert memories_container.create_item.call_args.kwargs["body"]["type"] == "fact"
    turns_container.method_calls == []
    summaries_container.method_calls == []


def test_persist_extracted_memories_409_skip_continues_to_next_doc() -> None:
    first = _fact_doc("The user prefers dark mode.")
    second = _fact_doc("The user prefers concise answers.")
    container = _Container(raise_for_ids={first["id"]})
    store = _Store(container)
    service = PipelineService(
        store,
        chat_client=object(),
        embeddings_client=_Embeddings(),
        containers=_containers_for_store(store),
    )

    result = service.persist_extracted_memories("u1", {"facts": [first, second], "episodic": [], "updates": []})

    assert result["fact_count"] == 1
    assert container.created_ids == [second["id"]]


@pytest.mark.asyncio
async def test_async_persist_extracted_memories_uses_deterministic_ids_and_skips_replay() -> None:
    container = _AsyncContainer()
    store = _AsyncStore(container)
    service = AsyncPipelineService(
        store,
        chat_client=object(),
        embeddings_client=_AsyncEmbeddings(),
        containers=_async_containers_for_store(store),
    )
    doc = _fact_doc()

    first = await service.persist_extracted_memories("u1", {"facts": [doc], "episodic": [], "updates": []})
    second = await service.persist_extracted_memories("u1", {"facts": [doc], "episodic": [], "updates": []})

    assert doc["id"] in container.docs
    assert first["fact_count"] == 1
    assert second["fact_count"] == 0
    assert container.created_ids == [doc["id"]]


@pytest.mark.asyncio
async def test_async_persist_extracted_memories_409_skip_continues_to_next_doc() -> None:
    first = _fact_doc("The user prefers dark mode.")
    second = _fact_doc("The user prefers concise answers.")
    container = _AsyncContainer(raise_for_ids={first["id"]})
    store = _AsyncStore(container)
    service = AsyncPipelineService(
        store,
        chat_client=object(),
        embeddings_client=_AsyncEmbeddings(),
        containers=_async_containers_for_store(store),
    )

    result = await service.persist_extracted_memories("u1", {"facts": [first, second], "episodic": [], "updates": []})

    assert result["fact_count"] == 1
    assert container.created_ids == [second["id"]]
