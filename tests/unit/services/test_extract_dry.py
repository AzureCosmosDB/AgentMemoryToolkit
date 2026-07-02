from __future__ import annotations

import json
from typing import Any
from unittest.mock import AsyncMock

import pytest

from azure.cosmos.agent_memory._container_routing import ContainerKey
from azure.cosmos.agent_memory.aio.services.pipeline import AsyncPipelineService, _AsyncStoreContainerAdapter
from azure.cosmos.agent_memory.services.pipeline import PipelineService, _StoreContainerAdapter


class _SyncChat:
    def __init__(self, responses: list[dict[str, Any]]):
        self.responses = list(responses)
        self.calls = 0
        self.messages: list[list[dict[str, Any]]] = []

    def generate(self, messages: list[dict[str, Any]], **opts: Any) -> str:
        del opts
        self.calls += 1
        self.messages.append(messages)
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
        self.search_calls: list[dict[str, Any]] = []
        self.search_results: list[dict[str, Any]] = []

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
        if "extracted_at" in sql:
            docs = [doc for doc in docs if not doc.get("extracted_at")]
        return docs

    def upsert_item(self, *, body: dict[str, Any]) -> dict[str, Any]:
        body = dict(body)
        doc_id = body.get("id")
        for i, doc in enumerate(self.docs):
            if doc.get("id") == doc_id:
                self.docs[i] = body
                return body
        self.docs.append(body)
        return body

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

    def search(
        self,
        *,
        search_terms: str,
        user_id: str,
        memory_types: list[str],
        top_k: int,
        include_superseded: bool = False,
    ) -> list[dict[str, Any]]:
        self.search_calls.append(
            {
                "search_terms": search_terms,
                "user_id": user_id,
                "memory_types": memory_types,
                "top_k": top_k,
                "include_superseded": include_superseded,
            }
        )
        return [dict(doc) for doc in self.search_results]


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
                "text": "CI retries resolved flaky tests.",
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

    assert set(output) == {"facts", "episodic", "updates", "processed_turn_docs"}
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


def test_extract_memories_dry_stage1_searches_user_turn_text_by_default() -> None:
    chat = _SyncChat([_response()])
    memories_store = _Store([])
    memories_store.search_results = [
        {
            "id": "memory-hybrid",
            "content": "Existing hybrid memory from search.",
            "type": "fact",
            "salience": 0.7,
        }
    ]
    turns = [_turn(1), _turn(2)]
    turns_store = _Store(turns)
    service = PipelineService(
        memories_store,
        chat,
        _SyncEmbeddings(),
        containers=_containers_for_store(memories_store, turns_store=turns_store),
    )

    service.extract_memories_dry("u1", "t1")

    assert memories_store.search_calls == [
        {
            "search_terms": "\n".join(turn["content"] for turn in turns),
            "user_id": "u1",
            "memory_types": ["fact"],
            "top_k": 10,
            "include_superseded": False,
        }
    ]
    assert "Existing hybrid memory from search." in json.dumps(chat.messages)


def test_extract_memories_dry_stage1_falls_back_to_transcript_without_user_turns() -> None:
    memories_store = _Store([])
    turns = [
        {
            **_turn(1),
            "id": "assistant-turn-1",
            "role": "assistant",
            "content": "Assistant response with no user-role content.",
        }
    ]
    service = PipelineService(
        memories_store,
        _SyncChat([_response()]),
        _SyncEmbeddings(),
        containers=_containers_for_store(memories_store, turns_store=_Store(turns)),
    )

    service.extract_memories_dry("u1", "t1")

    assert memories_store.search_calls == [
        {
            "search_terms": service._build_transcript(turns),
            "user_id": "u1",
            "memory_types": ["fact"],
            "top_k": 10,
            "include_superseded": False,
        }
    ]


def test_extract_memories_dry_stage1_legacy_context_does_not_call_search(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("azure.cosmos.agent_memory.thresholds.get_dedup_context_vector_enabled", lambda: False)
    chat = _SyncChat([_response()])
    memories_store = _Store(
        [
            {
                "id": "legacy-memory",
                "user_id": "u1",
                "type": "fact",
                "content": "Existing legacy memory from load.",
                "salience": 0.6,
            }
        ]
    )
    service = PipelineService(
        memories_store,
        chat,
        _SyncEmbeddings(),
        containers=_containers_for_store(memories_store, turns_store=_Store([_turn(1)])),
    )

    service.extract_memories_dry("u1", "t1")

    assert memories_store.search_calls == []
    assert "Existing legacy memory from load." in json.dumps(chat.messages)


@pytest.mark.asyncio
async def test_async_extract_memories_dry_shape_is_small_and_has_no_embeddings(monkeypatch) -> None:
    monkeypatch.setattr(
        "azure.cosmos.agent_memory.aio.services.pipeline.get_dedup_context_vector_enabled", lambda: False
    )
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

    assert set(output) == {"facts", "episodic", "updates", "processed_turn_docs"}
    assert len(json.dumps(output)) < 32 * 1024
    assert all("embedding" not in doc for docs in (output["facts"], output["episodic"]) for doc in docs)
    assert embeddings.calls == []


@pytest.mark.asyncio
async def test_async_extract_memories_dry_is_byte_deterministic_for_same_llm_response(monkeypatch) -> None:
    monkeypatch.setattr(
        "azure.cosmos.agent_memory.aio.services.pipeline.get_dedup_context_vector_enabled", lambda: False
    )
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


@pytest.mark.asyncio
async def test_async_extract_memories_dry_stage1_searches_user_turn_text_by_default() -> None:
    store = _AsyncStore([])
    store.search = AsyncMock(
        return_value=[
            {
                "id": "fact-1",
                "content": "The user prefers dark mode.",
                "type": "fact",
                "salience": 0.7,
            }
        ]
    )
    turns_store = _AsyncStore([_turn(1)])
    service = AsyncPipelineService(
        store,
        _AsyncChat([_response()]),
        _AsyncEmbeddings(),
        containers=_async_containers_for_store(store, turns_store=turns_store),
    )
    service._vector_candidates = AsyncMock(return_value=[])
    turns = [_turn(1)]

    await service.extract_memories_dry("u1", "t1")

    store.search.assert_awaited_once_with(
        search_terms="\n".join(turn["content"] for turn in turns),
        user_id="u1",
        memory_types=["fact"],
        top_k=10,
    )
    service._vector_candidates.assert_not_awaited()


@pytest.mark.asyncio
async def test_async_extract_memories_dry_stage1_falls_back_to_transcript_without_user_turns() -> None:
    store = _AsyncStore([])
    store.search = AsyncMock(return_value=[])
    turns = [
        {
            **_turn(1),
            "id": "assistant-turn-1",
            "role": "assistant",
            "content": "Assistant response with no user-role content.",
        }
    ]
    turns_store = _AsyncStore(turns)
    service = AsyncPipelineService(
        store,
        _AsyncChat([_response()]),
        _AsyncEmbeddings(),
        containers=_async_containers_for_store(store, turns_store=turns_store),
    )
    transcript = service._build_transcript(turns)

    await service.extract_memories_dry("u1", "t1")

    store.search.assert_awaited_once_with(
        search_terms=transcript,
        user_id="u1",
        memory_types=["fact"],
        top_k=10,
    )


@pytest.mark.asyncio
async def test_async_extract_memories_dry_stage1_legacy_path_when_context_vector_unset(monkeypatch) -> None:
    monkeypatch.setattr(
        "azure.cosmos.agent_memory.aio.services.pipeline.get_dedup_context_vector_enabled", lambda: False
    )
    store = _AsyncStore(
        [
            {
                "id": "fact-1",
                "user_id": "u1",
                "type": "fact",
                "content": "Existing fact.",
                "content_hash": "hash-1",
            }
        ]
    )
    store.search = AsyncMock(return_value=[])
    turns_store = _AsyncStore([_turn(1)])
    service = AsyncPipelineService(
        store,
        _AsyncChat([_response()]),
        _AsyncEmbeddings(),
        containers=_async_containers_for_store(store, turns_store=turns_store),
    )

    await service.extract_memories_dry("u1", "t1")

    store.search.assert_not_awaited()


def test_extract_memories_dry_stage1_search_failure_falls_back_to_hash_memories() -> None:
    """A failing dedup-context vector search must not abort extraction; it
    falls back to the hash-loaded existing memories (option 3 resilience)."""
    chat = _SyncChat([_response()])
    memories_store = _Store(
        [
            {
                "id": "hash-memory",
                "user_id": "u1",
                "type": "fact",
                "content": "Existing hash-based memory from load.",
                "content_hash": "h1",
                "salience": 0.6,
            }
        ]
    )

    def _boom(**_kwargs):
        raise RuntimeError("vector search down")

    memories_store.search = _boom
    service = PipelineService(
        memories_store,
        chat,
        _SyncEmbeddings(),
        containers=_containers_for_store(memories_store, turns_store=_Store([_turn(1)])),
    )

    # Must not raise despite the Stage-1 vector search failing.
    output = service.extract_memories_dry("u1", "t1")

    assert output["facts"]  # extraction proceeded
    # Fallback surfaced the hash-loaded memory into the extraction prompt.
    assert "Existing hash-based memory from load." in json.dumps(chat.messages)


@pytest.mark.asyncio
async def test_async_extract_memories_dry_stage1_search_failure_falls_back_to_hash_memories() -> None:
    class _RecordingAsyncChat(_AsyncChat):
        def __init__(self, responses):
            super().__init__(responses)
            self.messages: list = []

        async def generate(self, messages, **opts):
            self.messages.append(messages)
            del opts
            self.calls += 1
            return json.dumps(self.responses.pop(0))

    chat = _RecordingAsyncChat([_response()])
    store = _AsyncStore(
        [
            {
                "id": "hash-memory",
                "user_id": "u1",
                "type": "fact",
                "content": "Existing hash-based memory from load.",
                "content_hash": "h1",
                "salience": 0.6,
            }
        ]
    )
    store.search = AsyncMock(side_effect=RuntimeError("vector search down"))
    service = AsyncPipelineService(
        store,
        chat,
        _AsyncEmbeddings(),
        containers=_async_containers_for_store(store, turns_store=_AsyncStore([_turn(1)])),
    )

    output = await service.extract_memories_dry("u1", "t1")

    assert output["facts"]  # extraction proceeded
    store.search.assert_awaited_once()
    assert "Existing hash-based memory from load." in json.dumps(chat.messages)
