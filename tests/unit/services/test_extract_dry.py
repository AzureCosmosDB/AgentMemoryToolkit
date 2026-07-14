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


def test_extract_memories_dry_does_not_call_store_search() -> None:
    """Extraction is single-pass and existing-memory-free: it must not issue a
    dedup-context vector search (dedup is handled by hash + reconciliation)."""
    memories_store = _Store([])
    memories_store.search_results = [{"id": "should-not-appear", "content": "x", "type": "fact"}]
    service = PipelineService(
        memories_store,
        _SyncChat([_response()]),
        _SyncEmbeddings(),
        containers=_containers_for_store(memories_store, turns_store=_Store([_turn(1)])),
    )

    service.extract_memories_dry("u1", "t1")

    assert memories_store.search_calls == []


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

    assert set(output) == {"facts", "episodic", "updates", "processed_turn_docs"}
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


@pytest.mark.asyncio
async def test_async_extract_memories_dry_does_not_call_store_search() -> None:
    store = _AsyncStore([])

    store.search = AsyncMock(return_value=[])
    service = AsyncPipelineService(
        store,
        _AsyncChat([_response()]),
        _AsyncEmbeddings(),
        containers=_async_containers_for_store(store, turns_store=_AsyncStore([_turn(1)])),
    )

    await service.extract_memories_dry("u1", "t1")

    store.search.assert_not_awaited()


class _BatchChat:
    """Returns a unique fact per call; optionally raises on a chosen call index."""

    def __init__(self, fail_on_call=None, error=None):
        self.calls = 0
        self.fail_on_call = fail_on_call
        self.error = error

    def generate(self, messages, **opts):
        del messages, opts
        self.calls += 1
        if self.fail_on_call and self.calls == self.fail_on_call:
            raise self.error
        return json.dumps(
            {
                "facts": [
                    {
                        "text": f"Fact from call {self.calls}.",
                        "category": "other",
                        "confidence": 0.9,
                        "salience": 0.8,
                        "temporal_context": None,
                        "tags": [],
                    }
                ],
                "episodic": [],
            }
        )


def _one_turn_per_batch(monkeypatch):
    # Force each small turn into its own extraction batch.
    monkeypatch.setattr("azure.cosmos.agent_memory.thresholds.get_extraction_batch_max_tokens", lambda: 5)


def test_extract_batches_run_independently_one_call_per_batch(monkeypatch) -> None:
    _one_turn_per_batch(monkeypatch)
    chat = _BatchChat()
    memories_store = _Store([])
    turns_store = _Store([_turn(i) for i in range(3)])
    service = PipelineService(
        memories_store,
        chat,
        _SyncEmbeddings(),
        containers=_containers_for_store(memories_store, turns_store=turns_store),
    )

    out = service.extract_memories_dry("u1", "t1")

    assert chat.calls == 3  # one LLM call per batch
    assert len(out["facts"]) == 3
    assert len(out["processed_turn_docs"]) == 3


def test_extract_quarantines_non_retryable_batch_but_keeps_others(monkeypatch) -> None:
    _one_turn_per_batch(monkeypatch)
    chat = _BatchChat(fail_on_call=2, error=Exception("Error code: 400 content_filter"))
    memories_store = _Store([])
    turns_store = _Store([_turn(i) for i in range(3)])
    service = PipelineService(
        memories_store,
        chat,
        _SyncEmbeddings(),
        containers=_containers_for_store(memories_store, turns_store=turns_store),
    )

    out = service.extract_memories_dry("u1", "t1")

    # Batches 1 and 3 produced facts; batch 2 was quarantined (no fact) ...
    assert len(out["facts"]) == 2
    # ... but ALL 3 turns are stamped (quarantined turn included) so it never re-poisons.
    assert len(out["processed_turn_docs"]) == 3
    stats = [u for u in out["updates"] if u.get("op") == "stats" and "quarantined_turn_count" in u]
    assert stats and stats[0]["quarantined_turn_count"] == 1


def test_extract_defers_retryable_batch_leaving_turns_unstamped(monkeypatch) -> None:
    _one_turn_per_batch(monkeypatch)
    chat = _BatchChat(fail_on_call=2, error=Exception("Error code: 429 rate limit"))
    memories_store = _Store([])
    turns_store = _Store([_turn(i) for i in range(3)])
    service = PipelineService(
        memories_store,
        chat,
        _SyncEmbeddings(),
        containers=_containers_for_store(memories_store, turns_store=turns_store),
    )

    out = service.extract_memories_dry("u1", "t1")

    # Batches 1 and 3 produced facts; batch 2 deferred (retryable) ...
    assert len(out["facts"]) == 2
    # ... its turn is NOT in processed_turn_docs, so it stays un-extracted and retries.
    assert len(out["processed_turn_docs"]) == 2
    stats = [u for u in out["updates"] if u.get("op") == "stats" and "deferred_turn_count" in u]
    assert stats and stats[0]["deferred_turn_count"] == 1


def _agent_source_response() -> dict[str, Any]:
    # One agent-sourced fact, one user-sourced fact, one with source omitted
    # (must default to "user"). Mirrors the extract_memories.prompty schema.
    return {
        "facts": [
            {
                "text": "The agent booked the user on flight TP204 to Lisbon.",
                "category": "other",
                "source": "agent",
                "confidence": 0.95,
                "salience": 0.8,
                "temporal_context": None,
                "tags": ["travel"],
            },
            {
                "text": "The user prefers window seats.",
                "category": "preference",
                "source": "user",
                "confidence": 0.9,
                "salience": 0.7,
                "temporal_context": None,
                "tags": ["travel"],
            },
            {
                "text": "The user lives in Seattle.",
                "category": "biographical",
                "confidence": 0.9,
                "salience": 0.7,
                "temporal_context": None,
                "tags": ["identity"],
            },
        ],
        "episodic": [],
    }


def _fact_by_text(facts: list[dict[str, Any]], needle: str) -> dict[str, Any]:
    for doc in facts:
        if needle in doc["content"]:
            return doc
    raise AssertionError(f"no fact containing {needle!r}")


def test_agent_sourced_fact_is_tagged_and_stamped() -> None:
    memories_store = _Store([])
    service = PipelineService(
        memories_store,
        _SyncChat([_agent_source_response()]),
        _SyncEmbeddings(),
        containers=_containers_for_store(memories_store, turns_store=_Store([_turn(1)])),
    )

    out = service.extract_memories_dry("u1", "t1")

    agent_fact = _fact_by_text(out["facts"], "booked the user on flight")
    assert agent_fact["metadata"]["source"] == "agent"
    assert "sys:agent-fact" in agent_fact["tags"]

    user_fact = _fact_by_text(out["facts"], "prefers window seats")
    assert user_fact["metadata"]["source"] == "user"
    assert "sys:agent-fact" not in user_fact["tags"]

    # Source omitted by the model must default to user (no agent tag).
    defaulted = _fact_by_text(out["facts"], "lives in Seattle")
    assert defaulted["metadata"]["source"] == "user"
    assert "sys:agent-fact" not in defaulted["tags"]


@pytest.mark.asyncio
async def test_async_agent_sourced_fact_is_tagged_and_stamped() -> None:
    memories_store = _AsyncStore([])
    service = AsyncPipelineService(
        memories_store,
        _AsyncChat([_agent_source_response()]),
        _AsyncEmbeddings(),
        containers=_async_containers_for_store(memories_store, turns_store=_AsyncStore([_turn(1)])),
    )

    out = await service.extract_memories_dry("u1", "t1")

    agent_fact = _fact_by_text(out["facts"], "booked the user on flight")
    assert agent_fact["metadata"]["source"] == "agent"
    assert "sys:agent-fact" in agent_fact["tags"]

    defaulted = _fact_by_text(out["facts"], "lives in Seattle")
    assert defaulted["metadata"]["source"] == "user"
    assert "sys:agent-fact" not in defaulted["tags"]


def test_extraction_transcript_includes_turn_timestamps() -> None:
    # The extraction prompt must carry each turn's event time so the LLM can
    # resolve relative dates. _turn(i) stamps created_at=2025-01-01T00:0i:00.
    chat = _SyncChat([_response()])
    memories_store = _Store([])
    turns_store = _Store([_turn(1)])
    service = PipelineService(
        memories_store,
        chat,
        _SyncEmbeddings(),
        containers=_containers_for_store(memories_store, turns_store=turns_store),
    )

    service.extract_memories_dry("u1", "t1")

    prompt_text = json.dumps(chat.messages)
    assert "2025-01-01T00:01:00+00:00 | user" in prompt_text


def test_extraction_transcript_canonicalizes_speaker_role() -> None:
    # A turn written with the OpenAI-style role="assistant" must reach the
    # extraction prompt as the "agent" speaker, matching the prompt's rules.
    chat = _SyncChat([_response()])
    memories_store = _Store([])
    assistant_turn = dict(_turn(1))
    assistant_turn["role"] = "assistant"
    turns_store = _Store([assistant_turn])
    service = PipelineService(
        memories_store,
        chat,
        _SyncEmbeddings(),
        containers=_containers_for_store(memories_store, turns_store=turns_store),
    )

    service.extract_memories_dry("u1", "t1")

    prompt_text = json.dumps(chat.messages)
    assert "| agent]" in prompt_text
    assert "| assistant]" not in prompt_text
