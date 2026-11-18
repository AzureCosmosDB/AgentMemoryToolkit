from __future__ import annotations

import logging
from typing import Any

import pytest
from azure.cosmos.exceptions import CosmosResourceExistsError

from azure.cosmos.agent_memory.aio.services.pipeline import AsyncPipelineService
from tests.unit.services.test_extract_dry import (
    _async_containers_for_store,
    _AsyncChat,
    _AsyncEmbeddings,
    _AsyncStore,
    _response,
    _turn,
)
from tests.unit.services.test_extract_episodes import _episode


class _AsyncTrackingStore(_AsyncStore):
    def __init__(self, docs: list[dict[str, Any]]):
        super().__init__(docs)
        self.supersede_calls: list[dict[str, Any]] = []

    async def mark_superseded(self, old_doc: dict[str, Any], superseder_id: str, *, reason: str) -> bool:
        self.supersede_calls.append({"old_doc": old_doc, "superseder_id": superseder_id, "reason": reason})
        return True


def _service(
    responses: list[dict[str, Any]],
    *,
    memories_store: _AsyncTrackingStore | None = None,
    embeddings: _AsyncEmbeddings | None = None,
) -> tuple[AsyncPipelineService, _AsyncTrackingStore, _AsyncEmbeddings]:
    store = memories_store or _AsyncTrackingStore([])
    embedding_client = embeddings or _AsyncEmbeddings()
    turns_store = _AsyncStore([_turn(1), _turn(2)])
    service = AsyncPipelineService(
        store,
        _AsyncChat(responses),
        embedding_client,
        containers=_async_containers_for_store(store, turns_store=turns_store),
    )
    return service, store, embedding_client


@pytest.mark.asyncio
async def test_build_episode_docs_returns_multiple_docs_without_embeddings() -> None:
    outcome = {"status": "successful", "description": "The tests passed."}
    service, _, embeddings = _service(
        [
            {
                "episodes": [
                    _episode(outcome=outcome),
                    _episode(title="Planned vacation", summary="The user planned a vacation.", outcome=None),
                ]
            }
        ]
    )

    docs = await service._build_episode_docs("u1", "t1", [_turn(1), _turn(2)], segment_key="seg-1")

    assert [doc["content"] for doc in docs] == [
        "The user fixed flaky CI retries and the tests passed.",
        "The user planned a vacation.",
    ]
    assert all(doc["id"].startswith("ep_") for doc in docs)
    assert all("embedding" not in doc for doc in docs)
    assert docs[0]["source_turn_ids"] == ["turn-1", "turn-2"]
    assert docs[0]["events"][0]["source_turn_ids"] == ["turn-1"]
    assert docs[0]["outcome"] == outcome
    assert docs[1]["outcome"] is None
    assert embeddings.calls == []


@pytest.mark.asyncio
async def test_extract_episodes_embeds_content_persists_append_only(monkeypatch) -> None:
    monkeypatch.setenv("EPISODE_TOPIC_DRIFT", "0")
    service, store, embeddings = _service(
        [
            {
                "episodes": [
                    _episode(),
                    _episode(title="Planned vacation", summary="The user planned a vacation.", outcome=None),
                ]
            }
        ]
    )

    result = await service.extract_episodes("u1", "t1", flush=True)

    assert result == {"episodes": 2}
    assert embeddings.calls == [
        [
            "The user fixed flaky CI retries and the tests passed.",
            "The user planned a vacation.",
        ]
    ]
    episodes = [doc for doc in store.docs if doc.get("type") == "episodic"]
    assert [doc["content"] for doc in episodes] == [
        "The user fixed flaky CI retries and the tests passed.",
        "The user planned a vacation.",
    ]
    assert all(doc["id"].startswith("ep_") for doc in episodes)
    assert all(doc["embedding"] == [1.0] for doc in episodes)
    assert store.supersede_calls == []
    assert store.search_calls == []


@pytest.mark.asyncio
async def test_extract_episodes_empty_window_persists_nothing(monkeypatch) -> None:
    monkeypatch.setenv("EPISODE_TOPIC_DRIFT", "0")
    service, store, embeddings = _service([{"episodes": []}])

    result = await service.extract_episodes("u1", "t1", flush=True)

    assert result == {"episodes": 0}
    assert [doc for doc in store.docs if doc.get("type") == "episodic"] == []
    assert embeddings.calls == []


@pytest.mark.asyncio
async def test_extract_episodes_skips_malformed_episode_with_warning(caplog, monkeypatch) -> None:
    monkeypatch.setenv("EPISODE_TOPIC_DRIFT", "0")
    service, store, _ = _service(
        [
            {
                "episodes": [
                    {**_episode(), "title": ""},
                    _episode(title="Valid episode", summary="The valid episode persisted."),
                ]
            }
        ]
    )

    with caplog.at_level(logging.WARNING):
        result = await service.extract_episodes("u1", "t1", flush=True)

    assert result == {"episodes": 1}
    assert [doc["title"] for doc in store.docs if doc.get("type") == "episodic"] == ["Valid episode"]
    assert "dropping malformed episode" in caplog.text


@pytest.mark.asyncio
async def test_extract_memories_durable_keeps_episodic_empty_regression_guard() -> None:
    store = _AsyncTrackingStore([])
    service = AsyncPipelineService(
        store,
        _AsyncChat([_response()]),
        _AsyncEmbeddings(),
        containers=_async_containers_for_store(store, turns_store=_AsyncStore([_turn(1)])),
    )

    output = await service.extract_memories_durable("u1", "t1")

    assert output["facts"]
    assert output["episodic"] == []


class _AsyncIdUniqueStore(_AsyncTrackingStore):
    """Async store enforcing id-uniqueness on create, like Cosmos (409 on dup)."""

    async def create_item(self, *, body: dict[str, Any]) -> dict[str, Any]:
        if any(doc.get("id") == body.get("id") for doc in self.docs):
            raise CosmosResourceExistsError(message="conflict")
        self.docs.append(dict(body))
        return dict(body)


@pytest.mark.asyncio
async def test_build_episode_docs_id_stable_across_summary_text() -> None:
    service, _, _ = _service(
        [
            {"episodes": [_episode(summary="One phrasing of the CI-retry episode.")]},
            {"episodes": [_episode(summary="A completely different phrasing entirely.")]},
        ]
    )
    first = await service._build_episode_docs("u1", "t1", [_turn(1), _turn(2)], segment_key="seg-1")
    second = await service._build_episode_docs("u1", "t1", [_turn(1), _turn(2)], segment_key="seg-1")

    assert first[0]["content"] != second[0]["content"]
    assert first[0]["id"] == second[0]["id"]
    assert first[0]["content_hash"] != second[0]["content_hash"]


@pytest.mark.asyncio
async def test_extract_episodes_skips_duplicate_when_segment_reprocessed(monkeypatch) -> None:
    monkeypatch.setenv("EPISODE_TOPIC_DRIFT", "0")
    store = _AsyncIdUniqueStore([])
    turns = _AsyncStore([_turn(1), _turn(2)])
    service = AsyncPipelineService(
        store,
        _AsyncChat(
            [
                {"episodes": [_episode(summary="First run prose.")]},
                {"episodes": [_episode(summary="Second run, different prose.")]},
            ]
        ),
        _AsyncEmbeddings(),
        containers=_async_containers_for_store(store, turns_store=turns),
    )

    assert await service.extract_episodes("u1", "t1", flush=True) == {"episodes": 1}
    # Simulate a crash before the cursor advanced: the watermark never moved.
    store.docs = [doc for doc in store.docs if doc.get("type") != "episode_cursor"]
    assert await service.extract_episodes("u1", "t1", flush=True) == {"episodes": 0}
    episodic = [doc for doc in store.docs if doc.get("type") == "episodic"]
    assert len(episodic) == 1


@pytest.mark.asyncio
async def test_build_episode_docs_falls_back_to_segment_times_on_unparseable_llm_times() -> None:
    bad = _episode()
    bad["started_at"] = "March 9th"
    bad["ended_at"] = "2025-01-01T00:02:00+00:00"
    service, _, _ = _service([{"episodes": [bad]}])

    docs = await service._build_episode_docs("u1", "t1", [_turn(1), _turn(2)], segment_key="seg-1")

    assert len(docs) == 1
    assert docs[0]["started_at"] == "2025-01-01T00:01:00+00:00"
    assert docs[0]["ended_at"] == "2025-01-01T00:02:00+00:00"


@pytest.mark.asyncio
async def test_build_episode_docs_keeps_mixed_tz_llm_times_after_normalization() -> None:
    mixed = _episode()
    mixed["started_at"] = "2026-03-09"
    mixed["ended_at"] = "2026-03-10T09:08:00+00:00"
    service, _, _ = _service([{"episodes": [mixed]}])

    docs = await service._build_episode_docs("u1", "t1", [_turn(1), _turn(2)], segment_key="seg-1")

    assert len(docs) == 1
    assert docs[0]["started_at"] == "2026-03-09"
    assert docs[0]["ended_at"] == "2026-03-10T09:08:00+00:00"


@pytest.mark.asyncio
async def test_build_episode_docs_keeps_padded_timestamps_and_clamps_scores() -> None:
    ep = _episode()
    ep["started_at"] = " 2025-01-01T00:01:00+00:00"
    ep["ended_at"] = "2025-01-01T00:02:00+00:00 "
    ep["salience"] = 1.4
    ep["confidence"] = -0.2
    service, _, _ = _service([{"episodes": [ep]}])

    docs = await service._build_episode_docs("u1", "t1", [_turn(1), _turn(2)], segment_key="seg-1")

    assert len(docs) == 1
    assert docs[0]["started_at"] == "2025-01-01T00:01:00+00:00"
    assert docs[0]["ended_at"] == "2025-01-01T00:02:00+00:00"
    assert docs[0]["salience"] == 1.0
    assert docs[0]["confidence"] == 0.0


@pytest.mark.asyncio
async def test_the_same_turns_in_two_threads_make_episodes_and_watermarks_with_different_ids(monkeypatch) -> None:
    monkeypatch.setenv("EPISODE_TOPIC_DRIFT", "0")
    memories = _AsyncIdUniqueStore([])
    turns = _AsyncStore([_turn(1), _turn(2), {**_turn(1), "thread_id": "t2"}, {**_turn(2), "thread_id": "t2"}])
    service = AsyncPipelineService(
        memories,
        _AsyncChat([{"episodes": [_episode()]}, {"episodes": [_episode()]}]),
        _AsyncEmbeddings(),
        containers=_async_containers_for_store(memories, turns_store=turns),
    )

    assert await service.extract_episodes("u1", "t1", flush=True) == {"episodes": 1}
    assert await service.extract_episodes("u1", "t2", flush=True) == {"episodes": 1}

    episodes = [doc for doc in memories.docs if doc["type"] == "episodic"]
    cursors = [doc for doc in memories.docs if doc["type"] == "episode_cursor"]
    assert {doc["thread_id"] for doc in episodes} == {"t1", "t2"} and len({doc["id"] for doc in episodes}) == 2
    assert {doc["id"] for doc in cursors} == {"episode_cursor_u1_t1", "episode_cursor_u1_t2"}
