from __future__ import annotations

import logging
from typing import Any

from azure.cosmos.exceptions import CosmosResourceExistsError

from azure.cosmos.agent_memory.services.pipeline import PipelineService
from tests.unit.services.test_extract_dry import (
    _containers_for_store,
    _response,
    _Store,
    _SyncChat,
    _SyncEmbeddings,
    _turn,
)


class _TrackingStore(_Store):
    def __init__(self, docs: list[dict[str, Any]]):
        super().__init__(docs)
        self.supersede_calls: list[dict[str, Any]] = []

    def mark_superseded(self, old_doc: dict[str, Any], superseder_id: str, *, reason: str) -> bool:
        self.supersede_calls.append({"old_doc": old_doc, "superseder_id": superseder_id, "reason": reason})
        return True


def _episode(
    *,
    title: str = "Fixed CI retries",
    summary: str = "The user fixed flaky CI retries and the tests passed.",
    outcome: dict[str, str] | None = None,
) -> dict[str, Any]:
    return {
        "title": title,
        "summary": summary,
        "started_at": "2025-01-01T00:01:00+00:00",
        "ended_at": "2025-01-01T00:02:00+00:00",
        "participants": [],
        "events": [
            {
                "sequence": 1,
                "description": "CI retries were added.",
                "occurred_at": "2025-01-01T00:01:00+00:00",
                "source_turn_ids": ["turn-1", "missing"],
            },
            {
                "sequence": 2,
                "description": "The tests passed.",
                "occurred_at": "2025-01-01T00:02:00+00:00",
                "source_turn_ids": ["turn-2", "turn-1"],
            },
        ],
        "outcome": outcome,
        "lessons": ["Use retries for flaky CI."],
        "salience": 0.8,
        "confidence": 0.9,
    }


def _service(
    responses: list[dict[str, Any]],
    *,
    memories_store: _TrackingStore | None = None,
    embeddings: _SyncEmbeddings | None = None,
) -> tuple[PipelineService, _TrackingStore, _SyncEmbeddings]:
    store = memories_store or _TrackingStore([])
    embedding_client = embeddings or _SyncEmbeddings()
    turns_store = _Store([_turn(1), _turn(2)])
    service = PipelineService(
        store,
        _SyncChat(responses),
        embedding_client,
        containers=_containers_for_store(store, turns_store=turns_store),
    )
    return service, store, embedding_client


def test_build_episode_docs_returns_multiple_docs_without_embeddings() -> None:
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

    docs = service._build_episode_docs("u1", "t1", [_turn(1), _turn(2)], segment_key="seg-1")

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


def test_extract_episodes_embeds_content_persists_append_only(monkeypatch) -> None:
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

    result = service.extract_episodes("u1", "t1", flush=True)

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


def test_extract_episodes_empty_window_persists_nothing(monkeypatch) -> None:
    monkeypatch.setenv("EPISODE_TOPIC_DRIFT", "0")
    service, store, embeddings = _service([{"episodes": []}])

    result = service.extract_episodes("u1", "t1", flush=True)

    assert result == {"episodes": 0}
    assert [doc for doc in store.docs if doc.get("type") == "episodic"] == []
    assert embeddings.calls == []


def test_extract_episodes_skips_malformed_episode_with_warning(caplog, monkeypatch) -> None:
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
        result = service.extract_episodes("u1", "t1", flush=True)

    assert result == {"episodes": 1}
    assert [doc["title"] for doc in store.docs if doc.get("type") == "episodic"] == ["Valid episode"]
    assert "dropping malformed episode" in caplog.text


def test_extract_memories_durable_keeps_episodic_empty_regression_guard() -> None:
    store = _TrackingStore([])
    service = PipelineService(
        store,
        _SyncChat([_response()]),
        _SyncEmbeddings(),
        containers=_containers_for_store(store, turns_store=_Store([_turn(1)])),
    )

    output = service.extract_memories_durable("u1", "t1")

    assert output["facts"]
    assert output["episodic"] == []


class _IdUniqueStore(_TrackingStore):
    """Store that enforces id-uniqueness on create, like Cosmos (409 on dup)."""

    def create_item(self, *, body: dict[str, Any]) -> dict[str, Any]:
        if any(doc.get("id") == body.get("id") for doc in self.docs):
            raise CosmosResourceExistsError(message="conflict")
        self.docs.append(dict(body))
        return dict(body)


def test_build_episode_docs_id_stable_across_summary_text() -> None:
    service, _, _ = _service(
        [
            {"episodes": [_episode(summary="One phrasing of the CI-retry episode.")]},
            {"episodes": [_episode(summary="A completely different phrasing entirely.")]},
        ]
    )
    first = service._build_episode_docs("u1", "t1", [_turn(1), _turn(2)], segment_key="seg-1")
    second = service._build_episode_docs("u1", "t1", [_turn(1), _turn(2)], segment_key="seg-1")

    assert first[0]["content"] != second[0]["content"]
    assert first[0]["id"] == second[0]["id"]
    assert first[0]["content_hash"] != second[0]["content_hash"]


def test_build_episode_docs_multiple_episodes_get_distinct_ordinal_ids() -> None:
    service, _, _ = _service(
        [{"episodes": [_episode(), _episode(title="Vacation", summary="The user planned a vacation.")]}]
    )
    docs = service._build_episode_docs("u1", "t1", [_turn(1), _turn(2)], segment_key="seg-1")
    assert len({doc["id"] for doc in docs}) == 2


def test_extract_episodes_skips_duplicate_when_segment_reprocessed(monkeypatch) -> None:
    monkeypatch.setenv("EPISODE_TOPIC_DRIFT", "0")
    store = _IdUniqueStore([])
    turns = _Store([_turn(1), _turn(2)])
    service = PipelineService(
        store,
        _SyncChat(
            [
                {"episodes": [_episode(summary="First run prose.")]},
                {"episodes": [_episode(summary="Second run, different prose.")]},
            ]
        ),
        _SyncEmbeddings(),
        containers=_containers_for_store(store, turns_store=turns),
    )

    assert service.extract_episodes("u1", "t1", flush=True) == {"episodes": 1}

    # Simulate a crash before the cursor advanced: the episodic watermark never
    # moved, so the same open segment is re-loaded and re-segmented next run.
    store.docs = [doc for doc in store.docs if doc.get("type") != "episode_cursor"]

    assert service.extract_episodes("u1", "t1", flush=True) == {"episodes": 0}
    episodic = [doc for doc in store.docs if doc.get("type") == "episodic"]
    assert len(episodic) == 1


def test_build_episode_docs_falls_back_to_segment_times_on_unparseable_llm_times() -> None:
    bad = _episode()
    bad["started_at"] = "March 9th"
    bad["ended_at"] = "2025-01-01T00:02:00+00:00"
    service, _, _ = _service([{"episodes": [bad]}])

    docs = service._build_episode_docs("u1", "t1", [_turn(1), _turn(2)], segment_key="seg-1")

    assert len(docs) == 1
    assert docs[0]["started_at"] == "2025-01-01T00:01:00+00:00"
    assert docs[0]["ended_at"] == "2025-01-01T00:02:00+00:00"


def test_build_episode_docs_keeps_mixed_tz_llm_times_after_normalization() -> None:
    mixed = _episode()
    mixed["started_at"] = "2026-03-09"
    mixed["ended_at"] = "2026-03-10T09:08:00+00:00"
    service, _, _ = _service([{"episodes": [mixed]}])

    docs = service._build_episode_docs("u1", "t1", [_turn(1), _turn(2)], segment_key="seg-1")

    assert len(docs) == 1
    assert docs[0]["started_at"] == "2026-03-09"
    assert docs[0]["ended_at"] == "2026-03-10T09:08:00+00:00"


def test_build_episode_docs_keeps_episode_with_padded_timestamps() -> None:
    ep = _episode()
    ep["started_at"] = " 2025-01-01T00:01:00+00:00"
    ep["ended_at"] = "2025-01-01T00:02:00+00:00 "
    service, _, _ = _service([{"episodes": [ep]}])

    docs = service._build_episode_docs("u1", "t1", [_turn(1), _turn(2)], segment_key="seg-1")

    assert len(docs) == 1
    assert docs[0]["started_at"] == "2025-01-01T00:01:00+00:00"
    assert docs[0]["ended_at"] == "2025-01-01T00:02:00+00:00"


def test_build_episode_docs_clamps_out_of_range_salience_confidence() -> None:
    ep = _episode()
    ep["salience"] = 1.4
    ep["confidence"] = -0.2
    service, _, _ = _service([{"episodes": [ep]}])

    docs = service._build_episode_docs("u1", "t1", [_turn(1), _turn(2)], segment_key="seg-1")

    assert len(docs) == 1
    assert docs[0]["salience"] == 1.0
    assert docs[0]["confidence"] == 0.0


def test_build_episode_docs_drops_whitespace_only_summary() -> None:
    service, _, _ = _service([{"episodes": [_episode(summary="   ")]}])

    docs = service._build_episode_docs("u1", "t1", [_turn(1), _turn(2)], segment_key="seg-1")

    assert docs == []


def test_extract_episodes_reads_and_writes_only_inside_its_tenant(monkeypatch) -> None:
    monkeypatch.setenv("EPISODE_TOPIC_DRIFT", "0")
    ours = [{**_turn(1), "tenant_id": "acme"}, {**_turn(2), "tenant_id": "acme"}]
    theirs = {**_turn(3), "tenant_id": "other", "content": "Another tenant's private plan."}
    memories, turns, chat = _TrackingStore([]), _Store([*ours, theirs]), _SyncChat([{"episodes": [_episode()]}])
    service = PipelineService(
        memories,
        chat,
        _SyncEmbeddings(),
        containers=_containers_for_store(memories, turns_store=turns),
        tenant_id="acme",
    )

    assert service.extract_episodes("u1", "t1", flush=True) == {"episodes": 1}

    assert "private plan" not in str(chat.messages)
    assert turns.queries, "the open segment was read"
    for sql, params, partition_key in turns.queries:
        assert "c.tenant_id = @tenant_id AND c.user_id = @user_id AND c.thread_id = @thread_id" in sql
        assert (params["@tenant_id"], params["@user_id"], params["@thread_id"]) == ("acme", "u1", "t1")
        assert partition_key == ["acme", "u1"]
    assert memories.reads and all(partition_key == ["acme", "u1"] for _, partition_key in memories.reads)
    assert {doc["type"] for doc in memories.docs} == {"episodic", "episode_cursor"}
    assert {doc["tenant_id"] for doc in memories.docs} == {"acme"}, "the episode and the watermark carry the tenant"
    [cursor] = [doc for doc in memories.docs if doc["type"] == "episode_cursor"]
    assert (cursor["id"], cursor["user_id"], cursor["thread_id"]) == ("episode_cursor_u1_t1", "u1", "t1")


def test_the_same_turns_in_two_threads_make_episodes_and_watermarks_with_different_ids(monkeypatch) -> None:
    monkeypatch.setenv("EPISODE_TOPIC_DRIFT", "0")
    memories = _IdUniqueStore([])
    turns = _Store([_turn(1), _turn(2), {**_turn(1), "thread_id": "t2"}, {**_turn(2), "thread_id": "t2"}])
    service = PipelineService(
        memories,
        _SyncChat([{"episodes": [_episode()]}, {"episodes": [_episode()]}]),
        _SyncEmbeddings(),
        containers=_containers_for_store(memories, turns_store=turns),
    )

    assert service.extract_episodes("u1", "t1", flush=True) == {"episodes": 1}
    assert service.extract_episodes("u1", "t2", flush=True) == {"episodes": 1}

    episodes = [doc for doc in memories.docs if doc["type"] == "episodic"]
    cursors = [doc for doc in memories.docs if doc["type"] == "episode_cursor"]
    assert {doc["thread_id"] for doc in episodes} == {"t1", "t2"} and len({doc["id"] for doc in episodes}) == 2
    assert {doc["id"] for doc in cursors} == {"episode_cursor_u1_t1", "episode_cursor_u1_t2"}
