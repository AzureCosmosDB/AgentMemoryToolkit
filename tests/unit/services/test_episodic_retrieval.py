from __future__ import annotations

import logging
from typing import Any
from unittest.mock import MagicMock

from azure.cosmos.agent_memory._container_routing import ContainerKey
from azure.cosmos.agent_memory.cosmos_memory_client import CosmosMemoryClient
from azure.cosmos.agent_memory.store import MemoryStore

EPISODIC_OPT_IN_WARNING = "Episodic memories requested via memory_types are only returned when include_episodes=True"


def _containers(*, memories: Any = None, turns: Any = None, summaries: Any = None) -> dict[ContainerKey, Any]:
    return {
        ContainerKey.MEMORIES: memories if memories is not None else MagicMock(),
        ContainerKey.TURNS: turns if turns is not None else MagicMock(),
        ContainerKey.SUMMARIES: summaries if summaries is not None else MagicMock(),
    }


def _client_with_store(store: Any) -> CosmosMemoryClient:
    client = CosmosMemoryClient(
        use_default_credential=False,
        embeddings_client=MagicMock(),
        chat_client=MagicMock(),
    )
    client._get_store = MagicMock(return_value=store)  # type: ignore[method-assign]
    return client


def test_get_episodes_returns_user_episodes_newest_first() -> None:
    episodes = [
        {"id": "ep-new", "type": "episodic", "user_id": "u1", "thread_id": "t2", "created_at": "2026-01-02"},
        {"id": "ep-old", "type": "episodic", "user_id": "u1", "thread_id": "t1", "created_at": "2026-01-01"},
    ]
    memories = MagicMock()
    memories.query_items.return_value = episodes
    store = MemoryStore(containers=_containers(memories=memories))
    client = _client_with_store(store)

    result = client.get_episodes("u1", recent_k=2)

    assert result == episodes
    kwargs = memories.query_items.call_args.kwargs
    assert "c.type = @type" in kwargs["query"]
    assert "c.scope_key = @scope_key" in kwargs["query"]
    assert "ORDER BY c.created_at DESC" in kwargs["query"]
    assert kwargs["enable_cross_partition_query"] is True


def test_extract_episodes_routes_to_pipeline_with_flush() -> None:
    client = CosmosMemoryClient(
        use_default_credential=False,
        embeddings_client=MagicMock(),
        chat_client=MagicMock(),
    )
    pipeline = MagicMock()
    pipeline.extract_episodes.return_value = {"episodes": 2}
    client._get_pipeline = MagicMock(return_value=pipeline)  # type: ignore[method-assign]

    assert client.extract_episodes("u1", "t1", flush=True) == {"episodes": 2}
    pipeline.extract_episodes.assert_called_once_with("u1", "t1", flush=True)


def test_search_cosmos_base_is_facts_only_no_episodes_without_optin() -> None:
    # Episodes never enter via the base query: base search is scoped to facts,
    # and without include_episodes no episodic query runs at all.
    store = MagicMock()
    store.search.return_value = [{"id": "fact", "content": "fact", "type": "fact"}]
    store.search_summaries.return_value = [{"id": "summary", "content": "summary", "type": "thread_summary"}]
    store.search_turns.return_value = [{"id": "turn", "content": "turn", "type": "turn"}]
    client = _client_with_store(store)

    result = client.search_cosmos(
        "ci retries",
        user_id="u1",
        thread_id="t1",
        top_k=3,
        include_summaries=True,
        include_turns=True,
    )

    assert [doc["id"] for doc in result] == ["fact", "summary", "turn"]
    assert store.search.call_args.kwargs["memory_types"] == ["fact"]
    store.search_episodic.assert_not_called()


def test_search_cosmos_warns_when_episodic_requested_without_optin(caplog) -> None:
    store = MagicMock()
    store.search.return_value = []
    client = _client_with_store(store)
    caplog.set_level(logging.WARNING)

    result = client.search_cosmos("ci retries", user_id="u1", memory_types=["episodic"])

    assert result == []
    assert EPISODIC_OPT_IN_WARNING in caplog.text
    assert store.search.call_args.kwargs["memory_types"] == ["fact"]


def test_search_cosmos_does_not_warn_for_episodic_optin_or_other_types(caplog) -> None:
    store = MagicMock()
    store.search.return_value = []
    client = _client_with_store(store)
    caplog.set_level(logging.WARNING)

    client.search_cosmos("ci retries", user_id="u1", memory_types=["episodic"], include_episodes=True)
    assert EPISODIC_OPT_IN_WARNING not in caplog.text
    assert store.search.call_args.kwargs["memory_types"] == ["episodic"]

    caplog.clear()
    client.search_cosmos("ci retries", user_id="u1", memory_types=["fact"])
    assert EPISODIC_OPT_IN_WARNING not in caplog.text
    assert store.search.call_args.kwargs["memory_types"] == ["fact"]


def test_search_cosmos_include_episodes_combines_facts_and_episodes_in_base_query() -> None:
    store = MagicMock()
    # A single combined query returns facts + episodes ranked together.
    store.search.return_value = [
        {"id": "fact", "content": "fact", "type": "fact"},
        {"id": "episode", "content": "episode", "type": "episodic"},
    ]
    store.search_summaries.return_value = [{"id": "summary", "content": "summary", "type": "thread_summary"}]
    store.search_turns.return_value = [{"id": "turn", "content": "turn", "type": "turn"}]
    client = _client_with_store(store)

    result = client.search_cosmos(
        "ci retries",
        user_id="u1",
        thread_id="t1",
        top_k=100,
        include_episodes=True,
        include_summaries=True,
        include_turns=True,
    )

    # Combined base (facts + episodes) -> summaries -> turns.
    assert [doc["id"] for doc in result] == ["fact", "episode", "summary", "turn"]
    # One query, one shared top_k budget; episodic is folded into the base types.
    assert store.search.call_args.kwargs["top_k"] == 100
    assert store.search.call_args.kwargs["memory_types"] == ["fact", "episodic"]
    # No separate episodic query is issued.
    store.search_episodic.assert_not_called()


class _RankedEpisodeContainer:
    def __init__(self) -> None:
        self.query: str | None = None
        self.parameters: list[dict[str, Any]] | None = None
        self.docs = [
            {"id": "best-old", "created_at": "2026-01-01T00:00:00+00:00"},
            {"id": "best-new", "created_at": "2026-01-03T00:00:00+00:00"},
            {"id": "third-new", "created_at": "2026-01-04T00:00:00+00:00"},
        ]

    def query_items(self, **kwargs: Any) -> list[dict[str, Any]]:
        self.query = kwargs["query"]
        self.parameters = kwargs["parameters"]
        params = {p["name"]: p["value"] for p in self.parameters or []}
        after = params.get("@created_after")
        before = params.get("@created_before")
        docs = list(self.docs)
        if after is not None:
            docs = [doc for doc in docs if doc["created_at"] >= after]
        if before is not None:
            docs = [doc for doc in docs if doc["created_at"] <= before]
        return docs


def test_search_episodic_temporal_filter_narrows_without_time_ranking() -> None:
    memories = _RankedEpisodeContainer()
    embeddings = MagicMock()
    embeddings.generate.return_value = [0.1, 0.2]
    store = MemoryStore(containers=_containers(memories=memories), embeddings_client=embeddings)

    result = store.search_episodic(
        user_id="u1",
        search_terms="ci retries",
        created_after="2026-01-02T00:00:00+00:00",
    )

    assert [doc["id"] for doc in result] == ["best-new", "third-new"]
    assert memories.query is not None
    assert "c.created_at >= @created_after" in memories.query
    order_by = memories.query.split("ORDER BY", 1)[1]
    assert "created_at" not in order_by
