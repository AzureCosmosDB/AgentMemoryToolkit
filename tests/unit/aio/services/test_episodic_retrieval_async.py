from __future__ import annotations

import logging
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock

from azure.cosmos.agent_memory._container_routing import ContainerKey
from azure.cosmos.agent_memory.aio.cosmos_memory_client import AsyncCosmosMemoryClient
from azure.cosmos.agent_memory.aio.store import AsyncMemoryStore

EPISODIC_OPT_IN_WARNING = "Episodic memories requested via memory_types are only returned when include_episodes=True"


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


def _connected_client() -> tuple[AsyncCosmosMemoryClient, MagicMock]:
    client = AsyncCosmosMemoryClient(use_default_credential=False)
    memories = MagicMock()
    turns = MagicMock()
    summaries = MagicMock()
    for container in (memories, turns, summaries):
        container.query_items = MagicMock(return_value=AsyncIterator([]))
        container.upsert_item = AsyncMock()
    client._memories_container_client = memories
    client._turns_container_client = turns
    client._summaries_container_client = summaries
    return client, memories


def _containers(*, memories=None):
    return {
        ContainerKey.TURNS: MagicMock(),
        ContainerKey.MEMORIES: memories if memories is not None else MagicMock(),
        ContainerKey.SUMMARIES: MagicMock(),
    }


def _params_by_name(call_kwargs):
    return {p["name"]: p["value"] for p in call_kwargs["parameters"]}


async def test_async_get_episodes_user_wide_newest_first():
    mem, memories = _connected_client()
    docs = [
        {"id": "ep-new", "type": "episodic", "content": "new"},
        {"id": "ep-old", "type": "episodic", "content": "old"},
    ]
    memories.query_items = MagicMock(return_value=AsyncIterator(docs))

    results = await mem.get_episodes(user_id="u1", recent_k=2)

    assert [doc["id"] for doc in results] == ["ep-new", "ep-old"]
    call_kwargs = memories.query_items.call_args.kwargs
    assert "SELECT TOP @recent_k * FROM c" in call_kwargs["query"]
    assert "c.scope_key = @scope_key" in call_kwargs["query"]
    assert "c.type = @type" in call_kwargs["query"]
    assert "ORDER BY c.created_at DESC" in call_kwargs["query"]
    assert "partition_key" not in call_kwargs
    params = _params_by_name(call_kwargs)
    assert params["@scope_key"] == "user:u1"
    assert params["@type"] == "episodic"
    assert params["@recent_k"] == 2


async def test_async_search_cosmos_base_is_facts_only_no_episodes_without_optin():
    # Base search is facts-only; without include_episodes no episodic query runs.
    mem, _ = _connected_client()
    store = MagicMock()
    store.search = AsyncMock(return_value=[{"content": "fact A", "type": "fact"}])
    store.search_summaries = AsyncMock(return_value=[{"content": "summary B", "type": "thread_summary"}])
    store.search_episodic = AsyncMock(return_value=[])
    store.search_turns = AsyncMock(return_value=[{"content": "turn D", "type": "turn"}])
    mem._get_store = MagicMock(return_value=store)

    results = await mem.search_cosmos(
        "weather",
        user_id="u1",
        thread_id="t1",
        top_k=4,
        include_summaries=True,
        include_turns=True,
    )

    assert [doc["content"] for doc in results] == ["fact A", "summary B", "turn D"]
    assert store.search.call_args.kwargs["memory_types"] == ["fact"]
    store.search_episodic.assert_not_awaited()


async def test_async_search_cosmos_warns_when_episodic_requested_without_optin(caplog):
    mem, _ = _connected_client()
    store = MagicMock()
    store.search = AsyncMock(return_value=[])
    mem._get_store = MagicMock(return_value=store)
    caplog.set_level(logging.WARNING)

    results = await mem.search_cosmos("weather", user_id="u1", memory_types=["episodic"])

    assert results == []
    assert EPISODIC_OPT_IN_WARNING in caplog.text
    assert store.search.call_args.kwargs["memory_types"] == ["fact"]


async def test_async_search_cosmos_does_not_warn_for_episodic_optin_or_other_types(caplog):
    mem, _ = _connected_client()
    store = MagicMock()
    store.search = AsyncMock(return_value=[])
    mem._get_store = MagicMock(return_value=store)
    caplog.set_level(logging.WARNING)

    await mem.search_cosmos("weather", user_id="u1", memory_types=["episodic"], include_episodes=True)
    assert EPISODIC_OPT_IN_WARNING not in caplog.text
    assert store.search.call_args.kwargs["memory_types"] == ["episodic"]

    caplog.clear()
    await mem.search_cosmos("weather", user_id="u1", memory_types=["fact"])
    assert EPISODIC_OPT_IN_WARNING not in caplog.text
    assert store.search.call_args.kwargs["memory_types"] == ["fact"]


async def test_async_search_cosmos_include_episodes_combines_facts_and_episodes_in_base_query():
    mem, _ = _connected_client()
    store = MagicMock()
    store.search = AsyncMock(
        return_value=[
            {"content": "fact A", "type": "fact"},
            {"content": "episode C", "type": "episodic"},
        ]
    )
    store.search_episodic = AsyncMock()
    store.search_summaries = AsyncMock(return_value=[{"content": "summary B", "type": "thread_summary"}])
    store.search_turns = AsyncMock(return_value=[{"content": "turn D", "type": "turn"}])
    mem._get_store = MagicMock(return_value=store)

    results = await mem.search_cosmos(
        "weather",
        user_id="u1",
        thread_id="t1",
        top_k=100,
        include_episodes=True,
        include_summaries=True,
        include_turns=True,
    )
    # Combined base (facts + episodes) -> summaries -> turns; one shared budget.
    assert [doc["content"] for doc in results] == ["fact A", "episode C", "summary B", "turn D"]
    assert store.search.call_args.kwargs["top_k"] == 100
    assert store.search.call_args.kwargs["memory_types"] == ["fact", "episodic"]
    store.search_episodic.assert_not_awaited()


async def test_async_search_cosmos_include_episodes_combined_query_hits_real_store():
    # Drive the REAL AsyncMemoryStore.search (not a mock) through search_cosmos:
    # include_episodes folds "episodic" into a single combined base query that
    # applies the caller's tag/salience filters uniformly - no separate episodic
    # query, so facts and episodes share one top_k and the same filters.
    memories = MagicMock()
    memories.query_items = MagicMock(return_value=AsyncIterator([]))
    embeddings = MagicMock()
    embeddings.generate = AsyncMock(return_value=[0.1, 0.2])
    store = AsyncMemoryStore(containers=_containers(memories=memories), embeddings_client=embeddings)

    mem, _ = _connected_client()
    mem._get_store = MagicMock(return_value=store)

    results = await mem.search_cosmos(
        "weather",
        user_id="u1",
        thread_id="t1",
        top_k=5,
        include_episodes=True,
        tags_all=["trip"],
        min_salience=0.5,
    )

    assert results == []
    # Exactly one combined base query ran, scoped to fact + episodic, with the
    # caller's filters applied uniformly.
    assert memories.query_items.call_count == 1
    call_kwargs = memories.query_items.call_args.kwargs
    params = _params_by_name(call_kwargs)
    type_values = {v for k, v in params.items() if k.startswith("@memory_type")}
    assert type_values == {"fact", "episodic"}
    assert params["@min_salience"] == 0.5
    assert params["@tag_0"] == "trip"


async def test_async_search_episodic_temporal_filters_do_not_rank_by_time():
    memories = MagicMock()
    memories.query_items.return_value = AsyncIterator([])
    embeddings = MagicMock()
    embeddings.generate = AsyncMock(return_value=[0.1, 0.2])
    store = AsyncMemoryStore(containers=_containers(memories=memories), embeddings_client=embeddings)
    created_after = datetime(2026, 1, 1, tzinfo=timezone.utc)

    await store.search_episodic(
        user_id="u1",
        search_terms="checkout hotel",
        top_k=3,
        created_after=created_after,
        created_before="2026-02-01T00:00:00+00:00",
        started_after="2026-01-10T00:00:00+00:00",
        ended_before="2026-01-20T00:00:00+00:00",
    )

    call_kwargs = memories.query_items.call_args.kwargs
    query = call_kwargs["query"]
    assert "c.created_at >= @created_after" in query
    assert "c.created_at <= @created_before" in query
    assert "c.started_at >= @started_after" in query
    assert "c.ended_at <= @ended_before" in query
    assert "ORDER BY RANK RRF(VectorDistance(c.embedding, @embedding), FullTextScore(c.content, @kw0, @kw1))" in query
    assert "ORDER BY c.created_at" not in query
    assert "ORDER BY c.started_at" not in query
    params = _params_by_name(call_kwargs)
    assert params["@created_after"] == created_after.isoformat()
    assert params["@created_before"] == "2026-02-01T00:00:00+00:00"
    assert params["@started_after"] == "2026-01-10T00:00:00+00:00"
    assert params["@ended_before"] == "2026-01-20T00:00:00+00:00"
