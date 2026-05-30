from __future__ import annotations

from unittest.mock import MagicMock

from agent_memory_toolkit._container_routing import ContainerKey
from agent_memory_toolkit.aio.store import AsyncMemoryStore
from agent_memory_toolkit.store import MemoryStore


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


def _containers(*, turns=None, memories=None, summaries=None):
    return {
        ContainerKey.TURNS: turns if turns is not None else MagicMock(),
        ContainerKey.MEMORIES: memories if memories is not None else MagicMock(),
        ContainerKey.SUMMARIES: summaries if summaries is not None else MagicMock(),
    }


def test_list_tags_flattens_dedupes_sorts_and_hides_sys_tags():
    turns = MagicMock()
    memories = MagicMock()
    summaries = MagicMock()
    memories.query_items.return_value = [["topic:travel", "sys:fact"], ["topic:cooking"]]
    turns.query_items.return_value = [["topic:travel", "project:alpha"]]
    summaries.query_items.return_value = []
    store = MemoryStore(containers=_containers(turns=turns, memories=memories, summaries=summaries))

    assert store.list_tags("u1") == ["project:alpha", "topic:cooking", "topic:travel"]

    for target in (turns, memories, summaries):
        kwargs = target.query_items.call_args.kwargs
        assert kwargs["query"] == (
            "SELECT VALUE c.tags FROM c WHERE c.user_id = @user_id AND ARRAY_LENGTH(c.tags) > 0"
            " AND (NOT IS_DEFINED(c.superseded_by) OR IS_NULL(c.superseded_by))"
        )
        assert kwargs["enable_cross_partition_query"] is True


def test_list_tags_prefix_and_include_sys():
    turns = MagicMock()
    memories = MagicMock()
    summaries = MagicMock()
    turns.query_items.return_value = []
    memories.query_items.return_value = [["topic:travel", "topic:cooking", "sys:summary", "project:alpha"]]
    summaries.query_items.return_value = []
    store = MemoryStore(containers=_containers(turns=turns, memories=memories, summaries=summaries))

    assert store.list_tags("u1", prefix="topic:") == ["topic:cooking", "topic:travel"]
    assert store.list_tags("u1", prefix="sys:", include_sys=True) == ["sys:summary"]


def test_list_tags_thread_id_scopes_to_partition():
    turns = MagicMock()
    memories = MagicMock()
    summaries = MagicMock()
    turns.query_items.return_value = []
    memories.query_items.return_value = [["topic:thread"]]
    summaries.query_items.return_value = []
    store = MemoryStore(containers=_containers(turns=turns, memories=memories, summaries=summaries))

    assert store.list_tags("u1", thread_id="t1") == ["topic:thread"]

    for target in (turns, memories, summaries):
        kwargs = target.query_items.call_args.kwargs
        assert "AND c.thread_id = @thread_id" in kwargs["query"]
        assert kwargs["partition_key"] == ["u1", "t1"]
        assert "enable_cross_partition_query" not in kwargs


async def test_async_list_tags_flattens_dedupes_sorts_and_hides_sys_tags():
    turns = MagicMock()
    memories = MagicMock()
    summaries = MagicMock()
    memories.query_items.return_value = AsyncIterator([["topic:travel", "sys:fact"], ["topic:cooking"]])
    turns.query_items.return_value = AsyncIterator([["topic:travel", "project:alpha"]])
    summaries.query_items.return_value = AsyncIterator([])
    store = AsyncMemoryStore(containers=_containers(turns=turns, memories=memories, summaries=summaries))

    assert await store.list_tags("u1") == ["project:alpha", "topic:cooking", "topic:travel"]

    for target in (turns, memories, summaries):
        kwargs = target.query_items.call_args.kwargs
        assert kwargs["query"] == (
            "SELECT VALUE c.tags FROM c WHERE c.user_id = @user_id AND ARRAY_LENGTH(c.tags) > 0"
            " AND (NOT IS_DEFINED(c.superseded_by) OR IS_NULL(c.superseded_by))"
        )
        # Async SDK auto-detects cross-partition when partition_key is absent.
        # Forwarding `enable_cross_partition_query` would leak into aiohttp
        # because azure-cosmos.aio.ContainerProxy.query_items doesn't pop it.
        assert "enable_cross_partition_query" not in kwargs
        assert "partition_key" not in kwargs


async def test_async_list_tags_prefix_and_include_sys():
    turns = MagicMock()
    memories = MagicMock()
    summaries = MagicMock()
    turns.query_items.side_effect = lambda **_: AsyncIterator([])
    memories.query_items.side_effect = lambda **_: AsyncIterator([
        ["topic:travel", "topic:cooking", "sys:summary", "project:alpha"]
    ])
    summaries.query_items.side_effect = lambda **_: AsyncIterator([])
    store = AsyncMemoryStore(containers=_containers(turns=turns, memories=memories, summaries=summaries))

    assert await store.list_tags("u1", prefix="topic:") == ["topic:cooking", "topic:travel"]
    assert await store.list_tags("u1", prefix="sys:", include_sys=True) == ["sys:summary"]


async def test_async_list_tags_thread_id_scopes_to_partition():
    turns = MagicMock()
    memories = MagicMock()
    summaries = MagicMock()
    turns.query_items.return_value = AsyncIterator([])
    memories.query_items.return_value = AsyncIterator([["topic:thread"]])
    summaries.query_items.return_value = AsyncIterator([])
    store = AsyncMemoryStore(containers=_containers(turns=turns, memories=memories, summaries=summaries))

    assert await store.list_tags("u1", thread_id="t1") == ["topic:thread"]

    for target in (turns, memories, summaries):
        kwargs = target.query_items.call_args.kwargs
        assert "AND c.thread_id = @thread_id" in kwargs["query"]
        assert kwargs["partition_key"] == ["u1", "t1"]
        assert "enable_cross_partition_query" not in kwargs
