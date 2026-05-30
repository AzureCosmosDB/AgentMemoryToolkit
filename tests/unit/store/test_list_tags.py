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
    memories.query_items.return_value = [["topic:travel", "sys:fact"], ["topic:cooking", "project:alpha"]]
    store = MemoryStore(containers=_containers(turns=turns, memories=memories, summaries=summaries))

    assert store.list_tags("u1") == ["project:alpha", "topic:cooking", "topic:travel"]

    memories.query_items.assert_called_once()
    turns.query_items.assert_not_called()
    summaries.query_items.assert_not_called()
    kwargs = memories.query_items.call_args.kwargs
    assert kwargs["query"] == (
        "SELECT VALUE c.tags FROM c WHERE c.user_id = @user_id AND ARRAY_LENGTH(c.tags) > 0"
        " AND (NOT IS_DEFINED(c.superseded_by) OR IS_NULL(c.superseded_by))"
    )
    assert kwargs["enable_cross_partition_query"] is True


def test_list_tags_prefix_and_include_sys():
    turns = MagicMock()
    memories = MagicMock()
    summaries = MagicMock()
    memories.query_items.return_value = [["topic:travel", "topic:cooking", "sys:summary", "project:alpha"]]
    store = MemoryStore(containers=_containers(turns=turns, memories=memories, summaries=summaries))

    assert store.list_tags("u1", prefix="topic:") == ["topic:cooking", "topic:travel"]
    assert store.list_tags("u1", prefix="sys:", include_sys=True) == ["sys:summary"]
    turns.query_items.assert_not_called()
    summaries.query_items.assert_not_called()


def test_list_tags_thread_id_scopes_to_partition():
    turns = MagicMock()
    memories = MagicMock()
    summaries = MagicMock()
    memories.query_items.return_value = [["topic:thread"]]
    store = MemoryStore(containers=_containers(turns=turns, memories=memories, summaries=summaries))

    assert store.list_tags("u1", thread_id="t1") == ["topic:thread"]

    kwargs = memories.query_items.call_args.kwargs
    assert "AND c.thread_id = @thread_id" in kwargs["query"]
    assert kwargs["partition_key"] == ["u1", "t1"]
    assert "enable_cross_partition_query" not in kwargs
    turns.query_items.assert_not_called()
    summaries.query_items.assert_not_called()


async def test_async_list_tags_flattens_dedupes_sorts_and_hides_sys_tags():
    turns = MagicMock()
    memories = MagicMock()
    summaries = MagicMock()
    memories.query_items.return_value = AsyncIterator(
        [["topic:travel", "sys:fact"], ["topic:cooking", "project:alpha"]]
    )
    store = AsyncMemoryStore(containers=_containers(turns=turns, memories=memories, summaries=summaries))

    assert await store.list_tags("u1") == ["project:alpha", "topic:cooking", "topic:travel"]

    memories.query_items.assert_called_once()
    turns.query_items.assert_not_called()
    summaries.query_items.assert_not_called()
    kwargs = memories.query_items.call_args.kwargs
    assert kwargs["query"] == (
        "SELECT VALUE c.tags FROM c WHERE c.user_id = @user_id AND ARRAY_LENGTH(c.tags) > 0"
        " AND (NOT IS_DEFINED(c.superseded_by) OR IS_NULL(c.superseded_by))"
    )
    assert "enable_cross_partition_query" not in kwargs
    assert "partition_key" not in kwargs


async def test_async_list_tags_prefix_and_include_sys():
    turns = MagicMock()
    memories = MagicMock()
    summaries = MagicMock()
    memories.query_items.side_effect = lambda **_: AsyncIterator(
        [["topic:travel", "topic:cooking", "sys:summary", "project:alpha"]]
    )
    store = AsyncMemoryStore(containers=_containers(turns=turns, memories=memories, summaries=summaries))

    assert await store.list_tags("u1", prefix="topic:") == ["topic:cooking", "topic:travel"]
    assert await store.list_tags("u1", prefix="sys:", include_sys=True) == ["sys:summary"]
    turns.query_items.assert_not_called()
    summaries.query_items.assert_not_called()


async def test_async_list_tags_thread_id_scopes_to_partition():
    turns = MagicMock()
    memories = MagicMock()
    summaries = MagicMock()
    memories.query_items.return_value = AsyncIterator([["topic:thread"]])
    store = AsyncMemoryStore(containers=_containers(turns=turns, memories=memories, summaries=summaries))

    assert await store.list_tags("u1", thread_id="t1") == ["topic:thread"]

    kwargs = memories.query_items.call_args.kwargs
    assert "AND c.thread_id = @thread_id" in kwargs["query"]
    assert kwargs["partition_key"] == ["u1", "t1"]
    assert "enable_cross_partition_query" not in kwargs
    turns.query_items.assert_not_called()
    summaries.query_items.assert_not_called()
