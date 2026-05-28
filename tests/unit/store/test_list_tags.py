from __future__ import annotations

from unittest.mock import MagicMock

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


def test_list_tags_flattens_dedupes_sorts_and_hides_sys_tags():
    container = MagicMock()
    turns_container = MagicMock()
    container.query_items.return_value = [["topic:travel", "sys:fact"], ["topic:cooking"]]
    turns_container.query_items.return_value = [["topic:travel", "project:alpha"]]
    store = MemoryStore(container, turns_container=turns_container)

    assert store.list_tags("u1") == ["project:alpha", "topic:cooking", "topic:travel"]

    for target in (container, turns_container):
        kwargs = target.query_items.call_args.kwargs
        assert kwargs["query"] == (
            "SELECT VALUE c.tags FROM c WHERE c.user_id = @user_id AND ARRAY_LENGTH(c.tags) > 0"
            " AND (NOT IS_DEFINED(c.superseded_by) OR IS_NULL(c.superseded_by))"
        )
        assert kwargs["enable_cross_partition_query"] is True


def test_list_tags_prefix_and_include_sys():
    container = MagicMock()
    container.query_items.return_value = [["topic:travel", "topic:cooking", "sys:summary", "project:alpha"]]
    store = MemoryStore(container)

    assert store.list_tags("u1", prefix="topic:") == ["topic:cooking", "topic:travel"]
    assert store.list_tags("u1", prefix="sys:", include_sys=True) == ["sys:summary"]


def test_list_tags_thread_id_scopes_to_partition():
    container = MagicMock()
    container.query_items.return_value = [["topic:thread"]]
    store = MemoryStore(container)

    assert store.list_tags("u1", thread_id="t1") == ["topic:thread"]

    kwargs = container.query_items.call_args.kwargs
    assert "AND c.thread_id = @thread_id" in kwargs["query"]
    assert kwargs["partition_key"] == ["u1", "t1"]
    assert "enable_cross_partition_query" not in kwargs


async def test_async_list_tags_flattens_dedupes_sorts_and_hides_sys_tags():
    from agent_memory_toolkit.aio.store import AsyncMemoryStore

    container = MagicMock()
    turns_container = MagicMock()
    container.query_items.return_value = AsyncIterator([["topic:travel", "sys:fact"], ["topic:cooking"]])
    turns_container.query_items.return_value = AsyncIterator([["topic:travel", "project:alpha"]])
    store = AsyncMemoryStore(container, turns_container=turns_container)

    assert await store.list_tags("u1") == ["project:alpha", "topic:cooking", "topic:travel"]

    for target in (container, turns_container):
        kwargs = target.query_items.call_args.kwargs
        assert kwargs["query"] == (
            "SELECT VALUE c.tags FROM c WHERE c.user_id = @user_id AND ARRAY_LENGTH(c.tags) > 0"
            " AND (NOT IS_DEFINED(c.superseded_by) OR IS_NULL(c.superseded_by))"
        )
        assert kwargs["enable_cross_partition_query"] is True


async def test_async_list_tags_prefix_and_include_sys():
    from agent_memory_toolkit.aio.store import AsyncMemoryStore

    container = MagicMock()
    container.query_items.side_effect = [
        AsyncIterator([["topic:travel", "topic:cooking", "sys:summary", "project:alpha"]]),
        AsyncIterator([["topic:travel", "topic:cooking", "sys:summary", "project:alpha"]]),
    ]
    store = AsyncMemoryStore(container)

    assert await store.list_tags("u1", prefix="topic:") == ["topic:cooking", "topic:travel"]
    assert await store.list_tags("u1", prefix="sys:", include_sys=True) == ["sys:summary"]


async def test_async_list_tags_thread_id_scopes_to_partition():
    from agent_memory_toolkit.aio.store import AsyncMemoryStore

    container = MagicMock()
    container.query_items.return_value = AsyncIterator([["topic:thread"]])
    store = AsyncMemoryStore(container)

    assert await store.list_tags("u1", thread_id="t1") == ["topic:thread"]

    kwargs = container.query_items.call_args.kwargs
    assert "AND c.thread_id = @thread_id" in kwargs["query"]
    assert kwargs["partition_key"] == ["u1", "t1"]
    assert "enable_cross_partition_query" not in kwargs
