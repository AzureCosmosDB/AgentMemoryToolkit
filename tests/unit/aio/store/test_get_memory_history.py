"""Async parity tests for get_memory_history."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from azure.cosmos.agent_memory._container_routing import ContainerKey
from azure.cosmos.agent_memory.aio.store import AsyncMemoryStore
from azure.cosmos.agent_memory.exceptions import ValidationError


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


def _containers(*, memories=None):
    return {
        ContainerKey.TURNS: MagicMock(),
        ContainerKey.MEMORIES: memories if memories is not None else MagicMock(),
        ContainerKey.SUMMARIES: MagicMock(),
    }


def _fact(fact_id, *, superseded_at=None, reason=None):
    return {
        "id": fact_id,
        "user_id": "u1",
        "thread_id": "t1",
        "type": "fact",
        "content": fact_id,
        "metadata": {"category": "biographical"},
        "created_at": "2025-01-01T00:00:00+00:00",
        "superseded_at": superseded_at,
        "supersede_reason": reason,
    }


async def test_async_multi_level_chain_newest_first():
    memories = MagicMock()
    memories.query_items.side_effect = [
        AsyncIterator([_fact("v2", superseded_at="2025-03-01T00:00:00+00:00", reason="contradict")]),
        AsyncIterator([_fact("v1", superseded_at="2025-02-01T00:00:00+00:00", reason="update")]),
        AsyncIterator([]),
    ]
    store = AsyncMemoryStore(containers=_containers(memories=memories))

    history = await store.get_memory_history("current", user_id="u1", thread_id="t1")

    assert [d["id"] for d in history] == ["v2", "v1"]


async def test_async_missing_ids_raise():
    store = AsyncMemoryStore(containers=_containers())
    with pytest.raises(ValidationError):
        await store.get_memory_history("", user_id="u1")
    with pytest.raises(ValidationError):
        await store.get_memory_history("current", user_id="")
