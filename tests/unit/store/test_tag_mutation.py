from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest
from azure.core import MatchConditions
from azure.cosmos.exceptions import CosmosAccessConditionFailedError

from agent_memory_toolkit.exceptions import MemoryConflictError
from agent_memory_toolkit.store import MemoryStore


def _doc(etag: str, tags: list[str]) -> dict:
    return {
        "id": "m1",
        "user_id": "u1",
        "thread_id": "t1",
        "role": "user",
        "type": "turn",
        "content": "hello",
        "created_at": "2026-01-01T00:00:00+00:00",
        "tags": tags,
        "_etag": etag,
    }


def _conflict():
    return CosmosAccessConditionFailedError(message="412", response=None)


def test_add_tags_retries_once_after_etag_conflict_and_wins():
    container = MagicMock()
    container.read_item.side_effect = [_doc("v1", ["old"]), _doc("v2", ["old", "other"])]
    container.replace_item.side_effect = [_conflict(), None]
    store = MemoryStore(container)

    store.add_tags("m1", "u1", "t1", ["New"])

    assert container.read_item.call_count == 2
    assert container.replace_item.call_count == 2
    final_kwargs = container.replace_item.call_args.kwargs
    assert final_kwargs["etag"] == "v2"
    assert final_kwargs["match_condition"] == MatchConditions.IfNotModified
    assert final_kwargs["body"]["tags"] == ["new", "old", "other"]


def test_add_tags_raises_memory_conflict_after_retry_conflicts():
    container = MagicMock()
    container.read_item.side_effect = [_doc("v1", ["old"]), _doc("v2", ["old"])]
    container.replace_item.side_effect = [_conflict(), _conflict()]
    store = MemoryStore(container)

    with pytest.raises(MemoryConflictError):
        store.add_tags("m1", "u1", "t1", ["new"])

    assert container.read_item.call_count == 2
    assert container.replace_item.call_count == 2


async def test_async_add_tags_retries_once_after_etag_conflict_and_wins():
    from agent_memory_toolkit.aio.store import AsyncMemoryStore

    container = MagicMock()
    container.read_item = AsyncMock(side_effect=[_doc("v1", ["old"]), _doc("v2", ["old", "other"])])
    container.replace_item = AsyncMock(side_effect=[_conflict(), None])
    store = AsyncMemoryStore(container)

    await store.add_tags("m1", "u1", "t1", ["New"])

    assert container.read_item.await_count == 2
    assert container.replace_item.await_count == 2
    final_kwargs = container.replace_item.call_args.kwargs
    assert final_kwargs["etag"] == "v2"
    assert final_kwargs["match_condition"] == MatchConditions.IfNotModified
    assert final_kwargs["body"]["tags"] == ["new", "old", "other"]


async def test_async_add_tags_raises_memory_conflict_after_retry_conflicts():
    from agent_memory_toolkit.aio.store import AsyncMemoryStore

    container = MagicMock()
    container.read_item = AsyncMock(side_effect=[_doc("v1", ["old"]), _doc("v2", ["old"])])
    container.replace_item = AsyncMock(side_effect=[_conflict(), _conflict()])
    store = AsyncMemoryStore(container)

    with pytest.raises(MemoryConflictError):
        await store.add_tags("m1", "u1", "t1", ["new"])

    assert container.read_item.await_count == 2
    assert container.replace_item.await_count == 2
