from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest
from azure.core import MatchConditions
from azure.cosmos.exceptions import CosmosAccessConditionFailedError

from agent_memory_toolkit._container_routing import ContainerKey
from agent_memory_toolkit.aio.store import AsyncMemoryStore
from agent_memory_toolkit.exceptions import MemoryConflictError
from agent_memory_toolkit.store import MemoryStore


def _doc(etag: str, tags: list[str]) -> dict:
    return {
        "id": "m1",
        "user_id": "u1",
        "thread_id": "t1",
        "role": "user",
        "type": "fact",
        "content": "hello",
        "created_at": "2026-01-01T00:00:00+00:00",
        "tags": tags,
        "_etag": etag,
    }


def _conflict():
    return CosmosAccessConditionFailedError(message="412", response=None)


def _containers(*, turns=None, memories=None, summaries=None):
    return {
        ContainerKey.TURNS: turns if turns is not None else MagicMock(),
        ContainerKey.MEMORIES: memories if memories is not None else MagicMock(),
        ContainerKey.SUMMARIES: summaries if summaries is not None else MagicMock(),
    }


def test_add_tags_retries_once_after_etag_conflict_and_wins():
    container = MagicMock()
    container.read_item.side_effect = [_doc("v1", ["old"]), _doc("v2", ["old", "other"])]
    container.replace_item.side_effect = [_conflict(), None]
    store = MemoryStore(containers=_containers(memories=container))

    store.add_tags("m1", "u1", "t1", "fact", ["New"])

    assert container.read_item.call_count == 2
    assert container.replace_item.call_count == 2
    final_kwargs = container.replace_item.call_args.kwargs
    assert final_kwargs["etag"] == "v2"
    assert final_kwargs["match_condition"] == MatchConditions.IfNotModified
    assert final_kwargs["body"]["tags"] == ["new", "old", "other"]


def test_add_tags_raises_memory_conflict_after_max_retries(monkeypatch):
    monkeypatch.setattr("time.sleep", lambda *_a, **_kw: None)
    container = MagicMock()
    container.read_item.side_effect = [_doc(f"v{i}", ["old"]) for i in range(5)]
    container.replace_item.side_effect = [_conflict() for _ in range(5)]
    store = MemoryStore(containers=_containers(memories=container))

    with pytest.raises(MemoryConflictError, match="after 5 attempts"):
        store.add_tags("m1", "u1", "t1", "fact", ["new"])

    assert container.read_item.call_count == 5
    assert container.replace_item.call_count == 5


async def test_async_add_tags_retries_once_after_etag_conflict_and_wins():
    container = MagicMock()
    container.read_item = AsyncMock(side_effect=[_doc("v1", ["old"]), _doc("v2", ["old", "other"])])
    container.replace_item = AsyncMock(side_effect=[_conflict(), None])
    store = AsyncMemoryStore(containers=_containers(memories=container))

    await store.add_tags("m1", "u1", "t1", "fact", ["New"])

    assert container.read_item.await_count == 2
    assert container.replace_item.await_count == 2
    final_kwargs = container.replace_item.call_args.kwargs
    assert final_kwargs["etag"] == "v2"
    assert final_kwargs["match_condition"] == MatchConditions.IfNotModified
    assert final_kwargs["body"]["tags"] == ["new", "old", "other"]


async def test_async_add_tags_raises_memory_conflict_after_max_retries(monkeypatch):
    async def _noop_sleep(*_a, **_kw):
        return None

    monkeypatch.setattr("asyncio.sleep", _noop_sleep)

    container = MagicMock()
    container.read_item = AsyncMock(side_effect=[_doc(f"v{i}", ["old"]) for i in range(5)])
    container.replace_item = AsyncMock(side_effect=[_conflict() for _ in range(5)])
    store = AsyncMemoryStore(containers=_containers(memories=container))

    with pytest.raises(MemoryConflictError, match="after 5 attempts"):
        await store.add_tags("m1", "u1", "t1", "fact", ["new"])

    assert container.read_item.await_count == 5
    assert container.replace_item.await_count == 5
