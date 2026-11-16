"""``search_cosmos`` returns each fact with the facts it conflicts with (``conflicts_with``).

These run the real client and store over a fake memories container that answers the ranked search, the link query and
the partner query, so they cover the whole path a search takes.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from azure.cosmos.agent_memory.services._conflicts import CONFLICT_LINK_THREAD_ID
from tests.unit.aio import test_cosmos_memory_client as async_client_tests
from tests.unit.test_cosmos_memory_client import _connected_client

USER = "u1"


def fact(fact_id: str, content: str, created_at: str) -> dict[str, Any]:
    return {
        "id": fact_id,
        "user_id": USER,
        "thread_id": "t1",
        "type": "fact",
        "role": "system",
        "content": content,
        "created_at": created_at,
        "tags": ["sys:fact"],
        "metadata": {"category": "other"},
    }


RANKED = fact("f-new", "The user's deadline is March 15.", "2026-03-10T00:00:00Z")
OLDER = fact("f-old", "The user's deadline is March 1.", "2026-02-01T00:00:00Z")
LINKS = [{"first": "f-old", "later": "f-new", "kind": "update"}]


class FakeMemories:
    """Answers the three queries a search makes and records them."""

    def __init__(self, *, ranked: list[dict[str, Any]], links: list[dict[str, Any]], partners: list[dict[str, Any]]):
        self.ranked, self.links, self.partners = ranked, links, partners
        self.calls: list[dict[str, Any]] = []
        self.id = "memories"

    def _answer(self, kwargs: dict[str, Any]) -> list[dict[str, Any]]:
        self.calls.append(kwargs)
        query = kwargs["query"]
        if "c.type = @type" in query:
            return list(self.links)
        if "ARRAY_CONTAINS(@ids, c.id)" in query:
            return list(self.partners)
        return list(self.ranked)

    def query_items(self, **kwargs: Any) -> list[dict[str, Any]]:
        return self._answer(kwargs)


class AsyncFakeMemories(FakeMemories):
    """The async Cosmos client: ``query_items`` yields asynchronously and rejects the sync-only cross-partition flag."""

    def query_items(self, **kwargs: Any) -> Any:
        if "enable_cross_partition_query" in kwargs:
            raise TypeError("aiohttp rejects enable_cross_partition_query")
        rows = self._answer(kwargs)

        async def rows_iter() -> Any:
            for row in rows:
                yield row

        return rows_iter()


def _sync_client(container: FakeMemories) -> Any:
    mem, _ = _connected_client()
    mem._memories_container_client = container
    mem._embeddings_client = MagicMock()
    mem._embeddings_client.generate.return_value = [0.1, 0.2, 0.3]
    return mem


def _async_client(container: AsyncFakeMemories) -> Any:
    mem, _ = async_client_tests._connected_client()
    mem._memories_container_client = container
    mem._embeddings_client = AsyncMock()
    mem._embeddings_client.generate = AsyncMock(return_value=[0.1, 0.2, 0.3])
    return mem


def _partners() -> list[dict[str, Any]]:
    return [{"id": d["id"], "content": d["content"], "created_at": d["created_at"]} for d in (RANKED, OLDER)]


def _assert_shown_with_its_conflict(results: list[dict[str, Any]]) -> None:
    [shown] = results
    assert shown["id"] == "f-new"
    assert shown["conflicts_with"] == [
        {
            "id": "f-old",
            "content": "The user's deadline is March 1.",
            "created_at": "2026-02-01T00:00:00Z",
            "kind": "update",
        }
    ]


def test_a_fact_is_returned_with_the_fact_it_conflicts_with() -> None:
    container = FakeMemories(ranked=[dict(RANKED)], links=LINKS, partners=_partners())

    results = _sync_client(container).search_cosmos(search_terms="deadline", user_id=USER, top_k=3)

    _assert_shown_with_its_conflict(results)
    links_call = next(call for call in container.calls if "c.type = @type" in call["query"])
    assert links_call["partition_key"] == [USER, CONFLICT_LINK_THREAD_ID]


def test_a_fact_with_no_links_is_returned_as_it_is() -> None:
    container = FakeMemories(ranked=[dict(RANKED)], links=[], partners=_partners())

    [shown] = _sync_client(container).search_cosmos(search_terms="deadline", user_id=USER, top_k=3)

    assert "conflicts_with" not in shown
    assert not any("ARRAY_CONTAINS(@ids, c.id)" in call["query"] for call in container.calls)


def test_a_search_without_a_user_looks_up_no_conflicts() -> None:
    container = FakeMemories(ranked=[dict(RANKED)], links=LINKS, partners=_partners())

    [shown] = _sync_client(container).search_cosmos(search_terms="deadline", top_k=3)

    assert "conflicts_with" not in shown
    assert not any("c.type = @type" in call["query"] for call in container.calls)


def test_a_failing_conflict_lookup_does_not_break_the_search() -> None:
    container = FakeMemories(ranked=[dict(RANKED)], links=LINKS, partners=_partners())
    original = container._answer

    def answer(kwargs: dict[str, Any]) -> list[dict[str, Any]]:
        if "c.type = @type" in kwargs["query"]:
            raise RuntimeError("link query failed")
        return original(kwargs)

    container._answer = answer  # type: ignore[method-assign]

    [shown] = _sync_client(container).search_cosmos(search_terms="deadline", user_id=USER, top_k=3)

    assert shown["id"] == "f-new" and "conflicts_with" not in shown


@pytest.mark.asyncio
async def test_async_a_fact_is_returned_with_the_fact_it_conflicts_with() -> None:
    container = AsyncFakeMemories(ranked=[dict(RANKED)], links=LINKS, partners=_partners())

    results = await _async_client(container).search_cosmos(search_terms="deadline", user_id=USER, top_k=3)

    _assert_shown_with_its_conflict(results)
    links_call = next(call for call in container.calls if "c.type = @type" in call["query"])
    assert links_call["partition_key"] == [USER, CONFLICT_LINK_THREAD_ID]


@pytest.mark.asyncio
async def test_async_a_failing_conflict_lookup_does_not_break_the_search() -> None:
    container = AsyncFakeMemories(ranked=[dict(RANKED)], links=LINKS, partners=_partners())
    original = container._answer

    def answer(kwargs: dict[str, Any]) -> list[dict[str, Any]]:
        if "c.type = @type" in kwargs["query"]:
            raise RuntimeError("link query failed")
        return original(kwargs)

    container._answer = answer  # type: ignore[method-assign]

    [shown] = await _async_client(container).search_cosmos(search_terms="deadline", user_id=USER, top_k=3)

    assert shown["id"] == "f-new" and "conflicts_with" not in shown
