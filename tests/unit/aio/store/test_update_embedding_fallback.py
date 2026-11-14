"""A failed re-embed in async update() must not take the record out of vector search for good.

Only an update of a record that already has a vector rebuilds it, and nothing else backfills one, so these
tests run successive updates against a container that keeps what the earlier ones wrote.
"""

from __future__ import annotations

import copy
import itertools
import logging
from typing import Any
from unittest.mock import AsyncMock

from azure.core import MatchConditions
from azure.cosmos.exceptions import CosmosAccessConditionFailedError, CosmosResourceNotFoundError

from azure.cosmos.agent_memory._container_routing import ContainerKey
from azure.cosmos.agent_memory.aio.store import AsyncMemoryStore
from tests.unit.aio.store.test_secret_redaction import (
    SECRET,
    TOKEN,
    _container,
    _containers,
    _failing_embedder,
    _stored_doc,
)


class FakeMemories:
    """A memories container with the service's rules: hierarchical partition keys and ETag preconditions."""

    def __init__(self, *docs: dict[str, Any]) -> None:
        self._etags = itertools.count(1)
        self._docs: dict[tuple[str, tuple[str, ...]], dict[str, Any]] = {}
        for doc in docs:
            self._save(doc)

    def _save(self, body: dict[str, Any]) -> dict[str, Any]:
        saved = {**copy.deepcopy(body), "_etag": f"etag-{next(self._etags)}"}
        self._docs[saved["id"], (saved["user_id"], saved["thread_id"])] = saved
        return copy.deepcopy(saved)

    async def read_item(self, item: str, partition_key: list[str], **_kwargs: Any) -> dict[str, Any]:
        try:
            return copy.deepcopy(self._docs[item, tuple(partition_key)])
        except KeyError:
            raise CosmosResourceNotFoundError(
                status_code=404, message="Entity with the specified id does not exist"
            ) from None

    async def replace_item(
        self, item: str, body: dict[str, Any], *, match_condition: Any = None, etag: Any = None, **_kwargs: Any
    ) -> dict[str, Any]:
        current = await self.read_item(item, [body["user_id"], body["thread_id"]])
        if match_condition == MatchConditions.IfNotModified and etag != current["_etag"]:
            raise CosmosAccessConditionFailedError(status_code=412, message="Precondition failed")
        return self._save(body)


class FlakyEmbedder:
    """Embeds a text as ``[its length, 1.0]``; raises like a throttled endpoint while it is unavailable."""

    def __init__(self) -> None:
        self.available = True
        self.texts: list[str] = []

    async def generate(self, text: str) -> list[float]:
        self.texts.append(text)
        if not self.available:
            raise RuntimeError("429 Too Many Requests")
        return [float(len(text)), 1.0]


def _fact(content: str, embedding: list[float]) -> dict[str, Any]:
    return {
        "id": "fact_1",
        "user_id": "u1",
        "thread_id": "t1",
        "type": "fact",
        "role": "system",
        "content": content,
        "embedding": embedding,
        "metadata": {},
        "tags": [],
    }


def _store(memories: FakeMemories, embedder: FlakyEmbedder) -> AsyncMemoryStore:
    return AsyncMemoryStore(
        containers={
            ContainerKey.TURNS: FakeMemories(),
            ContainerKey.MEMORIES: memories,
            ContainerKey.SUMMARIES: FakeMemories(),
        },
        embeddings_client=embedder,
    )


async def test_a_failed_re_embed_keeps_the_record_in_vector_search_and_a_later_edit_refreshes_the_vector():
    memories = FakeMemories(_fact("I like tea", [10.0, 1.0]))
    embedder = FlakyEmbedder()
    store = _store(memories, embedder)

    embedder.available = False
    await store.update("fact_1", user_id="u1", thread_id="t1", memory_type="fact", content="I like green tea")

    stored = await memories.read_item("fact_1", ["u1", "t1"])
    assert stored["content"] == "I like green tea"
    assert stored["embedding"] == [10.0, 1.0], "the previous vector stays, so the record is still found"

    embedder.available = True
    await store.update("fact_1", user_id="u1", thread_id="t1", memory_type="fact", content="I like oolong tea")

    stored = await memories.read_item("fact_1", ["u1", "t1"])
    assert stored["embedding"] == [float(len("I like oolong tea")), 1.0]
    assert embedder.texts == ["I like green tea", "I like oolong tea"]


async def test_an_async_etag_conflict_retry_drops_a_re_read_vector_built_from_a_secret_when_embedding_fails(
    monkeypatch, caplog
):
    """Each attempt judges the vector it keeps by the text of the record it has just read, not of the first one."""
    caplog.set_level(logging.DEBUG)

    async def _noop_sleep(*_a, **_kw):
        return None

    monkeypatch.setattr("asyncio.sleep", _noop_sleep)
    memories = _container()
    memories.read_item = AsyncMock(
        side_effect=[
            _stored_doc(embedding=[9.9], _etag="v1"),
            # A writer without the guard (an older SDK) stored a credential, and its vector, meanwhile.
            _stored_doc(content=f"Use {SECRET}", embedding=[7.7], _etag="v2"),
        ]
    )
    memories.replace_item = AsyncMock(
        side_effect=[CosmosAccessConditionFailedError(message="412", response=None), None]
    )
    store = AsyncMemoryStore(containers=_containers(memories=memories), embeddings_client=_failing_embedder())

    await store.update("m1", user_id="u1", thread_id="t1", memory_type="fact", metadata={"note": "rotated"})

    body = memories.replace_item.call_args.kwargs["body"]
    assert memories.replace_item.await_count == 2
    assert "embedding" not in body and TOKEN not in repr(body)
    assert "dropping its previous vector" in caplog.text
    assert TOKEN not in caplog.text
