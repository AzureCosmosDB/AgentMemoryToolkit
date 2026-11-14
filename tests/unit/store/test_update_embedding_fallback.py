"""A failed re-embed in update() must not take the record out of vector search for good.

Only an update of a record that already has a vector rebuilds it, and nothing else backfills one, so these
tests run successive updates against a container that keeps what the earlier ones wrote.
"""

from __future__ import annotations

import copy
import itertools
from typing import Any

from azure.core import MatchConditions
from azure.cosmos.exceptions import CosmosAccessConditionFailedError, CosmosResourceNotFoundError

from azure.cosmos.agent_memory._container_routing import ContainerKey
from azure.cosmos.agent_memory.store import MemoryStore


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

    def read_item(self, item: str, partition_key: list[str], **_kwargs: Any) -> dict[str, Any]:
        try:
            return copy.deepcopy(self._docs[item, tuple(partition_key)])
        except KeyError:
            raise CosmosResourceNotFoundError(
                status_code=404, message="Entity with the specified id does not exist"
            ) from None

    def replace_item(
        self, item: str, body: dict[str, Any], *, match_condition: Any = None, etag: Any = None, **_kwargs: Any
    ) -> dict[str, Any]:
        current = self.read_item(item, [body["user_id"], body["thread_id"]])
        if match_condition == MatchConditions.IfNotModified and etag != current["_etag"]:
            raise CosmosAccessConditionFailedError(status_code=412, message="Precondition failed")
        return self._save(body)


class FlakyEmbedder:
    """Embeds a text as ``[its length, 1.0]``; raises like a throttled endpoint while it is unavailable."""

    def __init__(self) -> None:
        self.available = True
        self.texts: list[str] = []

    def generate(self, text: str) -> list[float]:
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


def _store(memories: FakeMemories, embedder: FlakyEmbedder) -> MemoryStore:
    return MemoryStore(
        containers={
            ContainerKey.TURNS: FakeMemories(),
            ContainerKey.MEMORIES: memories,
            ContainerKey.SUMMARIES: FakeMemories(),
        },
        embeddings_client=embedder,
    )


def test_a_failed_re_embed_keeps_the_record_in_vector_search_and_a_later_edit_refreshes_the_vector():
    memories = FakeMemories(_fact("I like tea", [10.0, 1.0]))
    embedder = FlakyEmbedder()
    store = _store(memories, embedder)

    embedder.available = False
    store.update("fact_1", user_id="u1", thread_id="t1", memory_type="fact", content="I like green tea")

    stored = memories.read_item("fact_1", ["u1", "t1"])
    assert stored["content"] == "I like green tea"
    assert stored["embedding"] == [10.0, 1.0], "the previous vector stays, so the record is still found"

    embedder.available = True
    store.update("fact_1", user_id="u1", thread_id="t1", memory_type="fact", content="I like oolong tea")

    stored = memories.read_item("fact_1", ["u1", "t1"])
    assert stored["embedding"] == [float(len("I like oolong tea")), 1.0]
    assert embedder.texts == ["I like green tea", "I like oolong tea"]
