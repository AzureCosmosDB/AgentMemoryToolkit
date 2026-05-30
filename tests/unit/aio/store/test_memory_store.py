from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock

import pytest

from agent_memory_toolkit._container_routing import ContainerKey
from agent_memory_toolkit.aio.store import AsyncMemoryStore
from agent_memory_toolkit.exceptions import MemoryNotFoundError


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


def _doc(**overrides):
    doc = {
        "id": "m1",
        "user_id": "u1",
        "thread_id": "t1",
        "role": "user",
        "type": "turn",
        "content": "hello",
        "metadata": {},
        "created_at": "2025-01-01T00:00:00+00:00",
        "tags": [],
    }
    doc.update(overrides)
    return doc


def _containers(*, turns=None, memories=None, summaries=None):
    return {
        ContainerKey.TURNS: turns if turns is not None else MagicMock(),
        ContainerKey.MEMORIES: memories if memories is not None else MagicMock(),
        ContainerKey.SUMMARIES: summaries if summaries is not None else MagicMock(),
    }


async def test_add_upserts_memory_document():
    turns = MagicMock()
    turns.upsert_item = AsyncMock()
    store = AsyncMemoryStore(containers=_containers(turns=turns))

    memory_id = await store.add(user_id="u1", role="user", content="hello", thread_id="t1")

    body = turns.upsert_item.call_args.kwargs["body"]
    assert memory_id == body["id"]
    assert body["content"] == "hello"
    assert body["ttl"] == 2_592_000


@pytest.mark.parametrize(
    ("memory_type", "expected_ttl"),
    [
        ("turn", 2_592_000),
        ("episodic", 7_776_000),
    ],
)
def test_prepare_doc_applies_default_ttl(memory_type, expected_ttl):
    store = AsyncMemoryStore(containers=_containers())

    body = store._prepare_doc(_doc(type=memory_type))

    assert body["ttl"] == expected_ttl


@pytest.mark.parametrize("ttl", [0, 60, -1])
def test_prepare_doc_preserves_caller_ttl(ttl):
    store = AsyncMemoryStore(containers=_containers())

    body = store._prepare_doc(_doc(type="episodic", ttl=ttl))

    assert body["ttl"] == ttl


@pytest.mark.parametrize("memory_type", ["fact", "thread_summary", "user_summary", "procedural", "unknown"])
def test_prepare_doc_omits_ttl_for_never_types(memory_type):
    store = AsyncMemoryStore(containers=_containers())

    body = store._prepare_doc(_doc(type=memory_type))

    assert "ttl" not in body


async def test_push_batches_and_embeds_non_turn_records():
    memories = MagicMock()
    memories.upsert_item = AsyncMock()
    embeddings = MagicMock()
    embeddings.generate_batch = AsyncMock(return_value=[[0.1, 0.2]])
    local = [_doc(id="f1", type="fact", content="fact", thread_id="facts")]
    store = AsyncMemoryStore(containers=_containers(memories=memories), embeddings_client=embeddings)

    await store.push(local, batch_size=10)

    embeddings.generate_batch.assert_awaited_once_with(["fact"])
    body = memories.upsert_item.call_args.kwargs["body"]
    assert body["embedding"] == [0.1, 0.2]
    assert local[0]["embedding"] == [0.1, 0.2]


async def test_query_wraps_query_items():
    memories = MagicMock()
    memories.query_items.return_value = AsyncIterator([_doc(type="fact")])
    store = AsyncMemoryStore(containers=_containers(memories=memories))

    results = await store.query(
        "SELECT * FROM c WHERE c.user_id = @user_id",
        [{"name": "@user_id", "value": "u1"}],
        container_key=ContainerKey.MEMORIES,
        cross_partition=True,
    )

    assert results == [_doc(type="fact")]
    # Async SDK auto-detects cross-partition when partition_key is absent;
    # we must NOT forward `enable_cross_partition_query` because the async SDK
    # forgets to pop it and leaks it into aiohttp (TypeError).
    call_kwargs = memories.query_items.call_args.kwargs
    assert "enable_cross_partition_query" not in call_kwargs
    assert "partition_key" not in call_kwargs


async def test_update_replaces_matching_doc():
    turns = MagicMock()
    memories = MagicMock()
    turns.query_items.return_value = AsyncIterator([])
    memories.query_items.return_value = AsyncIterator([_doc(type="fact")])
    memories.replace_item = AsyncMock()
    store = AsyncMemoryStore(containers=_containers(turns=turns, memories=memories))

    await store.update("m1", content="updated")

    body = memories.replace_item.call_args.kwargs["body"]
    assert body["content"] == "updated"
    assert "updated_at" in body


async def test_update_raises_when_missing():
    turns = MagicMock()
    memories = MagicMock()
    summaries = MagicMock()
    for container in (turns, memories, summaries):
        container.query_items.return_value = AsyncIterator([])
    store = AsyncMemoryStore(containers=_containers(turns=turns, memories=memories, summaries=summaries))

    with pytest.raises(MemoryNotFoundError):
        await store.update("missing")


async def test_delete_checks_existence_then_deletes():
    turns = MagicMock()
    memories = MagicMock()
    turns.query_items.return_value = AsyncIterator([])
    memories.query_items.return_value = AsyncIterator([{"id": "m1"}])
    memories.delete_item = AsyncMock()
    store = AsyncMemoryStore(containers=_containers(turns=turns, memories=memories))

    await store.delete("m1", thread_id="t1", user_id="u1")

    memories.delete_item.assert_awaited_once_with(item="m1", partition_key=["u1", "t1"])


async def test_read_and_tag_mutation_use_point_reads():
    turns = MagicMock()
    turns.read_item = AsyncMock(return_value=_doc(tags=["old"]))
    turns.replace_item = AsyncMock()
    store = AsyncMemoryStore(containers=_containers(turns=turns))

    assert (await store.read_item("m1", ["u1", "t1"], container_key=ContainerKey.TURNS))["id"] == "m1"
    await store.add_tags("m1", "u1", "t1", ["New"])
    await store.remove_tags("m1", "u1", "t1", ["old"])

    assert turns.read_item.call_args_list[0].kwargs == {"item": "m1", "partition_key": ["u1", "t1"]}
    assert turns.replace_item.await_count == 2


async def test_single_doc_and_simple_query_helpers():
    turns = MagicMock()
    memories = MagicMock()
    summaries = MagicMock()
    turns.query_items.side_effect = lambda **_: AsyncIterator([_doc(content="turn")])
    memories.query_items.side_effect = lambda **_: AsyncIterator([_doc(type="procedural", content="prompt", version=1)])
    summaries.read_item = AsyncMock(return_value={"id": "user_summary_u1"})
    store = AsyncMemoryStore(containers=_containers(turns=turns, memories=memories, summaries=summaries))

    assert await store.get_user_summary("u1") == {"id": "user_summary_u1"}
    assert await store.get_thread("t1")
    assert await store.get_procedural_prompt("u1") == "prompt"
    assert await store.get_procedural_history("u1", limit=1)
    assert await store.get_procedural_memories("u1")


def _params_by_name(call_kwargs):
    return {p["name"]: p["value"] for p in call_kwargs["parameters"]}


async def test_get_memories_adds_created_time_range_filters():
    memories = MagicMock()
    memories.query_items.return_value = AsyncIterator([])
    store = AsyncMemoryStore(containers=_containers(memories=memories))
    after = datetime(2026, 1, 1, tzinfo=timezone.utc)

    await store.get_memories(
        user_id="u1",
        memory_types=["fact"],
        created_after=after,
        created_before="2026-02-01T00:00:00+00:00",
    )

    call_kwargs = memories.query_items.call_args.kwargs
    assert "c.created_at >= @created_after" in call_kwargs["query"]
    assert "c.created_at <= @created_before" in call_kwargs["query"]
    params = _params_by_name(call_kwargs)
    assert params["@created_after"] == after.isoformat()
    assert params["@created_before"] == "2026-02-01T00:00:00+00:00"


async def test_get_thread_adds_created_time_range_filters():
    turns = MagicMock()
    turns.query_items.return_value = AsyncIterator([])
    store = AsyncMemoryStore(containers=_containers(turns=turns))

    # Scope to memory_types=["turn"] so get_thread fans out to TURNS only.
    # Post-split, get_thread without memory_types queries all 3 containers.
    await store.get_thread(
        "t1", user_id="u1", memory_types=["turn"], created_after="2026-01-01T00:00:00+00:00"
    )

    call_kwargs = turns.query_items.call_args.kwargs
    assert "c.created_at >= @created_after" in call_kwargs["query"]
    params = _params_by_name(call_kwargs)
    assert params["@created_after"] == "2026-01-01T00:00:00+00:00"


async def test_search_adds_created_time_range_filters():
    memories = MagicMock()
    memories.query_items.return_value = AsyncIterator([])
    embeddings = MagicMock()
    embeddings.generate = AsyncMock(return_value=[0.1, 0.2])
    store = AsyncMemoryStore(containers=_containers(memories=memories), embeddings_client=embeddings)

    await store.search("weather", user_id="u1", created_before="2026-03-01T00:00:00+00:00")

    call_kwargs = memories.query_items.call_args.kwargs
    assert "c.created_at <= @created_before" in call_kwargs["query"]
    params = _params_by_name(call_kwargs)
    assert params["@created_before"] == "2026-03-01T00:00:00+00:00"


async def test_add_cosmos_routes_by_type():
    turns = MagicMock()
    memories = MagicMock()
    summaries = MagicMock()
    for container in (turns, memories, summaries):
        container.upsert_item = AsyncMock()
    store = AsyncMemoryStore(containers=_containers(turns=turns, memories=memories, summaries=summaries))

    for memory_type in ("turn", "fact", "episodic", "procedural", "thread_summary", "user_summary"):
        await store.add_cosmos(_doc(id=f"{memory_type}_id", type=memory_type))

    assert turns.upsert_item.await_count == 1
    assert memories.upsert_item.await_count == 3
    assert summaries.upsert_item.await_count == 2
    assert turns.upsert_item.call_args.kwargs["body"]["type"] == "turn"
    assert {call.kwargs["body"]["type"] for call in memories.upsert_item.call_args_list} == {
        "fact",
        "episodic",
        "procedural",
    }
    assert {call.kwargs["body"]["type"] for call in summaries.upsert_item.call_args_list} == {
        "thread_summary",
        "user_summary",
    }


async def test_get_memories_fans_out_across_keys():
    turns = MagicMock()
    memories = MagicMock()
    summaries = MagicMock()
    memories.query_items.return_value = AsyncIterator([_doc(id="f1", type="fact")])
    summaries.query_items.return_value = AsyncIterator([_doc(id="s1", type="thread_summary")])
    store = AsyncMemoryStore(containers=_containers(turns=turns, memories=memories, summaries=summaries))

    results = await store.get_memories(user_id="u1", memory_types=["fact", "thread_summary"])

    assert [doc["id"] for doc in results] == ["f1", "s1"]
    memories.query_items.assert_called_once()
    summaries.query_items.assert_called_once()
    turns.query_items.assert_not_called()


async def test_query_unknown_type_raises_value_error():
    store = AsyncMemoryStore(containers=_containers())

    with pytest.raises(ValueError, match="Unknown memory type"):
        await store.get_memories(memory_types=["unknown"])
