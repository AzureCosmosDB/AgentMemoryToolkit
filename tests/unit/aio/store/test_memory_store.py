from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock

import pytest

from azure.cosmos.agent_memory._container_routing import ContainerKey
from azure.cosmos.agent_memory.aio.store import AsyncMemoryStore
from azure.cosmos.agent_memory.exceptions import MemoryNotFoundError, MemoryTypeMismatchError


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
    )

    assert results == [_doc(type="fact")]
    call_kwargs = memories.query_items.call_args.kwargs
    assert "enable_cross_partition_query" not in call_kwargs
    assert "partition_key" not in call_kwargs


async def test_update_replaces_matching_doc():
    memories = MagicMock()
    memories.read_item = AsyncMock(return_value=_doc(id="m1", type="fact"))
    memories.replace_item = AsyncMock()
    store = AsyncMemoryStore(containers=_containers(memories=memories))

    await store.update("m1", user_id="u1", thread_id="t1", memory_type="fact", content="updated")

    memories.read_item.assert_awaited_once_with(item="m1", partition_key=["u1", "t1"])
    body = memories.replace_item.call_args.kwargs["body"]
    assert body["content"] == "updated"
    assert body["type"] == "fact"
    assert "updated_at" in body


async def test_update_raises_when_missing():
    from azure.cosmos.exceptions import CosmosResourceNotFoundError

    memories = MagicMock()
    memories.read_item = AsyncMock(side_effect=CosmosResourceNotFoundError(message="404"))
    store = AsyncMemoryStore(containers=_containers(memories=memories))

    with pytest.raises(MemoryNotFoundError):
        await store.update("missing", user_id="u1", thread_id="t1", memory_type="fact")


async def test_update_raises_on_type_mismatch():
    memories = MagicMock()
    memories.read_item = AsyncMock(return_value=_doc(id="m1", type="fact"))
    memories.replace_item = AsyncMock()
    store = AsyncMemoryStore(containers=_containers(memories=memories))

    with pytest.raises(MemoryTypeMismatchError):
        await store.update("m1", user_id="u1", thread_id="t1", memory_type="episodic", content="x")

    memories.replace_item.assert_not_awaited()


async def test_delete_calls_delete_item_directly():
    memories = MagicMock()
    memories.read_item = AsyncMock(return_value=_doc(id="m1", type="fact"))
    memories.delete_item = AsyncMock()
    store = AsyncMemoryStore(containers=_containers(memories=memories))

    await store.delete("m1", user_id="u1", thread_id="t1", memory_type="fact")

    memories.delete_item.assert_awaited_once_with(item="m1", partition_key=["u1", "t1"])


async def test_delete_raises_when_missing():
    from azure.cosmos.exceptions import CosmosResourceNotFoundError

    memories = MagicMock()
    memories.read_item = AsyncMock(side_effect=CosmosResourceNotFoundError(message="404"))
    memories.delete_item = AsyncMock()
    store = AsyncMemoryStore(containers=_containers(memories=memories))

    with pytest.raises(MemoryNotFoundError):
        await store.delete("m1", user_id="u1", thread_id="t1", memory_type="fact")

    memories.delete_item.assert_not_awaited()


async def test_delete_raises_on_type_mismatch():
    memories = MagicMock()
    memories.read_item = AsyncMock(return_value=_doc(id="m1", type="fact"))
    memories.delete_item = AsyncMock()
    store = AsyncMemoryStore(containers=_containers(memories=memories))

    with pytest.raises(MemoryTypeMismatchError):
        await store.delete("m1", user_id="u1", thread_id="t1", memory_type="episodic")

    memories.delete_item.assert_not_awaited()


async def test_read_and_tag_mutation_use_point_reads():
    memories = MagicMock()
    memories.read_item = AsyncMock(return_value=_doc(type="fact", tags=["old"]))
    memories.replace_item = AsyncMock()
    store = AsyncMemoryStore(containers=_containers(memories=memories))

    assert (await store.read_item("m1", ["u1", "t1"], container_key=ContainerKey.MEMORIES))["id"] == "m1"
    await store.add_tags("m1", "u1", "t1", "fact", ["New"])
    await store.remove_tags("m1", "u1", "t1", "fact", ["old"])

    assert memories.read_item.call_args_list[0].kwargs == {"item": "m1", "partition_key": ["u1", "t1"]}
    assert memories.replace_item.await_count == 2


async def test_tag_mutation_rejects_non_memories_types():
    store = AsyncMemoryStore(containers=_containers())

    for bad in ("turn", "thread_summary", "user_summary", "unknown"):
        with pytest.raises(ValueError, match="memory_type for tag mutation"):
            await store.add_tags("m1", "u1", "t1", bad, ["x"])


async def test_single_doc_and_simple_query_helpers():
    turns = MagicMock()
    memories = MagicMock()
    summaries = MagicMock()
    turns.query_items.side_effect = lambda **_: AsyncIterator([_doc(content="turn")])
    memories.query_items.side_effect = lambda **_: AsyncIterator([_doc(type="procedural", content="prompt", version=1)])
    summaries.read_item = AsyncMock(return_value={"id": "user_summary_u1"})
    summaries.query_items.side_effect = lambda **_: AsyncIterator([_doc(type="thread_summary", content="ts")])
    store = AsyncMemoryStore(containers=_containers(turns=turns, memories=memories, summaries=summaries))

    assert await store.get_user_summary("u1") == {"id": "user_summary_u1"}
    assert await store.get_thread("t1")
    assert await store.get_thread_summary("u1", "t1")
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

    await store.get_thread("t1", user_id="u1", created_after="2026-01-01T00:00:00+00:00")

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


async def test_get_memories_queries_memories_container_only():
    turns = MagicMock()
    memories = MagicMock()
    summaries = MagicMock()
    memories.query_items.return_value = AsyncIterator([_doc(id="f1", type="fact")])
    store = AsyncMemoryStore(containers=_containers(turns=turns, memories=memories, summaries=summaries))

    results = await store.get_memories(user_id="u1", memory_types=["fact"])

    assert [doc["id"] for doc in results] == ["f1"]
    memories.query_items.assert_called_once()
    turns.query_items.assert_not_called()
    summaries.query_items.assert_not_called()


async def test_get_memories_default_types_include_all_memories_types():
    memories = MagicMock()
    memories.query_items.return_value = AsyncIterator([])
    store = AsyncMemoryStore(containers=_containers(memories=memories))

    await store.get_memories(user_id="u1")

    call_kwargs = memories.query_items.call_args.kwargs
    params = _params_by_name(call_kwargs)
    types = {params[k] for k in params if k.startswith("@memory_type_")}
    assert types == {"fact", "episodic", "procedural"}


async def test_get_memories_rejects_non_memories_types():
    store = AsyncMemoryStore(containers=_containers())

    for bad in (["turn"], ["thread_summary"], ["user_summary"], ["unknown"], ["fact", "turn"]):
        with pytest.raises(ValueError, match="memory_types must be a subset"):
            await store.get_memories(memory_types=bad)


async def test_get_thread_summary_queries_summaries_with_partition_key():
    summaries = MagicMock()
    summaries.query_items.return_value = AsyncIterator([_doc(type="thread_summary", id="s1")])
    store = AsyncMemoryStore(containers=_containers(summaries=summaries))

    results = await store.get_thread_summary("u1", "t1", recent_k=1)

    assert [doc["id"] for doc in results] == ["s1"]
    call_kwargs = summaries.query_items.call_args.kwargs
    assert call_kwargs["partition_key"] == ["u1", "t1"]
    assert "c.type = @type" in call_kwargs["query"]
    assert "TOP @recent_k" in call_kwargs["query"]
    params = _params_by_name(call_kwargs)
    assert params["@type"] == "thread_summary"
    assert params["@user_id"] == "u1"
    assert params["@thread_id"] == "t1"


# ---------------------------------------------------------------------------
# F-final: read-path translation for user-scoped types (episodic / procedural)
# ---------------------------------------------------------------------------


async def test_get_memories_with_episodic_and_thread_id_emits_or_clause():
    memories = MagicMock()
    memories.query_items.return_value = AsyncIterator([])
    store = AsyncMemoryStore(containers=_containers(memories=memories))

    await store.get_memories(user_id="u1", thread_id="t1", memory_types=["episodic"])

    call_kwargs = memories.query_items.call_args.kwargs
    assert "(c.thread_id = @thread_id OR c.type IN (@user_scoped_type_0))" in call_kwargs["query"]
    params = _params_by_name(call_kwargs)
    assert params["@thread_id"] == "t1"
    assert params["@user_scoped_type_0"] == "episodic"


async def test_get_memories_with_procedural_and_thread_id_emits_or_clause():
    memories = MagicMock()
    memories.query_items.return_value = AsyncIterator([])
    store = AsyncMemoryStore(containers=_containers(memories=memories))

    await store.get_memories(user_id="u1", thread_id="t1", memory_types=["procedural"])

    call_kwargs = memories.query_items.call_args.kwargs
    assert "(c.thread_id = @thread_id OR c.type IN (@user_scoped_type_0))" in call_kwargs["query"]


async def test_get_memories_fact_only_with_thread_id_keeps_plain_filter():
    memories = MagicMock()
    memories.query_items.return_value = AsyncIterator([])
    store = AsyncMemoryStore(containers=_containers(memories=memories))

    await store.get_memories(user_id="u1", thread_id="t1", memory_types=["fact"])

    call_kwargs = memories.query_items.call_args.kwargs
    assert "c.thread_id = @thread_id" in call_kwargs["query"]
    assert "@user_scoped_type_" not in call_kwargs["query"]


async def test_get_memories_no_memory_types_with_thread_id_emits_or_clause():
    memories = MagicMock()
    memories.query_items.return_value = AsyncIterator([])
    store = AsyncMemoryStore(containers=_containers(memories=memories))

    await store.get_memories(user_id="u1", thread_id="t1")

    call_kwargs = memories.query_items.call_args.kwargs
    assert "(c.thread_id = @thread_id OR c.type IN (@user_scoped_type_0, @user_scoped_type_1))" in call_kwargs["query"]


async def test_search_with_episodic_and_thread_id_omits_partition_key():
    memories = MagicMock()
    memories.query_items.return_value = AsyncIterator([])
    embeddings = MagicMock()
    embeddings.generate = AsyncMock(return_value=[0.1, 0.2])
    store = AsyncMemoryStore(containers=_containers(memories=memories), embeddings_client=embeddings)

    await store.search(
        search_terms="hotels",
        user_id="u1",
        thread_id="t1",
        memory_types=["episodic"],
    )

    call_kwargs = memories.query_items.call_args.kwargs
    # When user-scoped types are in scope, async search must NOT confine
    # the query to [u1, t1] (where no episodic ever lives). Async SDK
    # treats absent partition_key as cross-partition by default.
    assert "partition_key" not in call_kwargs
    assert "(c.thread_id = @thread_id OR c.type IN (@user_scoped_type_0))" in call_kwargs["query"]


async def test_search_fact_only_with_thread_id_uses_partition_path():
    memories = MagicMock()
    memories.query_items.return_value = AsyncIterator([])
    embeddings = MagicMock()
    embeddings.generate = AsyncMock(return_value=[0.1, 0.2])
    store = AsyncMemoryStore(containers=_containers(memories=memories), embeddings_client=embeddings)

    await store.search(
        search_terms="hotels",
        user_id="u1",
        thread_id="t1",
        memory_types=["fact"],
    )

    call_kwargs = memories.query_items.call_args.kwargs
    assert call_kwargs.get("partition_key") == ["u1", "t1"]
