from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock

import pytest

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


async def test_add_upserts_memory_document():
    container = MagicMock()
    container.upsert_item = AsyncMock()
    store = AsyncMemoryStore(container)

    memory_id = await store.add(user_id="u1", role="user", content="hello", thread_id="t1")

    body = container.upsert_item.call_args.kwargs["body"]
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
    store = AsyncMemoryStore(MagicMock())

    body = store._prepare_doc(_doc(type=memory_type))

    assert body["ttl"] == expected_ttl


@pytest.mark.parametrize("ttl", [0, 60, -1])
def test_prepare_doc_preserves_caller_ttl(ttl):
    store = AsyncMemoryStore(MagicMock())

    body = store._prepare_doc(_doc(type="episodic", ttl=ttl))

    assert body["ttl"] == ttl


@pytest.mark.parametrize("memory_type", ["fact", "summary", "user_summary", "procedural", "unknown"])
def test_prepare_doc_omits_ttl_for_never_types(memory_type):
    store = AsyncMemoryStore(MagicMock())

    body = store._prepare_doc(_doc(type=memory_type))

    assert "ttl" not in body


async def test_push_batches_and_embeds_non_turn_records():
    container = MagicMock()
    container.upsert_item = AsyncMock()
    embeddings = MagicMock()
    embeddings.generate_batch = AsyncMock(return_value=[[0.1, 0.2]])
    local = [_doc(id="f1", type="fact", content="fact", thread_id="facts")]
    store = AsyncMemoryStore(container, embeddings_client=embeddings)

    await store.push(local, batch_size=10)

    embeddings.generate_batch.assert_awaited_once_with(["fact"])
    body = container.upsert_item.call_args.kwargs["body"]
    assert body["embedding"] == [0.1, 0.2]
    assert local[0]["embedding"] == [0.1, 0.2]


async def test_query_wraps_query_items():
    container = MagicMock()
    container.query_items.return_value = AsyncIterator([_doc()])
    store = AsyncMemoryStore(container)

    results = await store.query(
        "SELECT * FROM c WHERE c.user_id = @user_id",
        [{"name": "@user_id", "value": "u1"}],
        cross_partition=True,
    )

    assert results == [_doc()]
    assert container.query_items.call_args.kwargs["enable_cross_partition_query"] is True


async def test_update_replaces_matching_doc():
    container = MagicMock()
    container.query_items.return_value = AsyncIterator([_doc()])
    container.replace_item = AsyncMock()
    store = AsyncMemoryStore(container)

    await store.update("m1", content="updated")

    body = container.replace_item.call_args.kwargs["body"]
    assert body["content"] == "updated"
    assert "updated_at" in body


async def test_update_raises_when_missing():
    container = MagicMock()
    container.query_items.return_value = AsyncIterator([])
    store = AsyncMemoryStore(container)

    with pytest.raises(MemoryNotFoundError):
        await store.update("missing")


async def test_delete_checks_existence_then_deletes():
    container = MagicMock()
    container.query_items.return_value = AsyncIterator([{"id": "m1"}])
    container.delete_item = AsyncMock()
    store = AsyncMemoryStore(container)

    await store.delete("m1", thread_id="t1", user_id="u1")

    container.delete_item.assert_awaited_once_with(item="m1", partition_key=["u1", "t1"])


async def test_read_and_tag_mutation_use_point_reads():
    container = MagicMock()
    container.read_item = AsyncMock(return_value=_doc(tags=["old"]))
    container.replace_item = AsyncMock()
    store = AsyncMemoryStore(container)

    assert (await store.read_item("m1", ["u1", "t1"]))["id"] == "m1"
    await store.add_tags("m1", "u1", "t1", ["New"])
    await store.remove_tags("m1", "u1", "t1", ["old"])

    assert container.read_item.call_args_list[0].kwargs == {"item": "m1", "partition_key": ["u1", "t1"]}
    assert container.replace_item.await_count == 2


async def test_single_doc_and_simple_query_helpers():
    container = MagicMock()
    container.read_item = AsyncMock(return_value={"id": "user_summary_u1"})
    container.query_items.side_effect = lambda **_: AsyncIterator([_doc(content="prompt", version=1)])
    store = AsyncMemoryStore(container)

    assert await store.get_user_summary("u1") == {"id": "user_summary_u1"}
    assert await store.get_thread("t1")
    assert await store.get_procedural_prompt("u1") == "prompt"
    assert await store.get_procedural_history("u1", limit=1)
    assert await store.get_procedural_memories("u1")


def _params_by_name(call_kwargs):
    return {p["name"]: p["value"] for p in call_kwargs["parameters"]}


async def test_get_memories_adds_created_time_range_filters():
    container = MagicMock()
    container.query_items.return_value = AsyncIterator([])
    store = AsyncMemoryStore(container)
    after = datetime(2026, 1, 1, tzinfo=timezone.utc)

    await store.get_memories(user_id="u1", created_after=after, created_before="2026-02-01T00:00:00+00:00")

    call_kwargs = container.query_items.call_args.kwargs
    assert "c.created_at >= @created_after" in call_kwargs["query"]
    assert "c.created_at <= @created_before" in call_kwargs["query"]
    params = _params_by_name(call_kwargs)
    assert params["@created_after"] == after.isoformat()
    assert params["@created_before"] == "2026-02-01T00:00:00+00:00"


async def test_get_thread_adds_created_time_range_filters():
    container = MagicMock()
    container.query_items.return_value = AsyncIterator([])
    store = AsyncMemoryStore(container)

    await store.get_thread("t1", user_id="u1", created_after="2026-01-01T00:00:00+00:00")

    call_kwargs = container.query_items.call_args.kwargs
    assert "c.created_at >= @created_after" in call_kwargs["query"]
    params = _params_by_name(call_kwargs)
    assert params["@created_after"] == "2026-01-01T00:00:00+00:00"


async def test_search_adds_created_time_range_filters():
    container = MagicMock()
    container.query_items.return_value = AsyncIterator([])
    embeddings = MagicMock()
    embeddings.generate = AsyncMock(return_value=[0.1, 0.2])
    store = AsyncMemoryStore(container, embeddings_client=embeddings)

    await store.search("weather", user_id="u1", created_before="2026-03-01T00:00:00+00:00")

    call_kwargs = container.query_items.call_args.kwargs
    assert "c.created_at <= @created_before" in call_kwargs["query"]
    params = _params_by_name(call_kwargs)
    assert params["@created_before"] == "2026-03-01T00:00:00+00:00"
