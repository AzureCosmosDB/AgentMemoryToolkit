from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import MagicMock

import pytest

from agent_memory_toolkit._container_routing import ContainerKey
from agent_memory_toolkit.exceptions import MemoryNotFoundError
from agent_memory_toolkit.store import MemoryStore


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


def test_add_upserts_memory_document():
    turns = MagicMock()
    store = MemoryStore(containers=_containers(turns=turns))

    memory_id = store.add(user_id="u1", role="user", content="hello", thread_id="t1")

    body = turns.upsert_item.call_args.kwargs["body"]
    assert memory_id == body["id"]
    assert body["user_id"] == "u1"
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
    store = MemoryStore(containers=_containers())

    body = store._prepare_doc(_doc(type=memory_type))

    assert body["ttl"] == expected_ttl


@pytest.mark.parametrize("ttl", [0, 60, -1])
def test_prepare_doc_preserves_caller_ttl(ttl):
    store = MemoryStore(containers=_containers())

    body = store._prepare_doc(_doc(type="episodic", ttl=ttl))

    assert body["ttl"] == ttl


@pytest.mark.parametrize("memory_type", ["fact", "thread_summary", "user_summary", "procedural", "unknown"])
def test_prepare_doc_omits_ttl_for_never_types(memory_type):
    store = MemoryStore(containers=_containers())

    body = store._prepare_doc(_doc(type=memory_type))

    assert "ttl" not in body


def test_push_batches_and_embeds_non_turn_records():
    memories = MagicMock()
    embeddings = MagicMock()
    embeddings.generate_batch.return_value = [[0.1, 0.2]]
    local = [_doc(id="f1", type="fact", content="fact", thread_id="facts")]
    store = MemoryStore(containers=_containers(memories=memories), embeddings_client=embeddings)

    store.push(local, batch_size=10)

    embeddings.generate_batch.assert_called_once_with(["fact"])
    body = memories.upsert_item.call_args.kwargs["body"]
    assert body["embedding"] == [0.1, 0.2]
    assert local[0]["embedding"] == [0.1, 0.2]


def test_query_wraps_query_items():
    memories = MagicMock()
    memories.query_items.return_value = [_doc(type="fact")]
    store = MemoryStore(containers=_containers(memories=memories))

    results = store.query(
        "SELECT * FROM c WHERE c.user_id = @user_id",
        [{"name": "@user_id", "value": "u1"}],
        container_key=ContainerKey.MEMORIES,
        cross_partition=True,
    )

    assert results == [_doc(type="fact")]
    assert memories.query_items.call_args.kwargs["enable_cross_partition_query"] is True


def test_update_replaces_matching_doc():
    turns = MagicMock()
    memories = MagicMock()
    turns.query_items.return_value = []
    memories.query_items.return_value = [_doc(type="fact")]
    store = MemoryStore(containers=_containers(turns=turns, memories=memories))

    store.update("m1", content="updated")

    body = memories.replace_item.call_args.kwargs["body"]
    assert body["content"] == "updated"
    assert "updated_at" in body


def test_update_raises_when_missing():
    turns = MagicMock()
    memories = MagicMock()
    summaries = MagicMock()
    for container in (turns, memories, summaries):
        container.query_items.return_value = []
    store = MemoryStore(containers=_containers(turns=turns, memories=memories, summaries=summaries))

    with pytest.raises(MemoryNotFoundError):
        store.update("missing")


def test_delete_checks_existence_then_deletes():
    turns = MagicMock()
    memories = MagicMock()
    turns.query_items.return_value = []
    memories.query_items.return_value = [{"id": "m1"}]
    store = MemoryStore(containers=_containers(turns=turns, memories=memories))

    store.delete("m1", thread_id="t1", user_id="u1")

    memories.delete_item.assert_called_once_with(item="m1", partition_key=["u1", "t1"])


def test_read_and_tag_mutation_use_point_reads():
    turns = MagicMock()
    turns.read_item.return_value = _doc(tags=["old"])
    store = MemoryStore(containers=_containers(turns=turns))

    assert store.read_item("m1", ["u1", "t1"], container_key=ContainerKey.TURNS)["id"] == "m1"
    store.add_tags("m1", "u1", "t1", ["New"])
    store.remove_tags("m1", "u1", "t1", ["old"])

    assert turns.read_item.call_args_list[0].kwargs == {"item": "m1", "partition_key": ["u1", "t1"]}
    assert turns.replace_item.call_count == 2


def test_single_doc_and_simple_query_helpers():
    turns = MagicMock()
    memories = MagicMock()
    summaries = MagicMock()
    turns.query_items.return_value = [_doc(content="turn")]
    memories.query_items.return_value = [_doc(type="procedural", content="prompt", version=1)]
    summaries.read_item.return_value = {"id": "user_summary_u1"}
    store = MemoryStore(containers=_containers(turns=turns, memories=memories, summaries=summaries))

    assert store.get_user_summary("u1") == {"id": "user_summary_u1"}
    assert store.get_thread("t1")
    assert store.get_procedural_prompt("u1") == "prompt"
    assert store.get_procedural_history("u1", limit=1)
    assert store.get_procedural_memories("u1")


def _params_by_name(call_kwargs):
    return {p["name"]: p["value"] for p in call_kwargs["parameters"]}


def test_get_memories_adds_created_time_range_filters():
    memories = MagicMock()
    memories.query_items.return_value = []
    store = MemoryStore(containers=_containers(memories=memories))
    after = datetime(2026, 1, 1, tzinfo=timezone.utc)

    store.get_memories(
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


def test_get_thread_adds_created_time_range_filters():
    turns = MagicMock()
    turns.query_items.return_value = []
    store = MemoryStore(containers=_containers(turns=turns))

    # Scope to memory_types=["turn"] so get_thread fans out to TURNS only.
    # Post-split, get_thread without memory_types queries all 3 containers.
    store.get_thread("t1", user_id="u1", memory_types=["turn"], created_after="2026-01-01T00:00:00+00:00")

    call_kwargs = turns.query_items.call_args.kwargs
    assert "c.created_at >= @created_after" in call_kwargs["query"]
    params = _params_by_name(call_kwargs)
    assert params["@created_after"] == "2026-01-01T00:00:00+00:00"


def test_search_adds_created_time_range_filters():
    memories = MagicMock()
    memories.query_items.return_value = []
    embeddings = MagicMock()
    embeddings.generate.return_value = [0.1, 0.2]
    store = MemoryStore(containers=_containers(memories=memories), embeddings_client=embeddings)

    store.search("weather", user_id="u1", created_before="2026-03-01T00:00:00+00:00")

    call_kwargs = memories.query_items.call_args.kwargs
    assert "c.created_at <= @created_before" in call_kwargs["query"]
    params = _params_by_name(call_kwargs)
    assert params["@created_before"] == "2026-03-01T00:00:00+00:00"


def test_add_cosmos_routes_by_type():
    turns = MagicMock()
    memories = MagicMock()
    summaries = MagicMock()
    store = MemoryStore(containers=_containers(turns=turns, memories=memories, summaries=summaries))

    for memory_type in ("turn", "fact", "episodic", "procedural", "thread_summary", "user_summary"):
        store.add_cosmos(_doc(id=f"{memory_type}_id", type=memory_type))

    assert turns.upsert_item.call_count == 1
    assert memories.upsert_item.call_count == 3
    assert summaries.upsert_item.call_count == 2
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


def test_get_memories_fans_out_across_keys():
    turns = MagicMock()
    memories = MagicMock()
    summaries = MagicMock()
    memories.query_items.return_value = [_doc(id="f1", type="fact")]
    summaries.query_items.return_value = [_doc(id="s1", type="thread_summary")]
    store = MemoryStore(containers=_containers(turns=turns, memories=memories, summaries=summaries))

    results = store.get_memories(user_id="u1", memory_types=["fact", "thread_summary"])

    assert [doc["id"] for doc in results] == ["f1", "s1"]
    memories.query_items.assert_called_once()
    summaries.query_items.assert_called_once()
    turns.query_items.assert_not_called()


def test_query_unknown_type_raises_value_error():
    store = MemoryStore(containers=_containers())

    with pytest.raises(ValueError, match="Unknown memory type"):
        store.get_memories(memory_types=["unknown"])
