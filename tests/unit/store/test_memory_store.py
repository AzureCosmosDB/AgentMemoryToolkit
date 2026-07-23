from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import MagicMock

import pytest

from azure.cosmos.agent_memory._container_routing import ContainerKey
from azure.cosmos.agent_memory.exceptions import MemoryNotFoundError, MemoryTypeMismatchError, ValidationError
from azure.cosmos.agent_memory.store import MemoryStore


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


def test_add_defaults_created_at_to_ingestion_time():
    turns = MagicMock()
    store = MemoryStore(containers=_containers(turns=turns))

    store.add(user_id="u1", role="user", content="hello", thread_id="t1")

    body = turns.upsert_item.call_args.kwargs["body"]
    # No created_at passed -> defaulted (present, non-empty ISO string).
    assert body["created_at"]


def test_add_honors_explicit_created_at_string():
    turns = MagicMock()
    store = MemoryStore(containers=_containers(turns=turns))

    store.add(
        user_id="u1",
        role="user",
        content="hello",
        thread_id="t1",
        created_at="2024-03-01T12:00:00+00:00",
    )

    body = turns.upsert_item.call_args.kwargs["body"]
    assert body["created_at"] == "2024-03-01T12:00:00+00:00"


def test_add_normalizes_datetime_created_at():
    turns = MagicMock()
    store = MemoryStore(containers=_containers(turns=turns))

    dt = datetime(2024, 3, 1, 12, 0, tzinfo=timezone.utc)
    store.add(user_id="u1", role="user", content="hello", thread_id="t1", created_at=dt)

    body = turns.upsert_item.call_args.kwargs["body"]
    assert body["created_at"] == dt.isoformat()


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
    memories = MagicMock()
    memories.read_item.return_value = _doc(id="m1", type="fact")
    store = MemoryStore(containers=_containers(memories=memories))

    store.update("m1", user_id="u1", thread_id="t1", memory_type="fact", content="updated")

    memories.read_item.assert_called_once_with(item="m1", partition_key=["u1", "t1"])
    body = memories.replace_item.call_args.kwargs["body"]
    assert body["content"] == "updated"
    assert body["type"] == "fact"
    assert "updated_at" in body


def test_update_raises_when_missing():
    from azure.cosmos.exceptions import CosmosResourceNotFoundError

    memories = MagicMock()
    memories.read_item.side_effect = CosmosResourceNotFoundError(message="404")
    store = MemoryStore(containers=_containers(memories=memories))

    with pytest.raises(MemoryNotFoundError):
        store.update("missing", user_id="u1", thread_id="t1", memory_type="fact")


def test_update_raises_on_type_mismatch():
    memories = MagicMock()
    memories.read_item.return_value = _doc(id="m1", type="fact")
    store = MemoryStore(containers=_containers(memories=memories))

    with pytest.raises(MemoryTypeMismatchError):
        store.update("m1", user_id="u1", thread_id="t1", memory_type="episodic", content="x")

    memories.replace_item.assert_not_called()


def test_delete_calls_delete_item_directly():
    memories = MagicMock()
    memories.read_item.return_value = _doc(id="m1", type="fact")
    store = MemoryStore(containers=_containers(memories=memories))

    store.delete("m1", user_id="u1", thread_id="t1", memory_type="fact")

    memories.delete_item.assert_called_once_with(item="m1", partition_key=["u1", "t1"])


def test_delete_raises_when_missing():
    from azure.cosmos.exceptions import CosmosResourceNotFoundError

    memories = MagicMock()
    memories.read_item.side_effect = CosmosResourceNotFoundError(message="404")
    store = MemoryStore(containers=_containers(memories=memories))

    with pytest.raises(MemoryNotFoundError):
        store.delete("m1", user_id="u1", thread_id="t1", memory_type="fact")

    memories.delete_item.assert_not_called()


def test_delete_raises_on_type_mismatch():
    memories = MagicMock()
    memories.read_item.return_value = _doc(id="m1", type="fact")
    store = MemoryStore(containers=_containers(memories=memories))

    with pytest.raises(MemoryTypeMismatchError):
        store.delete("m1", user_id="u1", thread_id="t1", memory_type="episodic")

    memories.delete_item.assert_not_called()


def test_read_and_tag_mutation_use_point_reads():
    memories = MagicMock()
    memories.read_item.return_value = _doc(type="fact", tags=["old"])
    store = MemoryStore(containers=_containers(memories=memories))

    assert store.read_item("m1", ["u1", "t1"], container_key=ContainerKey.MEMORIES)["id"] == "m1"
    store.add_tags("m1", "u1", "t1", "fact", ["New"])
    store.remove_tags("m1", "u1", "t1", "fact", ["old"])

    assert memories.read_item.call_args_list[0].kwargs == {"item": "m1", "partition_key": ["u1", "t1"]}
    assert memories.replace_item.call_count == 2


def test_tag_mutation_rejects_non_memories_types():
    store = MemoryStore(containers=_containers())

    for bad in ("turn", "thread_summary", "user_summary", "unknown"):
        with pytest.raises(ValueError, match="memory_type for tag mutation"):
            store.add_tags("m1", "u1", "t1", bad, ["x"])


def test_single_doc_and_simple_query_helpers():
    turns = MagicMock()
    memories = MagicMock()
    summaries = MagicMock()
    turns.query_items.return_value = [_doc(content="turn")]
    memories.query_items.return_value = [_doc(type="procedural", content="prompt", version=1)]
    summaries.read_item.return_value = {"id": "user_summary_u1"}
    summaries.query_items.return_value = [_doc(type="thread_summary", content="ts")]
    store = MemoryStore(containers=_containers(turns=turns, memories=memories, summaries=summaries))

    assert store.get_user_summary("u1") == {"id": "user_summary_u1"}
    assert store.get_thread("t1")
    assert store.get_thread_summary("u1", "t1")
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

    store.get_thread("t1", user_id="u1", created_after="2026-01-01T00:00:00+00:00")

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


def test_search_uses_keyword_params_for_hybrid_sql():
    memories = MagicMock()
    memories.query_items.return_value = []
    embeddings = MagicMock()
    embeddings.generate.return_value = [0.1, 0.2]
    store = MemoryStore(containers=_containers(memories=memories), embeddings_client=embeddings)

    store.search("weather in Seattle", user_id="u1")

    call_kwargs = memories.query_items.call_args.kwargs
    assert "ORDER BY RANK RRF" in call_kwargs["query"]
    assert "FullTextScore(c.content, @kw0, @kw1)" in call_kwargs["query"]
    params = _params_by_name(call_kwargs)
    assert params["@kw0"] == "weather"
    assert params["@kw1"] == "seattle"


def test_search_all_stopwords_falls_back_to_vector_only():
    memories = MagicMock()
    memories.query_items.return_value = []
    embeddings = MagicMock()
    embeddings.generate.return_value = [0.1, 0.2]
    store = MemoryStore(containers=_containers(memories=memories), embeddings_client=embeddings)

    store.search("what is the", user_id="u1")

    call_kwargs = memories.query_items.call_args.kwargs
    assert "ORDER BY VectorDistance" in call_kwargs["query"]
    assert "RANK RRF" not in call_kwargs["query"]
    assert not any(name.startswith("@kw") for name in _params_by_name(call_kwargs))


def test_search_episodic_forwards_search_options():
    store = MemoryStore(containers=_containers())
    store.search = MagicMock(return_value=[])

    store.search_episodic("u1", "weather")

    store.search.assert_called_once_with(
        search_terms="weather",
        user_id="u1",
        memory_types=["episodic"],
        top_k=5,
        min_salience=None,
        include_superseded=False,
    )


def test_build_episodic_context_forwards_search_options():
    store = MemoryStore(containers=_containers())
    store.search_episodic = MagicMock(return_value=[])

    assert store.build_episodic_context("u1", "weather") == ""

    store.search_episodic.assert_called_once_with("u1", "weather", top_k=3)


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


def test_get_memories_queries_memories_container_only():
    turns = MagicMock()
    memories = MagicMock()
    summaries = MagicMock()
    memories.query_items.return_value = [_doc(id="f1", type="fact")]
    store = MemoryStore(containers=_containers(turns=turns, memories=memories, summaries=summaries))

    results = store.get_memories(user_id="u1", memory_types=["fact"])

    assert [doc["id"] for doc in results] == ["f1"]
    memories.query_items.assert_called_once()
    turns.query_items.assert_not_called()
    summaries.query_items.assert_not_called()


def test_get_memories_default_types_include_all_memories_types():
    memories = MagicMock()
    memories.query_items.return_value = []
    store = MemoryStore(containers=_containers(memories=memories))

    store.get_memories(user_id="u1")

    call_kwargs = memories.query_items.call_args.kwargs
    params = _params_by_name(call_kwargs)
    types = {params[k] for k in params if k.startswith("@memory_type_")}
    assert types == {"fact", "episodic", "procedural"}


def test_get_memories_rejects_non_memories_types():
    store = MemoryStore(containers=_containers())

    for bad in (["turn"], ["thread_summary"], ["user_summary"], ["unknown"], ["fact", "turn"]):
        with pytest.raises(ValueError, match="memory_types must be a subset"):
            store.get_memories(memory_types=bad)


def test_get_thread_summary_queries_summaries_with_partition_key():
    summaries = MagicMock()
    summaries.query_items.return_value = [_doc(type="thread_summary", id="s1")]
    store = MemoryStore(containers=_containers(summaries=summaries))

    results = store.get_thread_summary("u1", "t1", recent_k=1)

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
#
# Episodic + procedural docs live in sentinel partitions
# ("__episodic__" / "__procedural__") under each user, not the originating
# thread's partition. Public read APIs that filter on c.thread_id must OR in
# an IN (...) clause for those types so they aren't silently excluded.
# ---------------------------------------------------------------------------


def test_get_memories_with_episodic_and_thread_id_emits_or_clause():
    memories = MagicMock()
    memories.query_items.return_value = []
    store = MemoryStore(containers=_containers(memories=memories))

    store.get_memories(user_id="u1", thread_id="t1", memory_types=["episodic"])

    call_kwargs = memories.query_items.call_args.kwargs
    assert "(c.thread_id = @thread_id OR c.type IN (@user_scoped_type_0))" in call_kwargs["query"]
    params = _params_by_name(call_kwargs)
    assert params["@thread_id"] == "t1"
    assert params["@user_scoped_type_0"] == "episodic"


def test_get_memories_with_procedural_and_thread_id_emits_or_clause():
    memories = MagicMock()
    memories.query_items.return_value = []
    store = MemoryStore(containers=_containers(memories=memories))

    store.get_memories(user_id="u1", thread_id="t1", memory_types=["procedural"])

    call_kwargs = memories.query_items.call_args.kwargs
    assert "(c.thread_id = @thread_id OR c.type IN (@user_scoped_type_0))" in call_kwargs["query"]
    params = _params_by_name(call_kwargs)
    assert params["@user_scoped_type_0"] == "procedural"


def test_get_memories_fact_only_with_thread_id_keeps_plain_filter():
    memories = MagicMock()
    memories.query_items.return_value = []
    store = MemoryStore(containers=_containers(memories=memories))

    store.get_memories(user_id="u1", thread_id="t1", memory_types=["fact"])

    call_kwargs = memories.query_items.call_args.kwargs
    assert "c.thread_id = @thread_id" in call_kwargs["query"]
    assert "@user_scoped_type_" not in call_kwargs["query"]


def test_get_memories_no_memory_types_with_thread_id_emits_or_clause():
    # Defaulting to "every type" means episodic + procedural are in scope and
    # would otherwise be silently dropped by a plain c.thread_id equality.
    memories = MagicMock()
    memories.query_items.return_value = []
    store = MemoryStore(containers=_containers(memories=memories))

    store.get_memories(user_id="u1", thread_id="t1")

    call_kwargs = memories.query_items.call_args.kwargs
    assert "(c.thread_id = @thread_id OR c.type IN (@user_scoped_type_0, @user_scoped_type_1))" in call_kwargs["query"]
    params = _params_by_name(call_kwargs)
    assert sorted(v for k, v in params.items() if k.startswith("@user_scoped_type_")) == ["episodic", "procedural"]


def test_search_with_episodic_and_thread_id_forces_cross_partition():
    memories = MagicMock()
    memories.query_items.return_value = []
    embeddings = MagicMock()
    embeddings.generate.return_value = [0.1, 0.2]
    store = MemoryStore(containers=_containers(memories=memories), embeddings_client=embeddings)

    store.search(
        search_terms="hotels",
        user_id="u1",
        thread_id="t1",
        memory_types=["episodic"],
    )

    call_kwargs = memories.query_items.call_args.kwargs
    # When user-scoped types are in scope, search must fan out across
    # partitions instead of confining to [u1, t1] (where no episodic
    # ever lives - they all use the "__episodic__" sentinel partition).
    assert call_kwargs.get("enable_cross_partition_query") is True
    assert "partition_key" not in call_kwargs
    assert "(c.thread_id = @thread_id OR c.type IN (@user_scoped_type_0))" in call_kwargs["query"]


def test_search_fact_only_with_thread_id_uses_partition_path():
    memories = MagicMock()
    memories.query_items.return_value = []
    embeddings = MagicMock()
    embeddings.generate.return_value = [0.1, 0.2]
    store = MemoryStore(containers=_containers(memories=memories), embeddings_client=embeddings)

    store.search(
        search_terms="hotels",
        user_id="u1",
        thread_id="t1",
        memory_types=["fact"],
    )

    call_kwargs = memories.query_items.call_args.kwargs
    assert call_kwargs.get("partition_key") == ["u1", "t1"]
    assert "enable_cross_partition_query" not in call_kwargs


def test_add_turn_skips_embedding_by_default():
    turns = MagicMock()
    embeddings = MagicMock()
    embeddings.generate.return_value = [0.1, 0.2]
    store = MemoryStore(containers=_containers(turns=turns), embeddings_client=embeddings)

    store.add(user_id="u1", role="user", content="hello", thread_id="t1")

    embeddings.generate.assert_not_called()
    body = turns.upsert_item.call_args.kwargs["body"]
    assert "embedding" not in body


def test_add_turn_embeds_when_enabled():
    turns = MagicMock()
    embeddings = MagicMock()
    embeddings.generate.return_value = [0.1, 0.2]
    store = MemoryStore(
        containers=_containers(turns=turns),
        embeddings_client=embeddings,
        enable_turn_embeddings=True,
    )

    store.add(user_id="u1", role="user", content="hello", thread_id="t1")

    embeddings.generate.assert_called_once_with("hello")
    body = turns.upsert_item.call_args.kwargs["body"]
    assert body["embedding"] == [0.1, 0.2]


def test_push_skips_turn_embedding_by_default():
    turns = MagicMock()
    embeddings = MagicMock()
    embeddings.generate_batch.return_value = [[0.1, 0.2]]
    local = [_doc(id="x1", type="turn", content="hello", thread_id="t1")]
    store = MemoryStore(containers=_containers(turns=turns), embeddings_client=embeddings)

    store.push(local, batch_size=10)

    embeddings.generate_batch.assert_not_called()
    body = turns.upsert_item.call_args.kwargs["body"]
    assert "embedding" not in body


def test_push_embeds_turns_when_enabled():
    turns = MagicMock()
    embeddings = MagicMock()
    embeddings.generate_batch.return_value = [[0.1, 0.2]]
    local = [_doc(id="x1", type="turn", content="hello", thread_id="t1")]
    store = MemoryStore(
        containers=_containers(turns=turns),
        embeddings_client=embeddings,
        enable_turn_embeddings=True,
    )

    store.push(local, batch_size=10)

    embeddings.generate_batch.assert_called_once_with(["hello"])
    body = turns.upsert_item.call_args.kwargs["body"]
    assert body["embedding"] == [0.1, 0.2]


def test_search_turns_queries_turns_container():
    turns = MagicMock()
    turns.query_items.return_value = []
    memories = MagicMock()
    memories.query_items.return_value = []
    embeddings = MagicMock()
    embeddings.generate.return_value = [0.1, 0.2]
    store = MemoryStore(
        containers=_containers(turns=turns, memories=memories),
        embeddings_client=embeddings,
    )

    store.search_turns(search_terms="hello", user_id="u1", thread_id="t1")

    turns.query_items.assert_called_once()
    memories.query_items.assert_not_called()
    sql = turns.query_items.call_args.kwargs["query"]
    assert "VectorDistance(c.embedding, @embedding)" in sql


def test_search_summaries_queries_summaries_container_scoped_to_user():
    summaries = MagicMock()
    summaries.query_items.return_value = []
    memories = MagicMock()
    memories.query_items.return_value = []
    embeddings = MagicMock()
    embeddings.generate.return_value = [0.1, 0.2]
    store = MemoryStore(
        containers=_containers(summaries=summaries, memories=memories),
        embeddings_client=embeddings,
    )

    # No thread_id: fans across the user's summary partitions (user + all threads).
    store.search_summaries(search_terms="summarize", user_id="u1")

    summaries.query_items.assert_called_once()
    memories.query_items.assert_not_called()
    kwargs = summaries.query_items.call_args.kwargs
    assert "VectorDistance(c.embedding, @embedding)" in kwargs["query"]
    assert {"name": "@user_id", "value": "u1"} in kwargs["parameters"]


def test_search_summaries_requires_user_id():
    store = MemoryStore(containers=_containers(), embeddings_client=MagicMock())
    with pytest.raises(ValidationError, match="user_id is required"):
        store.search_summaries(search_terms="x")


def test_search_turns_scopes_to_single_partition_with_thread_id():
    turns = MagicMock()
    turns.query_items.return_value = []
    embeddings = MagicMock()
    embeddings.generate.return_value = [0.1, 0.2]
    store = MemoryStore(
        containers=_containers(turns=turns),
        embeddings_client=embeddings,
    )

    store.search_turns(search_terms="hello", user_id="u1", thread_id="t1")

    kwargs = turns.query_items.call_args.kwargs
    assert kwargs["partition_key"] == ["u1", "t1"]
    assert "enable_cross_partition_query" not in kwargs


def test_search_turns_fans_out_across_partitions_without_thread_id():
    turns = MagicMock()
    turns.query_items.return_value = []
    embeddings = MagicMock()
    embeddings.generate.return_value = [0.1, 0.2]
    store = MemoryStore(
        containers=_containers(turns=turns),
        embeddings_client=embeddings,
    )

    store.search_turns(search_terms="hello", user_id="u1")

    kwargs = turns.query_items.call_args.kwargs
    assert "partition_key" not in kwargs
    assert kwargs["enable_cross_partition_query"] is True


def test_search_turns_requires_user_id():
    turns = MagicMock()
    embeddings = MagicMock()
    embeddings.generate.return_value = [0.1, 0.2]
    store = MemoryStore(
        containers=_containers(turns=turns),
        embeddings_client=embeddings,
    )

    with pytest.raises(ValidationError):
        store.search_turns(search_terms="hello", user_id=None)
    turns.query_items.assert_not_called()


def test_search_does_not_query_turns_container():
    turns = MagicMock()
    turns.query_items.return_value = []
    memories = MagicMock()
    memories.query_items.return_value = []
    embeddings = MagicMock()
    embeddings.generate.return_value = [0.1, 0.2]
    store = MemoryStore(
        containers=_containers(turns=turns, memories=memories),
        embeddings_client=embeddings,
    )

    store.search(search_terms="hello", user_id="u1", thread_id="t1")

    memories.query_items.assert_called_once()
    turns.query_items.assert_not_called()
