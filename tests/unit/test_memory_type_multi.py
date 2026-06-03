"""Tests for ``memory_types`` (multi-type filtering on read methods).

Covers ``_build_memory_query_builder`` and the public read-side methods that
forward to it: ``search_cosmos``, ``get_memories``, ``get_thread``.

A non-empty list emits ``c.type IN (@memory_type_0, @memory_type_1, ...)``.
``None`` (default) or an empty list disables the type filter so the call
returns every memory type.
"""

from __future__ import annotations

from unittest.mock import MagicMock

from azure.cosmos.agent_memory._utils import _build_memory_query_builder
from azure.cosmos.agent_memory.cosmos_memory_client import CosmosMemoryClient

# ---------------------------------------------------------------------------
# Helpers — a small replica of the patterns used in test_cosmos_memory_client.
# ---------------------------------------------------------------------------


def _connected_client() -> tuple[CosmosMemoryClient, MagicMock]:
    client = CosmosMemoryClient(use_default_credential=False)
    container = MagicMock()
    container.query_items.return_value = []
    client._memories_container_client = container
    return client, container


def _captured_query(container: MagicMock) -> str:
    assert container.query_items.called, "query_items was not called"
    return container.query_items.call_args.kwargs["query"]


def _captured_params(container: MagicMock) -> list[dict]:
    return container.query_items.call_args.kwargs["parameters"]


# ---------------------------------------------------------------------------
# _build_memory_query_builder behaviour
# ---------------------------------------------------------------------------


def test_memory_types_list_emits_in_clause():
    qb = _build_memory_query_builder(
        user_id="u1",
        memory_types=["fact", "procedural", "episodic"],
    )
    where = qb.build_where()
    params = qb.get_parameters()
    assert "c.type IN (@memory_type_0, @memory_type_1, @memory_type_2)" in where
    values = {p["value"] for p in params if p["name"].startswith("@memory_type_")}
    assert values == {"fact", "procedural", "episodic"}


def test_memory_types_single_element_list_still_uses_in_clause():
    qb = _build_memory_query_builder(user_id="u1", memory_types=["fact"])
    where = qb.build_where()
    assert "c.type IN (@memory_type_0)" in where


def test_memory_types_empty_list_skipped():
    qb = _build_memory_query_builder(user_id="u1", memory_types=[])
    where = qb.build_where()
    assert "c.type" not in where


def test_memory_types_none_skipped():
    qb = _build_memory_query_builder(user_id="u1", memory_types=None)
    where = qb.build_where()
    assert "c.type" not in where


def test_memory_types_list_combines_with_other_filters():
    qb = _build_memory_query_builder(
        user_id="u1",
        thread_id="t1",
        memory_types=["fact", "episodic"],
        min_confidence=0.5,
    )
    where = qb.build_where()
    assert "c.user_id = @user_id" in where
    # Episodic is user-scoped, so the thread_id filter becomes an OR clause
    # instead of the plain equality form.
    assert "(c.thread_id = @thread_id OR c.type IN (@user_scoped_type_0))" in where
    assert "c.type IN (@memory_type_0, @memory_type_1)" in where
    assert "c.confidence >= @min_confidence" in where


# ---------------------------------------------------------------------------
# thread_id translation for user-scoped types (episodic / procedural)
# ---------------------------------------------------------------------------


def test_thread_id_unchanged_when_only_non_user_scoped_types_requested():
    qb = _build_memory_query_builder(user_id="u1", thread_id="t1", memory_types=["fact"])
    where = qb.build_where()
    assert "c.thread_id = @thread_id" in where
    assert "@user_scoped_type_" not in where


def test_thread_id_unchanged_when_only_turn_requested():
    qb = _build_memory_query_builder(user_id="u1", thread_id="t1", memory_types=["turn"])
    where = qb.build_where()
    assert "c.thread_id = @thread_id" in where
    assert "@user_scoped_type_" not in where


def test_thread_id_or_clause_when_episodic_requested():
    qb = _build_memory_query_builder(user_id="u1", thread_id="t1", memory_types=["episodic"])
    where = qb.build_where()
    params = qb.get_parameters()
    assert "(c.thread_id = @thread_id OR c.type IN (@user_scoped_type_0))" in where
    user_scoped_values = sorted(p["value"] for p in params if p["name"].startswith("@user_scoped_type_"))
    assert user_scoped_values == ["episodic"]


def test_thread_id_or_clause_when_procedural_requested():
    qb = _build_memory_query_builder(user_id="u1", thread_id="t1", memory_types=["procedural"])
    where = qb.build_where()
    params = qb.get_parameters()
    assert "(c.thread_id = @thread_id OR c.type IN (@user_scoped_type_0))" in where
    user_scoped_values = sorted(p["value"] for p in params if p["name"].startswith("@user_scoped_type_"))
    assert user_scoped_values == ["procedural"]


def test_thread_id_or_clause_includes_both_when_both_in_scope():
    qb = _build_memory_query_builder(
        user_id="u1",
        thread_id="t1",
        memory_types=["episodic", "procedural"],
    )
    where = qb.build_where()
    params = qb.get_parameters()
    assert "(c.thread_id = @thread_id OR c.type IN (@user_scoped_type_0, @user_scoped_type_1))" in where
    user_scoped_values = sorted(p["value"] for p in params if p["name"].startswith("@user_scoped_type_"))
    assert user_scoped_values == ["episodic", "procedural"]


def test_thread_id_or_clause_when_no_memory_types_filter():
    # No memory_types means "all types", so user-scoped types are implicitly in scope.
    qb = _build_memory_query_builder(user_id="u1", thread_id="t1")
    where = qb.build_where()
    params = qb.get_parameters()
    assert "(c.thread_id = @thread_id OR c.type IN (@user_scoped_type_0, @user_scoped_type_1))" in where
    user_scoped_values = sorted(p["value"] for p in params if p["name"].startswith("@user_scoped_type_"))
    assert user_scoped_values == ["episodic", "procedural"]


def test_thread_id_or_clause_when_empty_memory_types_filter():
    qb = _build_memory_query_builder(user_id="u1", thread_id="t1", memory_types=[])
    where = qb.build_where()
    assert "(c.thread_id = @thread_id OR c.type IN (@user_scoped_type_0, @user_scoped_type_1))" in where


def test_no_thread_id_no_or_clause():
    qb = _build_memory_query_builder(user_id="u1", memory_types=["episodic"])
    where = qb.build_where()
    assert "c.thread_id" not in where
    assert "@user_scoped_type_" not in where


# ---------------------------------------------------------------------------
# Sync client surface — verifies the list reaches the generated SQL.
# ---------------------------------------------------------------------------


def test_get_memories_passes_list_to_in_clause():
    client, container = _connected_client()
    client.get_memories(user_id="u1", memory_types=["fact", "procedural", "episodic"])
    query = _captured_query(container)
    assert "c.type IN (@memory_type_0, @memory_type_1, @memory_type_2)" in query
    type_params = sorted(p["value"] for p in _captured_params(container) if p["name"].startswith("@memory_type_"))
    assert type_params == ["episodic", "fact", "procedural"]


def test_get_thread_does_not_accept_memory_types():
    import pytest

    client, _ = _connected_client()
    with pytest.raises(TypeError):
        client.get_thread(thread_id="t1", memory_types=["turn", "thread_summary"])


def test_search_cosmos_accepts_list():
    """search_cosmos must thread a list of memory types through to the WHERE."""
    client, container = _connected_client()
    client._embeddings_client = MagicMock()
    client._embeddings_client.generate.return_value = [0.0] * 8
    client.search_cosmos(
        search_terms="user preferences",
        user_id="u1",
        memory_types=["fact", "procedural", "episodic"],
    )
    query = _captured_query(container)
    assert "c.type IN (@memory_type_0, @memory_type_1, @memory_type_2)" in query


def test_search_cosmos_empty_list_disables_type_filter():
    client, container = _connected_client()
    client._embeddings_client = MagicMock()
    client._embeddings_client.generate.return_value = [0.0] * 8
    client.search_cosmos(search_terms="x", user_id="u1", memory_types=[])
    query = _captured_query(container)
    where_clause = query.split("FROM c", 1)[1]
    assert "c.type =" not in where_clause
    assert "c.type IN" not in where_clause
    assert all(not p["name"].startswith("@memory_type") for p in _captured_params(container))


def test_get_memories_default_uses_all_memories_types():
    """When ``memory_types`` is omitted, the query filters to all 3 MEMORIES types."""
    client, container = _connected_client()
    client.get_memories(user_id="u1")
    query = _captured_query(container)
    assert "c.type IN (@memory_type_0, @memory_type_1, @memory_type_2)" in query
    type_params = sorted(p["value"] for p in _captured_params(container) if p["name"].startswith("@memory_type_"))
    assert type_params == ["episodic", "fact", "procedural"]
