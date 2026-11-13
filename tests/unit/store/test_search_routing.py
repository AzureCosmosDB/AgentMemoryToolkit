"""Search routing: a keyword search runs cross-partition, confined to its records by the WHERE clause.

Cosmos DB ranks a ``FullTextScore`` (keyword or hybrid) query only when it runs across partitions. Sent with a
partition key, the same query came back in one fixed order for every question (azure-sdk-for-python#42241), so a
thread holding more memories than ``top_k`` returned the same memories whatever was asked. Vector-only searches rank
correctly either way and keep the key.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from azure.cosmos.agent_memory._container_routing import ContainerKey
from azure.cosmos.agent_memory.exceptions import ValidationError
from azure.cosmos.agent_memory.store import MemoryStore

KEYWORDS = "Hotels in Lisbon"
NO_KEYWORDS = "hello"
THREAD_SEARCHES = [
    ("search", ContainerKey.MEMORIES, {"memory_types": ["fact"]}),
    ("search_turns", ContainerKey.TURNS, {}),
    ("search_summaries", ContainerKey.SUMMARIES, {}),
    ("search_episodic", ContainerKey.MEMORIES, {}),
]


def _store() -> tuple[MemoryStore, dict[ContainerKey, MagicMock]]:
    containers = {key: MagicMock() for key in (ContainerKey.TURNS, ContainerKey.MEMORIES, ContainerKey.SUMMARIES)}
    for container in containers.values():
        container.query_items.return_value = []
    embeddings = MagicMock()
    embeddings.generate.return_value = [0.1, 0.2]
    return MemoryStore(containers=containers, embeddings_client=embeddings), containers


def _calls(container: MagicMock) -> list[dict]:
    return [call.kwargs for call in container.query_items.call_args_list]


def _params(call: dict) -> list[tuple[str, object]]:
    return [(param["name"], param["value"]) for param in call["parameters"]]


@pytest.mark.parametrize(("method", "container_key", "options"), THREAD_SEARCHES)
def test_a_keyword_search_of_a_thread_runs_cross_partition_inside_it(method, container_key, options):
    store, containers = _store()

    getattr(store, method)(user_id="u1", thread_id="t1", search_terms=KEYWORDS, **options)

    (call,) = _calls(containers[container_key])
    assert "partition_key" not in call
    assert call["enable_cross_partition_query"] is True
    assert "c.user_id = @user_id" in call["query"]
    assert "c.thread_id = @thread_id" in call["query"]
    params = _params(call)
    assert ("@user_id", "u1") in params
    assert ("@thread_id", "t1") in params
    assert (
        "ORDER BY RANK RRF(VectorDistance(c.embedding, @embedding), FullTextScore(c.content, @kw0, @kw1))"
        in call["query"]
    ), "keyword searches must rank with full-text, not vectors alone"
    assert [value for name, value in params if name.startswith("@kw")] == ["hotels", "lisbon"], (
        "keywords are bound lowercased, without stopwords"
    )
    assert call["full_text_score_scope"] == "Local"


@pytest.mark.parametrize(("method", "container_key", "options"), THREAD_SEARCHES)
def test_a_vector_only_search_of_a_thread_keeps_the_partition_key(method, container_key, options):
    store, containers = _store()

    getattr(store, method)(user_id="u1", thread_id="t1", search_terms=NO_KEYWORDS, **options)

    (call,) = _calls(containers[container_key])
    assert call["partition_key"] == ["u1", "t1"]
    assert "enable_cross_partition_query" not in call
    assert "FullTextScore" not in call["query"]
    assert "full_text_score_scope" not in call


@pytest.mark.parametrize(("method", "container_key", "options"), THREAD_SEARCHES)
def test_a_keyword_search_of_the_users_threads_ranks_with_local_statistics(method, container_key, options):
    store, containers = _store()

    getattr(store, method)(user_id="u1", search_terms=KEYWORDS, **options)

    (call,) = _calls(containers[container_key])
    assert "partition_key" not in call
    assert call["enable_cross_partition_query"] is True
    assert "c.user_id = @user_id" in call["query"]
    assert "c.thread_id = @thread_id" not in call["query"]
    assert call["full_text_score_scope"] == "Local"


def test_a_keyword_search_with_a_thread_keeps_the_user_scoped_types_reachable():
    store, containers = _store()

    store.search(user_id="u1", thread_id="t1", search_terms=KEYWORDS, memory_types=["fact", "procedural"])

    (call,) = _calls(containers[ContainerKey.MEMORIES])
    assert "partition_key" not in call
    assert "(c.thread_id = @thread_id OR c.type IN (@user_scoped_type_0))" in call["query"]


def test_a_full_text_query_sent_with_a_partition_key_is_rejected():
    """Cosmos DB would return it unranked; failing loudly keeps a future search path from repeating the bug."""
    store, containers = _store()

    with pytest.raises(ValidationError, match="42241"):
        store.query(
            "SELECT TOP 1 c.id FROM c ORDER BY RANK FullTextScore(c.content, 'hotels')",
            [],
            container_key=ContainerKey.MEMORIES,
            partition_key=["u1", "t1"],
        )
    containers[ContainerKey.MEMORIES].query_items.assert_not_called()


def test_a_callers_own_full_text_query_keeps_the_default_statistics():
    store, containers = _store()
    sql = "SELECT TOP 1 c.id FROM c WHERE c.user_id = @u ORDER BY RANK FullTextScore(c.content, 'hotels')"
    parameters = [{"name": "@u", "value": "u1"}]

    store.query(sql, parameters, container_key=ContainerKey.MEMORIES, cross_partition=True)
    store.query(
        sql, parameters, container_key=ContainerKey.MEMORIES, cross_partition=True, full_text_score_scope="Local"
    )

    default, local = _calls(containers[ContainerKey.MEMORIES])
    assert "full_text_score_scope" not in default
    assert local["full_text_score_scope"] == "Local"
