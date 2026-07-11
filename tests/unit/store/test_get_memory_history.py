"""Unit tests for get_memory_history - supersession-chain walking.

Covers the sync store logic (chain walking, ordering, partition scoping,
validation, cycle guard) plus the shared projection enrichment. The async
store shares the same helper module, so a single async smoke test guards
parity.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from azure.cosmos.agent_memory._container_routing import ContainerKey
from azure.cosmos.agent_memory.exceptions import ValidationError
from azure.cosmos.agent_memory.store import MemoryStore
from azure.cosmos.agent_memory.store._search_helpers import MEMORY_PROJECTION


def _containers(*, memories=None):
    return {
        ContainerKey.TURNS: MagicMock(),
        ContainerKey.MEMORIES: memories if memories is not None else MagicMock(),
        ContainerKey.SUMMARIES: MagicMock(),
    }


def _fact(fact_id, *, superseded_by=None, superseded_at=None, reason=None, content=""):
    return {
        "id": fact_id,
        "user_id": "u1",
        "thread_id": "t1",
        "type": "fact",
        "content": content or fact_id,
        "metadata": {"category": "biographical"},
        "created_at": "2025-01-01T00:00:00+00:00",
        "superseded_by": superseded_by,
        "superseded_at": superseded_at,
        "supersede_reason": reason,
    }


# ---------------------------------------------------------------------------
# Projection enrichment (Change 1)
# ---------------------------------------------------------------------------


def test_projection_includes_supersession_audit_fields():
    assert "c.superseded_at" in MEMORY_PROJECTION
    assert "c.supersede_reason" in MEMORY_PROJECTION
    # still carries the fields callers already relied on
    assert "c.superseded_by" in MEMORY_PROJECTION
    assert "c.created_at" in MEMORY_PROJECTION


# ---------------------------------------------------------------------------
# get_memory_history
# ---------------------------------------------------------------------------


def test_single_level_history_returns_direct_predecessor():
    memories = MagicMock()
    # First hop finds the doc superseded by 'current'; second hop finds nothing.
    memories.query_items.side_effect = [
        [_fact("v1", superseded_at="2025-02-01T00:00:00+00:00", reason="update")],
        [],
    ]
    store = MemoryStore(containers=_containers(memories=memories))

    history = store.get_memory_history("current", user_id="u1", thread_id="t1")

    assert [d["id"] for d in history] == ["v1"]
    assert history[0]["supersede_reason"] == "update"


def test_multi_level_chain_walks_transitively_newest_first():
    memories = MagicMock()
    memories.query_items.side_effect = [
        [_fact("v2", superseded_at="2025-03-01T00:00:00+00:00", reason="contradict")],
        [_fact("v1", superseded_at="2025-02-01T00:00:00+00:00", reason="update")],
        [],
    ]
    store = MemoryStore(containers=_containers(memories=memories))

    history = store.get_memory_history("current", user_id="u1", thread_id="t1")

    # Ordered most-recently-superseded first.
    assert [d["id"] for d in history] == ["v2", "v1"]


def test_no_history_returns_empty_list():
    memories = MagicMock()
    memories.query_items.return_value = []
    store = MemoryStore(containers=_containers(memories=memories))

    assert store.get_memory_history("current", user_id="u1", thread_id="t1") == []


def test_cycle_is_bounded_and_deduplicated():
    memories = MagicMock()
    # Pathological: the predecessor points back at an already-seen id.
    memories.query_items.side_effect = [
        [_fact("v1", superseded_at="2025-02-01T00:00:00+00:00")],
        [_fact("current")],  # 'current' already in seen -> skipped, loop stops
        [],
    ]
    store = MemoryStore(containers=_containers(memories=memories))

    history = store.get_memory_history("current", user_id="u1", thread_id="t1", max_depth=50)

    assert [d["id"] for d in history] == ["v1"]


def test_thread_scoped_query_uses_single_partition():
    memories = MagicMock()
    memories.query_items.return_value = []
    store = MemoryStore(containers=_containers(memories=memories))

    store.get_memory_history("current", user_id="u1", thread_id="t1")

    kwargs = memories.query_items.call_args.kwargs
    assert kwargs["partition_key"] == ["u1", "t1"]
    assert "enable_cross_partition_query" not in kwargs
    assert "c.thread_id = @thread_id" in kwargs["query"]


def test_history_without_thread_id_fans_out_cross_partition():
    memories = MagicMock()
    memories.query_items.return_value = []
    store = MemoryStore(containers=_containers(memories=memories))

    store.get_memory_history("current", user_id="u1")

    kwargs = memories.query_items.call_args.kwargs
    assert kwargs.get("enable_cross_partition_query") is True
    assert "c.thread_id = @thread_id" not in kwargs["query"]
    assert "c.superseded_by IN (@sid0)" in kwargs["query"]


def test_missing_memory_id_raises():
    store = MemoryStore(containers=_containers())
    with pytest.raises(ValidationError):
        store.get_memory_history("", user_id="u1")


def test_missing_user_id_raises():
    store = MemoryStore(containers=_containers())
    with pytest.raises(ValidationError):
        store.get_memory_history("current", user_id="")
