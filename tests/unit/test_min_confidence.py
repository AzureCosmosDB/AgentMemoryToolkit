"""Tests for `min_confidence` filter on get_memories / search_cosmos."""

from __future__ import annotations

from azure.cosmos.agent_memory._utils import _build_memory_query_builder


def test_min_confidence_adds_predicate_when_set():
    qb = _build_memory_query_builder(user_id="u1", min_confidence=0.7)
    where = qb.build_where()
    params = qb.get_parameters()
    assert "c.confidence >= @min_confidence" in where
    assert {"name": "@min_confidence", "value": 0.7} in params


def test_min_confidence_omitted_when_none():
    qb = _build_memory_query_builder(user_id="u1")
    where = qb.build_where()
    assert "c.confidence" not in where
    assert all(p["name"] != "@min_confidence" for p in qb.get_parameters())


def test_min_confidence_zero_is_no_op():
    """min_confidence=0 means 'no filter' (every memory >= 0 trivially)."""
    qb = _build_memory_query_builder(user_id="u1", min_confidence=0.0)
    where = qb.build_where()
    assert "c.confidence" not in where
