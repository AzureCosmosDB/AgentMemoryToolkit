"""Unit tests for shared helpers in agent_memory_toolkit._utils."""

import pytest

from agent_memory_toolkit._utils import (
    DEFAULT_TTL_BY_TYPE,
    _build_container_kwargs,
    _make_memory,
    compute_content_hash,
)
from agent_memory_toolkit.exceptions import ValidationError


def test_build_container_kwargs_includes_required_fields_and_extras():
    partition_key = object()
    throughput = object()

    kwargs = _build_container_kwargs(
        container_id="memories",
        partition_key=partition_key,
        offer_throughput=throughput,
        indexing_policy={"includedPaths": [{"path": "/*"}]},
        full_text_policy={"defaultLanguage": "en-US"},
    )

    assert kwargs["id"] == "memories"
    assert kwargs["partition_key"] is partition_key
    assert kwargs["offer_throughput"] is throughput
    assert kwargs["indexing_policy"] == {"includedPaths": [{"path": "/*"}]}
    assert kwargs["full_text_policy"] == {"defaultLanguage": "en-US"}


def test_build_container_kwargs_omits_offer_throughput_when_none():
    kwargs = _build_container_kwargs(
        container_id="leases",
        partition_key="/id",
        offer_throughput=None,
    )

    assert kwargs == {
        "id": "leases",
        "partition_key": "/id",
    }


# ---------------------------------------------------------------------------
# compute_content_hash
# ---------------------------------------------------------------------------


def test_compute_content_hash_basic():
    h = compute_content_hash("hello world")
    assert isinstance(h, str)
    assert len(h) == 64  # SHA-256 hex


def test_compute_content_hash_whitespace_normalized():
    h1 = compute_content_hash("hello   world")
    h2 = compute_content_hash("hello world")
    assert h1 == h2


def test_compute_content_hash_preserves_case():
    h1 = compute_content_hash("Hello World")
    h2 = compute_content_hash("hello world")
    assert h1 != h2  # case is preserved


def test_compute_content_hash_strips():
    h1 = compute_content_hash("  hello world  ")
    h2 = compute_content_hash("hello world")
    assert h1 == h2


def test_compute_content_hash_deterministic():
    h1 = compute_content_hash("test content")
    h2 = compute_content_hash("test content")
    assert h1 == h2


def test_compute_content_hash_different_content():
    h1 = compute_content_hash("hello")
    h2 = compute_content_hash("world")
    assert h1 != h2


# ---------------------------------------------------------------------------
# DEFAULT_TTL_BY_TYPE
# ---------------------------------------------------------------------------


def test_default_ttl_by_type():
    assert DEFAULT_TTL_BY_TYPE["turn"] == 2_592_000
    assert DEFAULT_TTL_BY_TYPE["summary"] is None
    assert DEFAULT_TTL_BY_TYPE["fact"] is None
    assert DEFAULT_TTL_BY_TYPE["user_summary"] is None
    assert DEFAULT_TTL_BY_TYPE["episodic"] == 7_776_000
    assert DEFAULT_TTL_BY_TYPE["procedural"] is None


# ---------------------------------------------------------------------------
# _make_memory
# ---------------------------------------------------------------------------


def test_make_memory_with_tags():
    m = _make_memory(user_id="u1", role="user", content="test", tags=["topic:x"])
    assert m["tags"] == ["topic:x"]


def test_make_memory_default_tags_empty():
    m = _make_memory(user_id="u1", role="user", content="test")
    assert m["tags"] == []


def test_make_memory_default_ttl_turn():
    m = _make_memory(user_id="u1", role="user", content="test", memory_type="turn")
    assert m["ttl"] == 2_592_000


def test_make_memory_default_ttl_fact():
    m = _make_memory(user_id="u1", role="user", content="test", memory_type="fact")
    assert "ttl" not in m  # None TTL should not be included


def test_make_memory_default_ttl_episodic():
    m = _make_memory(user_id="u1", role="user", content="test", memory_type="episodic")
    assert m["ttl"] == 7_776_000


def test_make_memory_override_ttl():
    m = _make_memory(user_id="u1", role="user", content="test", memory_type="turn", ttl=3600)
    assert m["ttl"] == 3600


def test_make_memory_new_types():
    m1 = _make_memory(user_id="u1", role="system", content="rule", memory_type="procedural")
    assert m1["type"] == "procedural"
    m2 = _make_memory(user_id="u1", role="system", content="exp", memory_type="episodic")
    assert m2["type"] == "episodic"


def test_make_memory_salience():
    m = _make_memory(user_id="u1", role="user", content="test", salience=0.85)
    assert m["salience"] == 0.85


def test_make_memory_salience_not_included_when_none():
    m = _make_memory(user_id="u1", role="user", content="test")
    assert "salience" not in m


def test_make_memory_content_hash():
    m = _make_memory(user_id="u1", role="user", content="test", content_hash="hash123")
    assert m["content_hash"] == "hash123"


def test_make_memory_content_hash_not_included_when_none():
    m = _make_memory(user_id="u1", role="user", content="test")
    assert "content_hash" not in m


def test_make_memory_required_fields():
    m = _make_memory(user_id="u1", role="user", content="test")
    assert m["user_id"] == "u1"
    assert m["role"] == "user"
    assert m["content"] == "test"
    assert m["type"] == "turn"
    assert "id" in m
    assert "thread_id" in m
    assert "created_at" in m
    assert m["metadata"] == {}


def test_make_memory_invalid_role():
    with pytest.raises(ValidationError):
        _make_memory(user_id="u1", role="invalid", content="test")


def test_make_memory_invalid_type():
    with pytest.raises(ValidationError):
        _make_memory(user_id="u1", role="user", content="test", memory_type="invalid")
