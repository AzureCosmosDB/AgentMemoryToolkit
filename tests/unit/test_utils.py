"""Unit tests for shared helpers in agent_memory_toolkit._utils."""

import pytest

from agent_memory_toolkit._utils import (
    DEFAULT_TTL_BY_TYPE,
    _build_container_kwargs,
    _make_memory,
    _resolve_distance_function,
    _resolve_embedding_data_type,
    _resolve_full_text_language,
    compute_content_hash,
)
from agent_memory_toolkit.exceptions import ConfigurationError, ValidationError


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
    assert len(h) == 32  # SHA-256 hex truncated to 32 chars (128 bits)


def test_compute_content_hash_whitespace_normalized():
    h1 = compute_content_hash("hello   world")
    h2 = compute_content_hash("hello world")
    assert h1 == h2


def test_compute_content_hash_case_insensitive():
    h1 = compute_content_hash("Hello World")
    h2 = compute_content_hash("hello world")
    assert h1 == h2  # normalized: lowercased


def test_compute_content_hash_truncated_to_32_chars():
    assert len(compute_content_hash("anything")) == 32


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


def test_resolve_embedding_data_type_defaults(monkeypatch):
    monkeypatch.delenv("AI_FOUNDRY_EMBEDDING_DATA_TYPE", raising=False)
    assert _resolve_embedding_data_type(None) == "float32"


def test_resolve_embedding_data_type_from_env(monkeypatch):
    monkeypatch.setenv("AI_FOUNDRY_EMBEDDING_DATA_TYPE", "int8")
    assert _resolve_embedding_data_type(None) == "int8"


def test_resolve_embedding_data_type_explicit_overrides_env(monkeypatch):
    monkeypatch.setenv("AI_FOUNDRY_EMBEDDING_DATA_TYPE", "int8")
    assert _resolve_embedding_data_type("uint8") == "uint8"


def test_resolve_embedding_data_type_invalid_raises(monkeypatch):
    monkeypatch.setenv("AI_FOUNDRY_EMBEDDING_DATA_TYPE", "bogus")
    with pytest.raises(ConfigurationError):
        _resolve_embedding_data_type(None)


def test_resolve_distance_function_defaults(monkeypatch):
    monkeypatch.delenv("AI_FOUNDRY_EMBEDDING_DISTANCE_FUNCTION", raising=False)
    assert _resolve_distance_function(None) == "cosine"


def test_resolve_distance_function_from_env(monkeypatch):
    monkeypatch.setenv("AI_FOUNDRY_EMBEDDING_DISTANCE_FUNCTION", "dotproduct")
    assert _resolve_distance_function(None) == "dotproduct"


def test_resolve_distance_function_invalid_raises(monkeypatch):
    monkeypatch.setenv("AI_FOUNDRY_EMBEDDING_DISTANCE_FUNCTION", "manhattan")
    with pytest.raises(ConfigurationError):
        _resolve_distance_function(None)


def test_resolve_full_text_language_defaults(monkeypatch):
    monkeypatch.delenv("COSMOS_DB_FULL_TEXT_LANGUAGE", raising=False)
    assert _resolve_full_text_language(None) == "en-US"


def test_resolve_full_text_language_from_env(monkeypatch):
    monkeypatch.setenv("COSMOS_DB_FULL_TEXT_LANGUAGE", "fr-FR")
    assert _resolve_full_text_language(None) == "fr-FR"
