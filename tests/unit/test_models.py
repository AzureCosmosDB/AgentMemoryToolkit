"""Unit tests for agent_memory_toolkit.models."""

import uuid

import pydantic
import pytest

from agent_memory_toolkit.models import (
    MemoryRecord,
    OrchestrationResult,
    SearchResult,
)

# ---------------------------------------------------------------------------
# MemoryRecord – defaults
# ---------------------------------------------------------------------------


def test_memory_record_defaults():
    """Required fields only → id/thread_id are UUIDs, type=turn, metadata={}."""
    rec = MemoryRecord(user_id="u1", role="user", content="hello")
    uuid.UUID(rec.id)  # valid UUID
    uuid.UUID(rec.thread_id)
    assert rec.memory_type == "turn"
    assert rec.metadata == {}
    assert rec.embedding is None
    assert rec.agent_id is None
    assert rec.updated_at is None


def test_memory_record_all_fields(sample_user_id, sample_thread_id, sample_embedding):
    """All fields populated are retained."""
    rec = MemoryRecord(
        id="custom-id",
        user_id=sample_user_id,
        thread_id=sample_thread_id,
        role="agent",
        memory_type="summary",
        content="summary content",
        metadata={"key": "value"},
        embedding=sample_embedding,
        agent_id="agent-1",
        created_at="2024-01-01T00:00:00+00:00",
        updated_at="2024-06-01T00:00:00+00:00",
    )
    assert rec.id == "custom-id"
    assert rec.user_id == sample_user_id
    assert rec.thread_id == sample_thread_id
    assert rec.role == "agent"
    assert rec.memory_type == "summary"
    assert rec.content == "summary content"
    assert rec.metadata == {"key": "value"}
    assert rec.embedding == sample_embedding
    assert rec.agent_id == "agent-1"
    assert rec.updated_at == "2024-06-01T00:00:00+00:00"


# ---------------------------------------------------------------------------
# Role validation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("role", ["user", "agent", "tool", "system"])
def test_valid_roles(role):
    rec = MemoryRecord(user_id="u", role=role, content="c")
    assert rec.role == role


def test_invalid_role():
    with pytest.raises(pydantic.ValidationError, match="role"):
        MemoryRecord(user_id="u", role="invalid_role", content="c")


# ---------------------------------------------------------------------------
# MemoryType validation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("mt", ["turn", "summary", "fact", "user_summary", "procedural", "episodic"])
def test_valid_memory_types(mt):
    rec = MemoryRecord(user_id="u", role="user", content="c", memory_type=mt)
    assert rec.memory_type == mt


def test_memory_type_procedural():
    rec = MemoryRecord(user_id="u1", role="system", content="test", memory_type="procedural")
    assert rec.memory_type == "procedural"


def test_memory_type_episodic():
    rec = MemoryRecord(user_id="u1", role="system", content="test", memory_type="episodic")
    assert rec.memory_type == "episodic"


def test_invalid_memory_type():
    with pytest.raises(pydantic.ValidationError, match="type"):
        MemoryRecord(user_id="u", role="user", content="c", memory_type="bad")


# ---------------------------------------------------------------------------
# to_cosmos_dict
# ---------------------------------------------------------------------------


def test_to_cosmos_dict_uses_type_key():
    rec = MemoryRecord(user_id="u", role="user", content="c")
    d = rec.to_cosmos_dict()
    assert "type" in d
    assert "memory_type" not in d
    assert d["type"] == "turn"


def test_to_cosmos_dict_omits_none():
    rec = MemoryRecord(user_id="u", role="user", content="c")
    d = rec.to_cosmos_dict()
    assert "embedding" not in d
    assert "agent_id" not in d
    assert "updated_at" not in d


def test_to_cosmos_dict_includes_optional_fields(sample_embedding):
    rec = MemoryRecord(
        user_id="u",
        role="user",
        content="c",
        embedding=sample_embedding,
        agent_id="a1",
        updated_at="2024-06-01T00:00:00+00:00",
    )
    d = rec.to_cosmos_dict()
    assert d["embedding"] == sample_embedding
    assert d["agent_id"] == "a1"
    assert d["updated_at"] == "2024-06-01T00:00:00+00:00"


# ---------------------------------------------------------------------------
# from_cosmos_dict round-trip
# ---------------------------------------------------------------------------


def test_from_cosmos_dict_round_trip(sample_embedding):
    original = MemoryRecord(
        id="rt-id",
        user_id="u",
        thread_id="t",
        role="agent",
        memory_type="fact",
        content="a fact",
        metadata={"k": 1},
        embedding=sample_embedding,
        agent_id="ag",
        created_at="2024-01-01T00:00:00+00:00",
        updated_at="2024-06-01T00:00:00+00:00",
    )
    cosmos = original.to_cosmos_dict()
    restored = MemoryRecord.from_cosmos_dict(cosmos)
    assert restored.id == original.id
    assert restored.user_id == original.user_id
    assert restored.thread_id == original.thread_id
    assert restored.role == original.role
    assert restored.memory_type == original.memory_type
    assert restored.content == original.content
    assert restored.metadata == original.metadata
    assert restored.embedding == original.embedding
    assert restored.agent_id == original.agent_id
    assert restored.created_at == original.created_at
    assert restored.updated_at == original.updated_at


def test_from_cosmos_dict_ignores_system_fields(sample_memory_dict):
    doc = {**sample_memory_dict, "_rid": "abc", "_ts": 123, "_etag": "e", "_self": "s"}
    rec = MemoryRecord.from_cosmos_dict(doc)
    assert rec.id == sample_memory_dict["id"]
    assert rec.content == sample_memory_dict["content"]


def test_from_cosmos_dict_handles_type_alias():
    doc = {
        "id": "x",
        "user_id": "u",
        "thread_id": "t",
        "role": "user",
        "type": "summary",
        "content": "c",
        "metadata": {},
        "created_at": "2024-01-01T00:00:00+00:00",
    }
    rec = MemoryRecord.from_cosmos_dict(doc)
    assert rec.memory_type == "summary"


# ---------------------------------------------------------------------------
# SearchResult / OrchestrationResult
# ---------------------------------------------------------------------------


def test_search_result():
    rec = MemoryRecord(user_id="u", role="user", content="c")
    sr = SearchResult(record=rec, score=0.95)
    assert sr.record is rec
    assert sr.score == 0.95

    sr_no_score = SearchResult(record=rec)
    assert sr_no_score.score is None


def test_orchestration_result():
    orch = OrchestrationResult(
        runtime_status="Completed",
        output={"result": 42},
        custom_status="done",
        instance_id="inst-1",
    )
    assert orch.runtime_status == "Completed"
    assert orch.output == {"result": 42}
    assert orch.custom_status == "done"
    assert orch.instance_id == "inst-1"


# ---------------------------------------------------------------------------
# Tags validation
# ---------------------------------------------------------------------------


def test_tags_default_empty():
    rec = MemoryRecord(user_id="u1", role="user", content="test")
    assert rec.tags == []


def test_tags_validation_valid():
    rec = MemoryRecord(user_id="u1", role="user", content="test", tags=["topic:travel", "sys:fact"])
    assert rec.tags == ["sys:fact", "topic:travel"]  # sorted and deduped


def test_tags_validation_lowercase():
    rec = MemoryRecord(user_id="u1", role="user", content="test", tags=["Topic:Travel"])
    assert rec.tags == ["topic:travel"]


def test_tags_validation_invalid_pattern():
    with pytest.raises(pydantic.ValidationError, match="Invalid tag format"):
        MemoryRecord(user_id="u1", role="user", content="test", tags=["invalid tag!"])


def test_tags_validation_too_long():
    with pytest.raises(pydantic.ValidationError, match="Invalid tag format"):
        MemoryRecord(user_id="u1", role="user", content="test", tags=["a" * 101])


def test_tags_none_becomes_empty_list():
    rec = MemoryRecord(user_id="u1", role="user", content="test", tags=None)
    assert rec.tags == []


def test_tags_deduplication():
    rec = MemoryRecord(user_id="u1", role="user", content="test", tags=["a", "b", "a"])
    assert rec.tags == ["a", "b"]


# ---------------------------------------------------------------------------
# Salience validation
# ---------------------------------------------------------------------------


def test_salience_valid():
    rec = MemoryRecord(user_id="u1", role="user", content="test", salience=0.85)
    assert rec.salience == 0.85


def test_salience_none_default():
    rec = MemoryRecord(user_id="u1", role="user", content="test")
    assert rec.salience is None


def test_salience_out_of_range_high():
    with pytest.raises(pydantic.ValidationError, match="salience must be between"):
        MemoryRecord(user_id="u1", role="user", content="test", salience=1.5)


def test_salience_out_of_range_low():
    with pytest.raises(pydantic.ValidationError, match="salience must be between"):
        MemoryRecord(user_id="u1", role="user", content="test", salience=-0.1)


def test_salience_boundary_zero():
    rec = MemoryRecord(user_id="u1", role="user", content="test", salience=0.0)
    assert rec.salience == 0.0


def test_salience_boundary_one():
    rec = MemoryRecord(user_id="u1", role="user", content="test", salience=1.0)
    assert rec.salience == 1.0


# ---------------------------------------------------------------------------
# Confidence validation
# ---------------------------------------------------------------------------


def test_confidence_valid():
    rec = MemoryRecord(user_id="u1", role="user", content="test", confidence=0.92)
    assert rec.confidence == 0.92


def test_confidence_none_default():
    rec = MemoryRecord(user_id="u1", role="user", content="test")
    assert rec.confidence is None


def test_confidence_out_of_range_high():
    with pytest.raises(pydantic.ValidationError, match="confidence must be between"):
        MemoryRecord(user_id="u1", role="user", content="test", confidence=1.5)


def test_confidence_out_of_range_low():
    with pytest.raises(pydantic.ValidationError, match="confidence must be between"):
        MemoryRecord(user_id="u1", role="user", content="test", confidence=-0.1)


def test_confidence_emitted_in_to_cosmos_dict():
    rec = MemoryRecord(user_id="u1", role="user", content="test", confidence=0.7)
    data = rec.to_cosmos_dict()
    assert data["confidence"] == 0.7


def test_confidence_omitted_when_none():
    rec = MemoryRecord(user_id="u1", role="user", content="test")
    data = rec.to_cosmos_dict()
    assert "confidence" not in data


# ---------------------------------------------------------------------------
# TTL field
# ---------------------------------------------------------------------------


def test_ttl_field():
    rec = MemoryRecord(user_id="u1", role="user", content="test", ttl=86400)
    assert rec.ttl == 86400


# ---------------------------------------------------------------------------
# Content hash field
# ---------------------------------------------------------------------------


def test_content_hash_field():
    rec = MemoryRecord(user_id="u1", role="user", content="test", content_hash="abc123")
    assert rec.content_hash == "abc123"


# ---------------------------------------------------------------------------
# Superseded / lineage fields
# ---------------------------------------------------------------------------


def test_superseded_by_field():
    rec = MemoryRecord(user_id="u1", role="user", content="test", superseded_by="new-id")
    assert rec.superseded_by == "new-id"


def test_supersedes_ids_field():
    rec = MemoryRecord(user_id="u1", role="user", content="test", supersedes_ids=["old1", "old2"])
    assert rec.supersedes_ids == ["old1", "old2"]


def test_source_memory_ids_field():
    rec = MemoryRecord(user_id="u1", role="user", content="test", source_memory_ids=["src1"])
    assert rec.source_memory_ids == ["src1"]


# ---------------------------------------------------------------------------
# to_cosmos_dict with new fields
# ---------------------------------------------------------------------------


def test_to_cosmos_dict_includes_tags():
    rec = MemoryRecord(user_id="u1", role="user", content="test", tags=["topic:test"])
    d = rec.to_cosmos_dict()
    assert d["tags"] == ["topic:test"]


def test_to_cosmos_dict_tags_always_present():
    rec = MemoryRecord(user_id="u1", role="user", content="test")
    d = rec.to_cosmos_dict()
    assert d["tags"] == []


def test_to_cosmos_dict_conditional_fields():
    rec = MemoryRecord(
        user_id="u1",
        role="user",
        content="test",
        ttl=86400,
        salience=0.8,
        content_hash="hash",
        superseded_by="new",
        supersedes_ids=["old"],
        source_memory_ids=["src"],
    )
    d = rec.to_cosmos_dict()
    assert d["ttl"] == 86400
    assert d["salience"] == 0.8
    assert d["content_hash"] == "hash"
    assert d["superseded_by"] == "new"
    assert d["supersedes_ids"] == ["old"]
    assert d["source_memory_ids"] == ["src"]


def test_to_cosmos_dict_omits_none_new_fields():
    rec = MemoryRecord(user_id="u1", role="user", content="test")
    d = rec.to_cosmos_dict()
    assert "ttl" not in d
    assert "salience" not in d
    assert "content_hash" not in d
    assert "superseded_by" not in d
    assert "supersedes_ids" not in d
    assert "source_memory_ids" not in d


# ---------------------------------------------------------------------------
# from_cosmos_dict round-trip with new fields
# ---------------------------------------------------------------------------


def test_from_cosmos_dict_round_trip_new_fields(sample_embedding):
    original = MemoryRecord(
        id="rt-new",
        user_id="u",
        thread_id="t",
        role="agent",
        memory_type="procedural",
        content="a rule",
        metadata={"k": 1},
        embedding=sample_embedding,
        tags=["sys:rule"],
        ttl=86400,
        salience=0.9,
        content_hash="abc",
        superseded_by="new-id",
        supersedes_ids=["old-id"],
        source_memory_ids=["src-id"],
    )
    cosmos = original.to_cosmos_dict()
    restored = MemoryRecord.from_cosmos_dict(cosmos)
    assert restored.memory_type == "procedural"
    assert restored.tags == ["sys:rule"]
    assert restored.ttl == 86400
    assert restored.salience == 0.9
    assert restored.content_hash == "abc"
    assert restored.superseded_by == "new-id"
    assert restored.supersedes_ids == ["old-id"]
    assert restored.source_memory_ids == ["src-id"]


def test_supersede_reason_and_at_fields():
    """supersede_reason (Literal) and superseded_at (str) fields behave correctly."""
    rec = MemoryRecord(user_id="u1", role="user", content="hello")
    assert rec.supersede_reason is None
    assert rec.superseded_at is None

    rec_dup = MemoryRecord(user_id="u1", role="user", content="hello", supersede_reason="duplicate")
    assert rec_dup.supersede_reason == "duplicate"

    rec_con = MemoryRecord(user_id="u1", role="user", content="hello", supersede_reason="contradiction")
    assert rec_con.supersede_reason == "contradiction"

    with pytest.raises(pydantic.ValidationError):
        MemoryRecord(user_id="u1", role="user", content="hello", supersede_reason="anything_else")

    rec_at = MemoryRecord(
        user_id="u1",
        role="user",
        content="hello",
        superseded_at="2024-01-01T00:00:00Z",
    )
    assert rec_at.superseded_at == "2024-01-01T00:00:00Z"

    rec_full = MemoryRecord(
        user_id="u1",
        role="user",
        content="hello",
        supersede_reason="duplicate",
        superseded_at="2024-01-01T00:00:00Z",
    )
    out = rec_full.to_cosmos_dict()
    assert out["supersede_reason"] == "duplicate"
    assert out["superseded_at"] == "2024-01-01T00:00:00Z"
