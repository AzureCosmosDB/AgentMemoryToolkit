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


@pytest.mark.parametrize("mt", ["turn", "summary", "fact", "user_summary"])
def test_valid_memory_types(mt):
    rec = MemoryRecord(user_id="u", role="user", content="c", memory_type=mt)
    assert rec.memory_type == mt


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
