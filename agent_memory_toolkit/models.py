"""Pydantic data models for the Agent Memory Toolkit.

Provides typed, validated models that replace raw dicts for memory records,
search results, and orchestration responses. All models serialize to/from
Cosmos DB-compatible JSON.
"""

import logging
import re
import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Literal, Optional

from pydantic import BaseModel, Field, field_validator

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class MemoryRole(str, Enum):
    """Allowed roles for a memory record."""

    user = "user"
    agent = "agent"
    tool = "tool"
    system = "system"


class MemoryType(str, Enum):
    """Allowed memory types stored in Cosmos DB."""

    turn = "turn"
    summary = "summary"
    fact = "fact"
    user_summary = "user_summary"
    procedural = "procedural"
    episodic = "episodic"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _uuid4_str() -> str:
    return str(uuid.uuid4())


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


# ---------------------------------------------------------------------------
# Tag validation
# ---------------------------------------------------------------------------

TAG_PATTERN = re.compile(r"^[a-z0-9][a-z0-9_:./-]{0,99}$")

# ---------------------------------------------------------------------------
# Core model
# ---------------------------------------------------------------------------


class MemoryRecord(BaseModel):
    """A single memory document stored in Cosmos DB.

    The ``memory_type`` field is named ``memory_type`` in Python to avoid
    shadowing the built-in ``type``, but it serializes to/from ``"type"``
    for Cosmos DB compatibility via a Pydantic alias.
    """

    model_config = {
        "populate_by_name": True,
        "use_enum_values": True,
    }

    id: str = Field(default_factory=_uuid4_str)
    user_id: str
    thread_id: str = Field(default_factory=_uuid4_str)
    role: MemoryRole
    memory_type: MemoryType = Field(alias="type", default=MemoryType.turn)
    content: str
    metadata: dict[str, Any] = Field(default_factory=dict)
    embedding: Optional[list[float]] = None
    agent_id: Optional[str] = None
    created_at: str = Field(default_factory=_utc_now_iso)
    updated_at: Optional[str] = None
    tags: list[str] = Field(default_factory=list)
    ttl: Optional[int] = None
    salience: Optional[float] = None
    confidence: Optional[float] = None
    content_hash: Optional[str] = None
    superseded_by: Optional[str] = None
    supersede_reason: Optional[Literal["duplicate", "contradiction", "update"]] = None
    superseded_at: Optional[str] = None
    supersedes_ids: list[str] = Field(default_factory=list)
    source_memory_ids: list[str] = Field(default_factory=list)

    # -- validators ----------------------------------------------------------

    @field_validator("role", mode="before")
    @classmethod
    def _validate_role(cls, v: Any) -> Any:
        if isinstance(v, str):
            try:
                return MemoryRole(v)
            except ValueError:
                valid = ", ".join(r.value for r in MemoryRole)
                raise ValueError(f"role must be one of {{{valid}}}, got '{v}'")
        return v

    @field_validator("memory_type", mode="before")
    @classmethod
    def _validate_memory_type(cls, v: Any) -> Any:
        if isinstance(v, str):
            try:
                return MemoryType(v)
            except ValueError:
                valid = ", ".join(t.value for t in MemoryType)
                raise ValueError(f"type must be one of {{{valid}}}, got '{v}'")
        return v

    @field_validator("tags", mode="before")
    @classmethod
    def _validate_tags(cls, v: Any) -> list[str]:
        if v is None:
            return []
        if not isinstance(v, list):
            raise ValueError("tags must be a list of strings")
        validated = []
        for tag in v:
            tag = str(tag).strip().lower()
            if not tag:
                continue
            if not TAG_PATTERN.match(tag):
                raise ValueError(f"Invalid tag format: '{tag}'. Must match [a-z0-9][a-z0-9_:./-]{{0,99}}")
            validated.append(tag)
        return sorted(set(validated))

    @field_validator("salience", mode="before")
    @classmethod
    def _validate_salience(cls, v: Any) -> Any:
        if v is not None and (v < 0.0 or v > 1.0):
            raise ValueError(f"salience must be between 0.0 and 1.0, got {v}")
        return v

    @field_validator("confidence", mode="before")
    @classmethod
    def _validate_confidence(cls, v: Any) -> Any:
        if v is not None and (v < 0.0 or v > 1.0):
            raise ValueError(f"confidence must be between 0.0 and 1.0, got {v}")
        return v

    # -- serialization helpers -----------------------------------------------

    def to_cosmos_dict(self) -> dict[str, Any]:
        """Return a dict suitable for Cosmos DB upsert.

        * Uses ``"type"`` as the key name (not ``"memory_type"``).
        * Always emits ``tags``.
        * Omits keys whose value is ``None`` or empty list (for optional fields).
        """
        data: dict[str, Any] = {
            "id": self.id,
            "user_id": self.user_id,
            "thread_id": self.thread_id,
            "role": self.role,
            "type": self.memory_type,
            "content": self.content,
            "metadata": self.metadata,
            "created_at": self.created_at,
            "tags": self.tags,
        }
        if self.embedding is not None:
            data["embedding"] = self.embedding
        if self.agent_id is not None:
            data["agent_id"] = self.agent_id
        if self.updated_at is not None:
            data["updated_at"] = self.updated_at
        if self.ttl is not None:
            data["ttl"] = self.ttl
        if self.salience is not None:
            data["salience"] = self.salience
        if self.confidence is not None:
            data["confidence"] = self.confidence
        if self.content_hash is not None:
            data["content_hash"] = self.content_hash
        if self.superseded_by is not None:
            data["superseded_by"] = self.superseded_by
        if self.supersede_reason is not None:
            data["supersede_reason"] = self.supersede_reason
        if self.superseded_at is not None:
            data["superseded_at"] = self.superseded_at
        if self.supersedes_ids:
            data["supersedes_ids"] = self.supersedes_ids
        if self.source_memory_ids:
            data["source_memory_ids"] = self.source_memory_ids
        return data

    @classmethod
    def from_cosmos_dict(cls, doc: dict[str, Any]) -> "MemoryRecord":
        """Create a ``MemoryRecord`` from a Cosmos DB document dict.

        Handles the ``"type"`` → ``memory_type`` mapping automatically via
        the Pydantic alias.  Extra Cosmos system fields (e.g. ``_rid``,
        ``_ts``) are silently ignored.
        """
        return cls.model_validate(doc)


# ---------------------------------------------------------------------------
# Search result wrapper
# ---------------------------------------------------------------------------


class SearchResult(BaseModel):
    """A memory record returned from a similarity or keyword search."""

    record: MemoryRecord
    score: Optional[float] = None


# ---------------------------------------------------------------------------
# Orchestration result
# ---------------------------------------------------------------------------


class OrchestrationResult(BaseModel):
    """Response envelope for Durable Functions orchestration calls."""

    runtime_status: str
    output: Optional[Any] = None
    custom_status: Optional[Any] = None
    instance_id: Optional[str] = None
