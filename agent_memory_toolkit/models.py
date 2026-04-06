"""Pydantic data models for the Agent Memory Toolkit.

Provides typed, validated models that replace raw dicts for memory records,
search results, and orchestration responses. All models serialize to/from
Cosmos DB-compatible JSON.
"""

import logging
import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Optional

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


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _uuid4_str() -> str:
    return str(uuid.uuid4())


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


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

    # -- validators ----------------------------------------------------------

    @field_validator("role", mode="before")
    @classmethod
    def _validate_role(cls, v: Any) -> Any:
        if isinstance(v, str):
            try:
                return MemoryRole(v)
            except ValueError:
                valid = ", ".join(r.value for r in MemoryRole)
                raise ValueError(
                    f"role must be one of {{{valid}}}, got '{v}'"
                )
        return v

    @field_validator("memory_type", mode="before")
    @classmethod
    def _validate_memory_type(cls, v: Any) -> Any:
        if isinstance(v, str):
            try:
                return MemoryType(v)
            except ValueError:
                valid = ", ".join(t.value for t in MemoryType)
                raise ValueError(
                    f"type must be one of {{{valid}}}, got '{v}'"
                )
        return v

    # -- serialization helpers -----------------------------------------------

    def to_cosmos_dict(self) -> dict[str, Any]:
        """Return a dict suitable for Cosmos DB upsert.

        * Uses ``"type"`` as the key name (not ``"memory_type"``).
        * Omits keys whose value is ``None``.
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
        }
        if self.embedding is not None:
            data["embedding"] = self.embedding
        if self.agent_id is not None:
            data["agent_id"] = self.agent_id
        if self.updated_at is not None:
            data["updated_at"] = self.updated_at
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
