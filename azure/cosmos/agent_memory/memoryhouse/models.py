"""MemoryHouse envelope wire models.

These models describe only the standardized MemoryHouse envelope. They do not
attempt to model AMT's semantic payload because that payload remains owned by
AMT and is reconstructed through ``MemoryRecordBase.from_doc()``.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationInfo, field_validator


class _EnvelopeModel(BaseModel):
    """Strict base model shared by every standardized envelope component."""

    # Accept Pythonic field names when constructing models while retaining the
    # camel-case aliases required by the MemoryHouse JSON wire contract.
    # ``extra="forbid"`` is important here: in particular, it rejects a stored
    # ``scope`` field that could drift away from ``owner.type``.
    model_config = ConfigDict(
        populate_by_name=True,
        extra="forbid",
        # Callers may pass an already-created model back through the adapter.
        # Revalidating instances keeps that boundary strict even if a mutable
        # model was changed after its original construction.
        revalidate_instances="always",
    )


def _require_non_empty(value: str, *, field_name: str) -> str:
    """Validate identifiers without rewriting caller-owned values."""
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string")
    return value


class MemoryOwner(_EnvelopeModel):
    """Entity that owns a memory and therefore defines its scope."""

    type: Literal["user", "team", "organization"]
    id: str

    @field_validator("id")
    @classmethod
    def _validate_id(cls, value: str) -> str:
        return _require_non_empty(value, field_name="owner.id")

    @property
    def scope(self) -> Literal["personal", "team", "organization"]:
        """Return the human-facing scope without serializing redundant state."""
        # ``owner.type`` is the single persisted source of truth. Keeping this
        # mapping as a property gives callers the requested scope vocabulary
        # without introducing a second field that must be synchronized.
        return {
            "user": "personal",
            "team": "team",
            "organization": "organization",
        }[self.type]


class MemoryACL(_EnvelopeModel):
    """Principals authorized for each MemoryHouse operation."""

    read: list[str]
    write: list[str]
    annotate: list[str]
    forget: list[str]

    @field_validator("read", "write", "annotate", "forget")
    @classmethod
    def _validate_principals(cls, values: list[str], info: ValidationInfo) -> list[str]:
        # Empty operation lists are valid and mean that no principal is granted
        # that operation. Empty principal strings are invalid because they make
        # authorization intent ambiguous.
        for value in values:
            _require_non_empty(value, field_name=f"acl.{info.field_name} principal")
        return values


class MemoryProvenance(_EnvelopeModel):
    """Origin and writer metadata for a MemoryHouse envelope."""

    application: str
    agent_id: str | None = Field(default=None, alias="agentId")
    created_by: str = Field(alias="createdBy")
    created_at: str = Field(alias="createdAt")
    source_id: str = Field(alias="sourceId")
    prompt_id: str | None = Field(default=None, alias="promptId")
    prompt_version: str | None = Field(default=None, alias="promptVersion")

    @field_validator("application", "created_by", "created_at", "source_id")
    @classmethod
    def _validate_required_string(cls, value: str, info: ValidationInfo) -> str:
        return _require_non_empty(value, field_name=str(info.field_name))

    @field_validator("agent_id", "prompt_id", "prompt_version")
    @classmethod
    def _validate_optional_string(cls, value: str | None, info: ValidationInfo) -> str | None:
        if value is None:
            return None
        return _require_non_empty(value, field_name=str(info.field_name))


class MemoryEnvelope(_EnvelopeModel):
    """Governed MemoryHouse envelope containing an opaque AMT payload."""

    id: str
    content_type: str = Field(alias="contentType")
    owner: MemoryOwner
    provenance: MemoryProvenance
    acl: MemoryACL
    annotations: dict[str, Any]
    payload: dict[str, Any]

    @field_validator("id", "content_type")
    @classmethod
    def _validate_required_string(cls, value: str, info: Any) -> str:
        return _require_non_empty(value, field_name=str(info.field_name))

    @property
    def scope(self) -> Literal["personal", "team", "organization"]:
        """Expose the owner-derived scope directly on the envelope."""
        return self.owner.scope


__all__ = [
    "MemoryACL",
    "MemoryEnvelope",
    "MemoryOwner",
    "MemoryProvenance",
]
