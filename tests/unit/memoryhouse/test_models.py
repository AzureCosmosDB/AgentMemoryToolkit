from __future__ import annotations

import pydantic
import pytest

from azure.cosmos.agent_memory.memoryhouse import MemoryACL, MemoryEnvelope, MemoryOwner, MemoryProvenance


def _envelope_kwargs() -> dict:
    """Return one complete wire-format envelope for focused model tests."""
    return {
        "id": "amt/default/turn-1",
        "contentType": "application/vnd.microsoft.amt.turn+json;version=1",
        "owner": {"type": "user", "id": "u1"},
        "provenance": {
            "application": "agent-memory-toolkit",
            "createdBy": "agent-1",
            "createdAt": "2026-08-20T20:00:00+00:00",
            "sourceId": "turn-1",
        },
        "acl": {
            "read": ["user:u1"],
            "write": ["user:u1"],
            "annotate": [],
            "forget": ["user:u1"],
        },
        "annotations": {},
        "payload": {"id": "turn-1", "type": "turn"},
    }


@pytest.mark.parametrize(
    ("owner_type", "scope"),
    [
        ("user", "personal"),
        ("team", "team"),
        ("organization", "organization"),
    ],
)
def test_owner_type_defines_scope(owner_type: str, scope: str):
    owner = MemoryOwner(type=owner_type, id="scope-id")
    assert owner.scope == scope


def test_scope_is_not_serialized():
    # Scope is intentionally derived from owner.type and must never appear as a
    # second persisted field.
    envelope = MemoryEnvelope(**_envelope_kwargs())
    assert envelope.scope == "personal"
    assert "scope" not in envelope.model_dump(mode="json", by_alias=True)
    assert "scope" not in envelope.owner.model_dump(mode="json", by_alias=True)


def test_serialized_scope_is_rejected():
    # Strict envelope validation prevents callers from creating contradictory
    # values such as owner.type="team" with scope="personal".
    kwargs = _envelope_kwargs()
    kwargs["scope"] = "personal"
    with pytest.raises(pydantic.ValidationError, match="scope"):
        MemoryEnvelope(**kwargs)


@pytest.mark.parametrize("model", [MemoryOwner, MemoryACL, MemoryProvenance, MemoryEnvelope])
def test_standardized_models_reject_unknown_fields(model):
    if model is MemoryOwner:
        kwargs = {"type": "user", "id": "u1"}
    elif model is MemoryACL:
        kwargs = {"read": [], "write": [], "annotate": [], "forget": []}
    elif model is MemoryProvenance:
        kwargs = {
            "application": "amt",
            "createdBy": "writer",
            "createdAt": "now",
            "sourceId": "source",
        }
    else:
        kwargs = _envelope_kwargs()
    kwargs["unexpected"] = True
    with pytest.raises(pydantic.ValidationError, match="unexpected"):
        model(**kwargs)


def test_acl_rejects_empty_principal():
    with pytest.raises(pydantic.ValidationError, match="non-empty"):
        MemoryACL(read=[""], write=[], annotate=[], forget=[])


def test_envelope_serializes_wire_aliases():
    doc = MemoryEnvelope(**_envelope_kwargs()).model_dump(mode="json", by_alias=True)
    assert "contentType" in doc
    assert doc["provenance"]["createdBy"] == "agent-1"
    assert "content_type" not in doc
