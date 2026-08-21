"""Lossless wrapping of AMT records in MemoryHouse envelopes.

The adapter deliberately duplicates only the minimum fields required by the
MemoryHouse contract:

* The envelope ID repeats the AMT ID inside a globally namespaced identifier.
* The content type repeats the AMT record type for format discovery.
* Provenance repeats the source ID and selected writer metadata.

All semantic memory data remains in the opaque payload. In particular, AMT
tags, confidence, salience, thread information, and lifecycle state are not
copied into annotations.
"""

from __future__ import annotations

import copy
import re
from collections.abc import Mapping
from typing import Any
from urllib.parse import quote, unquote

from pydantic import ValidationError as PydanticValidationError

from azure.cosmos.agent_memory.memoryhouse.exceptions import (
    MemoryEnvelopeIntegrityError,
    MemoryEnvelopeValidationError,
    UnsupportedMemoryContentTypeError,
)
from azure.cosmos.agent_memory.memoryhouse.models import (
    MemoryACL,
    MemoryEnvelope,
    MemoryOwner,
    MemoryProvenance,
)
from azure.cosmos.agent_memory.models import MemoryRecordBase

AMT_APPLICATION_NAME = "agent-memory-toolkit"

# A subtype-specific media type lets MemoryHouse scan by broad artifact format
# without introducing a duplicate ``annotations.amt.memoryType`` field.
AMT_MEMORY_CONTENT_TYPES: dict[str, str] = {
    "turn": "application/vnd.microsoft.amt.turn+json;version=1",
    "fact": "application/vnd.microsoft.amt.fact+json;version=1",
    "episodic": "application/vnd.microsoft.amt.episodic+json;version=1",
    "procedural": "application/vnd.microsoft.amt.procedural+json;version=1",
    "thread_summary": "application/vnd.microsoft.amt.thread-summary+json;version=1",
    "user_summary": "application/vnd.microsoft.amt.user-summary+json;version=1",
}

# Reverse lookup is precomputed so parsing and validation use the same
# authoritative mapping as serialization.
_CONTENT_TYPE_TO_MEMORY_TYPE = {
    content_type: memory_type for memory_type, content_type in AMT_MEMORY_CONTENT_TYPES.items()
}

# Insignificant whitespace around the semicolon and equals sign is accepted,
# but extra parameters and unsupported versions are rejected.
_CONTENT_TYPE_PATTERN = re.compile(r"^\s*([^;\s]+)\s*;\s*version\s*=\s*([^;\s]+)\s*$")

# ``urllib.parse.unquote`` tolerates malformed percent escapes by leaving them
# unchanged. Detect them first so IDs cannot acquire multiple textual forms.
_INVALID_PERCENT_ESCAPE = re.compile(r"%(?![0-9A-Fa-f]{2})")


def _require_identifier(value: str, *, name: str) -> str:
    """Require a usable identifier while preserving its exact spelling."""
    if not isinstance(value, str) or not value.strip():
        raise MemoryEnvelopeValidationError(f"{name} must be a non-empty string")
    return value


def build_envelope_id(store_id: str, memory_id: str) -> str:
    """Build the canonical MemoryHouse ID for an AMT record.

    Both components are encoded as path segments. This allows existing AMT IDs
    to contain slashes, spaces, or percent characters without changing their
    identity or adding ambiguous path levels.
    """
    store_id = _require_identifier(store_id, name="store_id")
    memory_id = _require_identifier(memory_id, name="memory_id")
    return f"amt/{quote(store_id, safe='')}/{quote(memory_id, safe='')}"


def parse_envelope_id(envelope_id: str) -> tuple[str, str]:
    """Parse a canonical AMT MemoryHouse ID into store and memory IDs."""
    envelope_id = _require_identifier(envelope_id, name="envelope_id")
    parts = envelope_id.split("/")
    if len(parts) != 3 or parts[0] != "amt":
        raise MemoryEnvelopeValidationError("envelope_id must use the canonical 'amt/{store-id}/{memory-id}' format")
    encoded_store_id, encoded_memory_id = parts[1:]
    if not encoded_store_id or not encoded_memory_id:
        raise MemoryEnvelopeValidationError("envelope_id store and memory segments must not be empty")
    if _INVALID_PERCENT_ESCAPE.search(encoded_store_id) or _INVALID_PERCENT_ESCAPE.search(encoded_memory_id):
        raise MemoryEnvelopeValidationError("envelope_id contains an invalid percent escape")

    store_id = unquote(encoded_store_id)
    memory_id = unquote(encoded_memory_id)

    # Rebuilding the value enforces one canonical representation. For example,
    # an unescaped space or a lower-case percent escape is rejected rather than
    # being accepted as an alias for another envelope ID.
    if build_envelope_id(store_id, memory_id) != envelope_id:
        raise MemoryEnvelopeValidationError("envelope_id is not canonically encoded")
    return store_id, memory_id


def content_type_for_memory_type(memory_type: str) -> str:
    """Return the versioned MemoryHouse content type for an AMT record type."""
    try:
        return AMT_MEMORY_CONTENT_TYPES[memory_type]
    except KeyError as exc:
        raise UnsupportedMemoryContentTypeError(f"Unsupported AMT memory type: {memory_type!r}") from exc


def memory_type_for_content_type(content_type: str) -> str:
    """Parse a supported MemoryHouse content type into its AMT record type."""
    if not isinstance(content_type, str):
        raise UnsupportedMemoryContentTypeError("Memory content type must be a string")
    match = _CONTENT_TYPE_PATTERN.fullmatch(content_type)
    if match is None:
        raise UnsupportedMemoryContentTypeError(f"Unsupported memory content type: {content_type!r}")
    normalized = f"{match.group(1)};version={match.group(2)}"
    try:
        return _CONTENT_TYPE_TO_MEMORY_TYPE[normalized]
    except KeyError as exc:
        raise UnsupportedMemoryContentTypeError(f"Unsupported memory content type: {content_type!r}") from exc


class AMTMemoryEnvelopeAdapter:
    """Convert between typed AMT records and governed MemoryHouse envelopes.

    Authorization is intentionally outside this adapter. MemoryHouse must apply
    the envelope ACL before a caller is allowed to obtain and unwrap a payload.
    """

    def wrap(
        self,
        record: MemoryRecordBase,
        *,
        store_id: str,
        owner: MemoryOwner | Mapping[str, Any],
        acl: MemoryACL | Mapping[str, Any],
        created_by: str,
        annotations: Mapping[str, Any] | None = None,
    ) -> MemoryEnvelope:
        if not isinstance(record, MemoryRecordBase):
            raise MemoryEnvelopeValidationError("record must be a MemoryRecordBase instance")

        try:
            # Validate governance separately from the AMT payload. ``user_id``
            # may supply personal ownership through a trusted caller policy,
            # but this adapter never infers team or organization ownership.
            validated_owner = MemoryOwner.model_validate(owner)
            validated_acl = MemoryACL.model_validate(acl)

            # ``to_doc`` is the canonical AMT wire representation. A deep copy
            # prevents mutations to the returned envelope from changing nested
            # dictionaries or lists owned by the source record.
            payload = copy.deepcopy(record.to_doc())
            provenance = MemoryProvenance(
                application=AMT_APPLICATION_NAME,
                agentId=record.agent_id,
                createdBy=created_by,
                createdAt=record.created_at,
                sourceId=record.id,
                promptId=record.prompt_id,
                promptVersion=record.prompt_version,
            )

            # Caller annotations are accepted only as independent enrichment.
            # The adapter adds no AMT discovery projection because the subtype
            # content type and opaque payload already carry that information.
            return MemoryEnvelope(
                id=build_envelope_id(store_id, record.id),
                # Read the wire discriminator from the serialized payload
                # instead of stringifying the model attribute. Literal enum
                # fields may stringify as ``MemoryType.fact`` even though the
                # AMT document correctly emits ``"fact"``.
                contentType=content_type_for_memory_type(payload["type"]),
                owner=validated_owner,
                provenance=provenance,
                acl=validated_acl,
                annotations=copy.deepcopy(dict(annotations or {})),
                payload=payload,
            )
        except PydanticValidationError as exc:
            raise MemoryEnvelopeValidationError(f"Invalid MemoryHouse envelope input: {exc}") from exc

    def unwrap(self, envelope: MemoryEnvelope | Mapping[str, Any]) -> MemoryRecordBase:
        try:
            # Revalidate model instances as well as mappings so this boundary
            # always applies the current strict wire contract.
            validated_envelope = MemoryEnvelope.model_validate(envelope)
        except PydanticValidationError as exc:
            raise MemoryEnvelopeValidationError(f"Invalid MemoryHouse envelope: {exc}") from exc

        payload = copy.deepcopy(validated_envelope.payload)
        payload_id = payload.get("id")
        payload_type = payload.get("type")
        if not isinstance(payload_id, str) or not payload_id:
            raise MemoryEnvelopeValidationError("MemoryHouse payload.id must be a non-empty string")
        if not isinstance(payload_type, str) or not payload_type:
            raise MemoryEnvelopeValidationError("MemoryHouse payload.type must be a non-empty string")

        # The global envelope ID, provenance source ID, and payload ID must all
        # identify the same AMT record. None is allowed to silently override the
        # others because that would make attribution and deletion unsafe.
        _, envelope_memory_id = parse_envelope_id(validated_envelope.id)
        if envelope_memory_id != payload_id:
            raise MemoryEnvelopeIntegrityError(
                f"Envelope ID memory component {envelope_memory_id!r} does not match payload.id {payload_id!r}"
            )
        if validated_envelope.provenance.source_id != payload_id:
            raise MemoryEnvelopeIntegrityError(
                f"provenance.sourceId {validated_envelope.provenance.source_id!r} "
                f"does not match payload.id {payload_id!r}"
            )

        content_memory_type = memory_type_for_content_type(validated_envelope.content_type)
        if content_memory_type != payload_type:
            raise MemoryEnvelopeIntegrityError(
                f"Content type represents {content_memory_type!r}, but payload.type is {payload_type!r}"
            )

        try:
            # Dispatch through AMT's existing discriminator so subtype-specific
            # validation remains centralized in the established record models.
            return MemoryRecordBase.from_doc(payload)
        except (PydanticValidationError, TypeError, ValueError) as exc:
            raise MemoryEnvelopeValidationError(f"Invalid AMT payload: {exc}") from exc


__all__ = [
    "AMT_APPLICATION_NAME",
    "AMT_MEMORY_CONTENT_TYPES",
    "AMTMemoryEnvelopeAdapter",
    "build_envelope_id",
    "content_type_for_memory_type",
    "memory_type_for_content_type",
    "parse_envelope_id",
]
