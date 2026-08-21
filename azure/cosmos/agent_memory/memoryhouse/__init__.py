"""MemoryHouse envelope integration for Agent Memory Toolkit records."""

from azure.cosmos.agent_memory.memoryhouse.adapter import (
    AMT_APPLICATION_NAME,
    AMT_MEMORY_CONTENT_TYPES,
    AMTMemoryEnvelopeAdapter,
    build_envelope_id,
    content_type_for_memory_type,
    memory_type_for_content_type,
    parse_envelope_id,
)
from azure.cosmos.agent_memory.memoryhouse.exceptions import (
    MemoryEnvelopeError,
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

__all__ = [
    "AMT_APPLICATION_NAME",
    "AMT_MEMORY_CONTENT_TYPES",
    "AMTMemoryEnvelopeAdapter",
    "MemoryACL",
    "MemoryEnvelope",
    "MemoryEnvelopeError",
    "MemoryEnvelopeIntegrityError",
    "MemoryEnvelopeValidationError",
    "MemoryOwner",
    "MemoryProvenance",
    "UnsupportedMemoryContentTypeError",
    "build_envelope_id",
    "content_type_for_memory_type",
    "memory_type_for_content_type",
    "parse_envelope_id",
]
