"""Exceptions raised by the MemoryHouse envelope integration.

The adapter has its own exception family because envelope validation failures
are different from AMT record validation and Cosmos persistence failures.
Callers can catch :class:`MemoryEnvelopeError` for the whole integration while
still distinguishing malformed input, unsupported formats, and integrity
violations.
"""

from azure.cosmos.agent_memory.exceptions import AgentMemoryError


class MemoryEnvelopeError(AgentMemoryError):
    """Base exception for MemoryHouse envelope failures."""

    error_code = "memory_envelope"


class MemoryEnvelopeValidationError(MemoryEnvelopeError):
    """Raised when an envelope or envelope input is structurally malformed."""

    error_code = "memory_envelope_validation"


class UnsupportedMemoryContentTypeError(MemoryEnvelopeError):
    """Raised when an envelope uses an unsupported content type or version."""

    error_code = "unsupported_memory_content_type"


class MemoryEnvelopeIntegrityError(MemoryEnvelopeError):
    """Raised when duplicated identity or type assertions disagree.

    MemoryHouse intentionally keeps the AMT payload opaque, but the envelope
    still repeats the AMT ID for global addressing and the AMT type in the
    content type. These checks prevent an envelope from pointing at one record
    while carrying another record as its payload.
    """

    error_code = "memory_envelope_integrity"


__all__ = [
    "MemoryEnvelopeError",
    "MemoryEnvelopeIntegrityError",
    "MemoryEnvelopeValidationError",
    "UnsupportedMemoryContentTypeError",
]
