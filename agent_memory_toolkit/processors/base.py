"""``MemoryProcessor`` Protocol and result dataclasses.

Defines the pluggable backend contract used by :class:`CosmosMemoryClient`
to turn raw turns into thread summaries / extracted memories / deduplicated
facts. Two built-in implementations satisfy the protocol:

* :class:`agent_memory_toolkit.processors.inprocess.InProcessProcessor`
* :class:`agent_memory_toolkit.processors.durable.DurableFunctionProcessor`
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional, Protocol, runtime_checkable


@dataclass
class ProcessThreadResult:
    """Outcome of a single ``process_thread`` invocation.

    ``extracted_counts`` carries the per-type extraction telemetry returned by
    :meth:`ProcessingPipeline.extract_memories` (e.g.
    ``{"facts_count": 2, "procedural_count": 1, "episodic_count": 0, "updated_count": 0}``).
    The actual extracted memory documents are persisted to Cosmos DB by the
    pipeline; query them back via the SDK's ``get_memories`` if you need the
    raw docs.
    """

    thread_summary: Optional[dict[str, Any]] = None
    extracted_counts: dict[str, int] = field(default_factory=dict)
    deduplicated_count: int = 0
    elapsed_ms: int = 0


@dataclass
class UserSummaryResult:
    """Outcome of a single ``generate_user_summary`` invocation."""

    summary: Optional[dict[str, Any]] = None


@runtime_checkable
class MemoryProcessor(Protocol):
    """Backend that turns raw turns into summaries + extracted memories.

    Implementations must be safe to call from a sync context. The async
    mirror lives at :mod:`agent_memory_toolkit.aio.processors`.
    """

    def process_thread(
        self,
        *,
        user_id: str,
        thread_id: str,
        turns: list[dict[str, Any]],
        existing_memories: Optional[list[dict[str, Any]]] = None,
    ) -> ProcessThreadResult: ...

    def generate_user_summary(
        self,
        *,
        user_id: str,
        thread_summaries: list[dict[str, Any]],
    ) -> UserSummaryResult: ...

    def close(self) -> None: ...


__all__ = [
    "MemoryProcessor",
    "ProcessThreadResult",
    "UserSummaryResult",
]
