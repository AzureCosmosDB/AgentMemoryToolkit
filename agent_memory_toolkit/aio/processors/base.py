"""Async :class:`AsyncMemoryProcessor` Protocol and result dataclasses.

Re-exports the sync result dataclasses (they are pure data) and defines
an ``async``-flavoured Protocol parallel to
:class:`agent_memory_toolkit.processors.base.MemoryProcessor`.
"""

from __future__ import annotations

from typing import Any, Optional, Protocol, runtime_checkable

from agent_memory_toolkit.processors.base import (
    ProcessThreadResult,
    UserSummaryResult,
)


@runtime_checkable
class AsyncMemoryProcessor(Protocol):
    """Async backend that turns raw turns into summaries + extracted memories."""

    async def process_thread(
        self,
        *,
        user_id: str,
        thread_id: str,
        turns: list[dict[str, Any]],
        existing_memories: Optional[list[dict[str, Any]]] = None,
    ) -> ProcessThreadResult: ...

    async def process_extract_memories(
        self,
        *,
        user_id: str,
        thread_id: str,
    ) -> dict[str, int]: ...

    async def process_thread_summary(
        self,
        *,
        user_id: str,
        thread_id: str,
    ) -> Optional[dict[str, Any]]: ...

    async def process_user_summary(
        self,
        *,
        user_id: str,
        thread_ids: Optional[list[str]] = None,
    ) -> UserSummaryResult: ...

    async def process_reconcile(
        self,
        *,
        user_id: str,
    ) -> int: ...

    async def generate_user_summary(
        self,
        *,
        user_id: str,
        thread_summaries: list[dict[str, Any]],
    ) -> UserSummaryResult: ...

    async def close(self) -> None: ...


__all__ = [
    "AsyncMemoryProcessor",
    "ProcessThreadResult",
    "UserSummaryResult",
]
