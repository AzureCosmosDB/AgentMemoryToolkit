"""Runtime-checkable Protocol satisfaction tests for ``AsyncMemoryProcessor``."""

from __future__ import annotations

from typing import Any, Optional
from unittest.mock import MagicMock

from agent_memory_toolkit.aio.processors import (
    AsyncDurableFunctionProcessor,
    AsyncInProcessProcessor,
    AsyncMemoryProcessor,
    ProcessThreadResult,
    UserSummaryResult,
)


class _FullAsyncDummy:
    """Custom class fully implementing the AsyncMemoryProcessor protocol."""

    async def process_thread(
        self,
        *,
        user_id: str,
        thread_id: str,
        turns: list[dict[str, Any]],
        existing_memories: Optional[list[dict[str, Any]]] = None,
    ) -> ProcessThreadResult:
        return ProcessThreadResult()

    async def process_extract_memories(
        self,
        *,
        user_id: str,
        thread_id: str,
    ) -> dict[str, int]:
        return {}

    async def process_thread_summary(
        self,
        *,
        user_id: str,
        thread_id: str,
    ) -> Optional[dict[str, Any]]:
        return {}

    async def process_user_summary(
        self,
        *,
        user_id: str,
        thread_ids: Optional[list[str]] = None,
    ) -> UserSummaryResult:
        return UserSummaryResult()

    async def process_reconcile(
        self,
        *,
        user_id: str,
    ) -> int:
        return 0

    async def generate_user_summary(
        self,
        *,
        user_id: str,
        thread_summaries: list[dict[str, Any]],
    ) -> UserSummaryResult:
        return UserSummaryResult()

    async def close(self) -> None:
        return None


class _MissingAsyncProcessThread:
    """Implements ``generate_user_summary`` and ``close`` but NOT ``process_thread``."""

    async def generate_user_summary(
        self,
        *,
        user_id: str,
        thread_summaries: list[dict[str, Any]],
    ) -> UserSummaryResult:
        return UserSummaryResult()

    async def close(self) -> None:
        return None


def test_async_inprocess_processor_satisfies_protocol():
    proc = AsyncInProcessProcessor(pipeline=MagicMock())
    assert isinstance(proc, AsyncMemoryProcessor)


def test_async_durable_function_processor_satisfies_protocol():
    assert isinstance(AsyncDurableFunctionProcessor(), AsyncMemoryProcessor)


def test_plain_object_does_not_satisfy_async_protocol():
    assert not isinstance(object(), AsyncMemoryProcessor)


def test_full_custom_async_implementation_satisfies_protocol():
    assert isinstance(_FullAsyncDummy(), AsyncMemoryProcessor)


def test_missing_method_fails_async_protocol_check():
    assert not isinstance(_MissingAsyncProcessThread(), AsyncMemoryProcessor)
