"""Runtime-checkable Protocol satisfaction tests for ``MemoryProcessor``."""

from __future__ import annotations

from typing import Any, Optional
from unittest.mock import MagicMock

from agent_memory_toolkit.processors import (
    DurableFunctionProcessor,
    InProcessProcessor,
    MemoryProcessor,
    ProcessThreadResult,
    UserSummaryResult,
)


class _FullDummy:
    """A custom class that fully implements the MemoryProcessor protocol."""

    def process_thread(
        self,
        *,
        user_id: str,
        thread_id: str,
        turns: list[dict[str, Any]],
        existing_memories: Optional[list[dict[str, Any]]] = None,
    ) -> ProcessThreadResult:
        return ProcessThreadResult()

    def process_extract_memories(
        self,
        *,
        user_id: str,
        thread_id: str,
    ) -> dict[str, int]:
        return {}

    def process_thread_summary(
        self,
        *,
        user_id: str,
        thread_id: str,
    ) -> Optional[dict[str, Any]]:
        return {}

    def process_user_summary(
        self,
        *,
        user_id: str,
        thread_ids: Optional[list[str]] = None,
    ) -> UserSummaryResult:
        return UserSummaryResult()

    def process_reconcile(
        self,
        *,
        user_id: str,
    ) -> int:
        return 0

    def generate_user_summary(
        self,
        *,
        user_id: str,
        thread_summaries: list[dict[str, Any]],
    ) -> UserSummaryResult:
        return UserSummaryResult()

    def close(self) -> None:
        return None


class _MissingProcessThread:
    """Implements ``generate_user_summary`` and ``close`` but NOT ``process_thread``."""

    def generate_user_summary(
        self,
        *,
        user_id: str,
        thread_summaries: list[dict[str, Any]],
    ) -> UserSummaryResult:
        return UserSummaryResult()

    def close(self) -> None:
        return None


def test_inprocess_processor_satisfies_protocol():
    proc = InProcessProcessor(pipeline=MagicMock())
    assert isinstance(proc, MemoryProcessor)


def test_durable_function_processor_satisfies_protocol():
    assert isinstance(DurableFunctionProcessor(), MemoryProcessor)


def test_plain_object_does_not_satisfy_protocol():
    assert not isinstance(object(), MemoryProcessor)


def test_full_custom_implementation_satisfies_protocol():
    assert isinstance(_FullDummy(), MemoryProcessor)


def test_missing_method_fails_protocol_check():
    assert not isinstance(_MissingProcessThread(), MemoryProcessor)
