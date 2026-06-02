"""Tests for the async MemoryProcessor protocol and result dataclasses."""

from __future__ import annotations

from unittest.mock import MagicMock

from azure.cosmos.agent_memory.aio.processors import (
    AsyncDurableFunctionProcessor,
    AsyncInProcessProcessor,
    AsyncMemoryProcessor,
    ProcessThreadResult,
    UserSummaryResult,
)


def test_async_inprocess_satisfies_protocol():
    proc = AsyncInProcessProcessor(pipeline=MagicMock())
    assert isinstance(proc, AsyncMemoryProcessor)


def test_async_durable_satisfies_protocol():
    assert isinstance(AsyncDurableFunctionProcessor(), AsyncMemoryProcessor)


def test_dataclass_defaults():
    assert ProcessThreadResult().extracted_counts == {}
    assert UserSummaryResult().summary is None


def test_async_inprocess_requires_pipeline_or_components():
    import pytest

    with pytest.raises(ValueError):
        AsyncInProcessProcessor()
