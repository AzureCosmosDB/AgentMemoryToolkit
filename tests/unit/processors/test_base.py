"""Tests for the MemoryProcessor Protocol and result dataclasses."""

from __future__ import annotations

from unittest.mock import MagicMock

from agent_memory_toolkit.processors import (
    DurableFunctionProcessor,
    InProcessProcessor,
    MemoryProcessor,
    ProcessThreadResult,
    UserSummaryResult,
)


def test_inprocess_satisfies_protocol():
    proc = InProcessProcessor(pipeline=MagicMock())
    assert isinstance(proc, MemoryProcessor)


def test_durable_satisfies_protocol():
    assert isinstance(DurableFunctionProcessor(), MemoryProcessor)


def test_process_thread_result_defaults():
    r = ProcessThreadResult()
    assert r.thread_summary is None
    assert r.extracted_counts == {}
    assert r.reconciled_count == 0
    assert r.elapsed_ms == 0


def test_user_summary_result_defaults():
    assert UserSummaryResult().summary is None


def test_inprocess_requires_pipeline_or_components():
    import pytest

    with pytest.raises(ValueError):
        InProcessProcessor()
