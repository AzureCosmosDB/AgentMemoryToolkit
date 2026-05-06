"""Tests for DurableFunctionProcessor — verifies all methods are no-ops."""

from __future__ import annotations

from agent_memory_toolkit.processors import (
    DurableFunctionProcessor,
    ProcessThreadResult,
    UserSummaryResult,
)


def test_process_thread_returns_empty_result():
    proc = DurableFunctionProcessor()
    result = proc.process_thread(
        user_id="u1",
        thread_id="t1",
        turns=[{"role": "user", "content": "hi"}],
    )
    assert isinstance(result, ProcessThreadResult)
    assert result.thread_summary is None
    assert result.extracted_counts == {}
    assert result.reconciled_count == 0
    assert result.elapsed_ms == 0


def test_generate_user_summary_returns_empty_result():
    proc = DurableFunctionProcessor()
    result = proc.generate_user_summary(user_id="u1", thread_summaries=[{"thread_id": "t1"}])
    assert isinstance(result, UserSummaryResult)
    assert result.summary is None


def test_close_is_noop():
    assert DurableFunctionProcessor().close() is None


def test_does_not_invoke_pipeline():
    """Sanity check: instantiating + calling never imports/uses ProcessingPipeline."""
    proc = DurableFunctionProcessor()
    # No pipeline attribute should exist
    assert not hasattr(proc, "_pipeline")
    proc.process_thread(user_id="u", thread_id="t", turns=[])
    proc.generate_user_summary(user_id="u", thread_summaries=[])
