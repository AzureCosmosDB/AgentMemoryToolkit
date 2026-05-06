"""Tests for AsyncDurableFunctionProcessor — verifies all methods are no-ops."""

from __future__ import annotations

import pytest

from agent_memory_toolkit.aio.processors import (
    AsyncDurableFunctionProcessor,
    ProcessThreadResult,
    UserSummaryResult,
)


@pytest.mark.asyncio
async def test_process_thread_returns_empty_result():
    proc = AsyncDurableFunctionProcessor()
    result = await proc.process_thread(user_id="u", thread_id="t", turns=[{"role": "user"}])
    assert isinstance(result, ProcessThreadResult)
    assert result.thread_summary is None
    assert result.extracted_counts == {}
    assert result.reconciled_count == 0
    assert result.elapsed_ms == 0


@pytest.mark.asyncio
async def test_generate_user_summary_returns_empty_result():
    proc = AsyncDurableFunctionProcessor()
    result = await proc.generate_user_summary(user_id="u", thread_summaries=[{"thread_id": "t"}])
    assert isinstance(result, UserSummaryResult)
    assert result.summary is None


@pytest.mark.asyncio
async def test_close_is_noop():
    assert await AsyncDurableFunctionProcessor().close() is None


def test_does_not_carry_pipeline():
    proc = AsyncDurableFunctionProcessor()
    assert not hasattr(proc, "_pipeline")
