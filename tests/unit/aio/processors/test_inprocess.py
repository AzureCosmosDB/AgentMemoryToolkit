"""Tests for AsyncInProcessProcessor."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from agent_memory_toolkit.aio.processors import AsyncInProcessProcessor, ProcessThreadResult


@pytest.mark.asyncio
async def test_process_thread_calls_summarize_extract_reconcile_in_order():
    pipeline = MagicMock()
    pipeline.generate_thread_summary.return_value = {"id": "summary"}
    pipeline.extract_memories.return_value = {"facts": 1}
    pipeline.reconcile_memories.return_value = {"merged": 2, "contradicted": 0, "kept": 3}

    proc = AsyncInProcessProcessor(pipeline=pipeline)
    result = await proc.process_thread(user_id="u", thread_id="t", turns=[])

    method_order = [c[0] for c in pipeline.method_calls]
    assert method_order == [
        "generate_thread_summary",
        "extract_memories",
        "reconcile_memories",
    ]
    assert isinstance(result, ProcessThreadResult)
    assert result.thread_summary == {"id": "summary"}
    assert result.reconciled_count == 2


@pytest.mark.asyncio
async def test_generate_user_summary_passes_thread_ids():
    pipeline = MagicMock()
    pipeline.generate_user_summary.return_value = {"id": "us"}
    proc = AsyncInProcessProcessor(pipeline=pipeline)
    res = await proc.generate_user_summary(
        user_id="u",
        thread_summaries=[{"thread_id": "t1"}, {"thread_id": "t2"}],
    )
    pipeline.generate_user_summary.assert_called_once_with("u", ["t1", "t2"])
    assert res.summary == {"id": "us"}


@pytest.mark.asyncio
async def test_close_is_noop():
    proc = AsyncInProcessProcessor(pipeline=MagicMock())
    assert await proc.close() is None


@pytest.mark.asyncio
async def test_process_reconcile_invokes_pipeline_with_env_pool_size(monkeypatch):
    """Regression: this was completely broken (ModuleNotFoundError on
    ``from ..thresholds``) — auto-trigger silently never reconciled in
    async deployments. Verify the call now succeeds and forwards the
    env-tunable pool size from ``get_dedup_pool_size``."""
    monkeypatch.setenv("DEDUP_POOL_SIZE", "37")
    pipeline = MagicMock()
    pipeline.reconcile_memories.return_value = {"merged": 4, "contradicted": 1, "kept": 9}

    proc = AsyncInProcessProcessor(pipeline=pipeline)
    count = await proc.process_reconcile(user_id="u")

    pipeline.reconcile_memories.assert_called_once_with("u", 37)
    assert count == 5  # merged + contradicted


@pytest.mark.asyncio
async def test_process_extract_memories_invokes_pipeline_and_filters_to_ints():
    pipeline = MagicMock()
    pipeline.extract_memories.return_value = {
        "facts_count": 3,
        "procedural_count": 1,
        "non_int_field": "skip me",
    }

    proc = AsyncInProcessProcessor(pipeline=pipeline)
    result = await proc.process_extract_memories(user_id="u", thread_id="t")

    pipeline.extract_memories.assert_called_once_with("u", "t")
    assert result == {"facts_count": 3, "procedural_count": 1}


@pytest.mark.asyncio
async def test_process_thread_summary_invokes_pipeline_and_returns_dict():
    pipeline = MagicMock()
    pipeline.generate_thread_summary.return_value = {"id": "summary-1", "content": "..."}

    proc = AsyncInProcessProcessor(pipeline=pipeline)
    result = await proc.process_thread_summary(user_id="u", thread_id="t")

    pipeline.generate_thread_summary.assert_called_once_with("u", "t")
    assert result == {"id": "summary-1", "content": "..."}


@pytest.mark.asyncio
async def test_process_thread_summary_returns_none_for_non_dict():
    pipeline = MagicMock()
    pipeline.generate_thread_summary.return_value = "not a dict"

    proc = AsyncInProcessProcessor(pipeline=pipeline)
    result = await proc.process_thread_summary(user_id="u", thread_id="t")
    assert result is None
