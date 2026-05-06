"""Tests for AsyncInProcessProcessor."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from agent_memory_toolkit.aio.processors import AsyncInProcessProcessor, ProcessThreadResult


@pytest.mark.asyncio
async def test_process_thread_calls_summarize_extract_dedup_in_order():
    pipeline = MagicMock()
    pipeline.generate_thread_summary.return_value = {"id": "summary"}
    pipeline.extract_memories.return_value = {"facts": 1}
    pipeline.deduplicate_facts.return_value = {"merged": 2, "superseded": 0, "kept": 3}

    proc = AsyncInProcessProcessor(pipeline=pipeline)
    result = await proc.process_thread(user_id="u", thread_id="t", turns=[])

    method_order = [c[0] for c in pipeline.method_calls]
    assert method_order == [
        "generate_thread_summary",
        "extract_memories",
        "deduplicate_facts",
    ]
    assert isinstance(result, ProcessThreadResult)
    assert result.thread_summary == {"id": "summary"}
    assert result.deduplicated_count == 2


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
