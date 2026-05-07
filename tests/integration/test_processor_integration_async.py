"""Async mirror of :mod:`tests.integration.test_processor_integration`."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from agent_memory_toolkit.aio.cosmos_memory_client import AsyncCosmosMemoryClient
from agent_memory_toolkit.aio.processors import (
    AsyncDurableFunctionProcessor,
    AsyncInProcessProcessor,
    ProcessThreadResult,
)


def _build_client(processor=None) -> AsyncCosmosMemoryClient:
    client = AsyncCosmosMemoryClient(use_default_credential=False, processor=processor)
    client._container_client = MagicMock()  # truthy → _require_cosmos passes
    return client


# ---------------------------------------------------------------------------
# (a) AsyncInProcessProcessor end-to-end with mocked Cosmos
# ---------------------------------------------------------------------------


class TestAsyncInProcessProcessNowEndToEnd:
    @pytest.mark.asyncio
    async def test_process_now_invokes_pipeline_with_correct_args(self):
        pipeline = MagicMock()
        pipeline.generate_thread_summary.return_value = {
            "id": "summary-1",
            "type": "summary",
            "content": "Conversation about Paris.",
        }
        pipeline.extract_memories.return_value = {
            "facts_count": 1,
            "procedural_count": 0,
            "episodic_count": 0,
            "updated_count": 0,
        }
        pipeline.reconcile_memories.return_value = {"kept": 0, "merged": 0, "contradicted": 0}

        processor = AsyncInProcessProcessor(pipeline=pipeline)
        client = _build_client(processor=processor)

        fake_turns = [
            {"id": "t1", "role": "user", "content": "Tell me about Paris."},
            {"id": "t2", "role": "agent", "content": "It's the capital of France."},
        ]
        client.get_thread = AsyncMock(return_value=fake_turns)

        result = await client.process_now(user_id="u-paris", thread_id="thread-paris")

        assert isinstance(result, ProcessThreadResult)
        assert result.thread_summary == {
            "id": "summary-1",
            "type": "summary",
            "content": "Conversation about Paris.",
        }
        assert result.extracted_counts == {
            "facts_count": 1,
            "procedural_count": 0,
            "episodic_count": 0,
            "updated_count": 0,
        }
        client.get_thread.assert_awaited_once_with(thread_id="thread-paris", user_id="u-paris", memory_type="turn")
        pipeline.generate_thread_summary.assert_called_once_with("u-paris", "thread-paris")
        pipeline.extract_memories.assert_called_once_with("u-paris", "thread-paris")
        pipeline.reconcile_memories.assert_called_once_with("u-paris", 50)


# ---------------------------------------------------------------------------
# (b) AsyncDurableFunctionProcessor end-to-end with mocked Cosmos
# ---------------------------------------------------------------------------


class TestAsyncDurableProcessNowEndToEnd:
    @pytest.mark.asyncio
    async def test_process_now_is_a_noop(self, caplog):
        client = _build_client(processor=AsyncDurableFunctionProcessor())
        pipeline = MagicMock()
        client._pipeline = pipeline
        client.get_thread = AsyncMock(return_value=[{"id": "t1", "role": "user"}])

        import logging

        with caplog.at_level(logging.DEBUG, logger="agent_memory_toolkit.aio.processors.durable"):
            result = await client.process_now(user_id="u-1", thread_id="th-1")

        assert isinstance(result, ProcessThreadResult)
        assert result.thread_summary is None
        assert result.extracted_counts == {}
        assert result.reconciled_count == 0

        pipeline.generate_thread_summary.assert_not_called()
        pipeline.extract_memories.assert_not_called()
        pipeline.reconcile_memories.assert_not_called()

        debug_msgs = [r.message for r in caplog.records if r.levelno == logging.DEBUG]
        assert any("process_thread no-op" in m for m in debug_msgs)


# ---------------------------------------------------------------------------
# (c) process_now_and_wait polling for AsyncDurableFunctionProcessor
# ---------------------------------------------------------------------------


class TestAsyncDurableProcessNowAndWaitPolling:
    @pytest.mark.asyncio
    async def test_returns_true_when_summary_appears_after_polling(self, monkeypatch):
        client = _build_client(processor=AsyncDurableFunctionProcessor())
        client.get_thread = AsyncMock(return_value=[])

        client.get_memories = AsyncMock(
            side_effect=[
                [],
                [],
                [{"id": "summary-1", "memory_type": "summary", "content": "..."}],
            ]
        )

        async def _no_sleep(*_a, **_k):
            return None

        monkeypatch.setattr("asyncio.sleep", _no_sleep)

        ok = await client.process_now_and_wait(user_id="u-poll", thread_id="th-poll", timeout=10.0)

        assert ok is True
        assert client.get_memories.await_count == 3
        for call in client.get_memories.await_args_list:
            kwargs = call.kwargs
            assert kwargs["user_id"] == "u-poll"
            assert kwargs["thread_id"] == "th-poll"
            assert kwargs["memory_type"] == "summary"


# ---------------------------------------------------------------------------
# (d) process_now_and_wait timeout for AsyncDurableFunctionProcessor
# ---------------------------------------------------------------------------


class TestAsyncDurableProcessNowAndWaitTimeout:
    @pytest.mark.asyncio
    async def test_returns_false_after_timeout(self, monkeypatch):
        client = _build_client(processor=AsyncDurableFunctionProcessor())
        client.get_thread = AsyncMock(return_value=[])
        client.get_memories = AsyncMock(return_value=[])

        async def _no_sleep(*_a, **_k):
            return None

        monkeypatch.setattr("asyncio.sleep", _no_sleep)

        # Tiny timeout — loop.time() advances normally, so this expires fast.
        ok = await client.process_now_and_wait(user_id="u-to", thread_id="th-to", timeout=0.001)

        assert ok is False

    @pytest.mark.asyncio
    async def test_timeout_swallows_search_errors(self, monkeypatch):
        client = _build_client(processor=AsyncDurableFunctionProcessor())
        client.get_thread = AsyncMock(return_value=[])
        client.get_memories = AsyncMock(side_effect=RuntimeError("transient"))

        async def _no_sleep(*_a, **_k):
            return None

        monkeypatch.setattr("asyncio.sleep", _no_sleep)

        ok = await client.process_now_and_wait(user_id="u-err", thread_id="th-err", timeout=0.001)

        assert ok is False
