"""Integration-style tests for the processor protocol surface.

Exercises ``CosmosMemoryClient.process_now`` / ``process_now_and_wait`` end-to-end with
a fully mocked Cosmos container — no live Azure calls — to validate that
the SDK wires the active :class:`MemoryProcessor` correctly through the
public API.
"""

from __future__ import annotations

from unittest.mock import MagicMock

from agent_memory_toolkit.cosmos_memory_client import CosmosMemoryClient
from agent_memory_toolkit.processors import (
    DurableFunctionProcessor,
    InProcessProcessor,
    ProcessThreadResult,
)


def _build_client(processor=None) -> CosmosMemoryClient:
    """Build a CosmosMemoryClient with a fake container client attached."""
    client = CosmosMemoryClient(use_default_credential=False, processor=processor)
    client._container_client = MagicMock()
    return client


# ---------------------------------------------------------------------------
# (a) InProcessProcessor end-to-end with mocked Cosmos
# ---------------------------------------------------------------------------


class TestInProcessProcessNowEndToEnd:
    def test_process_now_invokes_pipeline_with_correct_args(self):
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

        processor = InProcessProcessor(pipeline=pipeline)
        client = _build_client(processor=processor)

        fake_turns = [
            {"id": "t1", "role": "user", "content": "Tell me about Paris."},
            {"id": "t2", "role": "agent", "content": "It's the capital of France."},
        ]
        client.get_thread = MagicMock(return_value=fake_turns)

        result = client.process_now(user_id="u-paris", thread_id="thread-paris")

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
        client.get_thread.assert_called_once_with(thread_id="thread-paris", user_id="u-paris", memory_type="turn")
        pipeline.generate_thread_summary.assert_called_once_with("u-paris", "thread-paris")
        pipeline.extract_memories.assert_called_once_with("u-paris", "thread-paris")
        pipeline.reconcile_memories.assert_called_once_with("u-paris", 50)


# ---------------------------------------------------------------------------
# (b) DurableFunctionProcessor end-to-end with mocked Cosmos
# ---------------------------------------------------------------------------


class TestDurableProcessNowEndToEnd:
    def test_process_now_is_a_noop(self, caplog):
        client = _build_client(processor=DurableFunctionProcessor())

        # Replace the pipeline so we can prove no methods were called.
        pipeline = MagicMock()
        client._pipeline = pipeline
        client.get_thread = MagicMock(return_value=[{"id": "t1", "role": "user"}])

        import logging

        with caplog.at_level(logging.DEBUG, logger="agent_memory_toolkit.processors.durable"):
            result = client.process_now(user_id="u-1", thread_id="th-1")

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
# (c) process_now_and_wait polling for DurableFunctionProcessor
# ---------------------------------------------------------------------------


class TestDurableProcessNowAndWaitPolling:
    def test_returns_true_when_summary_appears_after_polling(self, monkeypatch):
        client = _build_client(processor=DurableFunctionProcessor())
        client.get_thread = MagicMock(return_value=[])

        # First two polls return empty; third returns a summary doc.
        client.get_memories = MagicMock(
            side_effect=[
                [],
                [],
                [{"id": "summary-1", "memory_type": "summary", "content": "..."}],
            ]
        )

        # Make sleep a no-op so the test stays fast.
        monkeypatch.setattr("time.sleep", lambda *_args, **_kwargs: None)

        ok = client.process_now_and_wait(user_id="u-poll", thread_id="th-poll", timeout=10.0)

        assert ok is True
        assert client.get_memories.call_count == 3
        # Each poll should query the same (user_id, thread_id) for summaries.
        for call in client.get_memories.call_args_list:
            kwargs = call.kwargs
            assert kwargs["user_id"] == "u-poll"
            assert kwargs["thread_id"] == "th-poll"
            assert kwargs["memory_type"] == "summary"


# ---------------------------------------------------------------------------
# (d) process_now_and_wait timeout for DurableFunctionProcessor
# ---------------------------------------------------------------------------


class TestDurableProcessNowAndWaitTimeout:
    def test_returns_false_after_timeout(self, monkeypatch):
        client = _build_client(processor=DurableFunctionProcessor())
        client.get_thread = MagicMock(return_value=[])
        client.get_memories = MagicMock(return_value=[])

        # Simulate a fast clock — each call to monotonic advances by 1s,
        # so a timeout of 0.5s expires after the first iteration.
        ticks = iter([0.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0])

        def _fake_monotonic():
            try:
                return next(ticks)
            except StopIteration:
                return 100.0

        monkeypatch.setattr("time.monotonic", _fake_monotonic)
        monkeypatch.setattr("time.sleep", lambda *_a, **_k: None)

        ok = client.process_now_and_wait(user_id="u-to", thread_id="th-to", timeout=0.5)

        assert ok is False
        # get_memories was tried at least once before the deadline expired.
        assert client.get_memories.call_count >= 1

    def test_timeout_swallows_search_errors(self, monkeypatch):
        client = _build_client(processor=DurableFunctionProcessor())
        client.get_thread = MagicMock(return_value=[])
        client.get_memories = MagicMock(side_effect=RuntimeError("transient"))

        monkeypatch.setattr("time.sleep", lambda *_a, **_k: None)

        ok = client.process_now_and_wait(user_id="u-err", thread_id="th-err", timeout=0.01)

        assert ok is False
