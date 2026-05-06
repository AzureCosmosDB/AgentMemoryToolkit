"""Tests for the InProcess push_to_cosmos auto-trigger.

Per-turn fact extraction is the new default (FACT_EXTRACTION_EVERY_N=1):
each turn flushed to Cosmos should immediately fire `process_thread` for
the in-process backend. The durable backend must remain a no-op (the
change-feed function app handles it).
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from agent_memory_toolkit.cosmos_memory_client import CosmosMemoryClient
from agent_memory_toolkit.processors import DurableFunctionProcessor, InProcessProcessor


def _connected(processor=None) -> CosmosMemoryClient:
    client = CosmosMemoryClient(use_default_credential=False, processor=processor)
    client._container_client = MagicMock()
    return client


def test_push_to_cosmos_fires_inprocess_trigger_per_turn(monkeypatch):
    monkeypatch.setenv("FACT_EXTRACTION_EVERY_N", "1")
    monkeypatch.setenv("THREAD_SUMMARY_EVERY_N", "0")

    client = _connected(processor=InProcessProcessor(pipeline=MagicMock()))
    counter_container = MagicMock()
    client._counter_container_client = counter_container

    pipeline = MagicMock()
    pipeline.generate_thread_summary.return_value = None
    pipeline.extract_memories.return_value = {"facts_count": 1}
    pipeline.reconcile_memories.return_value = {}
    client._processor._pipeline = pipeline

    with patch(
        "agent_memory_toolkit._counters.increment_counter_sync",
        return_value=(0, 1),
    ):
        client.add_local(user_id="u1", role="user", thread_id="t1", content="hi")
        client.push_to_cosmos()

    pipeline.extract_memories.assert_called_once_with("u1", "t1")


def test_push_to_cosmos_durable_does_not_fire_trigger(monkeypatch):
    monkeypatch.setenv("FACT_EXTRACTION_EVERY_N", "1")
    client = _connected(processor=DurableFunctionProcessor())
    client._counter_container_client = MagicMock()

    with patch(
        "agent_memory_toolkit._counters.increment_counter_sync",
        return_value=(0, 1),
    ) as inc:
        client.add_local(user_id="u1", role="user", thread_id="t1", content="hi")
        client.push_to_cosmos()

    inc.assert_not_called()


def test_push_to_cosmos_skips_trigger_when_thresholds_zero(monkeypatch):
    monkeypatch.setenv("FACT_EXTRACTION_EVERY_N", "0")
    monkeypatch.setenv("THREAD_SUMMARY_EVERY_N", "0")
    monkeypatch.setenv("USER_SUMMARY_EVERY_N", "0")

    client = _connected(processor=InProcessProcessor(pipeline=MagicMock()))
    client._counter_container_client = MagicMock()

    with patch(
        "agent_memory_toolkit._counters.increment_counter_sync",
        return_value=(0, 1),
    ) as inc:
        client.add_local(user_id="u1", role="user", thread_id="t1", content="hi")
        client.push_to_cosmos()

    inc.assert_not_called()


def test_push_to_cosmos_swallows_trigger_failures(monkeypatch):
    """Auto-trigger errors must never propagate from push_to_cosmos."""
    monkeypatch.setenv("FACT_EXTRACTION_EVERY_N", "1")

    pipeline = MagicMock()
    pipeline.generate_thread_summary.side_effect = RuntimeError("boom")
    client = _connected(processor=InProcessProcessor(pipeline=pipeline))
    client._counter_container_client = MagicMock()

    with patch(
        "agent_memory_toolkit._counters.increment_counter_sync",
        return_value=(0, 1),
    ):
        client.add_local(user_id="u1", role="user", thread_id="t1", content="hi")
        client.push_to_cosmos()  # must not raise


def test_push_to_cosmos_skips_when_counter_container_unavailable(monkeypatch):
    monkeypatch.setenv("FACT_EXTRACTION_EVERY_N", "1")

    client = _connected(processor=InProcessProcessor(pipeline=MagicMock()))
    # Counter container handle stays None; lazy getter would normally try to
    # build one but will return None on failure.
    client._get_counter_container = MagicMock(return_value=None)

    pipeline = MagicMock()
    client._processor._pipeline = pipeline

    client.add_local(user_id="u1", role="user", thread_id="t1", content="hi")
    client.push_to_cosmos()

    pipeline.extract_memories.assert_not_called()


# ---------------------------------------------------------------------------
# Per-step trigger gating — each *_EVERY_N fires its own pipeline step
# independently. The InProcess backend mirrors the function-app
# split-orchestrator behavior so the two backends produce the same memory
# contents for the same chat history.
# ---------------------------------------------------------------------------


class TestPerStepAutoTrigger:
    def test_extract_fires_independently_of_summary(self, monkeypatch):
        """N_facts=1 alone fires extract; summary/user-summary stay quiet."""
        monkeypatch.setenv("FACT_EXTRACTION_EVERY_N", "1")
        monkeypatch.setenv("THREAD_SUMMARY_EVERY_N", "10")
        monkeypatch.setenv("USER_SUMMARY_EVERY_N", "20")

        processor = InProcessProcessor(pipeline=MagicMock())
        processor.process_extract_memories = MagicMock(return_value={})
        processor.process_thread_summary = MagicMock(return_value={})
        processor.process_user_summary = MagicMock()

        client = _connected(processor=processor)
        client._counter_container_client = MagicMock()

        with patch(
            "agent_memory_toolkit._counters.increment_counter_sync",
            return_value=(0, 1),  # crosses 1 only
        ):
            client.add_local(user_id="u1", role="user", thread_id="t1", content="hi")
            client.push_to_cosmos()

        processor.process_extract_memories.assert_called_once_with(user_id="u1", thread_id="t1")
        processor.process_thread_summary.assert_not_called()
        processor.process_user_summary.assert_not_called()

    def test_summary_fires_independently_when_threshold_crossed(self, monkeypatch):
        """N_summary=10 boundary fires summary; N_facts=0 prevents extract."""
        monkeypatch.setenv("FACT_EXTRACTION_EVERY_N", "0")
        monkeypatch.setenv("THREAD_SUMMARY_EVERY_N", "10")
        monkeypatch.setenv("USER_SUMMARY_EVERY_N", "0")

        processor = InProcessProcessor(pipeline=MagicMock())
        processor.process_extract_memories = MagicMock()
        processor.process_thread_summary = MagicMock(return_value={})

        client = _connected(processor=processor)
        client._counter_container_client = MagicMock()

        with patch(
            "agent_memory_toolkit._counters.increment_counter_sync",
            return_value=(9, 10),  # crosses 10 only
        ):
            client.add_local(user_id="u1", role="user", thread_id="t1", content="hi")
            client.push_to_cosmos()

        processor.process_thread_summary.assert_called_once_with(user_id="u1", thread_id="t1")
        processor.process_extract_memories.assert_not_called()

    def test_user_summary_fires_at_user_threshold(self, monkeypatch):
        """The user-scoped counter is incremented separately from the thread counter."""
        monkeypatch.setenv("FACT_EXTRACTION_EVERY_N", "0")
        monkeypatch.setenv("THREAD_SUMMARY_EVERY_N", "0")
        monkeypatch.setenv("USER_SUMMARY_EVERY_N", "2")

        processor = InProcessProcessor(pipeline=MagicMock())
        processor.process_user_summary = MagicMock()

        client = _connected(processor=processor)
        client._counter_container_client = MagicMock()

        # Thread counter: (0,1) then (1,2); user counter: (1,2) crosses 2.
        with patch(
            "agent_memory_toolkit._counters.increment_counter_sync",
            side_effect=[(0, 1), (1, 2), (1, 2)],
        ):
            client.add_local(user_id="u1", role="user", thread_id="t1", content="hi")
            client.add_local(user_id="u1", role="agent", thread_id="t1", content="ok")
            client.push_to_cosmos()

        processor.process_user_summary.assert_called_once_with(user_id="u1")


# ---------------------------------------------------------------------------
# Owner exclusivity — MEMORY_PROCESSOR_OWNER ensures only one of
# {SDK auto-trigger, FA change-feed processor} runs against a shared
# container, preventing double-extraction / double-dedup.
# ---------------------------------------------------------------------------


class TestProcessorOwner:
    def test_durable_owner_suppresses_sdk_trigger(self, monkeypatch):
        monkeypatch.setenv("FACT_EXTRACTION_EVERY_N", "1")
        monkeypatch.setenv("MEMORY_PROCESSOR_OWNER", "durable")

        processor = InProcessProcessor(pipeline=MagicMock())
        processor.process_extract_memories = MagicMock()

        client = _connected(processor=processor)
        client._counter_container_client = MagicMock()

        with patch(
            "agent_memory_toolkit._counters.increment_counter_sync",
            return_value=(0, 1),
        ) as inc:
            client.add_local(user_id="u1", role="user", thread_id="t1", content="hi")
            client.push_to_cosmos()

        inc.assert_not_called()
        processor.process_extract_memories.assert_not_called()

    def test_inprocess_owner_allows_sdk_trigger(self, monkeypatch):
        monkeypatch.setenv("FACT_EXTRACTION_EVERY_N", "1")
        monkeypatch.setenv("THREAD_SUMMARY_EVERY_N", "0")
        monkeypatch.setenv("USER_SUMMARY_EVERY_N", "0")
        monkeypatch.setenv("MEMORY_PROCESSOR_OWNER", "inprocess")

        processor = InProcessProcessor(pipeline=MagicMock())
        processor.process_extract_memories = MagicMock(return_value={})

        client = _connected(processor=processor)
        client._counter_container_client = MagicMock()

        with patch(
            "agent_memory_toolkit._counters.increment_counter_sync",
            return_value=(0, 1),
        ):
            client.add_local(user_id="u1", role="user", thread_id="t1", content="hi")
            client.push_to_cosmos()

        processor.process_extract_memories.assert_called_once()
