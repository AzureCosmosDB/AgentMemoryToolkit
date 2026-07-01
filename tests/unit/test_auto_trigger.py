"""Tests for the InProcess push_to_cosmos auto-trigger.

Per-turn fact extraction is the new default (FACT_EXTRACTION_EVERY_N=1):
each turn flushed to Cosmos should immediately fire `process_thread` for
the in-process backend. The durable backend must remain a no-op (the
change-feed function app handles it).
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from azure.cosmos.exceptions import CosmosResourceNotFoundError

from azure.cosmos.agent_memory import _counters
from azure.cosmos.agent_memory.cosmos_memory_client import CosmosMemoryClient
from azure.cosmos.agent_memory.processors import DurableFunctionProcessor, InProcessProcessor


class _FakeCounterContainer:
    """Minimal in-memory stand-in for the Cosmos counter container.

    Exercises the REAL increment / watermark-read / watermark-advance helpers
    end-to-end (no mocking of counter math), so a watermark/recent_k regression
    can't slip through behind constant mocks.
    """

    def __init__(self) -> None:
        self.store: dict[str, dict] = {}
        self._etag = 0

    def read_item(self, *, item, partition_key):
        if item not in self.store:
            raise CosmosResourceNotFoundError(message="404")
        return dict(self.store[item])

    def create_item(self, *, body):
        self._etag += 1
        body = dict(body)
        body["_etag"] = f"e{self._etag}"
        self.store[body["id"]] = body
        return dict(body)

    def upsert_item(self, *, body, **_kwargs):
        self._etag += 1
        body = dict(body)
        body["_etag"] = f"e{self._etag}"
        self.store[body["id"]] = body
        return dict(body)

    def patch_item(self, *, item, partition_key, patch_operations):
        doc = self.store.setdefault(item, {"id": item})
        for op in patch_operations:
            doc[op["path"].lstrip("/")] = op["value"]
        return dict(doc)


def _connected(processor=None) -> CosmosMemoryClient:
    client = CosmosMemoryClient(use_default_credential=False, processor=processor)
    client._memories_container_client = MagicMock()
    client._turns_container_client = client._memories_container_client
    client._summaries_container_client = client._memories_container_client
    return client


def test_push_to_cosmos_fires_inprocess_trigger_per_turn(monkeypatch):
    monkeypatch.setenv("FACT_EXTRACTION_EVERY_N", "1")
    monkeypatch.setenv("THREAD_SUMMARY_EVERY_N", "0")

    client = _connected(processor=InProcessProcessor(pipeline=MagicMock()))
    counter_container = MagicMock()
    client._counter_container_client = counter_container

    pipeline = MagicMock()
    pipeline.generate_thread_summary.return_value = None
    pipeline.extract_memories.return_value = {"fact_count": 1}
    pipeline.reconcile_memories.return_value = {}
    client._processor._pipeline = pipeline

    with patch(
        "azure.cosmos.agent_memory._counters.increment_counter_sync",
        return_value=(0, 1),
    ):
        client.add_local(user_id="u1", role="user", thread_id="t1", content="hi")
        client.push_to_cosmos()

    pipeline.extract_memories.assert_called_once_with("u1", "t1", recent_k=1)


def test_push_to_cosmos_durable_does_not_fire_trigger(monkeypatch):
    monkeypatch.setenv("FACT_EXTRACTION_EVERY_N", "1")
    client = _connected(processor=DurableFunctionProcessor())
    client._counter_container_client = MagicMock()

    with patch(
        "azure.cosmos.agent_memory._counters.increment_counter_sync",
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
        "azure.cosmos.agent_memory._counters.increment_counter_sync",
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
        "azure.cosmos.agent_memory._counters.increment_counter_sync",
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
            "azure.cosmos.agent_memory._counters.increment_counter_sync",
            return_value=(0, 1),  # crosses 1 only
        ):
            client.add_local(user_id="u1", role="user", thread_id="t1", content="hi")
            client.push_to_cosmos()

        processor.process_extract_memories.assert_called_once_with(user_id="u1", thread_id="t1", recent_k=1)
        processor.process_thread_summary.assert_not_called()
        processor.process_user_summary.assert_not_called()

    @pytest.mark.parametrize(
        ("n_facts", "batch_count", "counter_result", "watermark", "expected_recent_k"),
        [
            (1, 1, (0, 1), None, 1),
            (1, 3, (0, 3), None, 3),
            (5, 1, (4, 5), None, 5),
            (1, 1, (5, 10), 5, 5),
            # Large backlog is NOT capped: recent_k spans every turn since the
            # watermark (newest-recent_k slice covers exactly those), so the
            # watermark can advance to new_count with no stranded turns.
            (1, 1, (98, 100), 0, 100),
            # BOOTSTRAP regression: no watermark yet but the counter is already
            # ahead of this batch (earlier extracts failed). base=0 so recent_k =
            # new_count (30) covers ALL turns — the old fallback max(n_facts,
            # batch_count) would return 2 and strand turns 1-28 forever.
            (1, 2, (20, 30), None, 30),
        ],
    )
    def test_extract_recent_k_uses_watermark_then_falls_back(
        self,
        monkeypatch,
        n_facts,
        batch_count,
        counter_result,
        watermark,
        expected_recent_k,
    ):
        monkeypatch.setenv("FACT_EXTRACTION_EVERY_N", str(n_facts))
        monkeypatch.setenv("THREAD_SUMMARY_EVERY_N", "0")
        monkeypatch.setenv("USER_SUMMARY_EVERY_N", "0")

        processor = InProcessProcessor(pipeline=MagicMock())
        processor.process_extract_memories = MagicMock(return_value={})

        client = _connected(processor=processor)
        client._counter_container_client = MagicMock()

        with (
            patch(
                "azure.cosmos.agent_memory._counters.increment_counter_sync",
                return_value=counter_result,
            ),
            patch(
                "azure.cosmos.agent_memory._counters.read_extract_watermark_sync",
                return_value=watermark,
            ),
            patch(
                "azure.cosmos.agent_memory._counters.advance_extract_watermark_sync",
            ) as advance,
        ):
            for i in range(batch_count):
                client.add_local(user_id="u1", role="user", thread_id="t1", content=f"hi {i}")
            client.push_to_cosmos()

        processor.process_extract_memories.assert_called_once_with(
            user_id="u1",
            thread_id="t1",
            recent_k=expected_recent_k,
        )
        advance.assert_called_once()

    def test_watermark_round_trip_fail_then_succeed_no_strand(self, monkeypatch):
        """End-to-end round-trip against a REAL in-memory counter (no constant
        mocks): a thread's first extract fails, the second succeeds, and the
        second must cover EVERY turn so far — not just its own batch — so turns
        from the failed batch are never stranded. This is the bootstrap case the
        constant-mock tests could not catch."""
        monkeypatch.setenv("FACT_EXTRACTION_EVERY_N", "1")
        monkeypatch.setenv("THREAD_SUMMARY_EVERY_N", "0")
        monkeypatch.setenv("USER_SUMMARY_EVERY_N", "0")

        counter = _FakeCounterContainer()
        recorded_recent_k: list[int] = []

        def extract(*, user_id, thread_id, recent_k):
            recorded_recent_k.append(recent_k)
            if len(recorded_recent_k) == 1:
                raise RuntimeError("transient LLM outage")  # first extract fails
            return {}

        processor = InProcessProcessor(pipeline=MagicMock())
        processor.process_extract_memories = MagicMock(side_effect=extract)
        client = _connected(processor=processor)
        client._counter_container_client = counter

        # Batch 1: 10 turns -> counter 0->10, extract FAILS (watermark not advanced).
        for i in range(10):
            client.add_local(user_id="u1", role="user", thread_id="t1", content=f"a{i}")
        client.push_to_cosmos()

        # Batch 2: 10 turns -> counter 10->20, extract SUCCEEDS.
        for i in range(10):
            client.add_local(user_id="u1", role="user", thread_id="t1", content=f"b{i}")
        client.push_to_cosmos()

        # First fired with 10 (all turns so far); second with 20 (ALL turns, since
        # the failed first extract left the watermark unset) — NOT 10.
        assert recorded_recent_k == [10, 20]
        # Watermark now seeded at the full count after the successful extract.
        cid = _counters.thread_counter_id("u1", "t1")
        assert _counters.read_extract_watermark_sync(counter, cid, "u1", "t1") == 20

    def test_watermark_not_advanced_when_extract_fails(self, monkeypatch):
        """advance-on-success: a failing extract must NOT move the watermark, so
        the skipped turns are retried next sweep; failure is stamped instead."""
        monkeypatch.setenv("FACT_EXTRACTION_EVERY_N", "1")
        monkeypatch.setenv("THREAD_SUMMARY_EVERY_N", "0")
        monkeypatch.setenv("USER_SUMMARY_EVERY_N", "0")

        processor = InProcessProcessor(pipeline=MagicMock())
        processor.process_extract_memories = MagicMock(side_effect=RuntimeError("llm down"))

        client = _connected(processor=processor)
        client._counter_container_client = MagicMock()

        with (
            patch(
                "azure.cosmos.agent_memory._counters.increment_counter_sync",
                return_value=(0, 1),
            ),
            patch(
                "azure.cosmos.agent_memory._counters.read_extract_watermark_sync",
                return_value=None,
            ),
            patch(
                "azure.cosmos.agent_memory._counters.advance_extract_watermark_sync",
            ) as advance,
            patch(
                "azure.cosmos.agent_memory._counters.stamp_failure_sync",
            ) as stamp,
        ):
            client.add_local(user_id="u1", role="user", thread_id="t1", content="hi")
            client.push_to_cosmos()

        advance.assert_not_called()
        stamp.assert_called_once()

    def test_reconcile_full_rebuild_on_persisted_counter_cadence(self, monkeypatch):
        """Symmetry with the durable backend: the in-process auto-trigger requests
        a full-pool reconcile (full_rebuild=True) on a PERSISTED-counter cadence —
        every DEDUP_FULL_RECLUSTER_EVERY_N-th reconcile — not via an in-memory
        per-instance sweep counter. Here that's every 2 turns."""
        monkeypatch.setenv("FACT_EXTRACTION_EVERY_N", "1")
        monkeypatch.setenv("THREAD_SUMMARY_EVERY_N", "0")
        monkeypatch.setenv("USER_SUMMARY_EVERY_N", "0")
        monkeypatch.setenv("DEDUP_EVERY_N", "1")
        monkeypatch.setattr("azure.cosmos.agent_memory.thresholds.get_dedup_full_recluster_every_n", lambda: 2)

        rebuilds: list[bool] = []
        processor = InProcessProcessor(pipeline=MagicMock())
        processor.process_extract_memories = MagicMock(return_value={})
        processor.synthesize_procedural = MagicMock(return_value=None)
        processor.process_reconcile = MagicMock(
            side_effect=lambda *, user_id, full_rebuild=False: rebuilds.append(full_rebuild)
        )

        client = _connected(processor=processor)
        client._counter_container_client = MagicMock()

        with (
            patch(
                "azure.cosmos.agent_memory._counters.increment_counter_sync",
                side_effect=[(0, 1), (1, 2)],
            ),
            patch(
                "azure.cosmos.agent_memory._counters.read_extract_watermark_sync",
                return_value=None,
            ),
            patch(
                "azure.cosmos.agent_memory._counters.advance_extract_watermark_sync",
            ),
        ):
            client.add_local(user_id="u1", role="user", thread_id="t1", content="a")
            client.push_to_cosmos()  # counter 0->1: reconcile, full crosses 2? no
            client.add_local(user_id="u1", role="user", thread_id="t1", content="b")
            client.push_to_cosmos()  # counter 1->2: full backstop threshold (2) crossed

        assert rebuilds == [False, True]

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
            "azure.cosmos.agent_memory._counters.increment_counter_sync",
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
            "azure.cosmos.agent_memory._counters.increment_counter_sync",
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
            "azure.cosmos.agent_memory._counters.increment_counter_sync",
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
            "azure.cosmos.agent_memory._counters.increment_counter_sync",
            return_value=(0, 1),
        ):
            client.add_local(user_id="u1", role="user", thread_id="t1", content="hi")
            client.push_to_cosmos()

        processor.process_extract_memories.assert_called_once()
