"""Tests for the AsyncCosmosMemoryClient push_to_cosmos auto-trigger.

The async client schedules ``_maybe_auto_trigger`` as a background
``asyncio.Task`` instead of awaiting it inline, so the user's write call
returns as soon as the Cosmos upserts complete.
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from azure.cosmos.exceptions import CosmosResourceNotFoundError

from azure.cosmos.agent_memory import _counters
from azure.cosmos.agent_memory.aio.cosmos_memory_client import AsyncCosmosMemoryClient
from azure.cosmos.agent_memory.aio.processors import AsyncInProcessProcessor


class _AsyncFakeCounterContainer:
    """Async in-memory counter container exercising the REAL increment /
    watermark-read / watermark-advance helpers end-to-end (no constant mocks)."""

    def __init__(self) -> None:
        self.store: dict[str, dict] = {}
        self._etag = 0

    async def read_item(self, *, item, partition_key):
        if item not in self.store:
            raise CosmosResourceNotFoundError(message="404")
        return dict(self.store[item])

    async def create_item(self, *, body):
        self._etag += 1
        body = dict(body)
        body["_etag"] = f"e{self._etag}"
        self.store[body["id"]] = body
        return dict(body)

    async def upsert_item(self, *, body, **_kwargs):
        self._etag += 1
        body = dict(body)
        body["_etag"] = f"e{self._etag}"
        self.store[body["id"]] = body
        return dict(body)

    async def patch_item(self, *, item, partition_key, patch_operations):
        doc = self.store.setdefault(item, {"id": item})
        for op in patch_operations:
            doc[op["path"].lstrip("/")] = op["value"]
        return dict(doc)



class TestAsyncAutoTriggerNonBlocking:
    @pytest.mark.asyncio
    async def test_push_to_cosmos_does_not_await_auto_trigger(self, monkeypatch):
        monkeypatch.setenv("FACT_EXTRACTION_EVERY_N", "1")
        monkeypatch.setenv("THREAD_SUMMARY_EVERY_N", "0")
        monkeypatch.setenv("USER_SUMMARY_EVERY_N", "0")

        processor = AsyncInProcessProcessor(pipeline=MagicMock())

        async def slow_trigger(user_id, thread_id, recent_k=None):
            # If push_to_cosmos awaited the trigger inline, the test would
            # block here for half a second before returning.
            await asyncio.sleep(0.5)
            return {}

        processor.process_extract_memories = MagicMock(side_effect=slow_trigger)

        client = AsyncCosmosMemoryClient(use_default_credential=False, processor=processor)

        async def fake_upsert(body):
            return body

        client._memories_container_client = MagicMock()
        client._memories_container_client.upsert_item = MagicMock(side_effect=fake_upsert)
        client._turns_container_client = client._memories_container_client
        client._summaries_container_client = client._memories_container_client
        client._counter_container_client = MagicMock()

        with patch(
            "azure.cosmos.agent_memory._counters.increment_counter_async",
            return_value=(0, 1),
        ):
            client.add_local(user_id="u1", role="user", thread_id="t1", content="hi")

            loop = asyncio.get_running_loop()
            t0 = loop.time()
            await client.push_to_cosmos()
            elapsed = loop.time() - t0

            # If push had awaited the slow trigger we'd see >= 0.5s here.
            assert elapsed < 0.4, f"push_to_cosmos awaited trigger inline (elapsed={elapsed:.3f}s)"

            # Drain the background task so pytest doesn't warn about a
            # destroyed-but-pending task at teardown.
            await asyncio.gather(*list(client._background_tasks), return_exceptions=True)


class TestAsyncExtractRecentK:
    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("n_facts", "batch_count", "counter_result", "expected_recent_k"),
        [
            (1, 1, (0, 1), 1),
            (1, 3, (0, 3), 3),
            (5, 1, (4, 5), 5),
        ],
    )
    async def test_extract_recent_k_uses_max_threshold_and_batch_count(
        self,
        monkeypatch,
        n_facts,
        batch_count,
        counter_result,
        expected_recent_k,
    ):
        monkeypatch.setenv("FACT_EXTRACTION_EVERY_N", str(n_facts))
        monkeypatch.setenv("THREAD_SUMMARY_EVERY_N", "0")
        monkeypatch.setenv("USER_SUMMARY_EVERY_N", "0")

        processor = AsyncInProcessProcessor(pipeline=MagicMock())
        processor.process_extract_memories = MagicMock(return_value={})

        client = AsyncCosmosMemoryClient(use_default_credential=False, processor=processor)

        async def fake_upsert(body):
            return body

        client._memories_container_client = MagicMock()
        client._memories_container_client.upsert_item = MagicMock(side_effect=fake_upsert)
        client._turns_container_client = client._memories_container_client
        client._summaries_container_client = client._memories_container_client
        client._counter_container_client = MagicMock()

        with patch(
            "azure.cosmos.agent_memory._counters.increment_counter_async",
            return_value=counter_result,
        ):
            for i in range(batch_count):
                client.add_local(user_id="u1", role="user", thread_id="t1", content=f"hi {i}")
            await client.push_to_cosmos()
            await asyncio.gather(*list(client._background_tasks), return_exceptions=True)

        processor.process_extract_memories.assert_called_once_with(
            user_id="u1",
            thread_id="t1",
            recent_k=expected_recent_k,
        )

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("counter_result", "watermark", "expected_recent_k"),
        [
            ((5, 10), 5, 5),       # backlog = new - watermark
            ((0, 1), 1, 1),        # new == watermark -> floored to 1
            ((98, 100), 0, 100),   # large backlog is NOT capped
            ((20, 30), None, 30),  # BOOTSTRAP: no watermark -> base=0 -> recent_k = new_count
        ],
    )
    async def test_extract_recent_k_uses_watermark(
        self, monkeypatch, counter_result, watermark, expected_recent_k
    ):
        monkeypatch.setenv("FACT_EXTRACTION_EVERY_N", "1")
        monkeypatch.setenv("THREAD_SUMMARY_EVERY_N", "0")
        monkeypatch.setenv("USER_SUMMARY_EVERY_N", "0")

        processor = AsyncInProcessProcessor(pipeline=MagicMock())
        processor.process_extract_memories = MagicMock(return_value={})
        client = AsyncCosmosMemoryClient(use_default_credential=False, processor=processor)
        client._memories_container_client = MagicMock()
        client._memories_container_client.upsert_item = AsyncMock(side_effect=lambda body: body)
        client._turns_container_client = client._memories_container_client
        client._summaries_container_client = client._memories_container_client
        client._counter_container_client = MagicMock()

        with patch(
            "azure.cosmos.agent_memory._counters.increment_counter_async",
            return_value=counter_result,
        ), patch(
            "azure.cosmos.agent_memory._counters.read_extract_watermark_async",
            new=AsyncMock(return_value=watermark),
        ), patch(
            "azure.cosmos.agent_memory._counters.advance_extract_watermark_async",
            new=AsyncMock(),
        ) as advance:
            client.add_local(user_id="u1", role="user", thread_id="t1", content="hi")
            await client.push_to_cosmos()
            await asyncio.gather(*list(client._background_tasks), return_exceptions=True)

        processor.process_extract_memories.assert_called_once_with(
            user_id="u1", thread_id="t1", recent_k=expected_recent_k
        )
        advance.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_watermark_not_advanced_when_extract_fails(self, monkeypatch):
        monkeypatch.setenv("FACT_EXTRACTION_EVERY_N", "1")
        monkeypatch.setenv("THREAD_SUMMARY_EVERY_N", "0")
        monkeypatch.setenv("USER_SUMMARY_EVERY_N", "0")

        processor = AsyncInProcessProcessor(pipeline=MagicMock())
        processor.process_extract_memories = MagicMock(side_effect=RuntimeError("llm down"))
        client = AsyncCosmosMemoryClient(use_default_credential=False, processor=processor)
        client._memories_container_client = MagicMock()
        client._memories_container_client.upsert_item = AsyncMock(side_effect=lambda body: body)
        client._turns_container_client = client._memories_container_client
        client._summaries_container_client = client._memories_container_client
        client._counter_container_client = MagicMock()

        with patch(
            "azure.cosmos.agent_memory._counters.increment_counter_async",
            return_value=(0, 1),
        ), patch(
            "azure.cosmos.agent_memory._counters.read_extract_watermark_async",
            new=AsyncMock(return_value=None),
        ), patch(
            "azure.cosmos.agent_memory._counters.advance_extract_watermark_async",
            new=AsyncMock(),
        ) as advance, patch(
            "azure.cosmos.agent_memory._counters.stamp_failure_async",
            new=AsyncMock(),
        ) as stamp:
            client.add_local(user_id="u1", role="user", thread_id="t1", content="hi")
            await client.push_to_cosmos()
            await asyncio.gather(*list(client._background_tasks), return_exceptions=True)

        advance.assert_not_awaited()
        stamp.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_watermark_round_trip_fail_then_succeed_no_strand(self, monkeypatch):
        """Async stateful round-trip against a REAL in-memory counter: first
        extract fails, second succeeds and must cover EVERY turn so far (20),
        not just its own batch (10) — the bootstrap strand regression."""
        monkeypatch.setenv("FACT_EXTRACTION_EVERY_N", "1")
        monkeypatch.setenv("THREAD_SUMMARY_EVERY_N", "0")
        monkeypatch.setenv("USER_SUMMARY_EVERY_N", "0")

        counter = _AsyncFakeCounterContainer()
        recorded: list[int] = []

        def extract(*, user_id, thread_id, recent_k):
            recorded.append(recent_k)
            if len(recorded) == 1:
                raise RuntimeError("transient LLM outage")
            return {}

        processor = AsyncInProcessProcessor(pipeline=MagicMock())
        processor.process_extract_memories = MagicMock(side_effect=extract)
        client = AsyncCosmosMemoryClient(use_default_credential=False, processor=processor)
        client._memories_container_client = MagicMock()
        client._memories_container_client.upsert_item = AsyncMock(side_effect=lambda body: body)
        client._turns_container_client = client._memories_container_client
        client._summaries_container_client = client._memories_container_client
        client._counter_container_client = counter

        for i in range(10):
            client.add_local(user_id="u1", role="user", thread_id="t1", content=f"a{i}")
        await client.push_to_cosmos()
        await asyncio.gather(*list(client._background_tasks), return_exceptions=True)

        for i in range(10):
            client.add_local(user_id="u1", role="user", thread_id="t1", content=f"b{i}")
        await client.push_to_cosmos()
        await asyncio.gather(*list(client._background_tasks), return_exceptions=True)

        assert recorded == [10, 20]
        cid = _counters.thread_counter_id("u1", "t1")
        assert await _counters.read_extract_watermark_async(counter, cid, "u1", "t1") == 20

    @pytest.mark.asyncio
    async def test_reconcile_full_rebuild_on_persisted_counter_cadence(self, monkeypatch):
        """Async symmetry: in-process auto-trigger requests full_rebuild on the
        persisted-counter cadence (every 2 turns here), like the durable backend."""
        monkeypatch.setenv("FACT_EXTRACTION_EVERY_N", "1")
        monkeypatch.setenv("THREAD_SUMMARY_EVERY_N", "0")
        monkeypatch.setenv("USER_SUMMARY_EVERY_N", "0")
        monkeypatch.setenv("DEDUP_EVERY_N", "1")
        monkeypatch.setattr(
            "azure.cosmos.agent_memory.thresholds.get_dedup_full_recluster_every_n", lambda: 2
        )

        rebuilds: list[bool] = []
        processor = AsyncInProcessProcessor(pipeline=MagicMock())
        processor.process_extract_memories = MagicMock(return_value={})
        processor.synthesize_procedural = MagicMock(return_value=None)
        processor.process_reconcile = MagicMock(
            side_effect=lambda *, user_id, full_rebuild=False: rebuilds.append(full_rebuild)
        )
        client = AsyncCosmosMemoryClient(use_default_credential=False, processor=processor)
        client._memories_container_client = MagicMock()
        client._memories_container_client.upsert_item = AsyncMock(side_effect=lambda body: body)
        client._turns_container_client = client._memories_container_client
        client._summaries_container_client = client._memories_container_client
        client._counter_container_client = MagicMock()

        with patch(
            "azure.cosmos.agent_memory._counters.increment_counter_async",
            new=AsyncMock(side_effect=[(0, 1), (1, 2)]),
        ), patch(
            "azure.cosmos.agent_memory._counters.read_extract_watermark_async",
            new=AsyncMock(return_value=None),
        ), patch(
            "azure.cosmos.agent_memory._counters.advance_extract_watermark_async",
            new=AsyncMock(),
        ):
            client.add_local(user_id="u1", role="user", thread_id="t1", content="a")
            await client.push_to_cosmos()
            await asyncio.gather(*list(client._background_tasks), return_exceptions=True)
            client.add_local(user_id="u1", role="user", thread_id="t1", content="b")
            await client.push_to_cosmos()
            await asyncio.gather(*list(client._background_tasks), return_exceptions=True)

        assert rebuilds == [False, True]


class TestPushToCosmosUnflushedDelta:
    """``push_to_cosmos`` must use the unflushed-add delta, not a recount
    of ``local_memory``, so callers that retain the buffer don't re-fire
    extract/dedup/summary on already-processed turns."""

    @pytest.mark.asyncio
    async def test_repeat_push_does_not_re_increment(self, monkeypatch):
        monkeypatch.setenv("FACT_EXTRACTION_EVERY_N", "1")

        client = AsyncCosmosMemoryClient(use_default_credential=False)

        async def fake_upsert(body):
            return body

        client._memories_container_client = MagicMock()
        client._memories_container_client.upsert_item = MagicMock(side_effect=fake_upsert)
        client._turns_container_client = client._memories_container_client
        client._summaries_container_client = client._memories_container_client
        client._counter_container_client = MagicMock()

        client.add_local(user_id="u1", role="user", thread_id="t1", content="a")
        client.add_local(user_id="u1", role="user", thread_id="t1", content="b")

        captured: list[dict] = []

        async def capture(turn_counts):
            captured.append(dict(turn_counts))

        with patch.object(client, "_maybe_auto_trigger", side_effect=capture):
            await client.push_to_cosmos()
            await asyncio.gather(*list(client._background_tasks), return_exceptions=True)

            # First push: trigger sees the 2 unflushed turns.
            assert captured == [{("u1", "t1"): 2}]
            # local_memory is intentionally retained.
            assert len(client.local_memory) == 2

            captured.clear()
            # Second push WITHOUT new add_local. The unflushed delta is now
            # empty so the trigger must NOT fire (or, if it fires, must see
            # an empty dict and short-circuit).
            await client.push_to_cosmos()
            await asyncio.gather(*list(client._background_tasks), return_exceptions=True)

            # No new background task with non-empty turn_counts.
            for tc in captured:
                assert tc == {}, f"Re-pushed buffer wrongly fired trigger: {tc}"

    @pytest.mark.asyncio
    async def test_only_new_adds_count_after_partial_push(self, monkeypatch):
        monkeypatch.setenv("FACT_EXTRACTION_EVERY_N", "1")

        client = AsyncCosmosMemoryClient(use_default_credential=False)

        async def fake_upsert(body):
            return body

        client._memories_container_client = MagicMock()
        client._memories_container_client.upsert_item = MagicMock(side_effect=fake_upsert)
        client._turns_container_client = client._memories_container_client
        client._summaries_container_client = client._memories_container_client
        client._counter_container_client = MagicMock()

        client.add_local(user_id="u1", role="user", thread_id="t1", content="a")

        captured: list[dict] = []

        async def capture(turn_counts):
            captured.append(dict(turn_counts))

        with patch.object(client, "_maybe_auto_trigger", side_effect=capture):
            await client.push_to_cosmos()
            await asyncio.gather(*list(client._background_tasks), return_exceptions=True)
            assert captured == [{("u1", "t1"): 1}]

            captured.clear()
            # Add ONE more turn. local_memory now has 2 entries but the
            # delta passed to the trigger must be 1.
            client.add_local(user_id="u1", role="user", thread_id="t1", content="b")
            await client.push_to_cosmos()
            await asyncio.gather(*list(client._background_tasks), return_exceptions=True)
            assert captured == [{("u1", "t1"): 1}]
