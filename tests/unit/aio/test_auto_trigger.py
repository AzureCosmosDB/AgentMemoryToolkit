"""Tests for the AsyncCosmosMemoryClient push_to_cosmos auto-trigger.

The async client schedules ``_maybe_auto_trigger`` as a background
``asyncio.Task`` instead of awaiting it inline, so the user's write call
returns as soon as the Cosmos upserts complete.
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from azure.cosmos.agent_memory.aio.cosmos_memory_client import AsyncCosmosMemoryClient
from azure.cosmos.agent_memory.aio.processors import AsyncInProcessProcessor


class TestAsyncAutoTriggerNonBlocking:
    @pytest.mark.asyncio
    async def test_push_to_cosmos_does_not_await_auto_trigger(self, monkeypatch):
        monkeypatch.setenv("FACT_EXTRACTION_EVERY_N", "1")
        monkeypatch.setenv("THREAD_SUMMARY_EVERY_N", "0")
        monkeypatch.setenv("USER_SUMMARY_EVERY_N", "0")

        processor = AsyncInProcessProcessor(pipeline=MagicMock())

        async def slow_trigger(user_id, thread_id):
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


class TestCadenceThresholdsForwarding:
    """A ``cadence_thresholds`` mapping on the client is forwarded to the auto-trigger.

    This lets callers set per-turn cadence in-process instead of mutating ``os.environ``.
    """

    @pytest.mark.asyncio
    async def test_cadence_thresholds_forwarded(self):
        thresholds = {"FACT_EXTRACTION_EVERY_N": 3, "DEDUP_EVERY_N": 2}
        client = AsyncCosmosMemoryClient(use_default_credential=False, cadence_thresholds=thresholds)
        client._get_processor = MagicMock(return_value=MagicMock())
        client._get_counter_container = MagicMock(return_value=MagicMock())

        with patch(
            "azure.cosmos.agent_memory.aio.cosmos_memory_client.maybe_trigger_steps",
            new=AsyncMock(),
        ) as mock_trigger:
            await client._maybe_auto_trigger({("u1", "t1"): 1})

        mock_trigger.assert_awaited_once()
        assert mock_trigger.await_args.kwargs["thresholds"] == thresholds

    @pytest.mark.asyncio
    async def test_defaults_to_none_when_unset(self):
        client = AsyncCosmosMemoryClient(use_default_credential=False)
        client._get_processor = MagicMock(return_value=MagicMock())
        client._get_counter_container = MagicMock(return_value=MagicMock())

        with patch(
            "azure.cosmos.agent_memory.aio.cosmos_memory_client.maybe_trigger_steps",
            new=AsyncMock(),
        ) as mock_trigger:
            await client._maybe_auto_trigger({("u1", "t1"): 1})

        # None preserves the env-only behavior (the auto-trigger treats None as defaults).
        assert mock_trigger.await_args.kwargs["thresholds"] is None


class TestCadenceThresholdsNormalization:
    """The async client normalizes ``cadence_thresholds`` at construction time."""

    def test_defensive_copy_isolates_later_mutation(self):
        thresholds = {"FACT_EXTRACTION_EVERY_N": 3}
        client = AsyncCosmosMemoryClient(use_default_credential=False, cadence_thresholds=thresholds)
        thresholds["FACT_EXTRACTION_EVERY_N"] = 99
        assert client._cadence_thresholds == {"FACT_EXTRACTION_EVERY_N": 3}

    def test_string_values_are_coerced_to_int(self):
        client = AsyncCosmosMemoryClient(use_default_credential=False, cadence_thresholds={"DEDUP_EVERY_N": "5"})
        assert client._cadence_thresholds == {"DEDUP_EVERY_N": 5}

    def test_negative_value_rejected(self):
        with pytest.raises(ValueError):
            AsyncCosmosMemoryClient(use_default_credential=False, cadence_thresholds={"DEDUP_EVERY_N": -1})

    def test_non_int_value_rejected(self):
        with pytest.raises(ValueError):
            AsyncCosmosMemoryClient(use_default_credential=False, cadence_thresholds={"DEDUP_EVERY_N": "x"})

    def test_non_mapping_rejected(self):
        with pytest.raises(TypeError):
            AsyncCosmosMemoryClient(use_default_credential=False, cadence_thresholds=[("DEDUP_EVERY_N", 5)])
