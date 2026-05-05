"""Tests for the AsyncCosmosMemoryClient push_to_cosmos auto-trigger.

The async client schedules ``_maybe_auto_trigger`` as a background
``asyncio.Task`` instead of awaiting it inline, so the user's write call
returns as soon as the Cosmos upserts complete (Round 4 fix #6).
"""

from __future__ import annotations

import asyncio
from unittest.mock import MagicMock, patch

import pytest

from agent_memory_toolkit.aio.cosmos_memory_client import AsyncCosmosMemoryClient
from agent_memory_toolkit.aio.processors import AsyncInProcessProcessor


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

        client._container_client = MagicMock()
        client._container_client.upsert_item = MagicMock(side_effect=fake_upsert)
        client._counter_container_client = MagicMock()

        with patch(
            "agent_memory_toolkit._counters.increment_counter_async",
            return_value=(0, 1),
        ):
            client.add_local(user_id="u1", role="user", thread_id="t1", content="hi")

            loop = asyncio.get_running_loop()
            t0 = loop.time()
            await client.push_to_cosmos()
            elapsed = loop.time() - t0

            # If push had awaited the slow trigger we'd see >= 0.5s here.
            assert elapsed < 0.4, (
                f"push_to_cosmos awaited trigger inline (elapsed={elapsed:.3f}s)"
            )

            # Drain the background task so pytest doesn't warn about a
            # destroyed-but-pending task at teardown.
            await asyncio.gather(*list(client._background_tasks), return_exceptions=True)
