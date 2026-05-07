"""Tests for AsyncCosmosMemoryClient.process_now() / process_now_and_wait() processor delegation."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agent_memory_toolkit.aio.cosmos_memory_client import AsyncCosmosMemoryClient
from agent_memory_toolkit.aio.processors import (
    AsyncDurableFunctionProcessor,
    AsyncInProcessProcessor,
    ProcessThreadResult,
)
from agent_memory_toolkit.exceptions import CosmosNotConnectedError


def _connected(processor=None) -> AsyncCosmosMemoryClient:
    client = AsyncCosmosMemoryClient(use_default_credential=False, processor=processor)
    client._container_client = MagicMock()  # truthy → _require_cosmos passes
    return client


def _patch_get_thread(client, turns):
    client.get_thread = AsyncMock(return_value=turns)


@pytest.mark.asyncio
async def test_process_now_with_inprocess_invokes_pipeline():
    client = _connected()
    pipeline = MagicMock()
    pipeline.generate_thread_summary.return_value = {"id": "s"}
    pipeline.extract_memories.return_value = {"facts": 1}
    pipeline.reconcile_memories.return_value = {"kept": 0, "merged": 0, "contradicted": 0}
    client._pipeline = pipeline
    _patch_get_thread(client, [{"role": "user"}])

    result = await client.process_now(user_id="u", thread_id="t")

    assert isinstance(result, ProcessThreadResult)
    assert isinstance(client._processor, AsyncInProcessProcessor)
    pipeline.generate_thread_summary.assert_called_once_with("u", "t")
    pipeline.extract_memories.assert_called_once_with("u", "t")
    pipeline.reconcile_memories.assert_called_once_with("u", 50)


@pytest.mark.asyncio
async def test_process_now_with_durable_is_noop():
    client = _connected(processor=AsyncDurableFunctionProcessor())
    pipeline = MagicMock()
    client._pipeline = pipeline
    _patch_get_thread(client, [])

    result = await client.process_now(user_id="u", thread_id="t")

    assert isinstance(result, ProcessThreadResult)
    assert result.thread_summary is None
    pipeline.generate_thread_summary.assert_not_called()


@pytest.mark.asyncio
async def test_process_now_requires_cosmos():
    client = AsyncCosmosMemoryClient(use_default_credential=False)
    with pytest.raises(CosmosNotConnectedError):
        await client.process_now(user_id="u", thread_id="t")


@pytest.mark.asyncio
async def test_process_now_and_wait_inprocess_returns_true():
    client = _connected()
    pipeline = MagicMock()
    pipeline.generate_thread_summary.return_value = {"id": "s"}
    pipeline.extract_memories.return_value = {}
    pipeline.reconcile_memories.return_value = {}
    client._pipeline = pipeline
    _patch_get_thread(client, [])

    assert await client.process_now_and_wait(user_id="u", thread_id="t") is True


@pytest.mark.asyncio
async def test_process_now_and_wait_durable_polls_until_summary_appears():
    client = _connected(processor=AsyncDurableFunctionProcessor())
    _patch_get_thread(client, [])
    client.get_memories = AsyncMock(side_effect=[[], [], [{"id": "summary"}]])

    async def _no_sleep(_):
        return None

    with patch("asyncio.sleep", new=_no_sleep):
        ok = await client.process_now_and_wait(user_id="u", thread_id="t", timeout=5.0)

    assert ok is True
    assert client.get_memories.await_count == 3


@pytest.mark.asyncio
async def test_process_now_and_wait_durable_returns_false_on_timeout():
    client = _connected(processor=AsyncDurableFunctionProcessor())
    _patch_get_thread(client, [])
    client.get_memories = AsyncMock(return_value=[])

    async def _no_sleep(_):
        return None

    with patch("asyncio.sleep", new=_no_sleep):
        ok = await client.process_now_and_wait(user_id="u", thread_id="t", timeout=0.01)

    assert ok is False
    assert client.get_memories.await_count >= 1


def test_constructor_accepts_processor_kwarg():
    durable = AsyncDurableFunctionProcessor()
    client = AsyncCosmosMemoryClient(use_default_credential=False, processor=durable)
    assert client._processor is durable
    assert client._processor_explicit is True


def test_constructor_default_processor_is_none():
    client = AsyncCosmosMemoryClient(use_default_credential=False)
    assert client._processor is None
    assert client._processor_explicit is False
