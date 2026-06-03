"""Tests for AsyncCosmosMemoryClient.process_now() / process_now_and_wait() processor delegation."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from azure.cosmos.agent_memory.aio.cosmos_memory_client import AsyncCosmosMemoryClient
from azure.cosmos.agent_memory.aio.processors import (
    AsyncDurableFunctionProcessor,
    AsyncInProcessProcessor,
    ProcessThreadResult,
)
from azure.cosmos.agent_memory.exceptions import CosmosNotConnectedError, LLMError, ValidationError


def _connected(processor=None) -> AsyncCosmosMemoryClient:
    client = AsyncCosmosMemoryClient(use_default_credential=False, processor=processor)
    client._memories_container_client = MagicMock()  # truthy → _require_cosmos passes
    client._turns_container_client = client._memories_container_client
    client._summaries_container_client = client._memories_container_client
    return client


def _patch_get_thread(client, turns):
    client.get_thread = AsyncMock(return_value=turns)


@pytest.mark.asyncio
async def test_process_now_with_inprocess_invokes_full_pipeline():
    """process_now must fire ALL FIVE steps for AsyncInProcess: thread_summary, extract,
    reconcile, procedural, user_summary. Pre-fix this was only the first 3, so
    procedural + user_summary never ran when callers used add_cosmos + process_now."""
    client = _connected()
    pipeline = AsyncMock()
    pipeline.generate_thread_summary.return_value = {"id": "s"}
    pipeline.extract_memories.return_value = {"facts": 1}
    pipeline.reconcile_memories.return_value = {"kept": 0, "merged": 0, "contradicted": 0}
    pipeline.synthesize_procedural.return_value = {"id": "proc1", "type": "procedural"}
    pipeline.generate_user_summary.return_value = {"id": "us1", "type": "user_summary"}
    pipeline._store = client._get_store()
    pipeline._containers = dict(client._containers)
    client._pipeline = pipeline
    _patch_get_thread(client, [{"role": "user"}])

    result = await client.process_now(user_id="u", thread_id="t")

    assert isinstance(result, ProcessThreadResult)
    assert isinstance(client._processor, AsyncInProcessProcessor)
    pipeline.generate_thread_summary.assert_awaited_once_with("u", "t")
    pipeline.extract_memories.assert_awaited_once_with("u", "t")
    pipeline.reconcile_memories.assert_awaited_once_with("u", 50)
    pipeline.synthesize_procedural.assert_awaited_once_with("u", force=False)
    pipeline.generate_user_summary.assert_awaited_once_with("u", None)
    assert result.procedural == {"id": "proc1", "type": "procedural"}
    assert result.user_summary == {"id": "us1", "type": "user_summary"}


@pytest.mark.asyncio
async def test_process_now_swallows_procedural_failure():
    """A transient LLM failure in synthesize_procedural must NOT erase the work
    already persisted by the per-thread steps. WARNING-log + continue."""
    client = _connected()
    pipeline = AsyncMock()
    pipeline.generate_thread_summary.return_value = {"id": "s"}
    pipeline.extract_memories.return_value = {"facts": 1}
    pipeline.reconcile_memories.return_value = {"kept": 0}
    pipeline.synthesize_procedural.side_effect = LLMError("LLM rate-limited")
    pipeline.generate_user_summary.return_value = {"id": "us1"}
    pipeline._store = client._get_store()
    pipeline._containers = dict(client._containers)
    client._pipeline = pipeline
    _patch_get_thread(client, [{"role": "user"}])

    result = await client.process_now(user_id="u", thread_id="t")

    assert result.procedural is None
    assert result.user_summary == {"id": "us1"}
    pipeline.synthesize_procedural.assert_awaited_once()
    pipeline.generate_user_summary.assert_awaited_once()


@pytest.mark.asyncio
async def test_process_now_swallows_user_summary_failure():
    """A transient LLM failure in generate_user_summary must NOT erase the work
    already persisted by the per-thread + procedural steps."""
    client = _connected()
    pipeline = AsyncMock()
    pipeline.generate_thread_summary.return_value = {"id": "s"}
    pipeline.extract_memories.return_value = {"facts": 1}
    pipeline.reconcile_memories.return_value = {"kept": 0}
    pipeline.synthesize_procedural.return_value = {"id": "proc1"}
    pipeline.generate_user_summary.side_effect = LLMError("LLM timeout")
    pipeline._store = client._get_store()
    pipeline._containers = dict(client._containers)
    client._pipeline = pipeline
    _patch_get_thread(client, [{"role": "user"}])

    result = await client.process_now(user_id="u", thread_id="t")

    assert result.procedural == {"id": "proc1"}
    assert result.user_summary is None


@pytest.mark.asyncio
async def test_process_now_swallows_transient_http_error_by_status_code():
    """HTTP exceptions with transient status codes must be swallowed."""

    class _FakeHttpExc(Exception):
        def __init__(self, status_code):
            super().__init__(f"HTTP {status_code}")
            self.status_code = status_code

    client = _connected()
    pipeline = AsyncMock()
    pipeline.generate_thread_summary.return_value = {"id": "s"}
    pipeline.extract_memories.return_value = {"facts": 1}
    pipeline.reconcile_memories.return_value = {"kept": 0}
    pipeline.synthesize_procedural.side_effect = _FakeHttpExc(429)
    pipeline.generate_user_summary.side_effect = _FakeHttpExc(503)
    pipeline._store = client._get_store()
    pipeline._containers = dict(client._containers)
    client._pipeline = pipeline
    _patch_get_thread(client, [{"role": "user"}])

    result = await client.process_now(user_id="u", thread_id="t")

    assert result.procedural is None
    assert result.user_summary is None


@pytest.mark.asyncio
async def test_process_now_propagates_permanent_procedural_failure():
    """Permanent failures (e.g. KeyError from a schema bug) must propagate."""
    client = _connected()
    pipeline = AsyncMock()
    pipeline.generate_thread_summary.return_value = {"id": "s"}
    pipeline.extract_memories.return_value = {"facts": 1}
    pipeline.reconcile_memories.return_value = {"kept": 0}
    pipeline.synthesize_procedural.side_effect = KeyError("missing_required_field")
    pipeline._store = client._get_store()
    pipeline._containers = dict(client._containers)
    client._pipeline = pipeline
    _patch_get_thread(client, [{"role": "user"}])

    with pytest.raises(KeyError):
        await client.process_now(user_id="u", thread_id="t")
    pipeline.generate_user_summary.assert_not_called()


@pytest.mark.asyncio
async def test_process_now_propagates_permanent_user_summary_failure():
    """ValidationError from generate_user_summary must propagate."""
    client = _connected()
    pipeline = AsyncMock()
    pipeline.generate_thread_summary.return_value = {"id": "s"}
    pipeline.extract_memories.return_value = {"facts": 1}
    pipeline.reconcile_memories.return_value = {"kept": 0}
    pipeline.synthesize_procedural.return_value = {"id": "proc1"}
    pipeline.generate_user_summary.side_effect = ValidationError("bad payload")
    pipeline._store = client._get_store()
    pipeline._containers = dict(client._containers)
    client._pipeline = pipeline
    _patch_get_thread(client, [{"role": "user"}])

    with pytest.raises(ValidationError):
        await client.process_now(user_id="u", thread_id="t")


@pytest.mark.asyncio
async def test_process_now_with_durable_skips_tail_steps():
    """Durable mode must NOT call synthesize_procedural or generate_user_summary —
    those are driven by the change-feed-fed sibling Function app."""
    client = _connected(processor=AsyncDurableFunctionProcessor())
    pipeline = AsyncMock()
    client._pipeline = pipeline
    _patch_get_thread(client, [])

    result = await client.process_now(user_id="u", thread_id="t")

    assert isinstance(result, ProcessThreadResult)
    assert result.thread_summary is None
    assert result.procedural is None
    assert result.user_summary is None
    pipeline.generate_thread_summary.assert_not_called()
    pipeline.synthesize_procedural.assert_not_called()
    pipeline.generate_user_summary.assert_not_called()


@pytest.mark.asyncio
async def test_process_now_requires_cosmos():
    client = AsyncCosmosMemoryClient(use_default_credential=False)
    with pytest.raises(CosmosNotConnectedError):
        await client.process_now(user_id="u", thread_id="t")


@pytest.mark.asyncio
async def test_process_now_and_wait_inprocess_returns_true():
    client = _connected()
    pipeline = AsyncMock()
    pipeline.generate_thread_summary.return_value = {"id": "s"}
    pipeline.extract_memories.return_value = {}
    pipeline.reconcile_memories.return_value = {}
    pipeline._store = client._get_store()
    pipeline._containers = dict(client._containers)
    client._pipeline = pipeline
    _patch_get_thread(client, [])

    assert await client.process_now_and_wait(user_id="u", thread_id="t") is True


@pytest.mark.asyncio
async def test_process_now_and_wait_durable_polls_until_summary_appears():
    client = _connected(processor=AsyncDurableFunctionProcessor())
    _patch_get_thread(client, [])
    client.get_thread_summary = AsyncMock(side_effect=[[], [], [{"id": "summary"}]])

    async def _no_sleep(_):
        return None

    with patch("asyncio.sleep", new=_no_sleep):
        ok = await client.process_now_and_wait(user_id="u", thread_id="t", timeout=5.0)

    assert ok is True
    assert client.get_thread_summary.await_count == 3


@pytest.mark.asyncio
async def test_process_now_and_wait_durable_returns_false_on_timeout():
    client = _connected(processor=AsyncDurableFunctionProcessor())
    _patch_get_thread(client, [])
    client.get_thread_summary = AsyncMock(return_value=[])

    async def _no_sleep(_):
        return None

    with patch("asyncio.sleep", new=_no_sleep):
        ok = await client.process_now_and_wait(user_id="u", thread_id="t", timeout=0.01)

    assert ok is False
    assert client.get_thread_summary.await_count >= 1


def test_constructor_accepts_processor_kwarg():
    durable = AsyncDurableFunctionProcessor()
    client = AsyncCosmosMemoryClient(use_default_credential=False, processor=durable)
    assert client._processor is durable
    assert client._processor_explicit is True


def test_constructor_default_processor_is_none():
    client = AsyncCosmosMemoryClient(use_default_credential=False)
    assert client._processor is None
    assert client._processor_explicit is False
