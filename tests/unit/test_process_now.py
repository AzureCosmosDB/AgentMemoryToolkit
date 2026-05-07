"""Tests for CosmosMemoryClient.process_now() / process_now_and_wait() processor delegation."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from agent_memory_toolkit.cosmos_memory_client import CosmosMemoryClient
from agent_memory_toolkit.exceptions import CosmosNotConnectedError
from agent_memory_toolkit.processors import (
    DurableFunctionProcessor,
    InProcessProcessor,
    ProcessThreadResult,
)


def _connected(processor=None) -> CosmosMemoryClient:
    client = CosmosMemoryClient(use_default_credential=False, processor=processor)
    client._container_client = MagicMock()
    return client


def _patch_get_thread(client, turns):
    """Make get_thread() return a fixed list without going through Cosmos."""
    client.get_thread = MagicMock(return_value=turns)


def test_process_now_with_inprocess_invokes_pipeline():
    client = _connected()  # default → InProcessProcessor lazily built
    pipeline = MagicMock()
    pipeline.generate_thread_summary.return_value = {"id": "s", "type": "summary"}
    pipeline.extract_memories.return_value = {"facts": 1}
    pipeline.reconcile_memories.return_value = {"kept": 0, "merged": 0, "contradicted": 0}
    client._pipeline = pipeline  # short-circuit lazy build
    _patch_get_thread(client, [{"role": "user", "content": "hi"}])

    result = client.process_now(user_id="u1", thread_id="t1")

    assert isinstance(result, ProcessThreadResult)
    assert isinstance(client._processor, InProcessProcessor)
    pipeline.generate_thread_summary.assert_called_once_with("u1", "t1")
    pipeline.extract_memories.assert_called_once_with("u1", "t1")
    pipeline.reconcile_memories.assert_called_once_with("u1", 50)


def test_process_now_with_durable_is_noop():
    client = _connected(processor=DurableFunctionProcessor())
    pipeline = MagicMock()
    client._pipeline = pipeline
    _patch_get_thread(client, [{"role": "user", "content": "hi"}])

    result = client.process_now(user_id="u1", thread_id="t1")

    assert isinstance(result, ProcessThreadResult)
    assert result.thread_summary is None
    pipeline.generate_thread_summary.assert_not_called()
    pipeline.extract_memories.assert_not_called()
    pipeline.reconcile_memories.assert_not_called()


def test_process_now_requires_cosmos():
    client = CosmosMemoryClient(use_default_credential=False)
    with pytest.raises(CosmosNotConnectedError):
        client.process_now(user_id="u1", thread_id="t1")


def test_process_now_and_wait_inprocess_returns_true():
    client = _connected()
    pipeline = MagicMock()
    pipeline.generate_thread_summary.return_value = {"id": "s"}
    pipeline.extract_memories.return_value = {}
    pipeline.reconcile_memories.return_value = {}
    client._pipeline = pipeline
    _patch_get_thread(client, [])

    assert client.process_now_and_wait(user_id="u", thread_id="t") is True


def test_process_now_and_wait_durable_polls_until_summary_appears():
    client = _connected(processor=DurableFunctionProcessor())
    _patch_get_thread(client, [])

    # First two polls return empty, third returns a summary
    client.get_memories = MagicMock(side_effect=[[], [], [{"id": "summary_u_t"}]])

    with patch("time.sleep"):
        ok = client.process_now_and_wait(user_id="u", thread_id="t", timeout=5.0)

    assert ok is True
    assert client.get_memories.call_count == 3


def test_process_now_and_wait_durable_returns_false_on_timeout():
    client = _connected(processor=DurableFunctionProcessor())
    _patch_get_thread(client, [])
    client.get_memories = MagicMock(return_value=[])

    with patch("time.sleep"):
        ok = client.process_now_and_wait(user_id="u", thread_id="t", timeout=0.01)

    assert ok is False
    assert client.get_memories.call_count >= 1


def test_process_now_and_wait_durable_swallows_search_errors_until_timeout():
    client = _connected(processor=DurableFunctionProcessor())
    _patch_get_thread(client, [])
    client.get_memories = MagicMock(side_effect=RuntimeError("transient"))

    with patch("time.sleep"):
        ok = client.process_now_and_wait(user_id="u", thread_id="t", timeout=0.01)

    assert ok is False
    assert client.get_memories.call_count >= 1


def test_constructor_accepts_processor_kwarg():
    durable = DurableFunctionProcessor()
    client = CosmosMemoryClient(use_default_credential=False, processor=durable)
    assert client._processor is durable
    assert client._processor_explicit is True


def test_constructor_default_processor_is_none_until_lazy_build():
    client = CosmosMemoryClient(use_default_credential=False)
    assert client._processor is None
    assert client._processor_explicit is False
