"""Tests for CosmosMemoryClient.process_now() / process_now_and_wait() processor delegation."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from azure.cosmos.agent_memory.cosmos_memory_client import CosmosMemoryClient
from azure.cosmos.agent_memory.exceptions import CosmosNotConnectedError, LLMError, ValidationError
from azure.cosmos.agent_memory.processors import (
    DurableFunctionProcessor,
    InProcessProcessor,
    ProcessThreadResult,
)


def _connected(processor=None) -> CosmosMemoryClient:
    client = CosmosMemoryClient(use_default_credential=False, processor=processor)
    client._memories_container_client = MagicMock()
    client._turns_container_client = client._memories_container_client
    client._summaries_container_client = client._memories_container_client
    return client


def _patch_get_thread(client, turns):
    """Make get_thread() return a fixed list without going through Cosmos."""
    client.get_thread = MagicMock(return_value=turns)


def test_process_now_with_inprocess_invokes_full_pipeline():
    """process_now must fire ALL FIVE steps for InProcess: thread_summary, extract,
    reconcile, procedural, user_summary. Pre-fix this was only the first 3, so
    procedural + user_summary never ran when callers used add_cosmos + process_now."""
    client = _connected()  # default → InProcessProcessor lazily built
    pipeline = MagicMock()
    pipeline.generate_thread_summary.return_value = {"id": "s", "type": "thread_summary"}
    pipeline.extract_memories.return_value = {"facts": 1}
    pipeline.reconcile_memories.return_value = {"kept": 0, "merged": 0, "contradicted": 0}
    pipeline.synthesize_procedural.return_value = {"id": "proc1", "type": "procedural"}
    pipeline.generate_user_summary.return_value = {"id": "us1", "type": "user_summary"}
    pipeline._store = client._get_store()
    pipeline._containers = dict(client._containers)
    client._pipeline = pipeline
    _patch_get_thread(client, [{"role": "user", "content": "hi"}])

    result = client.process_now(user_id="u1", thread_id="t1")

    assert isinstance(result, ProcessThreadResult)
    assert isinstance(client._processor, InProcessProcessor)
    pipeline.generate_thread_summary.assert_called_once_with("u1", "t1")
    pipeline.extract_memories.assert_called_once_with("u1", "t1")
    pipeline.reconcile_memories.assert_called_once_with("u1", 50)
    pipeline.synthesize_procedural.assert_called_once_with(user_id="u1", force=False)
    pipeline.generate_user_summary.assert_called_once_with("u1", None)
    assert result.procedural == {"id": "proc1", "type": "procedural"}
    assert result.user_summary == {"id": "us1", "type": "user_summary"}


def test_process_now_swallows_procedural_failure():
    """A transient LLM failure in synthesize_procedural must NOT erase the work
    already persisted by the per-thread steps. WARNING-log + continue."""
    client = _connected()
    pipeline = MagicMock()
    pipeline.generate_thread_summary.return_value = {"id": "s"}
    pipeline.extract_memories.return_value = {"facts": 1}
    pipeline.reconcile_memories.return_value = {"kept": 0}
    pipeline.synthesize_procedural.side_effect = LLMError("LLM rate-limited")
    pipeline.generate_user_summary.return_value = {"id": "us1"}
    pipeline._store = client._get_store()
    pipeline._containers = dict(client._containers)
    client._pipeline = pipeline
    _patch_get_thread(client, [{"role": "user", "content": "hi"}])

    result = client.process_now(user_id="u1", thread_id="t1")

    assert result.procedural is None
    assert result.user_summary == {"id": "us1"}
    pipeline.synthesize_procedural.assert_called_once()
    pipeline.generate_user_summary.assert_called_once()


def test_process_now_swallows_user_summary_failure():
    """A transient LLM failure in generate_user_summary must NOT erase the work
    already persisted by the per-thread + procedural steps."""
    client = _connected()
    pipeline = MagicMock()
    pipeline.generate_thread_summary.return_value = {"id": "s"}
    pipeline.extract_memories.return_value = {"facts": 1}
    pipeline.reconcile_memories.return_value = {"kept": 0}
    pipeline.synthesize_procedural.return_value = {"id": "proc1"}
    pipeline.generate_user_summary.side_effect = LLMError("LLM timeout")
    pipeline._store = client._get_store()
    pipeline._containers = dict(client._containers)
    client._pipeline = pipeline
    _patch_get_thread(client, [{"role": "user", "content": "hi"}])

    result = client.process_now(user_id="u1", thread_id="t1")

    assert result.procedural == {"id": "proc1"}
    assert result.user_summary is None


def test_process_now_swallows_transient_http_error_by_status_code():
    """Cosmos / HTTP exceptions with transient status codes (429, 503) must be
    swallowed — they're infrastructure hiccups, not bugs."""

    class _FakeHttpExc(Exception):
        def __init__(self, status_code):
            super().__init__(f"HTTP {status_code}")
            self.status_code = status_code

    client = _connected()
    pipeline = MagicMock()
    pipeline.generate_thread_summary.return_value = {"id": "s"}
    pipeline.extract_memories.return_value = {"facts": 1}
    pipeline.reconcile_memories.return_value = {"kept": 0}
    pipeline.synthesize_procedural.side_effect = _FakeHttpExc(429)
    pipeline.generate_user_summary.side_effect = _FakeHttpExc(503)
    pipeline._store = client._get_store()
    pipeline._containers = dict(client._containers)
    client._pipeline = pipeline
    _patch_get_thread(client, [{"role": "user", "content": "hi"}])

    result = client.process_now(user_id="u1", thread_id="t1")

    assert result.procedural is None
    assert result.user_summary is None


def test_process_now_propagates_permanent_procedural_failure():
    """A non-transient failure (e.g. ``KeyError`` from a schema bug) must NOT
    be silently swallowed — it should surface to the caller so config /
    programmer bugs do not turn into invisible ``WARNING`` lines."""
    client = _connected()
    pipeline = MagicMock()
    pipeline.generate_thread_summary.return_value = {"id": "s"}
    pipeline.extract_memories.return_value = {"facts": 1}
    pipeline.reconcile_memories.return_value = {"kept": 0}
    pipeline.synthesize_procedural.side_effect = KeyError("missing_required_field")
    pipeline._store = client._get_store()
    pipeline._containers = dict(client._containers)
    client._pipeline = pipeline
    _patch_get_thread(client, [{"role": "user", "content": "hi"}])

    with pytest.raises(KeyError):
        client.process_now(user_id="u1", thread_id="t1")
    # user_summary must NOT be attempted after a permanent procedural failure
    pipeline.generate_user_summary.assert_not_called()


def test_process_now_propagates_permanent_user_summary_failure():
    """ValidationError from generate_user_summary (e.g. schema bug) must
    surface to the caller — silencing config bugs is a bug."""
    client = _connected()
    pipeline = MagicMock()
    pipeline.generate_thread_summary.return_value = {"id": "s"}
    pipeline.extract_memories.return_value = {"facts": 1}
    pipeline.reconcile_memories.return_value = {"kept": 0}
    pipeline.synthesize_procedural.return_value = {"id": "proc1"}
    pipeline.generate_user_summary.side_effect = ValidationError("bad payload")
    pipeline._store = client._get_store()
    pipeline._containers = dict(client._containers)
    client._pipeline = pipeline
    _patch_get_thread(client, [{"role": "user", "content": "hi"}])

    with pytest.raises(ValidationError):
        client.process_now(user_id="u1", thread_id="t1")


def test_process_now_with_durable_skips_tail_steps():
    """Durable mode must NOT call synthesize_procedural or generate_user_summary —
    those are driven by the change-feed-fed sibling Function app."""
    client = _connected(processor=DurableFunctionProcessor())
    pipeline = MagicMock()
    client._pipeline = pipeline
    _patch_get_thread(client, [{"role": "user", "content": "hi"}])

    result = client.process_now(user_id="u1", thread_id="t1")

    assert isinstance(result, ProcessThreadResult)
    assert result.thread_summary is None
    assert result.procedural is None
    assert result.user_summary is None
    pipeline.generate_thread_summary.assert_not_called()
    pipeline.extract_memories.assert_not_called()
    pipeline.reconcile_memories.assert_not_called()
    pipeline.synthesize_procedural.assert_not_called()
    pipeline.generate_user_summary.assert_not_called()


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
    pipeline._store = client._get_store()
    pipeline._containers = dict(client._containers)
    client._pipeline = pipeline
    _patch_get_thread(client, [])

    assert client.process_now_and_wait(user_id="u", thread_id="t") is True


def test_process_now_and_wait_durable_polls_until_summary_appears():
    client = _connected(processor=DurableFunctionProcessor())
    _patch_get_thread(client, [])

    # First two polls return empty, third returns a summary
    client.get_thread_summary = MagicMock(side_effect=[[], [], [{"id": "summary_u_t"}]])

    with patch("time.sleep"):
        ok = client.process_now_and_wait(user_id="u", thread_id="t", timeout=5.0)

    assert ok is True
    assert client.get_thread_summary.call_count == 3


def test_process_now_and_wait_durable_returns_false_on_timeout():
    client = _connected(processor=DurableFunctionProcessor())
    _patch_get_thread(client, [])
    client.get_thread_summary = MagicMock(return_value=[])

    with patch("time.sleep"):
        ok = client.process_now_and_wait(user_id="u", thread_id="t", timeout=0.01)

    assert ok is False
    assert client.get_thread_summary.call_count >= 1


def test_process_now_and_wait_durable_swallows_search_errors_until_timeout():
    from azure.cosmos.exceptions import CosmosHttpResponseError

    client = _connected(processor=DurableFunctionProcessor())
    _patch_get_thread(client, [])
    client.get_thread_summary = MagicMock(side_effect=CosmosHttpResponseError(message="429 throttled", status_code=429))

    with patch("time.sleep"):
        ok = client.process_now_and_wait(user_id="u", thread_id="t", timeout=0.01)

    assert ok is False
    assert client.get_thread_summary.call_count >= 1


def test_process_now_and_wait_durable_propagates_non_cosmos_errors():
    """Non-Cosmos errors must NOT be silently swallowed in the polling loop —
    operators would otherwise wait the full timeout with no signal."""
    client = _connected(processor=DurableFunctionProcessor())
    _patch_get_thread(client, [])
    client.get_thread_summary = MagicMock(side_effect=RuntimeError("bug"))

    with patch("time.sleep"), pytest.raises(RuntimeError, match="bug"):
        client.process_now_and_wait(user_id="u", thread_id="t", timeout=0.01)


def test_constructor_accepts_processor_kwarg():
    durable = DurableFunctionProcessor()
    client = CosmosMemoryClient(use_default_credential=False, processor=durable)
    assert client._processor is durable
    assert client._processor_explicit is True


def test_constructor_default_processor_is_none_until_lazy_build():
    client = CosmosMemoryClient(use_default_credential=False)
    assert client._processor is None
    assert client._processor_explicit is False
