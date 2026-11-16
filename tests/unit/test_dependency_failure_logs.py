"""Dependency failures and the SDK's in-process processing log a description of the error, never its text.

Each case puts query, memory or model text in a failing dependency's error message (or a pydantic input value),
captures every record at DEBUG, and checks the text is absent from the log (and its JSON form) while the replacement
diagnostic is present.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import httpx
import openai
import pytest
from azure.cosmos.exceptions import CosmosHttpResponseError
from pydantic import BaseModel

from azure.cosmos.agent_memory import _counters
from azure.cosmos.agent_memory import auto_trigger as sync_trigger
from azure.cosmos.agent_memory._container_routing import ContainerKey
from azure.cosmos.agent_memory.aio import auto_trigger as async_trigger
from azure.cosmos.agent_memory.aio.chat import AsyncChatClient
from azure.cosmos.agent_memory.aio.cosmos_memory_client import AsyncCosmosMemoryClient, _log_auto_trigger_task_failure
from azure.cosmos.agent_memory.aio.store import AsyncMemoryStore
from azure.cosmos.agent_memory.chat import ChatClient
from azure.cosmos.agent_memory.cosmos_memory_client import CosmosMemoryClient
from azure.cosmos.agent_memory.exceptions import LLMError
from azure.cosmos.agent_memory.store import MemoryStore
from tests.unit.aio import test_process_now as async_process_now
from tests.unit.aio.test_auto_trigger import _AsyncFakeCounterContainer
from tests.unit.test_auto_trigger import _FakeCounterContainer
from tests.unit.test_chat import _reply
from tests.unit.test_log_redaction import MEMORY, MODEL, QUERY, _expect, _failure
from tests.unit.test_process_now import _connected, _patch_get_thread


@pytest.fixture
def logs(caplog: pytest.LogCaptureFixture) -> pytest.LogCaptureFixture:
    caplog.set_level(logging.DEBUG)
    return caplog


class _Answer(BaseModel):
    salience: float


def _invalid_answer() -> Exception:
    try:
        _Answer(salience=MODEL)
    except Exception as exc:  # noqa: BLE001 - the pydantic error, with the model's answer as its input value
        return exc
    raise AssertionError("expected a validation error")


_INVALID = "pydantic_core.ValidationError(model=_Answer, errors=1, fields=salience:float_parsing)"


# --- Azure OpenAI retries -------------------------------------------------------------------------------------


def _provider_error(status: int, error_type: type[openai.APIStatusError], text: str) -> openai.APIStatusError:
    request = httpx.Request("POST", "https://test.openai.azure.com/openai/deployments/test/chat/completions")
    response = httpx.Response(status, request=request)
    return error_type(message=f"Error code: {status} - prompt {text!r} was rejected", response=response, body=None)


def _provider_failures() -> list[Any]:
    return [
        _provider_error(429, openai.RateLimitError, QUERY),
        _provider_error(503, openai.InternalServerError, MEMORY),
        _reply("recovered"),
    ]


_CHAT_DIAGNOSTICS = (
    "LLM rate-limited (attempt 1/3), retrying in ",
    ": openai.RateLimitError(status=429) at ",
    "LLM API error 503 (attempt 2/3), retrying in ",
    ": openai.InternalServerError(status=503) at ",
)


def test_chat_retries_log_the_status_not_the_provider_message(logs, monkeypatch) -> None:
    monkeypatch.setattr("azure.cosmos.agent_memory.chat.time.sleep", lambda seconds: None)
    client = ChatClient(endpoint="https://test.openai.azure.com", api_key="test-key")
    client._client = MagicMock()
    client._client.chat.completions.create.side_effect = _provider_failures()

    assert client.generate([{"role": "user", "content": QUERY}], max_retries=3) == "recovered"
    _expect(logs, *_CHAT_DIAGNOSTICS)


async def test_async_chat_retries_log_the_status_not_the_provider_message(logs, monkeypatch) -> None:
    monkeypatch.setattr("azure.cosmos.agent_memory.aio.chat.asyncio.sleep", AsyncMock())
    client = AsyncChatClient(endpoint="https://test.openai.azure.com", api_key="key")
    client._client = MagicMock()
    client._client.chat.completions.create = AsyncMock(side_effect=_provider_failures())

    assert await client.generate([{"role": "user", "content": QUERY}], max_retries=3) == "recovered"
    _expect(logs, *_CHAT_DIAGNOSTICS)


# --- store operations -----------------------------------------------------------------------------------------


def _containers(memories: Any, *, asynchronous: bool = False) -> dict[ContainerKey, Any]:
    def container() -> MagicMock:
        mock = MagicMock()
        if asynchronous:
            for name in ("upsert_item", "create_item", "replace_item", "patch_item", "delete_item"):
                setattr(mock, name, AsyncMock(return_value={}))
        return mock

    return {ContainerKey.TURNS: container(), ContainerKey.MEMORIES: memories, ContainerKey.SUMMARIES: container()}


def _local_fact() -> dict[str, Any]:
    return {
        "id": "fact_local",
        "type": "fact",
        "role": "system",
        "user_id": "u1",
        "thread_id": "t1",
        "content": MEMORY,
        "created_at": "2025-01-01T00:00:00+00:00",
    }


_STORE_DIAGNOSTICS = (
    "upsert_memory: embedding generation failed for ",
    "(RuntimeError at ",
    "push_to_cosmos: batch embedding generation failed (RuntimeError at ",
    "supersede failed id=fact_1 superseder=fact_2 status=500: azure.cosmos.CosmosHttpResponseError(status=500) at ",
)


def _failed_write() -> CosmosHttpResponseError:
    return CosmosHttpResponseError(status_code=500, message=_failure(MEMORY).args[0])


def test_store_failures_log_the_failure_not_the_memory(logs) -> None:
    embeddings = MagicMock()
    embeddings.generate.side_effect = _failure(MEMORY)
    embeddings.generate_batch.side_effect = _failure(MEMORY)
    store = MemoryStore(containers=_containers(MagicMock()), embeddings_client=embeddings)
    failing = MagicMock()
    failing.upsert_item.side_effect = _failed_write()

    store.add(user_id="u1", role="user", content=MEMORY, memory_type="fact", thread_id="t1")
    store.push([_local_fact()])
    old = {"id": "fact_1", "type": "fact", "content": MEMORY}
    assert MemoryStore(containers=_containers(failing)).mark_superseded(old, "fact_2", reason="dup") is False
    _expect(logs, *_STORE_DIAGNOSTICS)


async def test_async_store_failures_log_the_failure_not_the_memory(logs) -> None:
    embeddings = MagicMock()
    embeddings.generate = AsyncMock(side_effect=_failure(MEMORY))
    embeddings.generate_batch = AsyncMock(side_effect=_failure(MEMORY))
    memories, failing = MagicMock(), MagicMock()
    memories.upsert_item = AsyncMock(return_value={})
    failing.upsert_item = AsyncMock(side_effect=_failed_write())
    store = AsyncMemoryStore(containers=_containers(memories, asynchronous=True), embeddings_client=embeddings)

    await store.add(user_id="u1", role="user", content=MEMORY, memory_type="fact", thread_id="t1")
    await store.push([_local_fact()])
    old = {"id": "fact_1", "type": "fact", "content": MEMORY}
    superseding = AsyncMemoryStore(containers=_containers(failing, asynchronous=True))
    assert await superseding.mark_superseded(old, "fact_2", reason="dup") is False
    _expect(logs, *_STORE_DIAGNOSTICS)


# --- search ---------------------------------------------------------------------------------------------------


def _ranked(top_k: int) -> list[dict[str, Any]]:
    return [{"id": f"fact_{i}", "content": f"fact {i}", "type": "fact"} for i in range(top_k)]


def _failing_search(**kwargs: Any) -> list[dict[str, Any]]:
    if kwargs.get("vector_only"):
        raise RuntimeError(f"slot query {QUERY!r} failed near {MEMORY!r}")
    return _ranked(kwargs["top_k"])


_SEARCH_KWARGS = dict(
    user_id="u1", top_k=5, include_summaries=True, include_turns=True, preference_slot=True, procedure_slot=True
)
_SEARCH_DIAGNOSTICS = (
    "search_cosmos: include_summaries search failed (RuntimeError at ",
    "search_cosmos: include_turns turn search failed (RuntimeError at ",
    "search_cosmos: slot search failed (RuntimeError at ",
)


def test_search_fallbacks_log_the_failure_not_the_query(logs) -> None:
    client = CosmosMemoryClient(use_default_credential=False)
    store = MagicMock()
    store.search.side_effect = _failing_search
    store.search_summaries.side_effect = RuntimeError(f"summary query {QUERY!r} failed")
    store.search_turns.side_effect = RuntimeError(f"turn query {QUERY!r} failed near {MEMORY!r}")
    client._get_store = MagicMock(return_value=store)

    out = client.search_cosmos(QUERY, **_SEARCH_KWARGS)

    assert [doc["id"] for doc in out] == [doc["id"] for doc in _ranked(5)]
    _expect(logs, *_SEARCH_DIAGNOSTICS)


async def test_async_search_fallbacks_log_the_failure_not_the_query(logs) -> None:
    client = AsyncCosmosMemoryClient(use_default_credential=False)
    store = MagicMock()
    store._embed = AsyncMock(return_value=[0.1, 0.2])
    store.search = AsyncMock(side_effect=_failing_search)
    store.search_summaries = AsyncMock(side_effect=RuntimeError(f"summary query {QUERY!r} failed"))
    store.search_turns = AsyncMock(side_effect=RuntimeError(f"turn query {QUERY!r} failed near {MEMORY!r}"))
    client._get_store = MagicMock(return_value=store)

    out = await client.search_cosmos(QUERY, **_SEARCH_KWARGS)

    assert [doc["id"] for doc in out] == [doc["id"] for doc in _ranked(5)]
    _expect(logs, *_SEARCH_DIAGNOSTICS)


# --- counters, summary lookup and client lifecycle ------------------------------------------------------------


def test_counter_and_client_failures_log_the_failure_not_the_text(logs) -> None:
    container = MagicMock()
    container.patch_item.side_effect = _failure(MEMORY, 503)
    _counters.stamp_failure_sync(container, "thread:u1:t1", "u1", "t1", "process_extract_memories: failed")

    client = CosmosMemoryClient(use_default_credential=False)
    client._cosmos_client = MagicMock()
    client._cosmos_client.get_database_client.side_effect = _failure(MEMORY)
    assert client._get_counter_container() is None
    client.get_thread_summary = MagicMock(side_effect=CosmosHttpResponseError(status_code=500, message=MEMORY))
    assert client._summary_exists(user_id="u1", thread_id="t1") is False
    client._cosmos_client = MagicMock()
    client._cosmos_client.close.side_effect = _failure(MEMORY)
    client._drain_cosmos_client()

    _expect(
        logs,
        "stamp_failure_sync failed counter_id=thread:u1:t1: RuntimeError(status=503) at ",
        "unreachable: RuntimeError at ",
        "_summary_exists: Cosmos error user_id=u1 thread_id=t1 status=500: "
        "azure.cosmos.CosmosHttpResponseError(status=500)",
        "Failed to close prior Cosmos client during reconnect: RuntimeError at ",
    )


async def test_async_counter_and_client_failures_log_the_failure_not_the_text(logs) -> None:
    container = MagicMock()
    container.patch_item = AsyncMock(side_effect=_failure(MEMORY, 503))
    await _counters.stamp_failure_async(container, "thread:u1:t1", "u1", "t1", "process_extract_memories: failed")

    client = AsyncCosmosMemoryClient(use_default_credential=False)
    client._cosmos_client = MagicMock()
    client._cosmos_client.get_database_client.side_effect = _failure(MEMORY)
    assert client._get_counter_container() is None
    client.get_thread_summary = AsyncMock(side_effect=CosmosHttpResponseError(status_code=500, message=MEMORY))
    assert await client._summary_exists(user_id="u1", thread_id="t1") is False

    _expect(
        logs,
        "stamp_failure_async failed counter_id=thread:u1:t1: RuntimeError(status=503) at ",
        "unreachable: RuntimeError at ",
        "_summary_exists: Cosmos error user_id=u1 thread_id=t1 status=500: "
        "azure.cosmos.CosmosHttpResponseError(status=500)",
    )


# --- the SDK's in-process processing --------------------------------------------------------------------------


class _FailingCounters(_FakeCounterContainer):
    def read_item(self, *, item, partition_key):
        raise _failure(MEMORY, 503)


class _AsyncFailingCounters(_AsyncFakeCounterContainer):
    async def read_item(self, *, item, partition_key):
        raise _failure(MEMORY, 503)


def _failing_processor(*, asynchronous: bool) -> MagicMock:
    processor = MagicMock()
    failures = {
        "process_extract_memories": _invalid_answer(),
        "process_thread_summary": LLMError(f"LLM returned invalid JSON (preview={MODEL!r})"),
        "process_user_summary": _failure(MEMORY),
    }
    for name, error in failures.items():
        setattr(processor, name, AsyncMock(side_effect=error) if asynchronous else MagicMock(side_effect=error))
    return processor


_STEPS = dict(new_count=4, fire_episode=False, fire_dedup=False)
_THREAD_ARGUMENTS = dict(n_facts=1, n_summary=0, n_episode=0, n_dedup_turns=0)


def _trigger_diagnostics(prefix: str, thread_counter: str, user_counter: str) -> tuple[str, ...]:
    return (
        f"{prefix} process_extract_memories failed for u1/t1: {_INVALID}",
        f"{prefix} process_thread_summary failed for u1/t1: agent_memory.LLMError(code=llm) at ",
        f"{prefix} process_user_summary failed for u1: RuntimeError at ",
        f"{thread_counter} increment failed for u1/t1: RuntimeError(status=503) at ",
        f"{user_counter} increment failed for u1: RuntimeError(status=503) at ",
    )


def _reason(counters: Any, counter_id: str) -> str:
    return counters.store[counter_id]["last_failure_reason"]


def test_in_process_trigger_failures_log_the_failure_not_the_text(logs) -> None:
    processor = _failing_processor(asynchronous=False)
    counters = _FakeCounterContainer()

    sync_trigger._fire_thread_steps(
        processor, counters, "thread:u1:t1", "u1", "t1", fire_extract=True, fire_summary=True, **_STEPS
    )
    sync_trigger._trigger_user_steps(processor, counters, {"u1": 4}, n_user=1)
    sync_trigger._trigger_thread_steps(processor, _FailingCounters(), {("u1", "t1"): 1}, **_THREAD_ARGUMENTS)
    sync_trigger._trigger_user_steps(processor, _FailingCounters(), {"u1": 4}, n_user=1)

    _expect(logs, *_trigger_diagnostics("Auto-trigger", "Counter", "User counter"))


async def test_async_in_process_trigger_failures_log_the_failure_not_the_text(logs) -> None:
    processor = _failing_processor(asynchronous=True)
    counters = _AsyncFakeCounterContainer()

    await async_trigger._fire_thread_steps(
        processor, counters, "thread:u1:t1", "u1", "t1", fire_extract=True, fire_summary=True, **_STEPS
    )
    await async_trigger._trigger_user_steps(processor, counters, {"u1": 4}, n_user=1)
    await async_trigger._trigger_thread_steps(
        processor, _AsyncFailingCounters(), {("u1", "t1"): 1}, **_THREAD_ARGUMENTS
    )
    await async_trigger._trigger_user_steps(processor, _AsyncFailingCounters(), {"u1": 4}, n_user=1)

    _expect(logs, *_trigger_diagnostics("Async auto-trigger", "Async counter", "Async user counter"))


_REASON_PREFIXES = (
    f"process_extract_memories: {_INVALID}",
    "process_thread_summary: agent_memory.LLMError(code=llm)",
    "process_user_summary: RuntimeError",
)


def _assert_reasons_describe_the_errors(reasons: list[str]) -> None:
    for reason, prefix in zip(reasons, _REASON_PREFIXES, strict=True):
        assert reason.startswith(prefix), reason
    assert [text for text in (QUERY, MEMORY, MODEL) if any(text in reason for reason in reasons)] == []


def test_a_failed_step_stamps_a_description_of_the_error_on_its_counter() -> None:
    processor = _failing_processor(asynchronous=False)
    counters = _FakeCounterContainer()

    sync_trigger._fire_thread_steps(
        processor, counters, "thread:u1:t1", "u1", "t1", fire_extract=True, fire_summary=False, **_STEPS
    )
    extract = _reason(counters, "thread:u1:t1")
    sync_trigger._fire_thread_steps(
        processor, counters, "thread:u1:t1", "u1", "t1", fire_extract=False, fire_summary=True, **_STEPS
    )
    summary = _reason(counters, "thread:u1:t1")
    sync_trigger._trigger_user_steps(processor, counters, {"u1": 4}, n_user=1)

    _assert_reasons_describe_the_errors([extract, summary, _reason(counters, "user:u1")])


async def test_async_failed_step_stamps_a_description_of_the_error_on_its_counter() -> None:
    processor = _failing_processor(asynchronous=True)
    counters = _AsyncFakeCounterContainer()

    await async_trigger._fire_thread_steps(
        processor, counters, "thread:u1:t1", "u1", "t1", fire_extract=True, fire_summary=False, **_STEPS
    )
    extract = _reason(counters, "thread:u1:t1")
    await async_trigger._fire_thread_steps(
        processor, counters, "thread:u1:t1", "u1", "t1", fire_extract=False, fire_summary=True, **_STEPS
    )
    summary = _reason(counters, "thread:u1:t1")
    await async_trigger._trigger_user_steps(processor, counters, {"u1": 4}, n_user=1)

    _assert_reasons_describe_the_errors([extract, summary, _reason(counters, "user:u1")])


def test_client_auto_trigger_failures_log_the_failure_not_the_text(logs) -> None:
    client = _connected()
    store = MagicMock()
    store.add.return_value = "turn_1"
    client._get_store = MagicMock(return_value=store)
    client._maybe_auto_trigger = MagicMock(side_effect=_failure(MEMORY))

    client.upsert_memory(user_id="u1", role="user", content=MEMORY, thread_id="t1")
    client.push_to_cosmos()
    _expect(
        logs,
        "Auto-trigger after upsert_memory failed: RuntimeError at ",
        "Auto-trigger after push_to_cosmos failed: RuntimeError at ",
    )


async def test_async_client_auto_trigger_failures_log_the_failure_not_the_text(logs) -> None:
    client = async_process_now._connected()
    store = MagicMock()
    store.add = AsyncMock(return_value="turn_1")
    client._get_store = MagicMock(return_value=store)
    client._maybe_auto_trigger = AsyncMock(side_effect=_failure(MEMORY))

    await client.upsert_memory(user_id="u1", role="user", content=MEMORY, thread_id="t1")
    tasks = list(client._background_tasks)
    await asyncio.wait(tasks)
    await asyncio.sleep(0)

    _expect(logs, "Background auto-trigger task failed: RuntimeError at ")


async def test_a_failed_background_trigger_task_logs_the_failure_not_the_text(logs) -> None:
    async def fail() -> None:
        raise _failure(MEMORY)

    task = asyncio.ensure_future(fail())
    await asyncio.gather(task, return_exceptions=True)

    _log_auto_trigger_task_failure(task)
    _expect(logs, "Background auto-trigger task failed: RuntimeError at ")


def _process_now_pipeline(*, asynchronous: bool) -> MagicMock:
    pipeline = MagicMock()
    returns = {
        "generate_thread_summary": {"id": "s"},
        "extract_memories": {"facts": 1},
        "reconcile_memories": {"kept": 0},
    }
    failures = ("synthesize_procedural", "generate_user_summary")
    for name, value in returns.items():
        setattr(pipeline, name, AsyncMock(return_value=value) if asynchronous else MagicMock(return_value=value))
    for name in failures:
        error = LLMError(f"LLM rate-limited answering {MODEL!r}")
        setattr(pipeline, name, AsyncMock(side_effect=error) if asynchronous else MagicMock(side_effect=error))
    return pipeline


_PROCESS_NOW_DIAGNOSTICS = tuple(
    f"process_now: {step} failed (transient) for user_id=u1: agent_memory.LLMError(code=llm) at "
    for step in ("synthesize_procedural", "process_user_summary")
)


def test_process_now_transient_failures_log_the_failure_not_the_answer(logs) -> None:
    client = _connected()
    pipeline = _process_now_pipeline(asynchronous=False)
    pipeline._store = client._get_store()
    pipeline._containers = dict(client._containers)
    client._pipeline = pipeline
    _patch_get_thread(client, [{"role": "user", "content": MEMORY}])

    client.process_now(user_id="u1", thread_id="t1")
    _expect(logs, *_PROCESS_NOW_DIAGNOSTICS)


async def test_async_process_now_transient_failures_log_the_failure_not_the_answer(logs) -> None:
    client = async_process_now._connected()
    pipeline = _process_now_pipeline(asynchronous=True)
    pipeline._store = client._get_store()
    pipeline._containers = dict(client._containers)
    client._pipeline = pipeline
    async_process_now._patch_get_thread(client, [{"role": "user", "content": MEMORY}])

    await client.process_now(user_id="u1", thread_id="t1")
    _expect(logs, *_PROCESS_NOW_DIAGNOSTICS)
