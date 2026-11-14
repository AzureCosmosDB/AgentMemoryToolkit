"""Failure and fallback logs keep query, memory and model text out.

Each case drives a distinctive string through a failing path, captures every record at DEBUG (as text, and through
the JSON formatter), and checks the string is absent while the diagnostic that replaced it is present.
"""

from __future__ import annotations

import json
import logging
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from azure.cosmos.agent_memory.aio.services.pipeline import AsyncPipelineService
from azure.cosmos.agent_memory.logging import JsonFormatter
from azure.cosmos.agent_memory.services._pipeline_helpers import parse_llm_json
from azure.cosmos.agent_memory.services.pipeline import PipelineService
from tests.unit.aio.services.test_episode_boundary_async import _service as _async_boundary_service
from tests.unit.aio.services.test_extract_episodes_async import _service as _async_episode_service
from tests.unit.aio.services.test_mark_turns_extracted_async import _AsyncPatchTurnsContainer
from tests.unit.aio.services.test_mark_turns_extracted_async import _service as _async_mark_service
from tests.unit.services.test_episode_boundary import _service as _boundary_service
from tests.unit.services.test_episode_boundary import _turn_at
from tests.unit.services.test_extract_dry import (
    _async_containers_for_store,
    _AsyncBatchChat,
    _AsyncEmbeddings,
    _AsyncStore,
    _BatchChat,
    _containers_for_store,
    _Store,
    _SyncEmbeddings,
    _turn,
)
from tests.unit.services.test_extract_episodes import _episode
from tests.unit.services.test_extract_episodes import _service as _episode_service
from tests.unit.services.test_mark_turns_extracted import _PatchTurnsContainer, _turn_doc
from tests.unit.services.test_mark_turns_extracted import _service as _mark_service

QUERY = "zx-query-lisbon-anniversary-surprise-7731"
MEMORY = "zx-memory-ana-allergic-to-cashews-4410"
MODEL = "zx-model-ana-prefers-window-seats-9902"


@pytest.fixture
def logs(caplog: pytest.LogCaptureFixture) -> pytest.LogCaptureFixture:
    caplog.set_level(logging.DEBUG)
    return caplog


def _assert_no_text(caplog: pytest.LogCaptureFixture) -> None:
    logged = caplog.text + "\n".join(JsonFormatter().format(record) for record in caplog.records)
    assert [text for text in (QUERY, MEMORY, MODEL) if text in logged] == []


def _expect(logs: pytest.LogCaptureFixture, *diagnostics: str) -> None:
    missing = [diagnostic for diagnostic in diagnostics if diagnostic not in logs.text]
    assert missing == [], logs.text
    _assert_no_text(logs)


def _failure(text: str, status: int | None = None) -> RuntimeError:
    error = RuntimeError(f"request for {text!r} failed")
    if status is not None:
        error.status_code = status
    return error


# --- episodes -------------------------------------------------------------------------------------------------


def _episodes_answer() -> dict[str, Any]:
    return {
        "episodes": [
            {**_episode(summary=MODEL), "title": ""},
            {**_episode(title=MODEL), "summary": "   "},
            {key: value for key, value in _episode(title=MODEL, summary=MEMORY).items() if key != "events"},
            MODEL,
            {**_episode(title=MODEL, summary=MEMORY), "participants": MODEL},
            _episode(title="Valid episode", summary="The valid episode persisted."),
        ]
    }


_EPISODE_DIAGNOSTICS = (
    "(missing title) user_id=u1 thread_id=t1 index=0 title=string(chars=0, blank)",
    "(missing summary) user_id=u1 thread_id=t1 index=1 summary=string(chars=3, blank)",
    "(missing events) user_id=u1 thread_id=t1 index=2 events=null",
    f"user_id=u1 thread_id=t1 index=3 episode=string(chars={len(MODEL)})",
    "index=4 err=pydantic_core.ValidationError(model=EpisodicRecord, errors=1, fields=participants:list_type)",
)


def test_malformed_episodes_are_logged_by_shape(logs: pytest.LogCaptureFixture, monkeypatch) -> None:
    monkeypatch.setenv("EPISODE_TOPIC_DRIFT", "0")
    service, store, _ = _episode_service([_episodes_answer()])

    assert service.extract_episodes("u1", "t1", flush=True) == {"episodes": 1}
    _expect(logs, *_EPISODE_DIAGNOSTICS)


async def test_async_malformed_episodes_are_logged_by_shape(logs: pytest.LogCaptureFixture, monkeypatch) -> None:
    monkeypatch.setenv("EPISODE_TOPIC_DRIFT", "0")
    service, store, _ = _async_episode_service([_episodes_answer()])

    assert await service.extract_episodes("u1", "t1", flush=True) == {"episodes": 1}
    _expect(logs, *_EPISODE_DIAGNOSTICS)


@pytest.mark.parametrize("asynchronous", [False, True])
async def test_a_malformed_episode_answer_is_logged_by_shape(
    logs: pytest.LogCaptureFixture, asynchronous: bool
) -> None:
    make = _async_episode_service if asynchronous else _episode_service
    service, _, _ = make([{"episodes": {"title": MODEL, "summary": MEMORY}}])

    docs = service._build_episode_docs("u1", "t1", [_turn(1), _turn(2)], segment_key="seg-1")

    assert (await docs if asynchronous else docs) == []
    _expect(logs, "dropping malformed response user_id=u1 thread_id=t1 episodes=object(keys=2)")


def _boundary_turns() -> list[dict[str, Any]]:
    return [_turn_at(1, 1), _turn_at(2, 2), _turn_at(3, 30)]


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize(
    ("error", "diagnostic"),
    [
        pytest.param(
            RuntimeError(f"transient rate limit 429 answering {MODEL!r}"),
            "extract_episodes: deferring 2 turns after retryable extraction error (will retry next run) "
            "user_id=u1 thread_id=t1 err=RuntimeError at ",
            id="retryable",
        ),
        pytest.param(
            RuntimeError(f"content_filter triggered by {MODEL!r}"),
            "extract_episodes: quarantining 2 turns after non-retryable extraction error "
            "(advancing the episode watermark past them so they do not re-poison future runs) "
            "user_id=u1 thread_id=t1 err=RuntimeError at ",
            id="non-retryable",
        ),
    ],
)
async def test_a_failed_episode_segment_logs_its_error_type_not_its_text(
    logs: pytest.LogCaptureFixture, monkeypatch, asynchronous: bool, error: Exception, diagnostic: str
) -> None:
    monkeypatch.setenv("EPISODE_IDLE_GAP_SECONDS", "120")
    monkeypatch.setenv("EPISODE_TOPIC_DRIFT", "0")
    service, *_ = (_async_boundary_service if asynchronous else _boundary_service)(_boundary_turns())
    service._run_prompty = AsyncMock(side_effect=error) if asynchronous else MagicMock(side_effect=error)

    result = service.extract_episodes("u1", "t1")

    assert (await result if asynchronous else result) == {"episodes": 0}
    _expect(logs, diagnostic)


@pytest.mark.parametrize("asynchronous", [False, True])
async def test_a_failed_drift_embedding_logs_the_error_type_not_its_text(
    logs: pytest.LogCaptureFixture, monkeypatch, asynchronous: bool
) -> None:
    monkeypatch.setenv("EPISODE_TOPIC_DRIFT", "0.5")
    embeddings = MagicMock()
    failure = RuntimeError(f"embedding {MEMORY!r} failed")
    embeddings.generate_batch = AsyncMock(side_effect=failure) if asynchronous else MagicMock(side_effect=failure)
    service, *_ = (_async_boundary_service if asynchronous else _boundary_service)(
        _boundary_turns(), embeddings=embeddings
    )

    result = service._episode_segment_embeddings([_turn(1), _turn(2)])

    assert (await result if asynchronous else result) == []
    _expect(logs, "episode drift embedding failed (RuntimeError at ")


class _FailingStamp(_PatchTurnsContainer):
    def patch_item(self, *, item, partition_key, patch_operations):
        raise _failure(MEMORY, 503)


class _AsyncFailingStamp(_AsyncPatchTurnsContainer):
    async def patch_item(self, *, item, partition_key, patch_operations):
        raise _failure(MEMORY, 503)


_STAMP_DIAGNOSTIC = "_mark_turns_extracted failed for turn_id=turn-1 err=RuntimeError(status=503) at "


def test_a_failed_turn_stamp_logs_the_status_not_the_text(logs: pytest.LogCaptureFixture) -> None:
    service = _mark_service(_FailingStamp([_turn_doc()]))

    assert service._mark_turns_extracted([_turn_doc()]) == 0
    _expect(logs, _STAMP_DIAGNOSTIC)


async def test_async_failed_turn_stamp_logs_the_status_not_the_text(logs: pytest.LogCaptureFixture) -> None:
    service = _async_mark_service(_AsyncFailingStamp([_turn_doc()]))

    assert await service._mark_turns_extracted([_turn_doc()]) == 0
    _expect(logs, _STAMP_DIAGNOSTIC)


# --- fact extraction ------------------------------------------------------------------------------------------

_BATCH_FAILURES = [
    pytest.param(
        Exception(f"Error code: 429 rate limit; the model answered {MODEL!r}"),
        "deferred_turn_count",
        "deferring 3 turns after retryable extraction error (will retry next run) user_id=u1 thread_id=t1 err=",
        id="retryable",
    ),
    pytest.param(
        Exception(f"Error code: 400 content_filter; the model answered {MODEL!r}"),
        "quarantined_turn_count",
        "quarantining 3 turns after non-retryable extraction error",
        id="non-retryable",
    ),
]


@pytest.mark.parametrize(("error", "count", "diagnostic"), _BATCH_FAILURES)
def test_a_failed_batch_logs_its_error_type_not_its_text(
    logs: pytest.LogCaptureFixture, error: Exception, count: str, diagnostic: str
) -> None:
    memories = _Store([])
    service = PipelineService(
        memories,
        _BatchChat(fail_on_call={1, 2}, error=error),
        _SyncEmbeddings(),
        containers=_containers_for_store(memories, turns_store=_Store([_turn(i) for i in range(3)])),
    )

    assert service.extract_memories("u1", "t1")[count] == 3
    _expect(logs, diagnostic, "Exception at ")


@pytest.mark.parametrize(("error", "count", "diagnostic"), _BATCH_FAILURES)
async def test_async_failed_batch_logs_its_error_type_not_its_text(
    logs: pytest.LogCaptureFixture, error: Exception, count: str, diagnostic: str
) -> None:
    memories = _AsyncStore([])
    service = AsyncPipelineService(
        memories,
        _AsyncBatchChat(fail_on_call={1, 2}, error=error),
        _AsyncEmbeddings(),
        containers=_async_containers_for_store(memories, turns_store=_AsyncStore([_turn(i) for i in range(3)])),
    )

    assert (await service.extract_memories("u1", "t1"))[count] == 3
    _expect(logs, diagnostic, "Exception at ")


def test_trailing_model_output_is_counted_not_logged(logs: pytest.LogCaptureFixture) -> None:
    assert parse_llm_json(f'{{"facts": [1]}} {MODEL} trailing') == {"facts": [1]}

    _expect(logs, "chars of non-JSON trailing data after 1 concatenated JSON object(s)")


# --- reconcile ------------------------------------------------------------------------------------------------


def _fact(fact_id: str) -> dict[str, Any]:
    return {"id": fact_id, "user_id": "u1", "thread_id": "t1", "type": "fact", "content": f"fact {fact_id}"}


def _contradictions_answer() -> str:
    return json.dumps(
        {
            "duplicate_groups": [],
            "contradicted_pairs": [
                {"winner_id": MODEL, "loser_id": "f1", "reason": MEMORY},
                {"winner_id": "f2", "loser_id": "f1", "reason": MEMORY},
                {"winner_id": "f3", "loser_id": "f1", "reason": MEMORY},
            ],
            "kept_ids": ["f1", "f2", "f3"],
        }
    )


_RECONCILE_DIAGNOSTICS = (
    f"reconcile_memories: hallucinated winner_id=string(chars={len(MODEL)}) not in pool; skipping pair object(keys=3)",
    "reconcile_memories: skipping chained/duplicate contradiction pair object(keys=3) "
    "(winner or loser already superseded this pass)",
)


def test_contradiction_pairs_the_model_got_wrong_are_logged_by_shape(logs: pytest.LogCaptureFixture) -> None:
    service = PipelineService.__new__(PipelineService)
    service._run_prompty = MagicMock(return_value=_contradictions_answer())
    service._mark_superseded = MagicMock(return_value=True)

    result = service._reconcile_contradictions("u1", "fact", [_fact("f1"), _fact("f2"), _fact("f3")])

    assert result["contradicted"] == 1
    _expect(logs, *_RECONCILE_DIAGNOSTICS)


async def test_async_contradiction_pairs_the_model_got_wrong_are_logged_by_shape(
    logs: pytest.LogCaptureFixture,
) -> None:
    service = AsyncPipelineService.__new__(AsyncPipelineService)
    service._run_prompty = AsyncMock(return_value=_contradictions_answer())
    service._mark_superseded = AsyncMock(return_value=True)

    result = await service._reconcile_contradictions("u1", "fact", [_fact("f1"), _fact("f2"), _fact("f3")])

    assert result["contradicted"] == 1
    _expect(logs, *_RECONCILE_DIAGNOSTICS)
