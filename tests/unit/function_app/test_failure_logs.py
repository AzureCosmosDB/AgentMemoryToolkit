"""Function App failure logs carry a safe code, the exception type and code locations, never the error's text.

Each failure carries a distinctive string in its message; the captured log (as text, with any traceback, and through
the JSON formatter) and what the Functions host records of an unhandled error must not contain it.
"""

from __future__ import annotations

import asyncio
import logging
import traceback
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import triggers.change_feed as change_feed_module
from azure.cosmos.exceptions import CosmosHttpResponseError
from orchestrators import extract_memories as em
from shared.counters import advance_extract_watermark, read_extract_watermark
from triggers.change_feed import process_changefeed_batch

from tests.unit.function_app.test_change_feed import _make_counter_container_starting_at, _make_starter, _turn
from tests.unit.test_log_redaction import MEMORY, _assert_no_text, _expect


@pytest.fixture
def logs(caplog: pytest.LogCaptureFixture) -> pytest.LogCaptureFixture:
    caplog.set_level(logging.DEBUG)
    return caplog


def _failure(status: int) -> Exception:
    error = Exception(f"request for {MEMORY!r} failed")
    error.status_code = status
    return error


def _host_log(error: BaseException) -> str:
    """What the Functions host records of an unhandled error: its traceback with every error it was raised from."""
    return "".join(traceback.format_exception(error))


async def test_watermark_failures_log_the_status_not_the_text(logs) -> None:
    container = MagicMock()
    container.read_item = AsyncMock(side_effect=_failure(503))
    container.patch_item = AsyncMock(
        side_effect=[_failure(503), CosmosHttpResponseError(status_code=500, message=MEMORY)]
    )

    assert await read_extract_watermark(container, "thread:u1:t1", "u1", "t1") is None
    await advance_extract_watermark(container, "thread:u1:t1", "u1", "t1", 4)
    await advance_extract_watermark(container, "thread:u1:t1", "u1", "t1", 4)

    expected = (
        "advance_extract_watermark failed counter_id=thread:u1:t1: azure.cosmos.CosmosHttpResponseError(status=500) at "
    )
    _expect(
        logs,
        "read_extract_watermark failed counter_id=thread:u1:t1: Exception(status=503) at ",
        "advance_extract_watermark failed counter_id=thread:u1:t1: Exception(status=503) at ",
        expected,
    )


class _Context:
    is_replaying = False

    def __init__(self, payload: dict) -> None:
        self.payload = payload

    def get_input(self) -> dict:
        return self.payload

    def call_activity_with_retry(self, name, retry, payload):
        return name, payload

    def call_sub_orchestrator_with_retry(self, name, retry, payload, instance_id=None):
        return name, payload


def _user_function(builder):
    if hasattr(builder, "_function"):
        return builder._function.get_user_function().orchestrator_function
    return builder


def test_a_failed_sub_orchestration_logs_its_type_not_its_text(logs) -> None:
    context = _Context({"user_id": "u1", "thread_id": "t1", "reconcile": True})

    with patch.object(em, "default_retry_options", return_value=MagicMock()):
        generator = _user_function(em.ExtractMemoriesOrchestrator)(context)
        next(generator)
        generator.send({"facts": []})
        generator.send({"fact_count": 0})
        generator.send({})
        with pytest.raises(StopIteration):
            generator.throw(RuntimeError(f"SynthesizeProceduralOrchestrator failed on {MEMORY!r}"))

    [record] = [
        record for record in logs.records if record.name == em.logger.name and record.levelno >= logging.WARNING
    ]
    assert record.exc_info is None
    assert record.getMessage().startswith("SynthesizeProceduralOrchestrator failed user=u1 thread=t1: RuntimeError at ")
    _assert_no_text(logs)


def _failing_start_environment(monkeypatch) -> MagicMock:
    monkeypatch.setenv("MEMORY_PROCESSOR_OWNER", "durable")
    monkeypatch.setattr(change_feed_module, "_topology_validated", True)
    for name, every_n in (("THREAD_SUMMARY_EVERY_N", "4"), ("FACT_EXTRACTION_EVERY_N", "4")):
        monkeypatch.setenv(name, every_n)
    for name in ("EPISODE_EVAL_EVERY_N", "USER_SUMMARY_EVERY_N"):
        monkeypatch.setenv(name, "0")
    starter = _make_starter()
    starter.start_new.side_effect = _failure(503)
    return starter


def test_a_failed_orchestration_start_logs_its_status_and_location_not_its_text(logs, monkeypatch) -> None:
    starter = _failing_start_environment(monkeypatch)

    with pytest.raises(RuntimeError) as failed:
        asyncio.run(
            process_changefeed_batch(
                [_turn(lsn=lsn) for lsn in range(10, 14)],
                starter,
                counter_container=_make_counter_container_starting_at(),
            )
        )

    records = [record for record in logs.records if "change-feed: failed to start" in record.getMessage()]
    assert len(records) == 2 and all(record.exc_info is None for record in records)
    assert all("(exc_type=Exception): Exception(status=503) at " in record.getMessage() for record in records)
    # The batch still fails, so the host retries it; what the host logs of that failure names the error safely.
    assert str(failed.value).startswith("Failed to start 2 orchestration(s): Exception(status=503) at ")
    assert (failed.value.__cause__, failed.value.__context__) == (None, None)
    assert MEMORY not in _host_log(failed.value)
    _assert_no_text(logs)


def test_an_unreadable_container_fails_the_trigger_without_the_sdk_text(monkeypatch) -> None:
    container = MagicMock()
    container.read = AsyncMock(side_effect=_failure(403))
    database = MagicMock()
    database.get_container_client.return_value = container
    monkeypatch.setattr(change_feed_module, "_topology_validated", False)
    monkeypatch.setattr("shared.cosmos_clients.get_cosmos_database_async", AsyncMock(return_value=database))

    with pytest.raises(RuntimeError) as failed:
        asyncio.run(change_feed_module._ensure_topology())

    assert str(failed.value).startswith("Topology error: cannot read container ")
    assert ": Exception(status=403) at " in str(failed.value)
    assert (failed.value.__cause__, failed.value.__context__) == (None, None)
    assert MEMORY not in _host_log(failed.value)
