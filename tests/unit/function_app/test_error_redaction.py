"""Activity failures reach Durable Functions and the log with a safe code and the exception type, not the error's text.

Each failure carries a distinctive string in its message, arguments or (for a validation error) its input; what Durable
would record and the captured log, including any traceback, must not contain it.
"""

from __future__ import annotations

import ast
import inspect
import json
import logging
import traceback
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from orchestrators import thread_summary as ts_mod
from pydantic import BaseModel
from shared.error_redaction import ActivityFailedError, redact_errors, safe_error_code

from azure.cosmos.agent_memory.exceptions import (
    CosmosOperationError,
    LLMError,
    NoSourceMemoriesError,
    ValidationError,
)

SECRET = "zx-memory-ana-allergic-to-cashews-4410"
ORCHESTRATORS = Path(__file__).resolve().parents[3] / "function_app" / "orchestrators"


class _Fact(BaseModel):
    salience: float


@redact_errors
def fa_Extract(payload: dict) -> dict:
    if payload.get("ok"):
        return {"facts": 1}
    answer = SECRET  # noqa: F841 - a local in the failing frame
    raise LLMError(f"LLM returned invalid JSON (preview={SECRET!r})")


@redact_errors
async def fa_Persist(payload: dict) -> dict:
    if payload.get("ok"):
        return {"persisted": True}
    _Fact(salience=SECRET)


def _http_error(status: int) -> Exception:
    error = Exception(f"HTTP {status} for {SECRET!r}")
    error.status_code = status
    return error


def _caused_by(cause: Exception) -> CosmosOperationError:
    try:
        raise CosmosOperationError(f"upsert failed for {SECRET!r}") from cause
    except CosmosOperationError as wrapped:
        return wrapped


def _host_log(error: BaseException) -> str:
    """What the Functions host records of an unhandled error: its traceback with every error it was raised from."""
    return "".join(traceback.format_exception(error))


@pytest.mark.parametrize(
    ("error", "code"),
    [
        (_http_error(412), "write_conflict"),
        (_http_error(404), "memory_not_found"),
        (_http_error(503), "dependency_failure"),
        (_caused_by(_http_error(412)), "write_conflict"),
        (_caused_by(RuntimeError(SECRET)), "cosmos_operation"),
        (ValidationError(SECRET), "validation"),
        (NoSourceMemoriesError(SECRET), "no_source_memories"),
        (LLMError(SECRET), "llm"),
        (KeyError(SECRET), "dependency_failure"),
    ],
)
def test_the_safe_code_comes_from_the_status_or_the_exception_type(error, code):
    assert safe_error_code(error) == code


def test_a_cause_cycle_does_not_hang_the_code_lookup():
    first, second = RuntimeError("a"), RuntimeError("b")
    first.__cause__, second.__cause__ = second, first

    assert safe_error_code(first) == "dependency_failure"


def test_a_failed_activity_raises_only_its_code_and_type(caplog):
    caplog.set_level(logging.DEBUG)

    with pytest.raises(ActivityFailedError) as failure:
        fa_Extract({})

    assert failure.value.args == ("llm (LLMError)",)
    assert (failure.value.__cause__, failure.value.__context__) == (None, None)
    assert SECRET not in _host_log(failure.value) + caplog.text


def test_the_log_names_the_activity_code_type_and_locations_but_not_the_text(caplog):
    caplog.set_level(logging.DEBUG)

    with pytest.raises(ActivityFailedError):
        fa_Extract({})

    [record] = [record for record in caplog.records if record.levelno >= logging.WARNING]
    assert record.exc_info is None
    message = record.getMessage()
    assert message.startswith("Activity fa_Extract failed error_code=llm: agent_memory.LLMError(code=llm) at ")
    assert "function_app/shared/error_redaction.py:" in message and "test_error_redaction.py:" in message
    assert SECRET not in caplog.text


async def test_a_failed_async_activity_raises_only_its_code_and_type(caplog):
    caplog.set_level(logging.DEBUG)

    with pytest.raises(ActivityFailedError) as failure:
        await fa_Persist({})

    assert failure.value.args == ("dependency_failure (ValidationError)",)
    assert (failure.value.__cause__, failure.value.__context__) == (None, None)
    assert "fields=salience:float_parsing" in caplog.text
    assert SECRET not in _host_log(failure.value) + caplog.text


async def test_a_successful_activity_returns_its_result_unchanged():
    assert fa_Extract({"ok": True}) == {"facts": 1}
    assert await fa_Persist({"ok": True}) == {"persisted": True}


def test_the_host_still_sees_the_activity_it_registered():
    assert fa_Extract.__name__ == "fa_Extract" and list(inspect.signature(fa_Extract).parameters) == ["payload"]
    assert inspect.iscoroutinefunction(fa_Persist) and not inspect.iscoroutinefunction(fa_Extract)
    assert fa_Extract(payload={"ok": True}) == {"facts": 1}


def test_an_exception_that_is_not_an_error_passes_through():
    @redact_errors
    def interrupted(payload: dict) -> None:
        raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        interrupted({})


class _Context:
    """Durable context double; ``history`` holds what Durable would record."""

    is_replaying = False

    def __init__(self, payload):
        self.payload = payload
        self.history = [payload]

    def get_input(self):
        return self.payload

    def call_activity_with_retry(self, name, retry, payload):
        return name, payload


def _user_function(builder):
    if hasattr(builder, "_function"):
        return builder._function.get_user_function().orchestrator_function
    return builder


def _run(orchestrator, context, activities):
    """Drive the orchestrator, running each activity it schedules and recording what Durable would."""
    generator = orchestrator(context)
    result, error = None, None
    while True:
        try:
            name, payload = generator.throw(error) if error else generator.send(result)
        except StopIteration as stopped:
            return stopped.value
        context.history.append({"activity": name, "input": payload})
        try:
            result, error = activities[name](payload), None
            context.history.append({"activity": name, "output": result})
        except Exception as exc:
            result, error = None, exc
            context.history.append({"activity": name, "error": str(exc)})


def test_a_failed_summary_step_keeps_its_error_text_out_of_the_history_and_the_log(monkeypatch, caplog):
    pipeline = MagicMock()
    pipeline.generate_thread_summary_durable.side_effect = LLMError(f"LLM returned invalid JSON (preview={SECRET!r})")
    monkeypatch.setattr(ts_mod, "get_pipeline", lambda: pipeline)
    monkeypatch.setattr(ts_mod, "default_retry_options", lambda: "retry")
    context = _Context({"user_id": "alice", "thread_id": "t1"})
    activities = {"ts_Extract": ts_mod.ts_Extract, "ts_PersistSummary": ts_mod.ts_PersistSummary}

    with caplog.at_level(logging.DEBUG), pytest.raises(ActivityFailedError) as failure:
        _run(_user_function(ts_mod.ThreadSummaryOrchestrator), context, activities)

    assert context.history[-1] == {"activity": "ts_Extract", "error": "llm (LLMError)"}
    assert "agent_memory.LLMError(code=llm) at " in caplog.text
    assert SECRET not in json.dumps(context.history) + str(failure.value) + caplog.text


def test_every_activity_hides_its_failure_text():
    """``@redact_errors`` sits directly under ``@bp.activity_trigger`` on every activity, so the host registers the
    wrapper (which keeps the activity's name and signature)."""
    activities, unprotected = [], []
    for path in sorted(ORCHESTRATORS.glob("*.py")):
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            decorators = [ast.unparse(d.func if isinstance(d, ast.Call) else d) for d in node.decorator_list]
            if "bp.activity_trigger" not in decorators:
                continue
            activities.append(node.name)
            expected_position = decorators.index("bp.activity_trigger") + 1
            if decorators[expected_position : expected_position + 1] != ["redact_errors"]:
                unprotected.append(f"{path.name}:{node.name}")

    assert activities and unprotected == []
