"""Tests for structured logging and JsonFormatter."""

from __future__ import annotations

import io
import json
import logging

import pytest

from agent_memory_toolkit.exceptions import MemoryNotFoundError, ValidationError
from agent_memory_toolkit.logging import (
    JsonFormatter,
    configure_logging,
    get_logger,
)


@pytest.fixture(autouse=True)
def _reset_root_logger():
    root = logging.getLogger("agent_memory_toolkit")
    saved_handlers = list(root.handlers)
    saved_level = root.level
    for h in list(root.handlers):
        root.removeHandler(h)
    root.setLevel(logging.NOTSET)
    yield
    for h in list(root.handlers):
        root.removeHandler(h)
    for h in saved_handlers:
        root.addHandler(h)
    root.setLevel(saved_level)


def _render(record: logging.LogRecord) -> dict:
    return json.loads(JsonFormatter().format(record))


class TestJsonFormatter:
    def _make_record(self, **extra) -> logging.LogRecord:
        record = logging.LogRecord(
            name="agent_memory_toolkit.test",
            level=logging.INFO,
            pathname=__file__,
            lineno=1,
            msg="hello %s",
            args=("world",),
            exc_info=None,
        )
        for k, v in extra.items():
            setattr(record, k, v)
        return record

    def test_minimal_payload_shape(self):
        payload = _render(self._make_record())
        assert payload["level"] == "INFO"
        assert payload["logger"] == "agent_memory_toolkit.test"
        assert payload["msg"] == "hello world"
        assert "ts" in payload
        # No correlation_id in the payload at all.
        assert "correlation_id" not in payload

    def test_optional_extra_keys_included_when_present(self):
        payload = _render(
            self._make_record(
                user_id="u1",
                thread_id="t1",
                operation="reconcile_memories",
                latency_ms=42.5,
                ru_charge=1.23,
                prompt_id="dedup.prompty",
                prompt_version="v1",
            )
        )
        assert payload["user_id"] == "u1"
        assert payload["thread_id"] == "t1"
        assert payload["operation"] == "reconcile_memories"
        assert payload["latency_ms"] == 42.5
        assert payload["ru_charge"] == 1.23
        assert payload["prompt_id"] == "dedup.prompty"
        assert payload["prompt_version"] == "v1"

    def test_optional_keys_omitted_when_none(self):
        payload = _render(self._make_record(user_id=None, latency_ms=None))
        assert "user_id" not in payload
        assert "latency_ms" not in payload

    def test_exception_info_serialised(self):
        try:
            raise ValidationError("bad input")
        except ValidationError:
            import sys

            record = logging.LogRecord(
                name="agent_memory_toolkit.test",
                level=logging.ERROR,
                pathname=__file__,
                lineno=1,
                msg="boom",
                args=(),
                exc_info=sys.exc_info(),
            )
        payload = _render(record)
        assert payload["exc_type"] == "ValidationError"
        assert payload["exc_msg"] == "bad input"
        assert payload["error_code"] == "validation"

    def test_error_code_from_agent_memory_error_subclass(self):
        try:
            raise MemoryNotFoundError(memory_id="m-1")
        except MemoryNotFoundError:
            import sys

            record = logging.LogRecord(
                name="agent_memory_toolkit.test",
                level=logging.ERROR,
                pathname=__file__,
                lineno=1,
                msg="not found",
                args=(),
                exc_info=sys.exc_info(),
            )
        payload = _render(record)
        assert payload["error_code"] == "memory_not_found"

    def test_single_line_output(self):
        line = JsonFormatter().format(self._make_record())
        assert "\n" not in line
        json.loads(line)


class TestConfigureLogging:
    def test_attaches_single_handler(self):
        configure_logging(force_json=True)
        root = logging.getLogger("agent_memory_toolkit")
        managed = [h for h in root.handlers if getattr(h, "_amt_managed", False)]
        assert len(managed) == 1

    def test_is_idempotent(self):
        configure_logging(force_json=True)
        configure_logging(force_json=True)
        configure_logging(force_json=True)
        root = logging.getLogger("agent_memory_toolkit")
        managed = [h for h in root.handlers if getattr(h, "_amt_managed", False)]
        assert len(managed) == 1

    def test_force_json_uses_json_formatter(self):
        configure_logging(force_json=True)
        root = logging.getLogger("agent_memory_toolkit")
        managed = [h for h in root.handlers if getattr(h, "_amt_managed", False)]
        assert isinstance(managed[0].formatter, JsonFormatter)

    def test_get_logger_participates_in_root(self):
        logger = get_logger("agent_memory_toolkit.some.submodule")
        buf = io.StringIO()
        capture = logging.StreamHandler(buf)
        capture.setFormatter(JsonFormatter())
        capture._amt_managed = True  # type: ignore[attr-defined]
        root = logging.getLogger("agent_memory_toolkit")
        for h in list(root.handlers):
            root.removeHandler(h)
        root.addHandler(capture)
        root.setLevel(logging.INFO)
        logger.info("ping", extra={"operation": "test_op", "user_id": "u9"})
        payload = json.loads(buf.getvalue().strip())
        assert payload["msg"] == "ping"
        assert payload["operation"] == "test_op"
        assert payload["user_id"] == "u9"
        assert payload["logger"] == "agent_memory_toolkit.some.submodule"
