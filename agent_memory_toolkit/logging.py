"""Structured logging for the Agent Memory Toolkit.

Provides a JSON log formatter and an idempotent :func:`configure_logging`
that attaches a single handler to the ``agent_memory_toolkit`` root logger.
"""

from __future__ import annotations

import json
import logging
import sys
from datetime import datetime, timezone
from typing import Any

from agent_memory_toolkit.exceptions import AgentMemoryError

_ROOT_LOGGER_NAME = "agent_memory_toolkit"

_OPTIONAL_EXTRA_KEYS: tuple[str, ...] = (
    "user_id",
    "thread_id",
    "operation",
    "latency_ms",
    "ru_charge",
    "error_code",
    "prompt_id",
    "prompt_version",
)


class JsonFormatter(logging.Formatter):
    """Serialize every log record to a fixed JSON shape on a single line."""

    def format(self, record: logging.LogRecord) -> str:
        ts = datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat()
        record_dict = record.__dict__

        payload: dict[str, Any] = {
            "ts": ts,
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }

        for key in _OPTIONAL_EXTRA_KEYS:
            value = record_dict.get(key)
            if value is None:
                continue
            payload[key] = value

        if record.exc_info:
            exc_type, exc_value, _ = record.exc_info
            if exc_type is not None:
                payload["exc_type"] = exc_type.__name__
            if exc_value is not None:
                payload["exc_msg"] = str(exc_value)
                if (
                    "error_code" not in payload
                    and isinstance(exc_value, AgentMemoryError)
                    and getattr(exc_value, "error_code", "")
                ):
                    payload["error_code"] = exc_value.error_code

        return json.dumps(payload, ensure_ascii=False, default=str)


class _AmtHandlerMarker:
    """Marker mixin so :func:`configure_logging` is idempotent."""

    _amt_managed = True


class _AmtStreamHandler(logging.StreamHandler, _AmtHandlerMarker):
    """Stream handler tagged so we can detect prior installs."""


def get_logger(name: str) -> logging.Logger:
    """Return a logger that participates in the toolkit's structured-logging tree."""
    return logging.getLogger(name)


def configure_logging(*, force_json: bool = False) -> None:
    """Idempotently attach a single handler to the toolkit root logger.

    Uses :class:`JsonFormatter` when ``force_json`` is set or the process is
    not attached to a TTY. Falls back to a human-readable formatter when
    running interactively. Repeat calls are no-ops because the handler
    carries an ``_amt_managed`` marker.
    """
    root = logging.getLogger(_ROOT_LOGGER_NAME)
    for existing in root.handlers:
        if getattr(existing, "_amt_managed", False):
            return

    handler = _AmtStreamHandler(stream=sys.stderr)
    stderr = sys.stderr
    is_tty = bool(getattr(stderr, "isatty", lambda: False)())
    if force_json or not is_tty:
        handler.setFormatter(JsonFormatter())
    else:
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
    root.addHandler(handler)


__all__ = [
    "JsonFormatter",
    "configure_logging",
    "get_logger",
]
