"""Structured logging for the Agent Memory Toolkit.

Provides a JSON log formatter, an idempotent :func:`configure_logging`
that attaches a single handler to the ``azure.cosmos.agent_memory`` root logger,
and log-safe descriptions of exceptions and model output.

An exception's message and arguments can quote a search query, memory text
or model output, and a model's answer is that output. Logs get them only
through :func:`describe_exception`, :func:`describe_shape` and
:func:`describe_turn_refs`, which keep bounded, code-defined facts.
"""

from __future__ import annotations

import json
import logging
import re
import sys
import sysconfig
import traceback
from collections.abc import Iterable, Mapping
from datetime import datetime, timezone
from pathlib import PurePath
from types import TracebackType
from typing import Any, Optional

from azure.cosmos.agent_memory.exceptions import AgentMemoryError

_ROOT_LOGGER_NAME = "azure.cosmos.agent_memory"

_MAX_CHAIN = 3
_MAX_FRAMES = 8
_MAX_FIELDS = 5
_MAX_REFS = 5
_MAX_TEXT = 2000
_TOKEN = re.compile(r"[A-Za-z0-9_.:-]{1,64}")
_FIELD = re.compile(r"[A-Za-z_][A-Za-z0-9_]{0,63}")
_TURN_LABEL = re.compile(r"turn-\d{1,6}(?:\.\d{1,3})?", re.IGNORECASE)
_STDLIB = tuple({path for path in (sysconfig.get_path("stdlib"), sysconfig.get_path("platstdlib")) if path})

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
                # Never the message: it can quote a query, memory text or model output.
                payload["exc_detail"] = describe_exception(exc_value, stack=True)
                if (
                    "error_code" not in payload
                    and isinstance(exc_value, AgentMemoryError)
                    and getattr(exc_value, "error_code", "")
                ):
                    payload["error_code"] = exc_value.error_code

        return json.dumps(payload, ensure_ascii=False, default=str)


def describe_exception(exc: BaseException, *, stack: bool = False) -> str:
    """Describe ``exc`` for a log without its message, arguments or local variables.

    Gives its type; its error code, HTTP status and sub-status, JSON decode
    position, or validation model, error count and failing fields with their
    error types; and where it was raised: the raise point and the nearest frame
    of our own code, or with ``stack`` up to eight frames of our code. Up to
    two exceptions it was raised from follow, each with its raise point.
    """
    try:
        parts: list[str] = []
        seen: set[int] = set()
        current: Optional[BaseException] = exc
        while current is not None and id(current) not in seen and len(parts) < _MAX_CHAIN:
            seen.add(id(current))
            parts.append(_describe_one(current, frames=_MAX_FRAMES if stack and not parts else 2))
            current = current.__cause__ or (None if current.__suppress_context__ else current.__context__)
        return " <- ".join(parts)[:_MAX_TEXT]
    except Exception:  # noqa: BLE001 - describing a failure must never raise
        return _type_name(exc)


def describe_shape(value: Any) -> str:
    """Describe a piece of model output for a log by its JSON shape, never its text.

    An object reports its key count; a string its length and whether it is
    blank; an array its length.
    """
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, (int, float)):
        return "number"
    if isinstance(value, str):
        return f"string(chars={len(value)}{', blank' if not value.strip() else ''})"
    if isinstance(value, Mapping):
        return f"object(keys={len(value)})"
    if isinstance(value, (list, tuple)):
        return f"array(len={len(value)})"
    return type(value).__name__


def describe_turn_refs(refs: Iterable[Any]) -> str:
    """Describe model-written turn citations for a log.

    A positional ``turn-N`` label is shown as written; any other reference
    (a turn ID or text) only by its shape.
    """
    refs = list(refs)
    shown = [
        ref.strip() if isinstance(ref, str) and _TURN_LABEL.fullmatch(ref.strip()) else describe_shape(ref)
        for ref in refs[:_MAX_REFS]
    ]
    return f"{len(refs)} [{', '.join(shown)}]"


def _describe_one(exc: BaseException, *, frames: int) -> str:
    facts = _exception_facts(exc)
    text = _type_name(exc) + (f"({', '.join(facts)})" if facts else "")
    where = _frames(exc.__traceback__, frames)
    return f"{text} at {where}" if where else text


def _type_name(exc: BaseException) -> str:
    cls = type(exc)
    module = getattr(cls, "__module__", "") or ""
    if module in ("", "builtins"):
        return cls.__qualname__
    if module == _ROOT_LOGGER_NAME or module.startswith(f"{_ROOT_LOGGER_NAME}."):
        package = "agent_memory"
    else:
        segments = module.split(".")
        package = ".".join(segments[:2]) if segments[0] == "azure" and len(segments) > 1 else segments[0]
    return f"{package}.{cls.__qualname__}"


def _exception_facts(exc: BaseException) -> list[str]:
    facts: list[str] = []
    code = (
        _token(getattr(exc, "error_code", None))
        or _token(getattr(exc, "code", None))
        or _token(getattr(getattr(exc, "error", None), "code", None))
    )
    if code:
        facts.append(f"code={code}")
    status = _integer(getattr(exc, "status_code", None))
    if status is None:
        status = _integer(getattr(exc, "status", None))
    if status is not None:
        facts.append(f"status={status}")
    sub_status = _integer(getattr(exc, "sub_status", None))
    if sub_status:
        facts.append(f"sub_status={sub_status}")
    if isinstance(exc, json.JSONDecodeError):
        facts.append(f"line={exc.lineno}, col={exc.colno}, pos={exc.pos}")
    facts.extend(_validation_facts(exc))
    return facts


def _validation_facts(exc: BaseException) -> list[str]:
    """A pydantic ``ValidationError``'s model, error count, and failing field paths with error types; never the
    input values or messages, which quote the data that failed."""
    errors_of, count_of = getattr(exc, "errors", None), getattr(exc, "error_count", None)
    if not callable(errors_of) or not callable(count_of):
        return []
    try:
        errors = errors_of(include_url=False, include_context=False, include_input=False)
    except TypeError:
        errors = errors_of()
    facts = [f"errors={int(count_of())}"]
    model = _token(getattr(exc, "title", None))
    if model:
        facts.insert(0, f"model={model}")
    fields = []
    for error in list(errors)[:_MAX_FIELDS]:
        kind = _token(error.get("type")) or "?"
        fields.append(f"{_field_path(error.get('loc'), kind)}:{kind}")
    if fields:
        facts.append(f"fields={' '.join(fields)}")
    return facts


def _field_path(loc: Any, kind: str) -> str:
    """The failing field's top-level name and list positions. A nested name can be a dictionary key, and an
    unexpected key is always one, named by the data itself; they show as ``*``."""
    if kind == "extra_forbidden" or not isinstance(loc, (tuple, list)) or not loc:
        return "*"
    parts = []
    for depth, part in enumerate(list(loc)[:3]):
        if isinstance(part, int) and not isinstance(part, bool):
            parts.append(str(part))
        else:
            parts.append(part if depth == 0 and isinstance(part, str) and _FIELD.fullmatch(part) else "*")
    return ".".join(parts)


def _frames(tb: Optional[TracebackType], limit: int) -> str:
    """The raise point and the innermost frames of our own code (not a dependency or the standard library), up to
    ``limit``, outermost first; ``…`` marks skipped frames."""
    frames = [(frame.f_code.co_filename, lineno, frame.f_code.co_name) for frame, lineno in traceback.walk_tb(tb)]
    if not frames:
        return ""
    ours = [index for index, (path, _, _) in enumerate(frames) if not _is_dependency(path)]
    keep = {len(frames) - 1}
    for index in [*reversed(ours), *reversed(range(len(frames)))]:
        if len(keep) >= limit:
            break
        keep.add(index)
    shown: list[str] = []
    previous = -1
    for index in sorted(keep):
        if shown and index != previous + 1:
            shown.append("…")
        path, lineno, name = frames[index]
        shown.append(f"{'/'.join(PurePath(path).parts[-3:])}:{lineno} {name}")
        previous = index
    return " > ".join(shown)


def _is_dependency(path: str) -> bool:
    parts = PurePath(path).parts
    if "agent_memory" in parts or "function_app" in parts:
        return False
    return "site-packages" in parts or "dist-packages" in parts or path.startswith(("<", *_STDLIB))


def _token(value: Any) -> Optional[str]:
    if isinstance(value, int) and not isinstance(value, bool):
        return str(value)
    return value if isinstance(value, str) and _TOKEN.fullmatch(value) else None


def _integer(value: Any) -> Optional[int]:
    return value if isinstance(value, int) and not isinstance(value, bool) else None


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
    "describe_exception",
    "describe_shape",
    "describe_turn_refs",
    "get_logger",
]
