"""No log call prints an exception's text.

An exception's message and arguments can quote a search query, memory text or model output, and a traceback prints
them. Logs describe a failure with ``describe_exception`` (its type, status, code and code locations) instead, and
so does the failure reason a counter stores. The last test scans the SDK and Function App sources so no log call or
stored failure reason prints exception text again; the others pin what the scan flags.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
PACKAGES = ("azure/cosmos/agent_memory", "function_app")

_LOG_METHODS = {"debug", "info", "warning", "error", "exception", "critical", "log"}
_FAILURE_STAMPS = {"stamp_failure_sync", "stamp_failure_async"}
_MESSAGE_ATTRIBUTES = {"args", "message", "msg", "body", "text", "response", "reason", "detail", "details"}


def _is_log_call(call: ast.Call) -> bool:
    return (
        isinstance(call.func, ast.Attribute)
        and call.func.attr in _LOG_METHODS
        and isinstance(call.func.value, ast.Name)
        and call.func.value.id.lstrip("_").lower() in ("logger", "log", "logging")
    )


def _is_failure_stamp(call: ast.Call) -> bool:
    name = call.func.attr if isinstance(call.func, ast.Attribute) else getattr(call.func, "id", None)
    return name in _FAILURE_STAMPS


def _exception_text_in_logs(source: str, name: str) -> list[str]:
    """Log calls that would print an exception's text: the exception itself, its ``str``/``repr``/message
    attributes, an f-string of it, or a traceback (``exc_info``, ``stack_info``, ``logger.exception``). The same
    goes for the failure reason a counter stores."""
    tree = ast.parse(source)
    bound: set[str] = {node.name for node in ast.walk(tree) if isinstance(node, ast.ExceptHandler) and node.name}
    bound |= {
        node.target.id
        for node in ast.walk(tree)
        if isinstance(node, ast.NamedExpr)
        and isinstance(node.value, ast.Call)
        and isinstance(node.value.func, ast.Attribute)
        and node.value.func.attr == "exception"
    }
    found = []
    for call in ast.walk(tree):
        if not isinstance(call, ast.Call):
            continue
        logged = _is_log_call(call)
        if not (logged or _is_failure_stamp(call)):
            continue
        where = f"{name}:{call.lineno}"
        if logged and (
            call.func.attr == "exception" or any(k.arg in ("exc_info", "stack_info") for k in call.keywords)
        ):
            found.append(f"{where} logs a traceback")
        for index, argument in enumerate(call.args):
            if index == 0 and isinstance(argument, ast.Constant):
                continue
            for node in _unwrapped(argument):
                if isinstance(node, ast.Name) and node.id in bound:
                    found.append(f"{where} {'logs' if logged else 'stores'} exception {node.id!r}")
    return found


def _unwrapped(node: ast.AST) -> list[ast.AST]:
    """``node`` and its parts, except inside ``describe_exception(...)``, ``type(...)`` or a safe attribute."""
    if (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id in ("describe_exception", "type")
    ):
        return []
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "getattr":
        return _unwrapped(node.args[0]) if len(node.args) > 1 and _is_message(node.args[1]) else []
    if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name):
        return [node.value] if node.attr in _MESSAGE_ATTRIBUTES else []
    return [node, *(part for child in ast.iter_child_nodes(node) for part in _unwrapped(child))]


def _is_message(node: ast.AST) -> bool:
    return not (isinstance(node, ast.Constant) and node.value not in _MESSAGE_ATTRIBUTES)


def _in_handler(statement: str) -> str:
    return f"try:\n    work()\nexcept Exception as exc:\n    {statement}\n"


@pytest.mark.parametrize(
    "statement",
    [
        'logger.warning("failed: %s", exc)',
        'logger.warning("failed: %r", exc)',
        'logger.warning("failed: %s", str(exc))',
        'logger.warning("failed: %s", exc.args)',
        'logger.warning("failed: %s", exc.message)',
        'logger.warning(f"failed: {exc}")',
        'logger.warning("failed: %s", getattr(exc, "message", None))',
        'logger.warning("failed", exc_info=True)',
        'logger.warning("failed", stack_info=True)',
        'logger.exception("failed")',
    ],
)
def test_the_scan_flags_a_log_call_that_prints_exception_text(statement: str) -> None:
    assert len(_exception_text_in_logs(_in_handler(statement), "sample.py")) == 1


@pytest.mark.parametrize(
    "statement",
    [
        'logger.warning("failed: %s", describe_exception(exc))',
        'logger.warning("failed: %s", describe_exception(exc, stack=True))',
        'logger.warning("failed: %s", type(exc).__name__)',
        'logger.warning("failed: %s", getattr(exc, "status_code", None))',
        'logger.warning("failed: %s", exc.status_code)',
        'logger.warning("failed")',
    ],
)
def test_the_scan_allows_a_description_of_the_error(statement: str) -> None:
    assert _exception_text_in_logs(_in_handler(statement), "sample.py") == []


def test_the_scan_flags_an_exception_taken_from_a_task() -> None:
    source = 'if (failure := task.exception()) is not None:\n    logger.warning("failed: %r", failure)\n'

    assert _exception_text_in_logs(source, "sample.py") == ["sample.py:2 logs exception 'failure'"]


@pytest.mark.parametrize(
    "statement",
    [
        '_counters.stamp_failure_sync(container, "id", "u1", "t1", f"step: {exc!r}")',
        'await _counters.stamp_failure_async(container, "id", "u1", "t1", f"step: {exc}")',
        'stamp_failure_sync(container, "id", "u1", "t1", str(exc))',
    ],
)
def test_the_scan_flags_a_stored_failure_reason_that_holds_exception_text(statement: str) -> None:
    assert _exception_text_in_logs(_in_handler(statement), "sample.py") == ["sample.py:4 stores exception 'exc'"]


def test_the_scan_allows_a_stored_failure_reason_that_describes_the_error() -> None:
    statement = 'error = describe_exception(exc)\n    stamp_failure_sync(container, "id", "u1", "t1", f"step: {error}")'

    assert _exception_text_in_logs(_in_handler(statement), "sample.py") == []


def test_no_log_call_or_stored_failure_reason_prints_exception_text() -> None:
    sources = [path for package in PACKAGES for path in sorted((ROOT / package).rglob("*.py"))]
    assert len(sources) > 50

    found = [
        finding
        for path in sources
        for finding in _exception_text_in_logs(path.read_text(encoding="utf-8"), str(path.relative_to(ROOT)))
    ]

    assert found == []
