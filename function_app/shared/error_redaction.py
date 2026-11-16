"""Failure reporting for Function App activities that keeps exception text out of the Durable history."""

from __future__ import annotations

import functools
import inspect
import logging
from typing import Any, Callable

from azure.cosmos.agent_memory.exceptions import AgentMemoryError
from azure.cosmos.agent_memory.logging import describe_exception

logger = logging.getLogger(__name__)


class ActivityFailedError(RuntimeError):
    """An activity failed. Durable Functions stores this error, so it carries only a safe code and an exception type."""


def safe_error_code(error: BaseException) -> str:
    """A fixed code for ``error``, chosen from the HTTP statuses and exception types along its cause chain, never from
    a message."""
    chain: list[BaseException] = []
    seen: set[int] = set()
    cause: BaseException | None = error
    while cause is not None and id(cause) not in seen:
        seen.add(id(cause))
        chain.append(cause)
        cause = cause.__cause__
    for cause in chain:
        status = getattr(cause, "status_code", None)
        if status == 412:
            return "write_conflict"
        if status == 404:
            return "memory_not_found"
    for cause in chain:
        if isinstance(cause, AgentMemoryError) and cause.error_code:
            return cause.error_code
    return "dependency_failure"


def redact_errors(activity: Callable) -> Callable:
    """Report an activity's failure to Durable Functions and the log without the original message.

    Durable keeps a failed activity's error in the orchestration history, and an error can quote model output or
    memory text. The history gets the safe error code and the exception type. The Function App log also gets the
    exception's status, validation fields and the code locations it passed through (``describe_exception``), but
    never its message, arguments or local variables. The failure is still raised, so the activity retry of
    ``orchestrators._retry`` applies as before; an exception that is not an ``Exception`` passes through untouched.
    """

    def safe(exc: Exception) -> ActivityFailedError:
        code = safe_error_code(exc)
        logger.error(
            "Activity %s failed error_code=%s: %s", activity.__name__, code, describe_exception(exc, stack=True)
        )
        return ActivityFailedError(f"{code} ({type(exc).__name__})")

    # The safe error is raised after the handler, so it is not chained to the exception it replaces.
    if inspect.iscoroutinefunction(activity):

        @functools.wraps(activity)
        async def run_async(*args: Any, **kwargs: Any) -> Any:
            try:
                return await activity(*args, **kwargs)
            except Exception as exc:
                failure = safe(exc)
            raise failure

        return run_async

    @functools.wraps(activity)
    def run(*args: Any, **kwargs: Any) -> Any:
        try:
            return activity(*args, **kwargs)
        except Exception as exc:
            failure = safe(exc)
        raise failure

    return run
