"""Synchronous Azure Durable Functions client for the Agent Memory Toolkit.

Provides :class:`ProcessingClient` (synchronous, stdlib-only) that
encapsulates the HTTP-start → poll-until-done lifecycle of Durable
Functions orchestrations.
"""

from __future__ import annotations

import json as _json
import logging
import time
import urllib.error
import urllib.request
from typing import Any

from .exceptions import (
    ConfigurationError,
    OrchestrationTimeoutError,
    ProcessingError,
)

logger = logging.getLogger(__name__)

_ORCHESTRATOR_PATH = "/orchestrators/memory_orchestrator"
_TERMINAL_STATUSES = frozenset(("Completed", "Failed", "Terminated"))


# ---------------------------------------------------------------------------
# Synchronous client
# ---------------------------------------------------------------------------


class ProcessingClient:
    """Synchronous Azure Durable Functions client using :mod:`urllib.request`.

    Parameters
    ----------
    endpoint:
        Base URL of the Azure Functions app hosting the orchestrator.
    key:
        Optional function-level API key appended as ``?code=…``.
    poll_interval:
        Seconds between status polls.  Defaults to ``2.0``.
    timeout:
        Maximum seconds to wait for orchestration completion.  Defaults to
        ``120.0``.
    """

    def __init__(
        self,
        endpoint: str | None = None,
        key: str | None = None,
        poll_interval: float = 2.0,
        timeout: float = 120.0,
    ) -> None:
        self._endpoint = endpoint
        self._key = key
        self._poll_interval = poll_interval
        self._timeout = timeout

    # -- core ---------------------------------------------------------------

    def invoke_orchestrator(
        self,
        payload: dict[str, Any],
        poll_interval: float | None = None,
        timeout: float | None = None,
    ) -> dict[str, Any]:
        """Start an orchestration and poll until it reaches a terminal state.

        Parameters
        ----------
        payload:
            JSON body sent to the orchestrator HTTP-start endpoint.
        poll_interval:
            Seconds between status polls.  Falls back to the constructor value.
        timeout:
            Maximum seconds to wait.  Falls back to the constructor value.

        Returns
        -------
        dict
            The full status response from the orchestration.

        Raises
        ------
        ConfigurationError
            If ``endpoint`` is not set.
        ProcessingError
            If the orchestration finishes with ``runtimeStatus == "Failed"``.
        OrchestrationTimeoutError
            If polling exceeds *timeout*.
        """
        if not self._endpoint:
            raise ConfigurationError(
                "Processing endpoint is required to invoke orchestrations",
                parameter="endpoint",
            )

        poll_interval = poll_interval if poll_interval is not None else self._poll_interval
        timeout = timeout if timeout is not None else self._timeout

        url = self._endpoint.rstrip("/") + _ORCHESTRATOR_PATH
        if self._key:
            url += f"?code={self._key}"

        logger.debug("POST %s with payload %s", url, payload)

        data = _json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"}, method="POST")
        try:
            with urllib.request.urlopen(req) as resp:
                start_response: dict[str, Any] = _json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            raise ProcessingError(f"Failed to start orchestration: HTTP {exc.code} {exc.reason}") from exc
        except urllib.error.URLError as exc:
            raise ProcessingError(f"Failed to reach orchestration endpoint: {exc.reason}") from exc

        status_url = start_response.get("statusQueryGetUri")
        if not status_url:
            return start_response

        logger.info(
            "Orchestration started (instance=%s), polling for completion",
            start_response.get("id"),
        )

        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            time.sleep(poll_interval)
            status_req = urllib.request.Request(status_url, method="GET")
            try:
                with urllib.request.urlopen(status_req) as resp:
                    status: dict[str, Any] = _json.loads(resp.read().decode("utf-8"))
            except urllib.error.HTTPError as exc:
                raise ProcessingError(f"Failed to poll orchestration status: HTTP {exc.code} {exc.reason}") from exc
            except urllib.error.URLError as exc:
                raise ProcessingError(f"Failed to reach orchestration status endpoint: {exc.reason}") from exc

            runtime_status = status.get("runtimeStatus", "")
            logger.debug("Poll runtimeStatus=%s", runtime_status)

            if runtime_status in _TERMINAL_STATUSES:
                if runtime_status == "Failed":
                    error_detail = status.get("output") or status.get("customStatus")
                    raise ProcessingError(f"Orchestration failed: {error_detail}")
                logger.info("Orchestration completed with status=%s", runtime_status)
                return status

        raise OrchestrationTimeoutError(timeout=timeout, status_url=status_url)

    # -- convenience wrappers -----------------------------------------------

    def generate_thread_summary(
        self,
        user_id: str,
        thread_id: str,
        recent_k: int | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Generate a summary for a single thread."""
        payload: dict[str, Any] = {
            "user_id": user_id,
            "thread_id": thread_id,
            "thread_summary_only": True,
        }
        if recent_k is not None:
            payload["recent_k"] = recent_k
        return self.invoke_orchestrator(payload, **kwargs)

    def extract_facts(
        self,
        user_id: str,
        thread_id: str,
        recent_k: int | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Extract factual knowledge from a thread."""
        payload: dict[str, Any] = {
            "user_id": user_id,
            "thread_id": thread_id,
            "extract_facts_only": True,
        }
        if recent_k is not None:
            payload["recent_k"] = recent_k
        return self.invoke_orchestrator(payload, **kwargs)

    def generate_user_summary(
        self,
        user_id: str,
        thread_ids: list[str] | None = None,
        recent_k: int | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Generate a cross-thread summary for a user."""
        payload: dict[str, Any] = {
            "user_id": user_id,
            "user_summary_only": True,
        }
        if thread_ids is not None:
            payload["thread_ids"] = thread_ids
        if recent_k is not None:
            payload["recent_k"] = recent_k
        return self.invoke_orchestrator(payload, **kwargs)
