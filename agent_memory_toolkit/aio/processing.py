"""Async Azure Durable Functions client for the Agent Memory Toolkit.

Provides :class:`AsyncProcessingClient` (asyncio + aiohttp) that
encapsulates the HTTP-start → poll-until-done lifecycle of Durable
Functions orchestrations.
"""

from __future__ import annotations

import logging
from typing import Any

from agent_memory_toolkit.exceptions import (
    ConfigurationError,
    OrchestrationTimeoutError,
    ProcessingError,
)

logger = logging.getLogger(__name__)

_ORCHESTRATOR_PATH = "/orchestrators/memory_orchestrator"
_TERMINAL_STATUSES = frozenset(("Completed", "Failed", "Terminated"))


class AsyncProcessingClient:
    """Async Azure Durable Functions client using aiohttp."""

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
        self._session: Any = None  # aiohttp.ClientSession, lazily created

    # -- async context manager ----------------------------------------------

    async def __aenter__(self) -> AsyncProcessingClient:
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        await self.close()

    async def close(self) -> None:
        """Close the underlying HTTP session."""
        if self._session is not None and not self._session.closed:
            await self._session.close()
            self._session = None

    # -- internal helpers ---------------------------------------------------

    async def _get_session(self) -> Any:
        """Return the shared :class:`aiohttp.ClientSession`, creating it on first use."""
        if self._session is None or self._session.closed:
            import aiohttp

            self._session = aiohttp.ClientSession()
        return self._session

    # -- core ---------------------------------------------------------------

    async def invoke_orchestrator(
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
            Seconds between status polls.  Falls back to constructor default.
        timeout:
            Maximum seconds to wait.  Falls back to constructor default.

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
        import asyncio

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

        import aiohttp

        session = await self._get_session()

        try:
            async with session.post(url, json=payload) as resp:
                resp.raise_for_status()
                start_response: dict[str, Any] = await resp.json()
        except aiohttp.ClientError as exc:
            raise ProcessingError(f"Failed to start orchestration: {exc}") from exc

        status_url = start_response.get("statusQueryGetUri")
        if not status_url:
            return start_response

        logger.info(
            "Orchestration started (instance=%s), polling for completion",
            start_response.get("id"),
        )

        loop = asyncio.get_running_loop()
        deadline = loop.time() + timeout
        while loop.time() < deadline:
            await asyncio.sleep(poll_interval)
            try:
                async with session.get(status_url) as resp:
                    resp.raise_for_status()
                    status: dict[str, Any] = await resp.json()
            except aiohttp.ClientError as exc:
                raise ProcessingError(f"Failed to poll orchestration status: {exc}") from exc

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

    async def generate_thread_summary(
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
        return await self.invoke_orchestrator(payload, **kwargs)

    async def extract_facts(
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
        return await self.invoke_orchestrator(payload, **kwargs)

    async def generate_user_summary(
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
        return await self.invoke_orchestrator(payload, **kwargs)
