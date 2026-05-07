"""Async in-process :class:`AsyncMemoryProcessor` backed by :class:`ProcessingPipeline`.

The underlying :class:`agent_memory_toolkit.pipeline.ProcessingPipeline` is
synchronous; this wrapper exposes ``async def`` methods that simply call
into the sync pipeline. This mirrors the existing pattern in
:class:`agent_memory_toolkit.aio.cosmos_memory_client.AsyncCosmosMemoryClient`,
which already runs the pipeline synchronously inside its async API surface.
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any, Optional

from agent_memory_toolkit.processors.base import (
    ProcessThreadResult,
    UserSummaryResult,
)

logger = logging.getLogger(__name__)


class AsyncInProcessProcessor:
    """Async wrapper around the in-process :class:`ProcessingPipeline`.

    The underlying pipeline is synchronous (multiple LLM + embedding + Cosmos
    calls). To avoid blocking the event loop, all calls are dispatched to the
    default thread pool via :func:`asyncio.to_thread`.
    """

    def __init__(
        self,
        pipeline: Any = None,
        *,
        cosmos_container: Any = None,
        chat_client: Any = None,
        embeddings_client: Any = None,
    ) -> None:
        if pipeline is None:
            if cosmos_container is None or chat_client is None or embeddings_client is None:
                raise ValueError(
                    "AsyncInProcessProcessor requires either a `pipeline` instance "
                    "or `cosmos_container`, `chat_client`, and `embeddings_client`."
                )
            from agent_memory_toolkit.pipeline import ProcessingPipeline

            pipeline = ProcessingPipeline(
                cosmos_container=cosmos_container,
                chat_client=chat_client,
                embeddings_client=embeddings_client,
            )

        self._pipeline = pipeline

    async def process_thread(
        self,
        *,
        user_id: str,
        thread_id: str,
        turns: list[dict[str, Any]],
        existing_memories: Optional[list[dict[str, Any]]] = None,
    ) -> ProcessThreadResult:
        start = time.monotonic()

        from ...thresholds import get_dedup_pool_size

        thread_summary = await asyncio.to_thread(self._pipeline.generate_thread_summary, user_id, thread_id)
        extracted = await asyncio.to_thread(self._pipeline.extract_memories, user_id, thread_id)
        reconciled = await asyncio.to_thread(self._pipeline.reconcile_memories, user_id, get_dedup_pool_size())

        deduped_count = self._extract_reconcile_count(reconciled)

        extracted_counts: dict[str, int] = (
            {k: v for k, v in extracted.items() if isinstance(v, int)} if isinstance(extracted, dict) else {}
        )

        elapsed_ms = int((time.monotonic() - start) * 1000)
        return ProcessThreadResult(
            thread_summary=thread_summary if isinstance(thread_summary, dict) else None,
            extracted_counts=extracted_counts,
            reconciled_count=deduped_count,
            elapsed_ms=elapsed_ms,
        )

    async def process_extract_memories(
        self,
        *,
        user_id: str,
        thread_id: str,
    ) -> dict[str, int]:
        extracted = await asyncio.to_thread(self._pipeline.extract_memories, user_id, thread_id)
        return {k: v for k, v in extracted.items() if isinstance(v, int)} if isinstance(extracted, dict) else {}

    async def process_thread_summary(
        self,
        *,
        user_id: str,
        thread_id: str,
    ) -> Optional[dict[str, Any]]:
        summary = await asyncio.to_thread(self._pipeline.generate_thread_summary, user_id, thread_id)
        return summary if isinstance(summary, dict) else None

    async def process_user_summary(
        self,
        *,
        user_id: str,
        thread_ids: Optional[list[str]] = None,
    ) -> UserSummaryResult:
        summary = await asyncio.to_thread(self._pipeline.generate_user_summary, user_id, thread_ids)
        return UserSummaryResult(summary=summary if isinstance(summary, dict) else None)

    async def process_reconcile(self, *, user_id: str) -> int:
        from ...thresholds import get_dedup_pool_size

        reconciled = await asyncio.to_thread(self._pipeline.reconcile_memories, user_id, get_dedup_pool_size())
        return self._extract_reconcile_count(reconciled)

    @staticmethod
    def _extract_reconcile_count(reconciled: Any) -> int:
        """Sum ``merged + contradicted`` from a ``reconcile_memories`` result.

        ``ProcessingPipeline.reconcile_memories`` returns a dict with
        ``{"kept", "merged", "contradicted"}`` — both ``merged`` and
        ``contradicted`` represent facts that were consolidated or retired,
        so they contribute to the dedup-count metric.
        """
        if not isinstance(reconciled, dict):
            return 0
        merged = reconciled.get("merged", 0) if isinstance(reconciled.get("merged"), int) else 0
        contradicted = reconciled.get("contradicted", 0) if isinstance(reconciled.get("contradicted"), int) else 0
        return merged + contradicted

    async def generate_user_summary(
        self,
        *,
        user_id: str,
        thread_summaries: list[dict[str, Any]],
    ) -> UserSummaryResult:
        thread_ids: Optional[list[str]] = None
        if thread_summaries:
            ids = [s.get("thread_id") for s in thread_summaries if s.get("thread_id")]
            thread_ids = ids or None

        summary = await asyncio.to_thread(self._pipeline.generate_user_summary, user_id, thread_ids)
        return UserSummaryResult(summary=summary if isinstance(summary, dict) else None)

    async def close(self) -> None:
        return None


__all__ = ["AsyncInProcessProcessor"]
