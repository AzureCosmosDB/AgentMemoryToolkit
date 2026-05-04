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

        thread_summary = await asyncio.to_thread(self._pipeline.generate_thread_summary, user_id, thread_id)
        extracted = await asyncio.to_thread(self._pipeline.extract_memories, user_id, thread_id)
        dedup = await asyncio.to_thread(self._pipeline.deduplicate_facts, user_id)

        deduped_count = 0
        if isinstance(dedup, dict):
            for key in ("deduplicated", "merged", "removed", "deduplicated_count"):
                if key in dedup and isinstance(dedup[key], int):
                    deduped_count = dedup[key]
                    break

        extracted_list: list[dict[str, Any]]
        if isinstance(extracted, list):
            extracted_list = extracted
        elif isinstance(extracted, dict):
            extracted_list = [extracted]
        else:
            extracted_list = []

        elapsed_ms = int((time.monotonic() - start) * 1000)
        return ProcessThreadResult(
            thread_summary=thread_summary if isinstance(thread_summary, dict) else None,
            extracted=extracted_list,
            deduplicated_count=deduped_count,
            elapsed_ms=elapsed_ms,
        )

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
