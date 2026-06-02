"""Async in-process :class:`AsyncMemoryProcessor` backed by :class:`AsyncPipelineService`.

The underlying :class:`azure.cosmos.agent_memory.aio.services.pipeline.AsyncPipelineService`
exposes native ``async def`` methods, so every call here is a direct
``await`` — no ``asyncio.to_thread`` adapter, no sync sub-clients.
"""

from __future__ import annotations

import time
from typing import Any, Optional

from azure.cosmos.agent_memory.processors.base import (
    ProcessThreadResult,
    UserSummaryResult,
)


class AsyncInProcessProcessor:
    """Async in-process orchestrator over :class:`AsyncPipelineService`."""

    def __init__(
        self,
        pipeline: Any = None,
        *,
        cosmos_container: Any = None,
        turns_container: Any = None,
        summaries_container: Any = None,
        chat_client: Any = None,
        embeddings_client: Any = None,
    ) -> None:
        if pipeline is None:
            if (
                cosmos_container is None
                or turns_container is None
                or summaries_container is None
                or chat_client is None
                or embeddings_client is None
            ):
                raise ValueError(
                    "AsyncInProcessProcessor requires either a `pipeline` instance "
                    "or `cosmos_container`, `turns_container`, `summaries_container`, "
                    "`chat_client`, and `embeddings_client`."
                )
            from azure.cosmos.agent_memory._container_routing import ContainerKey
            from azure.cosmos.agent_memory.aio.services.pipeline import AsyncPipelineService
            from azure.cosmos.agent_memory.aio.store import AsyncMemoryStore

            containers = {
                ContainerKey.TURNS: turns_container,
                ContainerKey.MEMORIES: cosmos_container,
                ContainerKey.SUMMARIES: summaries_container,
            }
            store = AsyncMemoryStore(containers=containers, embeddings_client=embeddings_client)
            pipeline = AsyncPipelineService(store, chat_client, embeddings_client, containers=containers)

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

        thread_summary = await self._pipeline.generate_thread_summary(user_id, thread_id)
        extracted = await self._pipeline.extract_memories(user_id, thread_id)
        reconciled = await self._pipeline.reconcile_memories(user_id, get_dedup_pool_size())

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
        extracted = await self._pipeline.extract_memories(user_id, thread_id)
        return {k: v for k, v in extracted.items() if isinstance(v, int)} if isinstance(extracted, dict) else {}

    async def process_thread_summary(
        self,
        *,
        user_id: str,
        thread_id: str,
    ) -> Optional[dict[str, Any]]:
        summary = await self._pipeline.generate_thread_summary(user_id, thread_id)
        return summary if isinstance(summary, dict) else None

    async def process_user_summary(
        self,
        *,
        user_id: str,
        thread_ids: Optional[list[str]] = None,
    ) -> UserSummaryResult:
        summary = await self._pipeline.generate_user_summary(user_id, thread_ids)
        return UserSummaryResult(summary=summary if isinstance(summary, dict) else None)

    async def process_reconcile(self, *, user_id: str) -> int:
        from ...thresholds import get_dedup_pool_size

        reconciled = await self._pipeline.reconcile_memories(user_id, get_dedup_pool_size())
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

        summary = await self._pipeline.generate_user_summary(user_id, thread_ids)
        return UserSummaryResult(summary=summary if isinstance(summary, dict) else None)

    async def synthesize_procedural(
        self,
        *,
        user_id: str,
        force: bool = False,
    ) -> dict[str, Any]:
        """Run procedural prompt synthesis through the in-process pipeline."""
        return await self._pipeline.synthesize_procedural(user_id, force=force)

    async def close(self) -> None:
        return None


__all__ = ["AsyncInProcessProcessor"]
