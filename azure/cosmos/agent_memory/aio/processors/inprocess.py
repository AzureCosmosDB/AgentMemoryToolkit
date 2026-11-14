"""Async in-process :class:`AsyncMemoryProcessor` backed by :class:`AsyncPipelineService`.

The underlying :class:`azure.cosmos.agent_memory.aio.services.pipeline.AsyncPipelineService`
exposes native ``async def`` methods, so every call here is a direct
``await`` - no ``asyncio.to_thread`` adapter, no sync sub-clients.
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
        """Summarize → extract → reconcile for a single thread.

        Fused convenience wrapper used by
        :meth:`AsyncCosmosMemoryClient.process_now`. The auto-trigger path uses
        the granular ``process_extract_memories`` / ``process_thread_summary``
        / ``process_reconcile`` methods instead so each step fires on its own
        threshold cadence (matching the function-app behavior).

        ``turns`` and ``existing_memories`` are accepted for protocol
        symmetry; the pipeline queries the container itself.
        """
        start = time.monotonic()

        from ...thresholds import get_dedup_pool_size

        thread_summary = await self._pipeline.generate_thread_summary(user_id, thread_id)
        extracted = await self._pipeline.extract_memories(user_id, thread_id)
        deduped_count = await self._reconcile_fact_and_episodic(user_id, get_dedup_pool_size())

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
        recent_k: Optional[int] = None,
    ) -> dict[str, int]:
        extracted = await self._pipeline.extract_memories(user_id, thread_id, recent_k=recent_k)
        return {k: v for k, v in extracted.items() if isinstance(v, int)} if isinstance(extracted, dict) else {}

    async def process_extract_episodes(
        self,
        *,
        user_id: str,
        thread_id: str,
    ) -> dict[str, int]:
        extracted = await self._pipeline.extract_episodes(user_id, thread_id)
        return {k: v for k, v in extracted.items() if isinstance(v, int)} if isinstance(extracted, dict) else {}

    async def process_thread_summary(
        self,
        *,
        user_id: str,
        thread_id: str,
    ) -> Optional[dict[str, Any]]:
        """Generate a thread summary. Used by the auto-trigger on THREAD_SUMMARY_EVERY_N."""
        summary = await self._pipeline.generate_thread_summary(user_id, thread_id)
        return summary if isinstance(summary, dict) else None

    async def process_user_summary(
        self,
        *,
        user_id: str,
        thread_ids: Optional[list[str]] = None,
    ) -> UserSummaryResult:
        """Generate a cross-thread user summary. Used by the auto-trigger on USER_SUMMARY_EVERY_N."""
        summary = await self._pipeline.generate_user_summary(user_id, thread_ids)
        return UserSummaryResult(summary=summary if isinstance(summary, dict) else None)

    async def process_reconcile(self, *, user_id: str) -> int:
        """Run reconciliation standalone. Returns the count of facts consolidated or retired.

        That is the dedup pass's merged and combined facts plus the facts the
        contradiction pass retired. Pool size is read from ``DEDUP_POOL_SIZE``
        (env-tunable, default 50, capped at 500) so the auto-trigger and the
        standalone path agree.
        """
        from ...thresholds import get_dedup_pool_size

        return await self._reconcile_fact_and_episodic(user_id, get_dedup_pool_size())

    async def _reconcile_fact_and_episodic(self, user_id: str, n: int) -> int:
        """Reconcile facts, then episodic memories; sum the consolidated counts.

        SDK in-process processing makes the same two ``reconcile_memories`` calls as the Durable
        backend. Facts get the duplicate, combine and contradiction passes; the pipeline treats an
        episodic reconcile as a no-op, so it adds nothing to the count.
        """
        total = 0
        for memory_type in ("fact", "episodic"):
            reconciled = await self._pipeline.reconcile_memories(user_id, n=n, memory_type=memory_type)
            total += self._extract_reconcile_count(reconciled)
        return total

    @staticmethod
    def _extract_reconcile_count(reconciled: Any) -> int:
        """Sum ``merged + contradicted + combined`` from a ``reconcile_memories`` result.

        ``AsyncPipelineService.reconcile_memories`` returns a dict with
        ``{"kept", "merged", "contradicted"}`` plus the dedup pass's counts;
        ``merged`` (duplicates), ``contradicted`` and ``combined`` (facts
        retired into a combined fact) all represent facts that were
        consolidated or retired, so they contribute to the dedup-count metric.
        """
        if not isinstance(reconciled, dict):
            return 0
        return sum(
            reconciled[key] if isinstance(reconciled.get(key), int) else 0
            for key in ("merged", "contradicted", "combined")
        )

    async def generate_user_summary(
        self,
        *,
        user_id: str,
        thread_summaries: list[dict[str, Any]],
    ) -> UserSummaryResult:
        """Run the cross-thread user summary pipeline."""
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
        """Learn procedures from the facts and episodes written since the last run.

        Runs ``AsyncPipelineService.synthesize_procedural``; ``force`` has no effect.
        """
        return await self._pipeline.synthesize_procedural(user_id, force=force)

    async def close(self) -> None:
        return None


__all__ = ["AsyncInProcessProcessor"]
