"""In-process :class:`MemoryProcessor` backed by :class:`ProcessingPipeline`."""

from __future__ import annotations

import logging
import time
from typing import Any, Optional

from .base import ProcessThreadResult, UserSummaryResult

logger = logging.getLogger(__name__)


class InProcessProcessor:
    """Runs the summarize → extract → dedup pipeline inline.

    This is the default backend. Wraps an existing
    :class:`agent_memory_toolkit.pipeline.ProcessingPipeline` instance, or
    constructs one from the supplied container / LLM / embeddings clients.
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
                    "InProcessProcessor requires either a `pipeline` instance or "
                    "`cosmos_container`, `chat_client`, and `embeddings_client`."
                )
            from ..pipeline import ProcessingPipeline

            pipeline = ProcessingPipeline(
                cosmos_container=cosmos_container,
                chat_client=chat_client,
                embeddings_client=embeddings_client,
            )

        self._pipeline = pipeline

    # -- MemoryProcessor protocol ------------------------------------------

    def process_thread(
        self,
        *,
        user_id: str,
        thread_id: str,
        turns: list[dict[str, Any]],
        existing_memories: Optional[list[dict[str, Any]]] = None,
    ) -> ProcessThreadResult:
        """Summarize → extract → deduplicate for a single thread.

        Fused convenience wrapper used by
        :meth:`CosmosMemoryClient.process_now`. The auto-trigger path uses
        the granular ``process_extract_memories`` / ``process_thread_summary``
        / ``process_reconcile`` methods instead so each step fires on its own
        threshold cadence (matching the function-app behavior).

        ``turns`` and ``existing_memories`` are accepted for protocol
        symmetry; the pipeline queries the container itself.
        """
        start = time.monotonic()

        from ..thresholds import get_dedup_pool_size

        thread_summary = self._pipeline.generate_thread_summary(user_id, thread_id)
        extracted = self._pipeline.extract_memories(user_id, thread_id)
        reconciled = self._pipeline.reconcile_memories(user_id, get_dedup_pool_size())

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

    def process_extract_memories(
        self,
        *,
        user_id: str,
        thread_id: str,
    ) -> dict[str, int]:
        extracted = self._pipeline.extract_memories(user_id, thread_id)
        return {k: v for k, v in extracted.items() if isinstance(v, int)} if isinstance(extracted, dict) else {}

    def process_thread_summary(
        self,
        *,
        user_id: str,
        thread_id: str,
    ) -> Optional[dict[str, Any]]:
        """Generate a thread summary. Used by the auto-trigger on THREAD_SUMMARY_EVERY_N."""
        summary = self._pipeline.generate_thread_summary(user_id, thread_id)
        return summary if isinstance(summary, dict) else None

    def process_user_summary(
        self,
        *,
        user_id: str,
        thread_ids: Optional[list[str]] = None,
    ) -> UserSummaryResult:
        """Generate a cross-thread user summary. Used by the auto-trigger on USER_SUMMARY_EVERY_N."""
        summary = self._pipeline.generate_user_summary(user_id, thread_ids)
        return UserSummaryResult(summary=summary if isinstance(summary, dict) else None)

    def process_reconcile(self, *, user_id: str) -> int:
        """Run reconciliation standalone. Returns count of facts merged + contradicted.

        Pool size is read from ``DEDUP_POOL_SIZE`` (env-tunable, default 50,
        capped at 500) so the auto-trigger and the standalone path agree.
        """
        from ..thresholds import get_dedup_pool_size

        reconciled = self._pipeline.reconcile_memories(user_id, n=get_dedup_pool_size())
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

    def generate_user_summary(
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

        summary = self._pipeline.generate_user_summary(user_id, thread_ids)
        return UserSummaryResult(summary=summary if isinstance(summary, dict) else None)

    def close(self) -> None:
        """No-op; the SDK owns the pipeline lifecycle."""
        return None


__all__ = ["InProcessProcessor"]
