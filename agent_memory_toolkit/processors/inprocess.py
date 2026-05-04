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

        ``turns`` and ``existing_memories`` are accepted for protocol
        symmetry; the pipeline queries the container itself.
        """
        start = time.monotonic()

        thread_summary = self._pipeline.generate_thread_summary(user_id, thread_id)
        extracted = self._pipeline.extract_memories(user_id, thread_id)
        dedup = self._pipeline.deduplicate_facts(user_id)

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
