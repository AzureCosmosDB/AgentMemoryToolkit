"""Marker :class:`MemoryProcessor` for the Durable-Function-app deployment.

When the customer has deployed the sibling Azure Durable Function app and
wired it to Cosmos DB Change Feed, the SDK does no in-process processing.
This stub class makes that intent explicit at construction time and turns
all ``process_*`` calls into debug-logged no-ops.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

from .base import ProcessThreadResult, UserSummaryResult

logger = logging.getLogger(__name__)


class DurableFunctionProcessor:
    """Signals "an Azure Durable Function app is the active processor."

    All ``process_*`` methods short-circuit and return empty results. The
    SDK still writes turn documents to Cosmos as usual; the function app
    picks them up via the Cosmos DB Change Feed trigger.
    """

    def process_thread(
        self,
        *,
        user_id: str,
        thread_id: str,
        turns: list[dict[str, Any]],
        existing_memories: Optional[list[dict[str, Any]]] = None,
    ) -> ProcessThreadResult:
        logger.debug(
            "DurableFunctionProcessor.process_thread no-op user_id=%s thread_id=%s n_turns=%d",
            user_id,
            thread_id,
            len(turns) if turns else 0,
        )
        return ProcessThreadResult(thread_summary=None, extracted_counts={}, reconciled_count=0, elapsed_ms=0)

    def process_extract_memories(
        self,
        *,
        user_id: str,
        thread_id: str,
    ) -> dict[str, int]:
        logger.debug(
            "DurableFunctionProcessor.process_extract_memories no-op user_id=%s thread_id=%s",
            user_id,
            thread_id,
        )
        return {}

    def process_thread_summary(
        self,
        *,
        user_id: str,
        thread_id: str,
    ) -> Optional[dict[str, Any]]:
        logger.debug(
            "DurableFunctionProcessor.process_thread_summary no-op user_id=%s thread_id=%s",
            user_id,
            thread_id,
        )
        return None

    def process_user_summary(
        self,
        *,
        user_id: str,
        thread_ids: Optional[list[str]] = None,
    ) -> UserSummaryResult:
        logger.debug(
            "DurableFunctionProcessor.process_user_summary no-op user_id=%s",
            user_id,
        )
        return UserSummaryResult(summary=None)

    def process_reconcile(self, *, user_id: str) -> int:
        logger.debug(
            "DurableFunctionProcessor.process_reconcile no-op user_id=%s",
            user_id,
        )
        return 0

    def generate_user_summary(
        self,
        *,
        user_id: str,
        thread_summaries: list[dict[str, Any]],
    ) -> UserSummaryResult:
        logger.debug(
            "DurableFunctionProcessor.generate_user_summary no-op user_id=%s n_summaries=%d",
            user_id,
            len(thread_summaries) if thread_summaries else 0,
        )
        return UserSummaryResult(summary=None)

    def close(self) -> None:
        logger.debug("DurableFunctionProcessor.close no-op")
        return None


__all__ = ["DurableFunctionProcessor"]
