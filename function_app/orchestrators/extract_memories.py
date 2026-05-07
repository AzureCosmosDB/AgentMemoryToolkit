"""Memory-extraction orchestrator + activities.

Chain: ``ExtractMemories`` followed by an optional ``ReconcileMemories``
activity. Reconciliation is gated by the change-feed trigger (which
tracks the per-user/thread turn counter) and signaled to the
orchestrator via the ``reconcile`` flag on its input payload. This
matches the SDK auto-trigger contract: extract every
``FACT_EXTRACTION_EVERY_N`` turns; reconcile every
``FACT_EXTRACTION_EVERY_N * DEDUP_EVERY_N`` turns.

The pipeline writes memories to Cosmos DB during ``ExtractMemories``; the
Function App does not delete or tombstone any memories on its own. Memories
are removed only via explicit user-driven SDK calls (``forget_memory`` /
``delete_memories``). Salience is preserved on each document for use as a
ranking signal at retrieval time, not as an automatic-deletion threshold.
"""

from __future__ import annotations

import logging

import azure.durable_functions as df
from shared import config
from shared.pipeline_factory import get_pipeline

from ._retry import default_retry_options

logger = logging.getLogger(__name__)

bp = df.Blueprint()


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------


@bp.orchestration_trigger(context_name="context")
def ExtractMemoriesOrchestrator(context: df.DurableOrchestrationContext):
    payload = context.get_input() or {}
    user_id = payload["user_id"]
    thread_id = payload["thread_id"]
    should_reconcile = bool(payload.get("reconcile", False))
    max_batch = config.get_max_batch_size()

    retry = default_retry_options()

    extracted = yield context.call_activity_with_retry(
        "em_ExtractMemories",
        retry,
        {"user_id": user_id, "thread_id": thread_id, "limit": max_batch},
    )

    reconciled = None
    if should_reconcile:
        reconciled = yield context.call_activity_with_retry(
            "em_ReconcileMemories",
            retry,
            {"user_id": user_id},
        )

    return {
        "persisted": True,
        "extracted": extracted,
        "reconciled": reconciled,
    }


# ---------------------------------------------------------------------------
# Activities
# ---------------------------------------------------------------------------


@bp.activity_trigger(input_name="payload")
def em_ExtractMemories(payload: dict) -> dict:
    """Run the LLM extraction step.

    Returns the per-type counts produced by ``pipeline.extract_memories``,
    shaped like
    ``{"facts_count": N, "procedural_count": N, "episodic_count": N, "updated_count": N}``.
    Salience-based filtering is delegated to the pipeline since it owns the
    schema.

    The pipeline loads recent turns internally, so we do NOT pre-load them in
    a separate activity (which would duplicate the query, waste RUs, and open
    a TOCTOU window between the load and the LLM call).
    """
    user_id = payload["user_id"]
    thread_id = payload["thread_id"]
    limit = payload.get("limit")
    pipeline = get_pipeline()
    counts = pipeline.extract_memories(
        user_id=user_id,
        thread_id=thread_id,
        recent_k=limit,
    )
    logger.info(
        "ExtractMemories user=%s thread=%s counts=%s",
        user_id,
        thread_id,
        counts,
    )
    return counts or {}


@bp.activity_trigger(input_name="payload")
def em_ReconcileMemories(payload: dict) -> dict:
    user_id = payload["user_id"]
    pipeline = get_pipeline()
    from agent_memory_toolkit.thresholds import get_dedup_pool_size

    return pipeline.reconcile_memories(user_id=user_id, n=get_dedup_pool_size()) or {}
