"""Thread-summary orchestrator + activities.

Chain: ``SummarizeThread`` → ``PersistSummary``.

``SummarizeThread`` calls ``ProcessingPipeline.generate_thread_summary`` which
loads turns, calls the LLM, and upserts the summary doc in a single
self-contained transaction. ``PersistSummary`` is a thin observability shim
that surfaces an explicit Persist event in App Insights / the Durable status
API.
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
def ThreadSummaryOrchestrator(context: df.DurableOrchestrationContext):
    payload = context.get_input() or {}
    user_id = payload["user_id"]
    thread_id = payload["thread_id"]
    max_batch = config.get_max_batch_size()

    retry = default_retry_options()

    summary = yield context.call_activity_with_retry(
        "ts_SummarizeThread", retry,
        {"user_id": user_id, "thread_id": thread_id, "limit": max_batch},
    )

    yield context.call_activity_with_retry(
        "ts_PersistSummary", retry,
        {"user_id": user_id, "thread_id": thread_id, "summary": summary},
    )

    return {
        "persisted": True,
        "summary_id": summary.get("id") if isinstance(summary, dict) else None,
    }


# ---------------------------------------------------------------------------
# Activities
# ---------------------------------------------------------------------------


@bp.activity_trigger(input_name="payload")
def ts_SummarizeThread(payload: dict) -> dict:
    """Generate (or incrementally update) the thread summary.

    The pipeline loads recent turns internally; we do NOT pre-load them in a
    separate activity (which would duplicate the query, waste RUs, and open a
    TOCTOU window between the load and the LLM call).
    """
    user_id = payload["user_id"]
    thread_id = payload["thread_id"]
    limit = payload.get("limit")
    pipeline = get_pipeline()
    summary = pipeline.generate_thread_summary(
        user_id=user_id, thread_id=thread_id, recent_k=limit,
    )
    logger.info("ThreadSummary generated user=%s thread=%s", user_id, thread_id)
    return summary


@bp.activity_trigger(input_name="payload")
def ts_PersistSummary(payload: dict) -> dict:
    """Observability shim — the pipeline has already upserted the summary doc.

    Kept as a separate activity so operators see explicit Persist events in
    App Insights / the Durable status API.
    """
    summary = payload.get("summary") or {}
    return {"id": summary.get("id"), "persisted": True}
