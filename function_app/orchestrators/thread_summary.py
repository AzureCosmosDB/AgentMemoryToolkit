"""Thread-summary orchestrator + activities.

Chain: ``Extract`` → ``PersistSummary``. Extract loads turns and calls the LLM;
PersistSummary computes the embedding and writes the deterministic summary doc.
"""

from __future__ import annotations

import logging

import azure.durable_functions as df
from shared import config
from shared.pipeline_factory import get_pipeline

from ._retry import default_retry_options

logger = logging.getLogger(__name__)

bp = df.Blueprint()


@bp.orchestration_trigger(context_name="context")
def ThreadSummaryOrchestrator(context: df.DurableOrchestrationContext):
    payload = context.get_input() or {}
    user_id = payload["user_id"]
    thread_id = payload["thread_id"]
    retry = default_retry_options()

    summary = yield context.call_activity_with_retry(
        "ts_Extract",
        retry,
        {"user_id": user_id, "thread_id": thread_id, "limit": config.get_max_batch_size()},
    )

    persisted = yield context.call_activity_with_retry(
        "ts_PersistSummary",
        retry,
        {"user_id": user_id, "thread_id": thread_id, "summary": summary},
    )

    return {
        "persisted": True,
        "summary_id": persisted.get("id") if isinstance(persisted, dict) else None,
    }


@bp.activity_trigger(input_name="payload")
def ts_Extract(payload: dict) -> dict:
    """Generate (or incrementally update) the thread summary body only."""
    user_id = payload["user_id"]
    thread_id = payload["thread_id"]
    summary = get_pipeline().generate_thread_summary_dry(
        user_id=user_id,
        thread_id=thread_id,
        recent_k=payload.get("limit"),
    )
    logger.info("ThreadSummary extracted user=%s thread=%s", user_id, thread_id)
    return summary


@bp.activity_trigger(input_name="payload")
def ts_PersistSummary(payload: dict) -> dict:
    """Compute the embedding and persist the thread summary."""
    user_id = payload["user_id"]
    thread_id = payload["thread_id"]
    summary = get_pipeline().persist_thread_summary(
        user_id=user_id,
        thread_id=thread_id,
        summary_doc=payload["summary"],
    )
    logger.info("ThreadSummary persisted user=%s thread=%s id=%s", user_id, thread_id, summary.get("id"))
    return summary
