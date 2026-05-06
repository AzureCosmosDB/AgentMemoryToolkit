"""User-summary orchestrator + activities.

Chain: ``GenerateUserSummary`` → ``PersistUserSummary``.
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
def UserSummaryOrchestrator(context: df.DurableOrchestrationContext):
    payload = context.get_input() or {}
    user_id = payload["user_id"]
    thread_ids = payload.get("thread_ids") or None
    max_batch = config.get_max_batch_size()

    retry = default_retry_options()

    user_summary = yield context.call_activity_with_retry(
        "us_GenerateUserSummary",
        retry,
        {"user_id": user_id, "limit": max_batch, "thread_ids": thread_ids},
    )

    yield context.call_activity_with_retry(
        "us_PersistUserSummary",
        retry,
        {"user_id": user_id, "user_summary": user_summary},
    )

    return {
        "persisted": True,
        "user_summary_id": (user_summary.get("id") if isinstance(user_summary, dict) else None),
    }


@bp.activity_trigger(input_name="payload")
def us_GenerateUserSummary(payload: dict) -> dict:
    """Generate a cross-thread user summary.

    The pipeline loads recent thread summaries internally; we do NOT pre-load
    them in a separate activity (which would duplicate the query and open a
    TOCTOU window between the load and the LLM call).
    """
    user_id = payload["user_id"]
    limit = payload.get("limit")
    thread_ids = payload.get("thread_ids") or None
    pipeline = get_pipeline()
    summary = pipeline.generate_user_summary(user_id=user_id, recent_k=limit, thread_ids=thread_ids)
    logger.info("UserSummary generated user=%s", user_id)
    return summary


@bp.activity_trigger(input_name="payload")
def us_PersistUserSummary(payload: dict) -> dict:
    """Observability shim — the pipeline has already upserted the user-summary doc."""
    summary = payload.get("user_summary") or {}
    return {"id": summary.get("id"), "persisted": True}
