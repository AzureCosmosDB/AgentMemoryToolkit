"""User-summary orchestrator + activities.

Chain: ``Extract`` → ``PersistUserSummary``. Extract loads memories and calls the
LLM; PersistUserSummary computes the embedding and writes the deterministic doc.
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
    retry = default_retry_options()

    user_summary = yield context.call_activity_with_retry(
        "us_Extract",
        retry,
        {"user_id": user_id, "limit": config.get_max_batch_size(), "thread_ids": thread_ids},
    )

    persisted = yield context.call_activity_with_retry(
        "us_PersistUserSummary",
        retry,
        {"user_id": user_id, "user_summary": user_summary},
    )

    return {
        "persisted": True,
        "user_summary_id": (persisted.get("id") if isinstance(persisted, dict) else None),
    }


@bp.activity_trigger(input_name="payload")
def us_Extract(payload: dict) -> dict:
    """Generate a cross-thread user summary body only."""
    user_id = payload["user_id"]
    summary = get_pipeline().generate_user_summary_dry(
        user_id=user_id,
        recent_k=payload.get("limit"),
        thread_ids=payload.get("thread_ids") or None,
    )
    logger.info("UserSummary extracted user=%s", user_id)
    return summary


@bp.activity_trigger(input_name="payload")
def us_PersistUserSummary(payload: dict) -> dict:
    """Compute the embedding and persist the user summary."""
    user_id = payload["user_id"]
    summary = get_pipeline().persist_user_summary(
        user_id=user_id,
        user_summary_doc=payload["user_summary"],
    )
    logger.info("UserSummary persisted user=%s id=%s", user_id, summary.get("id"))
    return summary
