"""User-summary orchestrator + activities.

Chain: ``Extract`` → ``PersistUserSummary``. Extract loads memories and calls the
LLM; PersistUserSummary computes the embedding and writes the deterministic doc.
"""

from __future__ import annotations

import logging
from datetime import timedelta

import azure.durable_functions as df
from shared import config
from shared.pipeline_factory import get_pipeline

from azure.cosmos.agent_memory._partitioning import tenant_scope
from azure.cosmos.agent_memory.exceptions import NoSourceMemoriesError

from ._retry import default_retry_options

logger = logging.getLogger(__name__)

bp = df.Blueprint()

# Sentinel returned by ``us_Extract`` when the user has no persisted memories
# yet, so the orchestrator waits for extraction to land instead of failing.
_NO_MEMORIES_YET_STATUS = "no_memories_yet"


@bp.orchestration_trigger(context_name="context")
def UserSummaryOrchestrator(context: df.DurableOrchestrationContext):
    payload = context.get_input() or {}
    tenant_id = payload.get("tenant_id")
    user_id = payload["user_id"]
    thread_ids = payload.get("thread_ids") or None
    retry = default_retry_options()
    extract_payload = {
        "tenant_id": tenant_id,
        "user_id": user_id,
        "limit": config.get_max_batch_size(),
        "thread_ids": thread_ids,
    }

    # A brand-new user can cross the user-summary threshold before fact
    # extraction (started independently by the change feed) has persisted
    # anything. Rather than fail after the short activity-retry window, poll on a
    # replay-safe Durable timer until memories land or a bounded budget is
    # exhausted, then skip this cadence (a later threshold retries).
    wait_budget = config.get_user_summary_wait_seconds()
    poll_interval = config.get_user_summary_wait_interval_seconds()
    deadline = context.current_utc_datetime + timedelta(seconds=wait_budget)

    user_summary = yield context.call_activity_with_retry("us_Extract", retry, extract_payload)
    while isinstance(user_summary, dict) and user_summary.get("status") == _NO_MEMORIES_YET_STATUS:
        if context.current_utc_datetime >= deadline:
            logger.warning(
                "UserSummary no memories persisted within %ss for user=%s; skipping this "
                "cadence (a later user-summary threshold will retry)",
                wait_budget,
                user_id,
            )
            return {"persisted": False, "user_summary_id": None, "skipped": _NO_MEMORIES_YET_STATUS}
        yield context.create_timer(context.current_utc_datetime + timedelta(seconds=poll_interval))
        user_summary = yield context.call_activity_with_retry("us_Extract", retry, extract_payload)

    persisted = yield context.call_activity_with_retry(
        "us_PersistUserSummary",
        retry,
        {"tenant_id": tenant_id, "user_id": user_id, "user_summary": user_summary},
    )

    return {
        "persisted": True,
        "user_summary_id": (persisted.get("id") if isinstance(persisted, dict) else None),
    }


@bp.activity_trigger(input_name="payload")
def us_Extract(payload: dict) -> dict:
    """Generate a cross-thread user summary body only.

    Returns a ``{"status": "no_memories_yet"}`` sentinel instead of raising when
    the user has no persisted memories yet, so the orchestrator can wait for
    extraction to land rather than exhausting its activity retries.
    """
    tenant_id = payload.get("tenant_id")
    user_id = payload["user_id"]
    try:
        with tenant_scope(tenant_id):
            summary = get_pipeline().generate_user_summary_durable(
                user_id=user_id,
                recent_k=payload.get("limit"),
                thread_ids=payload.get("thread_ids") or None,
            )
    except NoSourceMemoriesError:
        logger.info("UserSummary no source memories yet user=%s; will wait and retry", user_id)
        return {"status": _NO_MEMORIES_YET_STATUS}
    logger.info("UserSummary extracted user=%s", user_id)
    return summary


@bp.activity_trigger(input_name="payload")
def us_PersistUserSummary(payload: dict) -> dict:
    """Compute the embedding and persist the user summary."""
    tenant_id = payload.get("tenant_id")
    user_id = payload["user_id"]
    with tenant_scope(tenant_id):
        summary = get_pipeline().persist_user_summary(
            user_id=user_id,
            user_summary_doc=payload["user_summary"],
        )
    logger.info("UserSummary persisted user=%s id=%s", user_id, summary.get("id"))
    return summary
