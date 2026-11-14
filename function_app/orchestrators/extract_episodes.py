"""Episodic-extraction orchestrator + activities.

Chain: ``ExtractEpisodes``.

The pipeline writes episodes to Cosmos DB during ``ExtractEpisodes``; the
Function App returns only a slim status payload because Durable persists
activity outputs to orchestration history.
"""

from __future__ import annotations

import logging

import azure.durable_functions as df
from shared.error_redaction import redact_errors
from shared.pipeline_factory import get_pipeline

from ._retry import default_retry_options

logger = logging.getLogger(__name__)

bp = df.Blueprint()


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------


@bp.orchestration_trigger(context_name="context")
def ExtractEpisodesOrchestrator(context: df.DurableOrchestrationContext):
    payload = context.get_input() or {}
    user_id = payload["user_id"]
    thread_id = payload["thread_id"]

    retry = default_retry_options()

    result = yield context.call_activity_with_retry(
        "ee_ExtractEpisodes",
        retry,
        {"user_id": user_id, "thread_id": thread_id},
    )

    return result


# ---------------------------------------------------------------------------
# Activities
# ---------------------------------------------------------------------------


@bp.activity_trigger(input_name="payload")
@redact_errors
def ee_ExtractEpisodes(payload: dict) -> dict:
    user_id = payload["user_id"]
    thread_id = payload["thread_id"]
    pipeline = get_pipeline()
    result = pipeline.extract_episodes(user_id=user_id, thread_id=thread_id, flush=False) or {}
    slim = {"episodes": int(result.get("episodes", 0))}
    logger.info(
        "ExtractEpisodes user=%s thread=%s episodes=%s",
        user_id,
        thread_id,
        slim["episodes"],
    )
    return slim
