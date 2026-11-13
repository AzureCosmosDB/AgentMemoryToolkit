"""Procedural-synthesis orchestrator + activities.

Chain: ``SynthesizeProcedural``.

The pipeline writes the procedural doc to Cosmos DB during
``SynthesizeProcedural``; the Function App returns only a slim status payload
because Durable persists activity outputs to orchestration history.
"""

from __future__ import annotations

import logging
import time

import azure.durable_functions as df
from shared import config
from shared.error_redaction import redact_errors
from shared.pipeline_factory import get_pipeline

from ._retry import default_retry_options

logger = logging.getLogger(__name__)

bp = df.Blueprint()


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------


@bp.orchestration_trigger(context_name="context")
def SynthesizeProceduralOrchestrator(context: df.DurableOrchestrationContext):
    payload = context.get_input() or {}
    user_id = payload["user_id"]
    force = bool(payload.get("force", False))

    retry = default_retry_options()

    result = yield context.call_activity_with_retry(
        "sp_SynthesizeProcedural",
        retry,
        {"user_id": user_id, "force": force},
    )

    return result


# ---------------------------------------------------------------------------
# Activities
# ---------------------------------------------------------------------------


@bp.activity_trigger(input_name="payload")
@redact_errors
def sp_SynthesizeProcedural(payload: dict) -> dict:
    # One activity: the pass moves its cursor after each batch of sources and stops
    # starting model calls at the deadline, so a long backlog spreads over later runs
    # instead of outrunning the function timeout.
    user_id = payload["user_id"]
    force = bool(payload.get("force", False))
    pipeline = get_pipeline()
    deadline = time.monotonic() + config.get_activity_time_budget_seconds()
    result = pipeline.synthesize_procedural(user_id=user_id, force=force, deadline=deadline) or {}
    slim = {
        "status": result.get("status"),
        "procedures_created": int(result.get("procedures_created") or 0),
        "facts_retired": int(result.get("facts_retired") or 0),
    }
    logger.info(
        "SynthesizeProcedural user=%s status=%s procedures_created=%s facts_retired=%s",
        user_id,
        slim["status"],
        slim["procedures_created"],
        slim["facts_retired"],
    )
    return slim
