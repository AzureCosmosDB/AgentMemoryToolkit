"""Procedural-synthesis orchestrator + activities.

Chain: ``SynthesizeProcedural``.

The pipeline writes the procedural doc to Cosmos DB during
``SynthesizeProcedural``; the Function App returns only a slim status payload
because Durable persists activity outputs to orchestration history.
"""

from __future__ import annotations

import logging

import azure.durable_functions as df
from shared.pipeline_factory import get_pipeline

from azure.cosmos.agent_memory._partitioning import tenant_scope

from ._retry import default_retry_options

logger = logging.getLogger(__name__)

bp = df.Blueprint()


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------


@bp.orchestration_trigger(context_name="context")
def SynthesizeProceduralOrchestrator(context: df.DurableOrchestrationContext):
    payload = context.get_input() or {}
    tenant_id = payload.get("tenant_id")
    user_id = payload["user_id"]
    force = bool(payload.get("force", False))

    retry = default_retry_options()

    result = yield context.call_activity_with_retry(
        "sp_SynthesizeProcedural",
        retry,
        {"tenant_id": tenant_id, "user_id": user_id, "force": force},
    )

    return result


# ---------------------------------------------------------------------------
# Activities
# ---------------------------------------------------------------------------


@bp.activity_trigger(input_name="payload")
def sp_SynthesizeProcedural(payload: dict) -> dict:
    # Keep procedural synthesis single-activity for GA: chunked LLM rewrites already
    # retry internally, and the full procedural body is too bulky for history wins.
    tenant_id = payload.get("tenant_id")
    user_id = payload["user_id"]
    force = bool(payload.get("force", False))
    pipeline = get_pipeline()
    with tenant_scope(tenant_id):
        result = pipeline.synthesize_procedural(user_id=user_id, force=force) or {}
    slim = {
        "status": result.get("status"),
        "procedures_created": int(result.get("procedures_created") or 0),
    }
    logger.info(
        "SynthesizeProcedural user=%s status=%s procedures_created=%s",
        user_id,
        slim["status"],
        slim["procedures_created"],
    )
    return slim
