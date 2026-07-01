"""Memory-extraction orchestrator + activities.

Chain: ``Extract`` → ``Dedup`` → ``Persist`` followed by an optional
``ReconcileMemories`` activity, then a best-effort
``SynthesizeProceduralOrchestrator`` sub-call.
Reconciliation is gated by the change-feed trigger (which tracks the
per-user/thread turn counter) and signaled to the orchestrator via the
``reconcile`` flag on its input payload. Procedural synthesis fires only
after reconcile and only when ``PROCEDURAL_SYNTHESIS_AUTO`` is enabled, so
operators have a kill-switch for the extra LLM call. The prompt is always
derived from the deduped fact pool. Redundant concurrent runs across threads
are cheap because the pipeline short-circuits with ``status="unchanged"``
when the source fact/episodic IDs have not moved.
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
def ExtractMemoriesOrchestrator(context: df.DurableOrchestrationContext):
    payload = context.get_input() or {}
    user_id = payload["user_id"]
    thread_id = payload["thread_id"]
    should_reconcile = bool(payload.get("reconcile", False))
    full_rebuild = bool(payload.get("full_rebuild", False))
    recent_k = payload.get("recent_k")
    retry = default_retry_options()

    extract_payload = {"user_id": user_id, "thread_id": thread_id}
    if recent_k is not None:
        extract_payload["recent_k"] = recent_k
    extracted = yield context.call_activity_with_retry(
        "em_Extract",
        retry,
        extract_payload,
    )
    deduped = yield context.call_activity_with_retry(
        "em_Dedup",
        retry,
        {"user_id": user_id, "extracted": extracted},
    )
    persisted = yield context.call_activity_with_retry(
        "em_Persist",
        retry,
        {"user_id": user_id, "extracted": deduped},
    )

    count = payload.get("count")
    if count is not None:
        yield context.call_activity_with_retry(
            "em_AdvanceExtractWatermark",
            retry,
            {"user_id": user_id, "thread_id": thread_id, "count": count},
        )

    reconciled = None
    procedural = None
    if should_reconcile:
        reconciled = yield context.call_activity_with_retry(
            "em_ReconcileMemories",
            retry,
            {"user_id": user_id, "full_rebuild": full_rebuild},
        )
        if config.get_procedural_synthesis_auto():
            count = payload.get("count")
            instance_id = f"procedural:{user_id}:{thread_id}:{count}" if count is not None else None
            try:
                procedural = yield context.call_sub_orchestrator_with_retry(
                    "SynthesizeProceduralOrchestrator",
                    retry,
                    {"user_id": user_id, "force": False},
                    instance_id=instance_id,
                )
            except Exception as exc:
                if not context.is_replaying:
                    logger.warning(
                        "SynthesizeProceduralOrchestrator failed user=%s thread=%s: %s",
                        user_id,
                        thread_id,
                        exc,
                    )

    return {
        "persisted": True,
        "extracted": persisted,
        "reconciled": reconciled,
        "procedural": procedural,
    }


@bp.activity_trigger(input_name="payload")
def em_Extract(payload: dict) -> dict:
    """Load recent turns and run LLM extraction without embeddings or writes."""
    user_id = payload["user_id"]
    thread_id = payload["thread_id"]
    recent_k = payload.get("recent_k")
    if recent_k is None:
        recent_k = config.get_max_batch_size()
    extracted = get_pipeline().extract_memories_dry(
        user_id=user_id,
        thread_id=thread_id,
        recent_k=recent_k,
    )
    logger.info(
        "ExtractMemories extracted user=%s thread=%s facts=%d episodic=%d updates=%d",
        user_id,
        thread_id,
        len(extracted.get("facts", [])),
        len(extracted.get("episodic", [])),
        len(extracted.get("updates", [])),
    )
    return extracted


@bp.activity_trigger(input_name="payload")
def em_Dedup(payload: dict) -> dict:
    """vector-floor dedup ladder (gated; passthrough when disabled)."""
    return (
        get_pipeline().dedup_extracted_memories(
            user_id=payload["user_id"],
            extracted=payload["extracted"],
        )
        or payload["extracted"]
    )


@bp.activity_trigger(input_name="payload")
def em_Persist(payload: dict) -> dict:
    """Persist extracted docs with embeddings and deterministic create semantics."""
    user_id = payload["user_id"]
    counts = get_pipeline().persist_extracted_memories(
        user_id=user_id,
        extracted=payload["extracted"],
    )
    logger.info("ExtractMemories persisted user=%s counts=%s", user_id, counts)
    return counts or {}


@bp.activity_trigger(input_name="payload")
async def em_AdvanceExtractWatermark(payload: dict) -> bool:
    """Advance the extraction watermark after a successful extract→persist.

    Stamps ``last_extract_count`` on the thread counter so the next batch's
    recent_k spans only turns added since this run, never skipping any.
    Runs only on success (after persist) so failed extracts re-process.
    """
    from shared.cosmos_clients import get_counter_container_async
    from shared.counters import advance_extract_watermark, thread_counter_id

    user_id = payload["user_id"]
    thread_id = payload["thread_id"]
    count = payload["count"]
    container = await get_counter_container_async()
    await advance_extract_watermark(container, thread_counter_id(user_id, thread_id), user_id, thread_id, count)
    return True


@bp.activity_trigger(input_name="payload")
def em_ReconcileMemories(payload: dict) -> dict:
    # GA keeps reconcile single-activity: its LLM dedup decisions and supersession
    # operations are larger/more coupled than the extract→dedup→persist split handled here.
    user_id = payload["user_id"]
    # full_rebuild forces the full-pool single-LLM-pass path (catches dissimilar
    # contradictions). The change-feed sets it on a persisted-counter cadence so it
    # fires reliably on FA, where the in-memory candidate-mode sweep counter can't.
    full_rebuild = bool(payload.get("full_rebuild", False))
    pipeline = get_pipeline()
    from azure.cosmos.agent_memory.thresholds import get_dedup_pool_size

    n = get_dedup_pool_size()
    facts = pipeline.reconcile_memories(user_id=user_id, n=n, memory_type="fact", full_rebuild=full_rebuild) or {}
    episodic = (
        pipeline.reconcile_memories(user_id=user_id, n=n, memory_type="episodic", full_rebuild=full_rebuild) or {}
    )
    return {"fact": facts, "episodic": episodic}
