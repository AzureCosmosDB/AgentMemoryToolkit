"""Configuration helpers for the processor function app.

All knobs are read from environment variables / Azure Functions app settings.
Defaults:

* ``FACT_EXTRACTION_EVERY_N``      — default 1 (per-turn extraction)
* ``THREAD_SUMMARY_EVERY_N``       — default 10 (rolling summary cadence)
* ``USER_SUMMARY_EVERY_N``         — default 20
* ``PROCEDURAL_SYNTHESIS_AUTO``    — default true
* ``MAX_BATCH_SIZE``               — default 20

The fact-extraction default of ``1`` - every
new turn produces fresh facts. Operators can raise this for cost-sensitive
workloads. Summaries default to ``10`` because each summary call sees the
full recent context window and is the most expensive per-call operation.

Setting any ``*_EVERY_N`` env var to ``"0"`` disables that orchestrator
entirely. ``PROCEDURAL_SYNTHESIS_AUTO=false`` disables the chained
procedural-synthesis sub-orchestrator. When an env var is unset or invalid the
documented default is applied (so an out-of-the-box deploy actually
summarizes/extracts) and a warning is logged for invalid values. Use the
``get_*_every_n()`` helpers rather than calling ``_parse_threshold`` directly.

Threshold-crossing semantics in the change-feed trigger
-------------------------------------------------------
A single change-feed batch advances each counter by the entire batch's
contribution in one ETag-guarded write. Even if the resulting jump crosses
several ``EVERY_N`` boundaries (e.g. ``EVERY_N=10`` and a batch of 100
turns advances a counter from 0 → 100), exactly **one** orchestrator is
started — keyed on the new counter value — and it folds in only the
trailing ``MAX_BATCH_SIZE`` items. For high-throughput bursts this means
summarization is coarser than ``EVERY_N`` alone implies. Operators who
need finer per-boundary fan-out should lower ``MAX_BATCH_SIZE`` and raise
``EVERY_N`` proportionally, or process upstream in smaller batches.
"""

from __future__ import annotations

import logging
import os

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Cosmos DB binding / data-plane endpoints
# ---------------------------------------------------------------------------

CHANGE_FEED_DATABASE = os.environ.get("COSMOS_DB_DATABASE", "ai_memory")
MEMORIES_CONTAINER = os.environ.get("COSMOS_DB_MEMORIES_CONTAINER", "memories")
TURNS_CONTAINER = os.environ.get("COSMOS_DB_TURNS_CONTAINER", "memories_turns")
SUMMARIES_CONTAINER = os.environ.get("COSMOS_DB_SUMMARIES_CONTAINER", "memories_summaries")
LEASE_CONTAINER = os.environ.get("COSMOS_DB_LEASE_CONTAINER", "leases")
COUNTERS_CONTAINER = os.environ.get("COSMOS_DB_COUNTERS_CONTAINER", "counter")

USER_COUNTER_THREAD_ID = "__counters__"


# ---------------------------------------------------------------------------
# Defaults documented in local.settings.json.template
# ---------------------------------------------------------------------------

from azure.cosmos.agent_memory.thresholds import (  # noqa: E402
    DEFAULT_DEDUP_EVERY_N,
    DEFAULT_FACT_EXTRACTION_EVERY_N,
    DEFAULT_PROCEDURAL_SYNTHESIS_AUTO,
    DEFAULT_THREAD_SUMMARY_EVERY_N,
    DEFAULT_USER_SUMMARY_EVERY_N,
)

DEFAULT_MAX_BATCH_SIZE = 20


def _parse_threshold(name: str, default: int) -> int:
    """Parse an integer threshold env var.

    Returns ``default`` when the env var is unset, empty, or invalid (with a
    warning logged on invalid input). Negative values are also rejected and
    fall back to ``default`` so misconfiguration is explicit instead of
    silently disabling the orchestrator. Use this for the ``*_EVERY_N`` knobs
    where ``"0"`` is a valid explicit-disable value but anything else
    nonsensical should fall back.
    """
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    try:
        parsed = int(raw)
    except (ValueError, TypeError):
        logger.warning(
            "Invalid value for %s=%r, using default %d",
            name,
            raw,
            default,
        )
        return default
    if parsed < 0:
        logger.warning(
            "Negative value for %s=%r is not allowed; using default %d (set to 0 to explicitly disable)",
            name,
            raw,
            default,
        )
        return default
    return parsed


def _parse_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    try:
        return int(raw)
    except (ValueError, TypeError):
        logger.warning("Invalid value for %s=%r, using default %d", name, raw, default)
        return default


def _parse_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    try:
        return float(raw)
    except (ValueError, TypeError):
        logger.warning("Invalid value for %s=%r, using default %f", name, raw, default)
        return default


def _parse_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    normalized = raw.strip().lower()
    if normalized in {"true", "1", "yes", "on"}:
        return True
    if normalized in {"false", "0", "no", "off"}:
        return False
    logger.warning("Invalid value for %s=%r, using default %s", name, raw, default)
    return default


def get_max_batch_size() -> int:
    return _parse_int("MAX_BATCH_SIZE", DEFAULT_MAX_BATCH_SIZE)


def get_thread_summary_every_n() -> int:
    """Threshold for triggering ``ThreadSummaryOrchestrator``. ``0`` disables."""
    return _parse_threshold(
        "THREAD_SUMMARY_EVERY_N",
        DEFAULT_THREAD_SUMMARY_EVERY_N,
    )


def get_fact_extraction_every_n() -> int:
    """Threshold for triggering ``ExtractMemoriesOrchestrator``. ``0`` disables."""
    return _parse_threshold(
        "FACT_EXTRACTION_EVERY_N",
        DEFAULT_FACT_EXTRACTION_EVERY_N,
    )


def get_user_summary_every_n() -> int:
    """Threshold for triggering ``UserSummaryOrchestrator``. ``0`` disables."""
    return _parse_threshold(
        "USER_SUMMARY_EVERY_N",
        DEFAULT_USER_SUMMARY_EVERY_N,
    )


def get_dedup_every_n() -> int:
    """Threshold (in extract cycles) for triggering reconciliation. ``0`` disables.

    Reconcile fires every ``FACT_EXTRACTION_EVERY_N * DEDUP_EVERY_N`` turns,
    matching the SDK auto-trigger contract.
    """
    return _parse_threshold(
        "DEDUP_EVERY_N",
        DEFAULT_DEDUP_EVERY_N,
    )


def get_procedural_synthesis_auto() -> bool:
    """Enable chained procedural synthesis after extraction."""
    return _parse_bool(
        "PROCEDURAL_SYNTHESIS_AUTO",
        DEFAULT_PROCEDURAL_SYNTHESIS_AUTO,
    )


def get_cosmos_endpoint() -> str:
    """Return the Cosmos data-plane endpoint.

    The trigger binding uses ``COSMOS_DB__accountEndpoint`` (Azure Functions
    identity-based connection convention); all of our own clients use the
    plain ``COSMOS_DB_ENDPOINT`` env var.
    """
    endpoint = os.environ.get("COSMOS_DB_ENDPOINT") or os.environ.get("COSMOS_DB__accountEndpoint")
    if not endpoint:
        raise RuntimeError("COSMOS_DB_ENDPOINT (or COSMOS_DB__accountEndpoint) is not configured")
    return endpoint


def get_ai_foundry_endpoint() -> str:
    endpoint = os.environ.get("AI_FOUNDRY_ENDPOINT")
    if not endpoint:
        raise RuntimeError("AI_FOUNDRY_ENDPOINT is not configured")
    return endpoint


def get_chat_deployment_name() -> str:
    return os.environ.get("AI_FOUNDRY_CHAT_DEPLOYMENT_NAME", "gpt-4o-mini")


def get_embedding_deployment_name() -> str:
    return os.environ.get("AI_FOUNDRY_EMBEDDING_DEPLOYMENT_NAME") or "text-embedding-3-large"
