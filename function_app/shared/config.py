"""Configuration helpers for the processor function app.

All knobs are read from environment variables / Azure Functions app settings.
Defaults match the spec (section 8.4 / 11.2):

* ``THREAD_SUMMARY_EVERY_N``       — default 4
* ``FACT_EXTRACTION_EVERY_N``      — default 4
* ``USER_SUMMARY_EVERY_N``         — default 20
* ``MAX_BATCH_SIZE``               — default 20
* ``SALIENCE_THRESHOLD``           — default 0.0  (disabled)

Setting any ``*_EVERY_N`` to ``0`` disables that orchestrator entirely.
``_parse_threshold`` returns ``0`` when the value is missing or invalid so the
caller can rely on a single sentinel for "disabled".
"""

from __future__ import annotations

import logging
import os

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Cosmos DB binding / data-plane endpoints
# ---------------------------------------------------------------------------

CHANGE_FEED_DATABASE = os.environ.get("COSMOS_DB_DATABASE", "ai_memory")
CHANGE_FEED_CONTAINER = os.environ.get("COSMOS_DB_CONTAINER", "memories")
CHANGE_FEED_LEASE_CONTAINER = os.environ.get("COSMOS_DB_LEASE_CONTAINER", "leases")
COUNTERS_CONTAINER = os.environ.get("COSMOS_DB_COUNTERS_CONTAINER", "counter")

USER_COUNTER_THREAD_ID = "__counters__"


# ---------------------------------------------------------------------------
# Defaults documented in local.settings.json.template
# ---------------------------------------------------------------------------

DEFAULT_THREAD_SUMMARY_EVERY_N = 4
DEFAULT_FACT_EXTRACTION_EVERY_N = 4
DEFAULT_USER_SUMMARY_EVERY_N = 20
DEFAULT_MAX_BATCH_SIZE = 20
DEFAULT_SALIENCE_THRESHOLD = 0.0


def _parse_threshold(name: str) -> int:
    """Parse an integer threshold env var. Returns ``0`` if unset/invalid.

    The function is intentionally permissive: we want a misconfigured app to
    *disable* a threshold rather than crash on every change-feed batch. A
    warning is logged so the misconfiguration is visible.
    """
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return 0
    try:
        return int(raw)
    except (ValueError, TypeError):
        logger.warning("Invalid value for %s=%r, defaulting to 0 (disabled)", name, raw)
        return 0


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


def get_max_batch_size() -> int:
    return _parse_int("MAX_BATCH_SIZE", DEFAULT_MAX_BATCH_SIZE)


def get_salience_threshold() -> float:
    return _parse_float("SALIENCE_THRESHOLD", DEFAULT_SALIENCE_THRESHOLD)


def get_cosmos_endpoint() -> str:
    """Return the Cosmos data-plane endpoint.

    The trigger binding uses ``COSMOS_DB__accountEndpoint`` (Azure Functions
    identity-based connection convention); all of our own clients use the
    plain ``COSMOS_DB_ENDPOINT`` env var.
    """
    endpoint = os.environ.get("COSMOS_DB_ENDPOINT") or os.environ.get(
        "COSMOS_DB__accountEndpoint"
    )
    if not endpoint:
        raise RuntimeError(
            "COSMOS_DB_ENDPOINT (or COSMOS_DB__accountEndpoint) is not configured"
        )
    return endpoint


def get_ai_foundry_endpoint() -> str:
    endpoint = os.environ.get("AI_FOUNDRY_ENDPOINT")
    if not endpoint:
        raise RuntimeError("AI_FOUNDRY_ENDPOINT is not configured")
    return endpoint


def get_chat_deployment_name() -> str:
    return os.environ.get("AI_FOUNDRY_CHAT_DEPLOYMENT_NAME", "gpt-4o-mini")


def get_embedding_deployment_name() -> str:
    return (
        os.environ.get("AI_FOUNDRY_EMBEDDING_DEPLOYMENT_NAME")
        or "text-embedding-3-large"
    )
