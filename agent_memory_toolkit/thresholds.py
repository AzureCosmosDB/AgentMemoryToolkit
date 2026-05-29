"""SDK-side defaults for processing thresholds.

Mirror the function-app side (``function_app/shared/config.py``) so the
InProcess and Durable backends fire on the same turn boundaries by default.
Operators override via the documented env vars; both backends read the same
keys, so a single setting flips both.
"""

from __future__ import annotations

import os
from typing import Optional

from agent_memory_toolkit.logging import get_logger

logger = get_logger(__name__)

DEFAULT_FACT_EXTRACTION_EVERY_N = 1
DEFAULT_THREAD_SUMMARY_EVERY_N = 10
DEFAULT_USER_SUMMARY_EVERY_N = 20
# Dedup runs on its own cadence — every Nth extract (NOT every Nth turn),
# because dedup is O(N²) over all active facts and dominates per-push cost
# when FACT_EXTRACTION_EVERY_N=1. Default 5 = one dedup sweep per 5 extracts.
# Set to 1 to dedup on every extract; set to 0 to disable entirely.
DEFAULT_DEDUP_EVERY_N = 5
# Pool size for the auto-trigger reconcile sweep. Mirrors the ``n``
# parameter of :py:meth:`ProcessingPipeline.reconcile_memories`. Hard cap
# of 500 (enforced by the pipeline) bounds prompt size and LLM cost.
DEFAULT_DEDUP_POOL_SIZE = 50

DEFAULT_TTL_BY_TYPE: dict[str, int] = {
    "turn": 2_592_000,
    "episodic": 7_776_000,
    "thread_summary": -1,
    "user_summary": -1,
    "fact": -1,
    "procedural": -1,
}

_TRUTHY = {"true", "1", "yes", "on", "t", "y"}
_FALSY = {"false", "0", "no", "off", "f", "n"}

DEFAULT_PROCEDURAL_SYNTHESIS_AUTO = True

# Owner exclusivity — declares which backend is authoritative for the shared
# memories + counter container. When set, the *other* backend skips its
# auto-trigger and logs a loud warning. Default unset preserves today's
# behavior (no enforcement) for backward compatibility.
PROCESSOR_OWNER_INPROCESS = "inprocess"
PROCESSOR_OWNER_DURABLE = "durable"
_VALID_OWNERS = {PROCESSOR_OWNER_INPROCESS, PROCESSOR_OWNER_DURABLE}


def _parse_threshold(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    try:
        parsed = int(raw)
    except (ValueError, TypeError):
        logger.warning("Invalid value for %s=%r, using default %d", name, raw, default)
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


def _parse_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    value = raw.strip().lower()
    if value in _TRUTHY:
        return True
    if value in _FALSY:
        return False
    logger.warning("Invalid value for %s=%r, using default %s", name, raw, default)
    return default


def default_ttl_for(memory_type: str) -> Optional[int]:
    """Return the per-type default TTL, or None for 'use container default'.

    Per-doc ttl=-1 also means 'never'; never-expiring types return None so
    the container default applies and documents stay small. Unknown types do
    not expire by default.
    """
    ttl = DEFAULT_TTL_BY_TYPE.get(memory_type)
    if ttl is None or ttl == -1:
        return None
    return ttl


def get_fact_extraction_every_n() -> int:
    return _parse_threshold("FACT_EXTRACTION_EVERY_N", DEFAULT_FACT_EXTRACTION_EVERY_N)


def get_thread_summary_every_n() -> int:
    return _parse_threshold("THREAD_SUMMARY_EVERY_N", DEFAULT_THREAD_SUMMARY_EVERY_N)


def get_user_summary_every_n() -> int:
    return _parse_threshold("USER_SUMMARY_EVERY_N", DEFAULT_USER_SUMMARY_EVERY_N)


def get_dedup_every_n() -> int:
    """Run dedup once per N extracts. 0 disables dedup auto-trigger entirely."""
    return _parse_threshold("DEDUP_EVERY_N", DEFAULT_DEDUP_EVERY_N)


def get_dedup_pool_size() -> int:
    """Pool size for the auto-trigger reconcile sweep (``n`` param of
    :py:meth:`ProcessingPipeline.reconcile_memories`). Hard-capped at 500 by
    the pipeline; values above are clamped to 500 with a WARN."""
    raw = _parse_threshold("DEDUP_POOL_SIZE", DEFAULT_DEDUP_POOL_SIZE)
    if raw == 0:
        # 0 isn't meaningful for a pool size — fall back to default.
        logger.warning(
            "DEDUP_POOL_SIZE=0 is invalid for a pool size; using default %d",
            DEFAULT_DEDUP_POOL_SIZE,
        )
        return DEFAULT_DEDUP_POOL_SIZE
    if raw > 500:
        logger.warning("DEDUP_POOL_SIZE=%d exceeds hard cap; clamping to 500", raw)
        return 500
    return raw


def get_procedural_synthesis_auto() -> bool:
    """Whether procedural synthesis auto-fires after extract.

    Set ``PROCEDURAL_SYNTHESIS_AUTO=false`` to disable chained synthesis in
    function-app flows. In-process customers can still call
    :meth:`CosmosMemoryClient.synthesize_procedural` explicitly with
    ``force=True``.
    """
    return _parse_bool("PROCEDURAL_SYNTHESIS_AUTO", DEFAULT_PROCEDURAL_SYNTHESIS_AUTO)


def get_processor_owner() -> Optional[str]:
    """Return the configured ``MEMORY_PROCESSOR_OWNER`` or ``None``.

    Each side reads this to decide whether to run its auto-trigger. The
    contract is **asymmetric** by design — there is no cross-process lock,
    so the two sides default differently to avoid double-firing:

      * **SDK** (in-process) fires when the value is ``None`` (unset) or
        ``"inprocess"``; it skips on ``"durable"``. Pure SDK deployments
        therefore work without any environment configuration.
      * **Function App** (durable) fires **only** when the value is
        explicitly ``"durable"``; anything else (including ``None``) causes
        the change-feed trigger to skip. This default-deny posture is what
        prevents both backends from racing on the same writes when an
        operator deploys the FA next to an existing SDK install without
        setting the env var.

    . note::
       This is **operator-configured exclusivity, not enforced**. Counter
       writes still stamp ``last_owner`` and emit a one-shot WARN when the
       observed owner disagrees with the writer — treat that as a
       configuration audit signal, not a guarantee.
    """
    raw = os.environ.get("MEMORY_PROCESSOR_OWNER")
    if raw is None or raw == "":
        return None
    value = raw.strip().lower()
    if value not in _VALID_OWNERS:
        logger.warning(
            "Invalid MEMORY_PROCESSOR_OWNER=%r (expected one of %s); ignoring",
            raw,
            sorted(_VALID_OWNERS),
        )
        return None
    return value


__all__ = [
    "DEFAULT_FACT_EXTRACTION_EVERY_N",
    "DEFAULT_THREAD_SUMMARY_EVERY_N",
    "DEFAULT_USER_SUMMARY_EVERY_N",
    "DEFAULT_DEDUP_EVERY_N",
    "DEFAULT_DEDUP_POOL_SIZE",
    "DEFAULT_TTL_BY_TYPE",
    "DEFAULT_PROCEDURAL_SYNTHESIS_AUTO",
    "PROCESSOR_OWNER_INPROCESS",
    "PROCESSOR_OWNER_DURABLE",
    "default_ttl_for",
    "get_fact_extraction_every_n",
    "get_thread_summary_every_n",
    "get_user_summary_every_n",
    "get_dedup_every_n",
    "get_dedup_pool_size",
    "get_procedural_synthesis_auto",
    "get_processor_owner",
]
