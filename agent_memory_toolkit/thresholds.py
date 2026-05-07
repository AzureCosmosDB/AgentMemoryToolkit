"""SDK-side defaults for processing thresholds.

Mirror the function-app side (``function_app/shared/config.py``) so the
InProcess and Durable backends fire on the same turn boundaries by default.
Operators override via the documented env vars; both backends read the same
keys, so a single setting flips both.
"""

from __future__ import annotations

import logging
import os
from typing import Optional

logger = logging.getLogger(__name__)

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


def get_processor_owner() -> Optional[str]:
    """Return the configured ``MEMORY_PROCESSOR_OWNER`` or ``None``.

    Both the SDK and the function app should consult this to decide whether
    to run their auto-trigger. When unset, neither side enforces exclusivity
    (today's behavior). When set to a known value but mismatched, the side
    that does not own the container should skip and log.

    .. note::
       This is **operator-configured exclusivity, not enforced**. Each
       backend reads its own env var; there is no cross-process lock. If
       the SDK has ``inprocess`` but the FA is unset, both will run.
       As a backstop, counter writes stamp ``last_owner`` and a one-shot
       WARN is emitted when the observed owner doesn't match this process's
       owner — treat that as a configuration audit signal, not a guarantee.
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
    "PROCESSOR_OWNER_INPROCESS",
    "PROCESSOR_OWNER_DURABLE",
    "get_fact_extraction_every_n",
    "get_thread_summary_every_n",
    "get_user_summary_every_n",
    "get_dedup_every_n",
    "get_dedup_pool_size",
    "get_processor_owner",
]
