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
    "PROCESSOR_OWNER_INPROCESS",
    "PROCESSOR_OWNER_DURABLE",
    "get_fact_extraction_every_n",
    "get_thread_summary_every_n",
    "get_user_summary_every_n",
    "get_processor_owner",
]
