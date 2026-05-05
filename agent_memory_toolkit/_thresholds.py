"""SDK-side defaults for processing thresholds.

Mirror the function-app side (``function_app/shared/config.py``) so the
InProcess and Durable backends fire on the same turn boundaries by default.
Operators override via the documented env vars; both backends read the same
keys, so a single setting flips both.
"""

from __future__ import annotations

import logging
import os

logger = logging.getLogger(__name__)


# Defaults — match function_app/shared/config.py exactly.
DEFAULT_FACT_EXTRACTION_EVERY_N = 1
DEFAULT_THREAD_SUMMARY_EVERY_N = 10
DEFAULT_USER_SUMMARY_EVERY_N = 20


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


__all__ = [
    "DEFAULT_FACT_EXTRACTION_EVERY_N",
    "DEFAULT_THREAD_SUMMARY_EVERY_N",
    "DEFAULT_USER_SUMMARY_EVERY_N",
    "get_fact_extraction_every_n",
    "get_thread_summary_every_n",
    "get_user_summary_every_n",
]
