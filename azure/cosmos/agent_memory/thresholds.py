"""SDK-side defaults for processing thresholds.

Mirror the function-app side (``function_app/shared/config.py``) so the
InProcess and Durable backends fire on the same turn boundaries by default.
Operators override via the documented env vars; both backends read the same
keys, so a single setting flips both.

"""

from __future__ import annotations

import math
import os
from typing import Optional

from azure.cosmos.agent_memory.logging import get_logger

logger = get_logger(__name__)

DEFAULT_FACT_EXTRACTION_EVERY_N = 2
DEFAULT_THREAD_SUMMARY_EVERY_N = 10
# Episodic memory is boundary-based, not turn-cadence: the turn stream is
# segmented into coherent experiences and each *closed* segment becomes one
# immutable episode. EPISODE_EVAL_EVERY_N is the cheap boundary-evaluation
# cadence (how often we CHECK for a boundary), NOT how often we create an
# episode. 0 disables episodic memory entirely. When enabled, boundaries are
# detected automatically from an idle time-gap, a topic-drift shift, or a
# max-size safety cap - the caller never has to signal "session end". Default 4
# checks the open segment every 4 turns: responsive enough to close a boundary
# promptly while keeping the per-check overhead (a turn query + drift embeddings)
# low. Set to 0 to disable episodic memory.
DEFAULT_EPISODE_EVAL_EVERY_N = 4
# A gap larger than this many seconds between two consecutive turns closes the
# open episode (natural session / idle boundary).
DEFAULT_EPISODE_IDLE_GAP_SECONDS = 1800
# Cosine distance from the open segment's centroid past which a new turn counts
# as a topic/goal shift and closes the prior episode. Requires embeddings.
# Default 0 (OFF): episodes are segmented purely by the idle time-gap and the
# max-size cap, so a dated multi-session conversation yields one episode per
# session. Set to a positive value (e.g. 0.35) to additionally split a long
# single-session run into per-topic episodes; this is heuristic and adds one
# embedding pass over the open segment per boundary evaluation.
DEFAULT_EPISODE_TOPIC_DRIFT = 0.0
# Hard cap on an open segment: force a boundary so neither an episode nor its
# extraction prompt grows unbounded during a long single-topic session.
DEFAULT_EPISODE_MAX_TURNS = 40
# Minimum turns before a drift signal may close an episode, and a floor on all
# natural boundaries: idle-gap and drift boundaries below this many turns are
# suppressed so a lone turn is not emitted as a trivial episode (an explicit
# flush still drains a sub-min trailing segment). The max-size cap is a hard
# ceiling and is not floored.
DEFAULT_EPISODE_MIN_TURNS = 2
DEFAULT_USER_SUMMARY_EVERY_N = 20
# Dedup runs on its own cadence - every Nth extract (NOT every Nth turn),
# because dedup is O(N²) over all active facts and dominates per-push cost
# when FACT_EXTRACTION_EVERY_N=1. Default 5 = one dedup sweep per 5 extracts.
# Set to 1 to dedup on every extract; set to 0 to disable entirely.
DEFAULT_DEDUP_EVERY_N = 5
# Pool size for the auto-trigger reconcile sweep. Mirrors the ``n``
# parameter of :py:meth:`ProcessingPipeline.reconcile_memories`. Hard cap
# of 500 (enforced by the pipeline) bounds prompt size and LLM cost.
DEFAULT_DEDUP_POOL_SIZE = 50
# ---------------------------------------------------------------------------
# INTERNAL dedup/search tuning - NOT customer-configurable.
# These ship as fixed feature constants (no env vars, not in any settings
# template). They are maintainer-tunable here in code only; if a knob ever
# needs to become operator-facing we add the env plumbing back deliberately.
# ---------------------------------------------------------------------------
EXTRACTION_BATCH_MAX_TOKENS = 7000

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

# Whether raw conversation turns are embedded on write so they can be vector
# searched. Default ``False`` preserves today's behavior (turns are stored
# without an ``embedding`` field). The turns container always carries the
# vector index, so this only governs whether vectors are generated/searched.
DEFAULT_ENABLE_TURN_EMBEDDINGS = False

# Owner exclusivity - declares which backend is authoritative for the shared
# memories + counter container. When set, the *other* backend skips its
# auto-trigger and logs a loud warning. Default unset means no owner enforcement,
# so a lone SDK or Function App deployment still processes (the FA defaults to skip).
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


def _parse_threshold_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    try:
        parsed = float(raw)
    except (ValueError, TypeError):
        logger.warning("Invalid value for %s=%r, using default %s", name, raw, default)
        return default
    if not math.isfinite(parsed):
        logger.warning(
            "Non-finite value for %s=%r is not allowed; using default %s",
            name,
            raw,
            default,
        )
        return default
    if parsed < 0:
        logger.warning(
            "Negative value for %s=%r is not allowed; using default %s (set to 0 to disable)",
            name,
            raw,
            default,
        )
        return default
    return parsed


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


def get_episode_eval_every_n() -> int:
    """Boundary-evaluation cadence in turns. 0 disables episodic memory entirely.

    This is how often we cheaply CHECK the open segment for a boundary, not how
    often an episode is created; an episode is created only when a boundary is
    actually detected (idle gap, topic drift, or max-size cap).
    """
    return _parse_threshold("EPISODE_EVAL_EVERY_N", DEFAULT_EPISODE_EVAL_EVERY_N)


def get_episode_idle_gap_seconds() -> int:
    return _parse_threshold("EPISODE_IDLE_GAP_SECONDS", DEFAULT_EPISODE_IDLE_GAP_SECONDS)


def get_episode_topic_drift() -> float:
    return _parse_threshold_float("EPISODE_TOPIC_DRIFT", DEFAULT_EPISODE_TOPIC_DRIFT)


def get_episode_max_turns() -> int:
    return _parse_threshold("EPISODE_MAX_TURNS", DEFAULT_EPISODE_MAX_TURNS)


def get_episode_min_turns() -> int:
    return _parse_threshold("EPISODE_MIN_TURNS", DEFAULT_EPISODE_MIN_TURNS)


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
        # 0 isn't meaningful for a pool size - fall back to default.
        logger.warning(
            "DEDUP_POOL_SIZE=0 is invalid for a pool size; using default %d",
            DEFAULT_DEDUP_POOL_SIZE,
        )
        return DEFAULT_DEDUP_POOL_SIZE
    if raw > 500:
        logger.warning("DEDUP_POOL_SIZE=%d exceeds hard cap; clamping to 500", raw)
        return 500
    return raw


# ---------------------------------------------------------------------------
# Internal dedup/search feature accessors - return fixed constants (no env).
# Kept as thin functions so call sites stay stable and the values can be
# changed in one place; NOT customer-configurable.
# ---------------------------------------------------------------------------
def get_extraction_batch_max_tokens() -> int:
    """Token budget per extraction batch (internal; see EXTRACTION_BATCH_MAX_TOKENS)."""
    return EXTRACTION_BATCH_MAX_TOKENS


def get_procedural_synthesis_auto() -> bool:
    """Whether procedural synthesis auto-fires after extract.

    Set ``PROCEDURAL_SYNTHESIS_AUTO=false`` to disable chained synthesis in
    function-app flows. In-process customers can still call
    :meth:`CosmosMemoryClient.synthesize_procedural` explicitly with
    ``force=True``.
    """
    return _parse_bool("PROCEDURAL_SYNTHESIS_AUTO", DEFAULT_PROCEDURAL_SYNTHESIS_AUTO)


def get_enable_turn_embeddings() -> bool:
    """Whether raw turns are embedded on write and made vector-searchable.

    Set ``ENABLE_TURN_EMBEDDINGS=true`` to generate embeddings for ``turn``
    documents (so ``search_turns()`` returns ranked turns). Default
    ``False`` keeps turns un-embedded. The turns container always carries the
    vector index, so enabling this never requires recreating the container.
    """
    return _parse_bool("ENABLE_TURN_EMBEDDINGS", DEFAULT_ENABLE_TURN_EMBEDDINGS)


def get_processor_owner() -> Optional[str]:
    """Return the configured ``MEMORY_PROCESSOR_OWNER`` or ``None``.

    Each side reads this to decide whether to run its auto-trigger. The
    contract is **asymmetric** by design - there is no cross-process lock,
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
       observed owner disagrees with the writer - treat that as a
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
    "DEFAULT_EPISODE_EVAL_EVERY_N",
    "DEFAULT_EPISODE_IDLE_GAP_SECONDS",
    "DEFAULT_EPISODE_TOPIC_DRIFT",
    "DEFAULT_EPISODE_MAX_TURNS",
    "DEFAULT_EPISODE_MIN_TURNS",
    "DEFAULT_USER_SUMMARY_EVERY_N",
    "DEFAULT_DEDUP_EVERY_N",
    "DEFAULT_DEDUP_POOL_SIZE",
    "DEFAULT_TTL_BY_TYPE",
    "DEFAULT_PROCEDURAL_SYNTHESIS_AUTO",
    "DEFAULT_ENABLE_TURN_EMBEDDINGS",
    "PROCESSOR_OWNER_INPROCESS",
    "PROCESSOR_OWNER_DURABLE",
    "default_ttl_for",
    "get_fact_extraction_every_n",
    "get_thread_summary_every_n",
    "get_episode_eval_every_n",
    "get_episode_idle_gap_seconds",
    "get_episode_topic_drift",
    "get_episode_max_turns",
    "get_episode_min_turns",
    "get_user_summary_every_n",
    "get_dedup_every_n",
    "get_dedup_pool_size",
    "get_extraction_batch_max_tokens",
    "get_procedural_synthesis_auto",
    "get_enable_turn_embeddings",
    "get_processor_owner",
]
