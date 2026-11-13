"""Shared helpers for memory search query construction.

Used by both :class:`azure.cosmos.agent_memory.store.memory_store.MemoryStore` and
:class:`azure.cosmos.agent_memory.aio.store.memory_store.AsyncMemoryStore` to keep
search SQL building and result formatting in one place.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from datetime import datetime, timezone
from typing import Any, Optional

from azure.cosmos.agent_memory._query_builder import _QueryBuilder
from azure.cosmos.agent_memory.exceptions import ConfigurationError, ValidationError

MEMORY_PROJECTION = (
    "c.id, c.user_id, c.thread_id, c.role, c.type, c.content, "
    "c.metadata, c.created_at, c.tags, c.salience, c.confidence, "
    "c.name, c.summary, c.procedure_kind, c.applies, c.status, c.valid_until, c.activation_conditions, c.steps, "
    "c.title, c.started_at, c.ended_at, c.participants, c.events, "
    "c.assertion, c.subject, c.attribute, c.entities, "
    "c.outcome, c.lessons, c.source_turn_ids, "
    "c.superseded_by, c.superseded_at, c.supersede_reason, "
    "c.merged_from"
)


def require_search_terms(search_terms: Optional[str], query: Optional[str] = None) -> str:
    terms = query if query is not None else search_terms
    if not terms or not terms.strip():
        raise ValidationError("search_terms must be a non-empty string")
    return terms


def top_literal(value: int, *, name: str) -> int:
    try:
        top = int(value)
    except (TypeError, ValueError) as exc:
        raise ValidationError(f"{name} must be a positive integer") from exc
    if top <= 0:
        raise ValidationError(f"{name} must be a positive integer")
    return top


def add_tag_filters(
    qb: _QueryBuilder,
    *,
    tags_all: Optional[list[str]],
    tags_any: Optional[list[str]],
    exclude_tags: Optional[list[str]],
) -> None:
    if tags_all:
        for i, tag in enumerate(tags_all):
            qb.add_array_contains("c.tags", f"@tag_{i}", tag)
    if tags_any:
        qb.add_array_contains_any("c.tags", "@any_tag_", tags_any)
    if exclude_tags:
        for i, tag in enumerate(exclude_tags):
            qb.add_not_array_contains("c.tags", f"@exc_tag_{i}", tag)


def add_salience_filter(qb: _QueryBuilder, min_salience: Optional[float]) -> None:
    if min_salience is not None:
        qb.add_gte("c.salience", "@min_salience", min_salience)


# The procedures a user-facing search shows: active ones not past ``valid_until`` (an ISO date).
# Other memory types pass through unchanged.
PROCEDURES_IN_FORCE = (
    "(c.type != @procedure_type OR (c.status = @procedure_status "
    "AND (NOT IS_DEFINED(c.valid_until) OR IS_NULL(c.valid_until) OR c.valid_until >= @procedure_today)))"
)


def add_result_filters(
    qb: _QueryBuilder,
    *,
    active_procedures_only: bool = False,
    categories: Optional[list[str]] = None,
    applies: Optional[str] = None,
) -> None:
    """Limit a search to procedures in force, to facts of ``categories``, or to procedures that apply ``applies``."""
    if active_procedures_only:
        qb.add_condition(
            PROCEDURES_IN_FORCE,
            [
                {"name": "@procedure_type", "value": "procedural"},
                {"name": "@procedure_status", "value": "active"},
                {"name": "@procedure_today", "value": datetime.now(timezone.utc).date().isoformat()},
            ],
        )
    if categories:
        qb.add_in_filter("c.metadata.category", "@category_", list(categories))
    if applies:
        qb.add_filter("c.applies", "@applies", applies)


# The BM25 statistics scope of a keyword search: ``Local`` takes them from the partition ranges the query targets
# instead of running a statistics query against every range in the container.
FULL_TEXT_SCORE_SCOPE = "Local"

# SQL function names are case-insensitive and may be followed by whitespace before the parenthesis.
_FULL_TEXT_SCORE_CALL = re.compile(r"fulltextscore\s*\(", re.IGNORECASE)


def uses_full_text_score(query: str) -> bool:
    """Return whether ``query`` calls ``FullTextScore``."""
    return _FULL_TEXT_SCORE_CALL.search(query) is not None


def query_scope(user_id: Optional[str], thread_id: Optional[str], keyword_count: int = 0) -> tuple[Any, bool]:
    """Return ``(partition_key, needs_cross_partition)`` for a read.

    A search with keywords (``keyword_count`` > 0) ranks with ``FullTextScore``. The Python SDK runs a query sent with a
    partition key without planning it, and Cosmos DB then ignores the ranking: every question gets the same fixed order
    (Azure/azure-sdk-for-python#42241). Such a search therefore runs cross-partition: the ``c.user_id`` and
    ``c.thread_id`` filters in its ``WHERE`` clause confine it to the records the key would have read. A vector-only
    search ranks correctly either way and keeps the key.
    """
    if user_id is not None and thread_id is not None and not keyword_count:
        return [user_id, thread_id], False
    return None, True


def coerce_embedding(result: Any) -> list[float]:
    if result is None:
        raise ConfigurationError("Embedder returned no vector", parameter="embeddings_client")
    if isinstance(result, list) and result and isinstance(result[0], (int, float)):
        return result
    if isinstance(result, list) and not result:
        raise ConfigurationError(
            "Embedder returned an empty vector - likely an upstream embedding failure",
            parameter="embeddings_client",
        )
    raise ConfigurationError("Embedder must return list[float]", parameter="embeddings_client")


def format_episodic_context(memories: Iterable[dict[str, Any]]) -> str:
    memories_list = list(memories)
    if not memories_list:
        return ""
    lines = ["## Relevant Past Experiences"]
    for i, memory in enumerate(memories_list, 1):
        title = memory.get("title") or "Episode"
        outcome = memory.get("outcome") or {}
        status = outcome.get("status", "unknown") if isinstance(outcome, dict) else "unknown"
        lines.append(f"{i}. [{status}] {title}: {memory.get('content', '')}")
    return "\n".join(lines)


def build_search_sql(
    *,
    qb: _QueryBuilder,
    top: int,
    keyword_count: int,
    include_superseded: bool,
) -> str:
    """Build the search SQL.

    When ``keyword_count > 0`` the query is hybrid: ``RANK RRF`` fuses the vector
    distance with ``FullTextScore`` over ``keyword_count`` individual keyword
    parameters (``@kw0``..``@kw{n-1}``). When there are no keywords (e.g. an
    all-stopword query) it falls back to pure vector ranking. ``similarity_score``
    is always the vector distance and is *not* the RRF ranking basis under hybrid.
    """
    if not include_superseded:
        qb.add_is_null_or_undefined("c.superseded_by")
    vector_distance = "VectorDistance(c.embedding, @embedding)"
    if keyword_count > 0:
        keyword_params = ", ".join(f"@kw{i}" for i in range(keyword_count))
        order_by = f"ORDER BY RANK RRF({vector_distance}, FullTextScore(c.content, {keyword_params}))"
    else:
        order_by = f"ORDER BY {vector_distance}"
    return (
        f"SELECT TOP {top} {MEMORY_PROJECTION}, "
        f"{vector_distance} AS similarity_score "
        f"FROM c{qb.build_where()} {order_by}"
    )
