"""Shared helpers for memory search query construction.

Used by both :class:`agent_memory_toolkit.store.memory_store.MemoryStore` and
:class:`agent_memory_toolkit.aio.store.memory_store.AsyncMemoryStore` to keep
search SQL building and result formatting in one place.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any, Optional

from agent_memory_toolkit._query_builder import _QueryBuilder
from agent_memory_toolkit.exceptions import ConfigurationError, ValidationError

MEMORY_PROJECTION = (
    "c.id, c.user_id, c.thread_id, c.role, c.type, c.content, "
    "c.metadata, c.created_at, c.tags, c.salience, c.confidence, c.superseded_by"
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


def query_scope(user_id: Optional[str], thread_id: Optional[str]) -> tuple[Any, bool]:
    if user_id is not None and thread_id is not None:
        return [user_id, thread_id], False
    return None, True


def coerce_embedding(result: Any) -> list[float]:
    if result is None:
        raise ConfigurationError("Embedder returned no vector", parameter="embeddings_client")
    if isinstance(result, list) and result and isinstance(result[0], (int, float)):
        return result
    if isinstance(result, list) and not result:
        raise ConfigurationError(
            "Embedder returned an empty vector — likely an upstream embedding failure",
            parameter="embeddings_client",
        )
    raise ConfigurationError("Embedder must return list[float]", parameter="embeddings_client")


def format_episodic_context(memories: Iterable[dict[str, Any]]) -> str:
    memories_list = list(memories)
    if not memories_list:
        return ""
    lines = ["## Relevant Past Experiences"]
    for i, memory in enumerate(memories_list, 1):
        metadata = memory.get("metadata") or {}
        domain = metadata.get("domain", "general")
        valence = metadata.get("outcome_valence", "neutral")
        lines.append(f"{i}. [{domain}] {memory['content']} ({valence})")
    return "\n".join(lines)


def build_search_sql(
    *,
    qb: _QueryBuilder,
    top: int,
    hybrid_search: bool,
    include_superseded: bool,
) -> str:
    if not include_superseded:
        qb.add_is_null_or_undefined("c.superseded_by")
    order_by = "ORDER BY VectorDistance(c.embedding, @embedding)"
    if hybrid_search:
        order_by = "ORDER BY RANK RRF(VectorDistance(c.embedding, @embedding), FullTextScore(c.content, @key_terms))"
    return (
        f"SELECT TOP {top} {MEMORY_PROJECTION}, "
        "VectorDistance(c.embedding, @embedding) AS similarity_score "
        f"FROM c{qb.build_where()} {order_by}"
    )
