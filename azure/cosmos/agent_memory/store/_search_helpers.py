"""Shared helpers for memory search query construction.

Used by both :class:`azure.cosmos.agent_memory.store.memory_store.MemoryStore` and
:class:`azure.cosmos.agent_memory.aio.store.memory_store.AsyncMemoryStore` to keep
search SQL building and result formatting in one place.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any, Optional

from azure.cosmos.agent_memory._partitioning import (
    DEFAULT_TENANT_ID,
    partition_key_for_scope_thread,
    partition_key_for_user_thread,
)
from azure.cosmos.agent_memory._query_builder import _QueryBuilder
from azure.cosmos.agent_memory.exceptions import ConfigurationError, ValidationError

MEMORY_PROJECTION = (
    "c.id, c.thread_id, c.tenant_id, c.scope_type, c.scope_id, c.scope_key, c.role, c.type, c.content, "
    "c.metadata, c.acl, c.provenance, c.created_at, c.tags, c.salience, c.confidence, "
    "c.title, c.started_at, c.ended_at, c.participants, c.events, "
    "c.outcome, c.lessons, c.source_turn_ids, "
    "c.superseded_by, c.superseded_at, c.supersede_reason"
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


def query_scope(
    user_id: Optional[str],
    thread_id: Optional[str],
    tenant_id: Optional[str] = None,
    scope_key: Optional[str] = None,
) -> tuple[Any, bool]:
    if scope_key is not None and thread_id is not None:
        return partition_key_for_scope_thread(scope_key, thread_id, tenant_id), False
    if scope_key is not None:
        return [tenant_id or DEFAULT_TENANT_ID, scope_key], False
    if user_id is not None and thread_id is not None:
        return partition_key_for_user_thread(user_id, thread_id, tenant_id), False
    return None, True


def normalize_scope_keys(scopes: Optional[list[str]]) -> list[str] | None:
    """Strip and de-dupe an explicit scope-key list (order-preserving).

    Returns ``None`` when no scopes were supplied, which signals the caller to fall
    back to the single-user read path.
    """
    if scopes is not None:
        return list(
            dict.fromkeys(str(scope).strip() for scope in scopes if scope is not None and str(scope).strip())
        )
    return None


def add_tenant_scope_filter(qb: _QueryBuilder, *, tenant_id: Optional[str], scope_key: str) -> None:
    qb.add_filter("c.tenant_id", "@tenant_id", tenant_id or DEFAULT_TENANT_ID)
    qb.add_filter("c.scope_key", "@scope_key", scope_key)


def merge_ranked_results(scoped_results: Iterable[dict[str, Any]], *, top_k: int) -> list[dict[str, Any]]:
    deduped: dict[str, dict[str, Any]] = {}

    def score_value(doc: dict[str, Any]) -> float:
        score = doc.get("similarity_score")
        return float(score) if isinstance(score, (int, float)) else float("inf")

    for index, doc in enumerate(scoped_results):
        doc_id = str(doc.get("id") or "")
        if not doc_id:
            continue
        copied = dict(doc)
        copied.setdefault("_scope_union_rank", index)
        existing = deduped.get(doc_id)
        existing_rank = int(existing.get("_scope_union_rank") or 0) if existing is not None else 0
        if existing is None or (score_value(copied), index) < (score_value(existing), existing_rank):
            deduped[doc_id] = copied

    def sort_key(doc: dict[str, Any]) -> tuple[float, int]:
        score = doc.get("similarity_score")
        if isinstance(score, (int, float)):
            return float(score), int(doc.get("_scope_union_rank") or 0)
        return float("inf"), int(doc.get("_scope_union_rank") or 0)

    ranked = sorted(deduped.values(), key=sort_key)[:top_k]
    for doc in ranked:
        doc.pop("_scope_union_rank", None)
    return ranked


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
