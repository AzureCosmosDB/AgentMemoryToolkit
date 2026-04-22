"""Shared utilities for the Agent Memory Toolkit.

Houses helpers used by both the sync and async clients to avoid
duplication and hidden cross-module coupling.
"""

from __future__ import annotations

import os
import uuid
from datetime import datetime, timezone
from typing import Any, Optional

from ._query_builder import _QueryBuilder
from .exceptions import ConfigurationError, ValidationError

# ---------------------------------------------------------------------------
# Validation constants
# ---------------------------------------------------------------------------

VALID_ROLES = {"agent", "user", "tool", "system"}
VALID_TYPES = {"turn", "summary", "fact", "user_summary"}


# ---------------------------------------------------------------------------
# Memory factory
# ---------------------------------------------------------------------------


def _make_memory(
    user_id: str,
    role: str,
    content: str,
    memory_type: str = "turn",
    agent_id: Optional[str] = None,
    metadata: Optional[dict[str, Any]] = None,
    memory_id: Optional[str] = None,
    thread_id: Optional[str] = None,
) -> dict[str, Any]:
    """Create a validated memory dict."""
    if role not in VALID_ROLES:
        raise ValidationError(f"role must be one of {VALID_ROLES}, got '{role}'")
    if memory_type not in VALID_TYPES:
        raise ValidationError(f"type must be one of {VALID_TYPES}, got '{memory_type}'")

    memory: dict[str, Any] = {
        "id": memory_id or str(uuid.uuid4()),
        "user_id": user_id,
        "thread_id": thread_id or str(uuid.uuid4()),
        "role": role,
        "type": memory_type,
        "content": content,
        "metadata": metadata or {},
        "created_at": datetime.now(timezone.utc).isoformat(),
    }

    if agent_id is not None:
        memory["agent_id"] = agent_id

    return memory


def _resolve_embedding_dimensions(val: Optional[int]) -> Optional[int]:
    """Resolve embedding dimensions from explicit value or ``EMBEDDING_DIMENSIONS`` env var."""
    if val is not None:
        return val
    raw = os.environ.get("EMBEDDING_DIMENSIONS", "0") or "0"
    parsed = int(raw)
    return parsed if parsed else None


def _resolve_cosmos_throughput_mode(val: Optional[str]) -> str:
    """Resolve throughput mode from explicit value or env var.

    Allowed values are ``serverless`` and ``autoscale``.
    """
    raw = (val if val is not None else os.environ.get("COSMOS_DB_THROUGHPUT_MODE") or "serverless").strip().lower()

    if raw not in {"serverless", "autoscale"}:
        raise ConfigurationError(
            message=(
                f"Invalid configuration for cosmos_throughput_mode: expected 'serverless' or 'autoscale', got '{raw}'"
            ),
            parameter="cosmos_throughput_mode",
        )
    return raw


def _resolve_cosmos_autoscale_max_ru(val: Optional[int]) -> int:
    """Resolve autoscale max RU from explicit value or env var."""
    if val is not None:
        if not isinstance(val, int) or isinstance(val, bool) or val <= 0:
            raise ConfigurationError(
                message=f"Invalid configuration for cosmos_autoscale_max_ru: expected a positive integer, got '{val}'",
                parameter="cosmos_autoscale_max_ru",
            )
        return val
    raw = (os.environ.get("COSMOS_DB_AUTOSCALE_MAX_RU") or "1000").strip()
    try:
        parsed = int(raw)
    except ValueError as exc:
        raise ConfigurationError(
            message=(f"Invalid configuration for cosmos_autoscale_max_ru: expected an integer, got '{raw}'"),
            parameter="cosmos_autoscale_max_ru",
        ) from exc
    if parsed <= 0:
        raise ConfigurationError(
            message=(f"Invalid configuration for cosmos_autoscale_max_ru: expected a positive integer, got '{raw}'"),
            parameter="cosmos_autoscale_max_ru",
        )
    return parsed


def _resolve_cosmos_provisioning_autoscale_max_ru(
    *,
    throughput_mode: str,
    autoscale_max_ru: Optional[int],
) -> Optional[int]:
    """Resolve autoscale max RU only when autoscale throughput is enabled."""
    if throughput_mode != "autoscale":
        return None
    return _resolve_cosmos_autoscale_max_ru(autoscale_max_ru)


def _cosmos_container_offer_throughput(
    *,
    throughput_mode: str,
    autoscale_max_ru: Optional[int],
    throughput_properties_cls: Any,
) -> Any:
    """Return ``None`` for serverless mode or a throughput properties instance for autoscale mode."""
    if throughput_mode == "serverless":
        return None
    if autoscale_max_ru is None:
        raise ConfigurationError(
            message=("Invalid configuration for cosmos_autoscale_max_ru: autoscale mode requires a positive integer"),
            parameter="cosmos_autoscale_max_ru",
        )
    return throughput_properties_cls(auto_scale_max_throughput=autoscale_max_ru)


def _build_container_kwargs(
    *,
    container_id: str,
    partition_key: Any,
    offer_throughput: Optional[Any],
    **extras: Any,
) -> dict[str, Any]:
    """Build kwargs for ``create_container_if_not_exists`` with optional throughput."""
    kwargs: dict[str, Any] = {
        "id": container_id,
        "partition_key": partition_key,
        **extras,
    }
    if offer_throughput is not None:
        kwargs["offer_throughput"] = offer_throughput
    return kwargs


# ---------------------------------------------------------------------------
# Connection / query helpers (shared by sync & async Cosmos clients)
# ---------------------------------------------------------------------------


def _validate_connection(
    endpoint: str | None,
    credential: Any,
    database: str,
    container: str,
) -> None:
    """Raise :class:`ConfigurationError` if any required field is missing."""
    if not endpoint:
        raise ConfigurationError(parameter="endpoint")
    if not credential:
        raise ConfigurationError(parameter="credential")
    if not database:
        raise ConfigurationError(parameter="database")
    if not container:
        raise ConfigurationError(parameter="container")


def _build_memory_query_builder(
    *,
    memory_id: Optional[str] = None,
    user_id: Optional[str] = None,
    thread_id: Optional[str] = None,
    role: Optional[str] = None,
    memory_type: Optional[str] = None,
) -> _QueryBuilder:
    """Return a :class:`_QueryBuilder` pre-loaded with the standard filters."""
    qb = _QueryBuilder()
    qb.add_filter("c.id", "@memory_id", memory_id)
    qb.add_filter("c.user_id", "@user_id", user_id)
    qb.add_filter("c.thread_id", "@thread_id", thread_id)
    qb.add_filter("c.role", "@role", role)
    qb.add_filter("c.type", "@memory_type", memory_type)
    return qb


def _container_policies(
    *,
    embedding_dimensions: int,
    embedding_data_type: str,
    distance_function: str,
    full_text_language: str,
) -> tuple[dict, dict, dict]:
    """Build the vector, indexing, and full-text policies for container creation."""
    vector_embedding_policy = {
        "vectorEmbeddings": [
            {
                "path": "/embedding",
                "dataType": embedding_data_type,
                "distanceFunction": distance_function,
                "dimensions": embedding_dimensions,
            }
        ]
    }

    indexing_policy = {
        "includedPaths": [{"path": "/*"}],
        "excludedPaths": [{"path": "/embedding/*"}],
        "vectorIndexes": [{"path": "/embedding", "type": "quantizedFlat"}],
        "fullTextIndexes": [{"path": "/content"}],
    }

    full_text_policy = {
        "defaultLanguage": full_text_language,
        "fullTextPaths": [{"path": "/content", "language": full_text_language}],
    }

    return vector_embedding_policy, indexing_policy, full_text_policy


def _validate_hybrid_search(
    hybrid_search: bool,
    search_terms: Optional[str],
) -> None:
    """Raise :class:`ValidationError` if hybrid search is requested without search terms."""
    if hybrid_search and not search_terms:
        raise ValidationError("search_terms is required when hybrid_search is True")
