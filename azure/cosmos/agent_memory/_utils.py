"""Shared utilities for the Agent Memory Toolkit.

Houses helpers used by both the sync and async clients to avoid
duplication and hidden cross-module coupling.
"""

from __future__ import annotations

import hashlib
import os
import re
import uuid
from datetime import datetime, timezone
from typing import Any, Optional
from urllib.parse import urlsplit, urlunsplit

from ._container_routing import USER_SCOPED_MEMORIES_TYPES
from ._query_builder import _QueryBuilder
from .exceptions import ConfigurationError, ValidationError
from .thresholds import DEFAULT_TTL_BY_TYPE as DEFAULT_TTL_BY_TYPE
from .thresholds import default_ttl_for

VALID_ROLES = {"agent", "user", "tool", "system"}
VALID_TYPES = {"turn", "thread_summary", "fact", "user_summary", "procedural", "episodic"}


def new_id(memory_type: str) -> str:
    """Return a fresh, type-prefixed UUID-backed memory id."""
    prefix_map = {
        "fact": "fact_",
        "episodic": "ep_",
        "procedural": "proc_",
        "thread_summary": "summary_",
        "user_summary": "user_summary_",
    }
    prefix = prefix_map.get(memory_type, "")
    return f"{prefix}{uuid.uuid4()}"


def new_fact_id() -> str:
    """Return a fresh ``fact_*`` id."""
    return new_id("fact")


def new_episodic_id() -> str:
    """Return a fresh ``ep_*`` id."""
    return new_id("episodic")


def new_procedural_id() -> str:
    """Return a fresh ``proc_*`` id."""
    return new_id("procedural")


def new_thread_summary_id() -> str:
    """Return a fresh ``summary_*`` id for a thread-summary doc."""
    return new_id("thread_summary")


def new_user_summary_id() -> str:
    """Return a fresh ``user_summary_*`` id."""
    return new_id("user_summary")


_WHITESPACE_RE = re.compile(r"\s+")


def _coerce_datetime_iso(value: Optional[str | datetime]) -> Optional[str]:
    """Return ISO text for datetime values while leaving strings unchanged."""
    if isinstance(value, datetime):
        return value.isoformat()
    return value


def _normalize_for_hash(text: str) -> str:
    """Lowercase + collapse whitespace for write-time exact-dedup.

    Deliberately conservative: lowercase, strip, and collapse internal runs
    of whitespace to a single space. Punctuation and word order still matter.
    The point is to catch *identical* re-extractions cheaply — paraphrases
    are handled by the reconciliation LLM pass.
    """
    return _WHITESPACE_RE.sub(" ", text.strip().lower())


def compute_content_hash(content: str) -> str:
    """SHA-256 of normalized text, truncated to 32 hex chars.

    Normalization: lowercase + whitespace collapse (see ``_normalize_for_hash``).
    32 chars (128 bits) is plenty for collision avoidance within a single
    user's memory set and keeps the field compact in Cosmos documents.
    Used uniformly across facts, procedural, and episodic memories so the
    ``content_hash`` field has a single, stable shape regardless of type.
    """
    return hashlib.sha256(_normalize_for_hash(content).encode("utf-8")).hexdigest()[:32]


def _make_memory(
    user_id: str,
    role: str,
    content: str,
    memory_type: str = "turn",
    agent_id: Optional[str] = None,
    metadata: Optional[dict[str, Any]] = None,
    memory_id: Optional[str] = None,
    thread_id: Optional[str] = None,
    tags: Optional[list[str]] = None,
    ttl: Optional[int] = None,
    salience: Optional[float] = None,
    content_hash: Optional[str] = None,
) -> dict[str, Any]:
    """Create a validated memory dict."""
    if role not in VALID_ROLES:
        raise ValidationError(f"role must be one of {VALID_ROLES}, got '{role}'")
    if memory_type not in VALID_TYPES:
        raise ValidationError(f"type must be one of {VALID_TYPES}, got '{memory_type}'")

    if ttl is None:
        ttl = default_ttl_for(memory_type)

    memory: dict[str, Any] = {
        "id": memory_id or str(uuid.uuid4()),
        "user_id": user_id,
        "thread_id": thread_id or str(uuid.uuid4()),
        "role": role,
        "type": memory_type,
        "content": content,
        "metadata": metadata or {},
        "created_at": datetime.now(timezone.utc).isoformat(),
        "tags": tags if tags is not None else [],
    }

    if agent_id is not None:
        memory["agent_id"] = agent_id
    if ttl is not None:
        memory["ttl"] = ttl
    if salience is not None:
        memory["salience"] = salience
    if content_hash is not None:
        memory["content_hash"] = content_hash

    return memory


def _resolve_embedding_dimensions(val: Optional[int]) -> int:
    """Resolve embedding dimensions from explicit value or ``AI_FOUNDRY_EMBEDDING_DIMENSIONS`` env var.

    Defaults to 1536 (the dimension we ship with for ``text-embedding-3-large``
    truncated to 1536, which is the size our quantizedFlat vector indexes are
    tuned for in our containers).

    Raises :class:`ConfigurationError` if the env var is set but cannot be
    parsed as a positive integer.
    """
    if val is not None:
        return val
    raw = os.environ.get("AI_FOUNDRY_EMBEDDING_DIMENSIONS")
    if raw is None or raw == "":
        return 1536
    try:
        parsed = int(raw)
    except (ValueError, TypeError) as exc:
        raise ConfigurationError(
            message=(
                f"Invalid configuration for embedding_dimensions: AI_FOUNDRY_EMBEDDING_DIMENSIONS"
                f" must be a positive integer, got {raw!r}"
            ),
            parameter="embedding_dimensions",
        ) from exc
    if parsed <= 0:
        raise ConfigurationError(
            message=(f"Invalid configuration for embedding_dimensions: must be a positive integer, got {parsed}"),
            parameter="embedding_dimensions",
        )
    return parsed


_AI_FOUNDRY_PROJECT_PATH_RE = re.compile(r"/api/projects/[^/]+/?.*$", re.IGNORECASE)
_AI_FOUNDRY_HOST_SUFFIX = ".services.ai.azure.com"


def normalize_ai_foundry_endpoint(endpoint: Optional[str]) -> Optional[str]:
    """Normalize an AI Foundry / Azure OpenAI endpoint to the inference base URL.

    The toolkit reaches the model inference API through the OpenAI SDK
    (``AzureOpenAI(azure_endpoint=...)``), which expects the account-level
    inference endpoint, for example::

        https://<resource>.services.ai.azure.com
        https://<resource>.openai.azure.com

    The Azure AI Foundry portal, however, commonly surfaces a *project*-scoped
    endpoint of the form::

        https://<resource>.services.ai.azure.com/api/projects/<project-name>

    For ``*.services.ai.azure.com`` resources the project path lives on the same
    host that serves inference, so this helper strips a trailing
    ``/api/projects/<name>`` segment (plus any surrounding whitespace or trailing
    slash) to recover the base inference endpoint. Callers can therefore paste
    either form.

    The project-path stripping is applied **only** when the URL host ends with
    ``.services.ai.azure.com``, and only to the path component, so unrelated
    endpoints that happen to contain ``/api/projects/...`` in their path are left
    untouched. Endpoints that don't carry a project path are returned unchanged
    aside from whitespace/trailing-slash trimming, so non-Foundry endpoints keep
    working. ``None``/empty values are passed through untouched.
    """
    if not endpoint:
        return endpoint
    trimmed = endpoint.strip()
    parts = urlsplit(trimmed)
    host = parts.hostname or ""
    if host.lower().endswith(_AI_FOUNDRY_HOST_SUFFIX):
        new_path = _AI_FOUNDRY_PROJECT_PATH_RE.sub("", parts.path)
        trimmed = urlunsplit((parts.scheme, parts.netloc, new_path, parts.query, parts.fragment))
    return trimmed.rstrip("/")


_ALLOWED_EMBEDDING_DATA_TYPES = ("float32", "uint8", "int8")
_ALLOWED_DISTANCE_FUNCTIONS = ("cosine", "dotproduct", "euclidean")
_ALLOWED_VECTOR_INDEX_TYPES = ("diskANN", "quantizedFlat", "flat")


def _resolve_embedding_data_type(val: Optional[str]) -> str:
    """Resolve embedding data type from explicit value or ``AI_FOUNDRY_EMBEDDING_DATA_TYPE`` env var.

    Defaults to ``float32``. Raises :class:`ConfigurationError` for unknown values.
    """
    raw = (val if val is not None else os.environ.get("AI_FOUNDRY_EMBEDDING_DATA_TYPE") or "float32").strip()
    if raw not in _ALLOWED_EMBEDDING_DATA_TYPES:
        raise ConfigurationError(
            message=(
                f"Invalid configuration for embedding_data_type: must be one of "
                f"{_ALLOWED_EMBEDDING_DATA_TYPES}, got {raw!r}"
            ),
            parameter="embedding_data_type",
        )
    return raw


def _resolve_distance_function(val: Optional[str]) -> str:
    """Resolve distance function from explicit value or ``AI_FOUNDRY_EMBEDDING_DISTANCE_FUNCTION`` env var.

    Defaults to ``cosine``. Raises :class:`ConfigurationError` for unknown values.
    """
    raw = (val if val is not None else os.environ.get("AI_FOUNDRY_EMBEDDING_DISTANCE_FUNCTION") or "cosine").strip()
    if raw not in _ALLOWED_DISTANCE_FUNCTIONS:
        raise ConfigurationError(
            message=(
                f"Invalid configuration for distance_function: must be one of "
                f"{_ALLOWED_DISTANCE_FUNCTIONS}, got {raw!r}"
            ),
            parameter="distance_function",
        )
    return raw


def _resolve_vector_index_type(val: Optional[str]) -> str:
    """Resolve vector index type from explicit value or ``AI_FOUNDRY_EMBEDDING_VECTOR_INDEX_TYPE`` env var.

    Defaults to ``diskANN``. Raises :class:`ConfigurationError` for unknown values.

    ``diskANN`` requires the Cosmos DB account to have the DiskANN vector index
    capability enabled. Accounts that do not (for example the classic Cosmos DB
    emulator) can use ``quantizedFlat`` or ``flat`` instead.
    """
    raw = (val if val is not None else os.environ.get("AI_FOUNDRY_EMBEDDING_VECTOR_INDEX_TYPE") or "diskANN").strip()
    if raw not in _ALLOWED_VECTOR_INDEX_TYPES:
        raise ConfigurationError(
            message=(
                f"Invalid configuration for vector_index_type: must be one of "
                f"{_ALLOWED_VECTOR_INDEX_TYPES}, got {raw!r}"
            ),
            parameter="vector_index_type",
        )
    return raw


def _resolve_full_text_language(val: Optional[str]) -> str:
    """Resolve full-text language from explicit value or ``COSMOS_DB_FULL_TEXT_LANGUAGE`` env var.

    Defaults to ``en-US``. Empty values fall back to the default.
    """
    raw = (val if val is not None else os.environ.get("COSMOS_DB_FULL_TEXT_LANGUAGE") or "en-US").strip()
    return raw or "en-US"


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
    memory_types: Optional[list[str]] = None,
    min_confidence: Optional[float] = None,
) -> _QueryBuilder:
    """Return a :class:`_QueryBuilder` pre-loaded with the standard filters.

    ``memory_types`` is a list of types (e.g. ``["fact", "procedural",
    "episodic"]``); when ``None`` or empty no type filter is applied.
    """
    qb = _QueryBuilder()
    qb.add_filter("c.id", "@memory_id", memory_id)
    qb.add_filter("c.user_id", "@user_id", user_id)
    in_scope_user_types = _resolve_user_scoped_types_in_query(memory_types)
    if thread_id is not None and in_scope_user_types:
        qb.add_thread_id_or_user_scoped(thread_id, "@thread_id", sorted(in_scope_user_types))
    else:
        qb.add_filter("c.thread_id", "@thread_id", thread_id)
    qb.add_filter("c.role", "@role", role)
    if memory_types:
        qb.add_in_filter("c.type", "@memory_type_", list(memory_types))
    if min_confidence is not None and min_confidence > 0:
        qb.add_gte("c.confidence", "@min_confidence", min_confidence)
    return qb


def _resolve_user_scoped_types_in_query(memory_types: Optional[list[str]]) -> set[str]:
    """Return the user-scoped types this query may match."""
    if not memory_types:
        return set(USER_SCOPED_MEMORIES_TYPES)
    return set(memory_types) & USER_SCOPED_MEMORIES_TYPES


def _container_policies(
    *,
    embedding_dimensions: int,
    embedding_data_type: str,
    distance_function: str,
    full_text_language: str,
    include_salience_composite: bool = True,
    vector_index_type: str = "diskANN",
) -> tuple[dict, dict, dict]:
    """Build the vector, indexing, and full-text policies for container creation.

    ``include_salience_composite`` adds the ``(salience, created_at, id)``
    composite index required by procedural synthesis on the MEMORIES container.
    Turns reuse this builder with it disabled (turns are never synthesized).
    """
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
        "excludedPaths": [
            {"path": "/source_memory_ids/*"},
            {"path": "/supersedes_ids/*"},
            {"path": '/"_etag"/?'},
        ],
        "vectorIndexes": [{"path": "/embedding", "type": vector_index_type}],
        "fullTextIndexes": [{"path": "/content"}],
    }

    if include_salience_composite:
        # Procedural synthesis selects TOP N by (salience DESC, created_at ASC, id ASC).
        # Cosmos requires a composite index for multi-property ORDER BY; without it the
        # query returns a non-deterministic 50 of N when many docs share the default
        # salience (0.5), which makes the source-id short-circuit in synthesize_procedural
        # thrash and burn LLM calls on every reconcile.
        indexing_policy["compositeIndexes"] = [
            [
                {"path": "/salience", "order": "descending"},
                {"path": "/created_at", "order": "ascending"},
                {"path": "/id", "order": "ascending"},
            ]
        ]

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
