"""Shared base mixin for sync and async memory clients."""

from __future__ import annotations

import importlib
from datetime import datetime, timezone
from typing import Any, Optional

from azure.cosmos.agent_memory._container_routing import ContainerKey
from azure.cosmos.agent_memory._utils import (
    VALID_ROLES,
    VALID_TYPES,
    _make_memory,
    _resolve_cosmos_provisioning_autoscale_max_ru,
    _resolve_cosmos_throughput_mode,
    _resolve_embedding_dimensions,
    normalize_ai_foundry_endpoint,
)
from azure.cosmos.agent_memory.exceptions import CosmosNotConnectedError, MemoryNotFoundError, ValidationError
from azure.cosmos.agent_memory.logging import configure_logging, get_logger
from azure.cosmos.agent_memory.thresholds import get_enable_turn_embeddings

logger = get_logger(__name__)


class _BaseMemoryClient:
    """Small shared configuration and guard helpers for memory clients."""

    def _init_base_config(
        self,
        *,
        cosmos_endpoint: Optional[str],
        cosmos_credential: Optional[Any],
        cosmos_key: Optional[str],
        cosmos_database: Optional[str],
        cosmos_container: Optional[str],
        cosmos_turns_container: str = "memories_turns",
        cosmos_summaries_container: str = "memories_summaries",
        cosmos_counter_container: Optional[str],
        cosmos_lease_container: Optional[str],
        cosmos_throughput_mode: Optional[str],
        cosmos_autoscale_max_ru: Optional[int],
        ai_foundry_endpoint: Optional[str],
        ai_foundry_credential: Optional[Any],
        ai_foundry_api_key: Optional[str],
        embedding_deployment_name: str,
        embedding_dimensions: Optional[int],
        chat_deployment_name: str,
        use_default_credential: bool,
        enable_turn_embeddings: Optional[bool] = None,
        default_credential_module: str = "azure.identity",
    ) -> None:
        """Initialize shared local state, config values, and default credentials."""
        configure_logging()
        self.local_memory: list[dict[str, Any]] = []
        self._unflushed_turn_counts: dict[tuple[str, str], int] = {}
        self._warned_owner_skip: bool = False
        self._warned_counter_unreachable: bool = False

        self._cosmos_endpoint = cosmos_endpoint
        self._cosmos_credential = cosmos_credential
        self._cosmos_key = cosmos_key
        self._cosmos_database = cosmos_database or "ai_memory"
        self._cosmos_container = cosmos_container or "memories"
        self._cosmos_turns_container = cosmos_turns_container
        self._cosmos_summaries_container = cosmos_summaries_container
        self._cosmos_counter_container = cosmos_counter_container or "counter"
        self._cosmos_lease_container = cosmos_lease_container or "leases"
        self._cosmos_throughput_mode = _resolve_cosmos_throughput_mode(cosmos_throughput_mode)
        self._cosmos_autoscale_max_ru = _resolve_cosmos_provisioning_autoscale_max_ru(
            throughput_mode=self._cosmos_throughput_mode,
            autoscale_max_ru=cosmos_autoscale_max_ru,
        )

        self._ai_foundry_endpoint = normalize_ai_foundry_endpoint(ai_foundry_endpoint)
        self._ai_foundry_credential = ai_foundry_credential
        self._ai_foundry_api_key = ai_foundry_api_key
        self._embedding_deployment_name = embedding_deployment_name
        self._embedding_dimensions = _resolve_embedding_dimensions(embedding_dimensions)
        self._chat_deployment_name = chat_deployment_name
        self._enable_turn_embeddings = (
            enable_turn_embeddings if enable_turn_embeddings is not None else get_enable_turn_embeddings()
        )

        self._owns_cosmos_credential = False
        self._owns_ai_foundry_credential = False
        if self._cosmos_credential is None and self._cosmos_key:
            self._cosmos_credential = self._cosmos_key

        if use_default_credential:
            needs_cosmos = self._cosmos_credential is None
            needs_embed = self._ai_foundry_credential is None and not self._ai_foundry_api_key
            if needs_cosmos or needs_embed:
                try:
                    module = importlib.import_module(default_credential_module)
                    default_credential_cls = getattr(module, "DefaultAzureCredential")
                    if needs_cosmos:
                        self._cosmos_credential = default_credential_cls()
                        self._owns_cosmos_credential = True
                    if needs_embed:
                        self._ai_foundry_credential = default_credential_cls()
                        self._owns_ai_foundry_credential = True
                except (ImportError, AttributeError):
                    pass

        self._cosmos_client: Any = None
        self._memories_container_client: Any = None
        self._turns_container_client: Any = None
        self._summaries_container_client: Any = None
        self._counter_container_client: Any = None
        self._store: Any = None

    @property
    def _containers(self) -> dict[ContainerKey, Any]:
        return {
            ContainerKey.TURNS: self._turns_container_client,
            ContainerKey.MEMORIES: self._memories_container_client,
            ContainerKey.SUMMARIES: self._summaries_container_client,
        }

    def __enter__(self) -> Any:
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.close()

    def add_local(
        self,
        user_id: str,
        role: str,
        content: str,
        memory_type: str = "turn",
        agent_id: Optional[str] = None,
        metadata: Optional[dict[str, Any]] = None,
        thread_id: Optional[str] = None,
        tags: Optional[list[str]] = None,
        ttl: Optional[int] = None,
        salience: Optional[float] = None,
    ) -> None:
        """Add a new memory to the local store."""
        if memory_type == "turn" and not thread_id:
            raise ValidationError(
                "thread_id is required for memory_type='turn' so the auto-trigger "
                "counter can group turns per conversation. Set thread_id explicitly."
            )
        memory = _make_memory(
            user_id=user_id,
            role=role,
            content=content,
            memory_type=memory_type,
            agent_id=agent_id,
            metadata=metadata,
            thread_id=thread_id,
            tags=tags,
            ttl=ttl,
            salience=salience,
        )
        self.local_memory.append(memory)
        if memory_type == "turn":
            key = (user_id, thread_id)
            self._unflushed_turn_counts[key] = self._unflushed_turn_counts.get(key, 0) + 1
        logger.debug("add_local id=%s role=%s type=%s", memory["id"], role, memory_type)

    def get_local(
        self,
        memory_id: Optional[str] = None,
        user_id: Optional[str] = None,
        role: Optional[str] = None,
        memory_types: Optional[list[str]] = None,
    ) -> list[dict[str, Any]]:
        """Retrieve memories from the local store."""
        results = self.local_memory
        if memory_id is not None:
            results = [m for m in results if m["id"] == memory_id]
        if user_id is not None:
            results = [m for m in results if m["user_id"] == user_id]
        if role is not None:
            results = [m for m in results if m["role"] == role]
        if memory_types:
            type_set = set(memory_types)
            results = [m for m in results if m["type"] in type_set]
        return results

    def update_local(
        self,
        memory_id: str,
        content: Optional[str] = None,
        role: Optional[str] = None,
        memory_type: Optional[str] = None,
        metadata: Optional[dict[str, Any]] = None,
    ) -> None:
        """Update an existing memory in the local store."""
        for memory in self.local_memory:
            if memory["id"] != memory_id:
                continue
            if content is not None:
                memory["content"] = content
            if role is not None:
                if role not in VALID_ROLES:
                    raise ValidationError(f"role must be one of {VALID_ROLES}, got '{role}'")
                memory["role"] = role
            if memory_type is not None:
                if memory_type not in VALID_TYPES:
                    raise ValidationError(f"type must be one of {VALID_TYPES}, got '{memory_type}'")
                memory["type"] = memory_type
            if metadata is not None:
                memory["metadata"] = metadata
            memory["updated_at"] = datetime.now(timezone.utc).isoformat()
            return
        raise MemoryNotFoundError(memory_id=memory_id)

    def delete_local(self, memory_id: str) -> None:
        """Delete a memory from the local store by id."""
        for i, memory in enumerate(self.local_memory):
            if memory["id"] == memory_id:
                self.local_memory.pop(i)
                return
        raise MemoryNotFoundError(memory_id=memory_id)

    def _require_cosmos(self) -> None:
        """Raise if Cosmos DB is not connected."""
        if self._memories_container_client is None:
            raise CosmosNotConnectedError()

    def _warn_on_embedding_dim_mismatch(self, container: Any = None) -> None:
        """Log a warning when the configured embedding dim differs from the container policy."""
        container = container if container is not None else self._memories_container_client
        if container is None or self._embedding_dimensions is None:
            return
        try:
            props = container.read()
        except Exception:
            return
        if not isinstance(props, dict):
            return
        policy = props.get("vectorEmbeddingPolicy") or {}
        embeddings = policy.get("vectorEmbeddings") or []
        if not embeddings:
            return
        container_dim = embeddings[0].get("dimensions")
        if container_dim and container_dim != self._embedding_dimensions:
            logger.warning(
                "Embedding dimension mismatch: container '%s' is configured "
                "for %d-dim vectors but the SDK is set to write %d-dim vectors. "
                "Vector search will return empty/wrong results until both sides "
                "agree. Pass embedding_dimensions=%d (or recreate the container) "
                "to fix.",
                self._cosmos_container,
                container_dim,
                self._embedding_dimensions,
                container_dim,
            )

    @staticmethod
    def _close_sync_closeable(closeable: Any) -> None:
        close = getattr(closeable, "close", None)
        if callable(close):
            try:
                close()
            except Exception:
                pass


# Status codes that indicate transient, retry-able backend conditions.
# Permanent codes (401/403/404/409) and client-side bugs (400) must surface.
_TRANSIENT_HTTP_STATUS_CODES = frozenset({408, 429, 500, 502, 503, 504})


def is_transient_tail_step_error(exc: BaseException) -> bool:
    """Classify a tail-step exception as transient (swallow + log) or permanent (re-raise).

    Used by ``process_now`` to decide whether a failure in the
    ``synthesize_procedural`` / ``process_user_summary`` tail steps should be
    logged as a warning (so the per-thread work already persisted is not
    erased) or re-raised to the caller (so configuration / schema bugs do not
    become silent ``WARNING`` lines).

    Transient (swallow):
      * ``LLMError`` — LLM-side defensive raises (no-choices, no-content).
      * ``openai.RateLimitError`` / ``APITimeoutError`` / ``APIConnectionError``.
      * Any exception with ``status_code`` in
        :data:`_TRANSIENT_HTTP_STATUS_CODES` (covers ``CosmosHttpResponseError``
        and any other ``HttpResponseError`` subclass).

    Permanent (re-raise):
      * ``ValidationError`` / ``ConfigurationError`` / ``CosmosNotConnectedError``.
      * ``openai.AuthenticationError`` / ``PermissionDeniedError`` /
        ``BadRequestError`` (status 400/401/403).
      * ``CosmosHttpResponseError`` with status 400/401/403/404/409.
      * Python builtins (``KeyError``, ``TypeError``, ``AttributeError``,
        ``NameError`` …) — these are programmer bugs, not infra hiccups.
    """
    from azure.cosmos.agent_memory.exceptions import LLMError

    if isinstance(exc, LLMError):
        return True

    try:
        import openai
    except ImportError:
        openai = None
    if openai is not None and isinstance(
        exc,
        (openai.RateLimitError, openai.APITimeoutError, openai.APIConnectionError),
    ):
        return True
    if openai is not None and isinstance(
        exc,
        (openai.AuthenticationError, openai.PermissionDeniedError, openai.BadRequestError),
    ):
        return False

    status = getattr(exc, "status_code", None)
    if isinstance(status, int):
        return status in _TRANSIENT_HTTP_STATUS_CODES

    return False
