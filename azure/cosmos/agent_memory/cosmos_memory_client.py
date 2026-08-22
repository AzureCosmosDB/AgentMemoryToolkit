"""Thin synchronous Cosmos memory client orchestrator."""

from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING, Any, Iterable, Mapping, Optional

from azure.cosmos.agent_memory.logging import get_logger

from ._authz import authorize_scope_write, resolve_read_scopes, resolve_scope_access
from ._base import _BaseMemoryClient
from ._base.base_client import is_transient_tail_step_error
from ._container_routing import container_key_for_type
from ._partitioning import (
    USER_SUMMARY_THREAD_ID,
    ensure_scope_fields,
    private_scope_key_for_principal,
    tenant_scope,
    user_id_from_principal,
)
from ._security import SecurityContext
from ._utils import (
    _build_container_kwargs,
    _container_policies,
    _cosmos_container_offer_throughput,
    _resolve_cosmos_provisioning_autoscale_max_ru,
    _resolve_cosmos_throughput_mode,
    _resolve_distance_function,
    _resolve_embedding_data_type,
    _resolve_full_text_language,
    _resolve_vector_index_type,
    _validate_connection,
)
from .auto_trigger import maybe_trigger_steps
from .chat import ChatClient
from .embeddings import EmbeddingsClient
from .exceptions import CosmosOperationError, MemoryNotFoundError, ValidationError
from .processors import InProcessProcessor, MemoryProcessor
from .services._pipeline_helpers import _normalize_cadence_thresholds, _normalize_metadata_keys
from .services.pipeline import PipelineService
from .store import MemoryStore
from .thresholds import DEFAULT_TTL_BY_TYPE

if TYPE_CHECKING:  # pragma: no cover - typing-only import
    from .processors import ProcessThreadResult, UserSummaryResult  # noqa: F401

logger = get_logger(__name__)


class _BoundCosmosMemoryClient:
    """Lightweight scope-bound sync client view."""

    def __init__(
        self,
        client: "CosmosMemoryClient",
        ctx: SecurityContext,
        *,
        write_scope: str | None = None,
        read_scopes: list[str] | None = None,
        project_scope: str | None = None,
    ) -> None:
        self._client = client
        self.ctx = ctx
        self.write_scope = write_scope or private_scope_key_for_principal(ctx.principal)
        # When read_scopes are not given, auto-resolve the caller's entitled scopes from
        # the trusted context (own + agent + project + org + teams), capped. The LLM never
        # declares scopes - the adapter does. (Docs/shared-memory-design.md §4.10d)
        self.read_scopes = (
            list(read_scopes) if read_scopes is not None else resolve_read_scopes(ctx, project_scope=project_scope)
        )
        principal = ctx.principal
        # user_id is only meaningful for delegated ``user:`` principals. App-only
        # ``agent:`` principals own an ``agent:<id>`` scope (already reflected in
        # write_scope), so derive the id portion directly rather than through the
        # user-only helper, which rejects non-user principals.
        self.user_id = (
            user_id_from_principal(principal)
            if principal and principal.startswith("user:")
            else (principal.split(":", 1)[1] if principal and ":" in principal else "")
        )

    def __getattr__(self, name: str) -> Any:
        # Establish the caller's tenant for any client method not explicitly wrapped
        # above - notably the user-scoped getters (get_thread_summary, get_episodes,
        # ...) whose reads resolve tenant via the request-scoped ContextVar - so a
        # bound session operates entirely within ``ctx.tenant_id``.
        if name.startswith("__") and name.endswith("__"):
            raise AttributeError(name)
        attr = getattr(self._client, name)
        if not callable(attr):
            return attr

        def _tenant_bound(*args: Any, **kwargs: Any) -> Any:
            with tenant_scope(self.ctx.tenant_id):
                return attr(*args, **kwargs)

        return _tenant_bound

    def upsert_memory(self, *args: Any, **kwargs: Any) -> str:
        kwargs.setdefault("user_id", self.user_id)
        kwargs.setdefault("tenant_id", self.ctx.tenant_id)
        kwargs.setdefault("scope_key", self.write_scope)
        kwargs.setdefault("ctx", self.ctx)
        if self.ctx.agent_id:
            kwargs.setdefault("agent_id", self.ctx.agent_id)
        return self._client.upsert_memory(*args, **kwargs)

    def add_local(self, *args: Any, **kwargs: Any) -> None:
        kwargs.setdefault("user_id", self.user_id)
        # Authorize the buffered write at entry: add_local stamps ``self.write_scope`` and
        # appends to the shared client buffer, which push_to_cosmos later flushes with no
        # context. Gating here (not just at push) keeps an unauthorized scope from ever
        # entering the buffer, closing the buffer-flush bypass regardless of which
        # push_to_cosmos drains it.
        authorize_scope_write(self.ctx, self.write_scope, self.user_id)
        self._client.add_local(*args, **kwargs)
        if self._client.local_memory:
            record = self._client.local_memory[-1]
            record.pop("acl", None)
            ensure_scope_fields(
                record,
                tenant_id=self.ctx.tenant_id,
                scope_key=self.write_scope,
                principal=self.ctx.principal,
                agent_id=self.ctx.agent_id,
            )
            if self.ctx.agent_id:
                record.setdefault("provenance", {}).setdefault("agent_id", self.ctx.agent_id)

    def search_cosmos(self, *args: Any, **kwargs: Any) -> list[dict[str, Any]]:
        kwargs.setdefault("tenant_id", self.ctx.tenant_id)
        kwargs.setdefault("scopes", self.read_scopes)
        kwargs.setdefault("ctx", self.ctx)
        kwargs.setdefault("user_id", None)
        return self._client.search_cosmos(*args, **kwargs)

    def put_shared_record(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        kwargs.setdefault("ctx", self.ctx)
        kwargs.setdefault("scope_key", self.write_scope)
        return self._client.put_shared_record(*args, **kwargs)

    def get_shared_record(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        kwargs.setdefault("ctx", self.ctx)
        kwargs.setdefault("scope_key", self.write_scope)
        return self._client.get_shared_record(*args, **kwargs)

    def compare_and_swap_shared_record(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        kwargs.setdefault("ctx", self.ctx)
        kwargs.setdefault("scope_key", self.write_scope)
        return self._client.compare_and_swap_shared_record(*args, **kwargs)

    def update_shared_record(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        kwargs.setdefault("ctx", self.ctx)
        kwargs.setdefault("scope_key", self.write_scope)
        return self._client.update_shared_record(*args, **kwargs)

    def pin_memory(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        kwargs.setdefault("ctx", self.ctx)
        return self._client.pin_memory(*args, **kwargs)

    def unpin_memory(self, *args: Any, **kwargs: Any) -> bool:
        kwargs.setdefault("ctx", self.ctx)
        return self._client.unpin_memory(*args, **kwargs)

    def list_pins(self, *args: Any, **kwargs: Any) -> list[dict[str, Any]]:
        kwargs.setdefault("ctx", self.ctx)
        return self._client.list_pins(*args, **kwargs)

    def reconcile(self, *args: Any, **kwargs: Any) -> dict[str, int]:
        kwargs.setdefault("ctx", self.ctx)
        kwargs.setdefault("scope_key", self.write_scope)
        return self._client.reconcile(*args, **kwargs)


class CosmosMemoryClient(_BaseMemoryClient):
    """Manages local and Cosmos-backed memories via store and service layers."""

    def __init__(
        self,
        cosmos_endpoint: Optional[str] = None,
        cosmos_credential: Optional[Any] = None,
        cosmos_key: Optional[str] = None,
        cosmos_database: Optional[str] = None,
        cosmos_container: Optional[str] = None,
        cosmos_turns_container: str = "memories_turns",
        cosmos_summaries_container: str = "memories_summaries",
        cosmos_counter_container: Optional[str] = None,
        cosmos_lease_container: Optional[str] = None,
        cosmos_throughput_mode: Optional[str] = None,
        cosmos_autoscale_max_ru: Optional[int] = None,
        ai_foundry_endpoint: Optional[str] = None,
        ai_foundry_credential: Optional[Any] = None,
        ai_foundry_api_key: Optional[str] = None,
        embedding_deployment_name: str = "text-embedding-3-large",
        embedding_dimensions: Optional[int] = None,
        chat_deployment_name: str = "gpt-4o-mini",
        use_default_credential: bool = True,
        enable_turn_embeddings: Optional[bool] = None,
        user_agent: Optional[str] = None,
        embeddings_client: Optional[Any] = None,
        chat_client: Optional[Any] = None,
        processor: Optional[MemoryProcessor] = None,
        transcript_metadata_keys: Optional[Iterable[str]] = None,
        cadence_thresholds: Optional[Mapping[str, int]] = None,
    ) -> None:
        self._init_base_config(
            cosmos_endpoint=cosmos_endpoint,
            cosmos_credential=cosmos_credential,
            cosmos_key=cosmos_key,
            cosmos_database=cosmos_database,
            cosmos_container=cosmos_container,
            cosmos_turns_container=cosmos_turns_container,
            cosmos_summaries_container=cosmos_summaries_container,
            cosmos_counter_container=cosmos_counter_container,
            cosmos_lease_container=cosmos_lease_container,
            cosmos_throughput_mode=cosmos_throughput_mode,
            cosmos_autoscale_max_ru=cosmos_autoscale_max_ru,
            ai_foundry_endpoint=ai_foundry_endpoint,
            ai_foundry_credential=ai_foundry_credential,
            ai_foundry_api_key=ai_foundry_api_key,
            embedding_deployment_name=embedding_deployment_name,
            embedding_dimensions=embedding_dimensions,
            chat_deployment_name=chat_deployment_name,
            use_default_credential=use_default_credential,
            enable_turn_embeddings=enable_turn_embeddings,
            user_agent=user_agent,
        )
        # Embeddings/chat clients may be injected (e.g. an OpenAI-compatible backend, a
        # caller-configured client, or a deterministic fake for offline tests). When a client
        # is injected the caller owns its lifecycle, so the toolkit does not close it; otherwise
        # the toolkit builds the Azure-backed client and closes it in ``close()``.
        if embeddings_client is not None:
            self._embeddings_client = embeddings_client
            self._owns_embeddings_client = False
        else:
            self._embeddings_client = EmbeddingsClient(
                endpoint=self._ai_foundry_endpoint,
                credential=self._ai_foundry_credential,
                api_key=self._ai_foundry_api_key,
                model=self._embedding_deployment_name,
                dimensions=self._embedding_dimensions,
            )
            self._owns_embeddings_client = True
        if chat_client is not None:
            self._chat_client = chat_client
            self._owns_chat_client = False
        else:
            self._chat_client = ChatClient(
                endpoint=self._ai_foundry_endpoint,
                credential=self._ai_foundry_credential,
                api_key=self._ai_foundry_api_key,
                model=self._chat_deployment_name,
            )
            self._owns_chat_client = True
        self._pipeline: Optional[PipelineService] = None
        self._processor: Optional[MemoryProcessor] = processor
        self._processor_explicit = processor is not None
        self._transcript_metadata_keys: Optional[tuple[str, ...]] = _normalize_metadata_keys(transcript_metadata_keys)
        # Optional per-turn cadence override, keyed by the same names as the env vars (e.g.
        # ``FACT_EXTRACTION_EVERY_N``, ``DEDUP_EVERY_N``, ``THREAD_SUMMARY_EVERY_N``,
        # ``USER_SUMMARY_EVERY_N``). When provided, the auto-trigger uses these values instead of
        # reading ``os.environ``; any key not present falls back to the environment/defaults.
        # ``None`` preserves the env-only behavior. Normalized to a defensive ``dict[str, int]``
        # so later mutation of the caller's mapping cannot change client behavior, and invalid
        # values fail fast at construction rather than deep inside the trigger path.
        self._cadence_thresholds: Optional[dict[str, int]] = _normalize_cadence_thresholds(cadence_thresholds)
        if self._cosmos_endpoint:
            self.create_memory_store()
        logger.info("CosmosMemoryClient initialized")

    def __enter__(self) -> CosmosMemoryClient:
        return super().__enter__()

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        return super().__exit__(exc_type, exc_val, exc_tb)

    def close(self) -> None:
        """Close Cosmos, model clients, and owned credentials."""
        if self._cosmos_client is not None:
            self._cosmos_client.close()
            self._cosmos_client = None
            self._memories_container_client = None
            self._turns_container_client = None
            self._summaries_container_client = None
            self._counter_container_client = None
            self._store = None
            self._pipeline = None
        if self._processor is not None and not self._processor_explicit:
            self._close_sync_closeable(self._processor)
            self._processor = None
        if self._owns_chat_client:
            self._close_sync_closeable(self._chat_client)
        if self._owns_embeddings_client:
            self._close_sync_closeable(self._embeddings_client)
        for owns, cred in (
            (self._owns_cosmos_credential, self._cosmos_credential),
            (self._owns_ai_foundry_credential, self._ai_foundry_credential),
        ):
            if owns and cred is not None:
                self._close_sync_closeable(cred)

    def connect_cosmos(
        self,
        endpoint: Optional[str] = None,
        credential: Optional[Any] = None,
        key: Optional[str] = None,
        database: Optional[str] = None,
        container: Optional[str] = None,
        turns_container: Optional[str] = None,
        summaries_container: Optional[str] = None,
    ) -> None:
        """Establish a connection to the Cosmos DB memory containers.

        If container overrides are provided, they override the constructor's
        settings for this connection.
        """
        self._cosmos_endpoint = endpoint or self._cosmos_endpoint
        if credential is not None:
            self._cosmos_credential = credential
        elif key is not None:
            self._cosmos_credential = key
            self._cosmos_key = key
        self._cosmos_database = database or self._cosmos_database
        self._cosmos_container = container or self._cosmos_container
        if turns_container is not None:
            self._cosmos_turns_container = turns_container
        if summaries_container is not None:
            self._cosmos_summaries_container = summaries_container
        _validate_connection(
            self._cosmos_endpoint,
            self._cosmos_credential,
            self._cosmos_database,
            self._cosmos_container,
        )
        try:
            from azure.cosmos import CosmosClient

            self._drain_cosmos_client()
            client = CosmosClient(
                self._cosmos_endpoint,
                credential=self._cosmos_credential,
                user_agent=self._cosmos_user_agent,
            )
            db = client.get_database_client(self._cosmos_database)
            self._cosmos_client = client
            self._memories_container_client = db.get_container_client(self._cosmos_container)
            self._turns_container_client = db.get_container_client(self._cosmos_turns_container)
            self._summaries_container_client = db.get_container_client(self._cosmos_summaries_container)
            logger.info("Connected turns container: %s/%s", self._cosmos_database, self._cosmos_turns_container)
            logger.info(
                "Connected summaries container: %s/%s",
                self._cosmos_database,
                self._cosmos_summaries_container,
            )
            self._init_services()
        except Exception as exc:
            raise CosmosOperationError(f"Failed to connect to Cosmos DB: {exc}") from exc
        logger.info("Connected to Cosmos DB %s/%s", self._cosmos_database, self._cosmos_container)

    def create_memory_store(
        self,
        database: Optional[str] = None,
        container: Optional[str] = None,
        turns_container: Optional[str] = None,
        summaries_container: Optional[str] = None,
        counter_container: Optional[str] = None,
        lease_container: Optional[str] = None,
        endpoint: Optional[str] = None,
        credential: Optional[Any] = None,
        key: Optional[str] = None,
        embedding_dimensions: Optional[int] = None,
        embedding_data_type: Optional[str] = None,
        distance_function: Optional[str] = None,
        full_text_language: Optional[str] = None,
        throughput_mode: Optional[str] = None,
        autoscale_max_ru: Optional[int] = None,
        vector_index_type: Optional[str] = None,
    ) -> None:
        """Create the Cosmos DB database and memory/counter/lease containers."""
        self._cosmos_endpoint = endpoint or self._cosmos_endpoint
        if credential is not None:
            self._cosmos_credential = credential
        elif key is not None:
            self._cosmos_credential = key
            self._cosmos_key = key
        self._cosmos_database = database or self._cosmos_database
        self._cosmos_container = container or self._cosmos_container
        if turns_container is not None:
            self._cosmos_turns_container = turns_container
        if summaries_container is not None:
            self._cosmos_summaries_container = summaries_container
        self._cosmos_counter_container = counter_container or self._cosmos_counter_container
        self._cosmos_lease_container = lease_container or self._cosmos_lease_container
        self._cosmos_throughput_mode = _resolve_cosmos_throughput_mode(
            throughput_mode if throughput_mode is not None else self._cosmos_throughput_mode
        )
        self._cosmos_autoscale_max_ru = _resolve_cosmos_provisioning_autoscale_max_ru(
            throughput_mode=self._cosmos_throughput_mode,
            autoscale_max_ru=autoscale_max_ru if autoscale_max_ru is not None else self._cosmos_autoscale_max_ru,
        )
        _validate_connection(
            self._cosmos_endpoint,
            self._cosmos_credential,
            self._cosmos_database,
            self._cosmos_container,
        )
        try:
            from azure.cosmos import CosmosClient, PartitionKey, ThroughputProperties

            self._drain_cosmos_client()
            client = CosmosClient(
                self._cosmos_endpoint,
                credential=self._cosmos_credential,
                user_agent=self._cosmos_user_agent,
            )
            db = client.create_database_if_not_exists(id=self._cosmos_database)
            partition_key = PartitionKey(path=["/tenant_id", "/scope_key", "/thread_id"], kind="MultiHash")
            offer = _cosmos_container_offer_throughput(
                throughput_mode=self._cosmos_throughput_mode,
                autoscale_max_ru=self._cosmos_autoscale_max_ru,
                throughput_properties_cls=ThroughputProperties,
            )
            _policy_kwargs = dict(
                embedding_dimensions=embedding_dimensions or self._embedding_dimensions or 1536,
                embedding_data_type=_resolve_embedding_data_type(embedding_data_type),
                distance_function=_resolve_distance_function(distance_function),
                full_text_language=_resolve_full_text_language(full_text_language),
                vector_index_type=_resolve_vector_index_type(vector_index_type),
            )
            vec_policy, idx_policy, ft_policy = _container_policies(**_policy_kwargs)
            # Turns always carry the vector index (primed for search).
            turns_vec_policy, turns_idx_policy, turns_ft_policy = _container_policies(
                **{**_policy_kwargs, "vector_index_type": "quantizedFlat"},
            )
            self._memories_container_client = db.create_container_if_not_exists(
                **_build_container_kwargs(
                    container_id=self._cosmos_container,
                    partition_key=partition_key,
                    offer_throughput=offer,
                    default_ttl=-1,
                    indexing_policy=idx_policy,
                    vector_embedding_policy=vec_policy,
                    full_text_policy=ft_policy,
                )
            )
            self._turns_container_client = db.create_container_if_not_exists(
                **_build_container_kwargs(
                    container_id=self._cosmos_turns_container,
                    partition_key=partition_key,
                    offer_throughput=offer,
                    default_ttl=DEFAULT_TTL_BY_TYPE["turn"],
                    indexing_policy=turns_idx_policy,
                    vector_embedding_policy=turns_vec_policy,
                    full_text_policy=turns_ft_policy,
                )
            )
            logger.info("Created turns container: %s/%s", self._cosmos_database, self._cosmos_turns_container)
            # Summaries carry the vector + full-text index too, so summaries are
            # semantically searchable (search_summaries). Keep the point-lookup
            # composite (user_id, thread_id, version) get_*_summary relies on.
            summaries_vec_policy, summaries_idx_policy, summaries_ft_policy = _container_policies(
                **{**_policy_kwargs, "vector_index_type": "quantizedFlat"},
            )
            summaries_idx_policy["compositeIndexes"] = [
                [
                    {"path": "/user_id", "order": "ascending"},
                    {"path": "/thread_id", "order": "ascending"},
                    {"path": "/version", "order": "descending"},
                ]
            ]
            self._summaries_container_client = db.create_container_if_not_exists(
                **_build_container_kwargs(
                    container_id=self._cosmos_summaries_container,
                    partition_key=partition_key,
                    offer_throughput=offer,
                    default_ttl=-1,
                    indexing_policy=summaries_idx_policy,
                    vector_embedding_policy=summaries_vec_policy,
                    full_text_policy=summaries_ft_policy,
                )
            )
            logger.info(
                "Created summaries container: %s/%s",
                self._cosmos_database,
                self._cosmos_summaries_container,
            )
            db.create_container_if_not_exists(
                **_build_container_kwargs(
                    container_id=self._cosmos_counter_container,
                    partition_key=partition_key,
                    offer_throughput=offer,
                )
            )
            db.create_container_if_not_exists(
                **_build_container_kwargs(
                    container_id=self._cosmos_lease_container,
                    partition_key=PartitionKey(path="/id"),
                    offer_throughput=offer,
                )
            )
            self._cosmos_client = client
            self._init_services()
        except Exception as exc:
            raise CosmosOperationError(f"Failed to create memory store: {exc}") from exc
        logger.info("Created memory store %s/%s", self._cosmos_database, self._cosmos_container)

    def validate_topology(self) -> None:
        """Verify the Cosmos containers exist, are reachable, and use the expected key.

        Reads metadata for memories / memories_turns / memories_summaries and the counter
        container, and checks each partition key is exactly the hierarchical
        ``[/tenant_id, /scope_key, /thread_id]``. Raises ``RuntimeError`` on the first
        failure with a clear message instructing the customer to redeploy the
        infrastructure.

        Call this after ``connect_cosmos`` or ``create_memory_store`` to diagnose topology
        mismatches before any data is written.
        """
        from azure.cosmos.exceptions import CosmosResourceNotFoundError

        if self._memories_container_client is None:
            raise RuntimeError("validate_topology: Cosmos client is not connected; call connect_cosmos() first")
        expected_pk_paths = ["/tenant_id", "/scope_key", "/thread_id"]
        checks: list[tuple[str, Any]] = [(key.value, client) for key, client in self._containers.items()]
        counter_client = self._get_counter_container()
        if counter_client is not None:
            checks.append((self._cosmos_counter_container, counter_client))
        for label, client in checks:
            if client is None:
                raise RuntimeError(f"validate_topology: container for {label!r} is not connected")
            container_id = getattr(client, "id", label)
            try:
                props = client.read()
            except CosmosResourceNotFoundError as exc:
                raise RuntimeError(
                    f"validate_topology: container {container_id!r} does not exist; "
                    f"redeploy infrastructure (azd up) and ensure the SDK config matches"
                ) from exc
            except Exception as exc:
                raise RuntimeError(
                    f"validate_topology: cannot read container {container_id!r}: {type(exc).__name__}: {exc}"
                ) from exc
            # Only real container metadata (a dict) carries a partition key; test doubles
            # that return a non-dict are skipped. When metadata is present the key must be
            # exactly the hierarchical key - an empty/missing or 2-level key is rejected.
            if isinstance(props, dict):
                actual_pk_paths = list((props.get("partitionKey") or {}).get("paths") or [])
                if actual_pk_paths != expected_pk_paths:
                    raise RuntimeError(
                        f"validate_topology: container {container_id!r} has partition key "
                        f"{actual_pk_paths or 'none'} but this SDK requires {expected_pk_paths}. The partition key "
                        f"is immutable in Cosmos; redeploy the container with the current hierarchical key "
                        f"(tenant_id, scope_key, thread_id)"
                    )

    def _build_store(self) -> MemoryStore:
        return MemoryStore(
            containers=self._containers,
            embeddings_client=self._embeddings_client,
            enable_turn_embeddings=self._enable_turn_embeddings,
        )

    def _build_pipeline(self, store: MemoryStore) -> PipelineService:
        return PipelineService(
            store,
            self._chat_client,
            self._embeddings_client,
            containers=self._containers,
            transcript_metadata_keys=self._transcript_metadata_keys,
        )

    def _store_uses_current_clients(self) -> bool:
        if self._store is None:
            return False
        containers = getattr(self._store, "_containers", None)
        if not isinstance(containers, dict):
            return False
        return all(containers.get(key) is client for key, client in self._containers.items())

    def _pipeline_uses_current_clients(self) -> bool:
        if self._pipeline is None:
            return False
        containers = getattr(self._pipeline, "_containers", None)
        if not isinstance(containers, dict):
            return False
        return all(containers.get(key) is client for key, client in self._containers.items())

    def _init_services(self) -> None:
        self._store = self._build_store()
        self._pipeline = self._build_pipeline(self._store)
        self._warn_on_embedding_dim_mismatch()
        if not self._processor_explicit:
            self._processor = None

    def _init_pipeline(self) -> None:
        """Compatibility helper used by tests; initializes all service layers."""
        self._init_services()

    def _drain_cosmos_client(self) -> None:
        prior = self._cosmos_client
        if prior is not None:
            close = getattr(prior, "close", None)
            if callable(close):
                try:
                    close()
                except Exception:
                    logger.warning("Failed to close prior Cosmos client during reconnect", exc_info=True)
        self._cosmos_client = None
        self._memories_container_client = None
        self._turns_container_client = None
        self._summaries_container_client = None
        self._counter_container_client = None
        self._store = None
        self._pipeline = None
        if not self._processor_explicit:
            self._processor = None

    def _get_store(self) -> MemoryStore:
        self._require_cosmos()
        if (
            self._store is None
            or not self._store_uses_current_clients()
            or self._store._embeddings_client is not self._embeddings_client
        ):
            self._store = self._build_store()
        return self._store

    def _get_pipeline(self) -> PipelineService:
        self._require_cosmos()
        store = self._get_store()
        if self._pipeline is None or self._pipeline._store is not store or not self._pipeline_uses_current_clients():
            self._pipeline = self._build_pipeline(store)
        return self._pipeline

    def _get_processor(self) -> MemoryProcessor:
        if self._processor is None:
            self._processor = InProcessProcessor(pipeline=self._get_pipeline())
        return self._processor

    def _get_counter_container(self) -> Any:
        if self._counter_container_client is not None:
            return self._counter_container_client
        if self._cosmos_client is None:
            return None
        try:
            db = self._cosmos_client.get_database_client(self._cosmos_database)
            self._counter_container_client = db.get_container_client(self._cosmos_counter_container)
            return self._counter_container_client
        except Exception as exc:  # pragma: no cover - defensive
            if not self._warned_counter_unreachable:
                self._warned_counter_unreachable = True
                logger.warning(
                    "Counter container %s/%s unreachable: %s",
                    self._cosmos_database,
                    self._cosmos_counter_container,
                    exc,
                )
            return None

    def _maybe_auto_trigger(self, turn_counts: dict[tuple[str, str], int]) -> None:
        if not turn_counts:
            return
        maybe_trigger_steps(
            self._get_processor(),
            self._get_counter_container(),
            turn_counts,
            thresholds=self._cadence_thresholds,
        )

    def _container_for_type(self, memory_type: str) -> Any:
        """Return the Cosmos container client that owns ``memory_type``."""
        return self._containers[container_key_for_type(memory_type)]

    def session(
        self,
        ctx: SecurityContext,
        write_scope: str | None = None,
        read_scopes: list[str] | None = None,
        project_scope: str | None = None,
    ) -> _BoundCosmosMemoryClient:
        """Return a view bound to a trusted security context and scope set.

        When ``read_scopes`` is omitted it is auto-resolved from ``ctx`` (own + agent +
        optional ``project_scope`` + org + teams, capped at ``MAX_READ_SCOPES``).
        """
        return _BoundCosmosMemoryClient(
            self, ctx, write_scope=write_scope, read_scopes=read_scopes, project_scope=project_scope
        )

    def put_shared_record(
        self,
        *,
        ctx: SecurityContext,
        scope_key: str,
        key: str,
        content: str = "",
        data: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
        read_only: bool = False,
        created_by: str | None = None,
    ) -> dict[str, Any]:
        """Create or replace a small live shared coordination record."""
        return self._get_store().put_shared_record(
            ctx=ctx,
            scope_key=scope_key,
            key=key,
            content=content,
            data=data,
            metadata=metadata,
            read_only=read_only,
            created_by=created_by,
        )

    def get_shared_record(self, *, ctx: SecurityContext, scope_key: str, key: str) -> dict[str, Any]:
        """Read a live shared coordination record using read authorization."""
        return self._get_store().get_shared_record(ctx=ctx, scope_key=scope_key, key=key)

    def compare_and_swap_shared_record(
        self,
        *,
        ctx: SecurityContext,
        scope_key: str,
        key: str,
        record: dict[str, Any],
        etag: str,
    ) -> dict[str, Any]:
        """Replace a shared record only if the supplied Cosmos ETag still matches."""
        return self._get_store().compare_and_swap_shared_record(
            ctx=ctx, scope_key=scope_key, key=key, record=record, etag=etag
        )

    def update_shared_record(
        self,
        *,
        ctx: SecurityContext,
        scope_key: str,
        key: str,
        mutator: Any,
        max_retries: int = 3,
    ) -> dict[str, Any]:
        """Apply ``mutator`` with ETag compare-and-swap retry semantics."""
        return self._get_store().update_shared_record(
            ctx=ctx, scope_key=scope_key, key=key, mutator=mutator, max_retries=max_retries
        )

    def pin_memory(
        self,
        agent_id: str,
        resource: str,
        injection_mode: str = "summary",
        priority: int = 50,
        ctx: SecurityContext | None = None,
        resource_type: str | None = None,
    ) -> dict[str, Any]:
        """Bind a memory id or scope key to an agent for selective injection."""
        return self._get_store().pin_memory(
            agent_id,
            resource,
            injection_mode=injection_mode,
            priority=priority,
            ctx=ctx,
            resource_type=resource_type,
        )

    def unpin_memory(
        self, agent_id: str, resource: str, ctx: SecurityContext | None = None, resource_type: str | None = None
    ) -> bool:
        """Remove a selective-injection pin for an agent/resource binding."""
        return self._get_store().unpin_memory(agent_id, resource, ctx=ctx, resource_type=resource_type)

    def list_pins(self, agent_id: str, ctx: SecurityContext | None = None) -> list[dict[str, Any]]:
        """List selective-injection pins for an agent, highest priority first."""
        return self._get_store().list_pins(agent_id, ctx=ctx)

    def promote(self, memory_id: str, from_scope: str, to_scope: str, ctx: SecurityContext) -> dict[str, Any]:
        """Copy a memory into a target scope, gated by share/assign on that scope."""
        return self._get_store().promote(memory_id, from_scope, to_scope, ctx)

    def list_promotion_candidates(
        self,
        ctx: SecurityContext,
        *,
        from_scope: str | None = None,
        min_confidence: float | None = None,
        top: int = 100,
    ) -> list[dict[str, Any]]:
        """List private hinted records awaiting manual promotion review."""
        return self._get_store().list_promotion_candidates(
            ctx, from_scope=from_scope, min_confidence=min_confidence, top=top
        )

    def approve_promotion(
        self,
        memory_id: str,
        *,
        from_scope: str,
        to_scope: str,
        ctx: SecurityContext,
    ) -> dict[str, Any]:
        """Approve one promotion candidate into ``to_scope``."""
        return self._get_store().approve_promotion(memory_id, from_scope=from_scope, to_scope=to_scope, ctx=ctx)

    def auto_promote_candidates(
        self,
        ctx: SecurityContext,
        *,
        target_scopes_by_type: dict[str, str],
        confidence_threshold: float = 0.95,
        allow_memory_types: set[str] | None = None,
        from_scope: str | None = None,
        top: int = 100,
    ) -> list[dict[str, Any]]:
        """Auto-promote eligible hinted records when policy guards pass."""
        return self._get_store().auto_promote_candidates(
            ctx,
            target_scopes_by_type=target_scopes_by_type,
            confidence_threshold=confidence_threshold,
            allow_memory_types=allow_memory_types,
            from_scope=from_scope,
            top=top,
        )

    def upsert_memory(
        self,
        user_id: str,
        role: str,
        content: str,
        memory_type: str = "turn",
        metadata: Optional[dict[str, Any]] = None,
        thread_id: Optional[str] = None,
        tags: Optional[list[str]] = None,
        ttl: Optional[int] = None,
        salience: Optional[float] = None,
        embedding: Optional[list[float]] = None,
        embed: Optional[bool] = None,
        created_at: Optional[str | datetime] = None,
        tenant_id: Optional[str] = None,
        scope_key: Optional[str] = None,
        agent_id: Optional[str] = None,
        ctx: Optional[SecurityContext] = None,
    ) -> str:
        """Add a memory directly to Cosmos DB, bypassing the local buffer.

        For ``memory_type='turn'`` this also bumps the auto-trigger counter and
        schedules cadence-aware processing (extract / reconcile / procedural /
        thread_summary / user_summary) via :func:`maybe_trigger_steps`, exactly
        like :meth:`push_to_cosmos` does for buffered turns - so the
        ``FACT_EXTRACTION_EVERY_N`` / ``THREAD_SUMMARY_EVERY_N`` /
        ``USER_SUMMARY_EVERY_N`` / ``DEDUP_EVERY_N`` knobs apply uniformly
        whether the caller uses the buffer or writes through directly.

        ``agent_id`` is stamped on the record as provenance (which agent wrote it);
        it does not affect scope/placement. When ``scope_key`` targets a scope other
        than the caller's own ``user:<user_id>`` scope, ``ctx`` is required and must
        hold write permission for that scope (bound sessions supply it automatically).
        """
        if memory_type == "turn" and not thread_id:
            raise ValidationError(
                "thread_id is required for memory_type='turn' so the auto-trigger "
                "counter can group turns per conversation. Set thread_id explicitly."
            )
        memory_id = self._get_store().add(
            user_id,
            role,
            content,
            memory_type,
            metadata,
            thread_id,
            tags,
            ttl,
            salience,
            embedding,
            embed,
            created_at,
            tenant_id,
            scope_key,
            agent_id,
            ctx=ctx,
        )
        if memory_type == "turn" and thread_id:
            try:
                with tenant_scope(tenant_id):
                    self._maybe_auto_trigger({(user_id, thread_id): 1})
            except Exception as exc:
                logger.warning("Auto-trigger after upsert_memory failed: %s", exc)
        return memory_id

    def push_to_cosmos(self, batch_size: int = 25) -> None:
        """Insert all local memories into Cosmos DB."""
        self._get_store().push(self.local_memory, batch_size=batch_size)
        turn_counts, self._unflushed_turn_counts = self._unflushed_turn_counts, {}
        try:
            self._maybe_auto_trigger(turn_counts)
        except Exception as exc:
            logger.warning("Auto-trigger after push_to_cosmos failed: %s", exc)

    def get_memories(
        self,
        memory_id: Optional[str] = None,
        user_id: Optional[str] = None,
        thread_id: Optional[str] = None,
        role: Optional[str] = None,
        memory_types: Optional[list[str]] = None,
        recent_k: Optional[int] = None,
        tags_all: Optional[list[str]] = None,
        tags_any: Optional[list[str]] = None,
        exclude_tags: Optional[list[str]] = None,
        include_superseded: bool = False,
        min_salience: Optional[float] = None,
        min_confidence: Optional[float] = None,
        created_after: Optional[str | datetime] = None,
        created_before: Optional[str | datetime] = None,
    ) -> list[dict[str, Any]]:
        """Retrieve memories from Cosmos DB with optional filters."""
        return self._get_store().get_memories(
            memory_id=memory_id,
            user_id=user_id,
            thread_id=thread_id,
            role=role,
            memory_types=memory_types,
            recent_k=recent_k,
            tags_all=tags_all,
            tags_any=tags_any,
            exclude_tags=exclude_tags,
            include_superseded=include_superseded,
            min_salience=min_salience,
            min_confidence=min_confidence,
            created_after=created_after,
            created_before=created_before,
        )

    def update_cosmos(
        self,
        memory_id: str,
        *,
        user_id: str,
        thread_id: str,
        memory_type: str,
        content: Optional[str] = None,
        role: Optional[str] = None,
        metadata: Optional[dict[str, Any]] = None,
    ) -> None:
        """Update a memory in Cosmos DB."""
        return self._get_store().update(
            memory_id,
            user_id=user_id,
            thread_id=thread_id,
            memory_type=memory_type,
            content=content,
            role=role,
            metadata=metadata,
        )

    def delete_memory(
        self,
        memory_id: str,
        *,
        user_id: str,
        thread_id: str,
        memory_type: str,
    ) -> None:
        """Delete a memory from Cosmos DB."""
        return self._get_store().delete(
            memory_id,
            user_id=user_id,
            thread_id=thread_id,
            memory_type=memory_type,
        )

    def delete_turn(self, turn_id: str, *, user_id: str, thread_id: str) -> None:
        """Delete a single turn document. Raises if it does not exist."""
        return self.delete_memory(
            turn_id,
            user_id=user_id,
            thread_id=thread_id,
            memory_type="turn",
        )

    def delete_thread_summary(self, user_id: str, thread_id: str) -> bool:
        """Delete a thread's summary if present. Returns True when one was deleted.

        The summary id is deterministic, so callers need not look it up first; a
        missing summary is a no-op that returns False.
        """
        try:
            self.delete_memory(
                f"summary_{user_id}_{thread_id}",
                user_id=user_id,
                thread_id=thread_id,
                memory_type="thread_summary",
            )
            return True
        except MemoryNotFoundError:
            return False

    def delete_user_summary(self, user_id: str) -> bool:
        """Delete a user's summary if present. Returns True when one was deleted.

        A missing summary is a no-op that returns False.
        """
        try:
            self.delete_memory(
                f"user_summary_{user_id}",
                user_id=user_id,
                thread_id=USER_SUMMARY_THREAD_ID,
                memory_type="user_summary",
            )
            return True
        except MemoryNotFoundError:
            return False

    def delete_thread(self, user_id: str, thread_id: str, *, include_summary: bool = True) -> int:
        """Bulk-delete a conversation thread: all of its turns and, by default, its
        thread summary. Returns the number of documents deleted.

        This targets the conversation itself (turns + summary). Durable memories
        distilled from the thread - facts, episodes, procedures - are user-scoped
        knowledge and are left intact. Deletion is best-effort per document: a
        concurrently-removed turn is skipped rather than aborting the whole sweep.
        """
        if not user_id:
            raise ValidationError("user_id is required")
        if not thread_id:
            raise ValidationError("thread_id is required")

        deleted = 0
        turns = self.get_thread(thread_id=thread_id, user_id=user_id, include_superseded=True) or []
        for turn in turns:
            turn_id = turn.get("id")
            if not turn_id:
                continue
            try:
                self.delete_memory(turn_id, user_id=user_id, thread_id=thread_id, memory_type="turn")
                deleted += 1
            except MemoryNotFoundError:
                continue

        if include_summary and self.delete_thread_summary(user_id, thread_id):
            deleted += 1

        return deleted

    def search_cosmos(
        self,
        search_terms: str,
        memory_id: Optional[str] = None,
        user_id: Optional[str] = None,
        role: Optional[str] = None,
        memory_types: Optional[list[str]] = None,
        thread_id: Optional[str] = None,
        top_k: int = 5,
        tags_all: Optional[list[str]] = None,
        tags_any: Optional[list[str]] = None,
        exclude_tags: Optional[list[str]] = None,
        include_superseded: bool = False,
        min_salience: Optional[float] = None,
        min_confidence: Optional[float] = None,
        created_after: Optional[str | datetime] = None,
        created_before: Optional[str | datetime] = None,
        include_episodes: bool = False,
        include_turns: bool = False,
        turn_top_k: Optional[int] = None,
        include_summaries: bool = False,
        summary_top_k: Optional[int] = None,
        tenant_id: Optional[str] = None,
        scopes: Optional[list[str]] = None,
        ctx: Optional[SecurityContext] = None,
        agent_id: Optional[str] = None,
        include_pins: bool = False,
    ) -> list[dict[str, Any]]:
        """Search memories using vector similarity, with optional retrieval blending.

        The base search returns facts and, when ``include_episodes`` is True,
        episodes too - both in a single ranked query sharing one ``top_k`` budget,
        so facts and episodes compete on relevance rather than each getting a
        fixed slice. When ``include_episodes`` is False the base is facts only.
        Callers may pass ``memory_types`` to search other non-episodic types;
        ``episodic`` is added or removed based on ``include_episodes``. Optional
        ``include_summaries`` / ``include_turns`` blend those in after the base
        block, deduped by content (best-effort; a blend fetch failure never
        breaks the base result).
        """
        store = self._get_store()
        # Facts + episodes share one ranked query and one top_k budget: episodic
        # is added when include_episodes is True and stripped when it is False.
        if memory_types is not None and "episodic" in memory_types and not include_episodes:
            logger.warning(
                "Episodic memories requested via memory_types are only returned when include_episodes=True; "
                "proceeding without episodic memories and using facts or other requested memory types only."
            )
        if memory_types is not None:
            base_memory_types = [t for t in memory_types if t != "episodic"]
        else:
            base_memory_types = ["fact"]
        if include_episodes:
            base_memory_types = [*base_memory_types, "episodic"]
        base_memory_types = base_memory_types or ["fact"]
        search_kwargs = {
            "search_terms": search_terms,
            "memory_id": memory_id,
            "user_id": user_id,
            "role": role,
            "memory_types": base_memory_types,
            "thread_id": thread_id,
            "top_k": top_k,
            "tags_all": tags_all,
            "tags_any": tags_any,
            "exclude_tags": exclude_tags,
            "include_superseded": include_superseded,
            "min_salience": min_salience,
            "min_confidence": min_confidence,
            "created_after": created_after,
            "created_before": created_before,
        }
        if tenant_id is not None:
            search_kwargs["tenant_id"] = tenant_id
        if scopes is not None:
            search_kwargs["scopes"] = scopes
        if ctx is not None:
            search_kwargs["ctx"] = ctx

        pinned: list[dict[str, Any]] = []
        if include_pins and agent_id and ctx is not None:
            pinned = store.resolve_pinned_memories(
                agent_id=agent_id,
                ctx=ctx,
                memory_types=base_memory_types,
                top_k=top_k,
                include_superseded=include_superseded,
            )
        remaining_top_k = top_k - len(pinned)
        base = [] if remaining_top_k <= 0 else store.search(**{**search_kwargs, "top_k": remaining_top_k})

        results: list[dict[str, Any]] = []
        seen_content: set[str] = set()
        seen_ids: set[str] = set()

        def _extend(docs: list[dict[str, Any]]) -> None:
            for doc in docs:
                doc_id = str(doc.get("id") or "").strip()
                content = str(doc.get("content") or "").strip()
                if doc_id and doc_id in seen_ids:
                    continue
                if content and content in seen_content:
                    continue
                if doc_id:
                    seen_ids.add(doc_id)
                if content:
                    seen_content.add(content)
                results.append(doc)

        _extend(pinned)
        _extend(base)

        if not user_id and not scopes and not pinned:
            return base

        if include_summaries:
            try:
                summaries = store.search_summaries(
                    search_terms=search_terms,
                    user_id=user_id,
                    thread_id=thread_id,
                    top_k=summary_top_k if summary_top_k is not None else top_k,
                    exclude_tags=exclude_tags,
                    tenant_id=tenant_id,
                    scopes=scopes,
                    ctx=ctx,
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning("search_cosmos: include_summaries search failed (%s); skipping summaries", exc)
                summaries = []
            # Summaries come after facts/episodes (and before turns) so they
            # neither jump the ranking nor evict memories under truncation.
            _extend(summaries)

        if include_turns:
            try:
                turns = store.search_turns(
                    search_terms=search_terms,
                    user_id=user_id,
                    thread_id=thread_id,
                    role=role,
                    top_k=turn_top_k if turn_top_k is not None else top_k,
                    exclude_tags=exclude_tags,
                    created_after=created_after,
                    created_before=created_before,
                    tenant_id=tenant_id,
                    scopes=scopes,
                    ctx=ctx,
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning("search_cosmos: include_turns turn search failed (%s); returning memories only", exc)
                turns = []
            _extend(turns)
        return results

    def search_summaries(
        self,
        search_terms: str,
        user_id: str,
        thread_id: Optional[str] = None,
        top_k: int = 5,
        tags_all: Optional[list[str]] = None,
        tags_any: Optional[list[str]] = None,
        exclude_tags: Optional[list[str]] = None,
    ) -> list[dict[str, Any]]:
        """Vector-search the user's thread + user summaries (see ``MemoryStore.search_summaries``)."""
        return self._get_store().search_summaries(
            search_terms=search_terms,
            user_id=user_id,
            thread_id=thread_id,
            top_k=top_k,
            tags_all=tags_all,
            tags_any=tags_any,
            exclude_tags=exclude_tags,
        )

    def search_turns(
        self,
        search_terms: str,
        user_id: str,
        thread_id: Optional[str] = None,
        role: Optional[str] = None,
        top_k: int = 5,
        tags_all: Optional[list[str]] = None,
        tags_any: Optional[list[str]] = None,
        exclude_tags: Optional[list[str]] = None,
        created_after: Optional[str | datetime] = None,
        created_before: Optional[str | datetime] = None,
    ) -> list[dict[str, Any]]:
        """Vector-search the raw conversation log (requires turn embeddings).

        Searches the turns container directly. Only vector-searchable when
        ``enable_turn_embeddings`` was set when the turns were written.
        ``user_id`` is required and always filters the results. Passing
        ``thread_id`` as well scopes the search to a single partition; omitting
        it fans out across partitions, filtered by ``user_id`` in the WHERE
        clause.
        """
        return self._get_store().search_turns(
            search_terms=search_terms,
            user_id=user_id,
            thread_id=thread_id,
            role=role,
            top_k=top_k,
            tags_all=tags_all,
            tags_any=tags_any,
            exclude_tags=exclude_tags,
            created_after=created_after,
            created_before=created_before,
        )

    def get_memory_history(
        self,
        memory_id: str,
        user_id: str,
        thread_id: Optional[str] = None,
        *,
        max_depth: int = 20,
    ) -> list[dict[str, Any]]:
        """Return a memory's superseded predecessors, most-recently-superseded first.

        AMT supersedes rather than deletes, so the full history of a fact - how a
        preference, decision, or attribute changed over time - is preserved. This
        walks the ``superseded_by`` chain backwards from *memory_id* and returns
        every version it replaced (transitively), each carrying its
        ``superseded_at`` / ``supersede_reason`` audit fields. Useful for
        answering "what changed?" / "what was it before?" questions that a
        current-value-only view cannot.
        """
        return self._get_store().get_memory_history(
            memory_id,
            user_id,
            thread_id,
            max_depth=max_depth,
        )

    def get_thread(
        self,
        thread_id: str,
        user_id: Optional[str] = None,
        recent_k: Optional[int] = None,
        tags_all: Optional[list[str]] = None,
        tags_any: Optional[list[str]] = None,
        exclude_tags: Optional[list[str]] = None,
        include_superseded: bool = False,
        created_after: Optional[str | datetime] = None,
        created_before: Optional[str | datetime] = None,
    ) -> list[dict[str, Any]]:
        """Retrieve an entire thread (turns) from Cosmos DB."""
        return self._get_store().get_thread(
            thread_id=thread_id,
            user_id=user_id,
            recent_k=recent_k,
            tags_all=tags_all,
            tags_any=tags_any,
            exclude_tags=exclude_tags,
            include_superseded=include_superseded,
            created_after=created_after,
            created_before=created_before,
        )

    def get_thread_summary(
        self,
        user_id: str,
        thread_id: str,
        recent_k: Optional[int] = None,
    ) -> list[dict[str, Any]]:
        """Retrieve active thread summaries for ``(user_id, thread_id)``, newest first."""
        return self._get_store().get_thread_summary(
            user_id=user_id,
            thread_id=thread_id,
            recent_k=recent_k,
        )

    def get_episodes(
        self,
        user_id: str,
        thread_id: Optional[str] = None,
        recent_k: Optional[int] = None,
    ) -> list[dict[str, Any]]:
        """Retrieve active episodic memories for ``user_id``, newest first."""
        return self._get_store().get_episodes(
            user_id=user_id,
            thread_id=thread_id,
            recent_k=recent_k,
        )

    def get_user_summary(self, user_id: str) -> Optional[dict[str, Any]]:
        """Retrieve the user's summary document from Cosmos DB, or ``None`` if absent."""
        return self._get_store().get_user_summary(user_id=user_id)

    def list_tags(
        self,
        user_id: str,
        *,
        thread_id: Optional[str] = None,
        prefix: Optional[str] = None,
        include_sys: bool = False,
    ) -> list[str]:
        """Return sorted distinct tags for a user."""
        return self._get_store().list_tags(
            user_id,
            thread_id=thread_id,
            prefix=prefix,
            include_sys=include_sys,
        )

    def add_tags(
        self,
        memory_id: str,
        user_id: str,
        thread_id: str,
        memory_type: str,
        tags: list[str],
    ) -> None:
        """Add tags to an existing memory document."""
        return self._get_store().add_tags(memory_id, user_id, thread_id, memory_type, tags)

    def remove_tags(
        self,
        memory_id: str,
        user_id: str,
        thread_id: str,
        memory_type: str,
        tags: list[str],
    ) -> None:
        """Remove tags from an existing memory document."""
        return self._get_store().remove_tags(memory_id, user_id, thread_id, memory_type, tags)

    def get_procedural_prompt(self, user_id: str) -> Optional[str]:
        """Return the compiled procedural system prompt for a user.

        The prompt is a deterministic projection of the user's active procedures
        (compiled on demand), not a stored record. Returns None when the user has
        no active procedures.
        """
        prompt = self._get_pipeline().build_procedural_context(user_id)
        return prompt or None

    def get_procedural_history(self, user_id: str, limit: int = 10) -> list[dict[str, Any]]:
        """Return synthesized procedural docs for a user, newest first."""
        return self._get_store().get_procedural_history(user_id=user_id, limit=limit)

    def get_procedural_memories(
        self,
        user_id: str,
        priority: Optional[str] = None,
        category: Optional[str] = None,
        min_salience: Optional[float] = None,
        include_superseded: bool = False,
    ) -> list[dict[str, Any]]:
        """Retrieve active procedural memories for a user."""
        return self._get_store().get_procedural_memories(user_id, priority, category, min_salience, include_superseded)

    def search_episodic_memories(
        self,
        user_id: str,
        search_terms: str,
        top_k: int = 5,
        min_salience: Optional[float] = None,
        include_superseded: bool = False,
    ) -> list[dict[str, Any]]:
        """Semantic search across episodic memories for a user."""
        return self._get_store().search_episodic(
            user_id=user_id,
            search_terms=search_terms,
            top_k=top_k,
            min_salience=min_salience,
            include_superseded=include_superseded,
        )

    def retrieve_procedures(
        self,
        user_id: str,
        search_terms: str,
        top_k: int = 5,
        *,
        procedure_kind: Optional[str] = None,
        status: Optional[str] = "active",
        include_superseded: bool = False,
    ) -> list[dict[str, Any]]:
        """Context-aware semantic retrieval of a user's procedures.

        Ranks active procedures by relevance to ``search_terms`` (a task or
        situation), optionally narrowed by ``procedure_kind``. Pass ``status=None``
        to include non-active (e.g. candidate) procedures.
        """
        return self._get_store().retrieve_procedures(
            user_id=user_id,
            search_terms=search_terms,
            top_k=top_k,
            procedure_kind=procedure_kind,
            status=status,
            include_superseded=include_superseded,
        )

    def build_procedural_context(self, user_id: str, task: Optional[str] = None) -> str:
        """Compile a procedural system-prompt projection for prompt injection.

        With no ``task``, returns the user's always-on behavioral policies. With a
        ``task`` string, also folds in relevant task procedures (skills).
        """
        return self._get_pipeline().build_procedural_context(user_id, task)

    def build_episodic_context(
        self,
        user_id: str,
        query: str,
        top_k: int = 3,
    ) -> str:
        """Build formatted context of relevant past experiences."""
        return self._get_store().build_episodic_context(
            user_id=user_id,
            query=query,
            top_k=top_k,
        )

    def extract_memories(
        self,
        user_id: str,
        thread_id: str,
        recent_k: Optional[int] = None,
        *,
        tenant_id: Optional[str] = None,
    ) -> dict[str, int]:
        """Extract facts and episodic memories from a thread."""
        with tenant_scope(tenant_id):
            return self._get_pipeline().extract_memories(user_id, thread_id, recent_k)

    def extract_episodes(
        self,
        user_id: str,
        thread_id: str,
        *,
        flush: bool = False,
        tenant_id: Optional[str] = None,
    ) -> dict[str, int]:
        """Segment the thread's open turn stream into episodes at detected boundaries.

        Episodes are created at idle time-gaps (detected only once a later turn
        reveals the gap), topic shifts, and a max-size cap. A focused session
        shorter than the max-size cap therefore episodizes only lazily - on the
        next turn after the idle gap - and a one-shot session that never resumes
        is not episodized at all under the auto path. Pass ``flush=True`` at the
        end of a conversation (or benchmark run) to drain the trailing open
        segment immediately; integrators that know when a session ends should
        call this on session close.

        Only supported when the in-process backend owns processing; when a
        Durable Function app is the active processor this raises
        ``NotImplementedError`` so writes are not split away from that backend.
        """
        processor = self._get_processor()
        if not isinstance(processor, InProcessProcessor):
            raise NotImplementedError(
                "Episode extraction runs in-process; manual invocation via the SDK is not "
                "supported when the Durable Function app is the active processor."
            )
        with tenant_scope(tenant_id):
            return self._get_pipeline().extract_episodes(user_id, thread_id, flush=flush)

    def synthesize_procedural(self, user_id: str, *, force: bool = False) -> dict[str, Any]:
        processor = self._get_processor()
        if not isinstance(processor, InProcessProcessor):
            raise NotImplementedError(
                "Procedural synthesis runs automatically after reconcile in durable mode; "
                "manual invocation via the SDK is not supported when the Durable Function "
                "app is the active processor. Use get_procedural_prompt() to read the "
                "latest synthesized prompt."
            )
        return processor.synthesize_procedural(user_id=user_id, force=force)

    def generate_thread_summary(
        self,
        user_id: str,
        thread_id: str,
        recent_k: Optional[int] = None,
        *,
        tenant_id: Optional[str] = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Generate a thread summary."""
        with tenant_scope(tenant_id):
            return self._get_pipeline().generate_thread_summary(user_id, thread_id, recent_k)

    def generate_user_summary(
        self,
        user_id: str,
        thread_ids: Optional[list[str]] = None,
        recent_k: Optional[int] = None,
        *,
        tenant_id: Optional[str] = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Generate a cross-thread user summary."""
        with tenant_scope(tenant_id):
            return self._get_pipeline().generate_user_summary(user_id, thread_ids, recent_k)

    def reconcile(
        self,
        user_id: str | None = None,
        n: Optional[int] = None,
        *,
        scope_key: str | None = None,
        tenant_id: str | None = None,
        ctx: SecurityContext | None = None,
    ) -> dict[str, int]:
        """Reconcile facts in a user's private scope or an authorized shared scope."""
        from .thresholds import get_dedup_pool_size

        if scope_key is None and ctx is None and tenant_id is None:
            if not user_id:
                raise ValidationError("user_id is required")
            return self._get_pipeline().reconcile_memories(user_id, n if n is not None else get_dedup_pool_size())

        if ctx is None:
            if not user_id:
                raise ValidationError("ctx or user_id is required for scoped reconcile")
            ctx = SecurityContext.from_user_id(user_id)
        if tenant_id is not None and tenant_id != ctx.tenant_id:
            raise ValidationError(
                "tenant_id cannot override the SecurityContext tenant; the tenant is the hard "
                "isolation boundary set by the trusted host, not a request parameter"
            )
        scope_key = scope_key or private_scope_key_for_principal(ctx.principal)
        resolution = resolve_scope_access(ctx, [scope_key], "write")
        if scope_key not in resolution.allowed_scopes:
            raise ValidationError(f"write permission denied for scope_key={scope_key!r}")
        reconcile_user_id = user_id or (
            ctx.principal.split(":", 1)[1] if ctx.principal and ":" in ctx.principal else ""
        )
        return self._get_pipeline().reconcile_memories(
            reconcile_user_id,
            n if n is not None else get_dedup_pool_size(),
            tenant_id=ctx.tenant_id,
            scope_key=scope_key,
        )

    def process_now(self, *, user_id: str, thread_id: str) -> "ProcessThreadResult":
        """Force the processor to run the full pipeline RIGHT NOW for one thread.

        For the in-process processor this fires all five steps:
        ``thread_summary → extract → reconcile → procedural → user_summary``.
        For the durable processor this remains a no-op (the sibling Function
        app drives the pipeline off the Cosmos DB change feed).

        Transient failures in ``procedural`` and ``user_summary`` (LLM
        rate-limit / timeout, Cosmos 429 / 5xx, defensive ``LLMError``) are
        caught and logged as warnings so the per-thread work already
        persisted by the prior steps is not erased. Permanent failures
        (config bugs, auth errors, 4xx Cosmos errors, Python builtins like
        ``KeyError`` / ``TypeError``) are re-raised - silencing them turns
        operational issues into invisible ``WARNING`` lines.
        """
        self._require_cosmos()
        processor = self._get_processor()
        turns = self.get_thread(thread_id=thread_id, user_id=user_id) or []
        result = processor.process_thread(user_id=user_id, thread_id=thread_id, turns=turns)
        if isinstance(processor, InProcessProcessor):
            try:
                result.procedural = processor.synthesize_procedural(user_id=user_id)
            except Exception as exc:
                if not is_transient_tail_step_error(exc):
                    raise
                logger.warning(
                    "process_now: synthesize_procedural failed (transient) for user_id=%s: %s",
                    user_id,
                    exc,
                )
            try:
                user_summary_result = processor.process_user_summary(user_id=user_id)
                if user_summary_result is not None and user_summary_result.summary:
                    result.user_summary = user_summary_result.summary
            except Exception as exc:
                if not is_transient_tail_step_error(exc):
                    raise
                logger.warning(
                    "process_now: process_user_summary failed (transient) for user_id=%s: %s",
                    user_id,
                    exc,
                )
        return result

    def process_now_and_wait(self, *, user_id: str, thread_id: str, timeout: float = 30.0) -> bool:
        """Force processing and block until a thread summary exists (or timeout)."""
        import time

        self._require_cosmos()
        processor = self._get_processor()
        turns = self.get_thread(thread_id=thread_id, user_id=user_id) or []
        processor.process_thread(user_id=user_id, thread_id=thread_id, turns=turns)
        if isinstance(processor, InProcessProcessor):
            return True
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if self._summary_exists(user_id=user_id, thread_id=thread_id):
                return True
            time.sleep(0.5)
        return False

    def _summary_exists(self, *, user_id: str, thread_id: str) -> bool:
        from azure.cosmos.exceptions import (
            CosmosHttpResponseError,
            CosmosResourceNotFoundError,
        )

        try:
            results = self.get_thread_summary(user_id=user_id, thread_id=thread_id, recent_k=1)
        except CosmosResourceNotFoundError:
            return False
        except CosmosHttpResponseError as exc:
            logger.warning(
                "_summary_exists: Cosmos error user_id=%s thread_id=%s status=%s: %s",
                user_id,
                thread_id,
                getattr(exc, "status_code", None),
                exc,
            )
            return False
        return bool(results)
