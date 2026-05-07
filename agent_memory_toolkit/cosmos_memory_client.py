"""CosmosMemoryClient: unified local and cloud agent memory management.

Consolidates the former ``AgentMemory`` orchestrator and ``CosmosMemoryStore``
into a single class that owns local CRUD, Cosmos DB connection/CRUD,
embedding-based search, and LLM-powered processing pipeline.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, Optional

from ._query_builder import _QueryBuilder
from ._utils import (
    VALID_ROLES,
    VALID_TYPES,
    _build_container_kwargs,
    _build_memory_query_builder,
    _container_policies,
    _cosmos_container_offer_throughput,
    _make_memory,
    _resolve_cosmos_provisioning_autoscale_max_ru,
    _resolve_cosmos_throughput_mode,
    _resolve_distance_function,
    _resolve_embedding_data_type,
    _resolve_embedding_dimensions,
    _resolve_full_text_language,
    _validate_connection,
    _validate_hybrid_search,
)
from .chat import ChatClient
from .embeddings import EmbeddingsClient
from .exceptions import (
    CosmosNotConnectedError,
    CosmosOperationError,
    MemoryNotFoundError,
    ValidationError,
)
from .models import MemoryRecord
from .processors import (
    InProcessProcessor,
    MemoryProcessor,
)

if TYPE_CHECKING:  # pragma: no cover - typing-only import
    from .processors import ProcessThreadResult, UserSummaryResult  # noqa: F401

logger = logging.getLogger(__name__)


class CosmosMemoryClient:
    """Manages agent memories with local storage and Cosmos DB.

    Authentication uses ``azure-identity`` by default.  If no explicit
    credential is passed for Cosmos DB or AI Foundry, a
    ``DefaultAzureCredential`` is created automatically.

    For Cosmos DB control-plane operations (database/container creation),
    Entra ID RBAC is in private preview. If RBAC is unavailable, pass
    ``cosmos_key`` as a fallback — the SDK will use it when no
    ``TokenCredential`` is available.

    Parameters
    ----------
    cosmos_endpoint : str, optional
        The Cosmos DB account endpoint URL.
    cosmos_credential : TokenCredential, optional
        Azure credential for Cosmos DB (Entra ID / RBAC).
    cosmos_key : str, optional
        Cosmos DB account key. Used as fallback when ``cosmos_credential``
        is not provided and ``DefaultAzureCredential`` is unavailable.
    cosmos_database : str, optional
        Cosmos DB database name.
    cosmos_container : str, optional
        Cosmos DB container name.
    ai_foundry_endpoint : str, optional
        Azure OpenAI endpoint URL for embeddings and LLM.
    ai_foundry_credential : TokenCredential, optional
        Azure credential for the AI Foundry endpoint.
    ai_foundry_api_key : str, optional
        API key for Azure OpenAI (takes precedence over credential).
    embedding_deployment_name : str, optional
        Embedding model deployment name (default ``text-embedding-3-large``).
    embedding_dimensions : int, optional
        Dimensionality of embedding vectors.
    chat_deployment_name : str, optional
        LLM model deployment name (default ``gpt-4o-mini``).
    use_default_credential : bool, optional
        Automatically create ``DefaultAzureCredential`` when ``True``.
    """

    def __init__(
        self,
        cosmos_endpoint: Optional[str] = None,
        cosmos_credential: Optional[Any] = None,
        cosmos_key: Optional[str] = None,
        cosmos_database: Optional[str] = None,
        cosmos_container: Optional[str] = None,
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
        processor: Optional[MemoryProcessor] = None,
    ) -> None:
        # Local store
        self.local_memory: list[dict[str, Any]] = []
        self._unflushed_turn_counts: dict[tuple[str, str], int] = {}
        self._warned_owner_skip: bool = False
        self._warned_counter_unreachable: bool = False

        # Store kwargs directly
        self._cosmos_endpoint = cosmos_endpoint
        self._cosmos_credential = cosmos_credential
        self._cosmos_key = cosmos_key
        self._cosmos_database = cosmos_database or "ai_memory"
        self._cosmos_container = cosmos_container or "memories"
        self._cosmos_counter_container = cosmos_counter_container or "counter"
        self._cosmos_lease_container = cosmos_lease_container or "leases"
        self._cosmos_throughput_mode = _resolve_cosmos_throughput_mode(cosmos_throughput_mode)
        self._cosmos_autoscale_max_ru = _resolve_cosmos_provisioning_autoscale_max_ru(
            throughput_mode=self._cosmos_throughput_mode,
            autoscale_max_ru=cosmos_autoscale_max_ru,
        )

        self._ai_foundry_endpoint = ai_foundry_endpoint
        self._ai_foundry_credential = ai_foundry_credential
        self._ai_foundry_api_key = ai_foundry_api_key
        self._embedding_deployment_name = embedding_deployment_name
        self._embedding_dimensions = _resolve_embedding_dimensions(embedding_dimensions)
        self._chat_deployment_name = chat_deployment_name

        # Credential resolution priority:
        #   1. Explicit cosmos_credential / ai_foundry_credential (highest priority)
        #   2. Explicit cosmos_key / ai_foundry_api_key (relief for environments
        #      where Cosmos control-plane RBAC is in private preview, etc.)
        #   3. DefaultAzureCredential() when use_default_credential=True
        #
        # Track ownership separately so close() doesn't accidentally close a
        # user-supplied credential or leak one we created.
        self._owns_cosmos_credential = False
        self._owns_ai_foundry_credential = False
        if self._cosmos_credential is None and self._cosmos_key:
            self._cosmos_credential = self._cosmos_key

        if use_default_credential:
            needs_cosmos = self._cosmos_credential is None
            needs_embed = self._ai_foundry_credential is None and not self._ai_foundry_api_key
            if needs_cosmos or needs_embed:
                try:
                    from azure.identity import DefaultAzureCredential

                    if needs_cosmos:
                        self._cosmos_credential = DefaultAzureCredential()
                        self._owns_cosmos_credential = True
                    if needs_embed:
                        self._ai_foundry_credential = DefaultAzureCredential()
                        self._owns_ai_foundry_credential = True
                except ImportError:
                    pass

        # Internal Cosmos SDK handles
        self._cosmos_client: Any = None
        self._container_client: Any = None
        self._counter_container_client: Any = None

        # Composed sub-clients
        self._embeddings_client = EmbeddingsClient(
            endpoint=self._ai_foundry_endpoint,
            credential=self._ai_foundry_credential,
            api_key=self._ai_foundry_api_key,
            model=self._embedding_deployment_name,
            dimensions=self._embedding_dimensions,
        )
        self._chat_client = ChatClient(
            endpoint=self._ai_foundry_endpoint,
            credential=self._ai_foundry_credential,
            api_key=self._ai_foundry_api_key,
            model=self._chat_deployment_name,
        )

        # ProcessingPipeline is lazily initialized when Cosmos is connected
        self._pipeline: Any = None

        # Pluggable backend that owns summarize/extract/dedup. ``None`` means
        # "lazily construct an InProcessProcessor on first use, sharing the
        # client's pipeline / LLM / embeddings."
        self._processor: Optional[MemoryProcessor] = processor
        self._processor_explicit: bool = processor is not None

        # Auto-connect and create store when Cosmos endpoint is provided
        if self._cosmos_endpoint:
            self.create_memory_store()

        logger.info("CosmosMemoryClient initialized")

    # ------------------------------------------------------------------
    # Context manager / cleanup
    # ------------------------------------------------------------------

    def __enter__(self) -> CosmosMemoryClient:
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.close()

    def close(self) -> None:
        """Close the underlying Cosmos client and release resources."""
        if self._cosmos_client is not None:
            self._cosmos_client.close()
            self._cosmos_client = None
            self._container_client = None
            self._counter_container_client = None
            logger.info("Cosmos client closed")
        # Drain LLM/embeddings httpx pools — openai.AzureOpenAI keeps them
        # open across `with` blocks otherwise.
        try:
            self._chat_client.close_sync()
        except Exception:
            pass
        try:
            self._embeddings_client.close()
        except Exception:
            pass
        # Close credentials we created ourselves (sync DefaultAzureCredential
        # holds an underlying token cache + HTTP transport).
        for owns, cred in (
            (self._owns_cosmos_credential, self._cosmos_credential),
            (self._owns_ai_foundry_credential, self._ai_foundry_credential),
        ):
            if not owns or cred is None:
                continue
            close = getattr(cred, "close", None)
            if callable(close):
                try:
                    close()
                except Exception:
                    pass

    # ------------------------------------------------------------------
    # Read-only properties
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # Local operations
    # ------------------------------------------------------------------

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
        if memory_type == "turn" and not thread_id:
            raise ValidationError(
                "thread_id is required for memory_type='turn' so the auto-trigger "
                "counter can group turns per conversation. Set thread_id explicitly."
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
        memory_type: Optional[str] = None,
    ) -> list[dict[str, Any]]:
        """Retrieve memories from the local store.

        All filter parameters are optional. When none are provided every
        memory is returned. Filters are combined with AND logic.
        """
        logger.debug(
            "get_local memory_id=%s user_id=%s role=%s type=%s",
            memory_id,
            user_id,
            role,
            memory_type,
        )
        results = self.local_memory

        if memory_id is not None:
            results = [m for m in results if m["id"] == memory_id]
        if user_id is not None:
            results = [m for m in results if m["user_id"] == user_id]
        if role is not None:
            results = [m for m in results if m["role"] == role]
        if memory_type is not None:
            results = [m for m in results if m["type"] == memory_type]

        return results

    def update_local(
        self,
        memory_id: str,
        content: Optional[str] = None,
        role: Optional[str] = None,
        memory_type: Optional[str] = None,
        metadata: Optional[dict[str, Any]] = None,
    ) -> None:
        """Update an existing memory in the local store.

        Only the fields that are provided (not ``None``) will be updated.

        Raises
        ------
        MemoryNotFoundError
            If no memory with the given id exists.
        ValidationError
            If an invalid role or memory_type is provided.
        """
        for memory in self.local_memory:
            if memory["id"] == memory_id:
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
        """Delete a memory from the local store by id.

        Raises
        ------
        MemoryNotFoundError
            If no memory with the given id exists.
        """
        for i, memory in enumerate(self.local_memory):
            if memory["id"] == memory_id:
                self.local_memory.pop(i)
                return

        raise MemoryNotFoundError(memory_id=memory_id)

    # ------------------------------------------------------------------
    # Cosmos DB connection
    # ------------------------------------------------------------------

    def connect_cosmos(
        self,
        endpoint: Optional[str] = None,
        credential: Optional[Any] = None,
        key: Optional[str] = None,
        database: Optional[str] = None,
        container: Optional[str] = None,
    ) -> None:
        """Establish a connection to a Cosmos DB container.

        Parameters override whatever was set in ``__init__``.  After this
        call the Cosmos CRUD methods are ready to use.

        Either *credential* (Entra ID) or *key* (account key) may be
        provided.  *credential* takes precedence.
        """
        self._cosmos_endpoint = endpoint or self._cosmos_endpoint
        if credential is not None:
            self._cosmos_credential = credential
        elif key is not None:
            self._cosmos_credential = key
            self._cosmos_key = key
        self._cosmos_database = database or self._cosmos_database
        self._cosmos_container = container or self._cosmos_container

        _validate_connection(
            self._cosmos_endpoint,
            self._cosmos_credential,
            self._cosmos_database,
            self._cosmos_container,
        )

        try:
            from azure.cosmos import CosmosClient

            self._drain_cosmos_client()

            client = CosmosClient(self._cosmos_endpoint, credential=self._cosmos_credential)
            db = client.get_database_client(self._cosmos_database)
            container_handle = db.get_container_client(self._cosmos_container)

            self._cosmos_client = client
            self._container_client = container_handle
            self._init_pipeline()
        except Exception as exc:
            raise CosmosOperationError(f"Failed to connect to Cosmos DB: {exc}") from exc

        logger.info(
            "Connected to Cosmos DB %s/%s",
            self._cosmos_database,
            self._cosmos_container,
        )

    def create_memory_store(
        self,
        database: Optional[str] = None,
        container: Optional[str] = None,
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
    ) -> None:
        """Create the Cosmos DB database and container for memories.

        After successful creation the instance is connected and ready
        for CRUD operations.

        Either *credential* (Entra ID) or *key* (account key) may be
        provided.  *credential* takes precedence.  When neither is
        given the instance falls back to whatever was set at init time.

        The memories container is provisioned with:

        * Hierarchical partition key ``[/user_id, /thread_id]``
        * ``diskANN`` vector index on ``/embedding``
        * Full-text index on ``/content``
        * Throughput behavior controlled by *throughput_mode*

        Separate counter and lease containers are also provisioned.
        In ``serverless`` mode no RU/s throughput is specified.
        In ``autoscale`` mode all required containers use the same
        autoscale max RU from *autoscale_max_ru*.
        """
        self._cosmos_endpoint = endpoint or self._cosmos_endpoint
        if credential is not None:
            self._cosmos_credential = credential
        elif key is not None:
            self._cosmos_credential = key
            self._cosmos_key = key
        self._cosmos_database = database or self._cosmos_database
        self._cosmos_container = container or self._cosmos_container
        self._cosmos_counter_container = counter_container or self._cosmos_counter_container
        self._cosmos_lease_container = lease_container or self._cosmos_lease_container
        self._cosmos_throughput_mode = _resolve_cosmos_throughput_mode(
            throughput_mode if throughput_mode is not None else self._cosmos_throughput_mode
        )
        self._cosmos_autoscale_max_ru = _resolve_cosmos_provisioning_autoscale_max_ru(
            throughput_mode=self._cosmos_throughput_mode,
            autoscale_max_ru=(autoscale_max_ru if autoscale_max_ru is not None else self._cosmos_autoscale_max_ru),
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

            client = CosmosClient(self._cosmos_endpoint, credential=self._cosmos_credential)

            db = client.create_database_if_not_exists(id=self._cosmos_database)

            partition_key = PartitionKey(path=["/user_id", "/thread_id"], kind="MultiHash")
            lease_partition_key = PartitionKey(path="/id")
            vec_policy, idx_policy, ft_policy = _container_policies(
                embedding_dimensions=embedding_dimensions or self._embedding_dimensions or 1536,
                embedding_data_type=_resolve_embedding_data_type(embedding_data_type),
                distance_function=_resolve_distance_function(distance_function),
                full_text_language=_resolve_full_text_language(full_text_language),
            )
            offer_throughput = _cosmos_container_offer_throughput(
                throughput_mode=self._cosmos_throughput_mode,
                autoscale_max_ru=self._cosmos_autoscale_max_ru,
                throughput_properties_cls=ThroughputProperties,
            )

            container_handle = db.create_container_if_not_exists(
                **_build_container_kwargs(
                    container_id=self._cosmos_container,
                    partition_key=partition_key,
                    offer_throughput=offer_throughput,
                    default_ttl=-1,
                    indexing_policy=idx_policy,
                    vector_embedding_policy=vec_policy,
                    full_text_policy=ft_policy,
                )
            )

            db.create_container_if_not_exists(
                **_build_container_kwargs(
                    container_id=self._cosmos_counter_container,
                    partition_key=partition_key,
                    offer_throughput=offer_throughput,
                )
            )

            db.create_container_if_not_exists(
                **_build_container_kwargs(
                    container_id=self._cosmos_lease_container,
                    partition_key=lease_partition_key,
                    offer_throughput=offer_throughput,
                )
            )
            self._cosmos_client = client
            self._container_client = container_handle
            self._init_pipeline()
        except Exception as exc:
            raise CosmosOperationError(f"Failed to create memory store: {exc}") from exc

        logger.info(
            "Created memory store %s/%s with counter container %s and lease container %s",
            self._cosmos_database,
            self._cosmos_container,
            self._cosmos_counter_container,
            self._cosmos_lease_container,
        )

    def _init_pipeline(self) -> None:
        """Initialize the ProcessingPipeline with the current container client."""
        from .pipeline import ProcessingPipeline

        self._pipeline = ProcessingPipeline(
            cosmos_container=self._container_client,
            chat_client=self._chat_client,
            embeddings_client=self._embeddings_client,
        )
        self._warn_on_embedding_dim_mismatch()

    def _drain_cosmos_client(self) -> None:
        """Close any prior Cosmos client before reassigning the field.

        Repeated ``connect_cosmos`` / ``create_memory_store`` calls on the
        same instance must not leak the prior client's httpx pool / FDs.
        """
        prior = self._cosmos_client
        if prior is None:
            return
        close = getattr(prior, "close", None)
        if callable(close):
            try:
                close()
            except Exception:
                logger.warning("Failed to close prior Cosmos client during reconnect", exc_info=True)
        self._cosmos_client = None
        self._container_client = None
        self._counter_container_client = None
        self._pipeline = None
        if not self._processor_explicit:
            self._processor = None

    def _warn_on_embedding_dim_mismatch(self) -> None:
        """Log a WARNING if the resolved embedding dim differs from the container's policy.

        Writing vectors at a different dimensionality than the container's
        ``vectorEmbeddingPolicy`` produces vectors that will not match in
        similarity search — the failure mode is silent (empty / wrong
        results), so we surface it loudly at connect time. We never raise:
        the user may have a legitimate reason (e.g. the container was created
        outside the SDK and they are migrating).
        """
        if self._container_client is None or self._embedding_dimensions is None:
            return
        try:
            props = self._container_client.read()
        except Exception:
            return
        policy = (props or {}).get("vectorEmbeddingPolicy") or {}
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

    def _get_processor(self) -> MemoryProcessor:
        """Return the active processor, lazily building an in-process default."""
        if self._processor is None:
            if self._pipeline is None:
                self._init_pipeline()
            self._processor = InProcessProcessor(pipeline=self._pipeline)
        return self._processor

    def _get_counter_container(self) -> Any:
        """Return a lazy handle to the counter container.

        Best-effort: returns ``None`` if the container is unreachable (e.g. the
        operator brought their own memory container without provisioning a
        counter container). Auto-trigger callers must tolerate ``None`` and
        skip the increment.

        Logs a one-shot WARN on first failure so operators don't silently
        lose all memory processing because of a missing counter container.
        """
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
                    "Counter container %s/%s unreachable (%s: %s); "
                    "auto-trigger DISABLED for the lifetime of this client. "
                    "Provision the container or set MEMORY_PROCESSOR_OWNER=durable "
                    "if processing runs in the Function App.",
                    self._cosmos_database,
                    self._cosmos_counter_container,
                    type(exc).__name__,
                    exc,
                )
            return None

    def _maybe_auto_trigger(self, turn_counts: dict[tuple[str, str], int]) -> None:
        """After ``push_to_cosmos``, increment counters and run granular processor steps.

        Each ``*_EVERY_N`` threshold fires its own pipeline step independently
        (mirroring the function-app split-orchestrator behavior):

        - ``FACT_EXTRACTION_EVERY_N`` → ``processor.process_extract_memories``
        - ``THREAD_SUMMARY_EVERY_N``  → ``processor.process_thread_summary``
        - ``USER_SUMMARY_EVERY_N``    → ``processor.process_user_summary``

        Only fires for the in-process backend; the durable backend relies on
        the change-feed-driven function app, which uses the same counter
        container and would otherwise double-fire. Failures here are logged
        and stamped on the counter doc — the user's primary write must never
        fail because of an auto-trigger.
        """
        from ._counters import (
            USER_COUNTER_THREAD_ID,
            crosses_threshold,
            increment_counter_sync,
            stamp_failure_sync,
            thread_counter_id,
            user_counter_id,
        )
        from .processors import InProcessProcessor
        from .thresholds import (
            PROCESSOR_OWNER_DURABLE,
            PROCESSOR_OWNER_INPROCESS,
            get_dedup_every_n,
            get_fact_extraction_every_n,
            get_processor_owner,
            get_thread_summary_every_n,
            get_user_summary_every_n,
        )

        if not turn_counts:
            return

        # Only the InProcess backend needs the client-side trigger; Durable is
        # driven by the FA change-feed processor with the same thresholds.
        try:
            processor = self._get_processor()
        except Exception:  # pragma: no cover - defensive
            return
        if not isinstance(processor, InProcessProcessor):
            return

        # Owner exclusivity: when MEMORY_PROCESSOR_OWNER is set to the
        # opposite backend, defer to it and skip the SDK auto-trigger so we
        # don't double-extract / double-dedup against a shared container.
        owner = get_processor_owner()
        if owner == PROCESSOR_OWNER_DURABLE:
            if not self._warned_owner_skip:
                self._warned_owner_skip = True
                logger.warning(
                    "MEMORY_PROCESSOR_OWNER=durable is set; SDK auto-trigger will not run "
                    "(the function app owns processing for this container). Set "
                    "MEMORY_PROCESSOR_OWNER=inprocess (or unset it) to enable SDK auto-trigger. "
                    "Further skips will be logged at DEBUG level."
                )
            else:
                logger.debug("Skipping SDK auto-trigger: MEMORY_PROCESSOR_OWNER=durable")
            return
        # When unset (None) or PROCESSOR_OWNER_INPROCESS, proceed.

        n_facts = get_fact_extraction_every_n()
        n_summary = get_thread_summary_every_n()
        n_user = get_user_summary_every_n()
        n_dedup = get_dedup_every_n()
        if n_facts == 0 and n_summary == 0 and n_user == 0:
            return

        # Dedup fires every Nth EXTRACT, not every Nth turn. Express that
        # via the same thread counter using the combined threshold
        # n_facts * n_dedup. Disabled when either knob is 0.
        n_dedup_turns = n_facts * n_dedup if (n_facts > 0 and n_dedup > 0) else 0

        counter_container = self._get_counter_container()
        if counter_container is None:
            return

        # Per-user batch totals for the user-scoped counter.
        user_batch_counts: dict[str, int] = {}

        # ---- thread-scoped counter increments + per-step triggers ----
        for (user_id, thread_id), batch_count in turn_counts.items():
            if batch_count <= 0:
                continue
            user_batch_counts[user_id] = user_batch_counts.get(user_id, 0) + batch_count

            try:
                old_count, new_count = increment_counter_sync(
                    counter_container,
                    thread_counter_id(user_id, thread_id),
                    user_id,
                    thread_id,
                    batch_count,
                    owner=PROCESSOR_OWNER_INPROCESS,
                )
            except Exception as exc:  # pragma: no cover - defensive
                logger.warning("Counter increment failed for %s/%s: %s", user_id, thread_id, exc)
                continue

            fire_extract = n_facts > 0 and crosses_threshold(old_count, new_count, n_facts)
            fire_summary = n_summary > 0 and crosses_threshold(old_count, new_count, n_summary)
            fire_dedup = n_dedup_turns > 0 and crosses_threshold(old_count, new_count, n_dedup_turns)

            # Order matters: extract → dedup → summary.
            # Each step gates on its OWN threshold (independent crossings),
            # but when multiple thresholds cross in the same batch we still
            # want the data flow to be: write fresh facts, deduplicate them,
            # then fold the deduplicated set into the summary. Reordering
            # would risk a summary that includes since-removed duplicates
            # or omits just-extracted facts. DO NOT reorder without
            # updating the equivalent block in aio/cosmos_memory_client.py
            # and function_app/orchestrators/.
            if fire_extract:
                try:
                    processor.process_extract_memories(user_id=user_id, thread_id=thread_id)
                except Exception as exc:
                    logger.warning(
                        "Auto-trigger process_extract_memories failed for %s/%s: %s",
                        user_id,
                        thread_id,
                        exc,
                    )
                    stamp_failure_sync(
                        counter_container,
                        thread_counter_id(user_id, thread_id),
                        user_id,
                        thread_id,
                        f"process_extract_memories: {exc!r}",
                    )

            if fire_dedup:
                try:
                    processor.process_reconcile(user_id=user_id)
                except Exception as exc:
                    logger.warning(
                        "Auto-trigger process_reconcile failed for %s: %s",
                        user_id,
                        exc,
                    )
                    stamp_failure_sync(
                        counter_container,
                        thread_counter_id(user_id, thread_id),
                        user_id,
                        thread_id,
                        f"process_reconcile: {exc!r}",
                    )

            if fire_summary:
                try:
                    processor.process_thread_summary(user_id=user_id, thread_id=thread_id)
                except Exception as exc:
                    logger.warning(
                        "Auto-trigger process_thread_summary failed for %s/%s: %s",
                        user_id,
                        thread_id,
                        exc,
                    )
                    stamp_failure_sync(
                        counter_container,
                        thread_counter_id(user_id, thread_id),
                        user_id,
                        thread_id,
                        f"process_thread_summary: {exc!r}",
                    )

        # ---- user-scoped counter increment + per-step trigger ----
        if n_user > 0:
            for user_id, batch_count in user_batch_counts.items():
                try:
                    old_count, new_count = increment_counter_sync(
                        counter_container,
                        user_counter_id(user_id),
                        user_id,
                        USER_COUNTER_THREAD_ID,
                        batch_count,
                        owner=PROCESSOR_OWNER_INPROCESS,
                    )
                except Exception as exc:  # pragma: no cover - defensive
                    logger.warning("User counter increment failed for %s: %s", user_id, exc)
                    continue

                if crosses_threshold(old_count, new_count, n_user):
                    try:
                        processor.process_user_summary(user_id=user_id)
                    except Exception as exc:
                        logger.warning(
                            "Auto-trigger process_user_summary failed for %s: %s",
                            user_id,
                            exc,
                        )
                        stamp_failure_sync(
                            counter_container,
                            user_counter_id(user_id),
                            user_id,
                            USER_COUNTER_THREAD_ID,
                            f"process_user_summary: {exc!r}",
                        )

    def _require_cosmos(self) -> None:
        """Raise if Cosmos DB is not connected."""
        if self._container_client is None:
            raise CosmosNotConnectedError()

    # ------------------------------------------------------------------
    # Cosmos DB CRUD operations
    # ------------------------------------------------------------------

    def add_cosmos(
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
    ) -> str:
        """Add a memory to Cosmos DB.

        Returns the document id (whether newly generated or supplied via
        ``metadata``). For non-``turn`` memory types (``fact``, ``summary``,
        ``episodic``, ``procedural``, ``user_summary``) an embedding is
        generated automatically so vector / hybrid search and deduplication
        work out of the box. Pass ``embed=False`` to skip embedding, or
        ``embedding=[...]`` to supply one explicitly.
        """
        self._require_cosmos()
        kwargs: dict[str, Any] = {
            "user_id": user_id,
            "role": role,
            "content": content,
            "memory_type": memory_type,
            "metadata": metadata or {},
        }
        if thread_id is not None:
            kwargs["thread_id"] = thread_id
        if tags is not None:
            kwargs["tags"] = tags
        if ttl is not None:
            kwargs["ttl"] = ttl
        if salience is not None:
            kwargs["salience"] = salience
        record = MemoryRecord(**kwargs)
        body = record.to_cosmos_dict()

        # Auto-embed derived memories so search / dedup work out of the box.
        # Default: embed everything except raw "turn" memories.
        if embed is None:
            embed = memory_type != "turn"
        if embedding is not None:
            body["embedding"] = embedding
        elif embed and content:
            try:
                body["embedding"] = self._embeddings_client.generate(content)
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "add_cosmos: embedding generation failed for %s (%s); proceeding without embedding",
                    record.id,
                    exc,
                )

        try:
            self._container_client.upsert_item(body=body)
        except Exception as exc:
            raise CosmosOperationError(f"Upsert failed for record {record.id}: {exc}") from exc
        logger.info("add_cosmos id=%s role=%s type=%s", record.id, role, memory_type)
        return record.id

    def push_to_cosmos(self, batch_size: int = 25) -> None:
        """Insert all local memories into Cosmos DB.

        Each local memory is inserted as-is, preserving its existing
        ``id``, ``thread_id``, timestamps, and metadata.

        After the batched upserts complete, per-(user_id, thread_id) turn
        counts are accumulated and the in-process processor's auto-trigger
        runs. The trigger is best-effort: any failure (counter container
        missing, threshold-crossing logic raising, processor error) is
        logged and swallowed so writes never fail because of background
        processing.

        ``local_memory`` is **not** cleared on success — repeat calls
        re-upsert the same documents (idempotent on ``id``, but consumes
        RU and re-emits change-feed events). Auto-trigger uses a tracked
        per-(user, thread) delta so repeated pushes do not re-fire
        extraction on already-pushed turns. Call :meth:`clear_local`
        explicitly if you want strict flush-and-reset semantics.

        Parameters
        ----------
        batch_size : int
            Number of records per batch (for error isolation). All batches
            run sequentially. Defaults to 25.
        """
        self._require_cosmos()
        if batch_size <= 0:
            raise ValueError("batch_size must be greater than 0")
        logger.info(
            "push_to_cosmos count=%d batch_size=%d",
            len(self.local_memory),
            batch_size,
        )
        records = [MemoryRecord.from_cosmos_dict(dict(m)) for m in self.local_memory]
        for start in range(0, len(records), batch_size):
            batch = records[start : start + batch_size]
            bodies = [r.to_cosmos_dict() for r in batch]

            # Batch-embed non-turn memories that don't already carry a
            # vector — one /embeddings POST per Cosmos batch instead of
            # one per record.
            to_embed_idx: list[int] = []
            to_embed_text: list[str] = []
            for i, body in enumerate(bodies):
                if body.get("type") != "turn" and body.get("content") and not body.get("embedding"):
                    to_embed_idx.append(i)
                    to_embed_text.append(body["content"])
            if to_embed_text:
                try:
                    vectors = self._embeddings_client.generate_batch(to_embed_text)
                    for i, vec in zip(to_embed_idx, vectors):
                        bodies[i]["embedding"] = vec
                        # Persist the embedding back to local_memory so a
                        # repeat push_to_cosmos() doesn't re-embed.
                        self.local_memory[start + i]["embedding"] = vec
                except Exception as exc:  # noqa: BLE001
                    logger.warning(
                        "push_to_cosmos: batch embedding generation failed (%s); "
                        "proceeding without embeddings for %d records",
                        exc,
                        len(to_embed_text),
                    )

            for record, body in zip(batch, bodies):
                try:
                    self._container_client.upsert_item(body=body)
                except Exception as exc:
                    raise CosmosOperationError(f"Upsert failed for record {record.id}: {exc}") from exc
        logger.info("Upserted batch of %d records", len(records))

        turn_counts = self._unflushed_turn_counts
        self._unflushed_turn_counts = {}

        try:
            self._maybe_auto_trigger(turn_counts)
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("Auto-trigger after push_to_cosmos failed: %s", exc)

    def get_memories(
        self,
        memory_id: Optional[str] = None,
        user_id: Optional[str] = None,
        thread_id: Optional[str] = None,
        role: Optional[str] = None,
        memory_type: Optional[str] = None,
        recent_k: Optional[int] = None,
        tags: Optional[list[str]] = None,
        any_tags: Optional[list[str]] = None,
        exclude_tags: Optional[list[str]] = None,
        include_superseded: bool = False,
        min_salience: Optional[float] = None,
        min_confidence: Optional[float] = None,
    ) -> list[dict[str, Any]]:
        """Retrieve memories from Cosmos DB with optional filters.

        Args:
            memory_id: Filter by memory id.
            user_id: Filter by user id.
            thread_id: Filter by thread id.
            role: Filter by role.
            memory_type: Filter by type (raw, summary, fact, etc.).
            recent_k: If specified, return only the *k* most recent documents
                (ordered by ``_ts`` descending, then reversed to chronological).
            tags: AND filter — all specified tags must be present.
            any_tags: OR filter — any of the specified tags must be present.
            exclude_tags: NOT filter — none of these tags may be present.
            include_superseded: If False (default), exclude superseded memories.
            min_salience: Post-filter: only return memories with salience >= this value.
            min_confidence: Cosmos-side filter — only return memories with
                ``confidence >= min_confidence``. Memories without a confidence
                value (e.g. ``turn`` records) are excluded when this is set.
        """
        self._require_cosmos()
        logger.debug(
            "get_memories filters: memory_id=%s user_id=%s thread_id=%s role=%s type=%s recent_k=%s",
            memory_id,
            user_id,
            thread_id,
            role,
            memory_type,
            recent_k,
        )

        qb = _build_memory_query_builder(
            memory_id=memory_id,
            user_id=user_id,
            thread_id=thread_id,
            role=role,
            memory_type=memory_type,
            min_confidence=min_confidence,
        )

        # Tag filters
        if tags:
            for i, tag in enumerate(tags):
                qb.add_array_contains("c.tags", f"@tag_{i}", tag)
        if any_tags:
            qb.add_array_contains_any("c.tags", "@any_tag_", any_tags)
        if exclude_tags:
            for i, tag in enumerate(exclude_tags):
                qb.add_not_array_contains("c.tags", f"@exc_tag_{i}", tag)
        if not include_superseded:
            qb.add_is_null_or_undefined("c.superseded_by")

        where = qb.build_where()
        parameters = qb.get_parameters()

        if recent_k is not None:
            parameters.append({"name": "@recent_k", "value": recent_k})
            query = f"SELECT TOP @recent_k * FROM c{where} ORDER BY c._ts DESC"
        else:
            query = f"SELECT * FROM c{where}"

        logger.debug("get_memories query: %s", query)

        try:
            items = list(
                self._container_client.query_items(
                    query=query,
                    parameters=parameters or None,
                    enable_cross_partition_query=True,
                )
            )
        except Exception as exc:
            raise CosmosOperationError(f"get_memories query failed: {exc}") from exc

        if recent_k is not None:
            items.reverse()

        # Post-filter by salience
        if min_salience is not None:
            items = [i for i in items if (i.get("salience") or 0.0) >= min_salience]

        if not items:
            logger.warning("get_memories returned empty results")
        return items

    def update_cosmos(
        self,
        memory_id: str,
        content: Optional[str] = None,
        role: Optional[str] = None,
        memory_type: Optional[str] = None,
        metadata: Optional[dict[str, Any]] = None,
    ) -> None:
        """Update a memory in Cosmos DB.

        Raises
        ------
        MemoryNotFoundError
            If no document with *memory_id* exists.
        CosmosOperationError
            If the underlying Cosmos DB operation fails.
        """
        self._require_cosmos()

        try:
            results = list(
                self._container_client.query_items(
                    query="SELECT * FROM c WHERE c.id = @id",
                    parameters=[{"name": "@id", "value": memory_id}],
                    enable_cross_partition_query=True,
                )
            )
        except Exception as exc:
            raise CosmosOperationError(f"update query failed: {exc}") from exc

        if not results:
            raise MemoryNotFoundError(memory_id=memory_id)

        doc = results[0]
        if content is not None:
            doc["content"] = content
        if role is not None:
            doc["role"] = role
        if memory_type is not None:
            doc["type"] = memory_type
        if metadata is not None:
            doc["metadata"] = metadata
        doc["updated_at"] = datetime.now(timezone.utc).isoformat()

        try:
            self._container_client.replace_item(item=doc["id"], body=doc)
        except Exception as exc:
            raise CosmosOperationError(f"update replace failed for {memory_id}: {exc}") from exc

        logger.info("Updated record %s", memory_id)

    def delete_cosmos(self, memory_id: str, thread_id: str, user_id: str) -> None:
        """Delete a memory from Cosmos DB.

        Raises
        ------
        MemoryNotFoundError
            If no matching document is found.
        CosmosOperationError
            If the underlying Cosmos DB operation fails.
        """
        self._require_cosmos()

        try:
            results = list(
                self._container_client.query_items(
                    query=(
                        "SELECT TOP 1 c.id FROM c WHERE c.id = @id "
                        "AND c.thread_id = @thread_id AND c.user_id = @user_id"
                    ),
                    parameters=[
                        {"name": "@id", "value": memory_id},
                        {"name": "@thread_id", "value": thread_id},
                        {"name": "@user_id", "value": user_id},
                    ],
                    enable_cross_partition_query=True,
                )
            )
        except Exception as exc:
            raise CosmosOperationError(f"delete lookup failed: {exc}") from exc

        if not results:
            raise MemoryNotFoundError(memory_id=memory_id, user_id=user_id, thread_id=thread_id)

        try:
            self._container_client.delete_item(item=memory_id, partition_key=[user_id, thread_id])
        except Exception as exc:
            raise CosmosOperationError(f"delete failed for {memory_id}: {exc}") from exc

        logger.info("Deleted record %s", memory_id)

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    def search_cosmos(
        self,
        search_terms: str,
        memory_id: Optional[str] = None,
        user_id: Optional[str] = None,
        role: Optional[str] = None,
        memory_type: Optional[str] = None,
        thread_id: Optional[str] = None,
        hybrid_search: bool = False,
        top_k: int = 5,
        tags: Optional[list[str]] = None,
        any_tags: Optional[list[str]] = None,
        exclude_tags: Optional[list[str]] = None,
        include_superseded: bool = False,
        min_salience: Optional[float] = None,
        min_confidence: Optional[float] = None,
    ) -> list[dict[str, Any]]:
        """Search memories in Cosmos DB using vector similarity.

        1. Embeds *search_terms* via the configured embedding model.
        2. Runs a vector similarity query against the Cosmos DB container.
        3. Optionally filters by the remaining keyword parameters.
        4. Returns up to *top_k* results ordered by similarity.

        Args:
            min_confidence: Cosmos-side filter — only return memories with
                ``confidence >= min_confidence``. Memories without confidence
                (e.g. raw ``turn`` records) are excluded when set.
        """
        self._require_cosmos()
        _validate_hybrid_search(hybrid_search, search_terms)
        if not search_terms or not search_terms.strip():
            raise ValidationError("search_terms must be a non-empty string")

        logger.info(
            "search_cosmos terms_len=%d top_k=%d hybrid_search=%s",
            len(search_terms),
            top_k,
            hybrid_search,
        )
        logger.debug(
            "search_cosmos search_terms=%s",
            search_terms[:50] + "..." if len(search_terms) > 50 else search_terms,
        )

        query_vector = self._embeddings_client.generate(search_terms)

        qb = _build_memory_query_builder(
            user_id=user_id,
            role=role,
            memory_type=memory_type,
            thread_id=thread_id,
            min_confidence=min_confidence,
        )

        # Tag filters
        if tags:
            for i, tag in enumerate(tags):
                qb.add_array_contains("c.tags", f"@tag_{i}", tag)
        if any_tags:
            qb.add_array_contains_any("c.tags", "@any_tag_", any_tags)
        if exclude_tags:
            for i, tag in enumerate(exclude_tags):
                qb.add_not_array_contains("c.tags", f"@exc_tag_{i}", tag)
        if not include_superseded:
            qb.add_is_null_or_undefined("c.superseded_by")

        where = qb.build_where()
        parameters = qb.get_parameters()

        order_by = "ORDER BY VectorDistance(c.embedding, @embedding)"
        if hybrid_search:
            order_by = (
                "ORDER BY RANK RRF(VectorDistance(c.embedding, @embedding), FullTextScore(c.content, @key_terms))"
            )

        query = (
            f"SELECT TOP @top_k c.id, c.user_id, c.thread_id, c.role, c.type, c.content, "
            f"c.metadata, c.created_at, c.tags, c.salience, c.confidence, c.superseded_by "
            f"FROM c{where} "
            f"{order_by}"
        )

        parameters.extend(
            [
                {"name": "@top_k", "value": top_k},
                {"name": "@embedding", "value": query_vector},
            ]
        )
        if hybrid_search:
            parameters.append({"name": "@key_terms", "value": search_terms or ""})

        logger.debug("search_cosmos query: %s", query)

        try:
            results = list(
                self._container_client.query_items(
                    query=query,
                    parameters=parameters,
                    enable_cross_partition_query=True,
                )
            )
        except Exception as exc:
            raise CosmosOperationError(f"vector_search failed: {exc}") from exc

        # Post-filter by memory_id (not supported directly by vector search)
        if memory_id is not None:
            results = [r for r in results if r.get("id") == memory_id]
        # Post-filter by salience
        if min_salience is not None:
            results = [r for r in results if (r.get("salience") or 0.0) >= min_salience]
        if not results:
            logger.warning(
                "search_cosmos returned empty results (terms_len=%d)",
                len(search_terms),
            )
        return results

    def get_thread(
        self,
        thread_id: str,
        user_id: Optional[str] = None,
        memory_type: Optional[str] = None,
        recent_k: Optional[int] = None,
        tags: Optional[list[str]] = None,
        exclude_tags: Optional[list[str]] = None,
        include_superseded: bool = False,
    ) -> list[dict[str, Any]]:
        """Retrieve an entire thread from Cosmos DB.

        Returns memories sorted in chronological order (oldest first).
        """
        self._require_cosmos()

        qb = _QueryBuilder()
        qb.add_filter("c.thread_id", "@thread_id", thread_id)
        qb.add_filter("c.user_id", "@user_id", user_id)
        qb.add_filter("c.type", "@memory_type", memory_type)

        # Tag filters
        if tags:
            for i, tag in enumerate(tags):
                qb.add_array_contains("c.tags", f"@tag_{i}", tag)
        if exclude_tags:
            for i, tag in enumerate(exclude_tags):
                qb.add_not_array_contains("c.tags", f"@exc_tag_{i}", tag)
        if not include_superseded:
            qb.add_is_null_or_undefined("c.superseded_by")

        where = qb.build_where()
        parameters = qb.get_parameters()

        query = f"SELECT * FROM c{where} ORDER BY c.created_at DESC"
        logger.debug("get_thread query: %s", query)

        try:
            items = list(
                self._container_client.query_items(
                    query=query,
                    parameters=parameters,
                    enable_cross_partition_query=True,
                )
            )
        except Exception as exc:
            raise CosmosOperationError(f"get_thread query failed: {exc}") from exc

        if recent_k is not None:
            items = items[:recent_k]
        items.reverse()
        return items

    def get_user_summary(self, user_id: str) -> list[dict[str, Any]]:
        """Retrieve user summary documents from Cosmos DB, newest first."""
        self._require_cosmos()

        query = (
            "SELECT c.id, c.user_id, c.thread_id, c.role, c.type, "
            "c.content, c.metadata, c.created_at "
            "FROM c WHERE c.user_id = @user_id AND c.type = 'user_summary' "
            "ORDER BY c.created_at DESC"
        )
        parameters = [{"name": "@user_id", "value": user_id}]
        logger.debug("get_user_summary query: %s", query)

        try:
            return list(
                self._container_client.query_items(
                    query=query,
                    parameters=parameters,
                    enable_cross_partition_query=True,
                )
            )
        except Exception as exc:
            raise CosmosOperationError(f"get_user_summary query failed: {exc}") from exc

    # ------------------------------------------------------------------
    # Tag operations
    # ------------------------------------------------------------------

    def add_tags(self, memory_id: str, user_id: str, thread_id: str, tags: list[str]) -> None:
        """Add tags to an existing memory document."""
        self._require_cosmos()
        doc = self._container_client.read_item(item=memory_id, partition_key=[user_id, thread_id])
        existing_tags = set(doc.get("tags", []))
        existing_tags.update(t.strip().lower() for t in tags)
        doc["tags"] = sorted(existing_tags)
        doc["updated_at"] = datetime.now(timezone.utc).isoformat()
        self._container_client.replace_item(item=memory_id, body=doc)

    def remove_tags(self, memory_id: str, user_id: str, thread_id: str, tags: list[str]) -> None:
        """Remove tags from an existing memory document."""
        self._require_cosmos()
        doc = self._container_client.read_item(item=memory_id, partition_key=[user_id, thread_id])
        tags_to_remove = {t.strip().lower() for t in tags}
        doc["tags"] = sorted(set(doc.get("tags", [])) - tags_to_remove)
        doc["updated_at"] = datetime.now(timezone.utc).isoformat()
        self._container_client.replace_item(item=memory_id, body=doc)

    # ------------------------------------------------------------------
    # Procedural and episodic memory retrieval
    # ------------------------------------------------------------------

    def get_procedural_memories(
        self,
        user_id: str,
        priority: Optional[str] = None,
        category: Optional[str] = None,
        min_salience: Optional[float] = None,
        include_superseded: bool = False,
    ) -> list[dict[str, Any]]:
        """Retrieve active procedural memories for a user (thread_id='__procedural__')."""
        self._require_cosmos()
        qb = _QueryBuilder()
        qb.add_filter("c.user_id", "@user_id", user_id)
        qb.add_filter("c.thread_id", "@thread_id", "__procedural__")
        qb.add_filter("c.type", "@type", "procedural")
        if not include_superseded:
            qb.add_is_null_or_undefined("c.superseded_by")

        where = qb.build_where()
        parameters = qb.get_parameters()
        query = f"SELECT * FROM c{where} ORDER BY c.created_at DESC"

        try:
            items = list(
                self._container_client.query_items(
                    query=query,
                    parameters=parameters,
                    enable_cross_partition_query=True,
                )
            )
        except Exception as exc:
            raise CosmosOperationError(f"get_procedural_memories query failed: {exc}") from exc

        # Post-filter by salience
        if min_salience is not None:
            items = [i for i in items if (i.get("salience") or 0.0) >= min_salience]
        # Post-filter by priority/category from metadata
        if priority is not None:
            items = [i for i in items if i.get("metadata", {}).get("priority") == priority]
        if category is not None:
            items = [i for i in items if i.get("metadata", {}).get("category") == category]

        return items

    def search_episodic_memories(
        self,
        user_id: str,
        search_terms: str,
        top_k: int = 5,
        min_salience: Optional[float] = None,
        include_superseded: bool = False,
    ) -> list[dict[str, Any]]:
        """Semantic search across episodic memories for a user."""
        return self.search_cosmos(
            search_terms=search_terms,
            user_id=user_id,
            memory_type="episodic",
            top_k=top_k,
            min_salience=min_salience,
            include_superseded=include_superseded,
        )

    # ------------------------------------------------------------------
    # Context builders
    # ------------------------------------------------------------------

    def build_procedural_context(self, user_id: str) -> str:
        """Build formatted text for system prompt injection."""
        memories = self.get_procedural_memories(user_id)
        if not memories:
            return ""
        lines = ["## Learned User Preferences"]
        for m in memories:
            priority = m.get("metadata", {}).get("priority", "should")
            lines.append(f"- {m['content']} [{priority}]")
        return "\n".join(lines)

    def build_episodic_context(self, user_id: str, query: str, top_k: int = 3) -> str:
        """Build formatted context of relevant past experiences."""
        memories = self.search_episodic_memories(user_id, query, top_k=top_k)
        if not memories:
            return ""
        lines = ["## Relevant Past Experiences"]
        for i, m in enumerate(memories, 1):
            domain = m.get("metadata", {}).get("domain", "general")
            valence = m.get("metadata", {}).get("outcome_valence", "neutral")
            lines.append(f"{i}. [{domain}] {m['content']} ({valence})")
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Processing (LLM pipeline)
    # ------------------------------------------------------------------

    def extract_memories(
        self,
        user_id: str,
        thread_id: str,
        recent_k: Optional[int] = None,
    ) -> dict[str, int]:
        """Extract facts, procedural, and episodic memories from a thread."""
        self._require_cosmos()
        return self._pipeline.extract_memories(user_id, thread_id, recent_k)

    def generate_thread_summary(
        self,
        user_id: str,
        thread_id: str,
        recent_k: Optional[int] = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Generate a thread summary."""
        self._require_cosmos()
        return self._pipeline.generate_thread_summary(user_id, thread_id, recent_k)

    def generate_user_summary(
        self,
        user_id: str,
        thread_ids: Optional[list[str]] = None,
        recent_k: Optional[int] = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Generate a cross-thread user summary."""
        self._require_cosmos()
        return self._pipeline.generate_user_summary(user_id, thread_ids, recent_k)

    def reconcile(
        self,
        user_id: str,
        n: Optional[int] = None,
    ) -> dict[str, int]:
        """Reconcile a user's facts via the contradiction-aware dedup pass.

        ``n`` defaults to the ``DEDUP_POOL_SIZE`` env var (via
        :func:`agent_memory_toolkit.thresholds.get_dedup_pool_size`) so
        explicit calls honour the same operator knob the auto-trigger
        path uses. Pass an integer to override.
        """
        from .thresholds import get_dedup_pool_size

        self._require_cosmos()
        pool = n if n is not None else get_dedup_pool_size()
        return self._pipeline.reconcile_memories(user_id, pool)

    # ------------------------------------------------------------------
    # Processor delegation
    # ------------------------------------------------------------------

    def process_now(
        self,
        *,
        user_id: str,
        thread_id: str,
    ) -> "ProcessThreadResult":
        """Force the processor to run summarize/extract/dedup right now.

        Unlike the auto-trigger that fires from :meth:`push_to_cosmos` once
        a per-turn threshold is crossed, this method runs unconditionally.
        Useful for end-of-conversation cleanup, eval/test determinism, and
        agent handoff when you need a fresh summary regardless of cadence.

        For :class:`InProcessProcessor` this runs the summarize/extract/dedup
        pipeline inline. For :class:`DurableFunctionProcessor` this is a
        debug-logged no-op (the function app handles processing via
        Cosmos DB Change Feed). With per-turn extraction enabled (the
        default ``FACT_EXTRACTION_EVERY_N=1``) calling this after every
        turn is redundant — facts have already been extracted.
        """
        from .processors import ProcessThreadResult  # local import for forward ref

        self._require_cosmos()
        processor = self._get_processor()

        try:
            turns = self.get_thread(thread_id=thread_id, user_id=user_id, memory_type="turn")
        except Exception:  # pragma: no cover - best-effort load
            turns = []

        result = processor.process_thread(
            user_id=user_id,
            thread_id=thread_id,
            turns=turns,
        )
        assert isinstance(result, ProcessThreadResult)
        return result

    def process_now_and_wait(
        self,
        *,
        user_id: str,
        thread_id: str,
        timeout: float = 30.0,
    ) -> bool:
        """Force processing and block until a thread summary exists (or ``timeout``).

        For :class:`InProcessProcessor` this is just :meth:`process_now`
        followed by ``True`` (the in-process pipeline is synchronous). For
        :class:`DurableFunctionProcessor` this polls
        ``get_memories(memory_type="summary", ...)`` every 0.5s until a
        summary appears or the timeout expires.

        Polling uses ``get_memories`` (filter-only, no embeddings) so this
        path does *not* require AI Foundry / embeddings to be configured —
        the SDK can act as a pure Cosmos writer while the Function App owns
        all LLM calls.

        Returns ``True`` on success, ``False`` on timeout.
        """
        import time as _time

        self._require_cosmos()
        processor = self._get_processor()

        # In-process: pipeline is synchronous, so process_now() already finished work.
        if isinstance(processor, InProcessProcessor):
            self.process_now(user_id=user_id, thread_id=thread_id)
            return True

        # Durable: trigger the no-op flush (debug log) then poll Cosmos.
        self.process_now(user_id=user_id, thread_id=thread_id)

        deadline = _time.monotonic() + timeout
        poll_interval = 0.5
        while _time.monotonic() < deadline:
            try:
                results = self.get_memories(
                    user_id=user_id,
                    thread_id=thread_id,
                    memory_type="summary",
                    recent_k=1,
                )
            except Exception:
                results = []
            if results:
                return True
            _time.sleep(poll_interval)
        return False
