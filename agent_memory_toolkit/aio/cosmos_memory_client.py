"""Thin asynchronous Cosmos memory client orchestrator."""

from __future__ import annotations

import asyncio
from datetime import datetime
from typing import TYPE_CHECKING, Any, Optional

from agent_memory_toolkit._base import _BaseMemoryClient
from agent_memory_toolkit._utils import (
    _build_container_kwargs,
    _container_policies,
    _cosmos_container_offer_throughput,
    _resolve_cosmos_provisioning_autoscale_max_ru,
    _resolve_cosmos_throughput_mode,
    _resolve_distance_function,
    _resolve_embedding_data_type,
    _resolve_full_text_language,
    _validate_connection,
)
from agent_memory_toolkit.aio.auto_trigger import maybe_trigger_steps
from agent_memory_toolkit.aio.chat import AsyncChatClient
from agent_memory_toolkit.aio.embeddings import AsyncEmbeddingsClient
from agent_memory_toolkit.aio.processors import AsyncInProcessProcessor, AsyncMemoryProcessor
from agent_memory_toolkit.aio.services.pipeline import AsyncPipelineService
from agent_memory_toolkit.aio.store import AsyncMemoryStore
from agent_memory_toolkit.exceptions import CosmosNotConnectedError, CosmosOperationError
from agent_memory_toolkit.logging import get_logger
from agent_memory_toolkit.thresholds import DEFAULT_TTL_BY_TYPE

if TYPE_CHECKING:  # pragma: no cover - typing-only import
    from agent_memory_toolkit.processors.base import ProcessThreadResult, UserSummaryResult  # noqa: F401

logger = get_logger(__name__)


def _log_auto_trigger_task_failure(task: "asyncio.Task[Any]") -> None:
    if not task.cancelled() and (exc := task.exception()) is not None:
        logger.warning("Background auto-trigger task failed: %r", exc)


class AsyncCosmosMemoryClient(_BaseMemoryClient):
    """Async variant of :class:`agent_memory_toolkit.CosmosMemoryClient`."""

    def __init__(
        self,
        cosmos_endpoint: Optional[str] = None,
        cosmos_credential: Optional[Any] = None,
        cosmos_key: Optional[str] = None,
        cosmos_database: Optional[str] = None,
        cosmos_container: Optional[str] = None,
        cosmos_turns_container: Optional[str] = None,
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
        processor: Optional[AsyncMemoryProcessor] = None,
    ) -> None:
        self._init_base_config(
            cosmos_endpoint=cosmos_endpoint,
            cosmos_credential=cosmos_credential,
            cosmos_key=cosmos_key,
            cosmos_database=cosmos_database,
            cosmos_container=cosmos_container,
            cosmos_turns_container=cosmos_turns_container,
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
            default_credential_module="azure.identity.aio",
        )
        self._background_tasks: set[asyncio.Task[Any]] = set()
        self._pipeline_init_error: Exception | None = None
        self._embeddings_client = AsyncEmbeddingsClient(
            endpoint=self._ai_foundry_endpoint,
            credential=self._ai_foundry_credential,
            api_key=self._ai_foundry_api_key,
            model=self._embedding_deployment_name,
            dimensions=self._embedding_dimensions,
        )
        self._chat_client = AsyncChatClient(
            endpoint=self._ai_foundry_endpoint,
            credential=self._ai_foundry_credential,
            api_key=self._ai_foundry_api_key,
            model=self._chat_deployment_name,
        )
        self._pipeline: Optional[AsyncPipelineService] = None
        self._processor: Optional[AsyncMemoryProcessor] = processor
        self._processor_explicit = processor is not None
        logger.info("AsyncCosmosMemoryClient initialized")

    async def __aenter__(self) -> "AsyncCosmosMemoryClient":
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        await self.close()

    async def close(self) -> None:
        """Close async clients and owned credentials."""
        pending = list(self._background_tasks)
        for task in pending:
            task.cancel()
        if pending:
            await asyncio.wait(pending, timeout=5.0)
        if self._cosmos_client is not None:
            await self._cosmos_client.close()
            self._cosmos_client = None
            self._container_client = None
            self._turns_container_client = None
            self._counter_container_client = None
            self._store = None
            self._pipeline = None
        if self._processor is not None and not self._processor_explicit:
            await self._close_maybe_async(self._processor)
            self._processor = None
        await self._embeddings_client.close()
        await self._close_maybe_async(self._chat_client)
        for owns, cred in (
            (self._owns_cosmos_credential, self._cosmos_credential),
            (self._owns_ai_foundry_credential, self._ai_foundry_credential),
        ):
            if owns and cred is not None:
                await self._close_maybe_async(cred)
        logger.info("AsyncCosmosMemoryClient closed")

    async def _close_maybe_async(self, closeable: Any) -> None:
        if closeable is None:
            return
        close = getattr(closeable, "close", None)
        if callable(close):
            try:
                result = close()
                if asyncio.iscoroutine(result):
                    await result
            except Exception:
                pass

    async def connect_cosmos(
        self,
        endpoint: Optional[str] = None,
        credential: Optional[Any] = None,
        key: Optional[str] = None,
        database: Optional[str] = None,
        container: Optional[str] = None,
        turns_container: Optional[str] = None,
    ) -> None:
        """Establish an async connection to a Cosmos DB container.

        If *turns_container* is provided, it overrides the constructor's
        ``cosmos_turns_container`` setting for this connection.
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
        _validate_connection(
            self._cosmos_endpoint,
            self._cosmos_credential,
            self._cosmos_database,
            self._cosmos_container,
        )
        try:
            from azure.cosmos.aio import CosmosClient

            await self._drain_cosmos_client()
            client = CosmosClient(self._cosmos_endpoint, credential=self._cosmos_credential)
            db = client.get_database_client(self._cosmos_database)
            self._cosmos_client = client
            self._container_client = db.get_container_client(self._cosmos_container)
            if self._cosmos_turns_container:
                self._turns_container_client = db.get_container_client(self._cosmos_turns_container)
                logger.info(
                    "Async connected turns container: %s/%s",
                    self._cosmos_database,
                    self._cosmos_turns_container,
                )
            else:
                self._turns_container_client = None
            self._init_services()
        except Exception as exc:
            raise CosmosOperationError(f"Failed to connect to Cosmos DB (async): {exc}") from exc
        logger.info("Async connected to Cosmos DB %s/%s", self._cosmos_database, self._cosmos_container)

    async def create_memory_store(
        self,
        database: Optional[str] = None,
        container: Optional[str] = None,
        turns_container: Optional[str] = None,
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
            from azure.cosmos import PartitionKey, ThroughputProperties
            from azure.cosmos.aio import CosmosClient

            await self._drain_cosmos_client()
            client = CosmosClient(self._cosmos_endpoint, credential=self._cosmos_credential)
            db = await client.create_database_if_not_exists(id=self._cosmos_database)
            partition_key = PartitionKey(path=["/user_id", "/thread_id"], kind="MultiHash")
            offer = _cosmos_container_offer_throughput(
                throughput_mode=self._cosmos_throughput_mode,
                autoscale_max_ru=self._cosmos_autoscale_max_ru,
                throughput_properties_cls=ThroughputProperties,
            )
            vec_policy, idx_policy, ft_policy = _container_policies(
                embedding_dimensions=embedding_dimensions or self._embedding_dimensions or 1536,
                embedding_data_type=_resolve_embedding_data_type(embedding_data_type),
                distance_function=_resolve_distance_function(distance_function),
                full_text_language=_resolve_full_text_language(full_text_language),
            )
            self._container_client = await db.create_container_if_not_exists(
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
            await db.create_container_if_not_exists(
                **_build_container_kwargs(
                    container_id=self._cosmos_counter_container,
                    partition_key=partition_key,
                    offer_throughput=offer,
                )
            )
            await db.create_container_if_not_exists(
                **_build_container_kwargs(
                    container_id=self._cosmos_lease_container,
                    partition_key=PartitionKey(path="/id"),
                    offer_throughput=offer,
                )
            )
            self._cosmos_client = client
            if self._cosmos_turns_container:
                self._turns_container_client = await db.create_container_if_not_exists(
                    **_build_container_kwargs(
                        container_id=self._cosmos_turns_container,
                        partition_key=partition_key,
                        offer_throughput=offer,
                        default_ttl=DEFAULT_TTL_BY_TYPE["turn"],
                        indexing_policy=idx_policy,
                        vector_embedding_policy=vec_policy,
                        full_text_policy=ft_policy,
                    )
                )
                logger.info(
                    "Created turns container: %s/%s",
                    self._cosmos_database,
                    self._cosmos_turns_container,
                )
            else:
                self._turns_container_client = None
            self._init_services()
        except Exception as exc:
            raise CosmosOperationError(f"Failed to create memory store (async): {exc}") from exc
        logger.info("Async created memory store %s/%s", self._cosmos_database, self._cosmos_container)

    def _init_services(self) -> None:
        self._store = AsyncMemoryStore(
            self._container_client,
            embeddings_client=self._embeddings_client,
            turns_container=self._turns_container_client,
        )
        self._init_pipeline()
        if not self._processor_explicit:
            self._processor = None

    def _init_pipeline(self) -> None:
        """Initialize the AsyncPipelineService over the async store/clients."""
        self._pipeline = AsyncPipelineService(
            self._store,
            self._chat_client,
            self._embeddings_client,
            cosmos_turns_container=self._turns_container_client,
        )
        self._pipeline_init_error = None

    async def _drain_cosmos_client(self) -> None:
        prior = self._cosmos_client
        if prior is not None:
            close = getattr(prior, "close", None)
            if callable(close):
                await close()
        self._cosmos_client = None
        self._container_client = None
        self._turns_container_client = None
        self._counter_container_client = None
        self._store = None
        self._pipeline = None
        if not self._processor_explicit:
            self._processor = None

    def _require_pipeline(self) -> None:
        if self._pipeline is None:
            if self._pipeline_init_error is not None:
                raise CosmosNotConnectedError(
                    f"Processing pipeline failed to initialize "
                    f"({type(self._pipeline_init_error).__name__}: {self._pipeline_init_error})."
                ) from self._pipeline_init_error
            raise CosmosNotConnectedError("Processing pipeline requires Cosmos DB connection.")

    async def _require_cosmos(self) -> None:
        _BaseMemoryClient._require_cosmos(self)

    def _get_store(self) -> AsyncMemoryStore:
        _BaseMemoryClient._require_cosmos(self)
        if (
            self._store is None
            or self._store.container is not self._container_client
            or self._store._embeddings_client is not self._embeddings_client
            or self._store._turns_container is not self._turns_container_client
        ):
            self._store = AsyncMemoryStore(
                self._container_client,
                embeddings_client=self._embeddings_client,
                turns_container=self._turns_container_client,
            )
        return self._store

    def _get_pipeline(self) -> AsyncPipelineService:
        if self._pipeline is None:
            _BaseMemoryClient._require_cosmos(self)
            self._init_pipeline()
        self._require_pipeline()
        return self._pipeline

    def _get_processor(self) -> AsyncMemoryProcessor:
        if self._processor is None:
            self._processor = AsyncInProcessProcessor(pipeline=self._get_pipeline())
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

    async def _maybe_auto_trigger(self, turn_counts: dict[tuple[str, str], int]) -> None:
        if not turn_counts:
            return
        await maybe_trigger_steps(self._get_processor(), self._get_counter_container(), turn_counts)

    def _container_for_type(self, memory_type: str) -> Any:
        """Return the appropriate container client based on memory type.

        When ``cosmos_turns_container`` is configured, turn-type documents
        are routed to the dedicated turns container. All other memory types
        (fact, summary, episodic, procedural, user_summary) use the main
        memories container.
        """
        if memory_type == "turn" and self._turns_container_client is not None:
            return self._turns_container_client
        return self._container_client

    def _container_for_query(self, memory_types: Optional[list[str]] = None) -> Any:
        """Return a single container for a read query.

        This helper is intended for callers that target a known single
        container (e.g., ``get_memories`` for derived memories, ``search_cosmos``
        for vector search). It does NOT merge across containers.

        When a dedicated turns container is configured:

        * turn-only queries → turns container
        * non-turn / unspecified / mixed queries → main memories container

        Callers that need complete results across both containers (e.g.,
        ``get_thread``) should use ``_containers_for_query()`` instead.

        Note: ``get_memories(memory_types=None)`` returns derived memories
        only (facts, episodic, procedural, summaries). Raw turns are
        retrieved via ``get_thread()``.
        """
        if not memory_types:
            return self._container_client

        has_turn = any(t == "turn" for t in memory_types)
        has_not_turn = any(t != "turn" for t in memory_types)

        if self._turns_container_client is not None:
            if has_turn and not has_not_turn:
                return self._turns_container_client
            if not has_turn:
                return self._container_client
            # Mixed: fall back to main (turns in separate container won't
            # appear but this preserves backward-compat for callers that
            # don't handle multi-container merging)
            return self._container_client
        return self._container_client

    def _containers_for_query(self, memory_types: Optional[list[str]] = None) -> list[Any]:
        """Return candidate containers for a read query.

        Unlike :meth:`_container_for_query` which returns a single container,
        this returns all containers that should be queried to get complete
        results, enabling callers to merge across containers.
        """
        if self._turns_container_client is None:
            return [self._container_client]

        if not memory_types:
            # Query spans both the main memories container and the
            # turns container when memory_types is None.
            return [self._container_client, self._turns_container_client]

        has_turn = any(t == "turn" for t in memory_types)
        has_not_turn = any(t != "turn" for t in memory_types)

        if has_turn and has_not_turn:
            return [self._container_client, self._turns_container_client]

        if has_turn:
            return [self._turns_container_client]
        return [self._container_client]

    async def add_cosmos(
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
        return await self._get_store().add(
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
        )

    async def push_to_cosmos(self, batch_size: int = 25) -> None:
        """Insert all local memories into Cosmos DB and schedule processing."""
        await self._get_store().push(self.local_memory, batch_size=batch_size)
        turn_counts, self._unflushed_turn_counts = self._unflushed_turn_counts, {}
        if turn_counts:
            task = asyncio.create_task(self._maybe_auto_trigger(turn_counts))
            self._background_tasks.add(task)
            task.add_done_callback(self._background_tasks.discard)
            task.add_done_callback(_log_auto_trigger_task_failure)

    async def get_memories(
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
        return await self._get_store().get_memories(
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

    async def update_cosmos(
        self,
        memory_id: str,
        content: Optional[str] = None,
        role: Optional[str] = None,
        memory_type: Optional[str] = None,
        metadata: Optional[dict[str, Any]] = None,
    ) -> None:
        return await self._get_store().update(memory_id, content, role, memory_type, metadata)

    async def delete_cosmos(self, memory_id: str, thread_id: str, user_id: str) -> None:
        return await self._get_store().delete(memory_id=memory_id, thread_id=thread_id, user_id=user_id)

    async def search_cosmos(
        self,
        search_terms: str,
        memory_id: Optional[str] = None,
        user_id: Optional[str] = None,
        role: Optional[str] = None,
        memory_types: Optional[list[str]] = None,
        thread_id: Optional[str] = None,
        hybrid_search: bool = False,
        top_k: int = 5,
        tags_all: Optional[list[str]] = None,
        tags_any: Optional[list[str]] = None,
        exclude_tags: Optional[list[str]] = None,
        include_superseded: bool = False,
        min_salience: Optional[float] = None,
        min_confidence: Optional[float] = None,
        created_after: Optional[str | datetime] = None,
        created_before: Optional[str | datetime] = None,
    ) -> list[dict[str, Any]]:
        return await self._get_store().search(
            search_terms=search_terms,
            memory_id=memory_id,
            user_id=user_id,
            role=role,
            memory_types=memory_types,
            thread_id=thread_id,
            hybrid_search=hybrid_search,
            top_k=top_k,
            tags_all=tags_all,
            tags_any=tags_any,
            exclude_tags=exclude_tags,
            include_superseded=include_superseded,
            min_salience=min_salience,
            min_confidence=min_confidence,
            created_after=created_after,
            created_before=created_before,
        )

    async def get_thread(
        self,
        thread_id: str,
        user_id: Optional[str] = None,
        memory_types: Optional[list[str]] = None,
        recent_k: Optional[int] = None,
        tags_all: Optional[list[str]] = None,
        tags_any: Optional[list[str]] = None,
        exclude_tags: Optional[list[str]] = None,
        include_superseded: bool = False,
        created_after: Optional[str | datetime] = None,
        created_before: Optional[str | datetime] = None,
    ) -> list[dict[str, Any]]:
        return await self._get_store().get_thread(
            thread_id=thread_id,
            user_id=user_id,
            memory_types=memory_types,
            recent_k=recent_k,
            tags_all=tags_all,
            tags_any=tags_any,
            exclude_tags=exclude_tags,
            include_superseded=include_superseded,
            created_after=created_after,
            created_before=created_before,
        )

    async def get_user_summary(self, user_id: str) -> Optional[dict[str, Any]]:
        return await self._get_store().get_user_summary(user_id=user_id)

    async def list_tags(
        self,
        user_id: str,
        *,
        thread_id: Optional[str] = None,
        prefix: Optional[str] = None,
        include_sys: bool = False,
    ) -> list[str]:
        """Return sorted distinct tags for a user."""
        return await self._get_store().list_tags(
            user_id,
            thread_id=thread_id,
            prefix=prefix,
            include_sys=include_sys,
        )

    async def add_tags(self, memory_id: str, user_id: str, thread_id: str, tags: list[str]) -> None:
        return await self._get_store().add_tags(memory_id, user_id, thread_id, tags)

    async def remove_tags(self, memory_id: str, user_id: str, thread_id: str, tags: list[str]) -> None:
        return await self._get_store().remove_tags(memory_id, user_id, thread_id, tags)

    async def get_procedural_prompt(self, user_id: str) -> Optional[str]:
        return await self._get_store().get_procedural_prompt(user_id=user_id)

    async def get_procedural_history(self, user_id: str, limit: int = 10) -> list[dict[str, Any]]:
        return await self._get_store().get_procedural_history(user_id=user_id, limit=limit)

    async def get_procedural_memories(
        self,
        user_id: str,
        priority: Optional[str] = None,
        category: Optional[str] = None,
        min_salience: Optional[float] = None,
        include_superseded: bool = False,
    ) -> list[dict[str, Any]]:
        return await self._get_store().get_procedural_memories(
            user_id,
            priority,
            category,
            min_salience,
            include_superseded,
        )

    async def search_episodic_memories(
        self,
        user_id: str,
        search_terms: str,
        top_k: int = 5,
        min_salience: Optional[float] = None,
        include_superseded: bool = False,
    ) -> list[dict[str, Any]]:
        return await self._get_store().search_episodic(user_id, search_terms, top_k, min_salience, include_superseded)

    async def build_procedural_context(self, user_id: str) -> str:
        return await self._get_pipeline().build_procedural_context(user_id)

    async def build_episodic_context(self, user_id: str, query: str, top_k: int = 3) -> str:
        return await self._get_store().build_episodic_context(user_id, query, top_k)

    async def extract_memories(self, user_id: str, thread_id: str, recent_k: Optional[int] = None) -> dict[str, int]:
        return await self._get_pipeline().extract_memories(user_id, thread_id, recent_k)

    async def synthesize_procedural(self, user_id: str, *, force: bool = False) -> dict[str, Any]:
        processor = self._get_processor()
        if not isinstance(processor, AsyncInProcessProcessor):
            raise NotImplementedError(
                "Procedural synthesis runs automatically after reconcile in durable mode; "
                "manual invocation via the SDK is not supported when the Durable Function "
                "app is the active processor. Use get_procedural_prompt() to read the "
                "latest synthesized prompt."
            )
        return await processor.synthesize_procedural(user_id=user_id, force=force)

    async def generate_thread_summary(
        self,
        user_id: str,
        thread_id: str,
        recent_k: Optional[int] = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        return await self._get_pipeline().generate_thread_summary(user_id, thread_id, recent_k)

    async def generate_user_summary(
        self,
        user_id: str,
        thread_ids: Optional[list[str]] = None,
        recent_k: Optional[int] = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        return await self._get_pipeline().generate_user_summary(user_id, thread_ids, recent_k)

    async def reconcile(self, user_id: str, n: Optional[int] = None) -> dict[str, int]:
        from agent_memory_toolkit.thresholds import get_dedup_pool_size

        return await self._get_pipeline().reconcile_memories(user_id, n if n is not None else get_dedup_pool_size())

    async def process_now(self, *, user_id: str, thread_id: str) -> "ProcessThreadResult":
        _BaseMemoryClient._require_cosmos(self)
        turns = await self.get_thread(thread_id=thread_id, user_id=user_id, memory_types=["turn"]) or []
        return await self._get_processor().process_thread(user_id=user_id, thread_id=thread_id, turns=turns)

    async def process_now_and_wait(self, *, user_id: str, thread_id: str, timeout: float = 30.0) -> bool:
        _BaseMemoryClient._require_cosmos(self)
        processor = self._get_processor()
        turns = await self.get_thread(thread_id=thread_id, user_id=user_id, memory_types=["turn"]) or []
        await processor.process_thread(user_id=user_id, thread_id=thread_id, turns=turns)
        if isinstance(processor, AsyncInProcessProcessor):
            return True
        loop = asyncio.get_running_loop()
        deadline = loop.time() + timeout
        while loop.time() < deadline:
            if await self._summary_exists(user_id=user_id, thread_id=thread_id):
                return True
            await asyncio.sleep(0.5)
        return False

    async def _summary_exists(self, *, user_id: str, thread_id: str) -> bool:
        try:
            results = await self.get_memories(
                user_id=user_id,
                thread_id=thread_id,
                memory_types=["summary"],
                recent_k=1,
            )
        except Exception:
            return False
        return bool(results)
