"""CosmosMemoryClient: unified local and cloud agent memory management.

Consolidates the former ``AgentMemory`` orchestrator and ``CosmosMemoryStore``
into a single class that owns local CRUD, Cosmos DB connection/CRUD,
embedding-based search, and Azure Durable Functions processing.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Optional

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
    _resolve_embedding_dimensions,
    _validate_connection,
    _validate_hybrid_search,
)
from .embeddings import EmbeddingsClient
from .exceptions import (
    CosmosNotConnectedError,
    CosmosOperationError,
    MemoryNotFoundError,
    ValidationError,
)
from .models import MemoryRecord
from .processing import ProcessingClient

logger = logging.getLogger(__name__)


class CosmosMemoryClient:
    """Manages agent memories with local storage and Cosmos DB.

    Authentication uses ``azure-identity`` by default.  If no explicit
    credential is passed for Cosmos DB or AI Foundry, a
    ``DefaultAzureCredential`` is created automatically.

    Parameters
    ----------
    cosmos_endpoint : str, optional
        The Cosmos DB account endpoint URL.
    cosmos_credential : TokenCredential, optional
        Azure credential for Cosmos DB.
    cosmos_database : str, optional
        Cosmos DB database name.
    cosmos_container : str, optional
        Cosmos DB container name.
    ai_foundry_endpoint : str, optional
        Azure OpenAI endpoint URL for embeddings.
    ai_foundry_credential : TokenCredential, optional
        Azure credential for the AI Foundry endpoint.
    ai_foundry_api_key : str, optional
        API key for Azure OpenAI (takes precedence over credential).
    embedding_model : str, optional
        Embedding model deployment name (default ``text-embedding-3-large``).
    embedding_dimensions : int, optional
        Dimensionality of embedding vectors.
    adf_endpoint : str, optional
        Base URL for the Azure Durable Functions API.
    adf_key : str, optional
        Function-level key for authenticating to the Azure Function.
    use_default_credential : bool, optional
        Automatically create ``DefaultAzureCredential`` when ``True``.
    """

    def __init__(
        self,
        cosmos_endpoint: Optional[str] = None,
        cosmos_credential: Optional[Any] = None,
        cosmos_database: Optional[str] = None,
        cosmos_container: Optional[str] = None,
        cosmos_counter_container: Optional[str] = None,
        cosmos_lease_container: Optional[str] = None,
        cosmos_throughput_mode: Optional[str] = None,
        cosmos_autoscale_max_ru: Optional[int] = None,
        ai_foundry_endpoint: Optional[str] = None,
        ai_foundry_credential: Optional[Any] = None,
        ai_foundry_api_key: Optional[str] = None,
        embedding_model: str = "text-embedding-3-large",
        embedding_dimensions: Optional[int] = None,
        adf_endpoint: Optional[str] = None,
        adf_key: Optional[str] = None,
        use_default_credential: bool = True,
    ) -> None:
        # Local store
        self.local_memory: list[dict[str, Any]] = []

        # Store kwargs directly
        self._cosmos_endpoint = cosmos_endpoint
        self._cosmos_credential = cosmos_credential
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
        self._embedding_model = embedding_model
        self._embedding_dimensions = _resolve_embedding_dimensions(embedding_dimensions)

        self._adf_endpoint = adf_endpoint
        self._adf_key = adf_key

        # Resolve credentials via DefaultAzureCredential when needed
        if use_default_credential:
            needs_cosmos = self._cosmos_credential is None
            needs_embed = self._ai_foundry_credential is None
            if needs_cosmos or needs_embed:
                try:
                    from azure.identity import DefaultAzureCredential

                    _default = DefaultAzureCredential()
                except ImportError:
                    _default = None
                if needs_cosmos:
                    self._cosmos_credential = _default
                if needs_embed:
                    self._ai_foundry_credential = _default

        # Internal Cosmos SDK handles
        self._cosmos_client: Any = None
        self._container_client: Any = None

        # Composed sub-clients
        self._embeddings_client = EmbeddingsClient(
            endpoint=self._ai_foundry_endpoint,
            credential=self._ai_foundry_credential,
            api_key=self._ai_foundry_api_key,
            model=self._embedding_model,
            dimensions=self._embedding_dimensions,
        )
        self._processing_client = ProcessingClient(
            endpoint=self._adf_endpoint,
            key=self._adf_key,
        )

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
            logger.info("Cosmos client closed")

    # ------------------------------------------------------------------
    # Read-only properties
    # ------------------------------------------------------------------

    @property
    def adf_endpoint(self) -> str | None:
        """The Azure Durable Functions endpoint URL."""
        return self._adf_endpoint

    @property
    def adf_key(self) -> str | None:
        """The Azure Durable Functions key."""
        return self._adf_key

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
        )
        self.local_memory.append(memory)
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
        database: Optional[str] = None,
        container: Optional[str] = None,
    ) -> None:
        """Establish a connection to a Cosmos DB container.

        Parameters override whatever was set in ``__init__``.  After this
        call the Cosmos CRUD methods are ready to use.
        """
        self._cosmos_endpoint = endpoint or self._cosmos_endpoint
        self._cosmos_credential = credential or self._cosmos_credential
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

            client = CosmosClient(self._cosmos_endpoint, credential=self._cosmos_credential)
            db = client.get_database_client(self._cosmos_database)
            container_handle = db.get_container_client(self._cosmos_container)

            self._cosmos_client = client
            self._container_client = container_handle
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

        The memories container is provisioned with:

        * Hierarchical partition key ``[/user_id, /thread_id]``
        * ``quantizedFlat`` vector index on ``/embedding``
        * Full-text index on ``/content``
        * Throughput behavior controlled by *throughput_mode*

        Separate counter and lease containers are also provisioned.
        In ``serverless`` mode no RU/s throughput is specified.
        In ``autoscale`` mode all required containers use the same
        autoscale max RU from *autoscale_max_ru*.
        """
        self._cosmos_endpoint = endpoint or self._cosmos_endpoint
        self._cosmos_credential = credential or self._cosmos_credential
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

            client = CosmosClient(self._cosmos_endpoint, credential=self._cosmos_credential)

            db = client.create_database_if_not_exists(id=self._cosmos_database)

            partition_key = PartitionKey(path=["/user_id", "/thread_id"], kind="MultiHash")
            lease_partition_key = PartitionKey(path="/id")
            vec_policy, idx_policy, ft_policy = _container_policies(
                embedding_dimensions=embedding_dimensions or self._embedding_dimensions or 1536,
                embedding_data_type=embedding_data_type or "float32",
                distance_function=distance_function or "cosine",
                full_text_language=full_text_language or "en-US",
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
        except Exception as exc:
            raise CosmosOperationError(f"Failed to create memory store: {exc}") from exc

        logger.info(
            "Created memory store %s/%s with counter container %s and lease container %s",
            self._cosmos_database,
            self._cosmos_container,
            self._cosmos_counter_container,
            self._cosmos_lease_container,
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
    ) -> None:
        """Add a memory to Cosmos DB."""
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
        record = MemoryRecord(**kwargs)
        body = record.to_cosmos_dict()
        try:
            self._container_client.upsert_item(body=body)
        except Exception as exc:
            raise CosmosOperationError(f"Upsert failed for record {record.id}: {exc}") from exc
        logger.info("add_cosmos id=%s role=%s type=%s", record.id, role, memory_type)

    def push_to_cosmos(self, batch_size: int = 25) -> None:
        """Insert all local memories into Cosmos DB.

        Each local memory is inserted as-is, preserving its existing
        ``id``, ``thread_id``, timestamps, and metadata.

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
            for record in batch:
                body = record.to_cosmos_dict()
                try:
                    self._container_client.upsert_item(body=body)
                except Exception as exc:
                    raise CosmosOperationError(f"Upsert failed for record {record.id}: {exc}") from exc
        logger.info("Upserted batch of %d records", len(records))

    def get_memories(
        self,
        memory_id: Optional[str] = None,
        user_id: Optional[str] = None,
        thread_id: Optional[str] = None,
        role: Optional[str] = None,
        memory_type: Optional[str] = None,
        recent_k: Optional[int] = None,
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
        )
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
    ) -> list[dict[str, Any]]:
        """Search memories in Cosmos DB using vector similarity.

        1. Embeds *search_terms* via the configured embedding model.
        2. Runs a vector similarity query against the Cosmos DB container.
        3. Optionally filters by the remaining keyword parameters.
        4. Returns up to *top_k* results ordered by similarity.
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

        qb = _build_memory_query_builder(user_id=user_id, role=role, memory_type=memory_type, thread_id=thread_id)
        where = qb.build_where()
        parameters = qb.get_parameters()

        order_by = "ORDER BY VectorDistance(c.embedding, @embedding)"
        if hybrid_search:
            order_by = (
                "ORDER BY RANK RRF(VectorDistance(c.embedding, @embedding), FullTextScore(c.content, @key_terms))"
            )

        query = (
            f"SELECT TOP @top_k c.id, c.user_id, c.role, c.type, c.content, "
            f"c.metadata, c.created_at "
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
    ) -> list[dict[str, Any]]:
        """Retrieve an entire thread from Cosmos DB.

        Returns memories sorted in chronological order (oldest first).
        """
        self._require_cosmos()

        qb = _QueryBuilder()
        qb.add_filter("c.thread_id", "@thread_id", thread_id)
        qb.add_filter("c.user_id", "@user_id", user_id)
        qb.add_filter("c.type", "@memory_type", memory_type)

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
    # Processing (Azure Durable Functions)
    # ------------------------------------------------------------------

    def generate_thread_summary(
        self,
        user_id: str,
        thread_id: str,
        recent_k: Optional[int] = None,
        poll_interval: float = 2.0,
        timeout: float = 120.0,
    ) -> dict[str, Any]:
        """Trigger the Azure Durable Function to generate a thread summary."""
        logger.info(
            "generate_thread_summary started user_id=%s thread_id=%s",
            user_id,
            thread_id,
        )
        return self._processing_client.generate_thread_summary(
            user_id=user_id,
            thread_id=thread_id,
            recent_k=recent_k,
            poll_interval=poll_interval,
            timeout=timeout,
        )

    def extract_facts(
        self,
        user_id: str,
        thread_id: str,
        recent_k: Optional[int] = None,
        poll_interval: float = 2.0,
        timeout: float = 120.0,
    ) -> dict[str, Any]:
        """Trigger the Azure Durable Function to extract facts from a thread."""
        logger.info(
            "extract_facts started user_id=%s thread_id=%s",
            user_id,
            thread_id,
        )
        return self._processing_client.extract_facts(
            user_id=user_id,
            thread_id=thread_id,
            recent_k=recent_k,
            poll_interval=poll_interval,
            timeout=timeout,
        )

    def generate_user_summary(
        self,
        user_id: str,
        thread_ids: Optional[list[str]] = None,
        recent_k: Optional[int] = None,
        poll_interval: float = 2.0,
        timeout: float = 120.0,
    ) -> dict[str, Any]:
        """Trigger the Azure Durable Function to generate a cross-thread user summary."""
        logger.info("generate_user_summary started user_id=%s", user_id)
        return self._processing_client.generate_user_summary(
            user_id=user_id,
            thread_ids=thread_ids,
            recent_k=recent_k,
            poll_interval=poll_interval,
            timeout=timeout,
        )
