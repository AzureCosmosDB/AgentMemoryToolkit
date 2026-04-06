"""Cosmos DB client layer for the Agent Memory Toolkit.

Provides :class:`CosmosMemoryStore` — a synchronous client that owns **all**
Cosmos DB interaction logic: connection, container creation, CRUD, and
vector search.

Embedding generation is **not** this module's responsibility; callers must
supply pre-computed vectors to :meth:`vector_search`.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Optional

from ._query_builder import _QueryBuilder
from .exceptions import (
    ConfigurationError,
    CosmosNotConnectedError,
    CosmosOperationError,
    MemoryNotFoundError,
)
from .models import MemoryRecord

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Shared helpers
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


# ---------------------------------------------------------------------------
# Sync client
# ---------------------------------------------------------------------------


class CosmosMemoryStore:
    """Synchronous Cosmos DB client for memory CRUD and vector search.

    Parameters
    ----------
    endpoint:
        Cosmos DB account endpoint URL.
    credential:
        Azure ``TokenCredential`` or key string for authentication.
    database:
        Name of the Cosmos DB database.  Defaults to ``"ai_memory"``.
    container:
        Name of the Cosmos DB container.  Defaults to ``"memories"``.
    """

    def __init__(
        self,
        endpoint: str | None = None,
        credential: Any = None,
        database: str = "ai_memory",
        container: str = "memories",
    ) -> None:
        self._endpoint = endpoint
        self._credential = credential
        self._database = database
        self._container = container
        self._container_client: Any = None

    # -- connection ---------------------------------------------------------

    def connect(self) -> None:
        """Create a :class:`CosmosClient` and obtain the container handle.

        Raises
        ------
        ConfigurationError
            If any required configuration field is missing.
        CosmosOperationError
            If the connection to Cosmos DB fails.
        """
        _validate_connection(
            self._endpoint, self._credential, self._database, self._container
        )

        try:
            from azure.cosmos import CosmosClient

            client = CosmosClient(
                self._endpoint, credential=self._credential
            )
            db = client.get_database_client(self._database)
            self._container_client = db.get_container_client(self._container)
        except Exception as exc:
            raise CosmosOperationError(
                f"Failed to connect to Cosmos DB: {exc}"
            ) from exc

        logger.info(
            "Connected to Cosmos DB %s/%s",
            self._database,
            self._container,
        )

    def create_store(
        self,
        embedding_dimensions: int = 1536,
        embedding_data_type: str = "float32",
        distance_function: str = "cosine",
        full_text_language: str = "en-US",
        autoscale_max_ru: int = 1000,
    ) -> None:
        """Create the database and container, then connect.

        The container is provisioned with:

        * Hierarchical partition key ``[/user_id, /thread_id]``
        * ``quantizedFlat`` vector index on ``/embedding``
        * Full-text index on ``/content``
        * Autoscale throughput (max RU from *autoscale_max_ru*)

        Raises
        ------
        ConfigurationError
            If required fields are missing.
        CosmosOperationError
            If the Cosmos DB operation fails.
        """
        _validate_connection(
            self._endpoint, self._credential, self._database, self._container
        )

        try:
            from azure.cosmos import CosmosClient, PartitionKey, ThroughputProperties

            client = CosmosClient(
                self._endpoint, credential=self._credential
            )

            db = client.create_database_if_not_exists(id=self._database)

            partition_key = PartitionKey(
                path=["/user_id", "/thread_id"], kind="MultiHash"
            )
            vec_policy, idx_policy, ft_policy = _container_policies(
                embedding_dimensions=embedding_dimensions,
                embedding_data_type=embedding_data_type,
                distance_function=distance_function,
                full_text_language=full_text_language,
            )

            container = db.create_container_if_not_exists(
                id=self._container,
                partition_key=partition_key,
                indexing_policy=idx_policy,
                vector_embedding_policy=vec_policy,
                full_text_policy=ft_policy,
                offer_throughput=ThroughputProperties(
                    auto_scale_max_throughput=autoscale_max_ru,
                ),
            )
            self._container_client = container
        except Exception as exc:
            raise CosmosOperationError(
                f"Failed to create memory store: {exc}"
            ) from exc

        logger.info(
            "Created memory store %s/%s",
            self._database,
            self._container,
        )

    def _require_connected(self) -> None:
        """Raise if no active container client."""
        if self._container_client is None:
            raise CosmosNotConnectedError()

    # -- upsert -------------------------------------------------------------

    def upsert(self, record: MemoryRecord) -> None:
        """Upsert a single :class:`MemoryRecord` into Cosmos DB."""
        self._require_connected()
        body = record.to_cosmos_dict()
        try:
            self._container_client.upsert_item(body=body)
        except Exception as exc:
            raise CosmosOperationError(
                f"Upsert failed for record {record.id}: {exc}"
            ) from exc
        logger.info("Upserted record %s", record.id)

    def upsert_batch(self, records: list[MemoryRecord]) -> None:
        """Upsert multiple records sequentially."""
        self._require_connected()
        for record in records:
            self.upsert(record)
        logger.info("Upserted batch of %d records", len(records))

    # -- queries ------------------------------------------------------------

    def get_memories(
        self,
        memory_id: Optional[str] = None,
        user_id: Optional[str] = None,
        thread_id: Optional[str] = None,
        role: Optional[str] = None,
        memory_type: Optional[str] = None,
        recent_k: Optional[int] = None,
    ) -> list[dict[str, Any]]:
        """Query memories with optional filters.

        Returns raw dicts (callers convert to :class:`MemoryRecord` if needed).
        When *recent_k* is given the newest *k* documents are returned in
        chronological (oldest-first) order.
        """
        self._require_connected()

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
            raise CosmosOperationError(
                f"get_memories query failed: {exc}"
            ) from exc

        if recent_k is not None:
            items.reverse()
        return items

    def get_thread(
        self,
        thread_id: str,
        user_id: Optional[str] = None,
        memory_type: Optional[str] = None,
        recent_k: Optional[int] = None,
    ) -> list[dict[str, Any]]:
        """Retrieve an entire thread, oldest-first."""
        self._require_connected()

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
            raise CosmosOperationError(
                f"get_thread query failed: {exc}"
            ) from exc

        if recent_k is not None:
            items = items[:recent_k]
        items.reverse()
        return items

    def get_user_summary(self, user_id: str) -> list[dict[str, Any]]:
        """Retrieve user-summary documents, newest-first."""
        self._require_connected()

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
            raise CosmosOperationError(
                f"get_user_summary query failed: {exc}"
            ) from exc

    # -- update / delete ----------------------------------------------------

    def update(
        self,
        memory_id: str,
        content: Optional[str] = None,
        role: Optional[str] = None,
        memory_type: Optional[str] = None,
        metadata: Optional[dict[str, Any]] = None,
    ) -> None:
        """Update fields on an existing memory document.

        Raises
        ------
        MemoryNotFoundError
            If no document with *memory_id* exists.
        CosmosOperationError
            If the underlying Cosmos DB operation fails.
        """
        self._require_connected()

        try:
            results = list(
                self._container_client.query_items(
                    query="SELECT * FROM c WHERE c.id = @id",
                    parameters=[{"name": "@id", "value": memory_id}],
                    enable_cross_partition_query=True,
                )
            )
        except Exception as exc:
            raise CosmosOperationError(
                f"update query failed: {exc}"
            ) from exc

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
            raise CosmosOperationError(
                f"update replace failed for {memory_id}: {exc}"
            ) from exc

        logger.info("Updated record %s", memory_id)

    def delete(self, memory_id: str, user_id: str, thread_id: str) -> None:
        """Delete a memory document.

        Raises
        ------
        MemoryNotFoundError
            If no matching document is found.
        CosmosOperationError
            If the underlying Cosmos DB operation fails.
        """
        self._require_connected()

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
            raise CosmosOperationError(
                f"delete lookup failed: {exc}"
            ) from exc

        if not results:
            raise MemoryNotFoundError(
                memory_id=memory_id, user_id=user_id, thread_id=thread_id
            )

        try:
            self._container_client.delete_item(
                item=memory_id, partition_key=[user_id, thread_id]
            )
        except Exception as exc:
            raise CosmosOperationError(
                f"delete failed for {memory_id}: {exc}"
            ) from exc

        logger.info("Deleted record %s", memory_id)

    # -- vector search ------------------------------------------------------

    def vector_search(
        self,
        query_vector: list[float],
        user_id: Optional[str] = None,
        role: Optional[str] = None,
        memory_type: Optional[str] = None,
        thread_id: Optional[str] = None,
        hybrid_search: bool = False,
        search_terms: Optional[str] = None,
        top_k: int = 5,
    ) -> list[dict[str, Any]]:
        """Run a vector (or hybrid) similarity search.

        Parameters
        ----------
        query_vector : list[float]
            Pre-computed embedding vector.
        search_terms : str, optional
            Raw text for the full-text component of a hybrid search.
            Required when *hybrid_search* is ``True``.
        """
        self._require_connected()

        qb = _build_memory_query_builder(
            user_id=user_id, role=role, memory_type=memory_type, thread_id=thread_id
        )
        where = qb.build_where()
        parameters = qb.get_parameters()

        order_by = "ORDER BY VectorDistance(c.embedding, @embedding)"
        if hybrid_search:
            order_by = (
                "ORDER BY RANK RRF("
                "VectorDistance(c.embedding, @embedding), "
                "FullTextScore(c.content, @key_terms)"
                ")"
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

        logger.debug("vector_search query: %s", query)

        try:
            items = list(
                self._container_client.query_items(
                    query=query,
                    parameters=parameters,
                    enable_cross_partition_query=True,
                )
            )
        except Exception as exc:
            raise CosmosOperationError(
                f"vector_search failed: {exc}"
            ) from exc

        return items
