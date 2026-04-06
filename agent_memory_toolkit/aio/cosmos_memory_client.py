"""Async Cosmos DB client for the Agent Memory Toolkit.

Provides :class:`AsyncCosmosMemoryStore` which uses
``azure.cosmos.aio.CosmosClient`` for non-blocking operations.

Embedding generation is **not** this module's responsibility; callers must
supply pre-computed vectors to :meth:`vector_search`.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone
from typing import Any, Optional

from agent_memory_toolkit._query_builder import _QueryBuilder
from agent_memory_toolkit.exceptions import (
    ConfigurationError,
    CosmosNotConnectedError,
    CosmosOperationError,
    MemoryNotFoundError,
)
from agent_memory_toolkit.models import MemoryRecord

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _build_memory_query_builder(
    *,
    memory_id: Optional[str] = None,
    user_id: Optional[str] = None,
    thread_id: Optional[str] = None,
    role: Optional[str] = None,
    memory_type: Optional[str] = None,
) -> _QueryBuilder:
    """Return a :class:`_QueryBuilder` pre-loaded with standard filters."""
    qb = _QueryBuilder()
    qb.add_filter("c.id", "@memory_id", memory_id)
    qb.add_filter("c.user_id", "@user_id", user_id)
    qb.add_filter("c.thread_id", "@thread_id", thread_id)
    qb.add_filter("c.role", "@role", role)
    qb.add_filter("c.type", "@memory_type", memory_type)
    return qb


# ---------------------------------------------------------------------------
# Async client
# ---------------------------------------------------------------------------


class AsyncCosmosMemoryStore:
    """Async Cosmos DB client for the Agent Memory Toolkit.

    Uses ``azure.cosmos.aio.CosmosClient`` for non-blocking operations.
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
        self._cosmos_client: Any = None
        self._container_client: Any = None

    # -- context manager ----------------------------------------------------

    async def __aenter__(self) -> AsyncCosmosMemoryStore:
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        await self.close()

    async def close(self) -> None:
        """Close the underlying async Cosmos client."""
        if self._cosmos_client is not None:
            await self._cosmos_client.close()
            self._cosmos_client = None
            self._container_client = None
            logger.info("Async Cosmos client closed")

    # -- connection ---------------------------------------------------------

    async def connect(self) -> None:
        """Create an async :class:`CosmosClient` and obtain the container.

        Raises
        ------
        ConfigurationError
            If required fields are missing.
        CosmosOperationError
            If the connection fails.
        """
        if not self._endpoint:
            raise ConfigurationError(parameter="endpoint")
        if not self._credential:
            raise ConfigurationError(parameter="credential")

        try:
            from azure.cosmos.aio import CosmosClient

            self._cosmos_client = CosmosClient(
                self._endpoint, credential=self._credential
            )
            db = self._cosmos_client.get_database_client(self._database)
            self._container_client = db.get_container_client(self._container)
        except Exception as exc:
            raise CosmosOperationError(
                f"Failed to connect to Cosmos DB (async): {exc}"
            ) from exc

        logger.info(
            "Async connected to Cosmos DB %s/%s",
            self._database,
            self._container,
        )

    async def create_store(
        self,
        embedding_dimensions: int = 1536,
        embedding_data_type: str = "float32",
        distance_function: str = "cosine",
        full_text_language: str = "en-US",
        autoscale_max_ru: int = 1000,
    ) -> None:
        """Create the database and container (async), then set the container handle.

        The container is provisioned with:

        * Hierarchical partition key ``[/user_id, /thread_id]``
        * ``quantizedFlat`` vector index on ``/embedding``
        * Full-text index on ``/content``
        * Autoscale throughput (max RU)
        """
        if not self._endpoint:
            raise ConfigurationError(parameter="endpoint")
        if not self._credential:
            raise ConfigurationError(parameter="credential")

        try:
            from azure.cosmos import PartitionKey, ThroughputProperties
            from azure.cosmos.aio import CosmosClient

            self._cosmos_client = CosmosClient(
                self._endpoint, credential=self._credential
            )

            db = await self._cosmos_client.create_database_if_not_exists(
                id=self._database
            )

            partition_key = PartitionKey(
                path=["/user_id", "/thread_id"], kind="MultiHash"
            )

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
                "fullTextPaths": [
                    {"path": "/content", "language": full_text_language}
                ],
            }

            container = await db.create_container_if_not_exists(
                id=self._container,
                partition_key=partition_key,
                indexing_policy=indexing_policy,
                vector_embedding_policy=vector_embedding_policy,
                full_text_policy=full_text_policy,
                offer_throughput=ThroughputProperties(
                    auto_scale_max_throughput=autoscale_max_ru,
                ),
            )
            self._container_client = container
        except Exception as exc:
            raise CosmosOperationError(
                f"Failed to create memory store (async): {exc}"
            ) from exc

        logger.info(
            "Async created memory store %s/%s",
            self._database,
            self._container,
        )

    def _require_connected(self) -> None:
        """Raise if no active container client."""
        if self._container_client is None:
            raise CosmosNotConnectedError()

    # -- upsert -------------------------------------------------------------

    async def upsert(self, record: MemoryRecord) -> None:
        """Upsert a single :class:`MemoryRecord`."""
        self._require_connected()
        body = record.to_cosmos_dict()
        try:
            await self._container_client.upsert_item(body=body)
        except Exception as exc:
            raise CosmosOperationError(
                f"Async upsert failed for record {record.id}: {exc}"
            ) from exc
        logger.info("Async upserted record %s", record.id)

    async def upsert_batch(
        self, records: list[MemoryRecord], batch_size: int = 25
    ) -> None:
        """Upsert multiple records using ``asyncio.gather`` in batches."""
        self._require_connected()

        for start in range(0, len(records), batch_size):
            batch = records[start : start + batch_size]
            tasks = [self.upsert(record) for record in batch]
            await asyncio.gather(*tasks)

        logger.info("Async upserted batch of %d records", len(records))

    # -- queries ------------------------------------------------------------

    async def get_memories(
        self,
        memory_id: Optional[str] = None,
        user_id: Optional[str] = None,
        thread_id: Optional[str] = None,
        role: Optional[str] = None,
        memory_type: Optional[str] = None,
        recent_k: Optional[int] = None,
    ) -> list[dict[str, Any]]:
        """Query memories with optional filters.

        Returns raw dicts.  When *recent_k* is given the newest *k*
        documents are returned in chronological (oldest-first) order.
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

        logger.debug("async get_memories query: %s", query)

        try:
            items_iter = self._container_client.query_items(
                query=query,
                parameters=parameters or None,
            )
            results = [item async for item in items_iter]
        except Exception as exc:
            raise CosmosOperationError(
                f"async get_memories query failed: {exc}"
            ) from exc

        if recent_k is not None:
            results.reverse()
        return results

    async def get_thread(
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
        logger.debug("async get_thread query: %s", query)

        try:
            items_iter = self._container_client.query_items(
                query=query, parameters=parameters
            )
            items = [item async for item in items_iter]
        except Exception as exc:
            raise CosmosOperationError(
                f"async get_thread query failed: {exc}"
            ) from exc

        if recent_k is not None:
            items = items[:recent_k]
        items.reverse()
        return items

    async def get_user_summary(self, user_id: str) -> list[dict[str, Any]]:
        """Retrieve user-summary documents, newest-first."""
        self._require_connected()

        query = (
            "SELECT c.id, c.user_id, c.thread_id, c.role, c.type, "
            "c.content, c.metadata, c.created_at "
            "FROM c WHERE c.user_id = @user_id AND c.type = 'user_summary' "
            "ORDER BY c.created_at DESC"
        )
        parameters = [{"name": "@user_id", "value": user_id}]
        logger.debug("async get_user_summary query: %s", query)

        try:
            items_iter = self._container_client.query_items(
                query=query, parameters=parameters
            )
            return [item async for item in items_iter]
        except Exception as exc:
            raise CosmosOperationError(
                f"async get_user_summary query failed: {exc}"
            ) from exc

    # -- update / delete ----------------------------------------------------

    async def update(
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
            If the document does not exist.
        CosmosOperationError
            If the underlying Cosmos DB operation fails.
        """
        self._require_connected()

        try:
            items_iter = self._container_client.query_items(
                query="SELECT * FROM c WHERE c.id = @id",
                parameters=[{"name": "@id", "value": memory_id}],
            )
            docs = [item async for item in items_iter]
        except Exception as exc:
            raise CosmosOperationError(
                f"async update query failed: {exc}"
            ) from exc

        if not docs:
            raise MemoryNotFoundError(memory_id=memory_id)

        doc = docs[0]
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
            await self._container_client.replace_item(item=doc["id"], body=doc)
        except Exception as exc:
            raise CosmosOperationError(
                f"async update replace failed for {memory_id}: {exc}"
            ) from exc

        logger.info("Async updated record %s", memory_id)

    async def delete(self, memory_id: str, user_id: str, thread_id: str) -> None:
        """Delete a memory document.

        Raises
        ------
        MemoryNotFoundError
            If no matching document exists.
        CosmosOperationError
            If the underlying Cosmos DB operation fails.
        """
        self._require_connected()

        try:
            items_iter = self._container_client.query_items(
                query=(
                    "SELECT TOP 1 c.id FROM c WHERE c.id = @id "
                    "AND c.thread_id = @thread_id AND c.user_id = @user_id"
                ),
                parameters=[
                    {"name": "@id", "value": memory_id},
                    {"name": "@thread_id", "value": thread_id},
                    {"name": "@user_id", "value": user_id},
                ],
            )
            docs = [item async for item in items_iter]
        except Exception as exc:
            raise CosmosOperationError(
                f"async delete lookup failed: {exc}"
            ) from exc

        if not docs:
            raise MemoryNotFoundError(
                memory_id=memory_id, user_id=user_id, thread_id=thread_id
            )

        try:
            await self._container_client.delete_item(
                item=memory_id, partition_key=[user_id, thread_id]
            )
        except Exception as exc:
            raise CosmosOperationError(
                f"async delete failed for {memory_id}: {exc}"
            ) from exc

        logger.info("Async deleted record %s", memory_id)

    # -- vector search ------------------------------------------------------

    async def vector_search(
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

        logger.debug("async vector_search query: %s", query)

        try:
            items_iter = self._container_client.query_items(
                query=query, parameters=parameters
            )
            return [item async for item in items_iter]
        except Exception as exc:
            raise CosmosOperationError(
                f"async vector_search failed: {exc}"
            ) from exc
