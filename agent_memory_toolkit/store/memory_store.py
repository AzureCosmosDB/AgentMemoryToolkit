"""Synchronous Cosmos DB memory store primitives."""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Optional

from agent_memory_toolkit._query_builder import _QueryBuilder
from agent_memory_toolkit._utils import _build_memory_query_builder, _validate_hybrid_search
from agent_memory_toolkit.exceptions import (
    ConfigurationError,
    CosmosOperationError,
    MemoryNotFoundError,
)
from agent_memory_toolkit.models import MemoryRecord
from agent_memory_toolkit.store._search_helpers import (
    add_salience_filter,
    add_tag_filters,
    build_search_sql,
    coerce_embedding,
    format_episodic_context,
    query_scope,
    require_search_terms,
    top_literal,
)

logger = logging.getLogger(__name__)


class MemoryStore:
    """Typed CRUD and query primitives over a Cosmos DB container."""

    def __init__(self, container: Any, *, embeddings_client: Any = None) -> None:
        self._container = container
        self._embeddings_client = embeddings_client

    @property
    def container(self) -> Any:
        """Return the underlying Cosmos container client."""
        return self._container

    def read_item(self, item_id: str, partition_key: Any) -> dict[str, Any]:
        """Point-read a memory document by id and partition key."""
        from azure.cosmos.exceptions import CosmosResourceNotFoundError

        try:
            return self._container.read_item(item=item_id, partition_key=partition_key)
        except CosmosResourceNotFoundError as exc:
            raise MemoryNotFoundError(memory_id=item_id) from exc
        except Exception as exc:
            raise CosmosOperationError(f"read_item failed for {item_id}: {exc}") from exc

    def query(
        self,
        sql: str,
        parameters: Optional[list[dict[str, Any]]] = None,
        partition_key: Any = None,
        cross_partition: bool = False,
    ) -> list[dict[str, Any]]:
        """Run a parameterized Cosmos query and return all results."""
        return self._query_items(
            query=sql,
            parameters=parameters,
            partition_key=partition_key,
            cross_partition=cross_partition,
            operation="query",
        )

    def _query_items(
        self,
        *,
        query: str,
        parameters: Optional[list[dict[str, Any]]] = None,
        partition_key: Any = None,
        cross_partition: bool = False,
        operation: str,
    ) -> list[dict[str, Any]]:
        kwargs: dict[str, Any] = {"query": query, "parameters": parameters or None}
        if partition_key is not None:
            kwargs["partition_key"] = partition_key
        if cross_partition:
            kwargs["enable_cross_partition_query"] = True
        try:
            return list(self._container.query_items(**kwargs))
        except Exception as exc:
            raise CosmosOperationError(f"{operation} failed: {exc}") from exc

    def add_cosmos(self, record: dict[str, Any]) -> dict[str, Any]:
        """Upsert a pre-built Cosmos memory document and return the stored body."""
        try:
            response = self._container.upsert_item(body=record)
        except Exception as exc:
            raise CosmosOperationError(f"add_cosmos upsert failed for record {record.get('id')}: {exc}") from exc
        logger.info("add_cosmos id=%s role=%s type=%s", record.get("id"), record.get("role"), record.get("type"))
        return response if isinstance(response, dict) else record

    def add(
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
        """Add a memory document to Cosmos DB and return its id."""
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

        if embed is None:
            embed = memory_type != "turn"
        if embedding is not None:
            body["embedding"] = embedding
        elif embed and content and self._embeddings_client is not None:
            try:
                body["embedding"] = self._embeddings_client.generate(content)
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "add_cosmos: embedding generation failed for %s (%s); proceeding without embedding",
                    record.id,
                    exc,
                )

        try:
            self._container.upsert_item(body=body)
        except Exception as exc:
            raise CosmosOperationError(f"Upsert failed for record {record.id}: {exc}") from exc
        logger.info("add_cosmos id=%s role=%s type=%s", record.id, role, memory_type)
        return record.id

    def push(self, local_memory: list[dict[str, Any]], batch_size: int = 25) -> None:
        """Upsert all local memory records to Cosmos DB in sequential batches."""
        if batch_size <= 0:
            raise ValueError("batch_size must be greater than 0")
        logger.info(
            "push_to_cosmos count=%d batch_size=%d",
            len(local_memory),
            batch_size,
        )
        records = [MemoryRecord.from_cosmos_dict(dict(m)) for m in local_memory]
        for start in range(0, len(records), batch_size):
            batch = records[start : start + batch_size]
            bodies = [r.to_cosmos_dict() for r in batch]

            to_embed_idx: list[int] = []
            to_embed_text: list[str] = []
            for i, body in enumerate(bodies):
                if body.get("type") != "turn" and body.get("content") and not body.get("embedding"):
                    to_embed_idx.append(i)
                    to_embed_text.append(body["content"])
            if to_embed_text and self._embeddings_client is not None:
                try:
                    vectors = self._embeddings_client.generate_batch(to_embed_text)
                    for i, vec in zip(to_embed_idx, vectors):
                        bodies[i]["embedding"] = vec
                        local_memory[start + i]["embedding"] = vec
                except Exception as exc:  # noqa: BLE001
                    logger.warning(
                        "push_to_cosmos: batch embedding generation failed (%s); "
                        "proceeding without embeddings for %d records",
                        exc,
                        len(to_embed_text),
                    )

            for record, body in zip(batch, bodies):
                try:
                    self._container.upsert_item(body=body)
                except Exception as exc:
                    raise CosmosOperationError(f"Upsert failed for record {record.id}: {exc}") from exc
        logger.info("Upserted batch of %d records", len(records))

    def get_memories(
        self,
        memory_id: Optional[str] = None,
        user_id: Optional[str] = None,
        thread_id: Optional[str] = None,
        role: Optional[str] = None,
        memory_types: Optional[list[str]] = None,
        recent_k: Optional[int] = None,
        tags: Optional[list[str]] = None,
        any_tags: Optional[list[str]] = None,
        exclude_tags: Optional[list[str]] = None,
        include_superseded: bool = False,
        min_salience: Optional[float] = None,
        min_confidence: Optional[float] = None,
    ) -> list[dict[str, Any]]:
        """Retrieve memories from Cosmos DB with optional filters."""
        logger.debug(
            "get_memories filters: memory_id=%s user_id=%s thread_id=%s role=%s types=%s recent_k=%s",
            memory_id,
            user_id,
            thread_id,
            role,
            memory_types,
            recent_k,
        )

        qb = _build_memory_query_builder(
            memory_id=memory_id,
            user_id=user_id,
            thread_id=thread_id,
            role=role,
            memory_types=memory_types,
            min_confidence=min_confidence,
        )

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
        items = self._query_items(
            query=query,
            parameters=parameters or None,
            cross_partition=True,
            operation="get_memories query",
        )

        if recent_k is not None:
            items.reverse()
        if min_salience is not None:
            items = [i for i in items if (i.get("salience") or 0.0) >= min_salience]
        if not items:
            logger.warning("get_memories returned empty results")
        return items

    def update(
        self,
        memory_id: str,
        content: Optional[str] = None,
        role: Optional[str] = None,
        memory_type: Optional[str] = None,
        metadata: Optional[dict[str, Any]] = None,
    ) -> None:
        """Update a memory document in Cosmos DB."""
        results = self._query_items(
            query="SELECT * FROM c WHERE c.id = @id",
            parameters=[{"name": "@id", "value": memory_id}],
            cross_partition=True,
            operation="update query",
        )
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
            self._container.replace_item(item=doc["id"], body=doc)
        except Exception as exc:
            raise CosmosOperationError(f"update replace failed for {memory_id}: {exc}") from exc

        logger.info("Updated record %s", memory_id)

    def delete(self, memory_id: str, thread_id: str, user_id: str) -> None:
        """Delete a memory document from Cosmos DB."""
        results = self._query_items(
            query=(
                "SELECT TOP 1 c.id FROM c WHERE c.id = @id "
                "AND c.thread_id = @thread_id AND c.user_id = @user_id"
            ),
            parameters=[
                {"name": "@id", "value": memory_id},
                {"name": "@thread_id", "value": thread_id},
                {"name": "@user_id", "value": user_id},
            ],
            cross_partition=True,
            operation="delete lookup",
        )
        if not results:
            raise MemoryNotFoundError(memory_id=memory_id, user_id=user_id, thread_id=thread_id)

        try:
            self._container.delete_item(item=memory_id, partition_key=[user_id, thread_id])
        except Exception as exc:
            raise CosmosOperationError(f"delete failed for {memory_id}: {exc}") from exc

        logger.info("Deleted record %s", memory_id)

    def get_thread(
        self,
        thread_id: str,
        user_id: Optional[str] = None,
        memory_types: Optional[list[str]] = None,
        recent_k: Optional[int] = None,
        tags: Optional[list[str]] = None,
        exclude_tags: Optional[list[str]] = None,
        include_superseded: bool = False,
    ) -> list[dict[str, Any]]:
        """Retrieve an entire thread sorted oldest first."""
        qb = _QueryBuilder()
        qb.add_filter("c.thread_id", "@thread_id", thread_id)
        qb.add_filter("c.user_id", "@user_id", user_id)
        if memory_types:
            qb.add_in_filter("c.type", "@memory_type_", list(memory_types))
        if tags:
            for i, tag in enumerate(tags):
                qb.add_array_contains("c.tags", f"@tag_{i}", tag)
        if exclude_tags:
            for i, tag in enumerate(exclude_tags):
                qb.add_not_array_contains("c.tags", f"@exc_tag_{i}", tag)
        if not include_superseded:
            qb.add_is_null_or_undefined("c.superseded_by")

        query = f"SELECT * FROM c{qb.build_where()} ORDER BY c.created_at DESC"
        logger.debug("get_thread query: %s", query)
        items = self._query_items(
            query=query,
            parameters=qb.get_parameters(),
            cross_partition=True,
            operation="get_thread query",
        )
        if recent_k is not None:
            items = items[:recent_k]
        items.reverse()
        return items

    def get_user_summary(self, user_id: str) -> Optional[dict[str, Any]]:
        """Retrieve the user's summary document from Cosmos DB, or ``None`` if absent."""
        from azure.cosmos.exceptions import CosmosResourceNotFoundError

        try:
            return self._container.read_item(
                item=f"user_summary_{user_id}",
                partition_key=[user_id, "__user_summary__"],
            )
        except CosmosResourceNotFoundError:
            return None
        except Exception as exc:
            raise CosmosOperationError(f"get_user_summary read failed: {exc}") from exc

    def add_tags(self, memory_id: str, user_id: str, thread_id: str, tags: list[str]) -> None:
        """Add tags to an existing memory document."""
        doc = self._container.read_item(item=memory_id, partition_key=[user_id, thread_id])
        existing_tags = set(doc.get("tags", []))
        existing_tags.update(t.strip().lower() for t in tags)
        doc["tags"] = sorted(existing_tags)
        doc["updated_at"] = datetime.now(timezone.utc).isoformat()
        self._container.replace_item(item=memory_id, body=doc)

    def remove_tags(self, memory_id: str, user_id: str, thread_id: str, tags: list[str]) -> None:
        """Remove tags from an existing memory document."""
        doc = self._container.read_item(item=memory_id, partition_key=[user_id, thread_id])
        tags_to_remove = {t.strip().lower() for t in tags}
        doc["tags"] = sorted(set(doc.get("tags", [])) - tags_to_remove)
        doc["updated_at"] = datetime.now(timezone.utc).isoformat()
        self._container.replace_item(item=memory_id, body=doc)

    def mark_superseded(
        self,
        old_doc: dict[str, Any],
        superseder_id: str,
        *,
        reason: str,
    ) -> bool:
        """Set supersession audit fields using ETag protection when available."""
        from azure.core import MatchConditions
        from azure.cosmos.exceptions import CosmosAccessConditionFailedError

        etag = old_doc.get("_etag")
        new_doc = {
            **old_doc,
            "superseded_by": superseder_id,
            "supersede_reason": reason,
            "superseded_at": datetime.now(timezone.utc).isoformat(),
        }
        try:
            if etag:
                self._container.replace_item(
                    item=new_doc["id"],
                    body=new_doc,
                    match_condition=MatchConditions.IfNotModified,
                    etag=etag,
                )
            else:
                self._container.upsert_item(body=new_doc)
            return True
        except CosmosAccessConditionFailedError:
            logger.info(
                "supersede skipped (concurrent writer won) id=%s superseder=%s",
                old_doc.get("id"),
                superseder_id,
            )
            return False
        except Exception:
            logger.exception("supersede failed id=%s superseder=%s", old_doc.get("id"), superseder_id)
            return False

    def get_procedural_prompt(self, user_id: str) -> Optional[str]:
        """Return the active synthesized procedural prompt for a user."""
        qb = _QueryBuilder()
        qb.add_filter("c.user_id", "@user_id", user_id)
        qb.add_filter("c.thread_id", "@thread_id", "__procedural__")
        qb.add_filter("c.type", "@type", "procedural")
        qb.add_is_null_or_undefined("c.superseded_by")

        query = f"SELECT TOP 1 c.content, c.version FROM c{qb.build_where()} ORDER BY c.version DESC"
        items = self._query_items(
            query=query,
            parameters=qb.get_parameters(),
            cross_partition=True,
            operation="get_procedural_prompt query",
        )
        if not items:
            return None
        return items[0].get("content")

    def get_procedural_history(self, user_id: str, limit: int = 10) -> list[dict[str, Any]]:
        """Return synthesized procedural docs for a user, newest first."""
        if limit <= 0:
            return []

        qb = _QueryBuilder()
        qb.add_filter("c.user_id", "@user_id", user_id)
        qb.add_filter("c.thread_id", "@thread_id", "__procedural__")
        qb.add_filter("c.type", "@type", "procedural")

        query = f"SELECT * FROM c{qb.build_where()} ORDER BY c.version DESC"
        items = self._query_items(
            query=query,
            parameters=qb.get_parameters(),
            cross_partition=True,
            operation="get_procedural_history query",
        )

        def _is_active(doc: dict[str, Any]) -> bool:
            return not doc.get("superseded_by")

        items.sort(
            key=lambda doc: (
                1 if _is_active(doc) else 0,
                int(doc.get("version") or 0),
                int(doc.get("_ts") or 0),
            ),
            reverse=True,
        )
        return items[:limit]

    def get_procedural_memories(
        self,
        user_id: str,
        priority: Optional[str] = None,
        category: Optional[str] = None,
        min_salience: Optional[float] = None,
        include_superseded: bool = False,
    ) -> list[dict[str, Any]]:
        """Retrieve active procedural memories for a user."""
        qb = _QueryBuilder()
        qb.add_filter("c.user_id", "@user_id", user_id)
        qb.add_filter("c.thread_id", "@thread_id", "__procedural__")
        qb.add_filter("c.type", "@type", "procedural")
        if not include_superseded:
            qb.add_is_null_or_undefined("c.superseded_by")

        query = f"SELECT * FROM c{qb.build_where()} ORDER BY c.created_at DESC"
        items = self._query_items(
            query=query,
            parameters=qb.get_parameters(),
            cross_partition=True,
            operation="get_procedural_memories query",
        )

        if min_salience is not None:
            items = [i for i in items if (i.get("salience") or 0.0) >= min_salience]
        if priority is not None:
            items = [i for i in items if i.get("metadata", {}).get("priority") == priority]
        if category is not None:
            items = [i for i in items if i.get("metadata", {}).get("category") == category]
        return items

    # -- retrieval ----------------------------------------------------------

    def search(
        self,
        search_terms: Optional[str] = None,
        memory_id: Optional[str] = None,
        user_id: Optional[str] = None,
        role: Optional[str] = None,
        memory_types: Optional[list[str]] = None,
        thread_id: Optional[str] = None,
        hybrid_search: bool = False,
        top_k: int = 5,
        tags: Optional[list[str]] = None,
        any_tags: Optional[list[str]] = None,
        exclude_tags: Optional[list[str]] = None,
        include_superseded: bool = False,
        min_salience: Optional[float] = None,
        min_confidence: Optional[float] = None,
        *,
        query: Optional[str] = None,
        tags_any: Optional[list[str]] = None,
        tags_all: Optional[list[str]] = None,
    ) -> list[dict[str, Any]]:
        """Search memories using vector similarity with optional full-text hybrid ranking."""
        terms = require_search_terms(search_terms, query)
        _validate_hybrid_search(hybrid_search, terms)
        top = top_literal(top_k, name="top_k")
        query_vector = self._embed(terms)

        qb = _build_memory_query_builder(
            memory_id=memory_id,
            user_id=user_id,
            role=role,
            memory_types=memory_types,
            thread_id=thread_id,
            min_confidence=min_confidence,
        )
        add_tag_filters(
            qb,
            tags=tags,
            tags_all=tags_all,
            any_tags=any_tags,
            tags_any=tags_any,
            exclude_tags=exclude_tags,
        )
        add_salience_filter(qb, min_salience)

        sql = build_search_sql(qb=qb, top=top, hybrid_search=hybrid_search, include_superseded=include_superseded)
        parameters = qb.get_parameters()
        parameters.append({"name": "@embedding", "value": query_vector})
        if hybrid_search:
            parameters.append({"name": "@key_terms", "value": terms})

        partition_key, cross_partition = query_scope(user_id, thread_id)
        logger.debug("MemoryStore.search query: %s", sql)
        return self.query(
            sql,
            parameters,
            partition_key=partition_key,
            cross_partition=cross_partition,
        )

    def search_episodic(
        self,
        user_id: str,
        search_terms: str,
        top_k: int = 5,
        min_salience: Optional[float] = None,
        include_superseded: bool = False,
    ) -> list[dict[str, Any]]:
        """Semantic search across episodic memories for a user."""
        return self.search(
            search_terms=search_terms,
            user_id=user_id,
            memory_types=["episodic"],
            top_k=top_k,
            min_salience=min_salience,
            include_superseded=include_superseded,
        )

    def build_episodic_context(self, user_id: str, query: str, top_k: int = 3) -> str:
        """Build formatted context of relevant past experiences."""
        memories = self.search_episodic(user_id, query, top_k=top_k)
        return format_episodic_context(memories)

    def _embed(self, text: str) -> list[float]:
        if self._embeddings_client is None:
            raise ConfigurationError(
                "An embeddings_client is required for retrieval search",
                parameter="embeddings_client",
            )
        for method_name in ("generate", "embed_one"):
            method = getattr(self._embeddings_client, method_name, None)
            if callable(method):
                return coerce_embedding(method(text))
        raise ConfigurationError(
            "embeddings_client must expose generate or embed_one",
            parameter="embeddings_client",
        )
