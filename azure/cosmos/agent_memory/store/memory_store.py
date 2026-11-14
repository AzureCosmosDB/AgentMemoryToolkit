"""Synchronous Cosmos DB memory store primitives."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Optional

from azure.cosmos.agent_memory._container_routing import (
    _CONTAINER_FOR_TYPE,
    USER_SCOPED_MEMORIES_TYPES,
    ContainerKey,
    container_key_for_type,
)
from azure.cosmos.agent_memory._curation import (
    drop_secret_tags,
    embedding_text,
    mark_secret,
    sanitize_record,
    sanitize_text,
)
from azure.cosmos.agent_memory._query_builder import _QueryBuilder
from azure.cosmos.agent_memory._utils import (
    _build_memory_query_builder,
    _coerce_datetime_iso,
    compute_content_hash,
    extract_keywords,
    new_id,
    normalize_created_at_iso,
)
from azure.cosmos.agent_memory.exceptions import (
    ConfigurationError,
    CosmosOperationError,
    MemoryConflictError,
    MemoryNotFoundError,
    MemoryTypeMismatchError,
    ValidationError,
)
from azure.cosmos.agent_memory.logging import describe_exception, get_logger
from azure.cosmos.agent_memory.models import MemoryRecord
from azure.cosmos.agent_memory.services._absorbed import ABSORBING_TYPES, delete_absorbed
from azure.cosmos.agent_memory.store._search_helpers import (
    FULL_TEXT_SCORE_SCOPE,
    MEMORY_PROJECTION,
    add_result_filters,
    add_salience_filter,
    add_tag_filters,
    build_search_sql,
    coerce_embedding,
    format_episodic_context,
    query_scope,
    require_search_terms,
    top_literal,
    uses_full_text_score,
)
from azure.cosmos.agent_memory.thresholds import default_ttl_for

logger = get_logger(__name__)

_MEMORIES_TYPES: tuple[str, ...] = ("fact", "episodic", "procedural")

# Explicit turn-document projection used by get_thread(). The raw conversation
# log is the only place embeddings are stored on turns (when
# enable_turn_embeddings=True), so we project every turn field *except*
# ``embedding`` to keep the vector off the wire and out of the result.
_TURN_PROJECTION_FIELDS: tuple[str, ...] = (
    "id",
    "user_id",
    "thread_id",
    "role",
    "type",
    "content",
    "metadata",
    "created_at",
    "tags",
    "ttl",
)
_TURN_PROJECTION: str = ", ".join(f"c.{field}" for field in _TURN_PROJECTION_FIELDS)


def _validated_memories_types(memory_types: Optional[list[str]]) -> list[str]:
    types = list(memory_types) if memory_types else list(_MEMORIES_TYPES)
    invalid = [t for t in types if t not in _MEMORIES_TYPES]
    if invalid:
        raise ValueError(f"memory_types must be a subset of {list(_MEMORIES_TYPES)}; got {invalid}")
    return types


def _validate_taggable_type(memory_type: str) -> None:
    if memory_type not in _MEMORIES_TYPES:
        raise ValueError(f"memory_type for tag mutation must be one of {list(_MEMORIES_TYPES)}; got {memory_type!r}")


def _wrap_cosmos_exception(exc: BaseException, *, message: str) -> CosmosOperationError:
    """Wrap a Cosmos SDK exception with a contextual message."""
    return CosmosOperationError(message)


class MemoryStore:
    """Typed CRUD and query primitives over the Cosmos DB containers."""

    def __init__(
        self,
        *,
        containers: dict[ContainerKey, Any],
        embeddings_client: Any = None,
        enable_turn_embeddings: bool = False,
    ) -> None:
        self._containers = containers
        self._turns_container = containers[ContainerKey.TURNS]
        self._memories_container = containers[ContainerKey.MEMORIES]
        self._summaries_container = containers[ContainerKey.SUMMARIES]
        self._embeddings_client = embeddings_client
        self._enable_turn_embeddings = enable_turn_embeddings

    @property
    def container(self) -> Any:
        """Return the memories Cosmos container client."""
        return self._memories_container

    def _prepare_doc(self, doc: dict[str, Any]) -> dict[str, Any]:
        """Return a write-ready document: literal credentials redacted and type defaults applied."""
        body = dict(doc)
        redacted = sanitize_record(body)
        if redacted:
            logger.warning("redacted literal secrets before write id=%s patterns=%s", body.get("id"), redacted)
        if body.get("ttl") is None:
            body.pop("ttl", None)
            ttl = default_ttl_for(body.get("type"))
            if ttl is not None:
                body["ttl"] = ttl
        return body

    def _container_for_type(self, memory_type: str) -> Any:
        """Return the container that owns documents of ``memory_type``."""
        return self._containers[container_key_for_type(memory_type)]

    def _embed_for_write(self, record_id: Any, text: str, *, operation: str) -> Optional[list[float]]:
        """Return the vector of ``text``; ``None`` when there is no text or embeddings client, or embedding fails."""
        if not text or self._embeddings_client is None:
            return None
        try:
            return self._embeddings_client.generate(text)
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "%s: embedding generation failed for %s (%s); proceeding without embedding",
                operation,
                record_id,
                describe_exception(exc),
            )
            return None

    def _refresh_embedding(self, doc: dict[str, Any], built_from: str, vectors: dict[str, list[float]]) -> None:
        """Keep the vector of an updated ``doc`` in step with its text.

        ``built_from`` is the text the stored vector encodes. The write guard drops the vector when it redacts that
        text, and a changed text leaves it stale. Either way it is rebuilt from the text now stored, or left off the
        record when that cannot be done. ``vectors`` remembers what was embedded, so a retry does not embed it again.
        """
        text = embedding_text(doc)
        if doc.get("embedding") and text == built_from:
            return
        doc.pop("embedding", None)
        if text not in vectors:
            vector = self._embed_for_write(doc.get("id"), text, operation="update")
            if vector is None:
                return
            vectors[text] = vector
        doc["embedding"] = vectors[text]

    def read_item(self, item_id: str, partition_key: Any, *, container_key: ContainerKey) -> dict[str, Any]:
        """Point-read a document from the explicitly selected split container."""
        from azure.cosmos.exceptions import CosmosResourceNotFoundError

        try:
            return self._containers[container_key].read_item(item=item_id, partition_key=partition_key)
        except CosmosResourceNotFoundError as exc:
            raise MemoryNotFoundError(memory_id=item_id) from exc
        except Exception as exc:
            raise CosmosOperationError(f"read_item failed for {item_id}: {exc}") from exc

    def query(
        self,
        sql: str,
        parameters: Optional[list[dict[str, Any]]] = None,
        *,
        container_key: ContainerKey,
        partition_key: Any = None,
        cross_partition: bool = False,
        full_text_score_scope: Optional[str] = None,
    ) -> list[dict[str, Any]]:
        """Run a query against the explicitly selected split container.

        ``partition_key`` reads one partition; ``cross_partition`` lets the query fan out across
        partitions. A query that calls ``FullTextScore`` raises ``ValidationError`` when it is sent
        with a ``partition_key``: the Python SDK skips query planning for such a query, so the
        ranking would be ignored (Azure/azure-sdk-for-python#42241). Filter on the partition key
        fields in the ``WHERE`` clause instead (see ``query_scope``). ``full_text_score_scope``
        (``"Local"`` or ``"Global"``) sets where the keyword ranking takes its BM25 statistics from.
        """
        return self._query_items(
            query=sql,
            parameters=parameters,
            partition_key=partition_key,
            cross_partition=cross_partition,
            full_text_score_scope=full_text_score_scope,
            operation="query",
            container=self._containers[container_key],
        )

    def _query_items(
        self,
        *,
        query: str,
        parameters: Optional[list[dict[str, Any]]] = None,
        partition_key: Any = None,
        cross_partition: bool = False,
        full_text_score_scope: Optional[str] = None,
        operation: str,
        container: Any = None,
    ) -> list[dict[str, Any]]:
        if partition_key is not None and uses_full_text_score(query):
            raise ValidationError(
                "The Python SDK runs a query sent with a partition key without planning it, so FullTextScore "
                "ranking is ignored (Azure/azure-sdk-for-python#42241); filter on the partition key fields in the "
                "WHERE clause instead"
            )
        kwargs: dict[str, Any] = {"query": query, "parameters": parameters or None}
        if partition_key is not None:
            kwargs["partition_key"] = partition_key
        if cross_partition:
            kwargs["enable_cross_partition_query"] = True
        if full_text_score_scope is not None:
            kwargs["full_text_score_scope"] = full_text_score_scope
        target = container if container is not None else self._memories_container
        try:
            return list(target.query_items(**kwargs))
        except Exception as exc:
            raise CosmosOperationError(f"{operation} failed: {exc}") from exc

    def upsert_memory(self, record: dict[str, Any]) -> dict[str, Any]:
        """Upsert a pre-built Cosmos memory document and return the stored body.

        When the write guard redacts the text of a record that carries a vector, the vector is rebuilt from the
        redacted text; without an embeddings client (or if embedding fails) the record is stored without one.
        """
        body = self._prepare_doc(record)
        memory_type = body.get("type")
        if memory_type not in _CONTAINER_FOR_TYPE:
            raise ValueError(
                f"upsert_memory: record id={body.get('id')!r} has invalid type={memory_type!r}. "
                f"Set 'type' to one of {sorted(_CONTAINER_FOR_TYPE)} before calling upsert_memory."
            )
        if record.get("embedding") and not body.get("embedding"):
            vector = self._embed_for_write(body.get("id"), embedding_text(body), operation="upsert_memory")
            if vector is not None:
                body["embedding"] = vector
        container = self._container_for_type(memory_type)
        try:
            response = container.upsert_item(body=body)
        except Exception as exc:
            raise _wrap_cosmos_exception(
                exc, message=f"upsert_memory upsert failed for record {body.get('id')}: {exc}"
            ) from exc
        logger.info("upsert_memory id=%s role=%s type=%s", body.get("id"), body.get("role"), body.get("type"))
        return response if isinstance(response, dict) else body

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
        created_at: Optional[str | datetime] = None,
    ) -> str:
        """Add a memory document to Cosmos DB and return its id.

        Literal credentials are redacted before the memory is embedded or written, and the record is
        then marked with the ``sys:secret`` tag and ``metadata.classification="secret"``. It is
        embedded unless ``embed`` is False; ``embed`` defaults to on for every type but turns, which
        follow ``enable_turn_embeddings``. A supplied ``embedding`` is kept, unless it was built from
        text that held a credential.
        """
        content, secret_patterns = sanitize_text(content)
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
        if created_at is not None:
            kwargs["created_at"] = normalize_created_at_iso(created_at)
        if memory_type != "turn":
            kwargs.setdefault("content_hash", compute_content_hash(content))
            kwargs.setdefault("prompt_id", "manual:add")
            kwargs.setdefault("id", new_id(memory_type))
            meta = kwargs.get("metadata") or {}
            if memory_type == "fact":
                meta.setdefault("category", "unclassified:manual")
            elif memory_type == "episodic":
                kwargs.setdefault("title", content[:80] or "Manual episode")
                kwargs.setdefault("events", [])
                kwargs.setdefault("participants", [])
                kwargs.setdefault("lessons", [])
                kwargs.setdefault("source_turn_ids", [])
            elif memory_type == "procedural":
                kwargs.setdefault("source_fact_ids", ["manual"])
            kwargs["metadata"] = meta
        record = MemoryRecord(**kwargs)
        body = record.to_cosmos_dict()

        if secret_patterns and embedding is not None:
            logger.warning(
                "upsert_memory: discarding the supplied embedding for %s; it encodes text that held a literal secret",
                record.id,
            )
            embedding = None
            embed = embed is not False
        if embed is None:
            embed = memory_type != "turn" or self._enable_turn_embeddings
        if embedding is not None:
            body["embedding"] = embedding
        elif embed and content and self._embeddings_client is not None:
            try:
                body["embedding"] = self._embeddings_client.generate(content)
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "upsert_memory: embedding generation failed for %s (%s); proceeding without embedding",
                    record.id,
                    describe_exception(exc),
                )

        mark_secret(body, secret_patterns)
        body = self._prepare_doc(body)
        try:
            container = self._container_for_type(memory_type)
            container.upsert_item(body=body)
        except Exception as exc:
            raise _wrap_cosmos_exception(exc, message=f"Upsert failed for record {record.id}: {exc}") from exc
        logger.info("upsert_memory id=%s role=%s type=%s", record.id, role, memory_type)
        return record.id

    def push(self, local_memory: list[dict[str, Any]], batch_size: int = 25) -> None:
        """Upsert all local memory records to Cosmos DB in sequential batches.

        Each batch has literal credentials redacted, is embedded (turns only when
        ``enable_turn_embeddings`` is on), then written; a batch that cannot be embedded is written
        without vectors. ``local_memory`` keeps every record, each gaining the vector generated for it.
        """
        if batch_size <= 0:
            raise ValueError("batch_size must be greater than 0")
        logger.info(
            "push_to_cosmos count=%d batch_size=%d",
            len(local_memory),
            batch_size,
        )
        records = [dict(m) for m in local_memory]
        for start in range(0, len(records), batch_size):
            batch = records[start : start + batch_size]
            # Redact before embedding: no raw credential may reach the embedding provider, and a
            # supplied vector built from redacted text is dropped here and rebuilt below.
            bodies = [self._prepare_doc(dict(r)) for r in batch]

            to_embed_idx: list[int] = []
            to_embed_text: list[str] = []
            for i, body in enumerate(bodies):
                embeddable_type = body.get("type") != "turn" or self._enable_turn_embeddings
                if embeddable_type and body.get("content") and not body.get("embedding"):
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
                        describe_exception(exc),
                        len(to_embed_text),
                    )

            for body in bodies:
                memory_type = body.get("type")
                if memory_type not in _CONTAINER_FOR_TYPE:
                    raise ValueError(
                        f"push: record id={body.get('id')!r} has invalid type={memory_type!r}. "
                        f"Set 'type' to one of {sorted(_CONTAINER_FOR_TYPE)} on every local "
                        f"memory before calling push_to_cosmos."
                    )
            for record, body in zip(batch, bodies):
                container = self._container_for_type(body.get("type"))
                try:
                    container.upsert_item(body=body)
                except Exception as exc:
                    record_id = record.get("id") if isinstance(record, dict) else getattr(record, "id", None)
                    raise _wrap_cosmos_exception(exc, message=f"Upsert failed for record {record_id}: {exc}") from exc
        logger.info("Upserted batch of %d records", len(records))

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
        """Retrieve memories from the MEMORIES container with optional filters."""
        types = _validated_memories_types(memory_types)
        logger.debug(
            "get_memories filters: memory_id=%s user_id=%s thread_id=%s role=%s types=%s recent_k=%s",
            memory_id,
            user_id,
            thread_id,
            role,
            types,
            recent_k,
        )

        qb = _build_memory_query_builder(
            memory_id=memory_id,
            user_id=user_id,
            thread_id=thread_id,
            role=role,
            memory_types=types,
            min_confidence=min_confidence,
        )

        add_tag_filters(qb, tags_all=tags_all, tags_any=tags_any, exclude_tags=exclude_tags)
        qb.add_time_range(
            "c.created_at",
            after=_coerce_datetime_iso(created_after),
            before=_coerce_datetime_iso(created_before),
            after_param="@created_after",
            before_param="@created_before",
        )
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
            container=self._memories_container,
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
        *,
        user_id: str,
        thread_id: str,
        memory_type: str,
        content: Optional[str] = None,
        role: Optional[str] = None,
        metadata: Optional[dict[str, Any]] = None,
    ) -> None:
        """Update a memory document via point read in the container for ``memory_type``.

        The write guard redacts credentials in the stored document, which drops its vector. A vector is also stale
        once the text it was built from changes. Either way it is rebuilt from the redacted text, or left off the
        record when the store has no embeddings client or embedding fails.
        """
        import random
        import time

        from azure.core import MatchConditions
        from azure.cosmos.exceptions import (
            CosmosAccessConditionFailedError,
            CosmosResourceNotFoundError,
        )

        container = self._container_for_type(memory_type)
        max_attempts = 5
        attempts = 0
        vectors: dict[str, list[float]] = {}
        while True:
            try:
                doc = container.read_item(item=memory_id, partition_key=[user_id, thread_id])
            except CosmosResourceNotFoundError as exc:
                raise MemoryNotFoundError(memory_id=memory_id, user_id=user_id, thread_id=thread_id) from exc
            except Exception as exc:
                raise _wrap_cosmos_exception(exc, message=f"update read failed for {memory_id}: {exc}") from exc

            actual_type = doc.get("type")
            if actual_type != memory_type:
                raise MemoryTypeMismatchError(memory_id=memory_id, expected=memory_type, actual=actual_type)

            built_from = embedding_text(doc) if doc.get("embedding") else None
            if content is not None:
                doc["content"] = content
            if role is not None:
                doc["role"] = role
            if metadata is not None:
                doc["metadata"] = metadata
            redacted = sanitize_record(doc)
            if redacted:
                logger.warning("redacted literal secrets before write id=%s patterns=%s", memory_id, redacted)
            if built_from is not None:
                self._refresh_embedding(doc, built_from, vectors)
            doc["updated_at"] = datetime.now(timezone.utc).isoformat()

            kwargs: dict[str, Any] = {"item": doc["id"], "body": doc}
            if etag := doc.get("_etag"):
                kwargs.update(match_condition=MatchConditions.IfNotModified, etag=etag)
            try:
                container.replace_item(**kwargs)
                logger.info("Updated record %s", memory_id)
                return
            except CosmosAccessConditionFailedError as exc:
                attempts += 1
                if attempts >= max_attempts:
                    raise MemoryConflictError(
                        f"update conflicted after {max_attempts} attempts for memory_id={memory_id!r}"
                    ) from exc
                base = 0.02 * (2 ** (attempts - 1))
                time.sleep(base + random.uniform(0, base))
            except Exception as exc:
                raise _wrap_cosmos_exception(exc, message=f"update replace failed for {memory_id}: {exc}") from exc

    def delete(
        self,
        memory_id: str,
        *,
        user_id: str,
        thread_id: str,
        memory_type: str,
    ) -> None:
        """Delete a memory document from the container for ``memory_type``.

        Reads the doc first to verify its ``type`` matches ``memory_type`` and
        then issues the delete with ``If-Match`` on the read ETag, so a
        concurrent type mutation between read and delete is rejected (412)
        rather than silently dropping the wrong document. The records retired
        into a fact or procedure (duplicates, the sources of a combined fact
        and, for a procedure, the directives and corrections it absorbed - and
        anything retired into those; see ``services._absorbed``) are deleted
        first.
        """
        from azure.core import MatchConditions
        from azure.cosmos.exceptions import (
            CosmosAccessConditionFailedError,
            CosmosResourceNotFoundError,
        )

        container = self._container_for_type(memory_type)
        try:
            doc = container.read_item(item=memory_id, partition_key=[user_id, thread_id])
        except CosmosResourceNotFoundError as exc:
            raise MemoryNotFoundError(memory_id=memory_id, user_id=user_id, thread_id=thread_id) from exc
        except Exception as exc:
            raise _wrap_cosmos_exception(exc, message=f"delete read failed for {memory_id}: {exc}") from exc

        actual_type = doc.get("type")
        if actual_type != memory_type:
            raise MemoryTypeMismatchError(memory_id=memory_id, expected=memory_type, actual=actual_type)

        absorbed: list[dict[str, Any]] = []
        if memory_type in ABSORBING_TYPES:
            try:
                absorbed = delete_absorbed(container, user_id, memory_id)
            except Exception as exc:
                raise _wrap_cosmos_exception(
                    exc, message=f"delete of the records folded into {memory_id} failed: {exc}"
                ) from exc

        kwargs: dict[str, Any] = {"item": memory_id, "partition_key": [user_id, thread_id]}
        if etag := doc.get("_etag"):
            kwargs.update(match_condition=MatchConditions.IfNotModified, etag=etag)
        try:
            container.delete_item(**kwargs)
        except CosmosResourceNotFoundError as exc:
            raise MemoryNotFoundError(memory_id=memory_id, user_id=user_id, thread_id=thread_id) from exc
        except CosmosAccessConditionFailedError as exc:
            raise MemoryConflictError(
                f"delete conflicted for memory_id={memory_id!r} - document was modified after the type check"
            ) from exc
        except Exception as exc:
            raise _wrap_cosmos_exception(exc, message=f"delete failed for {memory_id}: {exc}") from exc

        logger.info("Deleted record %s and %d records folded into it", memory_id, len(absorbed))

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
        """Retrieve an entire thread (turns) sorted oldest first."""
        qb = _QueryBuilder()
        qb.add_filter("c.thread_id", "@thread_id", thread_id)
        qb.add_filter("c.user_id", "@user_id", user_id)
        add_tag_filters(qb, tags_all=tags_all, tags_any=tags_any, exclude_tags=exclude_tags)
        qb.add_time_range(
            "c.created_at",
            after=_coerce_datetime_iso(created_after),
            before=_coerce_datetime_iso(created_before),
            after_param="@created_after",
            before_param="@created_before",
        )
        if not include_superseded:
            qb.add_is_null_or_undefined("c.superseded_by")

        query = f"SELECT {_TURN_PROJECTION} FROM c{qb.build_where()} ORDER BY c.created_at DESC"
        logger.debug("get_thread query: %s", query)
        items = self._query_items(
            query=query,
            parameters=qb.get_parameters(),
            cross_partition=True,
            operation="get_thread query",
            container=self._turns_container,
        )
        if recent_k is not None:
            items = items[:recent_k]
        items.reverse()
        return items

    def get_thread_summary(
        self,
        user_id: str,
        thread_id: str,
        recent_k: Optional[int] = None,
    ) -> list[dict[str, Any]]:
        """Retrieve active thread summaries for ``(user_id, thread_id)``, newest first."""
        qb = _QueryBuilder()
        qb.add_filter("c.type", "@type", "thread_summary")
        qb.add_filter("c.user_id", "@user_id", user_id)
        qb.add_filter("c.thread_id", "@thread_id", thread_id)
        qb.add_is_null_or_undefined("c.superseded_by")
        parameters = qb.get_parameters()
        if recent_k is not None:
            parameters.append({"name": "@recent_k", "value": recent_k})
            sql = f"SELECT TOP @recent_k * FROM c{qb.build_where()} ORDER BY c.created_at DESC"
        else:
            sql = f"SELECT * FROM c{qb.build_where()} ORDER BY c.created_at DESC"
        return self._query_items(
            query=sql,
            parameters=parameters,
            partition_key=[user_id, thread_id],
            operation="get_thread_summary query",
            container=self._summaries_container,
        )

    def get_episodes(
        self,
        user_id: str,
        thread_id: Optional[str] = None,
        recent_k: Optional[int] = None,
    ) -> list[dict[str, Any]]:
        """Retrieve active episodic memories for ``user_id``, newest first."""
        if not user_id:
            raise ValidationError("user_id is required for get_episodes")
        qb = _QueryBuilder()
        qb.add_filter("c.type", "@type", "episodic")
        qb.add_filter("c.user_id", "@user_id", user_id)
        qb.add_filter("c.thread_id", "@thread_id", thread_id)
        qb.add_is_null_or_undefined("c.superseded_by")
        parameters = qb.get_parameters()
        if recent_k is not None:
            parameters.append({"name": "@recent_k", "value": recent_k})
            sql = f"SELECT TOP @recent_k * FROM c{qb.build_where()} ORDER BY c.created_at DESC"
        else:
            sql = f"SELECT * FROM c{qb.build_where()} ORDER BY c.created_at DESC"
        partition_key, cross_partition = query_scope(user_id, thread_id)
        return self.query(
            sql,
            parameters,
            container_key=ContainerKey.MEMORIES,
            partition_key=partition_key,
            cross_partition=cross_partition,
        )

    def get_user_summary(self, user_id: str) -> Optional[dict[str, Any]]:
        """Retrieve the user's summary document from Cosmos DB, or ``None`` if absent."""
        from azure.cosmos.exceptions import CosmosResourceNotFoundError

        try:
            return self._summaries_container.read_item(
                item=f"user_summary_{user_id}",
                partition_key=[user_id, "__user_summary__"],
            )
        except CosmosResourceNotFoundError:
            return None
        except Exception as exc:
            raise CosmosOperationError(f"get_user_summary read failed: {exc}") from exc

    def list_tags(
        self,
        user_id: str,
        *,
        thread_id: Optional[str] = None,
        prefix: Optional[str] = None,
        include_sys: bool = False,
        include_superseded: bool = False,
    ) -> list[str]:
        """Return sorted distinct tags for a user from the MEMORIES container."""
        query = "SELECT VALUE c.tags FROM c WHERE c.user_id = @user_id AND ARRAY_LENGTH(c.tags) > 0"
        parameters = [{"name": "@user_id", "value": user_id}]
        if thread_id is not None:
            query += " AND c.thread_id = @thread_id"
            parameters.append({"name": "@thread_id", "value": thread_id})
        if not include_superseded:
            query += " AND (NOT IS_DEFINED(c.superseded_by) OR IS_NULL(c.superseded_by))"

        prefix_norm = prefix.strip().lower() if prefix else None
        partition_key, cross_partition = query_scope(user_id, thread_id)
        rows = self._query_items(
            query=query,
            parameters=parameters,
            partition_key=partition_key,
            cross_partition=cross_partition,
            operation="list_tags query",
            container=self._memories_container,
        )
        tags: set[str] = set()
        for row in rows:
            values = row.get("tags", []) if isinstance(row, dict) else row
            for tag in values or []:
                tag_value = str(tag).strip().lower()
                if not tag_value:
                    continue
                if not include_sys and tag_value.startswith("sys:"):
                    continue
                if prefix_norm is not None and not tag_value.startswith(prefix_norm):
                    continue
                tags.add(tag_value)
        return sorted(tags)

    def _mutate_tags(
        self,
        memory_id: str,
        user_id: str,
        thread_id: str,
        memory_type: str,
        tags: list[str],
        *,
        add: bool,
    ) -> None:
        import random
        import time

        from azure.core import MatchConditions
        from azure.cosmos.exceptions import (
            CosmosAccessConditionFailedError,
            CosmosResourceNotFoundError,
        )

        _validate_taggable_type(memory_type)
        container = self._container_for_type(memory_type)
        secret_patterns: list[str] = []
        if add:
            tags, secret_patterns = drop_secret_tags(tags)
            if secret_patterns:
                logger.warning("redacted literal secrets before write id=%s patterns=%s", memory_id, secret_patterns)
        normalized = {t.strip().lower() for t in tags if t and t.strip()}
        max_attempts = 5
        attempts = 0
        while True:
            try:
                doc = container.read_item(item=memory_id, partition_key=[user_id, thread_id])
            except CosmosResourceNotFoundError as exc:
                raise MemoryNotFoundError(memory_id=memory_id, user_id=user_id, thread_id=thread_id) from exc
            mark_secret(doc, secret_patterns)
            existing_tags = set(doc.get("tags", []))
            if add:
                existing_tags.update(normalized)
            else:
                existing_tags.difference_update(normalized)
            doc["tags"] = sorted(existing_tags)
            doc["updated_at"] = datetime.now(timezone.utc).isoformat()

            kwargs: dict[str, Any] = {"item": memory_id, "body": doc}
            if etag := doc.get("_etag"):
                kwargs.update(match_condition=MatchConditions.IfNotModified, etag=etag)
            try:
                container.replace_item(**kwargs)
                return
            except CosmosAccessConditionFailedError as exc:
                attempts += 1
                if attempts >= max_attempts:
                    raise MemoryConflictError(
                        f"Tag update conflicted after {max_attempts} attempts for memory_id={memory_id!r}"
                    ) from exc
                base = 0.02 * (2 ** (attempts - 1))
                time.sleep(base + random.uniform(0, base))

    def add_tags(
        self,
        memory_id: str,
        user_id: str,
        thread_id: str,
        memory_type: str,
        tags: list[str],
    ) -> None:
        """Add tags to an existing fact, episodic or procedural memory.

        Tags are lower-cased. A tag that holds a literal credential is dropped, and the memory is
        marked ``sys:secret``.
        """
        self._mutate_tags(memory_id, user_id, thread_id, memory_type, tags, add=True)

    def remove_tags(
        self,
        memory_id: str,
        user_id: str,
        thread_id: str,
        memory_type: str,
        tags: list[str],
    ) -> None:
        """Remove tags from an existing fact, episodic or procedural memory."""
        self._mutate_tags(memory_id, user_id, thread_id, memory_type, tags, add=False)

    def mark_superseded(
        self,
        old_doc: dict[str, Any],
        superseder_id: str,
        *,
        reason: str,
    ) -> bool:
        """Set supersession audit fields using ETag protection when available."""
        from azure.core import MatchConditions
        from azure.cosmos.exceptions import (
            CosmosAccessConditionFailedError,
            CosmosHttpResponseError,
        )

        etag = old_doc.get("_etag")
        new_doc = {
            **old_doc,
            "superseded_by": superseder_id,
            "supersede_reason": reason,
            "superseded_at": datetime.now(timezone.utc).isoformat(),
        }
        target_container = self._container_for_type(old_doc.get("type"))
        try:
            if etag:
                target_container.replace_item(
                    item=new_doc["id"],
                    body=new_doc,
                    match_condition=MatchConditions.IfNotModified,
                    etag=etag,
                )
            else:
                target_container.upsert_item(body=new_doc)
            return True
        except CosmosAccessConditionFailedError as exc:
            logger.warning(
                "supersede skipped (concurrent writer won) id=%s superseder=%s",
                old_doc.get("id"),
                superseder_id,
                extra={"operation": "mark_superseded"},
            )
            del exc
            return False
        except CosmosHttpResponseError as exc:
            logger.warning(
                "supersede failed id=%s superseder=%s status=%s: %s",
                old_doc.get("id"),
                superseder_id,
                getattr(exc, "status_code", None),
                describe_exception(exc),
            )
            return False

    def get_procedural_prompt(self, user_id: str) -> Optional[str]:
        """Return the ``content`` of the user's highest-``version`` procedure that is not retired, or None.

        That is one procedure's rule text (candidates included), not the compiled standing
        instructions: ``CosmosMemoryClient.get_procedural_prompt`` compiles those from the active
        standing rules.
        """
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
            container=self._memories_container,
        )
        if not items:
            return None
        return items[0].get("content")

    def get_procedural_history(self, user_id: str, limit: int = 10) -> list[dict[str, Any]]:
        """Return up to ``limit`` of the user's procedures, retired (superseded) ones included.

        Procedures still in place come first and retired ones last. Within each group the highest
        ``version`` comes first, then the most recently written; ``status`` plays no part, so
        candidates are mixed in with active procedures.
        """
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
            container=self._memories_container,
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
        """Retrieve the user's procedural memories, candidates included, newest first by ``created_at``.

        Retired (superseded) procedures are left out unless ``include_superseded`` is True.
        ``priority``, ``category`` and ``min_salience`` match ``metadata.priority``,
        ``metadata.category`` and ``salience``, which procedures learned by
        ``synthesize_procedural`` do not carry, so setting one leaves them out.
        """
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
            container=self._memories_container,
        )

        if min_salience is not None:
            items = [i for i in items if (i.get("salience") or 0.0) >= min_salience]
        if priority is not None:
            items = [i for i in items if i.get("metadata", {}).get("priority") == priority]
        if category is not None:
            items = [i for i in items if i.get("metadata", {}).get("category") == category]
        return items

    def search(
        self,
        search_terms: Optional[str] = None,
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
        *,
        query: Optional[str] = None,
        active_procedures_only: bool = False,
        categories: Optional[list[str]] = None,
        applies: Optional[str] = None,
        vector_only: bool = False,
        query_embedding: Optional[list[float]] = None,
    ) -> list[dict[str, Any]]:
        """Search memories using vector similarity with optional full-text hybrid ranking.

        Searches the derived memories container (facts/episodic/procedural). Use
        :meth:`search_turns` to vector-search the raw conversation log instead.

        ``active_procedures_only`` hides procedures that are not active or are past their
        ``valid_until`` (user-facing search sets it). ``categories`` keeps facts of those
        ``metadata.category`` values, ``applies`` procedures with that ``applies`` value, and
        ``vector_only`` ranks by vector similarity alone, without the keyword ranking.
        ``query_embedding`` is the query's embedding when the caller already has it.

        ``thread_id`` limits facts to that thread; episodic and procedural memories stay user-wide.
        A keyword query, or a thread-scoped search that includes those user-wide types, runs across
        partitions with ``user_id`` and ``thread_id`` filters in the WHERE clause (keyword queries
        also set ``full_text_score_scope="Local"``); a vector-only query over facts with both ids
        reads one partition (see ``query_scope``).
        """
        terms = require_search_terms(search_terms, query)
        top = top_literal(top_k, name="top_k")
        query_vector = query_embedding if query_embedding is not None else self._embed(terms)
        keywords = [] if vector_only else extract_keywords(terms)

        qb = _build_memory_query_builder(
            memory_id=memory_id,
            user_id=user_id,
            role=role,
            memory_types=memory_types,
            thread_id=thread_id,
            min_confidence=min_confidence,
        )
        add_tag_filters(qb, tags_all=tags_all, tags_any=tags_any, exclude_tags=exclude_tags)
        qb.add_time_range(
            "c.created_at",
            after=_coerce_datetime_iso(created_after),
            before=_coerce_datetime_iso(created_before),
            after_param="@created_after",
            before_param="@created_before",
        )
        add_salience_filter(qb, min_salience)
        add_result_filters(
            qb,
            # The procedure filter matters only where procedures can match.
            active_procedures_only=active_procedures_only and (not memory_types or "procedural" in memory_types),
            categories=categories,
            applies=applies,
        )

        sql = build_search_sql(
            qb=qb,
            top=top,
            keyword_count=len(keywords),
            include_superseded=include_superseded,
        )
        parameters = qb.get_parameters()
        parameters.append({"name": "@embedding", "value": query_vector})
        for i, kw in enumerate(keywords):
            parameters.append({"name": f"@kw{i}", "value": kw})

        partition_key, cross_partition = query_scope(user_id, thread_id, len(keywords))
        if thread_id is not None and (not memory_types or set(memory_types) & USER_SCOPED_MEMORIES_TYPES):
            partition_key, cross_partition = None, True
        logger.debug("MemoryStore.search query: %s", sql)
        return self.query(
            sql,
            parameters,
            container_key=ContainerKey.MEMORIES,
            partition_key=partition_key,
            cross_partition=cross_partition,
            full_text_score_scope=FULL_TEXT_SCORE_SCOPE if keywords else None,
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

        AMT supersedes rather than deletes: when a fact is contradicted by a newer one, folded into
        a duplicate or combined fact, or retired into a procedure, the prior document is retained
        with ``superseded_by`` pointing at its replacement (see :meth:`mark_superseded`). This walks
        that chain backwards from *memory_id* so callers can reason about how a
        fact evolved - knowledge updates, preference reversals, relocations -
        instead of seeing only the current value.

        The document identified by *memory_id* is not itself included; the return
        value is everything it superseded, transitively, bounded by *max_depth*
        to guard against cycles. Each entry carries the ``superseded_at`` /
        ``supersede_reason`` audit fields so callers can order and explain the
        transitions.
        """
        if not memory_id:
            raise ValidationError("memory_id is required")
        if not user_id:
            raise ValidationError("user_id is required")
        partition_key, cross_partition = query_scope(user_id, thread_id)
        history: list[dict[str, Any]] = []
        seen: set[str] = {memory_id}
        frontier: list[str] = [memory_id]
        for _ in range(max(1, max_depth)):
            id_params = [{"name": f"@sid{i}", "value": sid} for i, sid in enumerate(frontier)]
            placeholders = ", ".join(param["name"] for param in id_params)
            parameters: list[dict[str, Any]] = [*id_params, {"name": "@user_id", "value": user_id}]
            where = f"c.superseded_by IN ({placeholders}) AND c.user_id = @user_id"
            if thread_id is not None:
                where += " AND c.thread_id = @thread_id"
                parameters.append({"name": "@thread_id", "value": thread_id})
            sql = f"SELECT {MEMORY_PROJECTION} FROM c WHERE {where}"
            rows = self.query(
                sql,
                parameters,
                container_key=ContainerKey.MEMORIES,
                partition_key=partition_key,
                cross_partition=cross_partition,
            )
            frontier = []
            for doc in rows:
                doc_id = doc.get("id")
                if not doc_id or doc_id in seen:
                    continue
                seen.add(doc_id)
                history.append(doc)
                frontier.append(doc_id)
            if not frontier:
                break
        history.sort(key=lambda d: d.get("superseded_at") or d.get("created_at") or "", reverse=True)
        return history

    def search_turns(
        self,
        search_terms: Optional[str] = None,
        user_id: Optional[str] = None,
        thread_id: Optional[str] = None,
        role: Optional[str] = None,
        top_k: int = 5,
        tags_all: Optional[list[str]] = None,
        tags_any: Optional[list[str]] = None,
        exclude_tags: Optional[list[str]] = None,
        created_after: Optional[str | datetime] = None,
        created_before: Optional[str | datetime] = None,
        *,
        query: Optional[str] = None,
    ) -> list[dict[str, Any]]:
        """Search raw conversation turns using vector similarity with hybrid ranking.

        Only vector-searchable when turn embeddings were enabled at write time
        (see ``enable_turn_embeddings``). ``user_id`` is required and always
        filters the results. When ``thread_id`` is also supplied the results are
        restricted to that thread: a vector-only query targets the thread's
        partition, while a keyword (hybrid) query runs across partitions with
        ``user_id`` and ``thread_id`` filters in the WHERE clause (a partition key
        breaks full-text ranking, Azure SDK issue #42241). When it is omitted the
        query fans out across the user's partitions.

        """
        if not user_id:
            raise ValidationError("user_id is required for search_turns")
        terms = require_search_terms(search_terms, query)
        top = top_literal(top_k, name="top_k")
        query_vector = self._embed(terms)
        keywords = extract_keywords(terms)

        qb = _QueryBuilder()
        qb.add_filter("c.user_id", "@user_id", user_id)
        qb.add_filter("c.thread_id", "@thread_id", thread_id)
        qb.add_filter("c.role", "@role", role)
        add_tag_filters(qb, tags_all=tags_all, tags_any=tags_any, exclude_tags=exclude_tags)
        qb.add_time_range(
            "c.created_at",
            after=_coerce_datetime_iso(created_after),
            before=_coerce_datetime_iso(created_before),
            after_param="@created_after",
            before_param="@created_before",
        )

        sql = build_search_sql(qb=qb, top=top, keyword_count=len(keywords), include_superseded=False)
        parameters = qb.get_parameters()
        parameters.append({"name": "@embedding", "value": query_vector})
        for i, kw in enumerate(keywords):
            parameters.append({"name": f"@kw{i}", "value": kw})

        partition_key, cross_partition = query_scope(user_id, thread_id, len(keywords))
        logger.debug("MemoryStore.search_turns query: %s", sql)
        return self.query(
            sql,
            parameters,
            container_key=ContainerKey.TURNS,
            partition_key=partition_key,
            cross_partition=cross_partition,
            full_text_score_scope=FULL_TEXT_SCORE_SCOPE if keywords else None,
        )

    def search_summaries(
        self,
        search_terms: Optional[str] = None,
        user_id: Optional[str] = None,
        thread_id: Optional[str] = None,
        top_k: int = 5,
        tags_all: Optional[list[str]] = None,
        tags_any: Optional[list[str]] = None,
        exclude_tags: Optional[list[str]] = None,
        *,
        query: Optional[str] = None,
    ) -> list[dict[str, Any]]:
        """Search the summaries container (thread_summary + user_summary).

        Searches across *all* of the user's summary docs at once - the single
        user_summary and every per-thread thread_summary - ranked by relevance,
        so a holistic ("summarize everything"), a session-scoped, or a
        topic-scoped summary question all surface the right summary. Ranking is
        by vector similarity, with full-text relevance added when the query has
        keywords. ``user_id`` is required. Pass ``thread_id`` to narrow to one
        thread's summary; as with :meth:`search_turns`, a keyword query then runs
        across partitions with ``user_id`` and ``thread_id`` filters in the WHERE
        clause, and a vector-only query targets the thread's partition.
        """
        if not user_id:
            raise ValidationError("user_id is required for search_summaries")
        terms = require_search_terms(search_terms, query)
        top = top_literal(top_k, name="top_k")
        query_vector = self._embed(terms)
        keywords = extract_keywords(terms)

        qb = _QueryBuilder()
        qb.add_filter("c.user_id", "@user_id", user_id)
        qb.add_filter("c.thread_id", "@thread_id", thread_id)
        add_tag_filters(qb, tags_all=tags_all, tags_any=tags_any, exclude_tags=exclude_tags)

        sql = build_search_sql(qb=qb, top=top, keyword_count=len(keywords), include_superseded=False)
        parameters = qb.get_parameters()
        parameters.append({"name": "@embedding", "value": query_vector})
        for i, kw in enumerate(keywords):
            parameters.append({"name": f"@kw{i}", "value": kw})

        partition_key, cross_partition = query_scope(user_id, thread_id, len(keywords))
        logger.debug("MemoryStore.search_summaries query: %s", sql)
        return self.query(
            sql,
            parameters,
            container_key=ContainerKey.SUMMARIES,
            partition_key=partition_key,
            cross_partition=cross_partition,
            full_text_score_scope=FULL_TEXT_SCORE_SCOPE if keywords else None,
        )

    def search_episodic(
        self,
        user_id: str,
        search_terms: str,
        top_k: int = 5,
        min_salience: Optional[float] = None,
        include_superseded: bool = False,
        thread_id: Optional[str] = None,
        tags_all: Optional[list[str]] = None,
        tags_any: Optional[list[str]] = None,
        exclude_tags: Optional[list[str]] = None,
        created_after: Optional[str | datetime] = None,
        created_before: Optional[str | datetime] = None,
        started_after: Optional[str | datetime] = None,
        started_before: Optional[str | datetime] = None,
        ended_after: Optional[str | datetime] = None,
        ended_before: Optional[str | datetime] = None,
    ) -> list[dict[str, Any]]:
        """Semantic search across episodic memories for a user.

        Temporal arguments are filters only; relevance ranking is vector/FTS-only.
        """
        if not user_id:
            raise ValidationError("user_id is required for search_episodic")
        terms = require_search_terms(search_terms)
        top = top_literal(top_k, name="top_k")
        query_vector = self._embed(terms)
        keywords = extract_keywords(terms)

        qb = _QueryBuilder()
        qb.add_filter("c.type", "@type", "episodic")
        qb.add_filter("c.user_id", "@user_id", user_id)
        qb.add_filter("c.thread_id", "@thread_id", thread_id)
        add_tag_filters(qb, tags_all=tags_all, tags_any=tags_any, exclude_tags=exclude_tags)
        qb.add_time_range(
            "c.created_at",
            after=_coerce_datetime_iso(created_after),
            before=_coerce_datetime_iso(created_before),
            after_param="@created_after",
            before_param="@created_before",
        )
        qb.add_time_range(
            "c.started_at",
            after=_coerce_datetime_iso(started_after),
            before=_coerce_datetime_iso(started_before),
            after_param="@started_after",
            before_param="@started_before",
        )
        qb.add_time_range(
            "c.ended_at",
            after=_coerce_datetime_iso(ended_after),
            before=_coerce_datetime_iso(ended_before),
            after_param="@ended_after",
            before_param="@ended_before",
        )
        add_salience_filter(qb, min_salience)

        sql = build_search_sql(
            qb=qb,
            top=top,
            keyword_count=len(keywords),
            include_superseded=include_superseded,
        )
        parameters = qb.get_parameters()
        parameters.append({"name": "@embedding", "value": query_vector})
        for i, kw in enumerate(keywords):
            parameters.append({"name": f"@kw{i}", "value": kw})

        partition_key, cross_partition = query_scope(user_id, thread_id, len(keywords))
        logger.debug("MemoryStore.search_episodic query: %s", sql)
        return self.query(
            sql,
            parameters,
            container_key=ContainerKey.MEMORIES,
            partition_key=partition_key,
            cross_partition=cross_partition,
            full_text_score_scope=FULL_TEXT_SCORE_SCOPE if keywords else None,
        )

    def build_episodic_context(self, user_id: str, query: str, top_k: int = 3) -> str:
        """Build formatted context of relevant past experiences."""
        memories = self.search_episodic(user_id, query, top_k=top_k)
        return format_episodic_context(memories)

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
        """Search a user's procedural memories by relevance to ``search_terms``.

        Ranks by vector similarity, with full-text relevance added when the query has keywords.
        Only ``active`` procedures are returned unless ``status`` is changed (``None`` also returns
        candidates); ``procedure_kind`` narrows to ``instruction``, ``habit`` or ``skill``, and
        retired ones need ``include_superseded``. Procedures past ``valid_until`` are not filtered
        out here (:meth:`search` hides them with ``active_procedures_only``).
        """
        if not user_id:
            raise ValidationError("user_id is required for retrieve_procedures")
        terms = require_search_terms(search_terms)
        top = top_literal(top_k, name="top_k")
        query_vector = self._embed(terms)
        keywords = extract_keywords(terms)

        qb = _QueryBuilder()
        qb.add_filter("c.type", "@type", "procedural")
        qb.add_filter("c.user_id", "@user_id", user_id)
        qb.add_filter("c.procedure_kind", "@procedure_kind", procedure_kind)
        qb.add_filter("c.status", "@status", status)

        sql = build_search_sql(
            qb=qb,
            top=top,
            keyword_count=len(keywords),
            include_superseded=include_superseded,
        )
        parameters = qb.get_parameters()
        parameters.append({"name": "@embedding", "value": query_vector})
        for i, kw in enumerate(keywords):
            parameters.append({"name": f"@kw{i}", "value": kw})

        partition_key, cross_partition = query_scope(user_id, None)
        logger.debug("MemoryStore.retrieve_procedures query: %s", sql)
        return self.query(
            sql,
            parameters,
            container_key=ContainerKey.MEMORIES,
            partition_key=partition_key,
            cross_partition=cross_partition,
            full_text_score_scope=FULL_TEXT_SCORE_SCOPE if keywords else None,
        )

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
