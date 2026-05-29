"""Async pipeline service for LLM-driven memory extraction, summaries, and reconciliation.

This module is the asynchronous sibling of
:class:`agent_memory_toolkit.services.pipeline.PipelineService`. The two
share all pure helpers via
:mod:`agent_memory_toolkit.services._pipeline_helpers`; only the IO call
sites differ — every Cosmos query, chat completion, and embedding call is
``await``-ed against the async clients/store.
"""

from __future__ import annotations

import hashlib
import inspect
import json
import time
from collections import defaultdict
from datetime import datetime, timezone
from typing import Any, Literal, Optional

from azure.cosmos.exceptions import CosmosResourceExistsError, CosmosResourceNotFoundError

from agent_memory_toolkit._utils import DEFAULT_TTL_BY_TYPE, compute_content_hash
from agent_memory_toolkit.aio.store import AsyncMemoryStore
from agent_memory_toolkit.exceptions import (
    LLMError,
    MemoryConflictError,
    ValidationError,
)
from agent_memory_toolkit.logging import get_logger
from agent_memory_toolkit.models import (
    EpisodicRecord,
    FactRecord,
    ProceduralRecord,
    ThreadSummaryRecord,
    UserSummaryRecord,
    construct_internal,
)
from agent_memory_toolkit.prompts._schemas import response_format_for
from agent_memory_toolkit.services._pipeline_helpers import (
    ID_SEED_SEP as _ID_SEED_SEP,
)
from agent_memory_toolkit.services._pipeline_helpers import (
    VALID_VALENCES,
    PromptyLoader,
    build_topic_tags,
    build_transcript,
    cap_structured_summary,
    chat_text,
    coerce_valence,
    parse_llm_json,
)
from agent_memory_toolkit.services._pipeline_helpers import (
    is_real_number as _is_real_number,
)
from agent_memory_toolkit.services._pipeline_helpers import (
    max_or_none as _max_or_none,
)
from agent_memory_toolkit.store._search_helpers import top_literal

logger = get_logger("agent_memory_toolkit.pipeline.aio")


_coerce_valence = coerce_valence
_cap_structured_summary = cap_structured_summary

_ACTIVE_DOC_FILTER = "(NOT IS_DEFINED(c.superseded_by) OR IS_NULL(c.superseded_by))"
_PROCEDURAL_MAX_CREATE_ATTEMPTS = 5


class _AsyncStoreContainerAdapter:
    """Async equivalent of ``services.pipeline._StoreContainerAdapter``.

    Exposes the ``query_items``/``read_item``/``upsert_item`` shape over
    :class:`AsyncMemoryStore` so the orchestration body can match the sync
    path.
    """

    def __init__(self, store: AsyncMemoryStore) -> None:
        self._store = store

    async def query_items(self, **kwargs: Any) -> list[dict[str, Any]]:
        return await self._store.query(
            kwargs["query"],
            parameters=kwargs.get("parameters"),
            partition_key=kwargs.get("partition_key"),
            cross_partition=bool(kwargs.get("enable_cross_partition_query")),
        )

    async def read_item(self, *, item: str, partition_key: Any) -> dict[str, Any]:
        return await self._store.read_item(item, partition_key)

    async def upsert_item(self, *, body: dict[str, Any]) -> dict[str, Any]:
        return await self._store.add_cosmos(body)

    async def create_item(self, *, body: dict[str, Any]) -> dict[str, Any]:
        container = getattr(self._store, "container", None)
        if container is not None and hasattr(container, "create_item"):
            response = container.create_item(body=body)
            if inspect.isawaitable(response):
                response = await response
            return response if isinstance(response, dict) else body
        create = getattr(self._store, "create_item", None)
        if create is not None:
            response = create(body=body)
            if inspect.isawaitable(response):
                response = await response
            return response if isinstance(response, dict) else body
        return await self._store.add_cosmos(body)


class AsyncPipelineService:
    """Async LLM orchestration service backed by an async typed memory store."""

    _turns_container: Any = None

    def __init__(
        self,
        store: AsyncMemoryStore,
        chat_client: Any,
        embeddings_client: Any,
        prompts_dir: str | None = None,
        cosmos_turns_container: Any | None = None,
    ) -> None:
        self._store = store
        self._container = _AsyncStoreContainerAdapter(store)
        self._turns_container = cosmos_turns_container
        self._chat_client = chat_client
        self._embeddings = embeddings_client
        self._prompty = PromptyLoader(prompts_dir)

    async def _run_prompty(
        self,
        filename: str,
        inputs: dict[str, Any],
    ) -> str:
        """Render a prompty template, run the LLM async, and return the response text."""
        messages, params = self._prompty.prepare(filename, inputs)
        schema_format = response_format_for(filename)
        if schema_format is not None:
            params["response_format"] = schema_format
        response = await self._chat_client.generate(messages, **params)
        return chat_text(response)

    async def _embed_one(self, text: str) -> list[float]:
        return await self._embeddings.generate(text)

    async def _embed_batch(self, texts: list[str]) -> list[list[float]]:
        return await self._embeddings.generate_batch(texts)

    def _prompt_lineage(self, filename: str) -> dict[str, str]:
        """Return ``{prompt_id, prompt_version}`` for stamping a doc.

        Safe no-op fallback (``prompt_version="v1"``) when the loader was
        never initialised — happens in unit tests that build the service
        via ``__new__`` to bypass real LLM/embedding clients.
        """
        loader = getattr(self, "_prompty", None)
        version = loader.prompt_version(filename) if loader is not None else "v1"
        return {"prompt_id": filename, "prompt_version": version}

    def _validate_extracted_doc(self, doc: dict[str, Any]) -> dict[str, Any]:
        """Run an extracted fact/episodic doc through its typed model."""
        if doc.get("type") == "fact":
            return construct_internal(FactRecord, doc).to_doc()
        if doc.get("type") == "episodic":
            return construct_internal(EpisodicRecord, doc).to_doc()
        return doc

    @staticmethod
    def _chat_text(response: Any) -> str:
        return chat_text(response)

    @staticmethod
    def _build_transcript(
        items: list[dict[str, Any]],
        *,
        group_by_thread: bool = False,
    ) -> str:
        return build_transcript(items, group_by_thread=group_by_thread)

    async def _load_existing_memories(
        self,
        user_id: str,
        memory_types: list[str],
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        """Query active (non-superseded) memories for reconciliation context.

        Results are ordered by ``c._ts DESC`` so the most recently written
        memories survive the cap — without ORDER BY, Cosmos returns rows
        in implementation-defined order and the dedup comparison set is
        non-deterministic.
        """
        type_placeholders = ", ".join(f"@mtype{i}" for i in range(len(memory_types)))
        capped_limit = top_literal(limit, name="_load_existing_memories.limit")
        query = (
            f"SELECT TOP {capped_limit} * FROM c "
            f"WHERE c.user_id = @user_id "
            f"AND c.type IN ({type_placeholders}) "
            f"AND {_ACTIVE_DOC_FILTER} "
            f"ORDER BY c._ts DESC"
        )
        parameters: list[dict[str, Any]] = [
            {"name": "@user_id", "value": user_id},
        ]
        for i, mt in enumerate(memory_types):
            parameters.append({"name": f"@mtype{i}", "value": mt})

        return await self._container.query_items(
            query=query,
            parameters=parameters,
            enable_cross_partition_query=True,
        )

    async def _upsert_memory(self, doc: dict[str, Any]) -> dict[str, Any]:
        """Upsert a single memory document via the async memory store."""
        response = await self._store.add_cosmos(doc)
        if isinstance(response, dict):
            return response
        return doc

    async def _create_memory(self, doc: dict[str, Any]) -> dict[str, Any]:
        """Create a memory document and let Cosmos raise 409 for duplicates."""
        if getattr(self, "_store", None) is None:
            return await self._upsert_memory(doc)
        response = await self._container.create_item(body=doc)
        return response if isinstance(response, dict) else doc

    @staticmethod
    def _empty_extract_counts() -> dict[str, int]:
        return {
            "fact_count": 0,
            "episodic_count": 0,
            "unclassified_count": 0,
            "updated_count": 0,
            "contradicted_count": 0,
            "exact_dedup_skipped": 0,
        }

    @staticmethod
    def _stable_source_timestamp(items: list[dict[str, Any]]) -> str:
        timestamps = [str(item.get("created_at")) for item in items if item.get("created_at")]
        if timestamps:
            return max(timestamps)
        return datetime.now(timezone.utc).isoformat()

    async def _mark_extracted_superseded(
        self,
        *,
        user_id: str,
        thread_id: str,
        supersedes_id: str,
        superseder_id: str,
        reason: Literal["update", "contradict"],
    ) -> bool:
        try:
            old_mem = await self._container.read_item(item=supersedes_id, partition_key=[user_id, thread_id])
            if old_mem.get("superseded_by"):
                logger.debug(
                    "extract_memories: skipping UPDATE — target %s already superseded by %s",
                    supersedes_id,
                    old_mem.get("superseded_by"),
                )
                return False
            return await self._mark_superseded(old_mem, superseder_id, reason=reason)
        except CosmosResourceNotFoundError:
            logger.debug(
                "extract_memories: %s not found at (user_id, thread_id) — retrying cross-partition",
                supersedes_id,
            )
        except Exception as exc:
            # Includes 429s, 503s, transient connection errors — surface at WARNING
            # so they're not masked by the silent cross-partition fallback below.
            logger.warning(
                "extract_memories: read_item failed for %s (%s); retrying cross-partition",
                supersedes_id,
                type(exc).__name__,
            )
        try:
            q = (
                "SELECT * FROM c WHERE c.id = @id AND c.user_id = @uid "
                f"AND {_ACTIVE_DOC_FILTER}"
            )
            results = await self._container.query_items(
                query=q,
                parameters=[
                    {"name": "@id", "value": supersedes_id},
                    {"name": "@uid", "value": user_id},
                ],
                enable_cross_partition_query=True,
            )
            if results and not results[0].get("superseded_by"):
                return await self._mark_superseded(results[0], superseder_id, reason=reason)
        except Exception as exc:
            logger.warning("Failed to mark superseded memory %s: %s", supersedes_id, exc)
        return False

    async def _mark_superseded(
        self,
        old_doc: dict[str, Any],
        superseder_id: str,
        *,
        reason: Literal["duplicate", "contradict", "update"],
    ) -> bool:
        """Atomically set ``superseded_by`` on ``old_doc`` via the async memory store."""
        try:
            return await self._store.mark_superseded(old_doc, superseder_id, reason=reason)
        except Exception:
            logger.exception(
                "supersede failed id=%s superseder=%s",
                old_doc.get("id"),
                superseder_id,
            )
            return False

    @staticmethod
    def _parse_llm_json(text: str | None) -> dict[str, Any]:
        return parse_llm_json(text)

    async def extract_memories_dry(
        self,
        user_id: str,
        thread_id: str,
        recent_k: int | None = None,
        *,
        turns: Optional[list[dict[str, Any]]] = None,
    ) -> dict[str, list[dict[str, Any]]]:
        """Load turns, call the LLM, and return memory docs without embeddings or writes."""
        if not user_id:
            raise ValidationError("user_id is required")
        if not thread_id:
            raise ValidationError("thread_id is required")

        logger.info("extract_memories_dry started user_id=%s thread_id=%s", user_id, thread_id)

        if turns is None:
            query = "SELECT * FROM c WHERE c.user_id = @user_id AND c.thread_id = @thread_id"
            parameters: list[dict[str, Any]] = [
                {"name": "@user_id", "value": user_id},
                {"name": "@thread_id", "value": thread_id},
            ]
            items = await self._container.query_items(
                query=query,
                parameters=parameters,
                partition_key=[user_id, thread_id],
            )
            if self._turns_container is not None:
                async for item in self._turns_container.query_items(
                    query=query,
                    parameters=parameters,
                    partition_key=[user_id, thread_id],
                ):
                    items.append(item)
        else:
            items = list(turns)

        items.sort(key=lambda m: m.get("created_at", ""), reverse=True)
        if recent_k is not None:
            items = items[:recent_k]
        items.reverse()

        if not items:
            logger.warning("extract_memories_dry no memories found user_id=%s thread_id=%s", user_id, thread_id)
            return {"facts": [], "episodic": [], "updates": []}

        existing = await self._load_existing_memories(user_id, ["fact"])
        existing_fact_hashes: set[str] = {
            m["content_hash"] for m in existing if m.get("type") == "fact" and m.get("content_hash")
        }
        if existing:
            existing_text = "\n".join(
                f"- [ID: {mem['id']}] {mem.get('content', '')} "
                f"(type={mem.get('type', 'fact')}, salience={mem.get('salience', 'N/A')})"
                for mem in existing
            )
        else:
            existing_text = "(none)"

        transcript = self._build_transcript(items)
        response_text = await self._run_prompty(
            "extract_memories.prompty",
            inputs={"existing_facts": existing_text, "transcript": transcript},
        )
        parsed = self._parse_llm_json(response_text)
        facts = parsed.get("facts", [])
        episodic = parsed.get("episodic", [])
        unclassified = parsed.get("unclassified", [])

        doc_timestamp = self._stable_source_timestamp(items)
        fact_docs: list[dict[str, Any]] = []
        episodic_docs: list[dict[str, Any]] = []
        updates: list[dict[str, Any]] = []
        exact_dedup_skipped = 0

        for fact in facts:
            action = fact.get("action", "ADD").upper()
            text = fact.get("text")
            if not text:
                logger.warning("extract_memories: dropping malformed fact (missing 'text'): %r", fact)
                continue

            new_content_hash = compute_content_hash(text)
            if action == "ADD" and new_content_hash in existing_fact_hashes:
                logger.debug(
                    "extract_memories: skipping exact-dup fact hash=%s user_id=%s thread_id=%s",
                    new_content_hash,
                    user_id,
                    thread_id,
                )
                exact_dedup_skipped += 1
                continue

            seed = _ID_SEED_SEP.join((user_id, thread_id, new_content_hash))
            det_id = f"fact_{hashlib.sha256(seed.encode()).hexdigest()[:32]}"
            topic_tags = build_topic_tags(fact.get("tags", []))
            confidence = fact.get("confidence")
            doc: dict[str, Any] = {
                "id": det_id,
                "user_id": user_id,
                "thread_id": thread_id,
                "role": "system",
                "type": "fact",
                "content": text,
                "content_hash": new_content_hash,
                "confidence": 0.5 if confidence is None else confidence,
                **self._prompt_lineage("extract_memories.prompty"),
                "metadata": {
                    "category": fact.get("category") or "general",
                    "subject": fact.get("subject"),
                    "predicate": fact.get("predicate"),
                    "object": fact.get("object"),
                    "temporal_context": fact.get("temporal_context"),
                },
                "salience": fact.get("salience") if fact.get("salience") is not None else 0.5,
                "tags": ["sys:fact", "sys:auto-extracted"] + topic_tags,
                "created_at": doc_timestamp,
                "updated_at": doc_timestamp,
            }

            if action in {"UPDATE", "CONTRADICT"} and fact.get("supersedes_id"):
                reason: Literal["update", "contradict"] = "contradict" if action == "CONTRADICT" else "update"
                if det_id == fact["supersedes_id"]:
                    logger.debug("extract_memories: skipping UPDATE — det_id == supersedes_id (%s)", det_id)
                    continue
                doc["supersedes_ids"] = [fact["supersedes_id"]]
                updates.append(
                    {
                        "op": "supersede",
                        "supersedes_id": fact["supersedes_id"],
                        "superseder_id": det_id,
                        "thread_id": thread_id,
                        "reason": reason,
                    }
                )

            fact_docs.append(self._validate_extracted_doc(doc))
            existing_fact_hashes.add(new_content_hash)

        for ep in episodic:
            scope_type_raw = ep.get("scope_type")
            scope_value_raw = ep.get("scope_value")
            scope_type = scope_type_raw.strip() if isinstance(scope_type_raw, str) else None
            scope_value = scope_value_raw.strip() if isinstance(scope_value_raw, str) else None
            if not scope_type or not scope_value:
                logger.warning(
                    "extract_memories: dropping malformed episodic (missing scope_type/scope_value): %r",
                    ep,
                )
                continue

            situation = ep.get("situation")
            action_taken = ep.get("action_taken")
            outcome = ep.get("outcome")
            if situation and action_taken and outcome:
                text = f"{situation} → {action_taken} → {outcome}"
            else:
                text = f"For the user's {scope_value} {scope_type}, intent recorded."

            content_hash = compute_content_hash(text)
            seed = _ID_SEED_SEP.join((user_id, thread_id, content_hash))
            det_id = f"ep_{hashlib.sha256(seed.encode()).hexdigest()[:32]}"
            topic_tags = build_topic_tags(ep.get("tags", []))
            confidence = ep.get("confidence")
            raw_valence = ep.get("outcome_valence")
            coerced_valence = _coerce_valence(raw_valence)
            if raw_valence is not None and raw_valence not in VALID_VALENCES:
                logger.warning(
                    "extract_memories: coercing unknown outcome_valence=%r → %r user_id=%s thread_id=%s",
                    raw_valence,
                    coerced_valence,
                    user_id,
                    thread_id,
                )
            doc = {
                "id": det_id,
                "user_id": user_id,
                "thread_id": thread_id,
                "role": "system",
                "type": "episodic",
                "content": text,
                "content_hash": content_hash,
                "confidence": 0.5 if confidence is None else confidence,
                "ttl": DEFAULT_TTL_BY_TYPE.get("episodic", 7_776_000),
                **self._prompt_lineage("extract_memories.prompty"),
                "metadata": {
                    "scope_type": scope_type,
                    "scope_value": scope_value,
                    "situation": situation,
                    "action_taken": action_taken,
                    "outcome": outcome,
                    "reasoning": ep.get("reasoning"),
                    "outcome_valence": coerced_valence,
                    "lesson": ep.get("lesson")
                    or (
                        f"{situation} → {action_taken} → {outcome}" if situation and action_taken and outcome else text
                    ),
                    "domain": ep.get("domain"),
                },
                "salience": ep.get("salience"),
                "tags": ["sys:episodic", "sys:auto-extracted"] + topic_tags,
                "created_at": doc_timestamp,
                "updated_at": doc_timestamp,
            }
            episodic_docs.append(self._validate_extracted_doc(doc))

        for item in unclassified:
            text = item.get("text")
            if not text:
                continue
            content_hash = compute_content_hash(text)
            if content_hash in existing_fact_hashes:
                logger.debug(
                    "extract_memories: skipping exact-dup unclassified hash=%s user_id=%s thread_id=%s",
                    content_hash,
                    user_id,
                    thread_id,
                )
                exact_dedup_skipped += 1
                continue
            seed = _ID_SEED_SEP.join((user_id, thread_id, content_hash))
            det_id = f"fact_{hashlib.sha256(seed.encode()).hexdigest()[:32]}"
            topic_tags = build_topic_tags(item.get("tags", []))
            confidence = item.get("confidence")
            doc = {
                "id": det_id,
                "user_id": user_id,
                "thread_id": thread_id,
                "role": "system",
                "type": "fact",
                "content": text,
                "content_hash": content_hash,
                "confidence": 0.5 if confidence is None else confidence,
                **self._prompt_lineage("extract_memories.prompty"),
                "metadata": {"category": "unclassified", "unclassified_reason": item.get("reason")},
                "salience": item.get("salience") if item.get("salience") is not None else 0.5,
                "tags": ["sys:fact", "sys:auto-extracted", "sys:unclassified"] + topic_tags,
                "created_at": doc_timestamp,
                "updated_at": doc_timestamp,
            }
            fact_docs.append(self._validate_extracted_doc(doc))
            existing_fact_hashes.add(content_hash)

        if exact_dedup_skipped:
            updates.append({"op": "stats", "exact_dedup_skipped": exact_dedup_skipped})

        result = {"facts": fact_docs, "episodic": episodic_docs, "updates": updates}
        logger.info(
            "extract_memories_dry completed user_id=%s thread_id=%s fact_docs=%d episodic_docs=%d updates=%d",
            user_id,
            thread_id,
            len(fact_docs),
            len(episodic_docs),
            len(updates),
        )
        return result

    async def persist_extracted_memories(
        self,
        user_id: str,
        extracted: dict[str, list[dict[str, Any]]],
    ) -> dict[str, int]:
        """Embed and create extracted memories, skipping deterministic-ID conflicts."""
        if not user_id:
            raise ValidationError("user_id is required")
        if not isinstance(extracted, dict):
            raise ValidationError("extracted must be a dict")

        result = self._empty_extract_counts()
        fact_docs = [dict(doc) for doc in extracted.get("facts", [])]
        episodic_docs = [dict(doc) for doc in extracted.get("episodic", [])]
        update_ops = [dict(op) for op in extracted.get("updates", [])]
        docs_to_create = fact_docs + episodic_docs

        docs_needing_embeddings = [doc for doc in docs_to_create if doc.get("content") and not doc.get("embedding")]
        if docs_needing_embeddings:
            embeddings = await self._embed_batch([str(doc["content"]) for doc in docs_needing_embeddings])
            for doc, embedding in zip(docs_needing_embeddings, embeddings):
                doc["embedding"] = embedding

        for doc in docs_to_create:
            validated = self._validate_extracted_doc(doc)
            try:
                await self._create_memory(validated)
            except CosmosResourceExistsError:
                logger.info("persist_extracted_memories skipped existing id=%s", validated.get("id"))
                continue

            tags = validated.get("tags", [])
            if validated.get("type") == "episodic":
                result["episodic_count"] += 1
            elif "sys:unclassified" in tags:
                result["unclassified_count"] += 1
            elif validated.get("type") == "fact":
                result["fact_count"] += 1

        for op in update_ops:
            if op.get("op") == "stats":
                result["exact_dedup_skipped"] += int(op.get("exact_dedup_skipped") or 0)
                continue
            if op.get("op") != "supersede":
                continue
            reason = op.get("reason")
            op_thread_id = op.get("thread_id")
            supersedes_id = op.get("supersedes_id")
            superseder_id = op.get("superseder_id")
            if reason not in {"update", "contradict"} or not op_thread_id or not supersedes_id or not superseder_id:
                continue
            marked = await self._mark_extracted_superseded(
                user_id=user_id,
                thread_id=op_thread_id,
                supersedes_id=supersedes_id,
                superseder_id=superseder_id,
                reason=reason,
            )
            if marked:
                if reason == "contradict":
                    result["contradicted_count"] += 1
                else:
                    result["updated_count"] += 1

        logger.info("persist_extracted_memories completed user_id=%s counts=%s", user_id, result)
        return result

    async def extract_memories(
        self,
        user_id: str,
        thread_id: str,
        recent_k: int | None = None,
        *,
        turns: Optional[list[dict[str, Any]]] = None,
    ) -> dict[str, int]:
        """Extract facts and episodic memories from a thread and persist them."""
        extracted = await self.extract_memories_dry(user_id, thread_id, recent_k, turns=turns)
        return await self.persist_extracted_memories(user_id, extracted)

    async def synthesize_procedural(
        self,
        user_id: str,
        *,
        force: bool = False,
    ) -> dict[str, Any]:
        """Synthesize the active procedural prompt for a user."""
        if not user_id:
            raise ValidationError("user_id is required")

        logger.info("synthesize_procedural started user_id=%s force=%s", user_id, force)

        async def _read_latest_procedural() -> Optional[dict[str, Any]]:
            docs = await self._container.query_items(
                query=(
                    "SELECT * FROM c WHERE c.user_id = @uid "
                    "AND c.thread_id = @thread_id "
                    "AND c.type = @type "
                    f"AND {_ACTIVE_DOC_FILTER}"
                ),
                parameters=[
                    {"name": "@uid", "value": user_id},
                    {"name": "@thread_id", "value": "__procedural__"},
                    {"name": "@type", "value": "procedural"},
                ],
                enable_cross_partition_query=True,
            )
            docs.sort(
                key=lambda doc: (int(doc.get("version") or 0), int(doc.get("_ts") or 0)),
                reverse=True,
            )
            if len(docs) > 1:
                logger.warning(
                    "synthesize_procedural found multiple active docs user_id=%s count=%d",
                    user_id,
                    len(docs),
                )
            return docs[0] if docs else None

        prior_doc = await _read_latest_procedural()

        behavioral_fact_docs = await self._container.query_items(
            query=(
                "SELECT TOP 50 * FROM c WHERE c.user_id = @uid "
                "AND c.type = @type "
                f"AND {_ACTIVE_DOC_FILTER} "
                "AND ((IS_DEFINED(c.metadata.category) "
                "AND c.metadata.category IN ('preference', 'requirement')) "
                "OR (IS_DEFINED(c.salience) AND c.salience >= @min_salience)) "
                "ORDER BY c.salience DESC, c.created_at ASC, c.id ASC"
            ),
            parameters=[
                {"name": "@uid", "value": user_id},
                {"name": "@type", "value": "fact"},
                {"name": "@min_salience", "value": 0.8},
            ],
            enable_cross_partition_query=True,
        )
        behavioral_fact_docs = [
            doc
            for doc in behavioral_fact_docs
            if isinstance(doc.get("content"), str) and doc.get("content", "").strip()
        ]
        behavioral_fact_ids = [doc["id"] for doc in behavioral_fact_docs]

        episodic_docs = await self._container.query_items(
            query=(
                "SELECT TOP 50 * FROM c WHERE c.user_id = @uid "
                "AND c.type = @type "
                f"AND {_ACTIVE_DOC_FILTER} "
                "AND IS_DEFINED(c.metadata.lesson) "
                "AND c.metadata.lesson != null "
                "ORDER BY c.salience DESC, c.created_at ASC, c.id ASC"
            ),
            parameters=[
                {"name": "@uid", "value": user_id},
                {"name": "@type", "value": "episodic"},
            ],
            enable_cross_partition_query=True,
        )
        episodic_with_lessons = [
            doc
            for doc in episodic_docs
            if isinstance(doc.get("metadata", {}).get("lesson"), str)
            and doc.get("metadata", {}).get("lesson", "").strip()
        ]
        source_episodic_ids = [doc["id"] for doc in episodic_with_lessons]

        current_source_ids = set(behavioral_fact_ids) | set(source_episodic_ids)

        def _covered_by(prior: Optional[dict[str, Any]]) -> bool:
            if prior is None:
                return False
            covered = set(prior.get("source_fact_ids") or []) | set(prior.get("source_episodic_ids") or [])
            return current_source_ids.issubset(covered)

        if prior_doc and not force and _covered_by(prior_doc):
            logger.info(
                "synthesize_procedural unchanged user_id=%s fact_count=%d episodic_count=%d",
                user_id,
                len(behavioral_fact_ids),
                len(source_episodic_ids),
            )
            return {"status": "unchanged", "procedural": prior_doc}

        if not current_source_ids:
            logger.info(
                "synthesize_procedural skipping LLM user_id=%s — no behavioral facts or episodic lessons",
                user_id,
            )
            return {"status": "unchanged", "procedural": prior_doc}

        name_docs = await self._container.query_items(
            query=(
                "SELECT TOP 1 * FROM c WHERE c.user_id = @uid "
                "AND c.type = @type "
                f"AND {_ACTIVE_DOC_FILTER} "
                "AND IS_DEFINED(c.metadata.category) "
                "AND c.metadata.category = @category "
                "AND IS_DEFINED(c.metadata.predicate) "
                "AND c.metadata.predicate = @predicate "
                "ORDER BY c._ts DESC"
            ),
            parameters=[
                {"name": "@uid", "value": user_id},
                {"name": "@type", "value": "fact"},
                {"name": "@category", "value": "biographical"},
                {"name": "@predicate", "value": "name"},
            ],
            enable_cross_partition_query=True,
        )
        user_name = "the user"
        if name_docs:
            metadata = name_docs[0].get("metadata") or {}
            name_candidate = metadata.get("object")
            if not isinstance(name_candidate, str) or not name_candidate.strip():
                name_candidate = name_docs[0].get("content")
            if isinstance(name_candidate, str) and name_candidate.strip():
                user_name = name_candidate.strip()

        def _render_bullets(values: list[str]) -> str:
            cleaned = [value.strip() for value in values if isinstance(value, str) and value.strip()]
            if not cleaned:
                return "(none)"
            return "\n".join(f"- {value}" for value in cleaned)

        static_prompty_inputs = {
            "behavioral_facts": _render_bullets([doc.get("content", "") for doc in behavioral_fact_docs]),
            "episodic_lessons": _render_bullets(
                [doc.get("metadata", {}).get("lesson", "") for doc in episodic_with_lessons]
            ),
            "user_name": user_name,
        }

        # Retry loop: LLM call lives inside so that on a race-induced 409
        # we (a) check whether the winner already covers our source set and
        # short-circuit if so, and (b) re-call the LLM with the winner as
        # the new prior if not — keeping synthesized content monotonic in
        # source coverage, not just version number.
        written_doc: Optional[dict[str, Any]] = None
        for attempt in range(1, _PROCEDURAL_MAX_CREATE_ATTEMPTS + 1):
            response_text = await self._run_prompty(
                "synthesize_procedural.prompty",
                inputs={
                    "prior_prompt": (prior_doc.get("content") or "") if prior_doc else "",
                    **static_prompty_inputs,
                },
            )

            parsed = self._parse_llm_json(response_text)
            system_prompt = parsed.get("system_prompt") if isinstance(parsed, dict) else None
            if not isinstance(system_prompt, str) or not system_prompt.strip():
                raise LLMError("synthesize_procedural returned JSON without a non-empty 'system_prompt' string")
            system_prompt = system_prompt.strip()

            new_seq = (int(prior_doc.get("version") or 0) + 1) if prior_doc else 1
            new_doc: dict[str, Any] = {
                "id": f"proc_{user_id}_{new_seq}",
                "user_id": user_id,
                "thread_id": "__procedural__",
                "type": "procedural",
                "version": new_seq,
                "content": system_prompt,
                "source_fact_ids": behavioral_fact_ids,
                "source_episodic_ids": source_episodic_ids,
                "supersedes_ids": [prior_doc["id"]] if prior_doc else [],
                "created_at": datetime.now(timezone.utc).isoformat(),
                "updated_at": datetime.now(timezone.utc).isoformat(),
                "role": "system",
                "tags": ["sys:procedural", "sys:synthesized"],
                **self._prompt_lineage("synthesize_procedural.prompty"),
                "metadata": {},
            }
            validated = construct_internal(ProceduralRecord, new_doc).to_doc()
            try:
                await self._container.create_item(body=validated)
                written_doc = validated
                break
            except CosmosResourceExistsError:
                logger.info(
                    "synthesize_procedural id collision user_id=%s seq=%d attempt=%d/%d — re-reading",
                    user_id,
                    new_seq,
                    attempt,
                    _PROCEDURAL_MAX_CREATE_ATTEMPTS,
                )
                latest = await _read_latest_procedural()
                if latest is None:
                    continue
                prior_doc = latest
                if _covered_by(prior_doc):
                    logger.info(
                        "synthesize_procedural race resolved by coverage user_id=%s winner=%s",
                        user_id,
                        prior_doc["id"],
                    )
                    return {"status": "unchanged", "procedural": prior_doc}
        if written_doc is None:
            raise MemoryConflictError(
                "synthesize_procedural failed after "
                f"{_PROCEDURAL_MAX_CREATE_ATTEMPTS} attempts due to id collisions "
                f"user_id={user_id!r}"
            )

        new_id = written_doc["id"]
        if prior_doc:
            await self._mark_superseded(prior_doc, new_id, reason="update")

        logger.info(
            "synthesize_procedural synthesized user_id=%s version=%d fact_count=%d episodic_count=%d",
            user_id,
            written_doc["version"],
            len(behavioral_fact_ids),
            len(source_episodic_ids),
        )
        return {"status": "synthesized", "procedural": written_doc}

    async def generate_thread_summary_dry(
        self,
        user_id: str,
        thread_id: str,
        recent_k: int | None = None,
    ) -> dict[str, Any]:
        """Generate or update a thread summary document without embedding or writing it."""
        if not user_id:
            raise ValidationError("user_id is required")
        if not thread_id:
            raise ValidationError("thread_id is required")

        logger.info("generate_thread_summary_dry started user_id=%s thread_id=%s", user_id, thread_id)

        summary_id = f"summary_{user_id}_{thread_id}"
        existing_summary: Optional[dict[str, Any]] = None
        try:
            existing_summary = await self._container.read_item(item=summary_id, partition_key=[user_id, thread_id])
        except CosmosResourceNotFoundError:
            pass

        query = "SELECT * FROM c WHERE c.user_id = @user_id AND c.thread_id = @thread_id AND c.type != 'summary'"
        parameters: list[dict[str, Any]] = [
            {"name": "@user_id", "value": user_id},
            {"name": "@thread_id", "value": thread_id},
        ]
        if existing_summary:
            since = existing_summary["updated_at"]
            query += " AND c.created_at > @since"
            parameters.append({"name": "@since", "value": since})

        items = await self._container.query_items(
            query=query,
            parameters=parameters,
            partition_key=[user_id, thread_id],
        )
        if self._turns_container is not None:
            async for item in self._turns_container.query_items(
                query=query,
                parameters=parameters,
                partition_key=[user_id, thread_id],
            ):
                items.append(item)

        if existing_summary and not items:
            logger.info("generate_thread_summary_dry no new memories, returning existing")
            summary_doc = dict(existing_summary)
            summary_doc.pop("embedding", None)
            return summary_doc
        if not existing_summary and not items:
            raise ValidationError(f"No memories found for user_id={user_id!r}, thread_id={thread_id!r}")

        items.sort(key=lambda m: m.get("created_at", ""), reverse=True)
        if recent_k is not None:
            items = items[:recent_k]
        items.reverse()

        transcript = self._build_transcript(items)
        if existing_summary:
            prior_json = existing_summary.get("metadata", {}).get("structured_summary")
            prior_text = json.dumps(prior_json, indent=2) if prior_json else existing_summary.get("content", "")
            response_text = await self._run_prompty(
                "summarize_update.prompty",
                inputs={"prior_summary": prior_text, "transcript": transcript},
            )
            summary_prompt_filename = "summarize_update.prompty"
        else:
            response_text = await self._run_prompty("summarize.prompty", inputs={"transcript": transcript})
            summary_prompt_filename = "summarize.prompty"

        parsed = self._parse_llm_json(response_text)
        parsed = _cap_structured_summary(parsed)
        overview = parsed.get("overview", response_text)
        topics = parsed.get("topics", [])
        total_source_count = (
            existing_summary.get("metadata", {}).get("source_count", 0) if existing_summary else 0
        ) + len(items)
        topic_tags = build_topic_tags(topics)
        doc_timestamp = self._stable_source_timestamp(items)
        summary_doc: dict[str, Any] = {
            "id": summary_id,
            "user_id": user_id,
            "thread_id": thread_id,
            "role": "system",
            "type": "summary",
            "content": overview,
            "salience": 1.0,
            "tags": ["sys:summary"] + topic_tags,
            **self._prompt_lineage(summary_prompt_filename),
            "metadata": {
                "structured_summary": parsed,
                "source_count": total_source_count,
                "recent_k": recent_k,
                "incremental_update": existing_summary is not None,
            },
            "created_at": existing_summary["created_at"] if existing_summary else doc_timestamp,
            "updated_at": doc_timestamp,
        }
        return construct_internal(ThreadSummaryRecord, summary_doc).to_doc()

    async def persist_thread_summary(
        self,
        user_id: str,
        thread_id: str,
        summary_doc: dict[str, Any],
    ) -> dict[str, Any]:
        """Compute the summary embedding and upsert the deterministic summary doc."""
        if not user_id:
            raise ValidationError("user_id is required")
        if not thread_id:
            raise ValidationError("thread_id is required")
        if not isinstance(summary_doc, dict):
            raise ValidationError("summary_doc must be a dict")

        doc = dict(summary_doc)
        doc["id"] = doc.get("id") or f"summary_{user_id}_{thread_id}"
        doc["user_id"] = user_id
        doc["thread_id"] = thread_id
        doc.setdefault("prompt_id", "summarize.prompty")
        doc.setdefault("prompt_version", "v1")
        if doc.get("content") and not doc.get("embedding"):
            doc["embedding"] = await self._embed_one(doc["content"])
        validated = construct_internal(ThreadSummaryRecord, doc).to_doc()
        stored = await self._upsert_memory(validated)
        logger.info("persist_thread_summary completed id=%s", validated.get("id"))
        return stored

    async def generate_thread_summary(
        self,
        user_id: str,
        thread_id: str,
        recent_k: int | None = None,
    ) -> dict[str, Any]:
        """Generate or incrementally update a thread summary and persist it."""
        summary_doc = await self.generate_thread_summary_dry(user_id, thread_id, recent_k=recent_k)
        return await self.persist_thread_summary(user_id, thread_id, summary_doc)

    async def generate_user_summary_dry(
        self,
        user_id: str,
        thread_ids: list[str] | None = None,
        recent_k: int | None = None,
    ) -> dict[str, Any]:
        """Generate a user summary document without embedding or writing it."""
        if not user_id:
            raise ValidationError("user_id is required")

        logger.info(
            "generate_user_summary_dry started user_id=%s observed_thread_ids=%s",
            user_id,
            len(thread_ids) if thread_ids else 0,
        )

        user_summary_id = f"user_summary_{user_id}"
        existing_summary: Optional[dict[str, Any]] = None
        try:
            existing_summary = await self._container.read_item(
                item=user_summary_id,
                partition_key=[user_id, "__user_summary__"],
            )
        except CosmosResourceNotFoundError:
            pass

        query = "SELECT * FROM c WHERE c.user_id = @user_id AND c.type != 'user_summary'"
        parameters: list[dict[str, Any]] = [{"name": "@user_id", "value": user_id}]
        if existing_summary:
            since = existing_summary["updated_at"]
            query += " AND c.created_at > @since"
            parameters.append({"name": "@since", "value": since})

        items = await self._container.query_items(query=query, parameters=parameters, enable_cross_partition_query=True)
        if self._turns_container is not None:
            async for item in self._turns_container.query_items(
                query=query,
                parameters=parameters,
                enable_cross_partition_query=True,
            ):
                items.append(item)

        if existing_summary and not items:
            logger.info("generate_user_summary_dry no new memories, returning existing")
            user_doc = dict(existing_summary)
            user_doc.pop("embedding", None)
            return user_doc
        if not existing_summary and not items:
            raise ValidationError(f"No memories found for user_id={user_id!r}")

        items.sort(key=lambda m: m.get("created_at", ""), reverse=True)
        if recent_k is not None:
            by_thread: dict[str, list[dict[str, Any]]] = defaultdict(list)
            for m in items:
                by_thread[m.get("thread_id", "")].append(m)
            trimmed: list[dict[str, Any]] = []
            for thread_items in by_thread.values():
                trimmed.extend(thread_items[:recent_k])
            trimmed.sort(key=lambda m: m.get("created_at", ""))
            items = trimmed
        else:
            items.reverse()

        transcript = self._build_transcript(items, group_by_thread=True)
        new_thread_ids = {m.get("thread_id", "") for m in items}
        if existing_summary:
            prior_json = existing_summary.get("metadata", {}).get("structured_summary")
            prior_text = json.dumps(prior_json, indent=2) if prior_json else existing_summary.get("content", "")
            response_text = await self._run_prompty(
                "user_summary_update.prompty",
                inputs={"prior_summary": prior_text, "transcript": transcript},
            )
            prompt_filename = "user_summary_update.prompty"
        else:
            response_text = await self._run_prompty("user_summary.prompty", inputs={"transcript": transcript})
            prompt_filename = "user_summary.prompty"

        parsed = self._parse_llm_json(response_text)
        parsed = _cap_structured_summary(parsed)
        key_facts = parsed.get("key_facts", [])
        overview = "; ".join(key_facts) if key_facts else response_text
        if existing_summary:
            old_thread_ids = set(existing_summary.get("metadata", {}).get("thread_ids", []))
            all_thread_ids = sorted(old_thread_ids | new_thread_ids)
            old_memory_count = existing_summary.get("metadata", {}).get("source_memory_count", 0)
            total_memory_count = old_memory_count + len(items)
        else:
            all_thread_ids = sorted(new_thread_ids)
            total_memory_count = len(items)

        topic_tags = build_topic_tags(parsed.get("topics", []))
        doc_timestamp = self._stable_source_timestamp(items)
        summary_doc: dict[str, Any] = {
            "id": user_summary_id,
            "user_id": user_id,
            "thread_id": "__user_summary__",
            "role": "system",
            "type": "user_summary",
            "content": overview,
            "salience": 1.0,
            "tags": ["sys:user-summary"] + topic_tags,
            **self._prompt_lineage(prompt_filename),
            "metadata": {
                "structured_summary": parsed,
                "source_thread_count": len(all_thread_ids),
                "source_memory_count": total_memory_count,
                "thread_ids": all_thread_ids,
                "recent_k": recent_k,
                "incremental_update": existing_summary is not None,
            },
            "created_at": existing_summary["created_at"] if existing_summary else doc_timestamp,
            "updated_at": doc_timestamp,
        }
        return construct_internal(UserSummaryRecord, summary_doc).to_doc()

    async def persist_user_summary(
        self,
        user_id: str,
        user_summary_doc: dict[str, Any],
    ) -> dict[str, Any]:
        """Compute the user-summary embedding and upsert the deterministic doc."""
        if not user_id:
            raise ValidationError("user_id is required")
        if not isinstance(user_summary_doc, dict):
            raise ValidationError("user_summary_doc must be a dict")

        doc = dict(user_summary_doc)
        doc["id"] = doc.get("id") or f"user_summary_{user_id}"
        doc["user_id"] = user_id
        doc["thread_id"] = "__user_summary__"
        doc.setdefault("prompt_id", "user_summary.prompty")
        doc.setdefault("prompt_version", "v1")
        structured_summary = doc.get("metadata", {}).get("structured_summary")
        topics = structured_summary.get("topics", []) if isinstance(structured_summary, dict) else []
        doc["tags"] = sorted({*(doc.get("tags") or []), "sys:user-summary", *build_topic_tags(topics)})
        if doc.get("content") and not doc.get("embedding"):
            doc["embedding"] = await self._embed_one(doc["content"])
        validated = construct_internal(UserSummaryRecord, doc).to_doc()
        stored = await self._upsert_memory(validated)
        logger.info("persist_user_summary completed id=%s", validated.get("id"))
        return stored

    async def generate_user_summary(
        self,
        user_id: str,
        thread_ids: list[str] | None = None,
        recent_k: int | None = None,
    ) -> dict[str, Any]:
        """Generate or incrementally update a user summary and persist it."""
        summary_doc = await self.generate_user_summary_dry(user_id, thread_ids=thread_ids, recent_k=recent_k)
        return await self.persist_user_summary(user_id, summary_doc)

    def _emit_reconcile_outcome(
        self,
        *,
        started_at: float,
        user_id: str,
        candidates: int,
        result: dict[str, int],
    ) -> None:
        duration_ms = (time.monotonic() - started_at) * 1000.0
        logger.info(
            "reconcile.outcome",
            extra={
                "operation": "reconcile_memories",
                "user_id": user_id,
                "candidates_considered": candidates,
                "kept": result["kept"],
                "merged": result["merged"],
                "contradicted": result["contradicted"],
                "duration_ms": duration_ms,
                "prompt_id": "dedup.prompty",
                "prompt_version": "v1",
            },
        )

    async def reconcile_memories(self, user_id: str, n: int = 50) -> dict[str, int]:
        """Reconcile a user's active facts in a single LLM pass.

        Loads the most recent ``n`` active (non-superseded) facts for
        ``user_id``, asks the dedup prompt to classify them into
        ``duplicate_groups``, ``contradicted_pairs``, and ``kept_ids``, then
        applies both kinds of resolutions:

        * **Duplicates** — a fresh merged fact is upserted; every source is
          soft-deleted with ``supersede_reason="duplicate"``.
        * **Contradictions** — the loser is soft-deleted with
          ``supersede_reason="contradict"`` and ``superseded_by`` set to
          the winner. Dangling references are resolved transparently when a
          contradicted id was just absorbed into a duplicate group.

        Returns ``{"kept": int, "merged": int, "contradicted": int}`` where
        ``merged`` and ``contradicted`` count the *losers* that were
        soft-deleted (duplicates and contradictions respectively).
        """
        if not user_id:
            raise ValidationError("user_id is required")
        if not isinstance(n, int) or isinstance(n, bool) or n < 1:
            raise ValidationError(f"n must be a positive integer, got {n!r}")
        if n > 500:
            raise ValidationError(f"n must be <= 500 to bound prompt size and LLM cost, got {n}")

        started_at = time.monotonic()
        logger.info("reconcile_memories started user_id=%s n=%d", user_id, n)

        # ---- 1. Load up to N most recent active facts ----
        # ORDER BY c.created_at DESC keeps the TOP cap deterministic across
        # physical partitions and matches the dedup prompt's tiebreaker
        # ("more recent created_at first"). Cosmos's _ts is the last-write
        # timestamp, which would diverge from created_at after any UPDATE.
        query = (
            f"SELECT TOP {n} * FROM c "
            "WHERE c.user_id = @user_id "
            "AND c.type = 'fact' "
            f"AND {_ACTIVE_DOC_FILTER} "
            "ORDER BY c.created_at DESC"
        )
        parameters: list[dict[str, Any]] = [
            {"name": "@user_id", "value": user_id},
        ]
        facts = await self._container.query_items(
            query=query,
            parameters=parameters,
            enable_cross_partition_query=True,
        )

        if len(facts) <= 1:
            logger.info(
                "reconcile_memories: %d facts, nothing to reconcile",
                len(facts),
            )
            early_result = {"kept": len(facts), "merged": 0, "contradicted": 0}
            self._emit_reconcile_outcome(
                started_at=started_at,
                user_id=user_id,
                candidates=len(facts),
                result=early_result,
            )
            return early_result

        # ---- 2. Format the facts pool for the prompt ----
        # ``json.dumps`` escapes embedded quotes and pipes inside content so
        # the visual grammar (`| Field:` separators, `"<text>"` quoting)
        # stays unambiguous even on adversarial inputs like
        # ``She said "hi" | weird``. IDs are kept raw because they're
        # deterministic alphanumerics — quoting them risks the LLM copying
        # the quotes back into ``source_ids``.
        lines: list[str] = []
        for i, cf in enumerate(facts, 1):
            content_quoted = json.dumps(cf.get("content", ""), ensure_ascii=False)
            conf_raw = cf.get("confidence")
            sal_raw = cf.get("salience")
            conf_str = conf_raw if _is_real_number(conf_raw) else "N/A"
            sal_str = sal_raw if _is_real_number(sal_raw) else "N/A"
            created_raw = cf.get("created_at")
            created_str = created_raw if created_raw else "N/A"
            lines.append(
                f"{i}. ID: {cf['id']} | Content: {content_quoted} | "
                f"Confidence: {conf_str} | "
                f"Salience: {sal_str} | "
                f"Created: {created_str}"
            )
        facts_text = "\n".join(lines)

        # ---- 3. Single LLM call over the entire pool ----
        response_text = await self._run_prompty(
            "dedup.prompty",
            inputs={"facts_text": facts_text},
        )
        parsed = self._parse_llm_json(response_text)

        duplicate_groups = parsed.get("duplicate_groups", []) or []
        contradicted_pairs = parsed.get("contradicted_pairs", []) or []
        # ``kept_ids`` from the LLM is used below as a cross-check for
        # accounting drift (hallucinated IDs, double-counting). The actual
        # kept count is computed from facts minus consumed losers.
        llm_kept_ids = list(parsed.get("kept_ids", []) or [])

        facts_by_id: dict[str, dict[str, Any]] = {f["id"]: f for f in facts}

        merged = 0
        contradicted = 0
        # Tracks source_id -> merged_id rewrites so contradictions whose
        # winner/loser landed in a duplicate group can be redirected to
        # the surviving merged document. Only updated on *successful*
        # supersede so stale redirects don't survive ETag races.
        source_to_merged_id: dict[str, str] = {}
        # Cache of merged docs we just upserted, keyed by merged_id. Lets
        # the contradiction redirector reuse the in-memory dict instead of
        # a cross-partition Cosmos round-trip for a doc we own. Also keeps
        # the chain ETag-stable when the same merged doc absorbs both a
        # duplicate group and a contradiction redirect in the same call.
        merged_docs_by_id: dict[str, dict[str, Any]] = {}
        # Set of source IDs that were *actually* superseded (counts toward
        # ``merged``). Used by the kept-count cross-check below — earlier
        # versions counted attempts and undercounted on ETag races.
        consumed_source_ids: set[str] = set()
        # Set of contradiction loser IDs that were *actually* superseded.
        consumed_loser_ids: set[str] = set()
        # Original-pool winner IDs from successfully-applied contradictions.
        # The LLM emits winners under ``contradicted_pairs``, never under
        # ``kept_ids`` — so the kept-cross-check at the end must subtract
        # them from the expected-kept set or every clean run looks like a
        # mismatch.
        contradiction_winner_ids_in_pool: set[str] = set()

        # ---- 4. Apply duplicate_groups FIRST ----
        for group in duplicate_groups:
            source_ids = list(group.get("source_ids") or [])
            merged_content = group.get("merged_content")
            if not merged_content or not source_ids:
                logger.debug(
                    "reconcile_memories: skipping malformed duplicate_group %r",
                    group,
                )
                continue

            source_docs = [facts_by_id[sid] for sid in source_ids if sid in facts_by_id]
            if not source_docs:
                logger.debug(
                    "reconcile_memories: duplicate_group references unknown ids %r",
                    source_ids,
                )
                continue

            # Filtered, hallucination-free view of the source ids that
            # actually exist in the pool. Used both for ``supersedes_ids``
            # on the merged record and for the deterministic merged-id
            # below so the merged doc faithfully represents reality.
            valid_source_ids = [sid for sid in source_ids if sid in facts_by_id]

            if len(valid_source_ids) < 2:
                logger.debug(
                    "reconcile_memories: skipping single-source duplicate_group %r",
                    source_ids,
                )
                continue

            # Sort source_docs by Cosmos _ts DESC so the merged record's
            # partition (thread_id) is picked deterministically from the
            # newest source — independent of the LLM's source_ids order.
            source_docs.sort(key=lambda d: d.get("_ts", 0), reverse=True)

            # Union tags across all source docs (preserve order, dedupe).
            merged_tags: list[str] = []
            seen_tags: set[str] = set()
            for src in source_docs:
                for t in src.get("tags", []) or []:
                    if t not in seen_tags:
                        seen_tags.add(t)
                        merged_tags.append(t)
            if not merged_tags:
                merged_tags = ["sys:fact"]

            # Union source_memory_ids across all source docs (provenance chain).
            merged_source_memory_ids: list[str] = []
            seen_smi: set[str] = set()
            for src in source_docs:
                for smi in src.get("source_memory_ids", []) or []:
                    if smi not in seen_smi:
                        seen_smi.add(smi)
                        merged_source_memory_ids.append(smi)

            # Transitive supersedes_ids: include any prior chain hops the
            # source docs already absorbed so the merged record carries
            # the full provenance, not just the immediate parent layer.
            merged_supersedes: list[str] = []
            seen_sup: set[str] = set()
            for sid in valid_source_ids:
                if sid not in seen_sup:
                    seen_sup.add(sid)
                    merged_supersedes.append(sid)
            for src in source_docs:
                for prior in src.get("supersedes_ids", []) or []:
                    if prior and prior not in seen_sup:
                        seen_sup.add(prior)
                        merged_supersedes.append(prior)

            # Newest source's thread_id wins (after _ts-desc sort above).
            recent_thread_id = source_docs[0].get("thread_id", "")

            # If LLM omitted confidence/salience, returned a non-positive
            # placeholder, returned a JSON ``true`` masquerading as numeric,
            # or returned an out-of-range value (e.g. 1.05 — common when
            # models confuse percent with [0,1]), fall back to max across
            # the source docs. Out-of-range without a fallback would let
            # ``MemoryRecord(...)`` raise on Pydantic validation and the
            # blanket except below would silently drop the entire group.
            llm_conf = group.get("confidence")
            confidence_val = (
                float(llm_conf)
                if _is_real_number(llm_conf) and 0 < llm_conf <= 1
                else _max_or_none(src.get("confidence") for src in source_docs)
            )
            llm_sal = group.get("salience")
            salience_val = (
                float(llm_sal)
                if _is_real_number(llm_sal) and 0 < llm_sal <= 1
                else _max_or_none(src.get("salience") for src in source_docs)
            )

            # Deterministic merged id keyed on (user, "merged", content_hash)
            # so re-running reconcile on the same merged content produces an
            # idempotent upsert instead of a fresh UUID each cycle. Stable
            # ids also keep the supersede chain shallow: a future paraphrase
            # that gets folded into the same canonical merged content will
            # see the same id rather than chaining through a new UUID.
            merged_content_hash = compute_content_hash(merged_content)
            merged_id_seed = _ID_SEED_SEP.join((user_id, "merged", merged_content_hash))
            merged_id = "fact_" + hashlib.sha256(merged_id_seed.encode()).hexdigest()[:32]

            try:
                merged_record = construct_internal(
                    FactRecord,
                    {
                        "id": merged_id,
                        "user_id": user_id,
                        "role": "system",
                        "type": "fact",
                        "content": merged_content,
                        "thread_id": recent_thread_id or f"__reconciled__:{user_id}",
                        "confidence": confidence_val if confidence_val is not None else 0.5,
                        "salience": salience_val if salience_val is not None else 0.5,
                        "supersedes_ids": merged_supersedes,
                        "source_memory_ids": merged_source_memory_ids,
                        "tags": merged_tags,
                        "content_hash": merged_content_hash,
                        "metadata": {
                            "category": "preference",
                            "merged_via": "reconcile",
                            "merged_from_count": len(valid_source_ids),
                        },
                        **self._prompt_lineage("dedup.prompty"),
                        "created_at": datetime.now(timezone.utc).isoformat(),
                        "updated_at": datetime.now(timezone.utc).isoformat(),
                    },
                )
            except Exception:
                logger.exception(
                    "reconcile_memories: failed to build merged record for group %r",
                    group,
                )
                continue

            # Generate embedding for the merged content so retrieval can
            # rank it against future queries from the moment it lands.
            # If embedding fails, abort this duplicate group entirely:
            # writing a merged doc with no embedding and then superseding
            # the sources would create a search-index hole until the next
            # reconcile retried. Better to leave the duplicates in place.
            try:
                merged_record.embedding = await self._embed_one(merged_content)
            except Exception:
                logger.exception(
                    "reconcile_memories: embedding failed for merged id=%s; "
                    "aborting duplicate group to avoid search-index hole",
                    merged_record.id,
                )
                continue

            merged_doc = merged_record.to_doc()
            try:
                merged_doc = await self._upsert_memory(merged_doc)
            except Exception:
                logger.exception(
                    "reconcile_memories: upsert failed for merged id=%s; aborting duplicate group",
                    merged_record.id,
                )
                continue
            merged_docs_by_id[merged_record.id] = merged_doc

            group_supersede_count = 0
            for sid in valid_source_ids:
                src_doc = facts_by_id.get(sid)
                if src_doc is None:
                    # Defensive — already filtered above, kept for clarity.
                    continue
                # Only update redirect/consumed-set on *successful* supersede.
                # Losing the ETag race means another writer beat us; the
                # source doc is still active from our perspective and should
                # not be treated as consumed.
                if await self._mark_superseded(src_doc, merged_record.id, reason="duplicate"):
                    merged += 1
                    group_supersede_count += 1
                    source_to_merged_id[sid] = merged_record.id
                    consumed_source_ids.add(sid)

            # If every supersede attempt for this group failed (typically
            # an ETag race against a concurrent reconcile that already
            # superseded the same sources to the *same* deterministic
            # merged id), do NOT delete the merged doc. A delete here
            # would orphan the sources whose ``superseded_by`` already
            # points at this merged id — they'd become invisible to
            # default reads (filter ``superseded_by IS NULL``) and to the
            # reconcile pool, causing permanent data loss. The merged doc
            # is idempotent (deterministic id), so leaving it in place is
            # consistent with whatever the winning concurrent writer
            # produced.
            if group_supersede_count == 0:
                logger.info(
                    "reconcile_memories: no sources superseded for merged id=%s "
                    "(likely ETag race with concurrent reconcile); leaving "
                    "merged doc in place — idempotent upsert is self-healing",
                    merged_record.id,
                )

        # ---- 5. Apply contradicted_pairs SECOND with dangling-id resolution ----
        for pair in contradicted_pairs:
            winner_id = pair.get("winner_id")
            loser_id = pair.get("loser_id")
            if not winner_id or not loser_id:
                logger.debug(
                    "reconcile_memories: skipping malformed contradicted_pair %r",
                    pair,
                )
                continue

            # Redirect through any duplicate-merge that absorbed the id.
            resolved_winner = source_to_merged_id.get(winner_id, winner_id)
            resolved_loser_id = source_to_merged_id.get(loser_id, loser_id)

            # Validate the (resolved) winner. The LLM is instructed never to
            # invent IDs — if it does, refuse to write a dangling
            # ``superseded_by`` pointer that breaks the audit trail.
            if resolved_winner not in facts_by_id and resolved_winner not in merged_docs_by_id:
                logger.warning(
                    "reconcile_memories: hallucinated winner_id=%s (resolved=%s) "
                    "not in pool or merged set; skipping pair %r",
                    winner_id,
                    resolved_winner,
                    pair,
                )
                continue

            if resolved_winner == resolved_loser_id:
                # Both sides collapsed into the same merged doc — the
                # contradiction is moot. Drop it silently.
                logger.debug(
                    "reconcile_memories: contradiction collapsed into duplicate group "
                    "(winner=%s loser=%s -> %s); skipping",
                    winner_id,
                    loser_id,
                    resolved_winner,
                )
                continue

            loser_doc = facts_by_id.get(resolved_loser_id)
            if loser_doc is None and resolved_loser_id != loser_id:
                # The original loser was just merged. Reuse the in-memory
                # merged doc so we skip a cross-partition re-fetch — we
                # own the (user_id, thread_id) partition and just wrote
                # it. This in-memory copy carries the ``_etag`` returned
                # by ``_upsert_memory``'s captured upsert response, so
                # the supersede below takes the ETag-protected
                # ``replace_item`` branch — concurrency-safe against any
                # other reconcile that may have touched the same merged
                # id in parallel.
                loser_doc = merged_docs_by_id.get(resolved_loser_id)

            if loser_doc is None:
                logger.warning(
                    "reconcile_memories: loser doc not found for pair %r (resolved_loser=%s)",
                    pair,
                    resolved_loser_id,
                )
                continue

            if await self._mark_superseded(loser_doc, resolved_winner, reason="contradict"):
                contradicted += 1
                # Track the *original* loser_id from the LLM so the kept
                # cross-check below can reconcile against the input pool.
                if loser_id in facts_by_id:
                    consumed_loser_ids.add(loser_id)
                # If the winner is an original pool member (not a freshly
                # minted merged doc), record it so the kept-cross-check
                # doesn't flag a clean run.
                if winner_id in facts_by_id:
                    contradiction_winner_ids_in_pool.add(winner_id)

        # The pipeline's "kept" semantic = facts that survive as live
        # records in the pool. The LLM's ``kept_ids`` semantic =
        # everything *not* mentioned in duplicate_groups or
        # contradicted_pairs. They differ by exactly the contradiction
        # winners (winners survive but are listed under contradicted_pairs).
        consumed_ids = consumed_source_ids | consumed_loser_ids
        kept_actual = {fid for fid in facts_by_id.keys() if fid not in consumed_ids}
        kept = len(kept_actual)
        # Cross-check: the LLM's kept_ids set should equal kept_actual
        # minus the contradiction winners. Mismatch usually means the LLM
        # hallucinated an id or double-counted a fact across categories.
        expected_llm_kept = kept_actual - contradiction_winner_ids_in_pool
        llm_kept_set = {kid for kid in llm_kept_ids if kid in facts_by_id}
        if llm_kept_set != expected_llm_kept:
            symdiff = sorted(llm_kept_set ^ expected_llm_kept)[:10]
            logger.info(
                "reconcile_memories: kept_ids mismatch (llm=%d valid=%d, expected=%d). "
                "Likely a hallucinated or double-counted fact id. Sample diff (≤10): %s",
                len(llm_kept_ids),
                len(llm_kept_set),
                len(expected_llm_kept),
                symdiff,
            )
        result = {"kept": kept, "merged": merged, "contradicted": contradicted}
        logger.info("reconcile_memories completed: %s", result)
        self._emit_reconcile_outcome(
            started_at=started_at,
            user_id=user_id,
            candidates=len(facts),
            result=result,
        )
        return result

    async def build_procedural_context(self, user_id: str) -> str:
        """Return the active synthesized procedural prompt for system injection."""
        if not user_id:
            raise ValidationError("user_id is required")
        query = (
            "SELECT TOP 1 c.content, c.version FROM c WHERE c.user_id = @user_id "
            "AND c.thread_id = @thread_id AND c.type = @type "
            f"AND {_ACTIVE_DOC_FILTER} "
            "ORDER BY c.version DESC"
        )
        items = await self._store.query(
            query,
            parameters=[
                {"name": "@user_id", "value": user_id},
                {"name": "@thread_id", "value": "__procedural__"},
                {"name": "@type", "value": "procedural"},
            ],
            cross_partition=True,
        )
        if not items:
            return ""
        content = items[0].get("content")
        return content if isinstance(content, str) else ""


__all__ = ["AsyncPipelineService"]
