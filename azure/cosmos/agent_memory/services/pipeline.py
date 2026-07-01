"""Pipeline service for LLM-driven memory extraction, summaries, and reconciliation.

Owns LLM-call orchestration; all persistence goes through the injected memory
store and all model calls go through the injected chat client. Pure helpers
(chat-text extraction, transcript building, prompty handling, LLM JSON parsing)
live in :mod:`azure.cosmos.agent_memory.services._pipeline_helpers` and are shared
with :class:`azure.cosmos.agent_memory.aio.services.pipeline.AsyncPipelineService`.
"""

from __future__ import annotations

import hashlib
import json
import time
from collections import defaultdict
from datetime import datetime, timezone
from typing import Any, Iterable, Literal, Optional

from azure.cosmos.exceptions import (
    CosmosResourceExistsError,
    CosmosResourceNotFoundError,
)

from azure.cosmos.agent_memory import thresholds as threshold_config
from azure.cosmos.agent_memory._container_routing import ContainerKey
from azure.cosmos.agent_memory._utils import (
    DEFAULT_TTL_BY_TYPE,
    compute_content_hash,
    distance_function_from_container_properties,
    vector_autodrop_supported,
    vector_order_direction,
    vector_similarity_at_least,
)
from azure.cosmos.agent_memory.exceptions import (
    LLMError,
    MemoryConflictError,
    ValidationError,
)
from azure.cosmos.agent_memory.logging import get_logger
from azure.cosmos.agent_memory.models import (
    EpisodicRecord,
    FactRecord,
    ProceduralRecord,
    ThreadSummaryRecord,
    UserSummaryRecord,
    construct_internal,
)
from azure.cosmos.agent_memory.prompts._schemas import response_format_for
from azure.cosmos.agent_memory.services import MemoryStoreProtocol
from azure.cosmos.agent_memory.services._pipeline_helpers import (
    ID_SEED_SEP as _ID_SEED_SEP,
)
from azure.cosmos.agent_memory.services._pipeline_helpers import (
    VALID_VALENCES,
    PromptyLoader,
    _normalize_metadata_keys,
    build_topic_tags,
    build_transcript,
    cap_structured_summary,
    chat_text,
    check_extracted_fact_grounding,
    coerce_valence,
    parse_llm_json,
)
from azure.cosmos.agent_memory.services._pipeline_helpers import (
    is_real_number as _is_real_number,
)
from azure.cosmos.agent_memory.services._pipeline_helpers import (
    max_or_none as _max_or_none,
)
from azure.cosmos.agent_memory.store._search_helpers import top_literal

logger = get_logger("azure.cosmos.agent_memory.pipeline")


_coerce_valence = coerce_valence
_cap_structured_summary = cap_structured_summary

# Standard SQL predicate that selects "active" (non-superseded) docs.
# Used in every query that should ignore superseded history. Kept as a single
# constant so any tweak (e.g. switching to a computed property) only happens
# once.
_ACTIVE_DOC_FILTER = "(NOT IS_DEFINED(c.superseded_by) OR IS_NULL(c.superseded_by))"

# Maximum number of create_item retries inside synthesize_procedural's
# race-recovery loop. Each attempt costs one LLM round-trip in the worst case;
# 5 is comfortably above the realistic worst-case race depth (two parallel
# threads per user, occasional triple-fanout) without inflating the cost cap.
_PROCEDURAL_MAX_CREATE_ATTEMPTS = 5


class _StoreContainerAdapter:
    """Expose one split ``MemoryStore`` container via Cosmos method shapes."""

    def __init__(self, store: MemoryStoreProtocol, container_key: ContainerKey) -> None:
        self._store = store
        self._container_key = container_key

    def _target_container(self) -> Any | None:
        containers = getattr(self._store, "_containers", None)
        if isinstance(containers, dict):
            return containers.get(self._container_key)
        if self._container_key is ContainerKey.MEMORIES:
            return getattr(self._store, "container", None)
        return None

    def query_items(self, **kwargs: Any) -> list[dict[str, Any]]:
        container = self._target_container()
        if container is not None and hasattr(container, "query_items"):
            return list(container.query_items(**kwargs))
        try:
            return self._store.query(
                kwargs["query"],
                parameters=kwargs.get("parameters"),
                container_key=self._container_key,
                partition_key=kwargs.get("partition_key"),
                cross_partition=bool(kwargs.get("enable_cross_partition_query")),
            )
        except TypeError:
            return self._store.query(
                kwargs["query"],
                parameters=kwargs.get("parameters"),
                partition_key=kwargs.get("partition_key"),
                cross_partition=bool(kwargs.get("enable_cross_partition_query")),
            )

    def read_item(self, *, item: str, partition_key: Any) -> dict[str, Any]:
        container = self._target_container()
        if container is not None and hasattr(container, "read_item"):
            return container.read_item(item=item, partition_key=partition_key)
        try:
            return self._store.read_item(item, partition_key, container_key=self._container_key)
        except TypeError:
            return self._store.read_item(item, partition_key)

    def upsert_item(self, *, body: dict[str, Any]) -> dict[str, Any]:
        container = self._target_container()
        if container is not None and hasattr(container, "upsert_item"):
            response = container.upsert_item(body=body)
            return response if isinstance(response, dict) else body
        upsert = getattr(self._store, "upsert_item", None)
        if upsert is not None:
            response = upsert(body=body)
            return response if isinstance(response, dict) else body
        return self._store.add_cosmos(body)

    def create_item(self, *, body: dict[str, Any]) -> dict[str, Any]:
        container = self._target_container()
        if container is not None and hasattr(container, "create_item"):
            response = container.create_item(body=body)
            return response if isinstance(response, dict) else body
        create = getattr(self._store, "create_item", None)
        if create is not None:
            response = create(body=body)
            return response if isinstance(response, dict) else body
        return self._store.add_cosmos(body)

    def replace_item(self, **kwargs: Any) -> Any:
        container = self._target_container()
        if container is not None and hasattr(container, "replace_item"):
            return container.replace_item(**kwargs)
        body = kwargs["body"]
        return self.upsert_item(body=body)


class PipelineService:
    """LLM orchestration service backed by a typed memory store."""

    def __init__(
        self,
        store: MemoryStoreProtocol,
        chat_client: Any,
        embeddings_client: Any,
        prompts_dir: str | None = None,
        *,
        containers: dict[ContainerKey, Any],
        transcript_metadata_keys: Optional[Iterable[str]] = None,
    ) -> None:
        self._store = store
        self._containers = containers
        self._memories_container = containers[ContainerKey.MEMORIES]
        self._turns_container = containers[ContainerKey.TURNS]
        self._summaries_container = containers[ContainerKey.SUMMARIES]
        self._container = self._memories_container
        self._chat_client = chat_client
        self._embeddings = embeddings_client
        self._prompty = PromptyLoader(prompts_dir)
        self._transcript_metadata_keys: Optional[tuple[str, ...]] = _normalize_metadata_keys(transcript_metadata_keys)

    def _run_prompty(
        self,
        filename: str,
        inputs: dict[str, Any],
    ) -> str:
        """Render a prompty template, run the LLM, and return the response text."""
        messages, params = self._prompty.prepare(filename, inputs)
        schema_format = response_format_for(filename)
        if schema_format is not None:
            params["response_format"] = schema_format
        response = self._chat_client.generate(messages, **params)
        return chat_text(response)

    def _embed_one(self, text: str) -> list[float]:
        return self._embeddings.generate(text)

    def _embed_batch(self, texts: list[str]) -> list[list[float]]:
        return self._embeddings.generate_batch(texts)

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

    def _build_transcript(
        self,
        items: list[dict[str, Any]],
        *,
        group_by_thread: bool = False,
    ) -> str:
        # getattr fallback covers unit tests that build PipelineService via
        # __new__ to bypass __init__ (and therefore the metadata-keys stash).
        return build_transcript(
            items,
            group_by_thread=group_by_thread,
            metadata_keys=getattr(self, "_transcript_metadata_keys", None),
        )

    def _load_existing_memories(
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

        items = list(
            self._memories_container.query_items(
                query=query,
                parameters=parameters,
                enable_cross_partition_query=True,
            )
        )
        return items

    def _vector_distance_function(self) -> str:
        """Return the container's configured Cosmos ``distanceFunction`` (cached).

        Read from the container's vector embedding policy (``container.read()``) —
        the authoritative, immutable source set when the container was created.
        Drives the ORDER BY direction and similarity-threshold comparisons so dedup
        never silently assumes cosine. Falls back to cosine when the policy can't be
        read (e.g. ``__new__``-built test instances with mocked containers).
        """
        fn = getattr(self, "_distance_function_cache", None)
        if fn is not None:
            return fn
        try:
            props = self._memories_container.read()
        except Exception:
            # Transient read failure (429/503/connection) is indistinguishable from
            # "no policy" once we drop to None — so DON'T cache here. Returning an
            # uncached cosine default lets the next call self-heal; caching it would
            # pin cosine for the instance's life and silently mis-handle a euclidean
            # container (cosine bands applied to euclidean distances → data loss).
            logger.debug(
                "vector dedup: could not read container vector policy; defaulting to cosine (not cached)",
                exc_info=True,
            )
            return "cosine"
        fn = distance_function_from_container_properties(props)
        self._distance_function_cache = fn
        return fn

    def _warn_euclidean_autodrop_once(self, distance_function: str) -> None:
        """One-shot WARN that the near-exact vector auto-drop is disabled.

        The ``DEDUP_SIM_HIGH`` thresholds are cosine-calibrated; on euclidean
        the destructive auto-drop is skipped (borderline tagging + LLM reconcile
        still run). Logged once per pipeline instance to avoid hot-path spam.
        """
        if getattr(self, "_warned_euclidean_autodrop", False):
            return
        self._warned_euclidean_autodrop = True
        logger.warning(
            "Container distanceFunction=%r: near-exact vector auto-drop is "
            "cosine-calibrated and has been DISABLED for this distance function. "
            "Duplicate detection falls back to borderline tagging + LLM reconcile. "
            "Use cosine/dotproduct embeddings for vector-floor auto-dedup.",
            distance_function,
        )

    def _vector_candidates(
        self,
        *,
        user_id: str,
        embedding: list[float],
        memory_type: str,
        top_k: int,
        exclude_ids: set[str],
    ) -> list[dict[str, Any]]:
        """Return nearest active same-type memories using Cosmos VectorDistance."""
        if not user_id or not embedding or top_k < 1:
            return []
        capped_top_k = top_literal(top_k, name="_vector_candidates.top_k")
        distance_function = self._vector_distance_function()
        order_direction = vector_order_direction(distance_function)
        field = "embedding"
        query = (
            f"SELECT TOP {capped_top_k} c.id, c.content, c.type, "
            f"VectorDistance(c.{field}, @vec) AS score "
            "FROM c WHERE c.user_id = @user_id "
            "AND c.type = @memory_type "
            f"AND {_ACTIVE_DOC_FILTER} "
            f"AND IS_DEFINED(c.{field}) "
            # Cosmos orders ORDER BY VectorDistance() most-similar-first per the
            # container's distanceFunction; an explicit ASC/DESC is rejected (BadRequest).
            f"ORDER BY VectorDistance(c.{field}, @vec)"
        )
        rows = list(
            self._memories_container.query_items(
                query=query,
                parameters=[
                    {"name": "@user_id", "value": user_id},
                    {"name": "@memory_type", "value": memory_type},
                    {"name": "@vec", "value": embedding},
                ],
                enable_cross_partition_query=True,
            )
        )
        excluded = set(exclude_ids or set())
        candidates = [
            {
                "id": row.get("id"),
                "content": row.get("content"),
                "type": row.get("type"),
                "score": float(row.get("score") or 0.0),
            }
            for row in rows
            if row.get("id") and row.get("id") not in excluded
        ]
        # Most-similar-first: descending score for cosine/dotproduct, ascending for euclidean.
        candidates.sort(
            key=lambda row: row.get("score", 0.0),
            reverse=order_direction == "DESC",
        )
        return candidates

    def _query_active_memories(
        self,
        user_id: str,
        memory_type: str,
        *,
        limit: int | None = None,
        tagged_only: bool = False,
    ) -> list[dict[str, Any]]:
        top_clause = f"TOP {top_literal(limit, name='_query_active_memories.limit')} " if limit else ""
        tag_clause = "AND ARRAY_CONTAINS(c.tags, @tag) " if tagged_only else ""
        query = (
            f"SELECT {top_clause}* FROM c WHERE c.user_id = @user_id "
            "AND c.type = @memory_type "
            f"AND {_ACTIVE_DOC_FILTER} "
            f"{tag_clause}"
            "ORDER BY c.created_at DESC"
        )
        parameters = [
            {"name": "@user_id", "value": user_id},
            {"name": "@memory_type", "value": memory_type},
        ]
        if tagged_only:
            parameters.append({"name": "@tag", "value": "sys:dup-candidate"})
        return list(
            self._memories_container.query_items(
                query=query,
                parameters=parameters,
                enable_cross_partition_query=True,
            )
        )

    def _load_memories_by_ids(
        self, user_id: str, memory_type: str, ids: Iterable[str]
    ) -> list[dict[str, Any]]:
        ids = [mid for mid in dict.fromkeys(ids) if mid]
        if not ids:
            return []
        placeholders = ", ".join(f"@id{i}" for i in range(len(ids)))
        parameters = [
            {"name": "@user_id", "value": user_id},
            {"name": "@memory_type", "value": memory_type},
        ]
        parameters.extend({"name": f"@id{i}", "value": mid} for i, mid in enumerate(ids))
        query = (
            "SELECT * FROM c WHERE c.user_id = @user_id "
            "AND c.type = @memory_type "
            f"AND c.id IN ({placeholders}) "
            f"AND {_ACTIVE_DOC_FILTER}"
        )
        return list(
            self._memories_container.query_items(
                query=query,
                parameters=parameters,
                enable_cross_partition_query=True,
            )
        )

    def _clear_dup_candidate_tags(self, docs: Iterable[dict[str, Any]]) -> None:
        for doc in docs:
            tags = [tag for tag in (doc.get("tags") or []) if tag != "sys:dup-candidate"]
            if tags == (doc.get("tags") or []):
                continue
            updated = dict(doc)
            updated["tags"] = tags
            metadata = dict(updated.get("metadata") or {})
            metadata.pop("dup_of", None)
            metadata.pop("dup_score", None)
            updated["metadata"] = metadata
            updated["updated_at"] = datetime.now(timezone.utc).isoformat()
            try:
                self._upsert_memory(updated)
            except Exception:
                logger.exception("reconcile_memories: failed to clear dup-candidate tag id=%s", doc.get("id"))

    def _upsert_memory(self, doc: dict[str, Any]) -> dict[str, Any]:
        """Upsert a fact, episodic, or procedural document to the memories container."""
        response = self._memories_container.upsert_item(body=doc)
        if isinstance(response, dict):
            return response
        return doc

    def _upsert_summary(self, doc: dict[str, Any]) -> dict[str, Any]:
        """Upsert a thread/user summary document to the summaries container."""
        response = self._summaries_container.upsert_item(body=doc)
        if isinstance(response, dict):
            return response
        return doc

    def _create_memory(self, doc: dict[str, Any]) -> dict[str, Any]:
        """Create a memory document and let Cosmos raise 409 for duplicates."""
        response = self._memories_container.create_item(body=doc)
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
            "dropped_episodic_count": 0,
        }

    @staticmethod
    def _stable_source_timestamp(items: list[dict[str, Any]]) -> str:
        timestamps = [str(item.get("created_at")) for item in items if item.get("created_at")]
        if timestamps:
            return max(timestamps)
        return datetime.now(timezone.utc).isoformat()

    def _mark_superseded(
        self,
        old_doc: dict[str, Any],
        superseder_id: str,
        *,
        reason: Literal["duplicate", "contradict", "update"],
    ) -> bool:
        """Atomically set ``superseded_by`` on ``old_doc`` via the memory store."""
        store = getattr(self, "_store", None)
        if store is not None:
            return store.mark_superseded(old_doc, superseder_id, reason=reason)
        return self._mark_superseded_via_container(old_doc, superseder_id, reason=reason)

    def _mark_superseded_via_container(
        self,
        old_doc: dict[str, Any],
        superseder_id: str,
        *,
        reason: Literal["duplicate", "contradict", "update"],
    ) -> bool:
        """Fallback for tests constructing the class via ``__new__``."""
        from azure.core import MatchConditions
        from azure.cosmos.exceptions import CosmosAccessConditionFailedError

        new_doc = {
            **old_doc,
            "superseded_by": superseder_id,
            "supersede_reason": reason,
            "superseded_at": datetime.now(timezone.utc).isoformat(),
        }
        try:
            if old_doc.get("_etag"):
                self._memories_container.replace_item(
                    item=new_doc["id"],
                    body=new_doc,
                    match_condition=MatchConditions.IfNotModified,
                    etag=old_doc["_etag"],
                )
            else:
                self._memories_container.upsert_item(body=new_doc)
            return True
        except CosmosAccessConditionFailedError:
            logger.info(
                "supersede skipped (concurrent writer won) id=%s superseder=%s",
                old_doc.get("id"),
                superseder_id,
            )
            return False

    @staticmethod
    def _parse_llm_json(text: str | None) -> dict[str, Any]:
        return parse_llm_json(text)

    def extract_memories_dry(
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
            query = (
                "SELECT * FROM c WHERE c.user_id = @user_id "
                "AND c.thread_id = @thread_id AND c.type = 'turn' "
                "AND (NOT IS_DEFINED(c.extracted_at) OR IS_NULL(c.extracted_at))"
            )
            parameters: list[dict[str, Any]] = [
                {"name": "@user_id", "value": user_id},
                {"name": "@thread_id", "value": thread_id},
            ]
            items = list(
                self._turns_container.query_items(
                    query=query,
                    parameters=parameters,
                    partition_key=[user_id, thread_id],
                )
            )
        else:
            items = list(turns)

        items.sort(key=lambda m: m.get("created_at", ""), reverse=True)
        if recent_k is not None:
            items = items[:recent_k]
        items.reverse()

        if not items:
            logger.warning("extract_memories_dry no memories found user_id=%s thread_id=%s", user_id, thread_id)
            return {"facts": [], "episodic": [], "updates": [], "processed_turn_docs": []}

        existing_for_hashes = self._load_existing_memories(user_id, ["fact"])
        existing_fact_hashes: set[str] = {
            m["content_hash"] for m in existing_for_hashes if m.get("type") == "fact" and m.get("content_hash")
        }
        transcript = self._build_transcript(items)
        existing = existing_for_hashes
        if threshold_config.get_dedup_context_vector_enabled():
            user_turns_text = "\n".join(
                str(it.get("content", "")) for it in items if it.get("role") == "user"
            ).strip()
            context_query = user_turns_text or transcript
            existing = self._store.search(
                search_terms=context_query,
                user_id=user_id,
                memory_types=["fact"],
                top_k=threshold_config.get_dedup_context_topk(),
            )
        if existing:
            existing_text = "\n".join(
                f"- [ID: {mem['id']}] {mem.get('content', '')} "
                f"(type={mem.get('type', 'fact')}, salience={mem.get('salience', 'N/A')})"
                for mem in existing
            )
        else:
            existing_text = "(none)"

        response_text = self._run_prompty(
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
        dropped_episodic_count = 0

        for fact in facts:
            text = fact.get("text")
            if not text:
                logger.warning("extract_memories: dropping malformed fact (missing 'text'): %r", fact)
                continue

            new_content_hash = compute_content_hash(text)
            if new_content_hash in existing_fact_hashes:
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

            fact_docs.append(self._validate_extracted_doc(doc))
            existing_fact_hashes.add(new_content_hash)

        for ep in episodic:
            scope_type_raw = ep.get("scope_type")
            scope_value_raw = ep.get("scope_value")
            scope_type = scope_type_raw.strip() if isinstance(scope_type_raw, str) else None
            scope_value = scope_value_raw.strip() if isinstance(scope_value_raw, str) else None
            if not scope_type or not scope_value:
                logger.warning(
                    "extract_memories: dropping malformed episodic (missing scope_type/scope_value) "
                    "user_id=%s thread_id=%s reason=malformed_scope payload=%r",
                    user_id,
                    thread_id,
                    ep,
                )
                dropped_episodic_count += 1
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
        if dropped_episodic_count:
            updates.append({"op": "stats", "dropped_episodic_count": dropped_episodic_count})

        check_extracted_fact_grounding(
            fact_docs,
            items,
            existing,
            user_id=user_id,
            thread_id=thread_id,
            logger=logger,
        )

        result = {
            "facts": fact_docs,
            "episodic": episodic_docs,
            "updates": updates,
            "processed_turn_docs": items,
        }
        logger.info(
            "extract_memories_dry completed user_id=%s thread_id=%s fact_docs=%d episodic_docs=%d updates=%d",
            user_id,
            thread_id,
            len(fact_docs),
            len(episodic_docs),
            len(updates),
        )
        return result

    def dedup_extracted_memories(
        self,
        user_id: str,
        extracted: dict[str, list[dict[str, Any]]],
    ) -> dict[str, list[dict[str, Any]]]:
        """Apply the gated Stage-3 vector dedup ladder to extracted docs."""
        if not threshold_config.get_dedup_vector_enabled():
            return extracted
        if not user_id:
            raise ValidationError("user_id is required")
        if not isinstance(extracted, dict):
            raise ValidationError("extracted must be a dict")

        high = threshold_config.get_dedup_sim_high()
        low = threshold_config.get_dedup_sim_low()
        top_k = threshold_config.get_dedup_candidate_topk()
        distance_function = self._vector_distance_function()
        autodrop_ok = vector_autodrop_supported(distance_function)
        if not autodrop_ok:
            self._warn_euclidean_autodrop_once(distance_function)
        vector_dedup_skipped = 0
        dup_candidates_tagged = 0

        result: dict[str, list[dict[str, Any]]] = {
            "facts": [dict(doc) for doc in extracted.get("facts", [])],
            "episodic": [dict(doc) for doc in extracted.get("episodic", [])],
            "updates": [dict(op) for op in extracted.get("updates", [])],
        }
        docs = [doc for bucket in ("facts", "episodic") for doc in result[bucket] if doc.get("content")]
        if not docs:
            return result

        missing_embeddings = [doc for doc in docs if not doc.get("embedding")]
        if missing_embeddings:
            embeddings = self._embed_batch([str(doc["content"]) for doc in missing_embeddings])
            for doc, embedding in zip(missing_embeddings, embeddings):
                doc["embedding"] = embedding

        kept_ids: set[str] = set()
        dropped_ids: set[str] = set()
        for doc in docs:
            doc_id = str(doc.get("id") or "")
            memory_type = str(doc.get("type") or "")
            if not doc_id or memory_type not in {"fact", "episodic"}:
                continue

            best: dict[str, Any] | None = None
            candidates = self._vector_candidates(
                user_id=user_id,
                embedding=doc.get("embedding") or [],
                memory_type=memory_type,
                top_k=top_k,
                exclude_ids=kept_ids | dropped_ids | {doc_id} | set(doc.get("supersedes_ids") or []),
            )
            if candidates:
                best = candidates[0]

            score = float(best.get("score") or 0.0) if best else 0.0
            if best and autodrop_ok and vector_similarity_at_least(score, high, distance_function):
                logger.info(
                    "vector dedup skipped new memory id=%s type=%s score=%.4f surviving_id=%s "
                    "new_content=%r surviving_content=%r",
                    doc_id,
                    memory_type,
                    score,
                    best.get("id"),
                    doc.get("content"),
                    best.get("content"),
                )
                vector_dedup_skipped += 1
                dropped_ids.add(doc_id)
                continue

            if best and vector_similarity_at_least(score, low, distance_function):
                tags = list(doc.get("tags") or [])
                if "sys:dup-candidate" not in tags:
                    tags.append("sys:dup-candidate")
                doc["tags"] = tags
                metadata = dict(doc.get("metadata") or {})
                metadata["dup_of"] = best.get("id")
                metadata["dup_score"] = score
                doc["metadata"] = metadata
                dup_candidates_tagged += 1

            kept_ids.add(doc_id)

        for bucket in ("facts", "episodic"):
            result[bucket] = [doc for doc in result[bucket] if doc.get("id") not in dropped_ids]
        if vector_dedup_skipped or dup_candidates_tagged:
            result["updates"].append(
                {
                    "op": "stats",
                    "vector_dedup_skipped": vector_dedup_skipped,
                    "dup_candidates_tagged": dup_candidates_tagged,
                }
            )
        return result

    def persist_extracted_memories(
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
            embeddings = self._embed_batch([str(doc["content"]) for doc in docs_needing_embeddings])
            for doc, embedding in zip(docs_needing_embeddings, embeddings):
                doc["embedding"] = embedding

        for doc in docs_to_create:
            # Intentionally re-validates even though extract_memories_dry did:
            # persist_extracted_memories is a public surface (recovery scripts,
            # custom processors, third-party callers) so we don't trust input
            # shape. Cost is microseconds per doc; the safety boundary is the
            # whole point.
            validated = self._validate_extracted_doc(doc)
            doc_type = validated.get("type")
            try:
                if doc_type == "episodic":
                    self._upsert_memory(validated)
                else:
                    self._create_memory(validated)
            except CosmosResourceExistsError:
                logger.info("persist_extracted_memories skipped existing id=%s", validated.get("id"))
                continue

            tags = validated.get("tags", [])
            if doc_type == "episodic":
                result["episodic_count"] += 1
            elif "sys:unclassified" in tags:
                result["unclassified_count"] += 1
            elif doc_type == "fact":
                result["fact_count"] += 1

        for op in update_ops:
            if op.get("op") == "stats":
                result["exact_dedup_skipped"] += int(op.get("exact_dedup_skipped") or 0)
                result["dropped_episodic_count"] += int(op.get("dropped_episodic_count") or 0)
                for key in ("vector_dedup_skipped", "dup_candidates_tagged"):
                    if key in op:
                        result[key] = result.get(key, 0) + int(op.get(key) or 0)

        logger.info("persist_extracted_memories completed user_id=%s counts=%s", user_id, result)

        processed_turns = extracted.get("processed_turn_docs") or []
        if processed_turns:
            marked = self._mark_turns_extracted(processed_turns)
            logger.info(
                "persist_extracted_memories marked turns as extracted user_id=%s marked=%d/%d",
                user_id,
                marked,
                len(processed_turns),
            )

        return result

    def _mark_turns_extracted(self, turn_docs: list[dict[str, Any]]) -> int:
        """Stamp ``extracted_at`` on each turn doc and upsert.

        We upsert the full doc (rather than patch) because the container
        adapter only exposes upsert. Per-turn failures are logged but do
        not raise — the worst case is one turn gets re-extracted on the
        next call, which is bounded and recoverable.
        """
        if not turn_docs:
            return 0
        now_iso = datetime.now(tz=timezone.utc).isoformat()
        marked = 0
        for turn in turn_docs:
            turn_id = turn.get("id")
            if not turn_id:
                continue
            try:
                doc_to_write = dict(turn)
                doc_to_write["extracted_at"] = now_iso
                self._turns_container.upsert_item(body=doc_to_write)
                marked += 1
            except Exception as exc:
                logger.warning(
                    "_mark_turns_extracted failed for turn_id=%s err=%s (turn may be re-extracted on next call)",
                    turn_id,
                    exc,
                )
        return marked

    def extract_memories(
        self,
        user_id: str,
        thread_id: str,
        recent_k: int | None = None,
        *,
        turns: Optional[list[dict[str, Any]]] = None,
    ) -> dict[str, int]:
        """Extract facts and episodic memories from a thread and persist them."""
        extracted = self.extract_memories_dry(user_id, thread_id, recent_k, turns=turns)
        if threshold_config.get_dedup_vector_enabled():
            extracted = self.dedup_extracted_memories(user_id, extracted)
        return self.persist_extracted_memories(user_id, extracted)

    def synthesize_procedural(
        self,
        user_id: str,
        *,
        force: bool = False,
    ) -> dict[str, Any]:
        """Synthesize the active procedural prompt for a user."""
        if not user_id:
            raise ValidationError("user_id is required")

        logger.info("synthesize_procedural started user_id=%s force=%s", user_id, force)

        def _read_latest_procedural() -> Optional[dict[str, Any]]:
            docs = list(
                self._memories_container.query_items(
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

        prior_doc = _read_latest_procedural()

        behavioral_fact_docs = list(
            self._memories_container.query_items(
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
        )
        behavioral_fact_docs = [
            doc
            for doc in behavioral_fact_docs
            if isinstance(doc.get("content"), str) and doc.get("content", "").strip()
        ]
        behavioral_fact_ids = [doc["id"] for doc in behavioral_fact_docs]

        episodic_docs = list(
            self._memories_container.query_items(
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

        name_docs = list(
            self._memories_container.query_items(
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
            response_text = self._run_prompty(
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
                self._memories_container.create_item(body=validated)
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
                latest = _read_latest_procedural()
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
            self._mark_superseded(prior_doc, new_id, reason="update")

        logger.info(
            "synthesize_procedural synthesized user_id=%s version=%d fact_count=%d episodic_count=%d",
            user_id,
            written_doc["version"],
            len(behavioral_fact_ids),
            len(source_episodic_ids),
        )
        return {"status": "synthesized", "procedural": written_doc}

    def generate_thread_summary_dry(
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
            existing_summary = self._summaries_container.read_item(item=summary_id, partition_key=[user_id, thread_id])
        except CosmosResourceNotFoundError:
            pass

        query = "SELECT * FROM c WHERE c.user_id = @user_id AND c.thread_id = @thread_id AND c.type = 'turn'"
        parameters: list[dict[str, Any]] = [
            {"name": "@user_id", "value": user_id},
            {"name": "@thread_id", "value": thread_id},
        ]
        if existing_summary:
            since = existing_summary["updated_at"]
            query += " AND c.created_at > @since"
            parameters.append({"name": "@since", "value": since})

        items = list(
            self._turns_container.query_items(
                query=query,
                parameters=parameters,
                partition_key=[user_id, thread_id],
            )
        )

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
            response_text = self._run_prompty(
                "summarize_update.prompty",
                inputs={"prior_summary": prior_text, "transcript": transcript},
            )
            summary_prompt_filename = "summarize_update.prompty"
        else:
            response_text = self._run_prompty("summarize.prompty", inputs={"transcript": transcript})
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
            "type": "thread_summary",
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

    def persist_thread_summary(
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
            doc["embedding"] = self._embed_one(doc["content"])
        validated = construct_internal(ThreadSummaryRecord, doc).to_doc()
        stored = self._upsert_summary(validated)
        logger.info("persist_thread_summary completed id=%s", validated.get("id"))
        return stored

    def generate_thread_summary(
        self,
        user_id: str,
        thread_id: str,
        recent_k: int | None = None,
    ) -> dict[str, Any]:
        """Generate or incrementally update a thread summary and persist it."""
        summary_doc = self.generate_thread_summary_dry(user_id, thread_id, recent_k=recent_k)
        return self.persist_thread_summary(user_id, thread_id, summary_doc)

    def generate_user_summary_dry(
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
            existing_summary = self._summaries_container.read_item(
                item=user_summary_id,
                partition_key=[user_id, "__user_summary__"],
            )
        except CosmosResourceNotFoundError:
            pass

        query_predicate = "c.user_id = @user_id"
        parameters: list[dict[str, Any]] = [{"name": "@user_id", "value": user_id}]
        if existing_summary:
            since = existing_summary["updated_at"]
            query_predicate += " AND c.created_at > @since"
            parameters.append({"name": "@since", "value": since})

        memories_query = f"SELECT * FROM c WHERE {query_predicate} AND c.type IN ('fact', 'episodic', 'procedural')"
        summaries_query = f"SELECT * FROM c WHERE {query_predicate} AND c.type = 'thread_summary'"

        items = list(
            self._memories_container.query_items(
                query=memories_query,
                parameters=parameters,
                enable_cross_partition_query=True,
            )
        )
        items.extend(
            self._summaries_container.query_items(
                query=summaries_query,
                parameters=parameters,
                enable_cross_partition_query=True,
            )
        )

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
            response_text = self._run_prompty(
                "user_summary_update.prompty",
                inputs={"prior_summary": prior_text, "transcript": transcript},
            )
            prompt_filename = "user_summary_update.prompty"
        else:
            response_text = self._run_prompty("user_summary.prompty", inputs={"transcript": transcript})
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

    def persist_user_summary(
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
            doc["embedding"] = self._embed_one(doc["content"])
        validated = construct_internal(UserSummaryRecord, doc).to_doc()
        stored = self._upsert_summary(validated)
        logger.info("persist_user_summary completed id=%s", validated.get("id"))
        return stored

    def generate_user_summary(
        self,
        user_id: str,
        thread_ids: list[str] | None = None,
        recent_k: int | None = None,
    ) -> dict[str, Any]:
        """Generate or incrementally update a user summary and persist it."""
        summary_doc = self.generate_user_summary_dry(user_id, thread_ids=thread_ids, recent_k=recent_k)
        return self.persist_user_summary(user_id, summary_doc)

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

    def _build_candidate_clusters(
        self, user_id: str, memory_type: str, n: int
    ) -> tuple[list[list[dict[str, Any]]], int, list[dict[str, Any]]]:
        """Cluster dup-candidate seeds (+ vector neighbors) into connected components.

        Returns ``(clusters, node_count, seeds)`` where each cluster has >= 2 members,
        ``node_count`` is the total distinct memories pulled into the graph (used to
        report ``reconcile_llm_calls_saved``), and ``seeds`` is the tagged seed scan
        (so the caller can clear stale tags on orphan seeds that never clustered).
        The seed scan is bounded to ``n`` so a single cluster can never exceed the
        reconcile prompt's pool cap.
        """
        cluster_sim = threshold_config.get_dedup_cluster_sim()
        top_k = threshold_config.get_dedup_candidate_topk()
        distance_function = self._vector_distance_function()
        seeds = self._query_active_memories(user_id, memory_type, limit=n, tagged_only=True)
        nodes_by_id: dict[str, dict[str, Any]] = {doc["id"]: doc for doc in seeds if doc.get("id")}
        edge_pairs: set[tuple[str, str]] = set()

        dup_of_ids = [
            (doc.get("metadata") or {}).get("dup_of")
            for doc in seeds
            if isinstance(doc.get("metadata"), dict) and (doc.get("metadata") or {}).get("dup_of")
        ]
        for doc in self._load_memories_by_ids(user_id, memory_type, dup_of_ids):
            if doc.get("id"):
                nodes_by_id[doc["id"]] = doc

        for seed in seeds:
            seed_id = seed.get("id")
            if not seed_id:
                continue
            dup_of = (seed.get("metadata") or {}).get("dup_of") if isinstance(seed.get("metadata"), dict) else None
            if dup_of:
                edge_pairs.add(tuple(sorted((seed_id, dup_of))))
            if not seed.get("embedding"):
                continue
            candidates = self._vector_candidates(
                user_id=user_id,
                embedding=seed.get("embedding") or [],
                memory_type=memory_type,
                top_k=top_k,
                exclude_ids={seed_id},
            )
            candidate_ids = [
                row["id"]
                for row in candidates
                if row.get("id")
                and vector_similarity_at_least(row.get("score", 0.0), cluster_sim, distance_function)
            ]
            for cid in candidate_ids:
                edge_pairs.add(tuple(sorted((seed_id, cid))))
            for doc in self._load_memories_by_ids(user_id, memory_type, candidate_ids):
                if doc.get("id"):
                    nodes_by_id[doc["id"]] = doc

        node_ids = list(nodes_by_id)
        node_id_set = set(node_ids)
        for node_id in node_ids:
            node = nodes_by_id[node_id]
            if not node.get("embedding"):
                continue
            for row in self._vector_candidates(
                user_id=user_id,
                embedding=node.get("embedding") or [],
                memory_type=memory_type,
                top_k=top_k,
                exclude_ids={node_id},
            ):
                neighbor_id = row.get("id")
                if neighbor_id in node_id_set and vector_similarity_at_least(
                    float(row.get("score") or 0.0), cluster_sim, distance_function
                ):
                    edge_pairs.add(tuple(sorted((node_id, neighbor_id))))

        adjacency: dict[str, set[str]] = {node_id: set() for node_id in node_ids}
        for left_id, right_id in edge_pairs:
            if left_id != right_id and left_id in adjacency and right_id in adjacency:
                adjacency[left_id].add(right_id)
                adjacency[right_id].add(left_id)

        clusters: list[list[dict[str, Any]]] = []
        seen: set[str] = set()
        for node_id in node_ids:
            if node_id in seen:
                continue
            stack = [node_id]
            component: list[str] = []
            seen.add(node_id)
            while stack:
                current = stack.pop()
                component.append(current)
                for neighbor in adjacency.get(current, set()):
                    if neighbor not in seen:
                        seen.add(neighbor)
                        stack.append(neighbor)
            if len(component) >= 2:
                # Cap cluster size at the reconcile pool limit: lowering the cluster
                # threshold can chain many facts into one giant transitive component
                # that would blow the prompt cap; keep the most-recent ``n``.
                if len(component) > n:
                    component = component[:n]
                clusters.append([nodes_by_id[cid] for cid in component])
        return clusters, len(nodes_by_id), seeds

    def _reconcile_candidate_mode(
        self, user_id: str, *, n: int, memory_type: str, started_at: float
    ) -> dict[str, int]:
        # Candidate clustering only. The periodic full-pool backstop that catches
        # dissimilar-embedding contradictions ("vegetarian" vs "loves steak") is
        # driven by the caller via ``full_rebuild`` on a PERSISTED-counter cadence
        # (in-process auto-trigger + durable change-feed), not an in-memory sweep
        # counter — the latter reset per worker/process and never fired reliably on
        # the Function-App backend.
        clusters, node_count, seeds = self._build_candidate_clusters(user_id, memory_type, n)
        aggregate = {"kept": 0, "merged": 0, "contradicted": 0}
        clustered_ids: set[str] = set()
        for cluster in clusters:
            # Mark members as clustered BEFORE the LLM call so a failed cluster's
            # seeds are not treated as orphans below (which would clear their tags
            # and prevent a retry). A truncated/malformed LLM response on one
            # cluster must not abort the sweep or starve the remaining clusters.
            clustered_ids.update(doc["id"] for doc in cluster if doc.get("id"))
            try:
                counts, consumed = self._reconcile_pool(user_id, memory_type, cluster)
            except Exception as exc:
                logger.warning(
                    "reconcile_memories: cluster reconcile failed user_id=%s memory_type=%s; "
                    "skipping cluster, tags retained for next sweep: %s",
                    user_id,
                    memory_type,
                    exc,
                )
                continue
            for key in aggregate:
                aggregate[key] += int(counts.get(key, 0))
            # Clear dup-candidate tags only on survivors. Re-upserting a doc that
            # was just superseded (duplicate source or contradiction loser) would
            # resurrect it, since the in-memory cluster copy lacks superseded_by.
            survivors = [doc for doc in cluster if doc.get("id") and doc["id"] not in consumed]
            self._clear_dup_candidate_tags(survivors)
        # Orphan seeds: tagged dup-candidates that never joined a cluster have no
        # near-duplicate, so clear the stale tag — otherwise every future sweep
        # re-scans them as seeds and they accumulate forever.
        orphan_seeds = [seed for seed in seeds if seed.get("id") and seed["id"] not in clustered_ids]
        self._clear_dup_candidate_tags(orphan_seeds)
        aggregate["reconcile_clusters_sent"] = len(clusters)
        aggregate["reconcile_llm_calls_saved"] = max(0, node_count - len(clusters))
        logger.info(
            "reconcile_memories candidate completed user_id=%s memory_type=%s result=%s",
            user_id,
            memory_type,
            aggregate,
        )
        self._emit_reconcile_outcome(
            started_at=started_at,
            user_id=user_id,
            candidates=node_count,
            result=aggregate,
        )
        return aggregate

    def reconcile_memories(
        self, user_id: str, n: int = 50, *, memory_type: str = "fact", full_rebuild: bool = False
    ) -> dict[str, int]:
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

        ``full_rebuild`` (set by the explicit public ``reconcile()`` entrypoint)
        makes candidate mode cluster the whole active pool instead of only
        Stage-3-tagged seeds, so a user-triggered reconcile dedups every active
        fact. Automatic background sweeps leave it ``False`` for the cheap
        tagged-only path.

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
        if memory_type not in {"fact", "episodic", "procedural"}:
            raise ValidationError(f"memory_type must be one of fact, episodic, procedural; got {memory_type!r}")
        if memory_type == "procedural":
            result = {
                "kept": 0,
                "merged": 0,
                "contradicted": 0,
                "reconcile_clusters_sent": 0,
                "reconcile_llm_calls_saved": 0,
            }
            logger.info("reconcile_memories procedural no-op user_id=%s result=%s", user_id, result)
            return result

        started_at = time.monotonic()
        logger.info("reconcile_memories started user_id=%s n=%d memory_type=%s", user_id, n, memory_type)

        # Explicit user-triggered reconcile (full_rebuild) always takes the
        # full-pool single-LLM-pass path: it sees every active fact together, so it
        # catches contradictions that aren't vector-similar (e.g. "vegetarian" vs
        # "loves steak") — which candidate clustering, keyed on near-duplicate
        # similarity, would never group. Automatic sweeps use cheap candidate mode.
        if threshold_config.get_dedup_reconcile_mode() == "candidate" and not full_rebuild:
            return self._reconcile_candidate_mode(
                user_id, n=n, memory_type=memory_type, started_at=started_at
            )

        facts = self._active_memories_for_reconcile(user_id, memory_type, n)
        result, consumed = self._reconcile_pool(user_id, memory_type, facts)
        # Clear dup-candidate tags on survivors so an explicit reconcile(full_rebuild=True)
        # doesn't leave stale sys:dup-candidate/dup_of metadata on user-visible memories.
        survivors = [doc for doc in facts if doc.get("id") and doc["id"] not in consumed]
        self._clear_dup_candidate_tags(survivors)
        self._emit_reconcile_outcome(
            started_at=started_at,
            user_id=user_id,
            candidates=len(facts),
            result=result,
        )
        return result

    def _active_memories_for_reconcile(
        self, user_id: str, memory_type: str, n: int
    ) -> list[dict[str, Any]]:
        # ---- Load up to N most recent active memories ----
        # ORDER BY c.created_at DESC keeps the TOP cap deterministic across
        # physical partitions and matches the dedup prompt's tiebreaker
        # ("more recent created_at first"). Cosmos's _ts is the last-write
        # timestamp, which would diverge from created_at after any UPDATE.
        query = (
            f"SELECT TOP {top_literal(n, name='reconcile_memories.n')} * FROM c "
            "WHERE c.user_id = @user_id "
            "AND c.type = @memory_type "
            f"AND {_ACTIVE_DOC_FILTER} "
            "ORDER BY c.created_at DESC"
        )
        return list(
            self._memories_container.query_items(
                query=query,
                parameters=[
                    {"name": "@user_id", "value": user_id},
                    {"name": "@memory_type", "value": memory_type},
                ],
                enable_cross_partition_query=True,
            )
        )

    def _reconcile_pool(
        self, user_id: str, memory_type: str, facts: list[dict[str, Any]]
    ) -> tuple[dict[str, int], set[str]]:
        """Reconcile an explicit pool of same-type memories in one LLM pass.

        Returns ``({"kept", "merged", "contradicted"}, consumed_ids)`` where
        ``consumed_ids`` are the source/loser ids that were actually superseded,
        so callers can skip them when clearing dup-candidate tags (re-upserting a
        superseded source would resurrect it). Does not emit telemetry — the
        caller owns the ``reconcile.outcome`` line.
        """
        # Polarity keyed on an explicit predicate so a future third type (e.g.
        # procedural, were it ever routed here) can't silently diverge from async.
        is_episodic = memory_type == "episodic"
        prompt_filename = "dedup_episodic.prompty" if is_episodic else "dedup.prompty"
        prompt_input_key = "episodics_text" if is_episodic else "facts_text"
        allow_contradictions = not is_episodic

        if len(facts) <= 1:
            logger.info(
                "reconcile_memories: %d %s memories, nothing to reconcile",
                len(facts),
                memory_type,
            )
            return {"kept": len(facts), "merged": 0, "contradicted": 0}, set()

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
        response_text = self._run_prompty(
            prompt_filename,
            inputs={prompt_input_key: facts_text},
        )
        parsed = self._parse_llm_json(response_text)

        duplicate_groups = parsed.get("duplicate_groups", []) or []
        contradicted_pairs = (parsed.get("contradicted_pairs", []) or []) if allow_contradictions else []
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
            # Strip sys:dup-candidate so the merged doc isn't reborn as a
            # permanent reconcile seed.
            merged_tags: list[str] = []
            seen_tags: set[str] = set()
            for src in source_docs:
                for t in src.get("tags", []) or []:
                    if t == "sys:dup-candidate":
                        continue
                    if t not in seen_tags:
                        seen_tags.add(t)
                        merged_tags.append(t)
            if not merged_tags:
                merged_tags = [f"sys:{memory_type}"]

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
            id_prefix = "fact_" if memory_type == "fact" else "ep_"
            merged_id = id_prefix + hashlib.sha256(merged_id_seed.encode()).hexdigest()[:32]

            try:
                merged_payload: dict[str, Any] = {
                        "id": merged_id,
                        "user_id": user_id,
                        "role": "system",
                        "type": memory_type,
                        "content": merged_content,
                        "thread_id": recent_thread_id or f"__reconciled__:{user_id}",
                        "confidence": confidence_val if confidence_val is not None else 0.5,
                        "salience": salience_val if salience_val is not None else 0.5,
                        "supersedes_ids": merged_supersedes,
                        "source_memory_ids": merged_source_memory_ids,
                        "tags": merged_tags,
                        "content_hash": merged_content_hash,
                        **self._prompt_lineage(prompt_filename),
                        "created_at": datetime.now(timezone.utc).isoformat(),
                        "updated_at": datetime.now(timezone.utc).isoformat(),
                }
                if memory_type == "fact":
                    merged_payload["metadata"] = {
                        "category": "preference",
                        "merged_via": "reconcile",
                        "merged_from_count": len(valid_source_ids),
                    }
                    record_cls = FactRecord
                else:
                    source_meta = dict(source_docs[0].get("metadata") or {})
                    source_meta.update(
                        {
                            "lesson": merged_content,
                            "merged_via": "reconcile",
                            "merged_from_count": len(valid_source_ids),
                        }
                    )
                    source_meta.setdefault("scope_type", source_meta.get("scope_type") or "general")
                    source_meta.setdefault("scope_value", source_meta.get("scope_value") or "general")
                    source_meta.setdefault("outcome_valence", source_meta.get("outcome_valence") or "neutral")
                    merged_payload["metadata"] = source_meta
                    record_cls = EpisodicRecord
                merged_record = construct_internal(record_cls, merged_payload)
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
                merged_record.embedding = self._embed_one(merged_content)
            except Exception:
                logger.exception(
                    "reconcile_memories: embedding failed for merged id=%s; "
                    "aborting duplicate group to avoid search-index hole",
                    merged_record.id,
                )
                continue

            merged_doc = merged_record.to_doc()
            try:
                merged_doc = self._upsert_memory(merged_doc)
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
                if self._mark_superseded(src_doc, merged_record.id, reason="duplicate"):
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

            if self._mark_superseded(loser_doc, resolved_winner, reason="contradict"):
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
        return result, consumed_ids

    def build_procedural_context(self, user_id: str) -> str:
        """Return the active synthesized procedural prompt for system injection."""
        if not user_id:
            raise ValidationError("user_id is required")
        query = (
            "SELECT TOP 1 c.content, c.version FROM c WHERE c.user_id = @user_id "
            "AND c.thread_id = @thread_id AND c.type = @type "
            f"AND {_ACTIVE_DOC_FILTER} "
            "ORDER BY c.version DESC"
        )
        items = list(
            self._memories_container.query_items(
                query=query,
                parameters=[
                    {"name": "@user_id", "value": user_id},
                    {"name": "@thread_id", "value": "__procedural__"},
                    {"name": "@type", "value": "procedural"},
                ],
                enable_cross_partition_query=True,
            )
        )
        if not items:
            return ""
        content = items[0].get("content")
        return content if isinstance(content, str) else ""


__all__ = ["PipelineService"]
