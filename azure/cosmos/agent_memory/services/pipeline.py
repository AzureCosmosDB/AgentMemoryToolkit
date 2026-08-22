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
import re
import time
from collections import defaultdict
from datetime import datetime, timezone
from typing import Any, Iterable, Literal, Optional

from azure.cosmos.exceptions import (
    CosmosResourceExistsError,
    CosmosResourceNotFoundError,
)
from pydantic import ValidationError as PydanticValidationError

from azure.cosmos.agent_memory import thresholds as threshold_config
from azure.cosmos.agent_memory._container_routing import ContainerKey
from azure.cosmos.agent_memory._curation import (
    CURATION_STATUS_CANDIDATE,
    apply_scope_hint_metadata,
)
from azure.cosmos.agent_memory._partitioning import (
    USER_SUMMARY_THREAD_ID,
    ensure_user_scope_fields,
    inherit_scope_fields,
    partition_key_for_scope_thread,
    partition_key_for_user_thread,
    scope_key_for_user,
    scope_values_for_scope_key,
    user_scope_parameters,
    user_scope_predicate,
)
from azure.cosmos.agent_memory._utils import (
    DEFAULT_TTL_BY_TYPE,
    compute_content_hash,
)
from azure.cosmos.agent_memory.exceptions import (
    NoSourceMemoriesError,
    ValidationError,
)
from azure.cosmos.agent_memory.logging import get_logger
from azure.cosmos.agent_memory.models import (
    TRUSTED_PROCEDURE_SOURCE_KINDS,
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
    PromptyLoader,
    _normalize_metadata_keys,
    batch_turns_by_tokens,
    build_topic_tags,
    build_transcript,
    cap_structured_summary,
    chat_text,
    clamp_unit_interval,
    created_at_sort_key,
    deterministic_episode_id,
    extract_memories_prompt_file,
    find_episode_boundary,
    is_retryable_llm_error,
    is_valid_time_pair,
    parse_llm_json,
    segment_time_bounds,
    turn_gap_seconds,
)
from azure.cosmos.agent_memory.services._pipeline_helpers import (
    is_real_number as _is_real_number,
)
from azure.cosmos.agent_memory.store._search_helpers import top_literal

logger = get_logger("azure.cosmos.agent_memory.pipeline")


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

# Safety cap on how many episodes one extract_episodes call may close in a
# single pass. Boundary evaluation runs on a small turn cadence so a backlog
# rarely exceeds one segment; the cap only bounds a pathological drain (e.g.
# first evaluation after a huge unprocessed backlog) so one call cannot fan out
# into an unbounded burst of LLM extractions.
_EPISODE_MAX_SEGMENTS_PER_RUN = 50


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
        return self._store.upsert_memory(body)

    @staticmethod
    def _apply_patch_operations(doc: dict[str, Any], patch_operations: list[dict[str, Any]]) -> dict[str, Any]:
        patched = dict(doc)
        for operation in patch_operations:
            if operation.get("op") != "set":
                raise ValueError(f"unsupported patch operation: {operation.get('op')!r}")
            path = operation.get("path")
            if not isinstance(path, str) or not path.startswith("/") or path == "/":
                raise ValueError(f"unsupported patch path: {path!r}")
            keys = [part.replace("~1", "/").replace("~0", "~") for part in path[1:].split("/")]
            target = patched
            for key in keys[:-1]:
                value = target.get(key)
                if not isinstance(value, dict):
                    value = {}
                    target[key] = value
                target = value
            target[keys[-1]] = operation.get("value")
        return patched

    def patch_item(self, *, item: str, partition_key: Any, patch_operations: list[dict[str, Any]]) -> dict[str, Any]:
        container = self._target_container()
        patch_item = getattr(container, "patch_item", None)
        if callable(patch_item):
            response = patch_item(item=item, partition_key=partition_key, patch_operations=patch_operations)
            return response if isinstance(response, dict) else self.read_item(item=item, partition_key=partition_key)
        doc = self.read_item(item=item, partition_key=partition_key)
        return self.upsert_item(body=self._apply_patch_operations(doc, patch_operations))

    def create_item(self, *, body: dict[str, Any]) -> dict[str, Any]:
        container = self._target_container()
        if container is not None and hasattr(container, "create_item"):
            response = container.create_item(body=body)
            return response if isinstance(response, dict) else body
        create = getattr(self._store, "create_item", None)
        if create is not None:
            response = create(body=body)
            return response if isinstance(response, dict) else body
        return self._store.upsert_memory(body)

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
        never initialised - happens in unit tests that build the service
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
        include_timestamp: bool = False,
    ) -> str:
        # getattr fallback covers unit tests that build PipelineService via
        # __new__ to bypass __init__ (and therefore the metadata-keys stash).
        return build_transcript(
            items,
            group_by_thread=group_by_thread,
            metadata_keys=getattr(self, "_transcript_metadata_keys", None),
            include_timestamp=include_timestamp,
        )

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
            f"SELECT {top_clause}* FROM c WHERE {user_scope_predicate()} "
            "AND c.type = @memory_type "
            f"AND {_ACTIVE_DOC_FILTER} "
            f"{tag_clause}"
            "ORDER BY c.created_at DESC"
        )
        parameters = [
            *user_scope_parameters(user_id),
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

    def _load_memories_by_ids(self, user_id: str, memory_type: str, ids: Iterable[str]) -> list[dict[str, Any]]:
        ids = [mid for mid in dict.fromkeys(ids) if mid]
        if not ids:
            return []
        placeholders = ", ".join(f"@id{i}" for i in range(len(ids)))
        parameters = [
            *user_scope_parameters(user_id),
            {"name": "@memory_type", "value": memory_type},
        ]
        parameters.extend({"name": f"@id{i}", "value": mid} for i, mid in enumerate(ids))
        query = (
            f"SELECT * FROM c WHERE {user_scope_predicate()} "
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

    def _upsert_memory(self, doc: dict[str, Any]) -> dict[str, Any]:
        """Upsert a fact, episodic, or procedural document to the memories container."""
        body = ensure_user_scope_fields(dict(doc))
        response = self._memories_container.upsert_item(body=body)
        if isinstance(response, dict):
            return response
        return body

    def _upsert_summary(self, doc: dict[str, Any]) -> dict[str, Any]:
        """Upsert a thread/user summary document to the summaries container."""
        body = ensure_user_scope_fields(dict(doc))
        response = self._summaries_container.upsert_item(body=body)
        if isinstance(response, dict):
            return response
        return body

    def _create_memory(self, doc: dict[str, Any]) -> dict[str, Any]:
        """Create a memory document and let Cosmos raise 409 for duplicates."""
        body = ensure_user_scope_fields(dict(doc))
        response = self._memories_container.create_item(body=body)
        return response if isinstance(response, dict) else body

    @staticmethod
    def _empty_extract_counts() -> dict[str, int]:
        return {
            "fact_count": 0,
            "episodic_count": 0,
            "updated_count": 0,
            "contradicted_count": 0,
            "exact_dedup_skipped": 0,
            "dropped_episodic_count": 0,
            "deferred_turn_count": 0,
            "quarantined_turn_count": 0,
        }

    @staticmethod
    def _stable_source_timestamp(items: list[dict[str, Any]]) -> str:
        timestamps = [str(item.get("created_at")) for item in items if item.get("created_at")]
        if timestamps:
            return max(timestamps)
        return datetime.now(timezone.utc).isoformat()

    @staticmethod
    def _inherit_source_scope(doc: dict[str, Any], sources: list[dict[str, Any]], user_id: str) -> dict[str, Any]:
        if sources:
            return inherit_scope_fields(doc, sources[0], fallback_user_id=user_id)
        return ensure_user_scope_fields(doc)

    @staticmethod
    def _apply_scope_hint(doc: dict[str, Any], source: dict[str, Any]) -> dict[str, Any]:
        try:
            metadata = apply_scope_hint_metadata(dict(doc.get("metadata") or {}), source)
        except ValidationError as exc:
            logger.debug("ignoring invalid scope hint: %s", exc)
            return doc
        if metadata != (doc.get("metadata") or {}):
            doc["metadata"] = metadata
            if doc.get("type") == "procedural":
                doc["metadata"].setdefault("curation_status", CURATION_STATUS_CANDIDATE)
            else:
                doc["status"] = CURATION_STATUS_CANDIDATE
        return doc

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

    def extract_memories_durable(
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

        logger.info("extract_memories_durable started user_id=%s thread_id=%s", user_id, thread_id)

        if turns is None:
            query = (
                f"SELECT * FROM c WHERE {user_scope_predicate()} "
                "AND c.thread_id = @thread_id AND c.type = 'turn' "
                "AND (NOT IS_DEFINED(c.extracted_at) OR IS_NULL(c.extracted_at))"
            )
            parameters: list[dict[str, Any]] = [
                *user_scope_parameters(user_id),
                {"name": "@thread_id", "value": thread_id},
            ]
            items = list(
                self._turns_container.query_items(
                    query=query,
                    parameters=parameters,
                    partition_key=partition_key_for_user_thread(user_id, thread_id),
                )
            )
        else:
            items = list(turns)

        items.sort(key=lambda m: m.get("created_at", ""), reverse=True)
        if recent_k is not None:
            items = items[:recent_k]
        items.reverse()

        if not items:
            logger.warning("extract_memories_durable no memories found user_id=%s thread_id=%s", user_id, thread_id)
            return {"facts": [], "episodic": [], "updates": [], "processed_turn_docs": []}

        # Exact-duplicate detection is in-batch only: a content_hash seen earlier
        # in THIS extraction is skipped. Cross-turn / cross-run exact duplicates
        # are handled at write time by the deterministic-id create (a repeat of
        # the same fact collides on id and is skipped with a 409), so there is no
        # per-extract query to preload the user's existing fact hashes.
        existing_fact_hashes: set[str] = set()

        # Token-bounded, per-batch extraction. Each batch is an independent LLM
        # call, so (a) each stays small enough to extract faithfully and (b) a
        # single poisoned turn fails only its own batch. Turns from succeeded and
        # quarantined (non-retryable, e.g. content-filter) batches go into
        # ``processed_turns``; the in-process caller marks them ``extracted_at``
        # so they are never re-processed (the Durable backend instead advances a
        # count-based watermark). Turns from batches that fail with a *retryable*
        # error are left out and retried on the next run.
        batches = batch_turns_by_tokens(items, threshold_config.get_extraction_batch_max_tokens())
        facts: list[dict[str, Any]] = []
        processed_turns: list[dict[str, Any]] = []
        deferred_turn_count = 0
        quarantined_turn_count = 0
        extract_prompt = extract_memories_prompt_file()
        for batch in batches:
            batch_transcript = self._build_transcript(batch, include_timestamp=True)
            try:
                response_text = self._run_prompty(extract_prompt, inputs={"transcript": batch_transcript})
                parsed = self._parse_llm_json(response_text)
                facts.extend(parsed.get("facts", []))
                processed_turns.extend(batch)
            except Exception as exc:  # noqa: BLE001
                if is_retryable_llm_error(exc):
                    deferred_turn_count += len(batch)
                    logger.warning(
                        "extract_memories: deferring %d turns after retryable extraction error "
                        "(will retry next run) user_id=%s thread_id=%s err=%s",
                        len(batch),
                        user_id,
                        thread_id,
                        exc,
                    )
                else:
                    processed_turns.extend(batch)
                    quarantined_turn_count += len(batch)
                    logger.warning(
                        "extract_memories: quarantining %d turns after non-retryable extraction error "
                        "(e.g. content filter) - marking extracted so they do not re-poison future runs "
                        "user_id=%s thread_id=%s err=%s",
                        len(batch),
                        user_id,
                        thread_id,
                        exc,
                    )

        doc_timestamp = self._stable_source_timestamp(items)
        fact_docs: list[dict[str, Any]] = []
        updates: list[dict[str, Any]] = []
        exact_dedup_skipped = 0

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
            raw_source = fact.get("source")
            if raw_source is not None and raw_source not in ("user", "agent"):
                logger.debug(
                    "extract_memories: coercing invalid fact source=%r to 'user' user_id=%s",
                    raw_source,
                    user_id,
                )
            fact_source = raw_source if raw_source in ("user", "agent") else "user"
            source_tags = ["sys:agent-fact"] if fact_source == "agent" else []
            confidence = fact.get("confidence")
            doc: dict[str, Any] = {
                "id": det_id,
                "user_id": user_id,
                "thread_id": thread_id,
                "role": "system",
                "type": "fact",
                "content": text,
                "content_hash": new_content_hash,
                "confidence": clamp_unit_interval(confidence, 0.5),
                **self._prompt_lineage(extract_prompt),
                "metadata": {
                    "category": fact.get("category") or "other",
                    "temporal_context": fact.get("temporal_context"),
                    "source": fact_source,
                },
                "salience": clamp_unit_interval(fact.get("salience"), 0.5),
                "tags": ["sys:fact", "sys:auto-extracted"] + source_tags + topic_tags,
                "created_at": doc_timestamp,
                "updated_at": doc_timestamp,
            }

            self._apply_scope_hint(doc, fact)
            self._inherit_source_scope(doc, items, user_id)
            fact_docs.append(self._validate_extracted_doc(doc))
            existing_fact_hashes.add(new_content_hash)

        if exact_dedup_skipped:
            updates.append({"op": "stats", "exact_dedup_skipped": exact_dedup_skipped})
        if deferred_turn_count or quarantined_turn_count:
            updates.append(
                {
                    "op": "stats",
                    "deferred_turn_count": deferred_turn_count,
                    "quarantined_turn_count": quarantined_turn_count,
                }
            )

        result = {
            "facts": fact_docs,
            "episodic": [],
            "updates": updates,
            "processed_turn_docs": processed_turns,
        }
        logger.info(
            "extract_memories_durable completed user_id=%s thread_id=%s fact_docs=%d updates=%d",
            user_id,
            thread_id,
            len(fact_docs),
            len(updates),
        )
        return result

    def _build_episode_transcript(self, items: list[dict[str, Any]]) -> str:
        """Build a timestamped transcript that exposes real turn ids to the LLM."""
        transcript_items: list[dict[str, Any]] = []
        for index, item in enumerate(items, start=1):
            turn_id = str(item.get("id") or f"turn-{index}")
            copied = dict(item)
            copied["content"] = f"Turn {turn_id}: {item.get('content', '')}"
            transcript_items.append(copied)
        return self._build_transcript(transcript_items, include_timestamp=True)

    @staticmethod
    def _ground_episode_events(
        events: Any,
        *,
        turn_ids: list[str],
    ) -> tuple[list[dict[str, Any]], list[str]]:
        """Normalize event source ids to real ids from the current turn window."""
        valid_turn_ids = set(turn_ids)
        label_to_id = {f"turn-{i}": turn_id for i, turn_id in enumerate(turn_ids, start=1)}
        grounded_events: list[dict[str, Any]] = []
        source_turn_ids: list[str] = []
        for event in events if isinstance(events, list) else []:
            if not isinstance(event, dict):
                continue
            grounded = dict(event)
            grounded_sources: list[str] = []
            for raw_source in event.get("source_turn_ids") or []:
                source = str(raw_source).strip()
                mapped = source if source in valid_turn_ids else label_to_id.get(source.lower())
                if mapped and mapped not in grounded_sources:
                    grounded_sources.append(mapped)
                if mapped and mapped not in source_turn_ids:
                    source_turn_ids.append(mapped)
            grounded["source_turn_ids"] = grounded_sources
            grounded_events.append(grounded)
        return grounded_events, source_turn_ids

    def _build_episode_docs(
        self,
        user_id: str,
        thread_id: str,
        items: list[dict[str, Any]],
        *,
        segment_key: str,
    ) -> list[dict[str, Any]]:
        """Run the episode-extraction prompt over one bounded, already-closed turn
        segment and return episode docs (no embeddings, no writes).

        Episode ids are DETERMINISTIC from the segment identity (its turn range)
        plus each episode's ordinal - not the summary text - so a re-run over the
        same segment (e.g. after a crash between persist and turn-stamping)
        collides on id and is skipped rather than duplicated.
        """
        if not items:
            return []

        transcript = self._build_episode_transcript(items)
        response_text = self._run_prompty("extract_episode.prompty", inputs={"transcript": transcript})
        parsed = self._parse_llm_json(response_text)
        episodes = parsed.get("episodes", [])
        if not isinstance(episodes, list):
            logger.warning(
                "_build_episode_docs dropping malformed response user_id=%s thread_id=%s payload=%r",
                user_id,
                thread_id,
                parsed,
            )
            return []

        doc_timestamp = self._stable_source_timestamp(items)
        turn_ids = [str(item.get("id")) for item in items if item.get("id")]
        # An episode's temporal span is grounded in its turn window: for the
        # conversational benchmarks each turn's created_at carries the session
        # date, so segment bounds give correct started_at/ended_at even when the
        # model omits them. We only trust model-supplied times when it provides a
        # complete, self-consistent pair.
        segment_started, segment_ended = segment_time_bounds(items)
        docs: list[dict[str, Any]] = []
        for index, episode in enumerate(episodes):
            if not isinstance(episode, dict):
                logger.warning(
                    "_build_episode_docs dropping malformed episode user_id=%s thread_id=%s payload=%r",
                    user_id,
                    thread_id,
                    episode,
                )
                continue

            summary = episode.get("summary")
            if not isinstance(summary, str) or not summary.strip():
                logger.warning(
                    "_build_episode_docs dropping malformed episode (missing summary) "
                    "user_id=%s thread_id=%s payload=%r",
                    user_id,
                    thread_id,
                    episode,
                )
                continue
            title = episode.get("title")
            if not isinstance(title, str) or not title.strip():
                logger.warning(
                    "_build_episode_docs dropping malformed episode (missing title) user_id=%s thread_id=%s payload=%r",
                    user_id,
                    thread_id,
                    episode,
                )
                continue
            if not isinstance(episode.get("events"), list):
                logger.warning(
                    "_build_episode_docs dropping malformed episode (missing events) "
                    "user_id=%s thread_id=%s payload=%r",
                    user_id,
                    thread_id,
                    episode,
                )
                continue

            events, source_turn_ids = self._ground_episode_events(episode.get("events"), turn_ids=turn_ids)
            content_hash = compute_content_hash(str(summary))
            llm_started, llm_ended = episode.get("started_at"), episode.get("ended_at")
            # Trust model-supplied times only when they form a valid, self-consistent
            # ISO pair; otherwise fall back to the grounded segment bounds rather than
            # dropping the whole episode (malformed or mixed-tz strings are common).
            if is_valid_time_pair(llm_started, llm_ended):
                started_at, ended_at = str(llm_started).strip(), str(llm_ended).strip()
            else:
                started_at, ended_at = segment_started, segment_ended
            try:
                doc = construct_internal(
                    EpisodicRecord,
                    {
                        "id": self._deterministic_episode_id(segment_key, index),
                        "user_id": user_id,
                        "thread_id": thread_id,
                        "role": "system",
                        "type": "episodic",
                        "content": summary,
                        "title": title,
                        "started_at": started_at,
                        "ended_at": ended_at,
                        "participants": episode.get("participants") or [],
                        "events": events,
                        "outcome": episode.get("outcome"),
                        "lessons": episode.get("lessons") or [],
                        "source_turn_ids": source_turn_ids,
                        "content_hash": content_hash,
                        "salience": clamp_unit_interval(episode.get("salience"), 0.5),
                        "confidence": clamp_unit_interval(episode.get("confidence"), 0.5),
                        "ttl": DEFAULT_TTL_BY_TYPE.get("episodic", 7_776_000),
                        "tags": ["sys:episodic", "sys:auto-extracted"],
                        "created_at": doc_timestamp,
                        "updated_at": doc_timestamp,
                        **self._prompt_lineage("extract_episode.prompty"),
                    },
                ).to_doc()
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "_build_episode_docs dropping malformed episode user_id=%s thread_id=%s err=%s payload=%r",
                    user_id,
                    thread_id,
                    exc,
                    episode,
                )
                continue
            self._apply_scope_hint(doc, episode)
            self._inherit_source_scope(doc, items, user_id)
            docs.append(doc)
        return docs

    @staticmethod
    def _deterministic_episode_id(segment_key: str, index: int) -> str:
        return deterministic_episode_id(segment_key, index)

    @staticmethod
    def _episode_cursor_id(user_id: str, thread_id: str) -> str:
        return f"episode_cursor_{user_id}_{thread_id}"

    def _read_episode_cursor(self, user_id: str, thread_id: str) -> tuple[str, str]:
        """Return the ``(created_at, id)`` watermark of the last turn folded into
        an episode for this thread, or ``("", "")`` when none has been extracted.

        The cursor is a single per-thread doc in the MEMORIES container - not a
        per-turn stamp - so advancing it never writes to the turns container and
        therefore never re-enters the turns change feed. That is load-bearing:
        the Durable cadence counter must observe each turn exactly once (at
        creation); a turn mutated by episodic segmentation would otherwise be
        redelivered and mis-counted. ``created_at`` is stored UTC-normalized, so
        the lexical ``>`` comparison below matches chronological order.
        """
        try:
            doc = self._memories_container.read_item(
                item=self._episode_cursor_id(user_id, thread_id),
                partition_key=partition_key_for_user_thread(user_id, thread_id),
            )
        except CosmosResourceNotFoundError:
            return "", ""
        return str(doc.get("last_episode_at") or ""), str(doc.get("last_episode_id") or "")

    def _advance_episode_cursor(self, user_id: str, thread_id: str, last_turn: dict[str, Any]) -> None:
        """Advance the episodic watermark to ``last_turn`` (newest turn just
        folded), never backwards. A single-doc write: unlike the former per-turn
        stamp it cannot partially fail and leave the open segment torn, and it
        writes to the memories container, never the change-feed-monitored turns
        container. The advance is atomic - ETag ``IfNotModified`` with a re-read
        retry - so a late, out-of-order concurrent ``episode:{u}:{t}:{count}``
        orchestration cannot regress the cursor (a regressed cursor would, with
        topic drift enabled, re-segment turns under a new segment key and
        duplicate episodes).
        """
        from azure.core import MatchConditions
        from azure.cosmos.exceptions import CosmosAccessConditionFailedError

        last_at = str(last_turn.get("created_at") or "")
        last_id = str(last_turn.get("id") or "")
        if not last_at:
            return
        cursor_id = self._episode_cursor_id(user_id, thread_id)
        partition_key = partition_key_for_user_thread(user_id, thread_id)
        for _ in range(3):
            etag: Optional[str] = None
            try:
                existing = self._memories_container.read_item(item=cursor_id, partition_key=partition_key)
            except CosmosResourceNotFoundError:
                existing = None
            if existing is not None:
                current = (
                    str(existing.get("last_episode_at") or ""),
                    str(existing.get("last_episode_id") or ""),
                )
                if (last_at, last_id) <= current:
                    return  # monotonic: never regress
                etag = existing.get("_etag")
            body = ensure_user_scope_fields(
                {
                    "id": cursor_id,
                    "type": "episode_cursor",
                    "user_id": user_id,
                    "thread_id": thread_id,
                    "last_episode_at": last_at,
                    "last_episode_id": last_id,
                    "updated_at": datetime.now(tz=timezone.utc).isoformat(),
                }
            )
            try:
                if existing is None:
                    self._memories_container.create_item(body=body)
                else:
                    self._memories_container.replace_item(
                        item=cursor_id,
                        body=body,
                        etag=etag,
                        match_condition=MatchConditions.IfNotModified,
                    )
                return
            except (CosmosResourceExistsError, CosmosAccessConditionFailedError):
                continue  # a concurrent run advanced first; re-read and re-check
        logger.debug("episode cursor advance retries exhausted user_id=%s thread_id=%s", user_id, thread_id)

    def _load_open_episode_segment(self, user_id: str, thread_id: str) -> list[dict[str, Any]]:
        """Return the open episode segment: turns created after the episodic
        watermark (not yet folded into an episode), oldest first.

        Uses a per-thread ``(created_at, id)`` cursor (see
        ``_read_episode_cursor``) instead of a per-turn stamp, so episodic
        segmentation writes nothing to the turns container and cannot perturb
        the change-feed cadence counter.
        """
        last_at, last_id = self._read_episode_cursor(user_id, thread_id)
        query = (
            f"SELECT * FROM c WHERE {user_scope_predicate()} "
            "AND c.thread_id = @thread_id AND c.type = 'turn' "
            "AND (c.created_at > @last_at "
            "OR (c.created_at = @last_at AND c.id > @last_id))"
        )
        parameters: list[dict[str, Any]] = [
            {"name": "@last_at", "value": last_at},
            {"name": "@last_id", "value": last_id},
            *user_scope_parameters(user_id),
            {"name": "@thread_id", "value": thread_id},
        ]
        items = list(
            self._turns_container.query_items(
                query=query,
                parameters=parameters,
                partition_key=partition_key_for_user_thread(user_id, thread_id),
            )
        )
        items.sort(key=created_at_sort_key)
        return items

    def _episode_segment_embeddings(self, segment: list[dict[str, Any]]) -> list[list[float]]:
        """Return one embedding per segment turn for drift detection, or ``[]`` when
        drift is disabled or embeddings are unavailable.

        Reuses a stored turn ``embedding`` when present (deployments with
        ``enable_turn_embeddings``); otherwise embeds turn text in one batch. The
        segment is bounded by ``EPISODE_MAX_TURNS`` so this stays cheap, and
        embeddings cost far less than the boundary-gated LLM extraction it guards.
        """
        if threshold_config.get_episode_topic_drift() <= 0:
            return []
        embeddings: list[Optional[list[float]]] = [
            turn.get("embedding") if isinstance(turn.get("embedding"), list) else None for turn in segment
        ]
        missing = [i for i, emb in enumerate(embeddings) if emb is None]
        if missing:
            try:
                fresh = self._embed_batch([str(segment[i].get("content") or "") for i in missing])
            except Exception as exc:  # noqa: BLE001
                logger.warning("episode drift embedding failed (%s); skipping drift this evaluation", exc)
                return []
            for pos, i in enumerate(missing):
                embeddings[i] = fresh[pos] if pos < len(fresh) else None
        if any(emb is None for emb in embeddings):
            return []
        return [emb for emb in embeddings if emb is not None]

    @staticmethod
    def _turn_gap_seconds(prev_turn: dict[str, Any], cur_turn: dict[str, Any]) -> Optional[float]:
        return turn_gap_seconds(prev_turn, cur_turn)

    def _find_episode_boundary(
        self,
        segment: list[dict[str, Any]],
        embeddings: list[list[float]],
    ) -> Optional[int]:
        return find_episode_boundary(
            segment,
            embeddings,
            max_turns=threshold_config.get_episode_max_turns(),
            idle_gap=threshold_config.get_episode_idle_gap_seconds(),
            drift=threshold_config.get_episode_topic_drift(),
            min_turns=threshold_config.get_episode_min_turns(),
        )

    def extract_episodes(
        self,
        user_id: str,
        thread_id: str,
        *,
        flush: bool = False,
    ) -> dict[str, int]:
        """Segment the open turn stream into episodes at detected boundaries.

        The open segment is every turn created after the episodic watermark (see
        ``_read_episode_cursor``). At each detected boundary - an idle time-gap, a
        topic-drift shift, or the max-size cap - the closed segment is extracted
        into one or more immutable episodes, embedded, and persisted, then the
        watermark advances past those turns so re-evaluation never re-extracts
        them. ``flush=True`` also drains the trailing open segment even without a
        boundary (end of conversation / explicit close); the default leaves the
        current, possibly-incomplete segment open. The caller never signals
        "session end" - boundaries are inferred from the stream itself.

        Idempotency: each episode's id is deterministic in its segment key and
        ordinal - not the LLM summary text - so re-running the same still-open
        segment collides on id and the duplicate write is skipped (409). The
        watermark is a single doc advanced only after a segment's episodes are
        created, so a crash between the create and the advance simply re-loads
        the same open segment next run (same turn set -> same ids -> 409); unlike
        the former per-turn stamp there is no partial-stamp state that could shift
        the boundary and admit a duplicate.
        """
        if not user_id:
            raise ValidationError("user_id is required")
        if not thread_id:
            raise ValidationError("thread_id is required")

        segment = self._load_open_episode_segment(user_id, thread_id)
        total = 0
        guard = 0
        while segment and guard < _EPISODE_MAX_SEGMENTS_PER_RUN:
            guard += 1
            embeddings = self._episode_segment_embeddings(segment)
            boundary = self._find_episode_boundary(segment, embeddings)
            if boundary is None:
                if not flush:
                    break
                boundary = len(segment)
            closing = segment[:boundary]
            if not closing:
                break
            first_id = str(closing[0].get("id") or "")
            last_id = str(closing[-1].get("id") or "")
            segment_key = _ID_SEED_SEP.join((user_id, thread_id, first_id, last_id))
            try:
                docs = self._build_episode_docs(user_id, thread_id, closing, segment_key=segment_key)
                embeddings_for_docs = self._embed_batch([str(doc["content"]) for doc in docs]) if docs else []
            except Exception as exc:  # noqa: BLE001
                if is_retryable_llm_error(exc):
                    # Transient provider error: leave the segment un-stamped and stop
                    # this run so it is retried intact next time (mirror of the fact path).
                    logger.warning(
                        "extract_episodes: deferring %d turns after retryable extraction error "
                        "(will retry next run) user_id=%s thread_id=%s err=%s",
                        len(closing),
                        user_id,
                        thread_id,
                        exc,
                    )
                    break
                # Non-retryable (e.g. content filter, context-length): quarantine the
                # poison segment - advance the watermark past it so it never
                # re-poisons future runs and the open segment cannot grow without
                # bound - then move to the next segment.
                logger.warning(
                    "extract_episodes: quarantining %d turns after non-retryable extraction error "
                    "(advancing the episode watermark past them so they do not re-poison future runs) "
                    "user_id=%s thread_id=%s err=%s",
                    len(closing),
                    user_id,
                    thread_id,
                    exc,
                )
                self._advance_episode_cursor(user_id, thread_id, closing[-1])
                segment = segment[boundary:]
                continue
            for doc, embedding in zip(docs, embeddings_for_docs):
                doc["embedding"] = embedding
                try:
                    self._create_memory(doc)
                    total += 1
                except CosmosResourceExistsError:
                    logger.info("extract_episodes idempotent skip duplicate episode id=%s", doc.get("id"))
            # Advance the episode watermark so the closed turns leave the open
            # segment even when the segment yielded no episode (nothing episodic
            # to remember) - otherwise they would be re-evaluated forever.
            self._advance_episode_cursor(user_id, thread_id, closing[-1])
            segment = segment[boundary:]
        return {"episodes": total}

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
            # Intentionally re-validates even though extract_memories_durable did:
            # persist_extracted_memories is a public surface (recovery scripts,
            # custom processors, third-party callers) so we don't trust input
            # shape. Cost is microseconds per doc; the safety boundary is the
            # whole point.
            validated = self._validate_extracted_doc(doc)
            doc_type = validated.get("type")
            try:
                self._create_memory(validated)
            except CosmosResourceExistsError:
                logger.info("persist_extracted_memories skipped existing id=%s", validated.get("id"))
                continue

            if doc_type == "episodic":
                result["episodic_count"] += 1
            elif doc_type == "fact":
                result["fact_count"] += 1

        for op in update_ops:
            if op.get("op") == "stats":
                result["exact_dedup_skipped"] += int(op.get("exact_dedup_skipped") or 0)
                result["dropped_episodic_count"] += int(op.get("dropped_episodic_count") or 0)
                for key in ("deferred_turn_count", "quarantined_turn_count"):
                    if key in op:
                        result[key] = result.get(key, 0) + int(op.get(key) or 0)

        logger.info("persist_extracted_memories completed user_id=%s counts=%s", user_id, result)

        return result

    def _mark_turns_extracted(self, turn_docs: list[dict[str, Any]]) -> int:
        """Stamp the fact-extraction ``extracted_at`` watermark on each turn doc.

        Used by the in-process fact path so a turn is not re-extracted. (Episodic
        segmentation uses a separate per-thread cursor doc - see
        ``_read_episode_cursor`` - and never stamps turns, so it does not perturb
        the change-feed cadence counter.)

        Per-turn failures are logged but do not raise - the worst case is one
        turn gets re-processed on the next call, which is bounded and
        recoverable.
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
                self._turns_container.patch_item(
                    item=turn_id,
                    partition_key=partition_key_for_scope_thread(
                        turn.get("scope_key") or f"user:{turn.get('user_id')}",
                        turn.get("thread_id"),
                        turn.get("tenant_id"),
                    ),
                    patch_operations=[{"op": "set", "path": "/extracted_at", "value": now_iso}],
                )
                marked += 1
            except Exception as exc:
                logger.warning(
                    "_mark_turns_extracted failed for turn_id=%s err=%s (turn may be re-processed on next call)",
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
        extracted = self.extract_memories_durable(user_id, thread_id, recent_k, turns=turns)
        # Capture the processed turns from the compute stage as the single source of
        # truth for stamping. Stamping happens here (not inside persist) so no
        # Persist uses this exact list for stamping after all creates finish.
        processed_turns = extracted.get("processed_turn_docs") or []
        counts = self.persist_extracted_memories(user_id, extracted)
        if processed_turns:
            marked = self._mark_turns_extracted(processed_turns)
            if marked < len(processed_turns):
                logger.warning(
                    "extract_memories stamped only %d/%d processed turns as extracted user_id=%s "
                    "thread_id=%s (unstamped turns will be re-extracted next run)",
                    marked,
                    len(processed_turns),
                    user_id,
                    thread_id,
                )
            else:
                logger.info(
                    "extract_memories stamped %d processed turns as extracted user_id=%s thread_id=%s",
                    marked,
                    user_id,
                    thread_id,
                )
        return counts

    def synthesize_procedural(
        self,
        user_id: str,
        *,
        force: bool = False,
    ) -> dict[str, Any]:
        """Extract atomic procedural memories from behavioral facts and lessons."""
        del force
        if not user_id:
            raise ValidationError("user_id is required")

        logger.info("synthesize_procedural extraction started user_id=%s", user_id)

        behavioral_fact_docs = list(
            self._memories_container.query_items(
                query=(
                    f"SELECT TOP 50 * FROM c WHERE {user_scope_predicate('@uid')} "
                    "AND c.type = @type "
                    f"AND {_ACTIVE_DOC_FILTER} "
                    "AND ((IS_DEFINED(c.metadata.category) "
                    "AND c.metadata.category IN ('preference', 'requirement')) "
                    "OR (IS_DEFINED(c.salience) AND c.salience >= @min_salience)) "
                    "ORDER BY c.created_at ASC"
                ),
                parameters=[
                    *user_scope_parameters(user_id, "@uid"),
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

        episodic_docs = list(
            self._memories_container.query_items(
                query=(
                    f"SELECT TOP 50 * FROM c WHERE {user_scope_predicate('@uid')} "
                    "AND c.type = @type "
                    f"AND {_ACTIVE_DOC_FILTER} "
                    "AND IS_DEFINED(c.lessons) AND ARRAY_LENGTH(c.lessons) > 0 "
                    "ORDER BY c.created_at ASC"
                ),
                parameters=[
                    *user_scope_parameters(user_id, "@uid"),
                    {"name": "@type", "value": "episodic"},
                ],
                enable_cross_partition_query=True,
            )
        )

        def _episodic_lessons(doc: dict[str, Any]) -> list[str]:
            lessons = doc.get("lessons")
            if isinstance(lessons, list):
                return [lesson.strip() for lesson in lessons if isinstance(lesson, str) and lesson.strip()]
            return []

        fact_lines: list[str] = []
        fact_label_to_id: dict[str, str] = {}
        for index, doc in enumerate(behavioral_fact_docs, start=1):
            label = f"fact-{index}"
            category = ""
            metadata = doc.get("metadata")
            if isinstance(metadata, dict) and isinstance(metadata.get("category"), str):
                category = metadata["category"].strip()
            fact_lines.append(f"{label} [{category or 'unknown'}]: {doc['content'].strip()}")
            if isinstance(doc.get("id"), str):
                fact_label_to_id[label] = doc["id"]

        episodic_lines: list[str] = []
        episodic_label_to_id: dict[str, str] = {}
        ep_index = 0
        for doc in episodic_docs:
            doc_id = doc.get("id")
            for lesson in _episodic_lessons(doc):
                ep_index += 1
                label = f"ep-{ep_index}"
                episodic_lines.append(f"{label}: {lesson}")
                if isinstance(doc_id, str):
                    episodic_label_to_id[label] = doc_id

        if not fact_lines and not episodic_lines:
            logger.info("synthesize_procedural unchanged user_id=%s - no procedure sources", user_id)
            return {"status": "unchanged", "procedures_created": 0}

        try:
            response_text = self._run_prompty(
                "extract_procedure.prompty",
                inputs={
                    "behavioral_facts": "\n".join(fact_lines),
                    "episodic_lessons": "\n".join(episodic_lines),
                },
            )
            parsed = self._parse_llm_json(response_text)
            procedures = parsed.get("procedures", []) if isinstance(parsed, dict) else []
            if not isinstance(procedures, list):
                procedures = []
        except Exception as exc:  # LLM/parsing quarantine: one bad call must not stop the pipeline.
            if is_retryable_llm_error(exc):
                logger.warning("synthesize_procedural deferred user_id=%s: %s", user_id, exc)
                return {"status": "deferred", "procedures_created": 0}
            logger.exception("synthesize_procedural skipped user_id=%s after non-retryable LLM error", user_id)
            return {"status": "skipped", "procedures_created": 0}

        trusted_source_kinds = {kind.value for kind in TRUSTED_PROCEDURE_SOURCE_KINDS}
        authority_by_source = {
            "explicit_user_instruction": "high",
            "organization_policy": "high",
            "observed_user_preference": "medium",
            "episode_distillation": "medium",
            "document_content": "low",
            "agent_inference": "low",
        }
        now = datetime.now(timezone.utc).isoformat()
        created = 0
        skipped = 0

        for proc in procedures:
            try:
                if not isinstance(proc, dict):
                    skipped += 1
                    continue
                name = proc.get("name")
                if not isinstance(name, str) or not name.strip():
                    skipped += 1
                    continue
                name = name.strip()

                grounded_in = proc.get("grounded_in")
                if isinstance(grounded_in, str):
                    labels = [grounded_in]
                elif isinstance(grounded_in, list):
                    labels = [label for label in grounded_in if isinstance(label, str)]
                else:
                    labels = []
                source_fact_ids = sorted({fact_label_to_id[label] for label in labels if label in fact_label_to_id})
                source_episodic_ids = sorted(
                    {episodic_label_to_id[label] for label in labels if label in episodic_label_to_id}
                )

                source_kind = proc.get("source_kind", "agent_inference")
                if not isinstance(source_kind, str):
                    source_kind = "agent_inference"
                # Episode-only grounding cannot corroborate a user/org instruction:
                # any trusted label backed solely by episodes (no behavioral fact)
                # downgrades to episode_distillation - a candidate, never an
                # auto-active policy.
                if source_episodic_ids and not source_fact_ids and source_kind in trusted_source_kinds:
                    source_kind = "episode_distillation"
                source_authority = authority_by_source.get(source_kind, "low")
                status = "active" if source_kind in trusted_source_kinds else "candidate"
                # Grounding is the trust anchor: a procedure whose ``grounded_in``
                # resolved to no persisted fact or episodic source is never
                # auto-activated, regardless of the LLM's self-declared
                # source_kind - this blocks an ungrounded self-labeled instruction
                # from being compiled into the runtime system prompt.
                if not source_fact_ids and not source_episodic_ids:
                    status = "candidate"

                summary = proc.get("summary") if isinstance(proc.get("summary"), str) else ""
                retrieval_text = proc.get("retrieval_text") if isinstance(proc.get("retrieval_text"), str) else ""
                procedure_kind = proc.get("procedure_kind", "behavioral_policy")
                proc_id = (
                    "proc_"
                    + hashlib.sha256(f"{user_id}|{procedure_kind}|{name.strip().lower()}".encode()).hexdigest()[:32]
                )
                doc: dict[str, Any] = {
                    "id": proc_id,
                    "user_id": user_id,
                    "thread_id": "__procedural__",
                    "type": "procedural",
                    "role": "system",
                    "tags": ["sys:procedural", "sys:auto-extracted"],
                    "created_at": now,
                    "updated_at": now,
                    "name": name,
                    "summary": summary.strip() or name,
                    "retrieval_text": retrieval_text.strip() or summary.strip() or name,
                    "procedure_kind": procedure_kind,
                    "activation_conditions": proc.get("activation_conditions", []),
                    "preconditions": proc.get("preconditions", []),
                    "steps": proc.get("steps", []),
                    "success_conditions": proc.get("success_conditions", []),
                    "failure_conditions": proc.get("failure_conditions", []),
                    "safety_constraints": proc.get("safety_constraints", []),
                    "status": status,
                    "priority": proc.get("priority", 0),
                    # Seed utility from the LLM's extraction confidence (the
                    # extract_procedure schema emits ``confidence``, not
                    # ``utility_score``). Procedures are create-only today with no
                    # promotion or outcome-scoring path, so this is the record's
                    # final utility value.
                    "utility_score": clamp_unit_interval(proc.get("confidence"), 0.5),
                    "successful_uses": proc.get("successful_uses", 0),
                    "failed_uses": proc.get("failed_uses", 0),
                    "source_kind": source_kind,
                    "source_authority": source_authority,
                    "source_fact_ids": source_fact_ids,
                    "source_episodic_ids": source_episodic_ids,
                    "source_turn_ids": proc.get("source_turn_ids", []),
                    "content": summary.strip() or name,
                    "version": proc.get("version", 1),
                    "metadata": {},
                    **self._prompt_lineage("extract_procedure.prompty"),
                }
                self._apply_scope_hint(doc, proc)
                validated = construct_internal(ProceduralRecord, doc).to_doc()
                validated["embedding"] = self._embed_one(validated["retrieval_text"])
                try:
                    self._create_memory(validated)
                    created += 1
                except CosmosResourceExistsError:
                    skipped += 1
            except (ValidationError, PydanticValidationError, ValueError) as exc:
                skipped += 1
                logger.warning("synthesize_procedural dropping malformed procedure user_id=%s: %s", user_id, exc)
            except Exception:
                skipped += 1
                logger.exception("synthesize_procedural failed to persist one procedure user_id=%s", user_id)

        logger.info(
            "synthesize_procedural extracted user_id=%s procedures_created=%d procedures_skipped=%d",
            user_id,
            created,
            skipped,
        )
        return {"status": "synthesized", "procedures_created": created, "procedures_skipped": skipped}

    def generate_thread_summary_durable(
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

        logger.info("generate_thread_summary_durable started user_id=%s thread_id=%s", user_id, thread_id)

        summary_id = f"summary_{user_id}_{thread_id}"
        existing_summary: Optional[dict[str, Any]] = None
        try:
            existing_summary = self._summaries_container.read_item(
                item=summary_id,
                partition_key=partition_key_for_user_thread(user_id, thread_id),
            )
        except CosmosResourceNotFoundError:
            pass

        query = f"SELECT * FROM c WHERE {user_scope_predicate()} AND c.thread_id = @thread_id AND c.type = 'turn'"
        parameters: list[dict[str, Any]] = [
            *user_scope_parameters(user_id),
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
                partition_key=partition_key_for_user_thread(user_id, thread_id),
            )
        )

        if existing_summary and not items:
            logger.info("generate_thread_summary_durable no new memories, returning existing")
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
        self._inherit_source_scope(summary_doc, items, user_id)
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
        doc["scope_key"] = scope_key_for_user(user_id)
        doc["thread_id"] = thread_id
        provenance = dict(doc.get("provenance") or {})
        provenance.setdefault("prompt_id", "summarize.prompty")
        provenance.setdefault("prompt_version", "v1")
        provenance.setdefault("created_by", scope_key_for_user(user_id))
        doc["provenance"] = provenance
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
        summary_doc = self.generate_thread_summary_durable(user_id, thread_id, recent_k=recent_k)
        return self.persist_thread_summary(user_id, thread_id, summary_doc)

    def generate_user_summary_durable(
        self,
        user_id: str,
        thread_ids: list[str] | None = None,
        recent_k: int | None = None,
    ) -> dict[str, Any]:
        """Generate a user summary document without embedding or writing it."""
        if not user_id:
            raise ValidationError("user_id is required")

        logger.info(
            "generate_user_summary_durable started user_id=%s observed_thread_ids=%s",
            user_id,
            len(thread_ids) if thread_ids else 0,
        )

        user_summary_id = f"user_summary_{user_id}"
        existing_summary: Optional[dict[str, Any]] = None
        try:
            existing_summary = self._summaries_container.read_item(
                item=user_summary_id,
                partition_key=partition_key_for_user_thread(user_id, USER_SUMMARY_THREAD_ID),
            )
        except CosmosResourceNotFoundError:
            pass

        query_predicate = user_scope_predicate()
        parameters: list[dict[str, Any]] = user_scope_parameters(user_id)
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
            logger.info("generate_user_summary_durable no new memories, returning existing")
            user_doc = dict(existing_summary)
            user_doc.pop("embedding", None)
            return user_doc
        if not existing_summary and not items:
            raise NoSourceMemoriesError(f"No memories found for user_id={user_id!r}")

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
            "thread_id": USER_SUMMARY_THREAD_ID,
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
        self._inherit_source_scope(summary_doc, items, user_id)
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
        doc["scope_key"] = scope_key_for_user(user_id)
        doc["thread_id"] = USER_SUMMARY_THREAD_ID
        provenance = dict(doc.get("provenance") or {})
        provenance.setdefault("prompt_id", "user_summary.prompty")
        provenance.setdefault("prompt_version", "v1")
        provenance.setdefault("created_by", scope_key_for_user(user_id))
        doc["provenance"] = provenance
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
        summary_doc = self.generate_user_summary_durable(user_id, thread_ids=thread_ids, recent_k=recent_k)
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
                "provenance": {
                    "prompt_id": "dedup.prompty",
                    "prompt_version": "v1",
                    "created_by": scope_key_for_user(user_id),
                },
            },
        )

    def reconcile_memories(
        self,
        user_id: str,
        n: int = 50,
        *,
        memory_type: str = "fact",
        tenant_id: str | None = None,
        scope_key: str | None = None,
    ) -> dict[str, int]:
        """Resolve contradictions among active memories in a user or explicit scope.

        Loads up to ``n`` active (non-superseded) ``memory_type`` records and
        asks the dedup prompt to identify ``contradicted_pairs`` - opposing
        claims about the same subject (e.g. "deadline March 1" vs "March 15").
        Each loser is soft-deleted with ``supersede_reason="contradict"`` and
        ``superseded_by`` set to the winner.

        Near-duplicate *paraphrases* are not merged here: reconcile is a bounded,
        convergent contradiction pass - no clustering, no synthesized merged
        documents, no re-merge churn. Episodic and procedural types are no-ops
        because they have no contradiction semantics.

        Returns ``{"kept", "merged", "contradicted"}``; ``merged`` is always 0.
        """
        if scope_key is None and not user_id:
            raise ValidationError("user_id is required")
        if scope_key is not None:
            tenant_id, _, _, scope_key = scope_values_for_scope_key(scope_key, tenant_id)
        if not isinstance(n, int) or isinstance(n, bool) or n < 1:
            raise ValidationError(f"n must be a positive integer, got {n!r}")
        if n > 500:
            raise ValidationError(f"n must be <= 500 to bound prompt size and LLM cost, got {n}")
        if memory_type not in {"fact", "episodic", "procedural"}:
            raise ValidationError(f"memory_type must be one of fact, episodic, procedural; got {memory_type!r}")
        if memory_type in {"episodic", "procedural"}:
            result = {"kept": 0, "merged": 0, "contradicted": 0}
            logger.info("reconcile_memories %s no-op user_id=%s result=%s", memory_type, user_id, result)
            return result

        started_at = time.monotonic()
        reconcile_scope = scope_key or f"user:{user_id}"
        logger.info(
            "reconcile_memories started user_id=%s tenant_id=%s scope_key=%s n=%d memory_type=%s",
            user_id,
            tenant_id or "default",
            reconcile_scope,
            n,
            memory_type,
        )

        facts = self._active_memories_for_reconcile(user_id, memory_type, n, tenant_id=tenant_id, scope_key=scope_key)
        result = self._reconcile_contradictions(user_id, memory_type, facts, scope_key=reconcile_scope)
        self._emit_reconcile_outcome(
            started_at=started_at,
            user_id=user_id,
            candidates=len(facts),
            result=result,
        )
        return result

    def _reconcile_contradictions(
        self, user_id: str, memory_type: str, facts: list[dict[str, Any]], *, scope_key: str | None = None
    ) -> dict[str, int]:
        """Resolve contradictions within an explicit pool of same-type memories.

        Runs the dedup prompt over the pool and applies its
        ``contradicted_pairs`` - the loser of each pair is soft-deleted with
        ``superseded_by`` set to the winner. Paraphrases are not merged here;
        the dedup prompt/schema no longer emits duplicate groups because
        write-time in-place dedup already folds them, so no merged documents are
        minted and the pass is convergent. Returns
        ``{"kept", "merged": 0, "contradicted"}``.
        """
        if len(facts) <= 1:
            return {"kept": len(facts), "merged": 0, "contradicted": 0}

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
                f"Confidence: {conf_str} | Salience: {sal_str} | Created: {created_str}"
            )
        facts_text = "\n".join(lines)

        response_text = self._run_prompty("dedup.prompty", inputs={"facts_text": facts_text})
        parsed = self._parse_llm_json(response_text)
        contradicted_pairs = parsed.get("contradicted_pairs", []) or []

        facts_by_id: dict[str, dict[str, Any]] = {f["id"]: f for f in facts}
        contradicted = 0
        consumed_loser_ids: set[str] = set()
        for pair in contradicted_pairs:
            first_id = pair.get("winner_id")
            second_id = pair.get("loser_id")
            if not first_id or not second_id or first_id == second_id:
                continue
            if first_id not in facts_by_id or second_id not in facts_by_id:
                logger.warning(
                    "reconcile_memories: hallucinated contradiction id(s) not in pool; skipping pair %r",
                    pair,
                )
                continue
            winner_id, loser_id = self._choose_contradiction_winner(first_id, second_id, facts_by_id)
            # Guard chained contradictions: never point a loser at a winner that
            # was itself already superseded earlier this pass (A>B then B>C would
            # tombstone C in favor of a now-dead B), and never re-supersede a loser.
            if winner_id in consumed_loser_ids or loser_id in consumed_loser_ids:
                logger.info(
                    "reconcile_memories: skipping chained/duplicate contradiction pair %r "
                    "(winner or loser already superseded this pass)",
                    pair,
                )
                continue
            loser_doc = facts_by_id.get(loser_id)
            if loser_doc is None:
                continue
            if self._mark_superseded(loser_doc, winner_id, reason="contradict"):
                contradicted += 1
                consumed_loser_ids.add(loser_id)

        kept = len([fid for fid in facts_by_id if fid not in consumed_loser_ids])
        result = {"kept": kept, "merged": 0, "contradicted": contradicted}
        logger.info(
            "reconcile_memories contradiction pass user_id=%s scope_key=%s memory_type=%s result=%s",
            user_id,
            scope_key,
            memory_type,
            result,
        )
        return result

    @staticmethod
    def _choose_contradiction_winner(
        first_id: str, second_id: str, facts_by_id: dict[str, dict[str, Any]]
    ) -> tuple[str, str]:
        first = facts_by_id[first_id]
        second = facts_by_id[second_id]
        first_created = str(first.get("created_at") or "")
        second_created = str(second.get("created_at") or "")
        # Contradiction winner selection (matches dedup.prompty):
        #   1. more recent created_at wins - newest information supersedes;
        #   2. higher confidence breaks a timestamp tie;
        #   3. the alphabetically-first id is the final deterministic tiebreak,
        #      so the outcome never flaps across runs.
        # Shared-scope owner/role priority remains deferred (design decision #6).
        if first_created != second_created:
            return (first_id, second_id) if first_created > second_created else (second_id, first_id)
        first_conf = first.get("confidence") if _is_real_number(first.get("confidence")) else float("-inf")
        second_conf = second.get("confidence") if _is_real_number(second.get("confidence")) else float("-inf")
        if first_conf != second_conf:
            return (first_id, second_id) if first_conf > second_conf else (second_id, first_id)
        return (first_id, second_id) if first_id <= second_id else (second_id, first_id)

    def _active_memories_for_reconcile(
        self,
        user_id: str,
        memory_type: str,
        n: int,
        *,
        tenant_id: str | None = None,
        scope_key: str | None = None,
    ) -> list[dict[str, Any]]:
        # ---- Load up to N most recent active memories ----
        # ORDER BY c.created_at DESC keeps the TOP cap deterministic across
        # physical partitions and matches the dedup prompt's tiebreaker
        # ("more recent created_at first"). Cosmos's _ts is the last-write
        # timestamp, which would diverge from created_at after any UPDATE.
        #
        # Agent-sourced facts (sys:agent-fact) are excluded: they record what the
        # agent did/recommended (historical events), not mutable user state, so
        # they must never be contradiction-superseded by a later user statement.
        if scope_key is not None:
            tenant_id, _, _, scope_key = scope_values_for_scope_key(scope_key, tenant_id)
            scope_predicate = "c.tenant_id = @tenant_id AND c.scope_key = @scope_key"
            scope_parameters = [
                {"name": "@tenant_id", "value": tenant_id},
                {"name": "@scope_key", "value": scope_key},
            ]
        else:
            scope_predicate = user_scope_predicate()
            scope_parameters = user_scope_parameters(user_id)
        query = (
            f"SELECT TOP {top_literal(n, name='reconcile_memories.n')} * FROM c "
            f"WHERE {scope_predicate} "
            "AND c.type = @memory_type "
            f"AND {_ACTIVE_DOC_FILTER} "
            "AND NOT ARRAY_CONTAINS(c.tags, 'sys:agent-fact') "
            "ORDER BY c.created_at DESC"
        )
        return list(
            self._memories_container.query_items(
                query=query,
                parameters=[
                    *scope_parameters,
                    {"name": "@memory_type", "value": memory_type},
                ],
                enable_cross_partition_query=True,
            )
        )

    def build_procedural_context(self, user_id: str, task: Optional[str] = None) -> str:
        """Build a deterministic system prompt projection from active procedures."""
        if not user_id:
            raise ValidationError("user_id is required")
        query = (
            "SELECT * FROM c WHERE c.scope_key=@scope_key AND c.type='procedural' "
            "AND c.status='active' "
            "AND (NOT IS_DEFINED(c.superseded_by) OR IS_NULL(c.superseded_by))"
        )
        procedures = list(
            self._memories_container.query_items(
                query=query,
                parameters=[{"name": "@scope_key", "value": scope_key_for_user(user_id)}],
                enable_cross_partition_query=True,
            )
        )
        procedures = [
            proc
            for proc in procedures
            if proc.get("type") == "procedural"
            and proc.get("status") == "active"
            and not proc.get("superseded_by")
        ]

        stopwords = {
            "a",
            "an",
            "and",
            "are",
            "as",
            "at",
            "be",
            "by",
            "for",
            "from",
            "in",
            "is",
            "it",
            "of",
            "on",
            "or",
            "the",
            "to",
            "with",
        }

        def _tokens(value: str) -> set[str]:
            return {token for token in re.findall(r"[a-z0-9]+", value.lower()) if token not in stopwords}

        task_tokens = _tokens(task) if isinstance(task, str) and task.strip() else set()
        policies: list[dict[str, Any]] = []
        task_procedures: list[dict[str, Any]] = []
        for proc in procedures:
            kind = proc.get("procedure_kind")
            if kind in {"behavioral_policy", "decision_rule"}:
                policies.append(proc)
                continue
            if task_tokens and kind in {"workflow", "recovery_strategy", "tool_usage"}:
                searchable = " ".join(
                    [
                        proc.get("retrieval_text") if isinstance(proc.get("retrieval_text"), str) else "",
                        *[
                            condition
                            for condition in proc.get("activation_conditions", [])
                            if isinstance(condition, str)
                        ],
                    ]
                )
                if task_tokens & _tokens(searchable):
                    task_procedures.append(proc)

        included = policies + task_procedures
        if not included:
            return ""

        authority_rank = {"mandatory": 3, "high": 2, "medium": 1, "low": 0}

        def _sort_key(proc: dict[str, Any]) -> tuple[int, int, str]:
            try:
                priority = int(proc.get("priority") or 0)
            except (TypeError, ValueError):
                priority = 0
            authority = proc.get("source_authority")
            rank = authority_rank.get(authority if isinstance(authority, str) else "low", 0)
            name = proc.get("name") if isinstance(proc.get("name"), str) else ""
            return (-priority, -rank, name.lower())

        policies.sort(key=_sort_key)
        task_procedures.sort(key=_sort_key)
        included = policies + task_procedures

        fingerprint_payload = sorted(
            (
                str(proc.get("id", "")),
                str(proc.get("version", "")),
                str(proc.get("status", "")),
                str(proc.get("priority", "")),
                str(proc.get("scope_key", "")),
            )
            for proc in included
        )
        fingerprint = hashlib.sha256(json.dumps(fingerprint_payload, separators=(",", ":")).encode()).hexdigest()
        logger.debug(
            "build_procedural_context fingerprint=%s included_ids=%s user_id=%s",
            fingerprint,
            [proc.get("id") for proc in included],
            user_id,
        )

        lines = ["# Learned procedures"]
        if policies:
            lines.append("")
            lines.append("## Behavioral policies")
            for proc in policies:
                name = proc.get("name") if isinstance(proc.get("name"), str) else "Unnamed procedure"
                summary = proc.get("summary") if isinstance(proc.get("summary"), str) else proc.get("content", "")
                lines.append(f"- {name}: {summary}")
        if task_procedures:
            lines.append("")
            lines.append("## Task procedures")
            for proc in task_procedures:
                name = proc.get("name") if isinstance(proc.get("name"), str) else "Unnamed procedure"
                summary = proc.get("summary") if isinstance(proc.get("summary"), str) else proc.get("content", "")
                lines.append(f"### {name}")
                if summary:
                    lines.append(f"Summary: {summary}")
                steps = proc.get("steps") if isinstance(proc.get("steps"), list) else []
                if steps:
                    sorted_steps = sorted(
                        [step for step in steps if isinstance(step, dict)],
                        key=lambda step: int(step.get("sequence") or 0),
                    )
                    for index, step in enumerate(sorted_steps, start=1):
                        instruction = step.get("instruction") if isinstance(step.get("instruction"), str) else ""
                        if instruction:
                            lines.append(f"{index}. {instruction}")
        return "\n".join(lines)
