"""Async pipeline service for LLM-driven memory extraction, summaries, and reconciliation.

This module is the asynchronous sibling of
:class:`azure.cosmos.agent_memory.services.pipeline.PipelineService`. The two
share all pure helpers via
:mod:`azure.cosmos.agent_memory.services._pipeline_helpers`; only the IO call
sites differ - every Cosmos query, chat completion, and embedding call is
``await``-ed against the async clients/store.
"""

from __future__ import annotations

import hashlib
import inspect
import json
import time
from collections import defaultdict
from datetime import datetime, timezone
from typing import Any, Iterable, Literal, Mapping, Optional

from azure.cosmos.exceptions import (
    CosmosAccessConditionFailedError,
    CosmosResourceExistsError,
    CosmosResourceNotFoundError,
)
from pydantic import ValidationError as PydanticValidationError

from azure.cosmos.agent_memory._container_routing import ContainerKey
from azure.cosmos.agent_memory._curation import mark_secret, sanitize_record, sanitize_text
from azure.cosmos.agent_memory._utils import (
    DEFAULT_TTL_BY_TYPE,
    compute_content_hash,
    cosine_similarity,
)
from azure.cosmos.agent_memory.aio.store import AsyncMemoryStore
from azure.cosmos.agent_memory.exceptions import (
    NoSourceMemoriesError,
    ValidationError,
)
from azure.cosmos.agent_memory.logging import describe_exception, describe_shape, get_logger
from azure.cosmos.agent_memory.models import (
    AGENT_FACT_TAG,
    EpisodicRecord,
    FactRecord,
    ProceduralRecord,
    ThreadSummaryRecord,
    UserSummaryRecord,
    construct_internal,
)
from azure.cosmos.agent_memory.prompts._schemas import response_format_for
from azure.cosmos.agent_memory.services._combine import (
    COMBINE_PROMPT,
    CombineCounts,
    CombineDraft,
    CombineOutcome,
    CombineWrite,
    arun_combine_group,
    combined_record,
    comparable_candidates,
    fold_verdicts,
    in_seed_order,
    plan_combine_groups,
    settled_seeds,
)
from azure.cosmos.agent_memory.services._dedup import (
    DEDUP_CURSOR_THREAD_ID,
    DEDUP_CURSOR_TYPE,
    DEDUP_PROMPT,
    Verdict,
    close_candidates,
    dedup_cursor_body,
    dedup_cursor_id,
    dedup_result,
    neighbor_query,
    next_dedup_cursor,
    parse_verdicts,
    plan_folds,
    survivor_patch,
    unchecked_memories_query,
    validate_consolidation_args,
    verify_inputs,
)
from azure.cosmos.agent_memory.services._extraction_context import (
    NONE,
    context_turns_text,
    earliest_created_at,
    known_facts_query,
    known_facts_text,
    latest_created_at,
    previous_turns_query,
    prior_turns,
    query_text,
    structured_fields,
    summary_text,
    takes_context,
    window_write_second,
)
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
    ground_fact_sources,
    is_retryable_llm_error,
    is_valid_time_pair,
    newest_window,
    parse_llm_json,
    segment_time_bounds,
    transcript_labels,
    turn_gap_seconds,
)
from azure.cosmos.agent_memory.services._pipeline_helpers import (
    is_real_number as _is_real_number,
)
from azure.cosmos.agent_memory.services._procedures import (
    PROCEDURE_CURSOR_ID,
    PROCEDURE_CURSOR_TYPE,
    ProcedureAnswer,
    ProcedureScope,
    advanced_procedure_cursor,
    behavioral_facts_query,
    existing_procedures_input,
    in_force,
    lesson_episodes_query,
    new_procedure,
    plan_answer,
    procedural_context,
    procedure_cursor_body,
    procedure_id,
    rank_related,
    reinforced_procedure,
    related_procedure_query,
    retirable_facts,
    revised_procedure,
    situational_rule_query,
    source_batches,
    sources_input,
    standing_rules,
    standing_rules_query,
    utc_today,
    with_support,
)
from azure.cosmos.agent_memory.store._search_helpers import top_literal
from azure.cosmos.agent_memory.thresholds import (
    get_combine_enabled,
    get_combine_workers,
    get_episode_idle_gap_seconds,
    get_episode_max_turns,
    get_episode_min_turns,
    get_episode_topic_drift,
    get_extraction_batch_max_tokens,
)

logger = get_logger("azure.cosmos.agent_memory.pipeline.aio")


_cap_structured_summary = cap_structured_summary

_ACTIVE_DOC_FILTER = "(NOT IS_DEFINED(c.superseded_by) OR IS_NULL(c.superseded_by))"

# Safety cap on how many episodes one extract_episodes call may close in a
# single pass (mirror of the sync pipeline). Boundary evaluation runs on a small
# turn cadence, so this only bounds a pathological drain and never fans out into
# an unbounded burst of LLM extractions.
_EPISODE_MAX_SEGMENTS_PER_RUN = 50


class _AsyncStoreContainerAdapter:
    """Expose one split ``AsyncMemoryStore`` container via Cosmos method shapes."""

    def __init__(self, store: AsyncMemoryStore, container_key: ContainerKey) -> None:
        self._store = store
        self._container_key = container_key

    def _target_container(self) -> Any | None:
        containers = getattr(self._store, "_containers", None)
        if isinstance(containers, dict):
            return containers.get(self._container_key)
        if self._container_key is ContainerKey.MEMORIES:
            return getattr(self._store, "container", None)
        return None

    async def _collect_query(self, result: Any) -> list[dict[str, Any]]:
        if inspect.isawaitable(result):
            result = await result
        if hasattr(result, "__aiter__"):
            return [item async for item in result]
        return list(result)

    async def query_items(self, **kwargs: Any) -> list[dict[str, Any]]:
        container = self._target_container()
        if container is not None and hasattr(container, "query_items"):
            kwargs.pop("enable_cross_partition_query", None)
            return await self._collect_query(container.query_items(**kwargs))
        try:
            return await self._store.query(
                kwargs["query"],
                parameters=kwargs.get("parameters"),
                container_key=self._container_key,
                partition_key=kwargs.get("partition_key"),
            )
        except TypeError:
            return await self._store.query(
                kwargs["query"],
                parameters=kwargs.get("parameters"),
                partition_key=kwargs.get("partition_key"),
            )

    async def read_item(self, *, item: str, partition_key: Any) -> dict[str, Any]:
        container = self._target_container()
        if container is not None and hasattr(container, "read_item"):
            response = container.read_item(item=item, partition_key=partition_key)
            if inspect.isawaitable(response):
                response = await response
            return response
        try:
            return await self._store.read_item(item, partition_key, container_key=self._container_key)
        except TypeError:
            return await self._store.read_item(item, partition_key)

    async def upsert_item(self, *, body: dict[str, Any]) -> dict[str, Any]:
        container = self._target_container()
        if container is not None and hasattr(container, "upsert_item"):
            response = container.upsert_item(body=body)
            if inspect.isawaitable(response):
                response = await response
            return response if isinstance(response, dict) else body
        upsert = getattr(self._store, "upsert_item", None)
        if upsert is not None:
            response = upsert(body=body)
            if inspect.isawaitable(response):
                response = await response
            return response if isinstance(response, dict) else body
        response = await self._store.upsert_memory(body)
        return response if isinstance(response, dict) else body

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

    async def patch_item(
        self, *, item: str, partition_key: Any, patch_operations: list[dict[str, Any]]
    ) -> dict[str, Any]:
        container = self._target_container()
        patch_item = getattr(container, "patch_item", None)
        if callable(patch_item):
            response = patch_item(item=item, partition_key=partition_key, patch_operations=patch_operations)
            if inspect.isawaitable(response):
                response = await response
            if isinstance(response, dict):
                return response
            return await self.read_item(item=item, partition_key=partition_key)
        doc = await self.read_item(item=item, partition_key=partition_key)
        return await self.upsert_item(body=self._apply_patch_operations(doc, patch_operations))

    async def create_item(self, *, body: dict[str, Any]) -> dict[str, Any]:
        container = self._target_container()
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
        response = await self._store.upsert_memory(body)
        return response if isinstance(response, dict) else body

    async def replace_item(self, **kwargs: Any) -> Any:
        container = self._target_container()
        if container is not None and hasattr(container, "replace_item"):
            response = container.replace_item(**kwargs)
            if inspect.isawaitable(response):
                response = await response
            return response
        return await self.upsert_item(body=kwargs["body"])


class AsyncPipelineService:
    """Async LLM orchestration service backed by an async typed memory store."""

    def __init__(
        self,
        store: AsyncMemoryStore,
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

    async def _query_items(self, container: Any, **kwargs: Any) -> list[dict[str, Any]]:
        result = container.query_items(**kwargs)
        if inspect.isawaitable(result):
            result = await result
        if hasattr(result, "__aiter__"):
            return [item async for item in result]
        return list(result)

    async def _read_item(self, container: Any, *, item: str, partition_key: Any) -> dict[str, Any]:
        result = container.read_item(item=item, partition_key=partition_key)
        if inspect.isawaitable(result):
            result = await result
        return result

    async def _upsert_item(self, container: Any, *, body: dict[str, Any]) -> dict[str, Any]:
        result = container.upsert_item(body=body)
        if inspect.isawaitable(result):
            result = await result
        return result if isinstance(result, dict) else body

    async def _create_item(self, container: Any, *, body: dict[str, Any]) -> dict[str, Any]:
        result = container.create_item(body=body)
        if inspect.isawaitable(result):
            result = await result
        return result if isinstance(result, dict) else body

    async def _replace_item(self, container: Any, **kwargs: Any) -> Any:
        result = container.replace_item(**kwargs)
        if inspect.isawaitable(result):
            result = await result
        return result

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
        return await self._embeddings.generate(sanitize_text(text)[0])

    async def _embed_batch(self, texts: list[str]) -> list[list[float]]:
        return await self._embeddings.generate_batch([sanitize_text(text)[0] for text in texts])

    @staticmethod
    def _redact_for_embedding(doc: dict[str, Any]) -> dict[str, Any]:
        """Redact ``doc`` in place before its text is embedded, so the vector matches the stored text."""
        redacted = sanitize_record(doc)
        if redacted:
            logger.warning("redacted literal secrets before embedding id=%s patterns=%s", doc.get("id"), redacted)
        return doc

    def _prompt_lineage(self, filename: str) -> dict[str, str]:
        """Return ``{prompt_id, prompt_version}`` for stamping a doc.

        Safe no-op fallback (``prompt_version="v1"``) when the loader was
        never initialized - happens in unit tests that build the service
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
        # getattr fallback covers unit tests that build AsyncPipelineService
        # via __new__ to bypass __init__ (and therefore the metadata-keys stash).
        return build_transcript(
            items,
            group_by_thread=group_by_thread,
            metadata_keys=getattr(self, "_transcript_metadata_keys", None),
            include_timestamp=include_timestamp,
        )

    def _sanitized_body(self, doc: dict[str, Any]) -> dict[str, Any]:
        """Copy ``doc`` with literal credentials redacted, ready for a direct container write."""
        body = dict(doc)
        redacted = sanitize_record(body)
        if redacted:
            logger.warning("redacted literal secrets before write id=%s patterns=%s", body.get("id"), redacted)
        return body

    async def _upsert_memory(self, doc: dict[str, Any]) -> dict[str, Any]:
        """Upsert a fact, episodic, or procedural document to the memories container."""
        return await self._upsert_item(self._memories_container, body=self._sanitized_body(doc))

    async def _upsert_summary(self, doc: dict[str, Any]) -> dict[str, Any]:
        """Upsert a thread/user summary document to the summaries container."""
        return await self._upsert_item(self._summaries_container, body=self._sanitized_body(doc))

    async def _create_memory(self, doc: dict[str, Any]) -> dict[str, Any]:
        """Create a memory document and let Cosmos raise 409 for duplicates."""
        return await self._create_item(self._memories_container, body=self._sanitized_body(doc))

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

    async def _mark_superseded(
        self,
        old_doc: dict[str, Any],
        superseder_id: str,
        *,
        reason: Literal["duplicate", "contradict", "update", "combined", "procedure"],
    ) -> bool:
        """Atomically set ``superseded_by`` on ``old_doc`` via the async memory store."""
        return await self._store.mark_superseded(old_doc, superseder_id, reason=reason)

    @staticmethod
    def _parse_llm_json(text: str | None) -> dict[str, Any]:
        return parse_llm_json(text)

    async def extract_memories_durable(
        self,
        user_id: str,
        thread_id: str,
        recent_k: int | None = None,
        *,
        turns: Optional[list[dict[str, Any]]] = None,
    ) -> dict[str, list[dict[str, Any]]]:
        """Load turns, call the LLM, and return memory docs without embeddings or writes.

        A prompt that takes context (``extract_memories-v3``) also sees, read-only, the thread's
        turns before each batch, its summary when written before the window, and the user's
        stored facts nearest the batch (see ``_extraction_context``); facts still come only from
        the new turns, each recording the turns it cites.
        """
        if not user_id:
            raise ValidationError("user_id is required")
        if not thread_id:
            raise ValidationError("thread_id is required")

        logger.info("extract_memories_durable started user_id=%s thread_id=%s", user_id, thread_id)

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
            items = await self._query_items(
                self._turns_container,
                query=query,
                parameters=parameters,
                partition_key=[user_id, thread_id],
            )
        else:
            items = list(turns)

        items = newest_window(items, recent_k)

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
        # call, so a single poisoned turn fails only its own batch. Turns from
        # succeeded and quarantined (non-retryable, e.g. content-filter) batches
        # go into ``processed_turns``; the in-process caller marks them
        # ``extracted_at`` so they are never re-processed (the Durable backend
        # instead advances a count-based watermark). Turns from batches that fail
        # with a *retryable* error are left out and retried on the next run.
        batches = batch_turns_by_tokens(items, get_extraction_batch_max_tokens())
        facts: list[tuple[dict[str, Any], list[Optional[str]], list[str]]] = []
        processed_turns: list[dict[str, Any]] = []
        deferred_turn_count = 0
        quarantined_turn_count = 0
        extract_prompt = extract_memories_prompt_file()
        # Read-only context for prompts that take it: the turns before the window and the thread summary.
        with_context = takes_context(extract_prompt)
        prior, summary = (
            await self._extraction_window_context(user_id, thread_id, items) if with_context else ([], NONE)
        )
        batch_start = 0
        for batch in batches:
            batch_turn_ids = [str(item["id"]) if item.get("id") else None for item in batch]
            batch_labels = transcript_labels(batch)
            if with_context:
                inputs = {
                    "transcript": self._build_grounded_transcript(batch, labels=batch_labels),
                    "previous_turns": context_turns_text([*prior, *items[:batch_start]]),
                    "thread_summary": summary,
                    "known_memories": await self._known_memories(user_id, batch),
                }
            else:
                inputs = {"transcript": self._build_transcript(batch, include_timestamp=True)}
            batch_start += len(batch)
            try:
                response_text = await self._run_extraction(extract_prompt, inputs)
                parsed = self._parse_llm_json(response_text)
                facts.extend(
                    (fact, batch_turn_ids, batch_labels) for fact in parsed.get("facts", []) if isinstance(fact, dict)
                )
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
                        describe_exception(exc),
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
                        describe_exception(exc),
                    )

        doc_timestamp = self._stable_source_timestamp(items)
        fact_docs: list[dict[str, Any]] = []
        updates: list[dict[str, Any]] = []
        exact_dedup_skipped = 0

        for fact, batch_turn_ids, batch_labels in facts:
            text = fact.get("text")
            if not text:
                logger.warning(
                    "extract_memories: dropping malformed fact (missing 'text') text=%s fact=%s",
                    describe_shape(text),
                    describe_shape(fact),
                )
                continue

            text, secret_patterns = sanitize_text(text)
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
                    "extract_memories: coercing an invalid fact source (%s) to 'user' user_id=%s",
                    describe_shape(raw_source),
                    user_id,
                )
            fact_source = raw_source if raw_source in ("user", "agent") else "user"
            source_tags = [AGENT_FACT_TAG] if fact_source == "agent" else []
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
                "source_turn_ids": ground_fact_sources(fact, turn_ids=batch_turn_ids, labels=batch_labels),
                **structured_fields(fact),
                "created_at": doc_timestamp,
                "updated_at": doc_timestamp,
            }

            mark_secret(doc, secret_patterns)
            # The model also writes metadata (temporal_context); the doc leaves this activity
            # through Durable history, so it is redacted here rather than only at persist.
            sanitize_record(doc)
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
            "extract_memories_durable completed user_id=%s thread_id=%s fact_docs=%d episodic_docs=%d updates=%d",
            user_id,
            thread_id,
            len(fact_docs),
            0,
            len(updates),
        )
        return result

    async def _run_extraction(self, prompt: str, inputs: dict[str, Any]) -> str:
        """Run the extraction prompt. When a request carrying read-only context fails for a reason a
        retry would not fix (a content filter, an over-long prompt), run it once more on the new
        turns alone: context text the filter rejects must not cost the batch its facts, nor
        follow the conversation into every later batch it is shown with."""
        try:
            return await self._run_prompty(prompt, inputs=inputs)
        except Exception as exc:  # noqa: BLE001
            if set(inputs) == {"transcript"} or is_retryable_llm_error(exc):
                raise
            logger.warning(
                "extract_memories: retrying a batch without its context after a non-retryable error: %s",
                describe_exception(exc),
            )
            return await self._run_prompty(prompt, inputs={"transcript": inputs["transcript"]})

    def _build_grounded_transcript(self, items: list[dict[str, Any]], labels: list[str]) -> str:
        """Build a timestamped transcript that labels each turn with the id the model cites it by."""
        transcript_items: list[dict[str, Any]] = []
        for item, label in zip(items, labels):
            copied = dict(item)
            copied["content"] = f"Turn {label}: {item.get('content', '')}"
            transcript_items.append(copied)
        return self._build_transcript(transcript_items, include_timestamp=True)

    async def _extraction_window_context(
        self, user_id: str, thread_id: str, items: list[dict[str, Any]]
    ) -> tuple[list[dict[str, Any]], str]:
        """The thread's turns before an extraction window, oldest first, and its summary as context.

        Both are read-only context, so a failure to read either costs only that context.
        """
        window_start = earliest_created_at(items)
        if window_start is None:
            return [], NONE
        window_ts = window_write_second(items, window_start)
        prior: list[dict[str, Any]] = []
        try:
            query, parameters = previous_turns_query(user_id, thread_id, window_start, window_ts)
            rows = await self._query_items(
                self._turns_container, query=query, parameters=parameters, partition_key=[user_id, thread_id]
            )
            prior = prior_turns(rows, items)
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "extract_memories: extracting without earlier turns user_id=%s thread_id=%s err=%s",
                user_id,
                thread_id,
                describe_exception(exc),
            )
        summary = NONE
        try:
            summary_doc = await self._read_item(
                self._summaries_container,
                item=f"summary_{user_id}_{thread_id}",
                partition_key=[user_id, thread_id],
            )
            summary = summary_text(summary_doc, window_start, window_ts)
        except CosmosResourceNotFoundError:
            pass
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "extract_memories: extracting without the thread summary user_id=%s thread_id=%s err=%s",
                user_id,
                thread_id,
                describe_exception(exc),
            )
        return prior, summary

    async def _known_memories(self, user_id: str, batch: list[dict[str, Any]]) -> str:
        """The user's stored facts nearest a batch of turns, as context for extracting it."""
        text = query_text(batch)
        if not text:
            return NONE
        try:
            query, parameters = known_facts_query(
                user_id, await self._embed_one(text), latest_created_at(batch), _ACTIVE_DOC_FILTER
            )
            rows = await self._query_items(
                self._memories_container, query=query, parameters=parameters, enable_cross_partition_query=True
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "extract_memories: extracting without stored memories user_id=%s err=%s",
                user_id,
                describe_exception(exc),
            )
            return NONE
        return known_facts_text(rows)

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

        # Redact before embedding: a vector built from the raw text would still encode
        # the secret even after the stored content is cleaned.
        for doc in docs_to_create:
            matched = sanitize_record(doc)
            if matched:
                logger.warning(
                    "persist_extracted_memories redacted literal secrets id=%s patterns=%s",
                    doc.get("id"),
                    matched,
                )

        docs_needing_embeddings = [doc for doc in docs_to_create if doc.get("content") and not doc.get("embedding")]
        if docs_needing_embeddings:
            embeddings = await self._embed_batch([str(doc["content"]) for doc in docs_needing_embeddings])
            for doc, embedding in zip(docs_needing_embeddings, embeddings):
                doc["embedding"] = embedding

        for doc in docs_to_create:
            validated = self._validate_extracted_doc(doc)
            doc_type = validated.get("type")
            try:
                await self._create_memory(validated)
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

    async def _mark_turns_extracted(self, turn_docs: list[dict[str, Any]]) -> int:
        """Stamp the fact-extraction ``extracted_at`` watermark on each turn doc
        (mirror of the sync helper). Episodic segmentation uses a separate
        per-thread cursor doc and never stamps turns. Per-turn failures are
        logged but never raise.
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
                await self._turns_container.patch_item(
                    item=turn_id,
                    partition_key=[turn.get("user_id"), turn.get("thread_id")],
                    patch_operations=[{"op": "set", "path": "/extracted_at", "value": now_iso}],
                )
                marked += 1
            except Exception as exc:
                logger.warning(
                    "_mark_turns_extracted failed for turn_id=%s err=%s (turn may be re-processed on next call)",
                    turn_id,
                    describe_exception(exc),
                )
        return marked

    async def extract_memories(
        self,
        user_id: str,
        thread_id: str,
        recent_k: int | None = None,
        *,
        turns: Optional[list[dict[str, Any]]] = None,
    ) -> dict[str, int]:
        """Extract facts and episodic memories from a thread and persist them."""
        extracted = await self.extract_memories_durable(user_id, thread_id, recent_k, turns=turns)
        # Capture the processed turns from the compute stage as the single source of
        # truth for stamping. Stamping happens here (not inside persist) so no
        # Persist uses this exact list for stamping after all creates finish.
        processed_turns = extracted.get("processed_turn_docs") or []
        counts = await self.persist_extracted_memories(user_id, extracted)
        if processed_turns:
            marked = await self._mark_turns_extracted(processed_turns)
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

    async def _load_turn_window(
        self,
        user_id: str,
        thread_id: str,
        recent_k: int | None = None,
    ) -> list[dict[str, Any]]:
        """Load the newest turns for a thread, returned in chronological order."""
        query = "SELECT * FROM c WHERE c.user_id = @user_id AND c.thread_id = @thread_id AND c.type = 'turn'"
        items = await self._query_items(
            self._turns_container,
            query=query,
            parameters=[
                {"name": "@user_id", "value": user_id},
                {"name": "@thread_id", "value": thread_id},
            ],
            partition_key=[user_id, thread_id],
        )
        return newest_window(items, recent_k)

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
            for turn_id in event.get("source_turn_ids") or []:
                source = str(turn_id).strip()
                mapped = source if source in valid_turn_ids else label_to_id.get(source.lower())
                if mapped and mapped not in grounded_sources:
                    grounded_sources.append(mapped)
                if mapped and mapped not in source_turn_ids:
                    source_turn_ids.append(mapped)
            grounded["source_turn_ids"] = grounded_sources
            grounded_events.append(grounded)
        return grounded_events, source_turn_ids

    async def _build_episode_docs(
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
        same segment collides on id and is skipped rather than duplicated. Mirror
        of the sync helper.
        """
        if not items:
            return []

        transcript_items: list[dict[str, Any]] = []
        for index, item in enumerate(items, start=1):
            turn_id = str(item.get("id") or f"turn-{index}")
            copied = dict(item)
            copied["content"] = f"Turn {turn_id}: {item.get('content', '')}"
            transcript_items.append(copied)
        transcript = self._build_transcript(transcript_items, include_timestamp=True)
        response_text = await self._run_prompty("extract_episode.prompty", inputs={"transcript": transcript})
        parsed = self._parse_llm_json(response_text)
        episodes = parsed.get("episodes", [])
        if not isinstance(episodes, list):
            logger.warning(
                "_build_episode_docs dropping malformed response user_id=%s thread_id=%s episodes=%s",
                user_id,
                thread_id,
                describe_shape(episodes),
            )
            return []

        doc_timestamp = self._stable_source_timestamp(items)
        turn_ids = [str(item.get("id")) for item in items if item.get("id")]
        segment_started, segment_ended = segment_time_bounds(items)

        episode_docs: list[dict[str, Any]] = []
        for index, episode in enumerate(episodes):
            if not isinstance(episode, dict):
                logger.warning(
                    "_build_episode_docs dropping malformed episode user_id=%s thread_id=%s index=%d episode=%s",
                    user_id,
                    thread_id,
                    index,
                    describe_shape(episode),
                )
                continue

            summary = episode.get("summary")
            if not isinstance(summary, str) or not summary.strip():
                logger.warning(
                    "_build_episode_docs dropping malformed episode (missing summary) "
                    "user_id=%s thread_id=%s index=%d summary=%s",
                    user_id,
                    thread_id,
                    index,
                    describe_shape(summary),
                )
                continue

            title = episode.get("title")
            if not isinstance(title, str) or not title.strip():
                logger.warning(
                    "_build_episode_docs dropping malformed episode (missing title) "
                    "user_id=%s thread_id=%s index=%d title=%s",
                    user_id,
                    thread_id,
                    index,
                    describe_shape(title),
                )
                continue

            if not isinstance(episode.get("events"), list):
                logger.warning(
                    "_build_episode_docs dropping malformed episode (missing events) "
                    "user_id=%s thread_id=%s index=%d events=%s",
                    user_id,
                    thread_id,
                    index,
                    describe_shape(episode.get("events")),
                )
                continue

            content = summary
            events, source_turn_ids = self._ground_episode_events(episode.get("events"), turn_ids=turn_ids)
            content_hash = compute_content_hash(content)
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
                        "content": content,
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
                    "_build_episode_docs dropping malformed episode user_id=%s thread_id=%s index=%d err=%s",
                    user_id,
                    thread_id,
                    index,
                    describe_exception(exc),
                )
                continue
            episode_docs.append(doc)

        return episode_docs

    @staticmethod
    def _deterministic_episode_id(segment_key: str, index: int) -> str:
        return deterministic_episode_id(segment_key, index)

    @staticmethod
    def _episode_cursor_id(user_id: str, thread_id: str) -> str:
        return f"episode_cursor_{user_id}_{thread_id}"

    async def _read_episode_cursor(self, user_id: str, thread_id: str) -> tuple[str, str]:
        """``(created_at, id)`` watermark of the last turn folded into an episode,
        or ``("", "")`` when none. Stored as a single doc in the MEMORIES
        container so advancing it never writes to the turns container and cannot
        re-enter the change feed (mirror of the sync helper). ``created_at`` is
        UTC-normalized, so the lexical ``>`` comparison matches chronology."""
        try:
            doc = await self._read_item(
                self._memories_container,
                item=self._episode_cursor_id(user_id, thread_id),
                partition_key=[user_id, thread_id],
            )
        except CosmosResourceNotFoundError:
            return "", ""
        return str(doc.get("last_episode_at") or ""), str(doc.get("last_episode_id") or "")

    async def _advance_episode_cursor(self, user_id: str, thread_id: str, last_turn: dict[str, Any]) -> None:
        """Advance the episodic watermark to ``last_turn``, never backwards
        (mirror of the sync helper): a single-doc write that cannot partially
        fail and never touches the change-feed-monitored turns container. The
        advance is atomic - ETag ``IfNotModified`` with a re-read retry - so a
        late, out-of-order concurrent orchestration cannot regress the cursor and
        duplicate episodes under topic drift."""
        from azure.core import MatchConditions
        from azure.cosmos.exceptions import CosmosAccessConditionFailedError

        last_at = str(last_turn.get("created_at") or "")
        last_id = str(last_turn.get("id") or "")
        if not last_at:
            return
        cursor_id = self._episode_cursor_id(user_id, thread_id)
        partition_key = [user_id, thread_id]
        for _ in range(3):
            etag: Optional[str] = None
            try:
                existing = await self._read_item(self._memories_container, item=cursor_id, partition_key=partition_key)
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
            body = {
                "id": cursor_id,
                "type": "episode_cursor",
                "user_id": user_id,
                "thread_id": thread_id,
                "last_episode_at": last_at,
                "last_episode_id": last_id,
                "updated_at": datetime.now(tz=timezone.utc).isoformat(),
            }
            try:
                if existing is None:
                    await self._create_item(self._memories_container, body=body)
                else:
                    await self._replace_item(
                        self._memories_container,
                        item=cursor_id,
                        body=body,
                        etag=etag,
                        match_condition=MatchConditions.IfNotModified,
                    )
                return
            except (CosmosResourceExistsError, CosmosAccessConditionFailedError):
                continue  # a concurrent run advanced first; re-read and re-check
        logger.debug("episode cursor advance retries exhausted user_id=%s thread_id=%s", user_id, thread_id)

    async def _load_open_episode_segment(self, user_id: str, thread_id: str) -> list[dict[str, Any]]:
        """Return the open episode segment: turns created after the episodic
        watermark (not yet folded), oldest first. Uses a per-thread
        ``(created_at, id)`` cursor doc instead of a per-turn stamp, so episodic
        segmentation writes nothing to the turns container (mirror of the sync
        helper)."""
        last_at, last_id = await self._read_episode_cursor(user_id, thread_id)
        items = await self._query_items(
            self._turns_container,
            query=(
                "SELECT * FROM c WHERE c.user_id = @user_id "
                "AND c.thread_id = @thread_id AND c.type = 'turn' "
                "AND (c.created_at > @last_at "
                "OR (c.created_at = @last_at AND c.id > @last_id))"
            ),
            parameters=[
                {"name": "@user_id", "value": user_id},
                {"name": "@thread_id", "value": thread_id},
                {"name": "@last_at", "value": last_at},
                {"name": "@last_id", "value": last_id},
            ],
            partition_key=[user_id, thread_id],
        )
        items.sort(key=created_at_sort_key)
        return items

    async def _episode_segment_embeddings(self, segment: list[dict[str, Any]]) -> list[list[float]]:
        """One embedding per segment turn for drift detection, or ``[]`` when drift
        is disabled or embeddings are unavailable (mirror of the sync helper)."""
        if get_episode_topic_drift() <= 0:
            return []
        embeddings: list[Optional[list[float]]] = [
            turn.get("embedding") if isinstance(turn.get("embedding"), list) else None for turn in segment
        ]
        missing = [i for i, emb in enumerate(embeddings) if emb is None]
        if missing:
            try:
                fresh = await self._embed_batch([str(segment[i].get("content") or "") for i in missing])
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "episode drift embedding failed (%s); skipping drift this evaluation", describe_exception(exc)
                )
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
            max_turns=get_episode_max_turns(),
            idle_gap=get_episode_idle_gap_seconds(),
            drift=get_episode_topic_drift(),
            min_turns=get_episode_min_turns(),
        )

    async def extract_episodes(
        self,
        user_id: str,
        thread_id: str,
        *,
        flush: bool = False,
    ) -> dict[str, int]:
        """Segment the open turn stream into episodes at detected boundaries.

        Mirror of the sync pipeline: the open segment is every turn created after
        the episodic watermark (see ``_read_episode_cursor``); at each boundary
        (idle time-gap, topic drift, or max-size cap) the closed segment is
        extracted, embedded, and persisted, then the watermark advances past those
        turns. ``flush=True`` drains the trailing open segment. The caller never
        signals "session end".

        Idempotency: each episode's id is deterministic in its segment key and
        ordinal - not the LLM summary text - so re-running the same still-open
        segment skips the duplicate write (409). The watermark is a single doc
        advanced only after a segment's episodes are created, so a crash between
        the create and the advance simply re-loads the same open segment next run
        (same turn set -> same ids -> 409); there is no partial-stamp state that
        could shift the boundary and admit a duplicate.
        """
        if not user_id:
            raise ValidationError("user_id is required")
        if not thread_id:
            raise ValidationError("thread_id is required")

        segment = await self._load_open_episode_segment(user_id, thread_id)
        total = 0
        guard = 0
        while segment and guard < _EPISODE_MAX_SEGMENTS_PER_RUN:
            guard += 1
            embeddings = await self._episode_segment_embeddings(segment)
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
                docs = await self._build_episode_docs(user_id, thread_id, closing, segment_key=segment_key)
                for doc in docs:
                    self._redact_for_embedding(doc)
                embeddings_for_docs = await self._embed_batch([str(doc["content"]) for doc in docs]) if docs else []
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
                        describe_exception(exc),
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
                    describe_exception(exc),
                )
                await self._advance_episode_cursor(user_id, thread_id, closing[-1])
                segment = segment[boundary:]
                continue
            for doc, embedding in zip(docs, embeddings_for_docs):
                doc["embedding"] = embedding
                try:
                    await self._create_memory(doc)
                    total += 1
                except CosmosResourceExistsError:
                    logger.info("extract_episodes idempotent skip duplicate episode id=%s", doc.get("id"))
            await self._advance_episode_cursor(user_id, thread_id, closing[-1])
            segment = segment[boundary:]
        return {"episodes": total}

    async def synthesize_procedural(
        self,
        user_id: str,
        *,
        force: bool = False,
        deadline: Optional[float] = None,
    ) -> dict[str, Any]:
        """Async mirror of the sync ``synthesize_procedural``: learn procedures from the behavioral
        facts and episodes written since the last run.

        Reads the user's active preferences, requirements, high-salience facts, directives and
        corrections, and episodes with lessons, written after the user's procedure cursor, and
        shows them to the model in the order they were written, in batches of ``PROMPT_BATCH``,
        each with the user's existing procedures most similar to it. Each answer adds a candidate
        procedure, revises an existing one in place (same ID, next version) or records new
        evidence for one; a procedure's status then follows from its evidence
        (``_procedures.procedure_status``). A directive or correction an active situational
        procedure now holds is retired into it (``facts_retired``). With no new sources it
        returns ``status="unchanged"`` without a model call.

        The cursor moves past each batch once it is done, so a run that stops early resumes where
        it stopped. A retryable model failure stops the run (``deferred``); any other failure
        skips that batch. ``deadline``, a ``time.monotonic()`` value, stops the run from starting
        another model call once it has passed, for callers with a time limit such as a Durable
        activity. ``force`` has no effect: a run only ever reads what is new.
        """
        del force
        if not user_id:
            raise ValidationError("user_id is required")

        logger.info("synthesize_procedural extraction started user_id=%s", user_id)

        scope = ProcedureScope(user_id)
        cursor = await self._read_procedure_cursor(scope)
        facts_read = await self._query_user_memories(*behavioral_facts_query(scope, cursor))
        episodes_read = await self._query_user_memories(*lesson_episodes_query(scope, cursor))
        batches = source_batches(facts_read, episodes_read)
        if not any(sources for sources, _, _ in batches):
            await self._advance_procedure_cursor(scope, cursor, facts_read, episodes_read)
            logger.info("synthesize_procedural unchanged user_id=%s - no new procedure sources", user_id)
            return {"status": "unchanged", "procedures_created": 0}

        counts: dict[str, int] = defaultdict(int)
        answered = 0
        for sources, facts, episodes in batches:
            if sources:
                if deadline is not None and time.monotonic() >= deadline:
                    logger.info("synthesize_procedural stopped at its deadline user_id=%s", user_id)
                    if not answered:
                        return {"status": "deferred", "procedures_created": 0}
                    break
                state, outcomes = await self._procedure_batch(user_id, scope, sources)
                if state == "deferred":
                    return {"status": "deferred", "procedures_created": counts["created"]}
                if state == "answered":
                    answered += 1
                    for outcome, number in outcomes.items():
                        counts[outcome] += number
            cursor = await self._advance_procedure_cursor(scope, cursor, facts, episodes)

        if not answered:
            return {"status": "skipped", "procedures_created": 0}
        logger.info(
            "synthesize_procedural extracted user_id=%s procedures_created=%d procedures_updated=%d "
            "procedures_reinforced=%d procedures_skipped=%d facts_retired=%d",
            user_id,
            counts["created"],
            counts["updated"],
            counts["reinforced"],
            counts["skipped"],
            counts["facts_retired"],
        )
        return {
            "status": "synthesized",
            "procedures_created": counts["created"],
            "procedures_updated": counts["updated"],
            "procedures_reinforced": counts["reinforced"],
            "procedures_skipped": counts["skipped"],
            "facts_retired": counts["facts_retired"],
        }

    async def _procedure_batch(
        self, user_id: str, scope: ProcedureScope, sources: list[dict[str, Any]]
    ) -> tuple[str, dict[str, int]]:
        """One model call over a batch of sources, with its answers applied.

        Returns ``("answered", outcomes)``; ``("skipped", {})`` when the call failed for good or
        its answer was unusable, which would fail the same way on a retry; or ``("deferred", {})``
        after a retryable model error.
        """
        source_inputs, labels = sources_input(sources)
        existing_text, existing_by_label = existing_procedures_input(await self._related_procedures(scope, sources))
        try:
            response_text = await self._run_prompty(
                "extract_procedure.prompty", inputs={**source_inputs, "existing_procedures": existing_text}
            )
        except Exception as exc:  # LLM quarantine: one bad call must not stop the pipeline.
            if is_retryable_llm_error(exc):
                logger.warning("synthesize_procedural deferred user_id=%s: %s", user_id, describe_exception(exc))
                return "deferred", {}
            logger.error(
                "synthesize_procedural skipped a batch user_id=%s after non-retryable LLM error: %s",
                user_id,
                describe_exception(exc, stack=True),
            )
            return "skipped", {}
        try:
            parsed = self._parse_llm_json(response_text)
        except Exception as exc:
            logger.error(
                "synthesize_procedural skipped a batch user_id=%s after an unusable answer: %s",
                user_id,
                describe_exception(exc, stack=True),
            )
            return "skipped", {}
        outcomes: dict[str, int] = defaultdict(int)
        answers = parsed.get("procedures", []) if isinstance(parsed, dict) else []
        for item in answers if isinstance(answers, list) else []:
            outcome, written = await self._apply_procedure_answer(user_id, scope, item, labels, existing_by_label)
            outcomes[outcome] += 1
            if written is not None:
                outcomes["facts_retired"] += await self._retire_taught_facts(user_id, written, labels.values())
        return "answered", outcomes

    async def _apply_procedure_answer(
        self,
        user_id: str,
        scope: ProcedureScope,
        item: Any,
        labels: Mapping[str, dict[str, Any]],
        existing_by_label: Mapping[str, dict[str, Any]],
    ) -> tuple[str, Optional[dict[str, Any]]]:
        """Apply one answer; returns ``created``, ``updated``, ``reinforced`` or ``skipped``, with
        the procedure as written (``None`` when nothing was written).

        An ``add`` whose procedure already exists (same kind and name) reinforces it, or revises
        it when the answer is the user's correction. An answer that cites no new source, or a
        revision that drops a number or negation the procedure stated, is skipped.
        """
        try:
            answer = plan_answer(item, labels, existing_by_label)
            if answer is None:
                return "skipped", None
            now = datetime.now(timezone.utc).isoformat()
            if answer.action == "add":
                target = await self._read_procedure(scope, procedure_id(user_id, answer.kind, answer.name))
                if target is None:
                    return await self._create_procedure(user_id, answer, now)
                action = "update" if answer.evidence == "corrected" else "reinforce"
            else:
                # Read it again: an earlier answer in this batch may have changed it since it was shown.
                target = await self._read_procedure(scope, str((answer.target or {}).get("id")))
                if target is None:
                    return "skipped", None
                action = answer.action
            outcome, changed = self._changed_procedure(target, answer, action, now)
            if changed is None:
                return "skipped", None
            written = await self._embedded_procedure(changed)
            replaced = await self._replace_procedure(written, target.get("_etag"))
            return (outcome, written) if replaced else ("skipped", None)
        except (ValidationError, PydanticValidationError, ValueError) as exc:
            logger.warning(
                "synthesize_procedural dropping malformed procedure user_id=%s: %s", user_id, describe_exception(exc)
            )
            return "skipped", None
        except Exception as exc:
            logger.error(
                "synthesize_procedural failed to persist one procedure user_id=%s: %s",
                user_id,
                describe_exception(exc, stack=True),
            )
            return "skipped", None

    async def _create_procedure(
        self, user_id: str, answer: ProcedureAnswer, now: str
    ) -> tuple[str, Optional[dict[str, Any]]]:
        created = with_support(new_procedure(answer, user_id=user_id, now=now), answer.sources, answer.evidence)
        if created is None:
            return "skipped", None
        created.update(self._prompt_lineage("extract_procedure.prompty"))
        written = await self._embedded_procedure(created)
        try:
            await self._create_memory(written)
        except CosmosResourceExistsError:
            return "skipped", None
        return "created", written

    async def _retire_taught_facts(
        self, user_id: str, procedure: Mapping[str, Any], sources: Iterable[dict[str, Any]]
    ) -> int:
        """Async mirror of :meth:`PipelineService._retire_taught_facts`."""
        sources = list(sources)
        retired = 0
        for fact in retirable_facts(procedure, sources):
            try:
                done = await self._mark_superseded(fact, str(procedure["id"]), reason="procedure")
            except Exception as exc:
                logger.info(
                    "synthesize_procedural kept a fact beside its procedure user_id=%s: %s",
                    user_id,
                    describe_exception(exc),
                )
                continue
            if not done:
                continue
            retired += 1
            # The fact may support another answer in this batch; mark it so it is not retired twice.
            for source in sources:
                if source.get("id") == fact["id"]:
                    source["superseded_by"] = procedure["id"]
        return retired

    def _changed_procedure(
        self, target: dict[str, Any], answer: ProcedureAnswer, action: str, now: str
    ) -> tuple[str, Optional[dict[str, Any]]]:
        """``target`` revised (``update``) or re-cited (``reinforce``) by ``answer``, with the answer's
        support added; ``None`` when there is nothing to write."""
        if action == "update":
            revised = revised_procedure(target, answer, now=now)
            if revised is None:
                logger.info(
                    "synthesize_procedural kept %s: its revision dropped a number or negation", target.get("id")
                )
                return "skipped", None
            revised.update(self._prompt_lineage("extract_procedure.prompty"))
            return "updated", with_support(revised, answer.sources, answer.evidence)
        reinforced = reinforced_procedure(target, answer, now=now)
        return "reinforced", with_support(reinforced, answer.sources, answer.evidence)

    async def _embedded_procedure(self, doc: dict[str, Any]) -> dict[str, Any]:
        """``doc`` validated as a procedure and embedded unless it already has a vector."""
        validated = construct_internal(ProceduralRecord, doc).to_doc()
        self._redact_for_embedding(validated)
        if not validated.get("embedding"):
            validated["embedding"] = await self._embed_one(validated["retrieval_text"])
        return validated

    async def _read_procedure(self, scope: ProcedureScope, doc_id: str) -> Optional[dict[str, Any]]:
        try:
            doc = await self._read_item(self._memories_container, item=doc_id, partition_key=scope.partition_key)
        except CosmosResourceNotFoundError:
            return None
        return doc if doc.get("type") == "procedural" and not doc.get("superseded_by") else None

    async def _query_user_memories(self, query: str, parameters: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Run ``query`` over all of the user's memories, which span their threads' partitions."""
        return await self._query_items(self._memories_container, query=query, parameters=parameters)

    async def _query_procedures(
        self, scope: ProcedureScope, query: str, parameters: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """Run ``query`` over the user's procedures, which all share one partition."""
        return await self._query_items(
            self._memories_container, query=query, parameters=parameters, partition_key=scope.partition_key
        )

    async def _related_procedures(self, scope: ProcedureScope, sources: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """The user's existing procedures most similar to ``sources``, so the model revises one
        instead of minting a paraphrase of it."""
        scored: list[tuple[dict[str, Any], float]] = []
        for source in sources:
            vector = source.get("embedding")
            if not isinstance(vector, list) or not vector:
                continue
            for doc in await self._query_procedures(scope, *related_procedure_query(scope, vector)):
                scored.append((doc, cosine_similarity(vector, doc.get("embedding"))))
        return rank_related(scored)

    async def _replace_procedure(self, doc: dict[str, Any], etag: Optional[str]) -> bool:
        """Write a revision over the procedure it revises, only if that one is unchanged since read."""
        from azure.core import MatchConditions

        kwargs: dict[str, Any] = {"item": doc["id"], "body": doc}
        if etag:
            kwargs.update(etag=etag, match_condition=MatchConditions.IfNotModified)
        try:
            await self._replace_item(self._memories_container, **kwargs)
        except (CosmosAccessConditionFailedError, CosmosResourceNotFoundError):
            logger.info("synthesize_procedural: %s changed since it was read; its revision is skipped", doc["id"])
            return False
        return True

    async def _read_procedure_cursor(self, scope: ProcedureScope) -> Optional[dict[str, Any]]:
        try:
            doc = await self._read_item(
                self._memories_container, item=PROCEDURE_CURSOR_ID, partition_key=scope.partition_key
            )
        except CosmosResourceNotFoundError:
            return None
        return doc if doc.get("type") == PROCEDURE_CURSOR_TYPE else None

    async def _advance_procedure_cursor(
        self,
        scope: ProcedureScope,
        existing: Optional[dict[str, Any]],
        facts_read: list[dict[str, Any]],
        episodes_read: list[dict[str, Any]],
    ) -> Optional[dict[str, Any]]:
        """Move the user's procedure cursor past the memories read, and return the cursor record."""
        positions = advanced_procedure_cursor(existing, facts_read, episodes_read)
        if positions is None:
            return existing
        return await self._write_cursor(existing, procedure_cursor_body(existing, scope, *positions)) or existing

    async def _write_cursor(self, existing: Optional[dict[str, Any]], body: dict[str, Any]) -> Optional[dict[str, Any]]:
        """Create or conditionally replace the cursor record and return it as written, or ``None``
        when a concurrent run moved it first."""
        from azure.core import MatchConditions

        try:
            if existing is None:
                written = await self._create_item(self._memories_container, body=body)
            else:
                kwargs: dict[str, Any] = {"item": body["id"], "body": body}
                if existing.get("_etag"):
                    kwargs.update(etag=existing["_etag"], match_condition=MatchConditions.IfNotModified)
                written = await self._replace_item(self._memories_container, **kwargs)
        except (CosmosResourceExistsError, CosmosAccessConditionFailedError, CosmosResourceNotFoundError):
            logger.info("%s moved concurrently; keeping it", body["id"])
            return None
        return written if isinstance(written, dict) else body

    async def generate_thread_summary_durable(
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
            existing_summary = await self._read_item(
                self._summaries_container,
                item=summary_id,
                partition_key=[user_id, thread_id],
            )
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

        items = await self._query_items(
            self._turns_container,
            query=query,
            parameters=parameters,
            partition_key=[user_id, thread_id],
        )

        if existing_summary and not items:
            logger.info("generate_thread_summary_durable no new memories, returning existing")
            summary_doc = dict(existing_summary)
            summary_doc.pop("embedding", None)
            return summary_doc
        if not existing_summary and not items:
            raise ValidationError(f"No memories found for user_id={user_id!r}, thread_id={thread_id!r}")

        items = newest_window(items, recent_k)

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
        self._redact_for_embedding(doc)
        if doc.get("content") and not doc.get("embedding"):
            doc["embedding"] = await self._embed_one(doc["content"])
        validated = construct_internal(ThreadSummaryRecord, doc).to_doc()
        stored = await self._upsert_summary(validated)
        logger.info("persist_thread_summary completed id=%s", validated.get("id"))
        return stored

    async def generate_thread_summary(
        self,
        user_id: str,
        thread_id: str,
        recent_k: int | None = None,
    ) -> dict[str, Any]:
        """Generate or incrementally update a thread summary and persist it."""
        summary_doc = await self.generate_thread_summary_durable(user_id, thread_id, recent_k=recent_k)
        return await self.persist_thread_summary(user_id, thread_id, summary_doc)

    async def generate_user_summary_durable(
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
            existing_summary = await self._read_item(
                self._summaries_container,
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

        items = await self._query_items(
            self._memories_container,
            query=memories_query,
            parameters=parameters,
        )
        items.extend(
            await self._query_items(
                self._summaries_container,
                query=summaries_query,
                parameters=parameters,
            )
        )

        if existing_summary and not items:
            logger.info("generate_user_summary_durable no new memories, returning existing")
            user_doc = dict(existing_summary)
            user_doc.pop("embedding", None)
            return user_doc
        if not existing_summary and not items:
            raise NoSourceMemoriesError(f"No memories found for user_id={user_id!r}")

        if recent_k is not None:
            by_thread: dict[str, list[dict[str, Any]]] = defaultdict(list)
            for m in items:
                by_thread[m.get("thread_id", "")].append(m)
            items = [m for thread_items in by_thread.values() for m in newest_window(thread_items, recent_k)]
        items = newest_window(items, None)

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
        self._redact_for_embedding(doc)
        if doc.get("content") and not doc.get("embedding"):
            doc["embedding"] = await self._embed_one(doc["content"])
        validated = construct_internal(UserSummaryRecord, doc).to_doc()
        stored = await self._upsert_summary(validated)
        logger.info("persist_user_summary completed id=%s", validated.get("id"))
        return stored

    async def generate_user_summary(
        self,
        user_id: str,
        thread_ids: list[str] | None = None,
        recent_k: int | None = None,
    ) -> dict[str, Any]:
        """Generate or incrementally update a user summary and persist it."""
        summary_doc = await self.generate_user_summary_durable(user_id, thread_ids=thread_ids, recent_k=recent_k)
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
                "facts_checked": result["checked"],
                "pairs_compared": result["compared"],
                "kept": result["kept"],
                "merged": result["merged"],
                "contradicted": result["contradicted"],
                "vetoed": result["vetoed"],
                "combined": result["combined"],
                "synthesized": result["synthesized"],
                "combine_groups": result["combine_groups"],
                "combine_failed": result["combine_failed"],
                "duration_ms": duration_ms,
                "prompt_id": "dedup.prompty",
                "prompt_version": "v1",
            },
        )

    async def _active_memories_for_reconcile(self, user_id: str, memory_type: str, n: int) -> list[dict[str, Any]]:
        capped_n = top_literal(n, name="reconcile_memories.n")
        # Agent-sourced facts (sys:agent-fact) are excluded: they record what the
        # agent did/recommended (historical events), not mutable user state, so
        # they must never be contradiction-superseded by a later user statement.
        query = (
            f"SELECT TOP {capped_n} * FROM c "
            "WHERE c.user_id = @user_id "
            "AND c.type = @memory_type "
            f"AND {_ACTIVE_DOC_FILTER} "
            "AND NOT ARRAY_CONTAINS(c.tags, 'sys:agent-fact') "
            "ORDER BY c.created_at DESC"
        )
        return await self._query_items(
            self._memories_container,
            query=query,
            parameters=[
                {"name": "@user_id", "value": user_id},
                {"name": "@memory_type", "value": memory_type},
            ],
        )

    async def _load_memories_by_ids(
        self,
        user_id: str,
        memory_type: str,
        ids: Iterable[str],
    ) -> list[dict[str, Any]]:
        id_list = [mid for mid in dict.fromkeys(ids) if mid]
        if not id_list:
            return []
        placeholders = ", ".join(f"@id{i}" for i in range(len(id_list)))
        query = (
            "SELECT * FROM c WHERE c.user_id = @user_id "
            "AND c.type = @memory_type "
            f"AND c.id IN ({placeholders}) "
            f"AND {_ACTIVE_DOC_FILTER}"
        )
        parameters = [
            {"name": "@user_id", "value": user_id},
            {"name": "@memory_type", "value": memory_type},
        ]
        parameters.extend({"name": f"@id{i}", "value": mid} for i, mid in enumerate(id_list))
        return await self._query_items(self._memories_container, query=query, parameters=parameters)

    async def reconcile_memories(
        self, user_id: str, n: int = 50, *, memory_type: str = "fact", deadline: Optional[float] = None
    ) -> dict[str, int]:
        """Fold duplicate facts, then resolve contradictions among a user's most-recent active memories.

        Async mirror of :meth:`~azure.cosmos.agent_memory.services.pipeline.PipelineService.reconcile_memories`:
        :meth:`consolidate_memories` retires facts another fact states in full
        (``supersede_reason="duplicate"``, and ``"combined"``), then one
        contradiction call over the latest ``n`` active facts retires the loser of each
        ``contradicted_pairs`` entry (``supersede_reason="contradict"``). Episodic and procedural types
        are no-ops. ``deadline`` (a ``time.monotonic()`` value) ends the passes early.

        Returns ``{"kept", "merged", "contradicted"}`` - ``merged`` counts the facts retired as
        duplicates - plus the dedup pass's ``checked``, ``compared``, ``vetoed``, ``combined``,
        ``synthesized``, ``combine_groups`` and ``combine_failed`` counts.
        """
        validate_consolidation_args(user_id, n, memory_type)
        if memory_type in {"episodic", "procedural"}:
            result = dedup_result(checked=0, retired_seeds=0, merged=0, compared=0, vetoed=0)
            logger.info("reconcile_memories %s no-op user_id=%s result=%s", memory_type, user_id, result)
            return result

        started_at = time.monotonic()
        logger.info("reconcile_memories started user_id=%s n=%d memory_type=%s", user_id, n, memory_type)

        result = await self.consolidate_memories(user_id, n, memory_type=memory_type, deadline=deadline)
        facts: list[dict[str, Any]] = []
        if deadline is None or time.monotonic() < deadline:
            # Folds retired facts, so the contradiction pool is read after them.
            facts = await self._active_memories_for_reconcile(user_id, memory_type, n)
            contradictions = await self._reconcile_contradictions(user_id, memory_type, facts)
            result = {**result, "kept": contradictions["kept"], "contradicted": contradictions["contradicted"]}
        else:
            logger.info("reconcile_memories skipped its contradiction pass at its deadline user_id=%s", user_id)
        self._emit_reconcile_outcome(
            started_at=started_at,
            user_id=user_id,
            candidates=len(facts),
            result=result,
        )
        return result

    async def consolidate_memories(
        self, user_id: str, n: int = 50, *, memory_type: str = "fact", deadline: Optional[float] = None
    ) -> dict[str, int]:
        """Async mirror of
        :meth:`~azure.cosmos.agent_memory.services.pipeline.PipelineService.consolidate_memories`:
        fold duplicate facts written since the user's last dedup pass.

        Each fact checked is compared with its nearest active neighbours; one is retired into another
        only when nothing it states would be lost. Contradictions are left to the contradiction pass of
        :meth:`reconcile_memories`, and a fact that has absorbed others is never retired. It also
        combines facts about one specific thing, as in the sync pipeline.
        Episodic and procedural types are left alone. ``deadline`` ends the pass early, as in the sync
        pipeline.
        """
        validate_consolidation_args(user_id, n, memory_type)
        if memory_type != "fact":
            result = dedup_result(checked=0, retired_seeds=0, merged=0, compared=0, vetoed=0)
            logger.info("consolidate_memories %s no-op user_id=%s result=%s", memory_type, user_id, result)
            return result

        started_at = time.monotonic()
        combining = get_combine_enabled()
        # Combined facts are written after every check, so checking stops halfway to the deadline.
        check_until = deadline
        if combining and deadline is not None:
            check_until = started_at + max(0.0, deadline - started_at) / 2
        logger.info("consolidate_memories started user_id=%s n=%d memory_type=%s", user_id, n, memory_type)
        cursor = await self._read_dedup_cursor(user_id, memory_type)
        query, parameters = unchecked_memories_query(
            user_id, memory_type, cursor, top_literal(n, name="consolidate_memories.n")
        )
        seeds = await self._query_items(self._memories_container, query=query, parameters=parameters)

        checked: list[tuple[dict[str, Any], list[Verdict]]] = []
        done: list[dict[str, Any]] = []
        judged: set[frozenset[str]] = set()  # a pair two new facts share is judged once
        compared = 0
        for seed in seeds:
            if check_until is not None and time.monotonic() >= check_until:
                logger.info(
                    "consolidate_memories stopped at its deadline user_id=%s after %d of %d facts",
                    user_id,
                    len(done),
                    len(seeds),
                )
                break
            candidates = [
                (doc, score)
                for doc, score in await self._dedup_candidates(user_id, memory_type, seed, combine=combining)
                if frozenset((seed["id"], doc["id"])) not in judged
            ]
            if candidates:
                try:
                    verdicts = await self._verify_duplicates(seed, candidates)
                except Exception as exc:
                    if not is_retryable_llm_error(exc):
                        raise
                    logger.warning(
                        "consolidate_memories deferred user_id=%s after %d of %d facts: %s",
                        user_id,
                        len(done),
                        len(seeds),
                        describe_exception(exc),
                    )
                    break
                compared += len(candidates)
                judged.update(frozenset((seed["id"], doc["id"])) for doc, _ in candidates)
                checked.append((seed, verdicts))
            done.append(seed)

        plan = plan_folds(fold_verdicts(checked) if combining else checked)
        retired: set[str] = set()
        for survivor_id, losers in plan.retired.items():
            survivor = plan.survivors[survivor_id]
            retired.update(
                await self._commit_fold(survivor_patch(survivor, losers), losers, survivor_etag=survivor.get("_etag"))
            )
        combine = CombineCounts()
        settled = done
        if combining:
            # Facts a fold touched changed since they were read; a later pass sees them anew.
            groups = in_seed_order(plan_combine_groups(checked, set(plan.survivors) | plan.retired_ids()), done)
            combine, untried = await self._run_combine_groups(user_id, groups, deadline)
            settled = settled_seeds(done, untried)
        await self._advance_dedup_cursor(user_id, memory_type, cursor, settled)

        gone = retired | combine.retired_ids
        result = dedup_result(
            checked=len(done),
            retired_seeds=sum(1 for seed in done if seed["id"] in gone),
            merged=len(retired),
            compared=compared,
            vetoed=plan.vetoed,
            combined=len(combine.retired_ids),
            synthesized=combine.written,
            combine_groups=combine.groups,
            combine_failed=combine.failed,
        )
        logger.info("consolidate_memories pass user_id=%s memory_type=%s result=%s", user_id, memory_type, result)
        return result

    async def _dedup_candidates(
        self, user_id: str, memory_type: str, seed: dict[str, Any], *, combine: bool = False
    ) -> list[tuple[dict[str, Any], float]]:
        """The seed's nearest neighbours at or above the similarity floor, and with ``combine`` those down to
        ``COMBINE_SIMILARITY_FLOOR`` it could be combined with; none without an embedding."""
        if not isinstance(seed.get("embedding"), list) or not seed["embedding"]:
            return []
        query, parameters = neighbor_query(user_id, memory_type, seed)
        neighbors = await self._query_items(self._memories_container, query=query, parameters=parameters)
        return comparable_candidates(seed, neighbors) if combine else close_candidates(seed, neighbors)

    async def _verify_duplicates(
        self, seed: dict[str, Any], candidates: list[tuple[dict[str, Any], float]]
    ) -> list[Verdict]:
        """Ask the model how ``seed`` relates to each candidate.

        A retryable failure of the call propagates. A non-retryable one, or an answer that
        cannot be parsed, keeps every fact: only a usable answer can retire one, and the
        pass must not stall on a fact whose answers keep failing to parse.
        """
        try:
            text = await self._run_prompty(DEDUP_PROMPT, inputs=verify_inputs(seed, candidates))
        except Exception as exc:
            if is_retryable_llm_error(exc):
                raise
            logger.warning(
                "consolidate_memories: keeping %s; its duplicate check failed: %s",
                seed.get("id"),
                describe_exception(exc),
            )
            return []
        try:
            parsed = self._parse_llm_json(text)
        except Exception as exc:
            logger.warning(
                "consolidate_memories: keeping %s; its duplicate check was unusable: %s",
                seed.get("id"),
                describe_exception(exc),
            )
            return []
        return parse_verdicts(parsed, candidates)

    async def _read_dedup_cursor(self, user_id: str, memory_type: str) -> Optional[dict[str, Any]]:
        try:
            doc = await self._read_item(
                self._memories_container,
                item=dedup_cursor_id(user_id, memory_type),
                partition_key=[user_id, DEDUP_CURSOR_THREAD_ID],
            )
        except CosmosResourceNotFoundError:
            return None
        return doc if isinstance(doc, dict) and doc.get("type") == DEDUP_CURSOR_TYPE else None

    async def _advance_dedup_cursor(
        self,
        user_id: str,
        memory_type: str,
        existing: Optional[dict[str, Any]],
        checked: list[dict[str, Any]],
    ) -> None:
        """Move the user's dedup cursor past ``checked``. A concurrent pass that moved it first wins;
        at worst a later pass checks some facts again, which retires nothing new."""
        from azure.core import MatchConditions
        from azure.cosmos.exceptions import CosmosAccessConditionFailedError

        position = next_dedup_cursor(existing, checked)
        if position is None:
            return
        body = dedup_cursor_body(existing, user_id, memory_type, position)
        try:
            if existing is None:
                await self._create_item(self._memories_container, body=body)
            else:
                await self._replace_item(
                    self._memories_container,
                    item=body["id"],
                    body=body,
                    etag=existing.get("_etag"),
                    match_condition=MatchConditions.IfNotModified,
                )
        except (CosmosResourceExistsError, CosmosAccessConditionFailedError):
            logger.info("consolidate_memories: the dedup cursor moved concurrently; keeping it user_id=%s", user_id)

    async def _commit_fold(
        self,
        survivor: dict[str, Any],
        losers: list[dict[str, Any]],
        *,
        survivor_etag: Optional[str] = None,
    ) -> list[str]:
        """Write ``survivor``, then retire each of ``losers`` into it as a duplicate.

        There is no multi-document transaction, so the survivor goes first: a crash between the writes
        leaves it listing a duplicate that is still active, and a later pass folds that again. Returns
        the IDs retired; none, having written nothing, when the survivor changed or disappeared since
        this pass read it. A loser that changed since then stays active.
        """
        from azure.core import MatchConditions
        from azure.cosmos.exceptions import CosmosAccessConditionFailedError

        try:
            if survivor_etag:
                await self._replace_item(
                    self._memories_container,
                    item=survivor["id"],
                    body=survivor,
                    etag=survivor_etag,
                    match_condition=MatchConditions.IfNotModified,
                )
            else:
                await self._upsert_item(self._memories_container, body=survivor)
        except (CosmosAccessConditionFailedError, CosmosResourceNotFoundError):
            logger.info("consolidate_memories: fold into %s skipped; it changed since this pass", survivor["id"])
            return []
        return [
            loser["id"] for loser in losers if await self._mark_superseded(loser, survivor["id"], reason="duplicate")
        ]

    async def _run_combine_groups(
        self, user_id: str, groups: list[list[dict[str, Any]]], deadline: Optional[float]
    ) -> tuple[CombineCounts, list[list[dict[str, Any]]]]:
        """Async mirror of the sync ``_run_combine_groups``: write the planned groups, a few at once, until
        ``deadline`` or a retryable model error; returns the counts and the groups not tried."""
        import asyncio

        counts = CombineCounts()
        queue = list(groups)
        untried: list[list[dict[str, Any]]] = []
        if not queue:
            return counts, untried
        workers = get_combine_workers()
        running: dict[asyncio.Task[CombineOutcome], list[dict[str, Any]]] = {}
        stopped = False
        try:
            while queue or running:
                while queue and len(running) < workers and not stopped:
                    if deadline is not None and time.monotonic() >= deadline:
                        logger.info("consolidate_memories stopped combining at its deadline user_id=%s", user_id)
                        stopped = True
                        break
                    group = queue.pop(0)
                    running[asyncio.ensure_future(arun_combine_group(group, self._combine_call))] = group
                if not running:
                    break
                finished, _ = await asyncio.wait(list(running), return_when=asyncio.FIRST_COMPLETED)
                for task in finished:
                    group = running.pop(task)
                    outcome = task.result()
                    write = (
                        await self._write_combined(outcome.draft)
                        if outcome.status == "combined"
                        else CombineWrite(outcome.status)
                    )
                    if write.status == "deferred":
                        logger.warning(
                            "consolidate_memories deferred combining user_id=%s: %s",
                            user_id,
                            outcome.error or "embedding failed",
                        )
                        stopped = True
                        untried.append(group)
                        continue
                    if outcome.status == "skipped":
                        logger.warning("consolidate_memories skipped a group user_id=%s: %s", user_id, outcome.error)
                    counts.groups += 1
                    if write.status == "written":
                        counts.written += 1
                        counts.retired_ids.update(write.retired)
                    else:
                        counts.failed += 1
        finally:
            # A failed write must not leave the other groups' model calls running.
            for task in running:
                task.cancel()
        return counts, untried + queue

    async def _combine_call(self, prompt: str, inputs: dict[str, str]) -> dict[str, Any]:
        text = await self._run_prompty(prompt, inputs=inputs)
        try:
            return self._parse_llm_json(text)
        except Exception:  # noqa: BLE001 - an unusable answer counts as an empty one
            logger.warning("consolidate_memories got an unusable answer from %s", prompt)
            return {}

    async def _write_combined(self, draft: Optional[CombineDraft]) -> CombineWrite:
        """Async mirror of the sync ``_write_combined``: validate, embed and apply a combined fact."""
        if draft is None:
            return CombineWrite("lost")
        members = list(draft.members)
        try:
            record = self._validate_extracted_doc(
                combined_record(members, draft.text, lineage=self._prompt_lineage(COMBINE_PROMPT))
            )
        except Exception as exc:  # noqa: BLE001 - a fact that cannot be built keeps its members
            logger.warning(
                "consolidate_memories: keeping %d facts; their combined fact is invalid: %s",
                len(members),
                describe_exception(exc),
            )
            return CombineWrite("lost")
        try:
            record["embedding"] = await self._embed_one(record["content"])
        except Exception as exc:
            if is_retryable_llm_error(exc):
                logger.warning("consolidate_memories deferred combining: embedding failed: %s", describe_exception(exc))
                return CombineWrite("deferred")
            logger.warning(
                "consolidate_memories: keeping %d facts; their embedding failed: %s",
                len(members),
                describe_exception(exc),
            )
            return CombineWrite("lost")
        retired = await self._commit_combine(record, members)
        return CombineWrite("lost") if retired is None else CombineWrite("written", tuple(retired))

    async def _commit_combine(self, record: dict[str, Any], members: list[dict[str, Any]]) -> Optional[list[str]]:
        """Async mirror of the sync ``_commit_combine``: create ``record``, then retire each of ``members`` into it
        as combined. Returns the IDs retired, or ``None`` when the fact that is there was retired since."""
        try:
            await self._create_memory(record)
        except CosmosResourceExistsError:
            try:
                existing = await self._read_item(
                    self._memories_container,
                    item=record["id"],
                    partition_key=[record["user_id"], record["thread_id"]],
                )
            except CosmosResourceNotFoundError:
                return None
            if isinstance(existing, dict) and existing.get("superseded_by"):
                logger.info(
                    "consolidate_memories: combined fact %s was retired since; keeping its members", record["id"]
                )
                return None
            logger.info("consolidate_memories: combined fact %s already exists", record["id"])
        return [doc["id"] for doc in members if await self._mark_superseded(doc, record["id"], reason="combined")]

    async def _reconcile_contradictions(
        self, user_id: str, memory_type: str, facts: list[dict[str, Any]]
    ) -> dict[str, int]:
        """Async mirror: resolve only ``contradicted_pairs`` within the pool.

        Paraphrases are not merged here; the dedup prompt/schema does not emit
        duplicate groups because the dedup pass that runs first
        (:meth:`consolidate_memories`) folds them, so no merged docs are minted
        and the pass is convergent. Returns
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

        response_text = await self._run_prompty("dedup.prompty", inputs={"facts_text": facts_text})
        parsed = self._parse_llm_json(response_text)
        contradicted_pairs = parsed.get("contradicted_pairs", []) or []

        facts_by_id: dict[str, dict[str, Any]] = {f["id"]: f for f in facts}
        contradicted = 0
        consumed_loser_ids: set[str] = set()
        for pair in contradicted_pairs:
            winner_id = pair.get("winner_id")
            loser_id = pair.get("loser_id")
            if not winner_id or not loser_id or winner_id == loser_id:
                continue
            if winner_id not in facts_by_id:
                logger.warning(
                    "reconcile_memories: hallucinated winner_id=%s not in pool; skipping pair %s",
                    describe_shape(winner_id),
                    describe_shape(pair),
                )
                continue
            # Guard chained contradictions (A>B then B>C) and re-supersession.
            if winner_id in consumed_loser_ids or loser_id in consumed_loser_ids:
                logger.info(
                    "reconcile_memories: skipping chained/duplicate contradiction pair %s "
                    "(winner or loser already superseded this pass)",
                    describe_shape(pair),
                )
                continue
            loser_doc = facts_by_id.get(loser_id)
            if loser_doc is None:
                continue
            if await self._mark_superseded(loser_doc, winner_id, reason="contradict"):
                contradicted += 1
                consumed_loser_ids.add(loser_id)

        kept = len([fid for fid in facts_by_id if fid not in consumed_loser_ids])
        result = {"kept": kept, "merged": 0, "contradicted": contradicted}
        logger.info(
            "reconcile_memories contradiction pass user_id=%s memory_type=%s result=%s",
            user_id,
            memory_type,
            result,
        )
        return result

    async def build_procedural_context(self, user_id: str, task: Optional[str] = None) -> str:
        """The standing-instructions block for a session start, or ``""`` when there is none.

        Lists the user's active procedures that apply to nearly every interaction and are not
        past ``valid_until``: taught before corrected before repeated, most recently updated
        first, capped at ``STANDING_RULES_MAX`` rules in ``STANDING_RULES_MAX_BYTES``. With a
        ``task``, it also shows the active situational procedure nearest to it (by vector
        similarity), rendered "When <conditions>: <rule>".
        """
        if not user_id:
            raise ValidationError("user_id is required")
        scope = ProcedureScope(user_id)
        today = utc_today()
        standing, left_out = standing_rules(
            await self._query_procedures(scope, *standing_rules_query(scope, today)), today
        )
        if left_out:
            logger.info(
                "build_procedural_context left out %d standing rules over the cap user_id=%s", left_out, user_id
            )
        situational: list[dict[str, Any]] = []
        if isinstance(task, str) and task.strip():
            query, parameters = situational_rule_query(scope, await self._embed_one(task), today)
            situational = [
                doc for doc in await self._query_procedures(scope, query, parameters) if in_force(doc, today)
            ]
        logger.debug(
            "build_procedural_context included_ids=%s user_id=%s",
            [doc.get("id") for doc in [*standing, *situational]],
            user_id,
        )
        return procedural_context(standing, situational)


__all__ = ["AsyncPipelineService"]
