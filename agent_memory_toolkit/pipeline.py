"""Processing pipeline for memory extraction, summarization, and dedup.

Shared by both the SDK (in-process calls) and Azure Functions (change feed trigger).
Uses ChatClient for chat completions and EmbeddingsClient for embeddings.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import os
from collections import defaultdict
from datetime import datetime, timezone
from typing import Any, Optional

from ._utils import DEFAULT_TTL_BY_TYPE, compute_content_hash
from .exceptions import LLMError, ValidationError

logger = logging.getLogger(__name__)


class ProcessingPipeline:
    """Memory processing engine.

    Parameters
    ----------
    cosmos_container : ContainerProxy or AsyncContainerProxy
        The Cosmos DB container client for reading/writing memories.
    chat_client : ChatClient
        Client for LLM chat completions.
    embeddings_client : EmbeddingsClient
        Client for embedding generation.
    prompts_dir : str, optional
        Directory containing ``.prompty`` prompt templates.  Defaults to
        ``agent_memory_toolkit/prompts/`` bundled with the package.
    """

    def __init__(
        self,
        cosmos_container: Any,
        chat_client: Any,
        embeddings_client: Any,
        prompts_dir: str | None = None,
    ) -> None:
        self._container = cosmos_container
        self._llm = chat_client
        self._embeddings = embeddings_client

        if prompts_dir is not None:
            self._prompts_dir = prompts_dir
        else:
            # Default: prompts/ directory bundled inside the package
            pkg_dir = os.path.dirname(os.path.abspath(__file__))
            self._prompts_dir = os.path.join(pkg_dir, "prompts")

        # Cache of loaded prompty.Prompty objects keyed by filename
        self._prompty_cache: dict[str, Any] = {}

    # -----------------------------------------------------------------------
    # Private helpers
    # -----------------------------------------------------------------------

    def _load_prompty(self, filename: str) -> Any:
        """Load and cache a ``.prompty`` template.

        The returned object exposes ``model.parameters`` (temperature,
        ``response_format``, etc.) and is consumed by ``prompty.prepare``
        to render the final ``messages`` list.
        """
        cached = self._prompty_cache.get(filename)
        if cached is not None:
            return cached

        import prompty  # local import to avoid a hard dependency at import time

        path = os.path.join(self._prompts_dir, filename)
        loaded = prompty.load(path)
        self._prompty_cache[filename] = loaded
        return loaded

    def _run_prompty(
        self,
        filename: str,
        inputs: dict[str, Any],
    ) -> str:
        """Render a prompty template, run the LLM, and return the response text.

        Model options from the prompty file (``temperature``,
        ``response_format``, etc.) are passed straight through to the
        underlying ``ChatClient.generate`` call — no per-call hardcoding.
        """
        import prompty

        p = self._load_prompty(filename)
        messages = self._messages_to_dicts(prompty.prepare(p, inputs=inputs))
        params = self._extract_prompty_params(p)
        return self._llm.generate(messages, **params)

    @staticmethod
    def _messages_to_dicts(messages: Any) -> list[dict[str, str]]:
        """Normalize prompty's prepared output to OpenAI-style message dicts.

        Prompty 2.x returns ``list[Message]`` dataclasses with ``role`` and
        ``parts`` (rich content parts). Older releases returned plain dicts.
        We collapse text parts into a single ``content`` string so the result
        is always the ``[{"role": ..., "content": ...}]`` shape OpenAI's
        chat completions API expects.
        """
        normalized: list[dict[str, str]] = []
        for msg in messages or []:
            if isinstance(msg, dict):
                normalized.append(msg)
                continue
            role = getattr(msg, "role", None)
            content = getattr(msg, "text", None)
            if content is None:
                parts = getattr(msg, "parts", None) or []
                content = "".join(getattr(part, "value", "") for part in parts)
            if role is None:
                continue
            normalized.append({"role": role, "content": content or ""})
        return normalized

    # Mapping from prompty 2.x ModelOptions field names (camelCase) to the
    # snake_case kwargs accepted by OpenAI's chat completions API.
    _PROMPTY_OPTION_ALIASES = {
        "topP": "top_p",
        "topK": "top_k",
        "frequencyPenalty": "frequency_penalty",
        "presencePenalty": "presence_penalty",
        "maxOutputTokens": "max_tokens",
        "stopSequences": "stop",
        "allowMultipleToolCalls": "parallel_tool_calls",
    }

    @classmethod
    def _extract_prompty_params(cls, p: Any) -> dict[str, Any]:
        """Pull model parameters from a Prompty object across library versions.

        - Prompty 2.x exposes ``model.options`` as a ``ModelOptions``
          dataclass with camelCase fields plus an ``additionalProperties``
          dict for things like ``response_format``.
        - Older 0.1.x releases expose ``model.parameters`` as a plain dict.

        We probe both, normalize camelCase → snake_case for known aliases,
        flatten ``additionalProperties``, and drop ``None`` values so the
        underlying ChatClient defaults still apply when a field is unset.
        """
        model = getattr(p, "model", None)
        if model is None:
            return {}

        # Prompty 0.1.x: parameters is already a dict.
        legacy = getattr(model, "parameters", None)
        if legacy:
            return {k: v for k, v in dict(legacy).items() if v is not None}

        options = getattr(model, "options", None)
        if options is None:
            return {}

        # Prompty 2.x: ModelOptions dataclass.
        try:
            import dataclasses

            raw = dataclasses.asdict(options) if dataclasses.is_dataclass(options) else dict(options)
        except Exception:
            raw = {}

        params: dict[str, Any] = {}
        for key, value in raw.items():
            if value is None:
                continue
            if key in ("additionalProperties", "additional_properties"):
                if isinstance(value, dict):
                    params.update(value)
                continue
            if isinstance(value, list) and not value:
                continue
            params[cls._PROMPTY_OPTION_ALIASES.get(key, key)] = value
        return params

    @staticmethod
    def _build_transcript(
        items: list[dict[str, Any]],
        *,
        group_by_thread: bool = False,
    ) -> str:
        """Build a formatted transcript from memory documents.

        Parameters
        ----------
        items:
            Memory dicts with ``role``, ``content``, and optional ``metadata``.
        group_by_thread:
            If *True*, group messages under ``=== Thread <id> ===`` headers.
        """
        if not group_by_thread:
            lines: list[str] = []
            for m in items:
                role = m.get("role", "unknown")
                content = m.get("content", "")
                metadata = m.get("metadata", {})
                meta_str = f" [metadata: {json.dumps(metadata)}]" if metadata else ""
                lines.append(f"[{role}]: {content}{meta_str}")
            return "\n".join(lines)

        threads: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for m in items:
            threads[m.get("thread_id", "")].append(m)

        parts: list[str] = []
        for tid, thread_items in threads.items():
            parts.append(f"=== Thread {tid} ===")
            for m in thread_items:
                role = m.get("role", "unknown")
                content = m.get("content", "")
                metadata = m.get("metadata", {})
                meta_str = f" [metadata: {json.dumps(metadata)}]" if metadata else ""
                parts.append(f"[{role}]: {content}{meta_str}")
            parts.append("")
        return "\n".join(parts)

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
        query = (
            f"SELECT TOP {limit} * FROM c "
            f"WHERE c.user_id = @user_id "
            f"AND c.type IN ({type_placeholders}) "
            f"AND (NOT IS_DEFINED(c.superseded_by) OR c.superseded_by = null) "
            f"ORDER BY c._ts DESC"
        )
        parameters: list[dict[str, Any]] = [
            {"name": "@user_id", "value": user_id},
        ]
        for i, mt in enumerate(memory_types):
            parameters.append({"name": f"@mtype{i}", "value": mt})

        items = list(
            self._container.query_items(
                query=query,
                parameters=parameters,
                enable_cross_partition_query=True,
            )
        )
        return items

    def _upsert_memory(self, doc: dict[str, Any]) -> dict[str, Any]:
        """Upsert a single memory document to Cosmos DB."""
        self._container.upsert_item(body=doc)
        return doc

    def _mark_superseded(self, old_doc: dict[str, Any], superseder_id: str) -> bool:
        """Atomically set ``superseded_by`` on ``old_doc`` using ETag protection.

        Supersession is advisory — losing a race here just means another writer
        already marked the same memory, so we log and return False instead of
        raising. Returns True on success.

        Using ``replace_item`` with ``MatchConditions.IfNotModified`` prevents
        the read-modify-write hazard where two concurrent extractions both
        load an old fact, both compute their own ``det_id``, and the
        slower writer overwrites the faster writer's ``superseded_by`` link.
        """
        from azure.core import MatchConditions
        from azure.cosmos.exceptions import CosmosAccessConditionFailedError

        etag = old_doc.get("_etag")
        new_doc = {**old_doc, "superseded_by": superseder_id}
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
            logger.exception(
                "supersede failed id=%s superseder=%s",
                old_doc.get("id"),
                superseder_id,
            )
            return False

    @staticmethod
    def _parse_llm_json(text: str | None) -> dict[str, Any]:
        """Parse JSON from an LLM response, stripping markdown fences."""
        if text is None:
            raise LLMError("LLM returned no content (None response body)")
        cleaned = text.strip()
        if cleaned.startswith("```"):
            first_newline = cleaned.find("\n")
            if first_newline >= 0:
                cleaned = cleaned[first_newline + 1 :]
            else:
                cleaned = cleaned.lstrip("`").lstrip()
        if cleaned.endswith("```"):
            cleaned = cleaned[:-3]
        try:
            return json.loads(cleaned.strip())
        except json.JSONDecodeError as exc:
            preview = (text or "")[:200].replace("\n", " ")
            raise LLMError(f"LLM returned invalid JSON (preview={preview!r}): {exc}") from exc

    # -----------------------------------------------------------------------
    # Public API
    # -----------------------------------------------------------------------

    def extract_memories(
        self,
        user_id: str,
        thread_id: str,
        recent_k: int | None = None,
    ) -> dict[str, int]:
        """Extract facts, procedural rules, and episodic memories from a thread.

        Returns a summary dict with counts of extracted items.
        """
        if not user_id:
            raise ValidationError("user_id is required")
        if not thread_id:
            raise ValidationError("thread_id is required")

        logger.info(
            "extract_memories started user_id=%s thread_id=%s",
            user_id,
            thread_id,
        )

        # ---- 1. Query thread memories ----
        query = "SELECT * FROM c WHERE c.user_id = @user_id AND c.thread_id = @thread_id"
        parameters: list[dict[str, Any]] = [
            {"name": "@user_id", "value": user_id},
            {"name": "@thread_id", "value": thread_id},
        ]
        items = list(
            self._container.query_items(
                query=query,
                parameters=parameters,
                partition_key=[user_id, thread_id],
            )
        )

        # Sort and trim
        items.sort(key=lambda m: m.get("created_at", ""), reverse=True)
        if recent_k is not None:
            items = items[:recent_k]
        items.reverse()  # chronological order

        if not items:
            logger.warning(
                "extract_memories no memories found user_id=%s thread_id=%s",
                user_id,
                thread_id,
            )
            return {"facts_count": 0, "procedural_count": 0, "episodic_count": 0, "updated_count": 0}

        # ---- 2. Load existing memories for reconciliation ----
        existing = self._load_existing_memories(user_id, ["fact", "procedural"])
        existing_text = ""
        if existing:
            lines = []
            for mem in existing:
                lines.append(
                    f"- [ID: {mem['id']}] {mem.get('content', '')} "
                    f"(type={mem.get('type', 'fact')}, salience={mem.get('salience', 'N/A')})"
                )
            existing_text = "\n".join(lines)
        else:
            existing_text = "(none)"

        # ---- 3. Build transcript and call LLM ----
        transcript = self._build_transcript(items)

        response_text = self._run_prompty(
            "extract_memories.prompty",
            inputs={"existing_facts": existing_text, "transcript": transcript},
        )

        # ---- 4. Parse LLM response ----
        parsed = self._parse_llm_json(response_text)
        facts = parsed.get("facts", [])
        procedural = parsed.get("procedural", [])
        episodic = parsed.get("episodic", [])
        unclassified = parsed.get("unclassified", [])

        now = datetime.now(timezone.utc).isoformat()
        docs_to_embed: list[dict[str, Any]] = []
        embed_texts: list[str] = []
        updated_count = 0

        # ---- 5. Process facts ----
        for fact in facts:
            action = fact.get("action", "ADD").upper()
            if action == "NONE":
                continue

            text = fact.get("text")
            if not text:
                logger.warning(
                    "extract_memories: dropping malformed fact (missing 'text'): %r",
                    fact,
                )
                continue
            content_hash = compute_content_hash(text)
            det_id = f"fact_{hashlib.sha256(f'{user_id}:{thread_id}:{content_hash}'.encode()).hexdigest()[:16]}"

            topic_tags = [f"topic:{t}" for t in fact.get("tags", [])]
            tags = ["sys:fact", "sys:auto-extracted"] + topic_tags

            confidence = fact.get("confidence")
            if confidence is None:
                confidence = 0.5

            doc: dict[str, Any] = {
                "id": det_id,
                "user_id": user_id,
                "thread_id": thread_id,
                "role": "system",
                "type": "fact",
                "content": text,
                "content_hash": content_hash,
                "confidence": confidence,
                "metadata": {
                    "category": fact.get("category"),
                    "subject": fact.get("subject"),
                    "predicate": fact.get("predicate"),
                    "object": fact.get("object"),
                    "temporal_context": fact.get("temporal_context"),
                },
                "salience": fact.get("salience"),
                "tags": tags,
                "created_at": now,
            }

            if action == "UPDATE" and fact.get("supersedes_id"):
                doc["supersedes_ids"] = [fact["supersedes_id"]]
                # Mark old memory as superseded
                try:
                    old_mem = self._container.read_item(
                        item=fact["supersedes_id"],
                        partition_key=[user_id, thread_id],
                    )
                    if self._mark_superseded(old_mem, det_id):
                        updated_count += 1
                except Exception:
                    # Try cross-partition query if direct read fails
                    logger.debug(
                        "Could not read superseded item %s directly, trying cross-partition",
                        fact["supersedes_id"],
                    )
                    try:
                        q = "SELECT * FROM c WHERE c.id = @id AND c.user_id = @uid"
                        results = list(
                            self._container.query_items(
                                query=q,
                                parameters=[
                                    {"name": "@id", "value": fact["supersedes_id"]},
                                    {"name": "@uid", "value": user_id},
                                ],
                                enable_cross_partition_query=True,
                            )
                        )
                        if results and self._mark_superseded(results[0], det_id):
                            updated_count += 1
                    except Exception as exc:
                        logger.warning(
                            "Failed to mark superseded memory %s: %s",
                            fact["supersedes_id"],
                            exc,
                        )

            docs_to_embed.append(doc)
            embed_texts.append(text)

        # ---- 6. Process procedural ----
        for proc in procedural:
            action = proc.get("action", "ADD").upper()
            if action == "NONE":
                continue

            text = proc.get("instruction")
            if not text:
                logger.warning(
                    "extract_memories: dropping malformed procedural (missing 'instruction'): %r",
                    proc,
                )
                continue
            content_hash = compute_content_hash(text)
            det_id = f"proc_{hashlib.sha256(f'{user_id}:{content_hash}'.encode()).hexdigest()[:16]}"

            topic_tags = [f"topic:{t}" for t in proc.get("tags", [])]
            tags = ["sys:procedural", "sys:auto-extracted"] + topic_tags

            confidence = proc.get("confidence")
            if confidence is None:
                confidence = 0.5

            doc = {
                "id": det_id,
                "user_id": user_id,
                "thread_id": "__procedural__",
                "role": "system",
                "type": "procedural",
                "content": text,
                "content_hash": content_hash,
                "confidence": confidence,
                "metadata": {
                    "trigger": proc.get("trigger"),
                    "category": proc.get("category"),
                    "source": proc.get("source"),
                    "priority": proc.get("priority"),
                },
                "salience": proc.get("salience"),
                "tags": tags,
                "created_at": now,
            }

            if action == "UPDATE" and proc.get("supersedes_id"):
                doc["supersedes_ids"] = [proc["supersedes_id"]]
                try:
                    q = "SELECT * FROM c WHERE c.id = @id AND c.user_id = @uid"
                    results = list(
                        self._container.query_items(
                            query=q,
                            parameters=[
                                {"name": "@id", "value": proc["supersedes_id"]},
                                {"name": "@uid", "value": user_id},
                            ],
                            enable_cross_partition_query=True,
                        )
                    )
                    if results and self._mark_superseded(results[0], det_id):
                        updated_count += 1
                except Exception as exc:
                    logger.warning(
                        "Failed to mark superseded procedural memory %s: %s",
                        proc["supersedes_id"],
                        exc,
                    )

            docs_to_embed.append(doc)
            embed_texts.append(text)

        # ---- 7. Process episodic ----
        for ep in episodic:
            situation = ep.get("situation")
            action_taken = ep.get("action_taken")
            outcome = ep.get("outcome")
            if not (situation and action_taken and outcome):
                logger.warning(
                    "extract_memories: dropping malformed episodic (missing situation/action_taken/outcome): %r",
                    ep,
                )
                continue
            text = f"{situation} → {action_taken} → {outcome}"
            content_hash = compute_content_hash(text)
            det_id = f"ep_{hashlib.sha256(f'{user_id}:{thread_id}:{content_hash}'.encode()).hexdigest()[:16]}"

            topic_tags = [f"topic:{t}" for t in ep.get("tags", [])]
            tags = ["sys:episodic", "sys:auto-extracted"] + topic_tags

            confidence = ep.get("confidence")
            if confidence is None:
                confidence = 0.5

            doc = {
                "id": det_id,
                "user_id": user_id,
                "thread_id": thread_id,
                "role": "system",
                "type": "episodic",
                "content": text,
                "content_hash": content_hash,
                "confidence": confidence,
                "ttl": DEFAULT_TTL_BY_TYPE.get("episodic", 7_776_000),
                "metadata": {
                    "situation": ep.get("situation"),
                    "action_taken": ep.get("action_taken"),
                    "outcome": ep.get("outcome"),
                    "reasoning": ep.get("reasoning"),
                    "outcome_valence": ep.get("outcome_valence"),
                    "lesson": ep.get("lesson"),
                    "domain": ep.get("domain"),
                },
                "salience": ep.get("salience"),
                "tags": tags,
                "created_at": now,
            }

            docs_to_embed.append(doc)
            embed_texts.append(text)

        # ---- 8. Process unclassified ----
        # The LLM uses the `unclassified` bucket when it cannot confidently
        # decide between fact / procedural / episodic. We persist these as
        # facts (the most common type, retrieval already handles them well)
        # tagged `sys:unclassified` so they're easy to audit and reclassify.
        for item in unclassified:
            text = item.get("text")
            if not text:
                continue
            content_hash = compute_content_hash(text)
            det_id = f"unc_{hashlib.sha256(f'{user_id}:{thread_id}:{content_hash}'.encode()).hexdigest()[:16]}"

            topic_tags = [f"topic:{t}" for t in item.get("tags", [])]
            tags = ["sys:fact", "sys:auto-extracted", "sys:unclassified"] + topic_tags

            confidence = item.get("confidence")
            if confidence is None:
                confidence = 0.5

            doc = {
                "id": det_id,
                "user_id": user_id,
                "thread_id": thread_id,
                "role": "system",
                "type": "fact",
                "content": text,
                "content_hash": content_hash,
                "confidence": confidence,
                "metadata": {
                    "unclassified_reason": item.get("reason"),
                },
                "salience": item.get("salience"),
                "tags": tags,
                "created_at": now,
            }

            docs_to_embed.append(doc)
            embed_texts.append(text)

        # ---- 9. Generate embeddings in batch ----
        if embed_texts:
            logger.info("extract_memories generating embeddings for %d items", len(embed_texts))
            embeddings = self._embeddings.generate_batch(embed_texts)
            for doc, emb in zip(docs_to_embed, embeddings):
                doc["embedding"] = emb

        # ---- 10. Upsert all documents ----
        for doc in docs_to_embed:
            self._upsert_memory(doc)

        result = {
            "facts_count": sum(
                1 for d in docs_to_embed if d["type"] == "fact" and "sys:unclassified" not in d.get("tags", [])
            ),
            "procedural_count": sum(1 for d in docs_to_embed if d["type"] == "procedural"),
            "episodic_count": sum(1 for d in docs_to_embed if d["type"] == "episodic"),
            "unclassified_count": sum(1 for d in docs_to_embed if "sys:unclassified" in d.get("tags", [])),
            "updated_count": updated_count,
        }
        logger.info("extract_memories completed: %s", result)
        return result

    def generate_thread_summary(
        self,
        user_id: str,
        thread_id: str,
        recent_k: int | None = None,
    ) -> dict[str, Any]:
        """Generate or incrementally update a thread summary.

        Returns the summary document dict.
        """
        if not user_id:
            raise ValidationError("user_id is required")
        if not thread_id:
            raise ValidationError("thread_id is required")

        logger.info(
            "generate_thread_summary started user_id=%s thread_id=%s",
            user_id,
            thread_id,
        )

        # ---- 1. Check for existing summary ----
        summary_id = f"summary_{user_id}_{thread_id}"
        existing_summary: Optional[dict[str, Any]] = None
        try:
            existing_summary = self._container.read_item(
                item=summary_id,
                partition_key=[user_id, thread_id],
            )
        except Exception:
            pass  # first time — full generation

        # ---- 2. Query memories (time-filtered if updating) ----
        query = "SELECT * FROM c WHERE c.user_id = @user_id AND c.thread_id = @thread_id AND c.type != 'summary'"
        parameters: list[dict[str, Any]] = [
            {"name": "@user_id", "value": user_id},
            {"name": "@thread_id", "value": thread_id},
        ]

        if existing_summary:
            since = existing_summary["updated_at"]
            query += " AND c.created_at > @since"
            parameters.append({"name": "@since", "value": since})

        query_started_at = datetime.now(timezone.utc).isoformat()

        items = list(
            self._container.query_items(
                query=query,
                parameters=parameters,
                partition_key=[user_id, thread_id],
            )
        )

        if existing_summary and not items:
            logger.info("generate_thread_summary no new memories, returning existing")
            return existing_summary

        if not existing_summary and not items:
            raise ValidationError(f"No memories found for user_id={user_id!r}, thread_id={thread_id!r}")

        # ---- 3. Sort and trim ----
        items.sort(key=lambda m: m.get("created_at", ""), reverse=True)
        if recent_k is not None:
            items = items[:recent_k]
        items.reverse()  # chronological order

        # ---- 4. Build transcript ----
        transcript = self._build_transcript(items)

        # ---- 5. Call LLM ----
        if existing_summary:
            prior_json = existing_summary.get("metadata", {}).get("structured_summary")
            if prior_json:
                prior_text = json.dumps(prior_json, indent=2)
            else:
                prior_text = existing_summary.get("content", "")
            response_text = self._run_prompty(
                "summarize_update.prompty",
                inputs={"prior_summary": prior_text, "transcript": transcript},
            )
        else:
            response_text = self._run_prompty(
                "summarize.prompty",
                inputs={"transcript": transcript},
            )

        # ---- 6. Parse response ----
        parsed = self._parse_llm_json(response_text)
        overview = parsed.get("overview", response_text)
        topics = parsed.get("topics", [])

        # ---- 7. Generate embedding from overview ----
        summary_embedding = self._embeddings.generate(overview)

        # ---- 8. Build and upsert summary doc ----
        if existing_summary:
            old_source_count = existing_summary.get("metadata", {}).get("source_count", 0)
            total_source_count = old_source_count + len(items)
        else:
            total_source_count = len(items)

        topic_tags = [f"topic:{t}" for t in topics]
        tags = ["sys:summary"] + topic_tags

        summary_doc: dict[str, Any] = {
            "id": summary_id,
            "user_id": user_id,
            "thread_id": thread_id,
            "role": "system",
            "type": "summary",
            "content": overview,
            "embedding": summary_embedding,
            "salience": 1.0,
            "tags": tags,
            "metadata": {
                "structured_summary": parsed,
                "source_count": total_source_count,
                "recent_k": recent_k,
                "incremental_update": existing_summary is not None,
            },
            "created_at": existing_summary["created_at"] if existing_summary else query_started_at,
            "updated_at": query_started_at,
        }

        self._upsert_memory(summary_doc)
        logger.info(
            "generate_thread_summary completed id=%s source_count=%d",
            summary_id,
            total_source_count,
        )
        return summary_doc

    def generate_user_summary(
        self,
        user_id: str,
        thread_ids: list[str] | None = None,
        recent_k: int | None = None,
    ) -> dict[str, Any]:
        """Generate or incrementally update a cross-thread user summary.

        ``thread_ids`` is observability metadata — recorded on the resulting
        document for debugging/auditing — but **not** used to filter the
        query. Filtering by ``thread_ids`` would silently drop memories from
        threads contributing earlier in the cross-counter window: if N
        change-feed batches accumulate before USER_SUMMARY_EVERY_N is
        crossed, only the threads in the *last* batch would be visible to
        the query, and pre-existing facts on other contributing threads
        would be permanently excluded from every subsequent incremental
        summary (the ``c.created_at > @since`` watermark moves past them).
        Cross-partition is unavoidable for a per-user roll-up.

        Returns the user summary document dict.
        """
        if not user_id:
            raise ValidationError("user_id is required")

        logger.info(
            "generate_user_summary started user_id=%s observed_thread_ids=%s",
            user_id,
            len(thread_ids) if thread_ids else 0,
        )

        # ---- 1. Check for existing user summary ----
        user_summary_id = f"user_summary_{user_id}"
        existing_summary: Optional[dict[str, Any]] = None
        try:
            existing_summary = self._container.read_item(
                item=user_summary_id,
                partition_key=[user_id, "__user_summary__"],
            )
        except Exception:
            pass  # first time — full generation

        # ---- 2. Query memories (time-filtered if updating) ----
        query = "SELECT * FROM c WHERE c.user_id = @user_id AND c.type != 'user_summary'"
        parameters: list[dict[str, Any]] = [
            {"name": "@user_id", "value": user_id},
        ]

        if existing_summary:
            since = existing_summary["updated_at"]
            query += " AND c.created_at > @since"
            parameters.append({"name": "@since", "value": since})

        query_started_at = datetime.now(timezone.utc).isoformat()

        items = list(
            self._container.query_items(
                query=query,
                parameters=parameters,
                enable_cross_partition_query=True,
            )
        )

        if existing_summary and not items:
            logger.info("generate_user_summary no new memories, returning existing")
            return existing_summary

        if not existing_summary and not items:
            raise ValidationError(f"No memories found for user_id={user_id!r}")

        # ---- 3. Sort and apply per-thread recent_k trimming ----
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
            items.reverse()  # chronological order

        # ---- 4. Build transcript grouped by thread ----
        transcript = self._build_transcript(items, group_by_thread=True)
        new_thread_ids = {m.get("thread_id", "") for m in items}

        # ---- 5. Call LLM ----
        if existing_summary:
            prior_json = existing_summary.get("metadata", {}).get("structured_summary")
            if prior_json:
                prior_text = json.dumps(prior_json, indent=2)
            else:
                prior_text = existing_summary.get("content", "")
            response_text = self._run_prompty(
                "user_summary_update.prompty",
                inputs={"prior_summary": prior_text, "transcript": transcript},
            )
        else:
            response_text = self._run_prompty(
                "user_summary.prompty",
                inputs={"transcript": transcript},
            )

        # ---- 6. Parse response ----
        parsed = self._parse_llm_json(response_text)
        # For user summaries, build a narrative overview from key_facts
        key_facts = parsed.get("key_facts", [])
        overview = "; ".join(key_facts) if key_facts else response_text

        # ---- 7. Generate embedding ----
        summary_embedding = self._embeddings.generate(overview)

        # ---- 8. Accumulate metadata and upsert ----
        if existing_summary:
            old_thread_ids = set(existing_summary.get("metadata", {}).get("thread_ids", []))
            all_thread_ids = sorted(old_thread_ids | new_thread_ids)
            old_memory_count = existing_summary.get("metadata", {}).get("source_memory_count", 0)
            total_memory_count = old_memory_count + len(items)
        else:
            all_thread_ids = sorted(new_thread_ids)
            total_memory_count = len(items)

        summary_doc: dict[str, Any] = {
            "id": user_summary_id,
            "user_id": user_id,
            "thread_id": "__user_summary__",
            "role": "system",
            "type": "user_summary",
            "content": overview,
            "embedding": summary_embedding,
            "salience": 1.0,
            "tags": ["sys:user-summary"],
            "metadata": {
                "structured_summary": parsed,
                "source_thread_count": len(all_thread_ids),
                "source_memory_count": total_memory_count,
                "thread_ids": all_thread_ids,
                "recent_k": recent_k,
                "incremental_update": existing_summary is not None,
            },
            "created_at": existing_summary["created_at"] if existing_summary else query_started_at,
            "updated_at": query_started_at,
        }

        self._upsert_memory(summary_doc)
        logger.info(
            "generate_user_summary completed thread_count=%d memory_count=%d",
            len(all_thread_ids),
            total_memory_count,
        )
        return summary_doc

    def deduplicate_facts(
        self,
        user_id: str,
        similarity_threshold: float = 0.9,
        max_facts: int = 200,
    ) -> dict[str, int]:
        """Deduplicate active facts for a user using cosine similarity + LLM.

        Returns counts: ``{"kept": N, "merged": N, "superseded": N}``.
        """
        if not user_id:
            raise ValidationError("user_id is required")

        logger.info("deduplicate_facts started user_id=%s threshold=%.2f", user_id, similarity_threshold)

        # ---- 1. Load all active facts ----
        # ORDER BY c._ts DESC makes the TOP cap deterministic - without it
        # Cosmos returns rows in implementation-defined order across physical
        # partitions, so two near-duplicates on opposite sides of the cap
        # would never get a chance to merge. Newest-first means recently
        # extracted facts are always considered.
        query = (
            f"SELECT TOP {max_facts} * FROM c "
            "WHERE c.user_id = @user_id "
            "AND c.type = 'fact' "
            "AND (NOT IS_DEFINED(c.superseded_by) OR c.superseded_by = null) "
            "ORDER BY c._ts DESC"
        )
        parameters: list[dict[str, Any]] = [
            {"name": "@user_id", "value": user_id},
        ]
        facts = list(
            self._container.query_items(
                query=query,
                parameters=parameters,
                enable_cross_partition_query=True,
            )
        )

        if len(facts) <= 1:
            logger.info("deduplicate_facts: %d facts, nothing to deduplicate", len(facts))
            return {"kept": len(facts), "merged": 0, "superseded": 0}

        # ---- 2. Compute pairwise cosine similarity ----
        embeddings = [f.get("embedding") for f in facts]
        # Filter out facts without embeddings
        valid = [(i, f, e) for i, (f, e) in enumerate(zip(facts, embeddings)) if e]
        if len(valid) <= 1:
            return {"kept": len(facts), "merged": 0, "superseded": 0}

        clusters = self._cluster_by_similarity(valid, similarity_threshold)

        # ---- 3. For clusters with >1 fact, call LLM for dedup decisions ----
        kept = 0
        merged = 0
        superseded = 0
        now = datetime.now(timezone.utc).isoformat()

        for cluster in clusters:
            if len(cluster) == 1:
                kept += 1
                continue

            # Build dedup prompt input
            cluster_facts = [facts[idx] for idx in cluster]
            lines = []
            for i, cf in enumerate(cluster_facts, 1):
                confidence = cf.get("confidence", "N/A")
                lines.append(
                    f'{i}. ID: {cf["id"]} | Content: "{cf.get("content", "")}" | '
                    f"Confidence: {confidence} | "
                    f"Salience: {cf.get('salience', 'N/A')} | "
                    f"Created: {cf.get('created_at', 'N/A')}"
                )
            cluster_text = "\n".join(lines)

            response_text = self._run_prompty(
                "dedup.prompty",
                inputs={"cluster_text": cluster_text},
            )

            parsed = self._parse_llm_json(response_text)
            actions = parsed.get("actions", [])

            for act in actions:
                action_type = act.get("action", "").upper()

                if action_type == "KEEP":
                    kept += 1

                elif action_type == "MERGE":
                    source_ids = act.get("source_ids", [])
                    merged_text = act.get("merged_text", "")
                    if not merged_text or not source_ids:
                        continue

                    content_hash = compute_content_hash(merged_text)
                    det_id = f"fact_{hashlib.sha256(f'{user_id}:merge:{content_hash}'.encode()).hexdigest()[:16]}"

                    # Use the thread_id from the first source
                    source_fact = next(
                        (f for f in cluster_facts if f["id"] in source_ids),
                        cluster_facts[0],
                    )

                    merged_tags: list[str] = []
                    seen_tags: set[str] = set()
                    for f in cluster_facts:
                        if f["id"] not in source_ids:
                            continue
                        for t in f.get("tags", []):
                            if t not in seen_tags:
                                seen_tags.add(t)
                                merged_tags.append(t)
                    if not merged_tags:
                        merged_tags = ["sys:fact"]

                    source_confidences = [
                        c for f in cluster_facts if f["id"] in source_ids and (c := f.get("confidence")) is not None
                    ]
                    merged_confidence = max(source_confidences) if source_confidences else None

                    source_saliences = [
                        s for f in cluster_facts if f["id"] in source_ids and (s := f.get("salience")) is not None
                    ]
                    merged_salience = act.get("salience")
                    if merged_salience is None and source_saliences:
                        merged_salience = max(source_saliences)

                    merged_doc: dict[str, Any] = {
                        "id": det_id,
                        "user_id": user_id,
                        "thread_id": source_fact.get("thread_id", ""),
                        "role": "system",
                        "type": "fact",
                        "content": merged_text,
                        "content_hash": content_hash,
                        "metadata": {
                            "merged_from": source_ids,
                            "merged_from_count": len(source_ids),
                        },
                        "salience": merged_salience,
                        "supersedes_ids": source_ids,
                        "tags": merged_tags,
                        "created_at": now,
                    }
                    if merged_confidence is not None:
                        merged_doc["confidence"] = merged_confidence

                    # Generate embedding for merged text
                    merged_doc["embedding"] = self._embeddings.generate(merged_text)
                    self._upsert_memory(merged_doc)

                    # Mark source facts as superseded
                    for sid in source_ids:
                        src = next((f for f in cluster_facts if f["id"] == sid), None)
                        if src and self._mark_superseded(src, det_id):
                            superseded += 1

                    merged += 1

                elif action_type == "SUPERSEDE":
                    old_id = act.get("old_id")
                    new_id = act.get("new_id")
                    if not old_id or not new_id:
                        continue

                    old_fact = next((f for f in cluster_facts if f["id"] == old_id), None)
                    if old_fact and self._mark_superseded(old_fact, new_id):
                        superseded += 1

                    kept += 1  # the new_id is effectively kept

        result = {"kept": kept, "merged": merged, "superseded": superseded}
        logger.info("deduplicate_facts completed: %s", result)
        return result

    @staticmethod
    def _cluster_by_similarity(
        valid: list[tuple[int, dict[str, Any], list[float]]],
        threshold: float,
    ) -> list[list[int]]:
        """Cluster facts by pairwise cosine similarity above *threshold*.

        Uses a simple union-find approach. Returns lists of original indices.
        """
        n = len(valid)

        def _cosine_sim(a: list[float], b: list[float]) -> float:
            dot = sum(x * y for x, y in zip(a, b))
            mag_a = math.sqrt(sum(x * x for x in a))
            mag_b = math.sqrt(sum(x * x for x in b))
            if mag_a == 0 or mag_b == 0:
                return 0.0
            return dot / (mag_a * mag_b)

        # Union-find
        parent = list(range(n))

        def find(x: int) -> int:
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(x: int, y: int) -> None:
            rx, ry = find(x), find(y)
            if rx != ry:
                parent[rx] = ry

        # Compute pairwise similarity
        for i in range(n):
            for j in range(i + 1, n):
                sim = _cosine_sim(valid[i][2], valid[j][2])
                if sim >= threshold:
                    union(i, j)

        # Group by root
        groups: dict[int, list[int]] = defaultdict(list)
        for i in range(n):
            groups[find(i)].append(valid[i][0])  # original index

        return list(groups.values())
