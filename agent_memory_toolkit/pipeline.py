"""Processing pipeline for memory extraction, summarization, and dedup.

Shared by both the SDK (in-process calls) and Azure Functions (change feed trigger).
Uses ChatClient for chat completions and EmbeddingsClient for embeddings.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
from collections import defaultdict
from datetime import datetime, timezone
from typing import Any, Literal, Optional

from ._utils import DEFAULT_TTL_BY_TYPE, compute_content_hash
from .exceptions import LLMError, ValidationError

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Reconciliation hashing helpers
# ---------------------------------------------------------------------------

_WHITESPACE_RE = re.compile(r"\s+")


def _normalize_for_hash(text: str) -> str:
    """Lowercase + collapse whitespace for write-time exact-dedup.

    Deliberately conservative: lowercase, strip, and collapse internal runs
    of whitespace to a single space. Punctuation and word order still matter.
    The point is to catch *identical* re-extractions cheaply — paraphrases
    are handled by the reconciliation LLM pass.
    """
    return _WHITESPACE_RE.sub(" ", text.strip().lower())


def _content_hash(text: str) -> str:
    """SHA-256 of the normalized text, truncated to 32 hex chars.

    32 chars (128 bits) is plenty for collision avoidance within a single
    user's fact set and keeps the field compact in Cosmos documents.
    """
    return hashlib.sha256(_normalize_for_hash(text).encode("utf-8")).hexdigest()[:32]


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

    def _mark_superseded(
        self,
        old_doc: dict[str, Any],
        superseder_id: str,
        *,
        reason: Literal["duplicate", "contradiction"],
    ) -> bool:
        """Atomically set ``superseded_by`` on ``old_doc`` using ETag protection.

        Also stamps ``supersede_reason`` and ``superseded_at`` so apps can
        distinguish a duplicate-collapse from a contradiction-resolution at
        audit time.

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
        # Pre-compute exact-content hashes from existing memories for the
        # write-time short-circuit. Saves the embedding call and the upsert
        # RU on identical re-extractions across runs.
        existing_hashes: set[str] = {f["content_hash"] for f in existing if f.get("content_hash")}
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
        exact_dedup_skipped = 0

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
            # Write-time exact-dedup short-circuit. ADDs whose normalized
            # content already exists are skipped before embedding/upsert.
            # UPDATEs go through unchanged - they explicitly target an old
            # record by id and need to write the supersession link.
            new_content_hash = _content_hash(text)
            if action == "ADD" and new_content_hash in existing_hashes:
                logger.debug(
                    "extract_memories: skipping exact-dup fact hash=%s user_id=%s thread_id=%s",
                    new_content_hash,
                    user_id,
                    thread_id,
                )
                exact_dedup_skipped += 1
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
                "content_hash": new_content_hash,
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
                    if self._mark_superseded(old_mem, det_id, reason="duplicate"):
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
                        if results and self._mark_superseded(results[0], det_id, reason="duplicate"):
                            updated_count += 1
                    except Exception as exc:
                        logger.warning(
                            "Failed to mark superseded memory %s: %s",
                            fact["supersedes_id"],
                            exc,
                        )

            docs_to_embed.append(doc)
            embed_texts.append(text)
            # Record this fact's hash so a later candidate in the same batch
            # with identical content also short-circuits.
            existing_hashes.add(new_content_hash)

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
                    if results and self._mark_superseded(results[0], det_id, reason="duplicate"):
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
            "exact_dedup_skipped": exact_dedup_skipped,
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

    def reconcile_memories(self, user_id: str, n: int = 50) -> dict[str, int]:
        """Reconcile a user's active facts in a single LLM pass.

        Loads the most recent ``n`` active (non-superseded) facts for
        ``user_id``, asks the dedup prompt to classify them into
        ``duplicate_groups``, ``contradicted_pairs``, and ``kept_ids``, then
        applies both kinds of resolutions:

        * **Duplicates** — a fresh merged fact is upserted; every source is
          soft-deleted with ``supersede_reason="duplicate"``.
        * **Contradictions** — the loser is soft-deleted with
          ``supersede_reason="contradiction"`` and ``superseded_by`` set to
          the winner. Dangling references are resolved transparently when a
          contradicted id was just absorbed into a duplicate group.

        Returns ``{"kept": int, "merged": int, "contradicted": int}`` where
        ``merged`` and ``contradicted`` count the *losers* that were
        soft-deleted (duplicates and contradictions respectively).
        """
        from .models import MemoryRecord, MemoryType

        if not user_id:
            raise ValidationError("user_id is required")
        if not isinstance(n, int) or isinstance(n, bool) or n < 1:
            raise ValidationError(f"n must be a positive integer, got {n!r}")

        logger.info("reconcile_memories started user_id=%s n=%d", user_id, n)

        # ---- 1. Load up to N most recent active facts ----
        # ORDER BY c._ts DESC keeps the TOP cap deterministic across
        # physical partitions and surfaces the freshest facts to the
        # LLM (recency is a tiebreaker the prompt relies on).
        query = (
            f"SELECT TOP {n} * FROM c "
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
            logger.info(
                "reconcile_memories: %d facts, nothing to reconcile",
                len(facts),
            )
            return {"kept": len(facts), "merged": 0, "contradicted": 0}

        # ---- 2. Format the facts pool for the prompt ----
        lines: list[str] = []
        for i, cf in enumerate(facts, 1):
            lines.append(
                f'{i}. ID: {cf["id"]} | Content: "{cf.get("content", "")}" | '
                f"Confidence: {cf.get('confidence', 'N/A')} | "
                f"Salience: {cf.get('salience', 'N/A')} | "
                f"Created: {cf.get('created_at', 'N/A')}"
            )
        facts_text = "\n".join(lines)

        # ---- 3. Single LLM call over the entire pool ----
        response_text = self._run_prompty(
            "dedup.prompty",
            inputs={"facts_text": facts_text},
        )
        parsed = self._parse_llm_json(response_text)

        duplicate_groups = parsed.get("duplicate_groups", []) or []
        contradicted_pairs = parsed.get("contradicted_pairs", []) or []
        # ``kept_ids`` is informational only; the actual ``kept`` count is
        # derived from len(facts) - merged - contradicted to match what the
        # pipeline really did, not what the LLM said.
        _ = parsed.get("kept_ids", []) or []

        facts_by_id: dict[str, dict[str, Any]] = {f["id"]: f for f in facts}

        merged = 0
        contradicted = 0
        # Tracks source_id -> merged_id rewrites so contradictions whose
        # winner/loser landed in a duplicate group can be redirected to
        # the surviving merged document.
        source_to_merged_id: dict[str, str] = {}

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

            # Facts come back newest-first (ORDER BY c._ts DESC), so the
            # first source doc is the most recent — pick its thread_id.
            recent_thread_id = source_docs[0].get("thread_id", "")

            confidence_val = group.get("confidence")
            salience_val = group.get("salience")

            try:
                merged_record = MemoryRecord(
                    user_id=user_id,
                    role="system",
                    memory_type=MemoryType.fact,
                    content=merged_content,
                    thread_id=recent_thread_id or "__reconciled__",
                    confidence=confidence_val,
                    salience=salience_val,
                    supersedes_ids=list(source_ids),
                    source_memory_ids=merged_source_memory_ids,
                    tags=merged_tags,
                    content_hash=_content_hash(merged_content),
                )
            except Exception:
                logger.exception(
                    "reconcile_memories: failed to build merged record for group %r",
                    group,
                )
                continue

            # Generate embedding for the merged content so retrieval can
            # rank it against future queries from the moment it lands.
            try:
                merged_record.embedding = self._embeddings.generate(merged_content)
            except Exception:
                logger.exception(
                    "reconcile_memories: embedding failed for merged id=%s",
                    merged_record.id,
                )

            self._upsert_memory(merged_record.to_cosmos_dict())

            for sid in source_ids:
                src_doc = facts_by_id.get(sid)
                if src_doc is None:
                    logger.debug(
                        "reconcile_memories: hallucinated source_id=%s (not in pool)",
                        sid,
                    )
                    continue
                if self._mark_superseded(src_doc, merged_record.id, reason="duplicate"):
                    merged += 1
                source_to_merged_id[sid] = merged_record.id

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
                # The original loser was just merged. Fetch the merged doc
                # from Cosmos so we can attach the contradiction reason.
                try:
                    q = "SELECT * FROM c WHERE c.id = @id AND c.user_id = @uid"
                    results = list(
                        self._container.query_items(
                            query=q,
                            parameters=[
                                {"name": "@id", "value": resolved_loser_id},
                                {"name": "@uid", "value": user_id},
                            ],
                            enable_cross_partition_query=True,
                        )
                    )
                    if results:
                        loser_doc = results[0]
                except Exception:
                    logger.exception(
                        "reconcile_memories: failed to fetch redirected loser id=%s",
                        resolved_loser_id,
                    )

            if loser_doc is None:
                logger.warning(
                    "reconcile_memories: loser doc not found for pair %r (resolved_loser=%s)",
                    pair,
                    resolved_loser_id,
                )
                continue

            if self._mark_superseded(loser_doc, resolved_winner, reason="contradiction"):
                contradicted += 1

        kept = max(0, len(facts) - merged - contradicted)
        result = {"kept": kept, "merged": merged, "contradicted": contradicted}
        logger.info("reconcile_memories completed: %s", result)
        return result
