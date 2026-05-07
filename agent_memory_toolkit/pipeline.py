"""Processing pipeline for memory extraction, summarization, and dedup.

Shared by both the SDK (in-process calls) and Azure Functions (change feed trigger).
Uses ChatClient for chat completions and EmbeddingsClient for embeddings.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
from collections import defaultdict
from datetime import datetime, timezone
from typing import Any, Literal, Optional

from ._utils import DEFAULT_TTL_BY_TYPE, compute_content_hash
from .exceptions import LLMError, ValidationError

logger = logging.getLogger(__name__)

# Separator for deterministic id seeds. Using NUL ensures user_id /
# thread_id values can never collide with literal section markers
# (e.g. a thread literally named ``"merged"`` cannot collide with the
# reconcile-merge id namespace). Defined as a module constant because
# escape sequences are not permitted inside f-strings on Python 3.11.
_ID_SEED_SEP = "\x00"


def _is_real_number(v: Any) -> bool:
    """True for ``int``/``float`` excluding ``bool`` (``isinstance(True, int)`` is True)."""
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def _max_or_none(values: Any) -> Optional[float]:
    """Return max of numeric values, ignoring None / non-numeric / bool. None if empty."""
    nums = [float(v) for v in values if _is_real_number(v)]
    return max(nums) if nums else None


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
            f"AND (NOT IS_DEFINED(c.superseded_by) OR IS_NULL(c.superseded_by)) "
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
        response = self._container.upsert_item(body=doc)
        if isinstance(response, dict):
            return response
        return doc

    def _mark_superseded(
        self,
        old_doc: dict[str, Any],
        superseder_id: str,
        *,
        reason: Literal["duplicate", "contradiction", "update"],
    ) -> bool:
        """Atomically set ``superseded_by`` on ``old_doc`` using ETag protection.

        Also stamps ``supersede_reason`` and ``superseded_at`` so apps can
        distinguish a duplicate-collapse from a contradiction-resolution
        from an extract-time refinement (``update``) at audit time.

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
            return {
                "facts_count": 0,
                "procedural_count": 0,
                "episodic_count": 0,
                "unclassified_count": 0,
                "updated_count": 0,
                "exact_dedup_skipped": 0,
            }

        # ---- 2. Load existing memories for reconciliation ----
        existing = self._load_existing_memories(user_id, ["fact", "procedural"])
        # Pre-compute exact-content hashes from existing memories for the
        # write-time short-circuit. Saves the embedding call and the upsert
        # RU on identical re-extractions across runs.
        #
        # Hashes are bucketed *by type* — a fact with the same normalized
        # text as an existing procedural must NOT be silently dropped, because
        # they're semantically different memory kinds and the LLM's
        # classification would be erased. Unclassified items are persisted as
        # facts so they share the fact bucket.
        existing_fact_hashes: set[str] = {
            m["content_hash"] for m in existing if m.get("type") == "fact" and m.get("content_hash")
        }
        existing_proc_hashes: set[str] = {
            m["content_hash"] for m in existing if m.get("type") == "procedural" and m.get("content_hash")
        }
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
                # If the new content hashes to the same deterministic id
                # as the target (UPDATE refines metadata but text is
                # unchanged), the upsert below would overwrite the
                # ``superseded_by``/``supersede_reason`` audit metadata
                # we are about to stamp on the target — and the new doc
                # would carry a self-referential ``supersedes_ids``
                # entry. Treat as a no-op and let the existing record
                # stand.
                if det_id == fact["supersedes_id"]:
                    logger.debug(
                        "extract_memories: skipping UPDATE — det_id == supersedes_id (%s)",
                        det_id,
                    )
                    continue
                doc["supersedes_ids"] = [fact["supersedes_id"]]
                # Mark old memory as superseded
                try:
                    old_mem = self._container.read_item(
                        item=fact["supersedes_id"],
                        partition_key=[user_id, thread_id],
                    )
                    # Skip if the target was already retired by an earlier
                    # extract or a reconcile cycle - re-superseding it
                    # would clobber the prior ``superseded_by`` link and
                    # narrow the audit chain.
                    if old_mem.get("superseded_by"):
                        logger.debug(
                            "extract_memories: skipping UPDATE — target %s already superseded by %s",
                            fact["supersedes_id"],
                            old_mem.get("superseded_by"),
                        )
                    elif self._mark_superseded(old_mem, det_id, reason="update"):
                        updated_count += 1
                except Exception:
                    # Try cross-partition query if direct read fails
                    logger.debug(
                        "Could not read superseded item %s directly, trying cross-partition",
                        fact["supersedes_id"],
                    )
                    try:
                        q = (
                            "SELECT * FROM c WHERE c.id = @id AND c.user_id = @uid "
                            "AND (NOT IS_DEFINED(c.superseded_by) OR IS_NULL(c.superseded_by))"
                        )
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
                        if results and self._mark_superseded(results[0], det_id, reason="update"):
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
            existing_fact_hashes.add(new_content_hash)

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
            # Same write-time exact-dedup short-circuit as facts. UPDATEs
            # bypass — they explicitly target an old record by id.
            if action == "ADD" and content_hash in existing_proc_hashes:
                logger.debug(
                    "extract_memories: skipping exact-dup procedural hash=%s user_id=%s",
                    content_hash,
                    user_id,
                )
                exact_dedup_skipped += 1
                continue
            seed = _ID_SEED_SEP.join((user_id, content_hash))
            det_id = f"proc_{hashlib.sha256(seed.encode()).hexdigest()[:32]}"

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
                # Same self-overwrite guard as the fact UPDATE branch:
                # if the new content hashes to the same det_id as the
                # target, the upsert below would erase the audit
                # metadata we are about to stamp on it.
                if det_id == proc["supersedes_id"]:
                    logger.debug(
                        "extract_memories: skipping procedural UPDATE — det_id == supersedes_id (%s)",
                        det_id,
                    )
                    continue
                doc["supersedes_ids"] = [proc["supersedes_id"]]
                try:
                    q = (
                        "SELECT * FROM c WHERE c.id = @id AND c.user_id = @uid "
                        "AND (NOT IS_DEFINED(c.superseded_by) OR IS_NULL(c.superseded_by))"
                    )
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
                    if results and self._mark_superseded(results[0], det_id, reason="update"):
                        updated_count += 1
                except Exception as exc:
                    logger.warning(
                        "Failed to mark superseded procedural memory %s: %s",
                        proc["supersedes_id"],
                        exc,
                    )

            docs_to_embed.append(doc)
            embed_texts.append(text)
            # Record this procedural's hash so a later candidate in the same
            # batch with identical content also short-circuits.
            existing_proc_hashes.add(content_hash)

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
            seed = _ID_SEED_SEP.join((user_id, thread_id, content_hash))
            det_id = f"ep_{hashlib.sha256(seed.encode()).hexdigest()[:32]}"

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
            # Unclassified items are persisted as facts → share the fact
            # bucket for the write-time exact-dedup short-circuit.
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
            det_id = f"unc_{hashlib.sha256(seed.encode()).hexdigest()[:32]}"

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
            # Unclassified items are persisted as facts → record in the
            # fact bucket so later batch candidates short-circuit.
            existing_fact_hashes.add(content_hash)

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
        if n > 500:
            raise ValidationError(f"n must be <= 500 to bound prompt size and LLM cost, got {n}")

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
            "AND (NOT IS_DEFINED(c.superseded_by) OR IS_NULL(c.superseded_by)) "
            "ORDER BY c.created_at DESC"
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
                merged_record = MemoryRecord(
                    id=merged_id,
                    user_id=user_id,
                    role="system",
                    memory_type=MemoryType.fact,
                    content=merged_content,
                    thread_id=recent_thread_id or f"__reconciled__:{user_id}",
                    confidence=confidence_val,
                    salience=salience_val,
                    supersedes_ids=merged_supersedes,
                    source_memory_ids=merged_source_memory_ids,
                    tags=merged_tags,
                    content_hash=merged_content_hash,
                    metadata={
                        "merged_via": "reconcile",
                        "merged_from_count": len(valid_source_ids),
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
                merged_record.embedding = self._embeddings.generate(merged_content)
            except Exception:
                logger.exception(
                    "reconcile_memories: embedding failed for merged id=%s; "
                    "aborting duplicate group to avoid search-index hole",
                    merged_record.id,
                )
                continue

            merged_doc = merged_record.to_cosmos_dict()
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

            if self._mark_superseded(loser_doc, resolved_winner, reason="contradiction"):
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
        return result
