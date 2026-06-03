"""Pure helpers shared between sync and async pipeline services.

Anything that's a function of its inputs only (no LLM/Cosmos/embedding IO)
lives here so :class:`PipelineService` and :class:`AsyncPipelineService` can
share it without duplication. The :class:`PromptyLoader` cache-and-prepare
helper is here too because it is pure (filesystem read + dict shaping); the
LLM call itself stays on the service.
"""

from __future__ import annotations

import json
import os
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Optional

from azure.cosmos.agent_memory.exceptions import LLMError

# Separator for deterministic id seeds. Using NUL ensures user_id /
# thread_id values can never collide with literal section markers
# (e.g. a thread literally named ``"merged"`` cannot collide with the
# reconcile-merge id namespace). Defined as a module constant because
# escape sequences are not permitted inside f-strings on Python 3.11.
ID_SEED_SEP = "\x00"

# Mapping from prompty 2.x ModelOptions field names (camelCase) to the
# snake_case kwargs accepted by OpenAI's chat completions API. We include
# snake_case variants too because some prompty releases serialize options
# already lowercased.
PROMPTY_OPTION_ALIASES = {
    "topP": "top_p",
    "top_p": "top_p",
    "topK": "top_k",
    "top_k": "top_k",
    "frequencyPenalty": "frequency_penalty",
    "frequency_penalty": "frequency_penalty",
    "presencePenalty": "presence_penalty",
    "presence_penalty": "presence_penalty",
    "maxOutputTokens": "max_completion_tokens",
    "max_output_tokens": "max_completion_tokens",
    "maxTokens": "max_completion_tokens",
    "max_tokens": "max_completion_tokens",
    "stopSequences": "stop",
    "stop_sequences": "stop",
    "allowMultipleToolCalls": "parallel_tool_calls",
    "allow_multiple_tool_calls": "parallel_tool_calls",
}

_FRONT_MATTER_VERSION = re.compile(r"^version:\s*(\S+)\s*$", re.MULTILINE)
DEFAULT_PROMPT_VERSION = "v1"
_TOPIC_TAG_UNSAFE = re.compile(r"[^a-z0-9_:./-]+")


def build_topic_tags(values: Any) -> list[str]:
    tags: set[str] = set()
    for value in values or []:
        raw = str(value).strip().lower()
        if raw.startswith("topic:"):
            raw = raw[len("topic:") :]
        topic = _TOPIC_TAG_UNSAFE.sub("-", raw).strip("-")
        if topic:
            tags.add(f"topic:{topic}")
    return sorted(tags)


def is_real_number(v: Any) -> bool:
    """True for ``int``/``float`` excluding ``bool`` (``isinstance(True, int)`` is True)."""
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def max_or_none(values: Any) -> Optional[float]:
    """Return max of numeric values, ignoring None / non-numeric / bool. None if empty."""
    nums = [float(v) for v in values if is_real_number(v)]
    return max(nums) if nums else None


def chat_text(response: Any) -> str:
    """Extract assistant text from the chat client response.

    Sync/async ``ChatClient.generate`` returns a plain string. The remaining
    branches handle legacy dict/object shapes still emitted by mocks in
    the unit tests.
    """
    if response is None:
        raise LLMError("LLM returned no response")
    if isinstance(response, str):
        return response
    if isinstance(response, dict):
        content = response.get("content") or response.get("text")
        if isinstance(content, str):
            return content
        message = response.get("message")
        if isinstance(message, dict) and isinstance(message.get("content"), str):
            return message["content"]
        choices = response.get("choices")
        if isinstance(choices, list) and choices:
            first = choices[0]
            if isinstance(first, dict):
                first_message = first.get("message")
                if isinstance(first_message, dict) and isinstance(first_message.get("content"), str):
                    return first_message["content"]
                if isinstance(first.get("text"), str):
                    return first["text"]
    content_attr = getattr(response, "content", None)
    if isinstance(content_attr, str):
        return content_attr
    raise LLMError(f"LLM response did not contain text content: {type(response).__name__}")


def messages_to_dicts(messages: Any) -> list[dict[str, str]]:
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


def extract_prompty_params(p: Any) -> dict[str, Any]:
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
        params[PROMPTY_OPTION_ALIASES.get(key, key)] = value
    return params


def _normalize_metadata_keys(
    value: Optional[Iterable[str]],
) -> Optional[tuple[str, ...]]:
    """Validate + coerce a ``transcript_metadata_keys`` argument to a tuple.

    Rejects ``str`` outright (a bare string is iterable char-by-char, which
    would silently produce a one-letter allow-list). Returns ``None`` for
    empty or missing input.
    """
    if value is None:
        return None
    if isinstance(value, str):
        raise TypeError(
            "transcript_metadata_keys must be a sequence of keys "
            "(list/tuple/set), not a single str. "
            f"Got: {value!r}. Did you mean [{value!r}]?"
        )
    keys = tuple(str(k) for k in value if str(k))
    return keys or None


def _format_metadata_segment(
    metadata: Any,
    metadata_keys: Optional[tuple[str, ...]],
) -> str:
    """Render the trailing ``[metadata: {...}]`` segment for a transcript line.

    Returns an empty string unless ``metadata_keys`` is a non-empty tuple
    AND at least one of those keys is present in ``metadata``. Only the
    explicitly allow-listed keys are serialized, in the iteration order of
    ``metadata_keys``.
    """
    if not metadata_keys or not isinstance(metadata, dict):
        return ""
    filtered = {k: metadata[k] for k in metadata_keys if k in metadata}
    if not filtered:
        return ""
    payload = json.dumps(filtered, separators=(",", ":"), ensure_ascii=False, default=str)
    return f" [metadata: {payload}]"


def build_transcript(
    items: list[dict[str, Any]],
    *,
    group_by_thread: bool = False,
    metadata_keys: Optional[Iterable[str]] = None,
) -> str:
    """Build a formatted transcript from memory documents.

    Parameters
    ----------
    items:
        Memory dicts with ``role``, ``content``, and optional ``metadata``.
    group_by_thread:
        If *True*, group messages under ``=== Thread <id> ===`` headers.
    metadata_keys:
        Allow-list of metadata keys to surface in each transcript line.
        Defaults to ``None`` (no metadata serialized — only ``[role]:
        content`` lines). When provided, only the listed keys are emitted,
        in iteration order. Keys absent from a given turn's metadata are
        silently skipped.

        Set this when callers stash semantically useful breadcrumbs in
        ``TurnRecord.metadata`` that the extraction LLM should see
        (e.g. ``["agent_id", "timestamp"]``). Leaving it unset keeps free-form
        metadata blobs (raw tool calls, IDE schema, etc.) out of every
        prompt — they're often 10-100x larger than the dialog itself and
        dilute extraction quality.

        Accepts any iterable of strings except ``str`` itself (which would
        be interpreted char-by-char). Generators are coerced to a tuple so
        the allow-list is reusable across turns.
    """
    keys = _normalize_metadata_keys(metadata_keys)
    if not group_by_thread:
        lines: list[str] = []
        for m in items:
            role = m.get("role", "unknown")
            content = m.get("content", "")
            meta_str = _format_metadata_segment(m.get("metadata", {}), keys)
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
            meta_str = _format_metadata_segment(m.get("metadata", {}), keys)
            parts.append(f"[{role}]: {content}{meta_str}")
        parts.append("")
    return "\n".join(parts)


def format_existing_episodics(memories: list[dict[str, Any]]) -> str:
    """Render existing episodic memories for the extract_memories prompt.

    Groups by ``(scope_type, scope_value)`` so the LLM can see, per-scope,
    which intent is already captured. Episodics use **scope-as-identity**:
    the deterministic id is seeded from ``(user_id, scope_type, scope_value)``,
    so any re-emission for the same scope (paraphrased intent, added detail,
    or a reversal) collides and overwrites the prior record via upsert. The
    LLM does NOT make ``ADD``/``UPDATE``/``CONTRADICT`` decisions on
    episodics — that vocabulary is not in the episodic schema.

    What this rendering gives the model is per-scope context so it can:

    1. Emit a single coherent ``text`` that reflects the *current* intent
       for the scope (the upsert will overwrite the prior one).
    2. Avoid re-emitting an episodic when the new turn carries no
       additional signal beyond what the existing one already records.

    Distinct events under the same umbrella (e.g. hotel booking vs lost
    wallet, both under a Tokyo trip) belong under distinct ``scope_value``
    strings so they don't collide on the deterministic id.
    """
    if not memories:
        return "(none)"
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for mem in memories:
        meta = mem.get("metadata") or {}
        scope_type = (meta.get("scope_type") or "(none)").strip() or "(none)"
        scope_value = (meta.get("scope_value") or "(none)").strip() or "(none)"
        grouped[(scope_type, scope_value)].append(mem)
    lines: list[str] = []
    for (scope_type, scope_value), bucket in grouped.items():
        lines.append(f"- {scope_type} = {scope_value} ({len(bucket)} episodic{'s' if len(bucket) != 1 else ''})")
        for mem in bucket:
            mem_id = mem.get("id", "(no-id)")
            salience = mem.get("salience", "N/A")
            content = (mem.get("content") or "").strip() or "(empty content)"
            lines.append(f"  - [ID: {mem_id}] (salience {salience}) {content}")
    return "\n".join(lines)


# Stopwords stripped from grounding checks. Keep this list short and focused
# on tokens that carry no factual content; any word a memory might legitimately
# differ on (e.g. "not", "no") must NOT be added here.
_GROUNDING_STOPWORDS = frozenset(
    {
        "the",
        "a",
        "an",
        "is",
        "are",
        "was",
        "were",
        "be",
        "been",
        "being",
        "and",
        "or",
        "but",
        "to",
        "of",
        "for",
        "on",
        "in",
        "at",
        "by",
        "with",
        "from",
        "as",
        "that",
        "this",
        "these",
        "those",
        "it",
        "its",
        "user",
        "they",
        "them",
        "their",
        "he",
        "she",
        "his",
        "her",
        "him",
        "has",
        "have",
        "had",
        "do",
        "does",
        "did",
        "will",
        "would",
        "should",
        "can",
        "could",
        "may",
        "might",
        "must",
    }
)

_GROUNDING_TOKEN_RE = re.compile(r"[a-zA-Z]{3,}")


def _grounding_tokens(text: str) -> set[str]:
    """Tokenize text into lowercased content words (>=3 chars, stopwords removed)."""
    if not text:
        return set()
    return {t for t in _GROUNDING_TOKEN_RE.findall(text.lower()) if t not in _GROUNDING_STOPWORDS}


def check_extracted_fact_grounding(
    fact_docs: list[dict[str, Any]],
    turn_items: list[dict[str, Any]],
    existing_facts: list[dict[str, Any]],
    *,
    user_id: str,
    thread_id: str,
    logger: Any,
) -> None:
    """Warn when an extracted fact's content is not grounded in the new user turns.

    Catches two known LLM failure modes that previously corrupted the fact store:

    1. **Synthesis from existing facts** — the LLM emits an ADD whose content
       paraphrase-merges two or more existing facts (e.g. existing
       "user eats meat" + "user loves steak" → emitted "user loves steak,
       indicating they eat meat") even though the new user turn says nothing
       on the topic. Reconciliation later catches the resulting duplicates
       but the visible artefact is a chain of "duplicate" supersedes that the
       user never triggered.

    2. **Phantom explicit-negation** — the LLM emits a second CONTRADICT fact
       alongside the literal user statement (e.g. user says "I love steak and
       seafood"; LLM emits both "user loves steak and seafood" and an invented
       "user eats meat" CONTRADICT) when the supersedes_id on the literal fact
       would have sufficed. Pollutes the store with claims the user didn't make.

    Heuristic: tokenize each emitted fact's content into lowercased content
    words; subtract tokens present in the new user-turn transcript; the
    remainder is "ungrounded". If ungrounded tokens come from 2+ existing
    facts → strong synthesis signal. If they come from a single existing
    fact with >=50%% overlap → weaker phantom-negation signal.

    Logs a WARNING for each suspected fact. Does NOT drop facts — downstream
    reconciliation remains the dedup authority — but the WARNING is the
    deterministic test signal that catches regressions.
    """
    if not fact_docs or not turn_items:
        return

    user_turn_text = " ".join(
        str(m.get("content") or "") for m in turn_items if (m.get("role") or "").lower() == "user"
    )
    user_tokens = _grounding_tokens(user_turn_text)

    existing_with_tokens: list[tuple[str, set[str]]] = []
    for mem in existing_facts:
        toks = _grounding_tokens(str(mem.get("content") or ""))
        if toks:
            existing_with_tokens.append((str(mem.get("id") or ""), toks))

    for doc in fact_docs:
        content = str(doc.get("content") or "")
        fact_tokens = _grounding_tokens(content)
        if not fact_tokens:
            continue

        ungrounded = fact_tokens - user_tokens
        if not ungrounded:
            continue

        contributors: list[tuple[str, set[str]]] = [
            (eid, ungrounded & toks) for eid, toks in existing_with_tokens if ungrounded & toks
        ]

        if len(contributors) >= 2:
            logger.warning(
                "extract_memories: emitted fact appears synthesized from %d existing facts "
                "(ungrounded in user turns) — extract should ground only in this turn's [user] lines. "
                "doc_id=%s content=%r ungrounded_tokens=%s contributor_ids=%s "
                "user_id=%s thread_id=%s",
                len(contributors),
                doc.get("id"),
                content,
                sorted(ungrounded),
                [eid for eid, _ in contributors],
                user_id,
                thread_id,
            )
        elif len(contributors) == 1 and len(ungrounded) >= 2:
            eid, overlap = contributors[0]
            overlap_ratio = len(overlap) / len(ungrounded)
            if overlap_ratio >= 0.5:
                logger.warning(
                    "extract_memories: emitted fact has ungrounded tokens overlapping a single existing fact "
                    "(possible phantom-negation/restatement) — extract should ground only in this turn's "
                    "[user] lines. doc_id=%s content=%r ungrounded_tokens=%s overlap_existing_id=%s "
                    "overlap_ratio=%.2f user_id=%s thread_id=%s",
                    doc.get("id"),
                    content,
                    sorted(ungrounded),
                    eid,
                    overlap_ratio,
                    user_id,
                    thread_id,
                )


def parse_llm_json(text: str | None) -> dict[str, Any]:
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


def default_prompts_dir() -> str:
    """Default ``prompts/`` directory location: under ``azure/cosmos/agent_memory/``."""
    pkg_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(pkg_dir, "prompts")


def _read_prompty_version(path: str | Path) -> str:
    """Read the ``version:`` key from a prompty file's YAML front-matter."""
    text = Path(path).read_text(encoding="utf-8")
    if text.startswith("---"):
        end = text.find("\n---", 3)
        if end != -1:
            text = text[:end]
    match = _FRONT_MATTER_VERSION.search(text)
    return match.group(1) if match else DEFAULT_PROMPT_VERSION


class PromptyLoader:
    """Caching prompty template loader.

    Pure-IO module-aware: only reads the filesystem; never calls the LLM.
    Shared by sync and async pipeline services so the prepared (messages,
    params) pair has identical formatting on both code paths.
    """

    def __init__(self, prompts_dir: str | None = None) -> None:
        self._prompts_dir = prompts_dir if prompts_dir is not None else default_prompts_dir()
        self._cache: dict[str, Any] = {}
        self._version_cache: dict[str, str] = {}

    @property
    def prompts_dir(self) -> str:
        return self._prompts_dir

    def _path_for(self, filename: str) -> str:
        return os.path.join(self._prompts_dir, filename)

    def load(self, filename: str) -> Any:
        cached = self._cache.get(filename)
        if cached is not None:
            return cached
        import prompty

        loaded = prompty.load(self._path_for(filename))
        self._cache[filename] = loaded
        return loaded

    def prompt_version(self, filename: str) -> str:
        """Return the ``version:`` declared in the prompty front-matter."""
        cached = self._version_cache.get(filename)
        if cached is not None:
            return cached
        version = _read_prompty_version(self._path_for(filename))
        self._version_cache[filename] = version
        return version

    def prepare(self, filename: str, inputs: dict[str, Any]) -> tuple[list[dict[str, str]], dict[str, Any]]:
        """Render a prompty template and return ``(messages, model_params)``."""
        import prompty

        p = self.load(filename)
        messages = messages_to_dicts(prompty.prepare(p, inputs=inputs))
        params = extract_prompty_params(p)
        return messages, params


# Allowed values for the EpisodicRecord ``outcome_valence`` field — mirrors
# ``azure.cosmos.agent_memory.models._EPISODIC_ALLOWED_VALENCES`` but kept inline
# to avoid an import cycle (helpers must not import models).
VALID_VALENCES = frozenset({"positive", "negative", "neutral", "mixed"})


def coerce_valence(value: Any) -> str:
    """Map an LLM-emitted ``outcome_valence`` to a record-safe value.

    The strict response schema permits ``positive | negative | mixed | neutral
    | null``; null and any unknown value fall through to ``"neutral"`` so a
    single drifted episode never aborts the whole extract batch.
    """
    if isinstance(value, str) and value in VALID_VALENCES:
        return value
    return "neutral"


# Per-section caps on the persisted ``structured_summary``. Strict-mode JSON
# output does not enforce ``maxItems``, so the LLM grows lists unboundedly
# across incremental updates. Capping at persist time keeps both Cosmos
# document size and the next call's ``prior_summary`` prompt bounded.
SUMMARY_LIST_CAPS: dict[str, int] = {
    "key_facts": 50,
    "personal_preferences": 30,
    "account_environment": 30,
    "goals_current_work": 30,
    "behavioral_patterns": 30,
    "compliance_requirements": 30,
    "open_items": 30,
    "topics": 15,
    "goals": 30,
    "relationships": 30,
    "entities": 30,
}
SUMMARY_DEFAULT_CAP = 30


def cap_structured_summary(parsed: Optional[dict[str, Any]]) -> Optional[dict[str, Any]]:
    """Truncate every list field of a parsed summary to its configured cap.

    Returns a shallow copy with list fields replaced by their first-N slice.
    Non-list values are passed through unchanged. ``None`` returns ``None``.
    """
    if not isinstance(parsed, dict):
        return parsed
    out = dict(parsed)
    for key, value in list(out.items()):
        if isinstance(value, list):
            cap = SUMMARY_LIST_CAPS.get(key, SUMMARY_DEFAULT_CAP)
            if len(value) > cap:
                out[key] = value[:cap]
    return out
