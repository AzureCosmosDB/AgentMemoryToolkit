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
from typing import Any, Optional

from agent_memory_toolkit.exceptions import LLMError

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


def build_transcript(
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
    """Default ``prompts/`` directory location: under ``agent_memory_toolkit/``."""
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
# ``agent_memory_toolkit.models._EPISODIC_ALLOWED_VALENCES`` but kept inline
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
