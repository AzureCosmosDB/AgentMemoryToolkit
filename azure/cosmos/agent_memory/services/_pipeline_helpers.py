"""Pure helpers shared between sync and async pipeline services.

Anything that's a function of its inputs only (no LLM/Cosmos/embedding IO)
lives here so :class:`PipelineService` and :class:`AsyncPipelineService` can
share it without duplication. The :class:`PromptyLoader` cache-and-prepare
helper is here too because it is pure (filesystem read + dict shaping); the
LLM call itself stays on the service.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional

from azure.cosmos.agent_memory._embedding_tokens import count_tokens
from azure.cosmos.agent_memory._utils import cosine_similarity, vector_centroid
from azure.cosmos.agent_memory.exceptions import LLMError
from azure.cosmos.agent_memory.logging import describe_turn_refs, get_logger

logger = get_logger(__name__)


def parse_iso_datetime(value: Any) -> Optional[datetime]:
    """Parse an ISO-8601 string to a tz-aware datetime (UTC assumed for naive
    values), or return ``None`` if it is missing or unparseable.

    Pure: used to validate/normalize model-supplied episode timestamps without
    raising, so callers can fall back to grounded segment bounds.
    """
    if not isinstance(value, str) or not value.strip():
        return None
    try:
        dt = datetime.fromisoformat(value.strip().replace("Z", "+00:00"))
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


def is_valid_time_pair(started: Any, ended: Any) -> bool:
    """True iff both values parse as ISO-8601 datetimes with ``started <= ended``.

    Naive and tz-aware values are normalized to UTC before comparison, so a
    mixed pair (e.g. a naive date plus a tz-aware datetime) compares safely
    instead of raising ``TypeError``.
    """
    start = parse_iso_datetime(started)
    end = parse_iso_datetime(ended)
    return start is not None and end is not None and start <= end


def clamp_unit_interval(value: Any, default: float) -> float:
    """Clamp a number into ``[0.0, 1.0]``; return ``default`` if it is not a
    finite number.

    The strict LLM ``json_schema`` constrains the field type (number) but not
    its range, so salience / confidence can arrive out of range (e.g. 1.4 or
    -0.2). Normalizing rather than raising keeps the episode instead of dropping
    it over a slightly-off score.
    """
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return default
    v = float(value)
    if not math.isfinite(v):
        return default
    return max(0.0, min(1.0, v))


# ---------------------------------------------------------------------------
# Episode boundary segmentation - pure helpers shared by the sync and aio
# pipelines. IO-free: functions of their inputs (plus threshold values passed
# in), so the two pipelines share one implementation instead of hand-mirroring.
# ---------------------------------------------------------------------------


def turn_gap_seconds(prev_turn: dict[str, Any], cur_turn: dict[str, Any]) -> Optional[float]:
    """Seconds between two consecutive turns' ``created_at``, or None if either
    timestamp is missing/unparseable (the idle-gap check then abstains)."""
    prev_ts = parse_iso_datetime(prev_turn.get("created_at"))
    cur_ts = parse_iso_datetime(cur_turn.get("created_at"))
    if prev_ts is None or cur_ts is None:
        return None
    return (cur_ts - prev_ts).total_seconds()


def segment_time_bounds(items: list[dict[str, Any]]) -> tuple[Optional[str], Optional[str]]:
    """Return (earliest, latest) turn ``created_at`` in a segment, or (None, None).

    Used to ground an episode's started_at/ended_at in its actual turn window
    when the model omits them - for the conversational benchmarks each turn's
    created_at carries the session date, so this yields correct temporal spans.

    Ordering is by PARSED datetime, not lexical string order: mixed UTC offsets
    (e.g. ``+05:00`` vs ``Z``) sort correctly instead of producing an inverted
    ``(started, ended)`` pair that would later fail ``_validate_time_order``. The
    original ISO strings are returned unchanged.
    """
    parsed: list[tuple[datetime, str]] = []
    for item in items:
        raw = item.get("created_at")
        if not isinstance(raw, str) or not raw:
            continue
        dt = parse_iso_datetime(raw)
        if dt is not None:
            parsed.append((dt, raw))
    if not parsed:
        return None, None
    parsed.sort(key=lambda pair: pair[0])
    return parsed[0][1], parsed[-1][1]


def created_at_sort_key(item: dict[str, Any]) -> tuple[int, float, str]:
    """Total, chronological, stable sort key for a turn by ``created_at``.

    Orders by PARSED instant (so mixed UTC offsets sort by true time, not lexical
    string) and breaks ties on ``id`` so turns sharing one timestamp - common
    when many turns carry the same session date - keep a deterministic order. A
    stable order matters because the open segment's first/last turn ids seed the
    ``segment_key`` that grounds the deterministic episode id: an unstable order
    could shift that id between runs and admit a duplicate episode. Missing or
    unparseable timestamps sort last, deterministically by id.
    """
    dt = parse_iso_datetime(item.get("created_at"))
    if dt is None:
        return (1, 0.0, str(item.get("id") or ""))
    return (0, dt.timestamp(), str(item.get("id") or ""))


def chronological_key(item: Mapping[str, Any]) -> tuple[str, int]:
    """Sort key for conversation order: event time (``created_at``), then write order (Cosmos ``_ts``).

    Turns sharing an event time, as an imported session's do, keep the order they were written
    in; a stable sort keeps the order they were read in within one write second.
    """
    return str(item.get("created_at") or ""), int(item.get("_ts") or 0)


def newest_window(items: Iterable[dict[str, Any]], limit: Optional[int]) -> list[dict[str, Any]]:
    """The newest ``limit`` of ``items`` (all of them when ``limit`` is ``None``), oldest first."""
    ordered = sorted(items, key=chronological_key)
    return ordered if limit is None else ordered[max(len(ordered) - limit, 0) :]


def deterministic_episode_id(segment_key: str, index: int) -> str:
    """Identity from the STABLE segment key plus the episode's ordinal within the
    segment - never the LLM summary text - so re-running the same un-stamped
    segment yields the same id and the duplicate write is skipped (409)."""
    seed = ID_SEED_SEP.join((segment_key, str(index)))
    return f"ep_{hashlib.sha256(seed.encode()).hexdigest()[:32]}"


def find_episode_boundary(
    segment: list[dict[str, Any]],
    embeddings: list[list[float]],
    *,
    max_turns: int,
    idle_gap: int,
    drift: float,
    min_turns: int,
) -> Optional[int]:
    """Return the exclusive end index of the earliest episode boundary in the
    open segment, or ``None`` if the segment is still open.

    Signals, earliest wins: (1) an idle time-gap between two consecutive turns,
    (2) a topic-drift shift of a new turn away from the segment centroid, and
    (3) a max-size cap that force-closes an over-long segment. Idle-gap and drift
    boundaries below ``min_turns`` are suppressed so a lone turn is not emitted as
    a trivial episode; the max-size cap is a hard ceiling and is not floored.
    Boundaries are temporal/semantic only; prior episodes are never mutated.
    """
    n = len(segment)
    if n == 0:
        return None
    min_turns = max(1, min_turns)
    has_embeddings = bool(embeddings) and len(embeddings) == n
    for i in range(1, n):
        if max_turns > 0 and i >= max_turns:
            return i
        if idle_gap > 0 and i >= min_turns:
            gap = turn_gap_seconds(segment[i - 1], segment[i])
            if gap is not None and gap > idle_gap:
                return i
        if drift > 0 and i >= min_turns and has_embeddings:
            centroid = vector_centroid(embeddings[:i])
            if centroid and (1.0 - cosine_similarity(embeddings[i], centroid)) > drift:
                return i
    if max_turns > 0 and n >= max_turns:
        return max_turns
    return None


_NON_RETRYABLE_LLM_MARKERS = (
    "content_filter",
    "model_refusal",
    "unexpected_tool_response",
    "content management policy",
    "context_length_exceeded",
    "maximum context length",
)

# Programming errors, not provider failures: a bug in our own extraction/parse
# code (e.g. calling ``.get`` on a non-dict, indexing past the end) raises one of
# these. They are deterministic, so retrying re-fails identically and would wedge
# the segment/batch forever - classify them non-retryable so the poison input is
# quarantined and surfaced rather than deferred indefinitely. Note this is a
# narrow allow-list of true code-bug types, NOT "any non-LLMError": genuine
# transient provider failures (rate-limit / timeout / connection) are their own
# SDK exception types and must stay retryable.
_NON_RETRYABLE_EXC_TYPES = (AttributeError, KeyError, TypeError, IndexError, NameError)


def is_retryable_llm_error(exc: BaseException) -> bool:
    """Classify an extraction LLM failure as retryable (transient) or not."""
    if isinstance(exc, _NON_RETRYABLE_EXC_TYPES):
        return False
    text = str(exc).lower()
    return not any(marker in text for marker in _NON_RETRYABLE_LLM_MARKERS)


def batch_turns_by_tokens(
    items: list[dict[str, Any]],
    max_tokens: int,
    *,
    model: str = "gpt-5.4",
) -> list[list[dict[str, Any]]]:
    """Greedily pack ordered *items* into batches whose combined content stays
    within *max_tokens*.

    Token-bounded batching keeps each extraction call small enough that (a) the
    model can attend to every turn (more complete extraction - smaller windows
    extract more faithfully) and (b) a single poisoned turn fails only its own
    batch instead of the whole backlog. A turn larger than *max_tokens* on its
    own becomes a singleton batch (never dropped).
    """
    if not items:
        return []
    batches: list[list[dict[str, Any]]] = []
    current: list[dict[str, Any]] = []
    current_tokens = 0
    for item in items:
        item_tokens = count_tokens(str(item.get("content") or ""), model)
        if current and current_tokens + item_tokens > max_tokens:
            batches.append(current)
            current = []
            current_tokens = 0
        current.append(item)
        current_tokens += item_tokens
    if current:
        batches.append(current)
    return batches


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


def _normalize_cadence_thresholds(
    value: Optional[Mapping[str, int]],
) -> Optional[dict[str, int]]:
    """Validate + defensively copy a ``cadence_thresholds`` argument.

    Coerces each value to ``int`` and rejects negatives (``0`` disables the
    corresponding step). Returns a new ``dict`` so later mutation of the
    caller's mapping cannot change client behavior. Returns ``None`` for
    missing or empty input.
    """
    if value is None:
        return None
    if not isinstance(value, Mapping):
        raise TypeError(
            "cadence_thresholds must be a mapping of env-var name to int "
            f"(e.g. {{'FACT_EXTRACTION_EVERY_N': 5}}), not {type(value).__name__}."
        )
    normalized: dict[str, int] = {}
    for key, raw in value.items():
        try:
            coerced = int(raw)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"cadence_thresholds[{key!r}] must be an int, got {raw!r}.") from exc
        if coerced < 0:
            raise ValueError(f"cadence_thresholds[{key!r}] must be >= 0 (0 disables the step), got {coerced}.")
        normalized[str(key)] = coerced
    return normalized or None


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
    include_timestamp: bool = False,
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
        Defaults to ``None`` (no metadata serialized - only ``[role]:
        content`` lines). When provided, only the listed keys are emitted,
        in iteration order. Keys absent from a given turn's metadata are
        silently skipped.

        Set this when callers stash semantically useful breadcrumbs in
        ``TurnRecord.metadata`` that the extraction LLM should see
        (e.g. ``["agent_id", "timestamp"]``). Leaving it unset keeps free-form
        metadata blobs (raw tool calls, IDE schema, etc.) out of every
        prompt - they're often 10-100x larger than the dialog itself and
        dilute extraction quality.

        Accepts any iterable of strings except ``str`` itself (which would
        be interpreted char-by-char). Generators are coerced to a tuple so
        the allow-list is reusable across turns.
    include_timestamp:
        If *True*, prefix each line with the turn's top-level ``created_at``
        (its event time) as ``[<created_at> | role]: content``. This lets the
        extraction LLM anchor relative time expressions ("3 weeks ago", "last
        June") to absolute dates instead of leaving them unresolved. Turns
        without a ``created_at`` fall back to the plain ``[role]:`` form.
    """
    keys = _normalize_metadata_keys(metadata_keys)

    def _line(m: dict[str, Any]) -> str:
        role = _canonical_speaker(m.get("role", "unknown"))
        content = m.get("content", "")
        meta_str = _format_metadata_segment(m.get("metadata", {}), keys)
        created_at = m.get("created_at") if include_timestamp else None
        prefix = f"[{created_at} | {role}]" if created_at else f"[{role}]"
        return f"{prefix}: {content}{meta_str}"

    if not group_by_thread:
        return "\n".join(_line(m) for m in items)

    threads: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for m in items:
        threads[m.get("thread_id", "")].append(m)

    parts: list[str] = []
    for tid, thread_items in threads.items():
        parts.append(f"=== Thread {tid} ===")
        for m in thread_items:
            parts.append(_line(m))
        parts.append("")
    return "\n".join(parts)


def transcript_labels(turns: list[Mapping[str, Any]]) -> list[str]:
    """Return the label each turn shows in a grounded transcript.

    A turn's own id where it has one; otherwise ``turn-N`` for its 1-based position, made
    unique so it can never read as another turn's real id.
    """
    taken = {str(turn["id"]) for turn in turns if turn.get("id")}
    labels: list[str] = []
    for index, turn in enumerate(turns, start=1):
        if turn.get("id"):
            labels.append(str(turn["id"]))
            continue
        label, suffix = f"turn-{index}", 0
        while label in taken:
            suffix += 1
            label = f"turn-{index}.{suffix}"
        taken.add(label)
        labels.append(label)
    return labels


def ground_fact_sources(fact: Mapping[str, Any], *, turn_ids: list[Optional[str]], labels: list[str]) -> list[str]:
    """Resolve a fact's model-declared turn refs to real ids from its batch.

    ``turn_ids`` holds one slot per transcript line, ``None`` where the turn has no id, and
    ``labels`` the label each line showed (see :func:`transcript_labels`). A ref is matched
    against those exact labels first, then read as a positional ``turn-N``, so it resolves to
    exactly the line the model saw. A ref to an unidentified turn or to anything outside the
    batch is dropped rather than recorded as provenance, so a fact whose citations all fail
    records none, as episodes do. Only a fact that cites nothing (a v1 or v2 response) falls
    back to every identified turn in the batch.
    """
    identified = [turn_id for turn_id in turn_ids if turn_id]
    refs = fact.get("source_turn_ids")
    if not isinstance(refs, list):
        refs = [refs] if refs else []
    if not refs:
        return identified
    by_label = dict(zip(labels, turn_ids))
    by_folded_label = {label.lower(): turn_id for label, turn_id in by_label.items()}
    by_position = {f"turn-{i}": turn_id for i, turn_id in enumerate(turn_ids, start=1)}
    grounded: list[str] = []
    for raw in refs:
        ref = str(raw).strip()
        if ref in by_label:
            mapped = by_label[ref]
        elif ref.lower() in by_folded_label:
            mapped = by_folded_label[ref.lower()]
        else:
            mapped = by_position.get(ref.lower())
        if mapped and mapped not in grounded:
            grounded.append(mapped)
    if not grounded:
        logger.warning(
            "extract_memories: no cited turn resolves in the batch; recording none refs=%s",
            describe_turn_refs(refs),
        )
    return grounded


# Map common role synonyms onto the toolkit's canonical speaker labels. Callers
# may write turns with any role string (e.g. OpenAI's ``assistant``); the
# extraction prompt reasons about ``user`` vs ``agent``, so synonyms are folded
# onto those. ``tool`` and ``system`` are distinct roles and kept as-is;
# unrecognized labels pass through unchanged rather than being misattributed.
_SPEAKER_ALIASES = {
    "user": "user",
    "human": "user",
    "customer": "user",
    "end_user": "user",
    "person": "user",
    "agent": "agent",
    "assistant": "agent",
    "ai": "agent",
    "bot": "agent",
    "chatbot": "agent",
    "model": "agent",
    "copilot": "agent",
    "tool": "tool",
    "system": "system",
}


def _canonical_speaker(role: Any) -> str:
    """Normalize a free-form role to the canonical speaker label for prompts."""
    normalized = str(role or "").strip().lower()
    return _SPEAKER_ALIASES.get(normalized, str(role or "unknown"))


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
    cleaned = cleaned.strip()
    decoder = json.JSONDecoder()
    try:
        obj, end = decoder.raw_decode(cleaned)
    except json.JSONDecodeError as exc:
        # The decode error names only a position; the response itself stays out of the error, which is logged.
        if _looks_truncated(cleaned, exc):
            raise LLMError(
                "LLM JSON output appears TRUNCATED (decode error at the very end of a "
                f"{len(cleaned)}-char body - the model almost certainly hit its output-token "
                "cap mid-object). Increase 'maxOutputTokens' in the calling prompty, or reduce "
                "the amount of input per call (e.g. lower the fact-extraction batch size / "
                f"recent_k, or split oversized turns). Decode error: {exc}."
            ) from exc
        raise LLMError(f"LLM returned invalid JSON ({len(cleaned)}-char body): {exc}") from exc
    remainder = cleaned[end:]
    if not isinstance(obj, dict):
        # Type contract: this returns a JSON object. Some deployments emit a bare
        # array or scalar root (e.g. ``[{...}]``); every caller then does
        # ``parsed.get("facts"/"episodes")``, so returning a non-dict would
        # surface downstream as an AttributeError that the extraction
        # error-handlers misclassify as a transient (retryable) failure and defer
        # forever. Raise a typed LLMError so a malformed root is handled like any
        # other bad output instead of wedging the segment/batch.
        raise LLMError(
            f"LLM returned a non-object JSON root ({type(obj).__name__}); expected a JSON "
            f"object such as {{'facts': [...]}}."
        )
    if not remainder.strip():
        # Fast path: exactly one JSON object, no trailing content (the vast
        # majority of responses).
        return obj

    # Some models (notably under strict json_schema on certain deployments) emit
    # MULTIPLE back-to-back top-level JSON objects for a single call - e.g.
    # ``{"facts":[...]}{"facts":[...]}``. ``raw_decode`` only returns the first,
    # so keeping just it would silently DROP every item in the trailing
    # object(s) - and the batch's turns get stamped ``extracted_at``, so those
    # facts are never recovered. Instead, decode every top-level object and merge
    # them: list-valued keys (``facts``, ``events``, ...) are concatenated and any
    # other/scalar key keeps its first-seen value. Downstream exact-dup hashing
    # removes any repeats, so merging is always safe.
    merged: dict[str, Any] = obj
    object_count = 1
    pos = end
    length = len(cleaned)
    while pos < length:
        # Skip inter-object whitespace and stray separators between objects.
        while pos < length and cleaned[pos] in " \t\r\n,":
            pos += 1
        if pos >= length:
            break
        try:
            nxt, pos = decoder.raw_decode(cleaned, pos)
        except json.JSONDecodeError:
            leftover = cleaned[pos:].strip()
            logger.warning(
                "LLM response had %d chars of non-JSON trailing data after %d concatenated "
                "JSON object(s); merged the objects and ignored the remainder",
                len(leftover),
                object_count,
            )
            break
        object_count += 1
        if isinstance(nxt, dict):
            _merge_json_objects(merged, nxt)
        # A non-dict follow-on cannot be merged into a dict result; skip it.

    if object_count > 1:
        logger.info(
            "LLM response contained %d concatenated JSON objects; merged their list fields into "
            "one result so no items were dropped.",
            object_count,
        )
    return merged


def _merge_json_objects(base: dict[str, Any], extra: dict[str, Any]) -> None:
    """Merge ``extra`` into ``base`` in place.

    List-valued keys present in both are concatenated (this is what recovers the
    facts/events an LLM splits across multiple concatenated JSON objects); keys
    only in ``extra`` are added; any other conflict keeps ``base``'s (first-seen)
    value. Exact-duplicate items are pruned later by the pipeline's content-hash
    dedup, so concatenating without de-duping here is safe.
    """
    for key, value in extra.items():
        if key not in base:
            base[key] = value
        elif isinstance(base[key], list) and isinstance(value, list):
            base[key].extend(value)


def _looks_truncated(cleaned: str, exc: json.JSONDecodeError) -> bool:
    """Heuristic: did the JSON fail because the model ran out of output tokens?"""
    if not cleaned:
        return False
    unbalanced = cleaned.count("{") > cleaned.count("}") or cleaned.count("[") > cleaned.count("]")
    unterminated_string = "Unterminated string" in str(exc)
    return unbalanced or unterminated_string


def default_prompts_dir() -> str:
    """Default ``prompts/`` directory location: under ``azure/cosmos/agent_memory/``."""
    pkg_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(pkg_dir, "prompts")


_EXTRACT_MEMORIES_PROMPT_DEFAULT = "extract_memories-v3.prompty"
_EXTRACT_MEMORIES_PROMPT_ALLOWED = frozenset(
    {"extract_memories.prompty", "extract_memories-v2.prompty", "extract_memories-v3.prompty"}
)


def extract_memories_prompt_file() -> str:
    """Return the fact-extraction prompt filename, env-selectable.

    Defaults to the v3 ``extract_memories-v3.prompty`` extractor: the v2 rules (higher recall;
    also captures assistant-provided information: lists, tables, instructions, and researched
    answers the user may later reference) plus read-only context (the thread's earlier turns
    and summary, and the user's related stored memories), how each fact was asserted, its
    subject, attribute and entities, and the turns it came from. Set
    ``AMT_EXTRACT_MEMORIES_PROMPT=extract_memories-v2.prompty`` (or ``extract_memories.prompty``
    for v1) to select an earlier extractor without touching the default. An unknown value falls
    back to the v3 default so a typo can never point extraction at an arbitrary or missing
    prompt file. The chosen filename must be registered in ``PROMPTY_SCHEMAS`` so it still gets
    the structured-output response format.
    """
    name = os.environ.get("AMT_EXTRACT_MEMORIES_PROMPT", _EXTRACT_MEMORIES_PROMPT_DEFAULT)
    return name if name in _EXTRACT_MEMORIES_PROMPT_ALLOWED else _EXTRACT_MEMORIES_PROMPT_DEFAULT


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
        self._disable_strict_format(prompty, loaded)
        self._cache[filename] = loaded
        return loaded

    def _disable_strict_format(self, prompty: Any, loaded: Any) -> None:
        """Disable prompty strict nonce parsing for SDK-owned internal prompts."""
        template = getattr(loaded, "template", None)
        if template is None:
            template_type = getattr(prompty, "Template", None)
            if template_type is None:
                return
            template = template_type()
            loaded.template = template

        format_config = getattr(template, "format", None)
        if format_config is None:
            format_type = getattr(prompty, "FormatConfig", None)
            if format_type is None:
                return
            format_config = format_type()
            template.format = format_config

        if hasattr(format_config, "strict"):
            format_config.strict = False

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
