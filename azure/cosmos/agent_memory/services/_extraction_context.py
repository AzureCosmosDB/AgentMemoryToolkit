"""Read-only context for fact extraction, shared by the sync and async pipelines.

A prompt that takes context (``extract_memories-v3``) sees, beside the new turns it extracts
from, the turns just before them, the thread summary when it was written before them, and the
user's stored facts nearest them. The context lets the model resolve references ("yes, do
that"), keep the rationale a new turn points at, and skip what is already stored; facts still
come only from the new turns. Each part is bounded, and each part predates the new turns, so
extracting an old conversation never shows the model what came after it.
"""

from __future__ import annotations

from typing import Any, Iterable, Mapping, Optional, Sequence

from azure.cosmos.agent_memory._embedding_tokens import count_tokens
from azure.cosmos.agent_memory.models import FACT_ASSERTIONS
from azure.cosmos.agent_memory.services._dedup import is_agent_sourced
from azure.cosmos.agent_memory.services._pipeline_helpers import build_transcript
from azure.cosmos.agent_memory.thresholds import (
    EXTRACTION_CONTEXT_MAX_TOKENS,
    EXTRACTION_CONTEXT_TURN_MAX_CHARS,
    EXTRACTION_CONTEXT_TURNS,
    EXTRACTION_KNOWN_FACTS,
    EXTRACTION_SUMMARY_MAX_CHARS,
)

# Extraction prompts that take the context inputs and label each new turn with its id; any
# other sees only the plain transcript.
CONTEXT_PROMPTS = frozenset({"extract_memories-v3.prompty"})
NONE = "(none)"
CLIP_MARK = " [...] "
# The text embedded to find stored facts near a batch.
QUERY_TURN_MAX_CHARS = 1500
QUERY_MAX_CHARS = 6000
KNOWN_FACT_MAX_CHARS = 600

# The fields a v3 answer gives each fact beyond the v2 ones; ``FactRecord`` normalizes them.
STRUCTURED_FACT_FIELDS = ("assertion", "subject", "attribute", "entities")


def takes_context(prompt_file: str) -> bool:
    return prompt_file in CONTEXT_PROMPTS


def clip(text: str, max_chars: int) -> str:
    """``text`` cut to about ``max_chars``, keeping its start and its end, where a reply usually
    asks what to do next."""
    text = text.strip()
    if len(text) <= max_chars:
        return text
    head = int(max_chars * 0.6)
    tail = max(max_chars - head - len(CLIP_MARK), 0)
    return text[:head].rstrip() + CLIP_MARK + (text[-tail:].lstrip() if tail else "")


def _created_at(turn: Mapping[str, Any]) -> str:
    return str(turn.get("created_at") or "")


def earliest_created_at(turns: Iterable[Mapping[str, Any]]) -> Optional[str]:
    return min((_created_at(turn) for turn in turns if _created_at(turn)), default=None)


def latest_created_at(turns: Iterable[Mapping[str, Any]]) -> Optional[str]:
    return max((_created_at(turn) for turn in turns if _created_at(turn)), default=None)


def _write_second(doc: Mapping[str, Any]) -> Optional[int]:
    value = doc.get("_ts")
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def window_write_second(turns: Iterable[Mapping[str, Any]], start: Optional[str]) -> Optional[int]:
    """When the window's earliest turns were written (Cosmos ``_ts``), or ``None`` when unknown.

    Turns imported from one session often share an event time (``created_at``), so the event
    time alone cannot tell which of them came before the window; the write order can.
    """
    seconds = [_write_second(turn) for turn in turns if _created_at(turn) == start]
    return None if not seconds or any(second is None for second in seconds) else min(seconds)


def previous_turns_query(
    user_id: str,
    thread_id: str,
    before: str,
    before_ts: Optional[int] = None,
) -> tuple[str, list[dict[str, Any]]]:
    """The thread's turns before ``before`` (an event time), most recently written first.

    With ``before_ts``, a turn sharing the event time ``before`` also counts when it was written
    before that second, as the earlier turns of the same session are.
    """
    earlier = "c.created_at < @before"
    if before_ts is not None:
        earlier = f"({earlier} OR (c.created_at = @before AND c._ts < @before_ts))"
    query = (
        f"SELECT TOP {EXTRACTION_CONTEXT_TURNS} c.id, c.role, c.content, c.created_at, c._ts FROM c "
        f"WHERE c.user_id = @user_id AND c.thread_id = @thread_id AND c.type = 'turn' AND {earlier} "
        "ORDER BY c._ts DESC"
    )
    parameters = [
        {"name": "@user_id", "value": user_id},
        {"name": "@thread_id", "value": thread_id},
        {"name": "@before", "value": before},
    ]
    if before_ts is not None:
        parameters.append({"name": "@before_ts", "value": before_ts})
    return query, parameters


def prior_turns(rows: Iterable[Mapping[str, Any]], window: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """The queried earlier turns that are not in the window, in conversation order (event time,
    then write order)."""
    window_ids = {str(turn.get("id")) for turn in window if turn.get("id")}
    kept = [dict(row) for row in rows if str(row.get("id")) not in window_ids]
    return sorted(kept, key=lambda turn: (_created_at(turn), _write_second(turn) or 0))


def context_turns_text(history: Sequence[Mapping[str, Any]]) -> str:
    """The last turns of ``history`` within the context budget, each clipped, as transcript lines
    without turn ids (so the model cannot cite them); ``(none)`` when there are none."""
    chosen: list[dict[str, Any]] = []
    used = 0
    for turn in reversed(history):
        if len(chosen) == EXTRACTION_CONTEXT_TURNS:
            break
        content = clip(str(turn.get("content") or ""), EXTRACTION_CONTEXT_TURN_MAX_CHARS)
        if not content:
            continue
        tokens = count_tokens(content)
        if chosen and used + tokens > EXTRACTION_CONTEXT_MAX_TOKENS:
            break
        chosen.append({"role": turn.get("role"), "content": content, "created_at": turn.get("created_at")})
        used += tokens
    chosen.reverse()
    return build_transcript(chosen, include_timestamp=True) if chosen else NONE


def summary_text(doc: Optional[Mapping[str, Any]], window_start: Optional[str], window_ts: Optional[int] = None) -> str:
    """The thread summary as context: its overview and decisions, clipped. ``(none)`` when there
    is no summary, or when it may cover turns from ``window_start`` on (its ``updated_at`` is the
    event time of the newest turn it summarized). A summary whose newest turn shares the window's
    event time is used only when it was written before the window's turns (``window_ts``)."""
    if not doc or not window_start:
        return NONE
    covered_to = str(doc.get("updated_at") or doc.get("created_at") or "")
    written = _write_second(doc)
    before_window = covered_to < window_start or (
        covered_to == window_start and window_ts is not None and written is not None and written < window_ts
    )
    if not covered_to or not before_window:
        return NONE
    metadata = doc.get("metadata") if isinstance(doc.get("metadata"), Mapping) else {}
    structured = metadata.get("structured_summary") if isinstance(metadata.get("structured_summary"), Mapping) else {}
    parts = [str(structured.get("overview") or doc.get("content") or "").strip()]
    decisions = [str(item).strip() for item in structured.get("decisions") or [] if str(item).strip()]
    if decisions:
        parts.append("Decisions: " + "; ".join(decisions[:5]))
    text = "\n".join(part for part in parts if part)
    return clip(text, EXTRACTION_SUMMARY_MAX_CHARS) if text else NONE


def query_text(batch: Sequence[Mapping[str, Any]]) -> str:
    """The text embedded to find the stored facts nearest a batch of turns."""
    parts = [clip(str(turn.get("content") or ""), QUERY_TURN_MAX_CHARS) for turn in batch]
    return clip("\n".join(part for part in parts if part), QUERY_MAX_CHARS)


def known_facts_query(
    user_id: str,
    vector: list[float],
    before: Optional[str],
    active_filter: str,
) -> tuple[str, list[dict[str, Any]]]:
    """The user's active facts nearest ``vector``, written no later than ``before`` (an event time)."""
    window = " AND c.created_at <= @before" if before else ""
    query = (
        f"SELECT TOP {EXTRACTION_KNOWN_FACTS} c.id, c.content, c.created_at, c.tags, c.metadata.source AS source, "
        "c.assertion, c.subject, c.attribute, c.entities FROM c "
        f"WHERE c.user_id = @user_id AND c.type = @fact_type AND {active_filter} "
        f"AND IS_DEFINED(c.embedding){window} ORDER BY VectorDistance(c.embedding, @embedding)"
    )
    parameters = [
        {"name": "@user_id", "value": user_id},
        {"name": "@fact_type", "value": "fact"},
        {"name": "@embedding", "value": vector},
    ]
    if before:
        parameters.append({"name": "@before", "value": before})
    return query, parameters


def _from_agent(doc: Mapping[str, Any]) -> bool:
    """Whether a stored-fact row (its ``tags`` and its flattened ``metadata.source``) records agent content."""
    return is_agent_sourced({"tags": doc.get("tags"), "metadata": {"source": doc.get("source")}})


def _known_fact_line(doc: Mapping[str, Any]) -> Optional[str]:
    content = clip(" ".join(str(doc.get("content") or "").split()), KNOWN_FACT_MAX_CHARS)
    if not content:
        return None
    assertion = doc.get("assertion") if doc.get("assertion") in FACT_ASSERTIONS else None
    source = "agent" if _from_agent(doc) else "user"
    head = " · ".join(part for part in (str(doc.get("created_at") or "")[:10], source, assertion) if part)
    entities = ", ".join(str(name) for name in doc.get("entities") or [] if isinstance(name, str) and name.strip())
    named = (("subject", doc.get("subject")), ("attribute", doc.get("attribute")), ("entities", entities))
    keys = "; ".join(f"{name}: {value}" for name, value in named if isinstance(value, str) and value.strip())
    line = f"- ({head}) {content}"
    return f"{line} [{keys}]" if keys else line


def known_facts_text(docs: Iterable[Mapping[str, Any]]) -> str:
    """Stored facts as context lines: date, who it came from (user or agent), assertion, text, then
    subject, attribute and entities."""
    lines = [line for line in map(_known_fact_line, docs) if line]
    return "\n".join(lines) or NONE


def structured_fields(fact: Mapping[str, Any]) -> dict[str, Any]:
    """The v3 fields a model answer gave one fact, unnormalized; absent for older prompts."""
    return {key: fact[key] for key in STRUCTURED_FACT_FIELDS if key in fact}
