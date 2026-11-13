"""Procedure pass helpers shared by the sync and async pipelines.

``synthesize_procedural`` reads the behavioral facts and episodes written since the user's
procedure cursor and shows them to the model in small batches, in the order they were written,
each with the user's existing procedures most similar to it. Each answer adds a procedure,
revises an existing one (``update``) or records new evidence for one (``reinforce``). The
cursor moves past each batch once it is done, so a run that stops early resumes there.

A procedure's kind says how it was learned, and its status follows from the evidence it has
gathered (``procedure_status``): an instruction is active once the user taught it, a habit
after a correction or on two dates, a skill after two successful practices or a user
endorsement. Anything else stays a candidate, which search does not show, until more evidence
arrives. Reading only new sources means a source is distilled once; showing existing
procedures lets the model revise or reinforce one instead of minting a paraphrase of it.

Teaching or correcting needs a fact extraction recorded as the user's ``directive`` or
``correction`` (facts extracted before assertions were recorded still count). Once an active
situational procedure holds such a fact, the fact is retired into it, so the rule is stored and
retrieved once, as a procedure.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Iterable, Mapping, Optional, Sequence

from azure.cosmos.agent_memory.models import FACT_ASSERTIONS, USER_ONLY_ASSERTIONS, ProcedureEvidence, ProcedureKind
from azure.cosmos.agent_memory.services._dedup import (
    ACTIVE_FILTER,
    AFTER_POSITION,
    checkpoint_position,
    is_agent_sourced,
    next_position,
    position_parameters,
    specifics_preserved,
    without_system_fields,
)
from azure.cosmos.agent_memory.services._pipeline_helpers import clamp_unit_interval, is_retryable_llm_error

PROCEDURAL_THREAD_ID = "__procedural__"
PROCEDURE_CURSOR_ID = "procedure_cursor"
PROCEDURE_CURSOR_TYPE = "procedure_cursor"
SOURCE_BATCH = 50
# Sources per model call: few enough that repeated evidence across batches meets the
# procedures earlier batches created.
PROMPT_BATCH = 10
RELATED_PER_SOURCE = 3
MAX_RELATED_PROCEDURES = 20
BEHAVIORAL_MIN_SALIENCE = 0.8

PROCEDURE_KINDS = frozenset(kind.value for kind in ProcedureKind)
PROCEDURE_EVIDENCE = frozenset(evidence.value for evidence in ProcedureEvidence)
# Evidence that the user taught or corrected the assistant, which only their directives and corrections give.
TEACHING_EVIDENCE = frozenset({ProcedureEvidence.taught.value, ProcedureEvidence.corrected.value})
GOOD_OUTCOMES = frozenset({"successful", "partially_successful"})
_ISO_DATE = re.compile(r"\d{4}-\d{2}-\d{2}")

# The session-start block of standing rules: at most this many, in at most this many bytes.
STANDING_RULES_MAX = 10
STANDING_RULES_MAX_BYTES = 2048
STANDING_HEADER = "# Standing instructions from the user"
SITUATIONAL_HEADER = "## Apply only when the situation matches"

# Statuses of a failed embedding call or Cosmos write that a later run can expect to get past.
_TRANSIENT_STATUSES = frozenset({408, 429, 449, 500, 502, 503, 504})


def is_transient_write_failure(exc: BaseException) -> bool:
    """Whether a failed embedding call or Cosmos write is worth retrying on a later run:
    throttling, a timeout, a server error or a connection failure, but not a bad request or a conflict."""
    status = getattr(exc, "status_code", None)
    if isinstance(status, int):
        return status in _TRANSIENT_STATUSES
    return is_retryable_llm_error(exc)


@dataclass(frozen=True)
class ProcedureScope:
    """One user's procedure data: what selects their memories, and where their procedures live."""

    user_id: str

    @property
    def predicate(self) -> str:
        return "c.user_id = @uid"

    @property
    def parameters(self) -> list[dict[str, Any]]:
        return [{"name": "@uid", "value": self.user_id}]

    @property
    def partition_key(self) -> list[str]:
        """Where procedures and the procedure cursor are stored (a point read or write)."""
        return [self.user_id, PROCEDURAL_THREAD_ID]

    @property
    def fields(self) -> dict[str, str]:
        return {"user_id": self.user_id, "thread_id": PROCEDURAL_THREAD_ID}


# ---------------------------------------------------------------- shared helpers
def procedure_text(doc: Mapping[str, Any]) -> str:
    """The text a procedure is compared by: its name, summary, conditions, constraints and steps,
    so a comparison cannot miss what only those state. (A fact is compared by its content alone.)"""
    head = str(doc.get("name") or "").strip()
    summary = str(doc.get("summary") or doc.get("content") or "").strip()
    parts = [f"{head}: {summary}" if head and summary else head or summary]
    for label, key in (
        ("When", "activation_conditions"),
        ("Requires", "preconditions"),
        ("Constraints", "safety_constraints"),
    ):
        items = [str(item).strip() for item in (doc.get(key) or []) if str(item).strip()]
        if items:
            parts.append(f"{label}: " + "; ".join(items))
    steps = [
        str(step.get("instruction") or "").strip() for step in (doc.get("steps") or []) if isinstance(step, Mapping)
    ]
    if any(steps):
        parts.append("Steps: " + "; ".join(step for step in steps if step))
    return " | ".join(part for part in parts if part)


# ---------------------------------------------------------------- reading sources
def _positions(cursor: Optional[Mapping[str, Any]]) -> tuple[dict[str, Any], dict[str, Any]]:
    stored = cursor or {}
    return checkpoint_position(stored.get("facts")), checkpoint_position(stored.get("episodes"))


def behavioral_facts_query(
    scope: ProcedureScope, cursor: Optional[Mapping[str, Any]]
) -> tuple[str, list[dict[str, Any]]]:
    """Active facts written after the cursor, oldest first, that may teach a procedure: preferences,
    requirements, high-salience facts, and the user's directives and corrections."""
    query = (
        f"SELECT TOP {SOURCE_BATCH} * FROM c WHERE {scope.predicate} AND c.type = @type AND {ACTIVE_FILTER} "
        "AND ((IS_DEFINED(c.metadata.category) AND c.metadata.category IN ('preference', 'requirement')) "
        "OR (IS_DEFINED(c.salience) AND c.salience >= @min_salience) "
        "OR (IS_DEFINED(c.assertion) AND c.assertion IN ('directive', 'correction')))" + AFTER_POSITION
    )
    parameters = [
        *scope.parameters,
        {"name": "@type", "value": "fact"},
        {"name": "@min_salience", "value": BEHAVIORAL_MIN_SALIENCE},
        *position_parameters(_positions(cursor)[0]),
    ]
    return query, parameters


def lesson_episodes_query(
    scope: ProcedureScope, cursor: Optional[Mapping[str, Any]]
) -> tuple[str, list[dict[str, Any]]]:
    """Active episodes with lessons written after the cursor, oldest first."""
    query = (
        f"SELECT TOP {SOURCE_BATCH} * FROM c WHERE {scope.predicate} AND c.type = @type AND {ACTIVE_FILTER} "
        "AND IS_DEFINED(c.lessons) AND ARRAY_LENGTH(c.lessons) > 0" + AFTER_POSITION
    )
    parameters = [
        *scope.parameters,
        {"name": "@type", "value": "episodic"},
        *position_parameters(_positions(cursor)[1]),
    ]
    return query, parameters


def related_procedure_query(scope: ProcedureScope, vector: list[float]) -> tuple[str, list[dict[str, Any]]]:
    """The active procedures nearest one source."""
    query = (
        f"SELECT TOP {RELATED_PER_SOURCE} * FROM c WHERE {scope.predicate} AND c.type = @memory_type "
        f"AND {ACTIVE_FILTER} AND IS_DEFINED(c.embedding) ORDER BY VectorDistance(c.embedding, @embedding)"
    )
    parameters = [
        *scope.parameters,
        {"name": "@memory_type", "value": "procedural"},
        {"name": "@embedding", "value": vector},
    ]
    return query, parameters


_IN_FORCE = "(NOT IS_DEFINED(c.valid_until) OR IS_NULL(c.valid_until) OR c.valid_until >= @today)"


def _in_force_procedures(scope: ProcedureScope, applies: str, today: str) -> tuple[str, list[dict[str, Any]]]:
    predicate = (
        f"{scope.predicate} AND c.type = @memory_type AND c.status = @status AND c.applies = @applies "
        f"AND {ACTIVE_FILTER} AND {_IN_FORCE}"
    )
    parameters = [
        *scope.parameters,
        {"name": "@memory_type", "value": "procedural"},
        {"name": "@status", "value": "active"},
        {"name": "@applies", "value": applies},
        {"name": "@today", "value": today},
    ]
    return predicate, parameters


def standing_rules_query(scope: ProcedureScope, today: str) -> tuple[str, list[dict[str, Any]]]:
    """The active procedures in force that apply to nearly every interaction."""
    predicate, parameters = _in_force_procedures(scope, "always", today)
    return f"SELECT * FROM c WHERE {predicate}", parameters


def situational_rule_query(scope: ProcedureScope, vector: list[float], today: str) -> tuple[str, list[dict[str, Any]]]:
    """The active situational procedure in force nearest a task."""
    predicate, parameters = _in_force_procedures(scope, "when", today)
    query = (
        f"SELECT TOP 1 * FROM c WHERE {predicate} AND IS_DEFINED(c.embedding) "
        "ORDER BY VectorDistance(c.embedding, @embedding)"
    )
    return query, [*parameters, {"name": "@embedding", "value": vector}]


def rank_related(scored: Iterable[tuple[dict[str, Any], float]]) -> list[dict[str, Any]]:
    """Distinct procedures by their best similarity to any source, most similar first."""
    best: dict[str, tuple[dict[str, Any], float]] = {}
    for doc, score in scored:
        doc_id = doc.get("id")
        if doc_id and (doc_id not in best or score > best[doc_id][1]):
            best[doc_id] = (doc, score)
    ranked = sorted(best.values(), key=lambda item: item[1], reverse=True)
    return [doc for doc, _ in ranked[:MAX_RELATED_PROCEDURES]]


def is_source(doc: Mapping[str, Any]) -> bool:
    """Whether a memory the pass read is something to learn from: a fact with text, or an episode with a lesson."""
    if doc.get("type") == "episodic":
        return any(isinstance(lesson, str) and lesson.strip() for lesson in doc.get("lessons") or [])
    return isinstance(doc.get("content"), str) and bool(doc["content"].strip())


def source_batches(
    facts: Iterable[Mapping[str, Any]], episodes: Iterable[Mapping[str, Any]]
) -> list[tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]]:
    """The model calls a run makes, in write order (``_ts``): ``(sources, facts, episodes)``.

    Each batch holds up to ``PROMPT_BATCH`` sources; ``facts`` and ``episodes`` are every fact
    and episode read from the end of the previous batch through its last source, including ones
    with nothing to learn from. In write order they are the next stretch of each cursor position,
    so the cursor can move past a batch as soon as it is done. Memories left after the last
    source form a final batch without sources.
    """
    ordered = sorted([*facts, *episodes], key=lambda doc: (doc.get("_ts") or 0, str(doc.get("id"))))
    batches: list[tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]] = []
    sources: list[dict[str, Any]] = []
    read: list[dict[str, Any]] = []

    def close() -> None:
        episodic = [doc for doc in read if doc.get("type") == "episodic"]
        batches.append((sources, [doc for doc in read if doc.get("type") != "episodic"], episodic))

    for doc in ordered:
        read.append(dict(doc))
        if is_source(doc):
            sources.append(dict(doc))
            if len(sources) == PROMPT_BATCH:
                close()
                sources, read = [], []
    if read:
        close()
    return batches


# ---------------------------------------------------------------- what the model reads
def source_speaker(doc: Mapping[str, Any]) -> str:
    """Who stated a source fact: the ``assistant`` for agent content, otherwise the ``user``."""
    return "assistant" if is_agent_sourced(doc) else "user"


def can_teach(source: Mapping[str, Any]) -> bool:
    """Whether a source can show the user taught or corrected the assistant: a fact the user stated
    that extraction recorded as a directive or correction, or one extracted before facts recorded
    how they were asserted. A preference or a request about one task is not teaching."""
    if source.get("type") == "episodic" or source_speaker(source) != "user":
        return False
    assertion = source.get("assertion")
    return not assertion or assertion in USER_ONLY_ASSERTIONS


def source_date(doc: Mapping[str, Any]) -> str:
    """A source's ISO date, or ``unknown`` when its ``created_at`` does not start with one."""
    value = str(doc.get("created_at") or "")[:10]
    return value if _ISO_DATE.fullmatch(value) else "unknown"


def _outcome_status(doc: Mapping[str, Any]) -> str:
    outcome = doc.get("outcome")
    return str(outcome.get("status") or "") if isinstance(outcome, Mapping) else ""


def _fact_line(label: str, doc: Mapping[str, Any]) -> str:
    metadata = doc.get("metadata")
    category = metadata.get("category") if isinstance(metadata, Mapping) else None
    category = category.strip() if isinstance(category, str) and category.strip() else "unknown"
    assertion = doc.get("assertion") if doc.get("assertion") in FACT_ASSERTIONS else None
    details = [category, source_speaker(doc), *([assertion] if assertion else []), source_date(doc)]
    return f"{label} [{' · '.join(details)}]: {str(doc['content']).strip()}"


def _episode_line(label: str, doc: Mapping[str, Any]) -> str:
    outcome = doc.get("outcome") if isinstance(doc.get("outcome"), Mapping) else {}
    description = str(outcome.get("description") or "").strip()
    status = (_outcome_status(doc) or "none") + (f" - {description[:240]}" if description else "")
    lessons = "; ".join(
        lesson.strip() for lesson in doc.get("lessons") or [] if isinstance(lesson, str) and lesson.strip()
    )
    return (
        f"{label} [{source_date(doc)}] {str(doc.get('title') or '').strip()} (outcome: {status}). "
        f"Summary: {str(doc.get('content') or '').strip()[:700]} Lessons: {lessons}"
    )


def sources_input(batch: Sequence[Mapping[str, Any]]) -> tuple[dict[str, str], dict[str, dict[str, Any]]]:
    """Prompt inputs for a batch of sources, and the source each ``fact-N`` / ``ep-N`` label names."""
    facts: list[str] = []
    episodes: list[str] = []
    labels: dict[str, dict[str, Any]] = {}
    for doc in batch:
        if doc.get("type") == "episodic":
            label = f"ep-{len(episodes) + 1}"
            episodes.append(_episode_line(label, doc))
        else:
            label = f"fact-{len(facts) + 1}"
            facts.append(_fact_line(label, doc))
        labels[label] = dict(doc)
    inputs = {"behavioral_facts": "\n".join(facts) or "(none)", "episodes": "\n".join(episodes) or "(none)"}
    return inputs, labels


def support_count(doc: Mapping[str, Any]) -> int:
    return len(doc.get("source_fact_ids") or []) + len(doc.get("source_episodic_ids") or [])


def existing_procedures_input(procedures: Sequence[Mapping[str, Any]]) -> tuple[str, dict[str, dict[str, Any]]]:
    """Prompt lines labeled ``proc-N`` and the procedure each label names."""
    if not procedures:
        return "(none)", {}
    lines: list[str] = []
    by_label: dict[str, dict[str, Any]] = {}
    for index, doc in enumerate(procedures, start=1):
        label = f"proc-{index}"
        by_label[label] = dict(doc)
        evidence = "/".join(sorted(str(item) for item in doc.get("evidence") or [])) or "none"
        count = support_count(doc)
        lines.append(
            f"{label} [{doc.get('procedure_kind') or 'unknown'} · {doc.get('status') or 'candidate'} · {evidence} · "
            f"{count} source{'' if count == 1 else 's'}]: {procedure_text(doc)}"
        )
    return "\n".join(lines), by_label


# ---------------------------------------------------------------- evidence and status
def procedure_status(doc: Mapping[str, Any]) -> str:
    """``active`` once a procedure has the evidence its kind needs, otherwise ``candidate``.

    An instruction needs the user to have taught or corrected it; a habit, a correction by
    the user or support on two dates; a skill, two successful practices on different dates
    or the user's endorsement. Support "by the user" means a supporting fact the user stated.
    """
    evidence = set(doc.get("evidence") or [])
    by_user = int(doc.get("user_support_count") or 0) > 0
    kind = doc.get("procedure_kind")
    if kind == ProcedureKind.instruction.value:
        active = by_user and bool(evidence & {"taught", "corrected"})
    elif kind == ProcedureKind.habit.value:
        active = ("corrected" in evidence and by_user) or len(doc.get("support_dates") or []) >= 2
    elif kind == ProcedureKind.skill.value:
        active = len(doc.get("practiced_dates") or []) >= 2 or ("endorsed" in evidence and by_user)
    else:
        active = False
    return "active" if active else "candidate"


# Provenance for conflict handling, from the strongest evidence an active procedure has.
_TRUST_BY_EVIDENCE = (
    ("taught", "explicit_user_instruction", "high"),
    ("endorsed", "explicit_user_instruction", "high"),
    ("corrected", "observed_user_preference", "medium"),
    ("repeated", "observed_user_preference", "medium"),
    ("practiced", "episode_distillation", "medium"),
)


def _trust(evidence: Iterable[str], status: str) -> tuple[str, str]:
    """``(source_kind, source_authority)`` from the strongest evidence of an active procedure;
    a candidate's are unverified."""
    if status == "active":
        labels = set(evidence)
        for label, source_kind, authority in _TRUST_BY_EVIDENCE:
            if label in labels:
                return source_kind, authority
    return "agent_inference", "low"


def with_support(
    doc: Mapping[str, Any], sources: Sequence[Mapping[str, Any]], evidence: str
) -> Optional[dict[str, Any]]:
    """``doc`` with ``sources`` and ``evidence`` added to its support and its status re-derived;
    ``None`` when every source already supports it. An active procedure never becomes a candidate.

    ``taught`` or ``corrected`` evidence is recorded only when a new source can teach
    (``can_teach``), so a preference or a one-task request the model read as an instruction
    leaves the procedure a candidate."""
    fact_ids = set(doc.get("source_fact_ids") or [])
    episode_ids = set(doc.get("source_episodic_ids") or [])
    new: list[Mapping[str, Any]] = []
    for source in sources:
        source_id = source.get("id")
        known = episode_ids if source.get("type") == "episodic" else fact_ids
        if isinstance(source_id, str) and source_id and source_id not in known:
            known.add(source_id)
            new.append(source)
    if not new:
        return None
    supported = evidence not in TEACHING_EVIDENCE or any(can_teach(source) for source in new)
    out = dict(doc)
    out["source_fact_ids"] = sorted(fact_ids)
    out["source_episodic_ids"] = sorted(episode_ids)
    user_facts = [s for s in new if s.get("type") != "episodic" and source_speaker(s) == "user"]
    good_episodes = [s for s in new if s.get("type") == "episodic" and _outcome_status(s) in GOOD_OUTCOMES]
    # Support dates count the user's own statements and episodes that went well, not what the
    # assistant said or episodes that failed, so a habit cannot become active on assistant-only evidence.
    dates = {source_date(s) for s in [*user_facts, *good_episodes]} - {"unknown"}
    out["support_dates"] = sorted({*(doc.get("support_dates") or []), *dates})
    out["user_support_count"] = int(doc.get("user_support_count") or 0) + len(user_facts)
    practiced = {source_date(s) for s in good_episodes} - {"unknown"}
    out["practiced_dates"] = sorted({*(doc.get("practiced_dates") or []), *practiced})
    out["evidence"] = sorted({*(doc.get("evidence") or []), *([evidence] if supported else [])})
    out["status"] = "active" if doc.get("status") == "active" else procedure_status(out)
    out["source_kind"], out["source_authority"] = _trust(out["evidence"], out["status"])
    return out


# ---------------------------------------------------------------- answers
@dataclass(frozen=True)
class ProcedureAnswer:
    """One usable answer from the model: what to do, with which sources, to which procedure."""

    action: str  # add, update or reinforce
    kind: str
    evidence: str
    name: str
    sources: tuple[dict[str, Any], ...]
    target: Optional[dict[str, Any]]
    item: Mapping[str, Any]


def plan_answer(
    item: Any, labels: Mapping[str, dict[str, Any]], existing_by_label: Mapping[str, dict[str, Any]]
) -> Optional[ProcedureAnswer]:
    """Resolve an answer's labels, or ``None`` when its kind, evidence or name is unusable.

    An ``update`` or ``reinforce`` naming no shown procedure is treated as an ``add``.
    """
    if not isinstance(item, Mapping):
        return None
    kind, evidence, name = item.get("kind"), item.get("evidence"), item.get("name")
    if (
        kind not in PROCEDURE_KINDS
        or evidence not in PROCEDURE_EVIDENCE
        or not isinstance(name, str)
        or not name.strip()
    ):
        return None
    grounded = item.get("grounded_in")
    names = [grounded] if isinstance(grounded, str) else [g for g in grounded or [] if isinstance(g, str)]
    sources = tuple({id(labels[g]): labels[g] for g in names if g in labels}.values())
    action = item.get("action")
    target = existing_by_label.get(item.get("updates")) if action in ("update", "reinforce") else None
    return ProcedureAnswer(
        action=action if target is not None else "add",
        kind=kind,
        evidence=evidence,
        name=name.strip(),
        sources=sources,
        target=target,
        item=item,
    )


def is_user_correction(answer: ProcedureAnswer) -> bool:
    """Whether an answer records the user correcting the assistant, in a fact that can teach (``can_teach``)."""
    return answer.evidence == "corrected" and any(can_teach(source) for source in answer.sources)


def retirable_facts(procedure: Mapping[str, Any], sources: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """The user's directives and corrections among ``sources`` that ``procedure`` now holds in their place.

    Only an active situational procedure (``applies: when``) takes their place, because it is found
    by matching the task at hand; a standing rule reaches the agent only through
    ``build_procedural_context``, so the facts behind it stay searchable. The procedure must cite them.
    """
    if procedure.get("status") != "active" or procedure.get("applies") != "when" or not procedure.get("id"):
        return []
    cited = set(procedure.get("source_fact_ids") or [])
    return [
        dict(source)
        for source in sources
        if source.get("type") != "episodic"
        and source.get("id") in cited
        and source.get("assertion") in USER_ONLY_ASSERTIONS
        and source_speaker(source) == "user"
        and not source.get("superseded_by")
    ]


def procedure_id(user_id: str, kind: str, name: str) -> str:
    return "proc_" + hashlib.sha256(f"{user_id}|{kind}|{name.strip().lower()}".encode()).hexdigest()[:32]


_CONTENT_LISTS = (
    "activation_conditions",
    "preconditions",
    "steps",
    "success_conditions",
    "failure_conditions",
    "safety_constraints",
)


def _content_fields(item: Mapping[str, Any], name: str) -> dict[str, Any]:
    """The fields an answer writes: name, rule text, conditions, steps and validity."""
    summary = str(item.get("summary") or "").strip() or name
    retrieval_text = str(item.get("retrieval_text") or "").strip() or summary
    fields: dict[str, Any] = {
        "name": name,
        "summary": summary,
        "content": summary,
        "retrieval_text": retrieval_text,
        "applies": "always" if item.get("applies") == "always" else "when",
        "valid_until": item.get("valid_until") if isinstance(item.get("valid_until"), str) else None,
    }
    for key in _CONTENT_LISTS:
        value = item.get(key)
        fields[key] = value if isinstance(value, list) else []
    return fields


def new_procedure(answer: ProcedureAnswer, *, user_id: str, now: str) -> dict[str, Any]:
    """A new candidate procedure from an ``add`` answer, with no support yet."""
    return {
        "id": procedure_id(user_id, answer.kind, answer.name),
        "user_id": user_id,
        "thread_id": PROCEDURAL_THREAD_ID,
        "type": "procedural",
        "role": "system",
        "tags": ["sys:procedural", "sys:auto-extracted"],
        "created_at": now,
        "updated_at": now,
        **_content_fields(answer.item, answer.name),
        "procedure_kind": answer.kind,
        "status": "candidate",
        "utility_score": clamp_unit_interval(answer.item.get("confidence"), 0.5),
        "source_fact_ids": [],
        "source_episodic_ids": [],
        "source_turn_ids": [],
        "evidence": [],
        "support_dates": [],
        "user_support_count": 0,
        "practiced_dates": [],
        "version": 1,
        "metadata": {},
    }


def revised_procedure(target: Mapping[str, Any], answer: ProcedureAnswer, *, now: str) -> Optional[dict[str, Any]]:
    """``target`` rewritten by an ``update`` answer, in place: the same ID and support, the next
    version and no embedding (the caller re-embeds). ``None`` when the rewrite drops a number or
    negation the procedure stated, unless it is the user's correction."""
    revised = {**target, **_content_fields(answer.item, answer.name), "updated_at": now}
    # A user's correction is meant to change the rule's numbers or negations; any other rewrite must keep them.
    if not is_user_correction(answer) and not specifics_preserved(
        procedure_text(target), procedure_text(revised), names=False
    ):
        return None
    if target.get("procedure_kind") not in PROCEDURE_KINDS:
        revised["procedure_kind"] = answer.kind
    try:
        revised["version"] = int(target.get("version") or 1) + 1
    except (TypeError, ValueError):
        revised["version"] = 2
    revised["embedding"] = None
    return revised


def reinforced_procedure(target: Mapping[str, Any], answer: ProcedureAnswer, *, now: str) -> dict[str, Any]:
    """``target`` unchanged except for its timestamp; the caller adds the answer's support."""
    reinforced = {**target, "updated_at": now}
    if target.get("procedure_kind") not in PROCEDURE_KINDS:
        reinforced["procedure_kind"] = answer.kind
    return reinforced


# ---------------------------------------------------------------- cursor
def advanced_procedure_cursor(
    existing: Optional[Mapping[str, Any]],
    facts_read: Sequence[Mapping[str, Any]],
    episodes_read: Sequence[Mapping[str, Any]],
) -> Optional[tuple[dict[str, Any], dict[str, Any]]]:
    """The fact and episode positions after the sources a run read; ``None`` if it read none."""
    if not facts_read and not episodes_read:
        return None
    facts, episodes = _positions(existing)
    return (
        next_position(facts, facts_read) or facts,
        next_position(episodes, episodes_read) or episodes,
    )


def procedure_cursor_body(
    existing: Optional[Mapping[str, Any]],
    scope: ProcedureScope,
    facts: Mapping[str, Any],
    episodes: Mapping[str, Any],
) -> dict[str, Any]:
    body = without_system_fields(existing)
    body.update(
        {
            "id": PROCEDURE_CURSOR_ID,
            "type": PROCEDURE_CURSOR_TYPE,
            **scope.fields,
            "facts": {"last_ts": facts["ts"], "ids_at_last_ts": list(facts["ids"])},
            "episodes": {"last_ts": episodes["ts"], "ids_at_last_ts": list(episodes["ids"])},
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }
    )
    return body


# ---------------------------------------------------------------- the prompt block
def utc_today() -> str:
    return datetime.now(timezone.utc).date().isoformat()


def in_force(doc: Mapping[str, Any], today: str) -> bool:
    """Whether a procedure is not past its ``valid_until`` date."""
    until = doc.get("valid_until")
    return not isinstance(until, str) or not until or until[:10] >= today


def rendered_rule(doc: Mapping[str, Any]) -> str:
    """How the agent sees a procedure: its rule, prefixed with its conditions when situational."""
    summary = str(doc.get("summary") or doc.get("name") or doc.get("content") or "").strip()
    conditions = [str(c).strip() for c in doc.get("activation_conditions") or [] if str(c).strip()]
    if doc.get("applies") == "always" or not conditions:
        return summary
    return f"When {'; '.join(conditions)}: {summary}"


_AUTHORITY = {"taught": 0, "corrected": 1, "repeated": 2, "endorsed": 3, "practiced": 4}


def standing_rules(procedures: Iterable[Mapping[str, Any]], today: str) -> tuple[list[dict[str, Any]], int]:
    """The always-on procedures in force, best first and capped, and how many the cap left out.

    Taught rules come before corrected, then repeated ones; within that, the most recently
    updated first. The cap is ``STANDING_RULES_MAX`` rules in ``STANDING_RULES_MAX_BYTES``.
    """
    rules = [
        dict(doc)
        for doc in procedures
        if doc.get("type") == "procedural"
        and doc.get("status") == "active"
        and doc.get("applies") == "always"
        and not doc.get("superseded_by")
        and in_force(doc, today)
        and rendered_rule(doc)
    ]
    rules.sort(key=lambda doc: str(doc.get("updated_at") or doc.get("created_at") or ""), reverse=True)
    rules.sort(
        key=lambda doc: min(
            (_AUTHORITY.get(str(e), len(_AUTHORITY)) for e in doc.get("evidence") or []), default=len(_AUTHORITY)
        )
    )
    kept: list[dict[str, Any]] = []
    used = 0
    for doc in rules:
        if len(kept) == STANDING_RULES_MAX:
            break
        size = len(f"- {rendered_rule(doc)}\n".encode())
        if used + size > STANDING_RULES_MAX_BYTES:
            continue  # one oversized rule must not empty the block; later, smaller ones may still fit
        kept.append(doc)
        used += size
    return kept, len(rules) - len(kept)


def procedural_context(standing: Sequence[Mapping[str, Any]], situational: Sequence[Mapping[str, Any]]) -> str:
    """The prompt block for standing rules and, for a task, the situational rules that match it."""
    if not standing and not situational:
        return ""
    lines = [STANDING_HEADER]
    lines.extend(f"- {rendered_rule(doc)}" for doc in standing)
    if situational:
        lines.extend(["", SITUATIONAL_HEADER])
        lines.extend(f"- {rendered_rule(doc)}" for doc in situational)
    return "\n".join(lines)
