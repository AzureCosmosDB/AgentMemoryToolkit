"""Duplicate checks for the dedup pass, shared by the sync and async pipelines.

``consolidate_memories`` checks each fact written since the user's last pass against its
nearest neighbours:

1. **Seeds**: the user's active facts written after the dedup cursor, oldest first.
2. **Candidates**: a seed's nearest active facts whose cosine similarity reaches
   ``DEDUP_SIMILARITY_FLOOR``.
3. **Verdicts**: ``dedup_verify.prompty`` labels each candidate's relation to the seed
   and lists the details each side states that the other lacks. The seed and the candidates
   reach it as JSON whose strings cannot end a line (``json_text``), so that no text in a memory can
   start a line of the prompt.
4. **Folds**: a fact is retired into another only when the two state the same claim or
   the other contains it, the model lists nothing it would lose, and
   :func:`specifics_preserved` agrees. The survivor keeps its content verbatim.

Contradictions are never retired here: ``reconcile_memories`` resolves them in its own
pass after this one. A fact that has absorbed others is never retired, and a user's
statement is never retired into an assistant's. :mod:`azure.cosmos.agent_memory.services._combine`
also combines facts about one specific thing in the same pass.

The module also holds what the other passes share with this one: the active-document filter, the
``_ts`` cursor positions and the query window after one, agent-source detection, the numbers, names
and negation checks, and the ranking of a seed's neighbours by similarity.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Mapping, Optional, Sequence

from azure.cosmos.agent_memory._utils import cosine_similarity
from azure.cosmos.agent_memory.exceptions import ValidationError
from azure.cosmos.agent_memory.models import AGENT_FACT_TAG, MAX_TAGS_PER_RECORD, USER_ONLY_ASSERTIONS
from azure.cosmos.agent_memory.services._pipeline_helpers import is_real_number, json_text
from azure.cosmos.agent_memory.thresholds import DEDUP_MAX_NEIGHBORS, DEDUP_SIMILARITY_FLOOR

DEDUP_PROMPT = "dedup_verify.prompty"
DEDUP_CURSOR_TYPE = "dedup_cursor"
# The cursor is user-wide, so it lives under a sentinel thread id like the user summary does.
DEDUP_CURSOR_THREAD_ID = "__dedup__"
# A fold is one write per retired fact after its survivor's, and the survivor lists them all in
# ``merged_from``: bound both for one pass.
MAX_FOLD_LOSERS = 50

ACTIVE_FILTER = "(NOT IS_DEFINED(c.superseded_by) OR IS_NULL(c.superseded_by))"

SAME_CLAIM = "same_claim"
SEED_CONTAINS = "seed_contains_candidate"
CANDIDATE_CONTAINS = "candidate_contains_seed"
CONFLICT_RELATIONS = frozenset({"update", "inconsistent"})
RELATIONS = frozenset(
    {SAME_CLAIM, SEED_CONTAINS, CANDIDATE_CONTAINS, "partial_overlap", "related", "unrelated", *CONFLICT_RELATIONS}
)


@dataclass(frozen=True)
class ModelCall:
    """A prompt to run; the pipeline sends back its parsed JSON answer (``{}`` when unusable)."""

    prompt: str
    inputs: dict[str, str]


def validate_consolidation_args(user_id: str, n: int, memory_type: str) -> None:
    """Check the arguments ``reconcile_memories`` and ``consolidate_memories`` share."""
    if not user_id:
        raise ValidationError("user_id is required")
    if not isinstance(n, int) or isinstance(n, bool) or n < 1:
        raise ValidationError(f"n must be a positive integer, got {n!r}")
    if n > 500:
        raise ValidationError(f"n must be <= 500 to bound prompt size and LLM cost, got {n}")
    if memory_type not in {"fact", "episodic", "procedural"}:
        raise ValidationError(f"memory_type must be one of fact, episodic, procedural; got {memory_type!r}")


# ---------------------------------------------------------------- cursor
def without_system_fields(doc: Optional[Mapping[str, Any]]) -> dict[str, Any]:
    """``doc`` without its Cosmos DB system fields (``_etag``, ``_ts``, ...), as the base of a body to write."""
    return {key: value for key, value in (doc or {}).items() if not key.startswith("_")}


def cursor_position(value: Any) -> dict[str, Any]:
    """A cursor position as ``{"ts": int, "ids": [...]}``: the server write second (``_ts``) of the
    last memory checked and the memories checked within that second. Anything unrecognized is the
    empty position, which starts from the user's first memory."""
    if not isinstance(value, Mapping):
        return {"ts": 0, "ids": []}
    return {"ts": int(value.get("ts") or 0), "ids": sorted({str(i) for i in value.get("ids") or [] if i})}


def further_position(first: Any, second: Any) -> dict[str, Any]:
    """The further of two positions; within one write second their ids combine."""
    a, b = cursor_position(first), cursor_position(second)
    if a["ts"] != b["ts"]:
        return a if a["ts"] > b["ts"] else b
    return {"ts": a["ts"], "ids": sorted(set(a["ids"]) | set(b["ids"]))}


def checkpoint_position(doc: Optional[Mapping[str, Any]]) -> dict[str, Any]:
    """The ``{ts, ids}`` position a stored cursor document records."""
    stored = doc or {}
    return cursor_position({"ts": stored.get("last_ts"), "ids": stored.get("ids_at_last_ts")})


def next_position(position: Any, checked: Sequence[Mapping[str, Any]]) -> Optional[dict[str, Any]]:
    """The position after ``checked``, a prefix of the memories read in ``_ts`` order after ``position``;
    ``None`` if empty."""
    if not checked:
        return None
    last_ts = int(checked[-1].get("_ts") or 0)
    ids = [str(doc["id"]) for doc in checked if doc.get("id") and int(doc.get("_ts") or 0) == last_ts]
    return further_position(position, {"ts": last_ts, "ids": ids})


def next_dedup_cursor(
    existing: Optional[Mapping[str, Any]], checked: Sequence[Mapping[str, Any]]
) -> Optional[dict[str, Any]]:
    """The cursor position after ``checked``, a prefix of the seeds in ``_ts`` order; ``None`` if empty."""
    return next_position(checkpoint_position(existing), checked)


def dedup_cursor_id(user_id: str, memory_type: str) -> str:
    return f"dedup_cursor_{user_id}_{memory_type}"


def dedup_cursor_body(
    existing: Optional[Mapping[str, Any]], user_id: str, memory_type: str, position: Mapping[str, Any]
) -> dict[str, Any]:
    body = without_system_fields(existing)
    body.update(
        {
            "id": dedup_cursor_id(user_id, memory_type),
            "type": DEDUP_CURSOR_TYPE,
            "user_id": user_id,
            "thread_id": DEDUP_CURSOR_THREAD_ID,
            "last_ts": position["ts"],
            "ids_at_last_ts": list(position["ids"]),
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }
    )
    return body


# ---------------------------------------------------------------- queries
# The memories written after a cursor position, oldest first. ``_ts`` is always indexed, so the window needs no
# composite index.
AFTER_POSITION = (
    " AND c._ts >= @after_ts AND NOT (c._ts = @after_ts AND ARRAY_CONTAINS(@after_ids, c.id)) ORDER BY c._ts ASC"
)


def position_parameters(position: Mapping[str, Any]) -> list[dict[str, Any]]:
    """The query parameters ``AFTER_POSITION`` reads."""
    return [{"name": "@after_ts", "value": position["ts"]}, {"name": "@after_ids", "value": position["ids"]}]


def unchecked_memories_query(
    user_id: str, memory_type: str, existing_cursor: Optional[Mapping[str, Any]], top: int
) -> tuple[str, list[dict[str, Any]]]:
    """The user's active ``memory_type`` memories written after the cursor, oldest first."""
    query = (
        f"SELECT TOP {top} * FROM c WHERE c.user_id = @user_id AND c.type = @memory_type AND {ACTIVE_FILTER}"
        + AFTER_POSITION
    )
    parameters = [
        {"name": "@user_id", "value": user_id},
        {"name": "@memory_type", "value": memory_type},
        *position_parameters(checkpoint_position(existing_cursor)),
    ]
    return query, parameters


def neighbor_query(user_id: str, memory_type: str, seed: Mapping[str, Any]) -> tuple[str, list[dict[str, Any]]]:
    """The seed's nearest active memories of the same type for the same user."""
    query = (
        f"SELECT TOP {DEDUP_MAX_NEIGHBORS} * FROM c WHERE c.user_id = @user_id AND c.type = @memory_type "
        f"AND {ACTIVE_FILTER} AND c.id != @seed_id AND IS_DEFINED(c.embedding) "
        "ORDER BY VectorDistance(c.embedding, @embedding)"
    )
    parameters = [
        {"name": "@user_id", "value": user_id},
        {"name": "@memory_type", "value": memory_type},
        {"name": "@seed_id", "value": seed["id"]},
        {"name": "@embedding", "value": seed["embedding"]},
    ]
    return query, parameters


# ---------------------------------------------------------------- candidates and verdicts
def is_agent_sourced(doc: Mapping[str, Any]) -> bool:
    """True when ``doc`` records agent content rather than the user's own words, by its tag or
    by ``metadata.source``."""
    metadata = doc.get("metadata")
    return AGENT_FACT_TAG in (doc.get("tags") or []) or (
        isinstance(metadata, Mapping) and metadata.get("source") == "agent"
    )


def ranked_candidates(
    seed: Mapping[str, Any],
    neighbors: Sequence[Mapping[str, Any]],
    accept: Callable[[Mapping[str, Any], float], bool],
) -> list[tuple[dict[str, Any], float]]:
    """The ``neighbors`` of ``seed`` that ``accept`` (given a neighbour and its similarity to the seed),
    most similar first and at most ``DEDUP_MAX_NEIGHBORS``.

    Similarity is computed here, so the result does not depend on the container's vector
    distance function.
    """
    scored = [
        (dict(doc), cosine_similarity(seed.get("embedding") or [], doc.get("embedding") or []))
        for doc in neighbors
        if doc.get("id") != seed.get("id")
    ]
    close = [(doc, score) for doc, score in scored if accept(doc, score)]
    close.sort(key=lambda item: item[1], reverse=True)
    return close[:DEDUP_MAX_NEIGHBORS]


def close_candidates(
    seed: Mapping[str, Any], neighbors: Sequence[Mapping[str, Any]]
) -> list[tuple[dict[str, Any], float]]:
    """Neighbours at or above the similarity floor, most similar first."""
    return ranked_candidates(seed, neighbors, lambda _doc, score: score >= DEDUP_SIMILARITY_FLOOR)


def memory_text(doc: Mapping[str, Any]) -> str:
    """The text a fact is compared by: its content."""
    return str(doc.get("content") or "")


def memory_json(doc: Mapping[str, Any], label: Optional[str] = None) -> dict[str, Any]:
    entry: dict[str, Any] = {"label": label} if label else {}
    entry["said"] = str(doc.get("created_at") or "unknown")[:10]
    entry["by"] = "the assistant" if is_agent_sourced(doc) else "the user"
    entry["text"] = memory_text(doc)
    return entry


def verify_inputs(seed: Mapping[str, Any], candidates: Sequence[tuple[Mapping[str, Any], float]]) -> dict[str, str]:
    """Prompt inputs: candidates are labeled ``C1..Cn`` so the model never handles IDs.

    The seed and the candidates are each JSON whose strings cannot end a line (``json_text``), so that no text in a
    memory can start a line of the prompt."""
    return {
        "seed": json_text(memory_json(seed)),
        "candidates": json_text(
            [memory_json(doc, f"C{i}") for i, (doc, _) in enumerate(candidates, start=1)], indent=1
        ),
    }


@dataclass(frozen=True)
class Verdict:
    candidate: dict[str, Any]
    similarity: float
    relation: str
    # Details the seed states that the candidate lacks, and the reverse. ``None`` when the
    # model's list was unusable, which blocks the fold that would lose those details.
    seed_missing: Optional[tuple[str, ...]]
    candidate_missing: Optional[tuple[str, ...]]


def _details(value: Any) -> Optional[tuple[str, ...]]:
    if not isinstance(value, list):
        return None
    return tuple(str(item).strip() for item in value if str(item).strip())


def parse_verdicts(parsed: Mapping[str, Any], candidates: Sequence[tuple[Mapping[str, Any], float]]) -> list[Verdict]:
    """Map the model's labeled verdicts back to candidates; unknown or repeated labels are dropped."""
    by_label = {f"C{i}": (dict(doc), score) for i, (doc, score) in enumerate(candidates, start=1)}
    verdicts: list[Verdict] = []
    seen: set[str] = set()
    for item in parsed.get("candidates") or []:
        if not isinstance(item, Mapping):
            continue
        label, relation = item.get("label"), item.get("relation")
        if label not in by_label or label in seen or relation not in RELATIONS:
            continue
        seen.add(label)
        doc, score = by_label[label]
        verdicts.append(
            Verdict(
                candidate=doc,
                similarity=score,
                relation=relation,
                seed_missing=_details(item.get("seed_details_missing_from_candidate")),
                candidate_missing=_details(item.get("candidate_details_missing_from_seed")),
            )
        )
    return verdicts


# ---------------------------------------------------------------- deterministic veto
_NUMBER = re.compile(r"\d[\d.,:/-]*\d|\d")
_NEGATION = re.compile(r"\b(?:not|never|no longer|without|none|nobody|nothing)\b|n't\b", re.IGNORECASE)
_NAME = re.compile(r"\b[A-Z][a-zA-Z'\u2019-]{2,}\b")
_POSSESSIVE = re.compile(r"['\u2019]s$")
_MONTHS = "January February March April May June July August September October November December".split()
_NOT_NAMES = (
    frozenset(
        "The This That These Those When Since After Before During While If For And But Or So Then User Users "
        "They Their Them His Her She He It Its We Our You Your On In At By With From To As Also Each Every Some "
        "Any Over Into About Monday Tuesday Wednesday Thursday Friday Saturday Sunday".split()
    )
    | frozenset(_MONTHS)
    | frozenset(month[:3] for month in _MONTHS)
)


def _names(text: str) -> set[str]:
    return {_POSSESSIVE.sub("", word) for word in _NAME.findall(text)} - _NOT_NAMES


def numbers_kept(retired: str, survivor: str) -> bool:
    """Whether ``survivor`` states every number ``retired`` does."""
    return set(_NUMBER.findall(retired)) <= set(_NUMBER.findall(survivor))


def names_kept(retired: str, survivor: str) -> bool:
    """Whether ``survivor`` mentions every name ``retired`` does."""
    survivor_lower = survivor.lower()
    return all(name.lower() in survivor_lower for name in _names(retired))


def has_negation(text: str) -> bool:
    return bool(_NEGATION.search(text))


def specifics_preserved(retired: str, survivor: str, *, names: bool = True) -> bool:
    """Whether ``survivor`` keeps the numbers and names ``retired`` states, and its negation.

    A cheap veto behind the model's check, not a substitute for it: it errs toward keeping both
    facts (a reworded date or a spelled-out number blocks a safe fold) and cannot see a lost
    qualifier, so a fold still needs the model's check to list nothing missing. Procedures skip
    the name check (``names=False``): their imperative steps ("Ask", "Verify") read as names, so
    rewording a step would block a safe revision.
    """
    if not numbers_kept(retired, survivor) or (names and not names_kept(retired, survivor)):
        return False
    return has_negation(retired) == has_negation(survivor)


# ---------------------------------------------------------------- fold planning
def is_absorber(doc: Mapping[str, Any]) -> bool:
    """A fact that has absorbed others: retiring it would chain merges."""
    return bool(doc.get("merged_from"))


def _keeps_assertion(retired: Mapping[str, Any], survivor: Mapping[str, Any]) -> bool:
    """Whether retiring ``retired`` into ``survivor`` keeps how it was asserted. A suggestion and the
    later decision are a transition, not a duplicate; and a directive or correction folded into a
    fact that is neither would no longer teach the procedure pass. A fact extracted before
    assertions were recorded has none, so only a directive or correction refuses it as survivor."""
    mine, theirs = retired.get("assertion"), survivor.get("assertion")
    if not mine:
        return True
    if not theirs:
        return mine not in USER_ONLY_ASSERTIONS
    return mine == theirs


def may_retire(retired: Mapping[str, Any], survivor: Mapping[str, Any]) -> bool:
    if is_absorber(retired) or not _keeps_assertion(retired, survivor):
        return False
    # A user's statement stays correctable: it is never retired into an assistant fact.
    return is_agent_sourced(retired) or not is_agent_sourced(survivor)


def _rank(doc: Mapping[str, Any]) -> tuple[str, float, float]:
    confidence = doc.get("confidence") if is_real_number(doc.get("confidence")) else float("-inf")
    salience = doc.get("salience") if is_real_number(doc.get("salience")) else float("-inf")
    return (str(doc.get("created_at") or ""), confidence, salience)


def preferred_survivor(first: dict[str, Any], second: dict[str, Any]) -> dict[str, Any]:
    """For two statements of the same claim: the most recent, then the more confident, then the
    more salient, then the lexicographically first ID."""
    ordered = sorted([first, second], key=lambda doc: str(doc.get("id")))
    ordered.sort(key=_rank, reverse=True)
    return ordered[0]


@dataclass
class FoldPlan:
    survivors: dict[str, dict[str, Any]] = field(default_factory=dict)
    retired: dict[str, list[dict[str, Any]]] = field(default_factory=dict)  # survivor ID -> retired facts
    vetoed: int = 0

    def retired_ids(self) -> set[str]:
        return {doc["id"] for docs in self.retired.values() for doc in docs}


def _fold(seed: dict[str, Any], verdict: Verdict) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    """The ``(retired, survivor)`` pairs the verdict permits, preferred first."""
    candidate = verdict.candidate
    seed_safe = verdict.seed_missing == ()  # retiring the seed loses nothing
    candidate_safe = verdict.candidate_missing == ()
    if verdict.relation == SAME_CLAIM:
        options = []
        if candidate_safe:
            options.append((candidate, seed))
        if seed_safe:
            options.append((seed, candidate))
        survivor_id = preferred_survivor(seed, candidate)["id"]
        options.sort(key=lambda pair: pair[1]["id"] != survivor_id)
        return options
    if verdict.relation == SEED_CONTAINS and candidate_safe:
        return [(candidate, seed)]
    if verdict.relation == CANDIDATE_CONTAINS and seed_safe:
        return [(seed, candidate)]
    return []


# A seed is first offered to candidates that contain it, so statements of one claim converge
# on the most complete one rather than on whichever paraphrase happened to be compared first.
_FOLD_ORDER = {CANDIDATE_CONTAINS: 0, SAME_CLAIM: 1, SEED_CONTAINS: 2}


def plan_folds(checked: Sequence[tuple[dict[str, Any], Sequence[Verdict]]]) -> FoldPlan:
    """Choose folds from each seed's verdicts: candidates that contain the seed first, then
    statements of the same claim, then candidates the seed contains; most similar first.

    A fact is retired at most once, never after it absorbed another in this pass, and
    never into a fact this pass already retired.
    """
    plan = FoldPlan()
    gone: set[str] = set()
    for seed, verdicts in checked:
        ordered = sorted(verdicts, key=lambda v: (_FOLD_ORDER.get(v.relation, 3), -v.similarity))
        for verdict in ordered:
            if seed["id"] in gone:
                break
            for retired, survivor in _fold(seed, verdict):
                if retired["id"] in gone or survivor["id"] in gone or retired["id"] in plan.survivors:
                    continue
                if not may_retire(retired, survivor):
                    continue
                if not specifics_preserved(memory_text(retired), memory_text(survivor)):
                    plan.vetoed += 1
                    continue
                losers = plan.retired.setdefault(survivor["id"], [])
                if len(losers) >= MAX_FOLD_LOSERS:
                    continue
                plan.survivors.setdefault(survivor["id"], survivor)
                losers.append(retired)
                gone.add(retired["id"])
                break
    return plan


def survivor_patch(survivor: dict[str, Any], losers: list[dict[str, Any]]) -> dict[str, Any]:
    """The survivor of a fold, content kept verbatim, with its duplicates' provenance.

    Unions tags (and source turns, on facts that record them), takes the highest salience and
    confidence (corroboration does not lower trust), and lists the duplicates in ``merged_from``.
    """
    docs = [survivor, *losers]
    tags = {tag for doc in docs for tag in (doc.get("tags") or [])}
    if not is_agent_sourced(survivor):
        # A user fact that absorbs an agent paraphrase must stay open to contradiction.
        tags.discard(AGENT_FACT_TAG)
    # The model refuses a record with more tags than its cap: a survivor keeps its own and takes what fits.
    own = sorted(tags & set(survivor.get("tags") or []))
    salience = [doc.get("salience") for doc in docs if is_real_number(doc.get("salience"))]
    confidence = [doc.get("confidence") for doc in docs if is_real_number(doc.get("confidence"))]
    patch = without_system_fields(survivor)
    patch["tags"] = sorted([*own, *sorted(tags - set(own))][:MAX_TAGS_PER_RECORD])
    if any("source_turn_ids" in doc for doc in docs):
        patch["source_turn_ids"] = sorted({turn for doc in docs for turn in (doc.get("source_turn_ids") or [])})
    if salience:
        patch["salience"] = max(salience)
    if confidence:
        patch["confidence"] = max(confidence)
    patch["merged_from"] = sorted(set(survivor.get("merged_from") or []) | {loser["id"] for loser in losers})
    patch["updated_at"] = datetime.now(timezone.utc).isoformat()
    return patch


def dedup_result(
    *,
    checked: int,
    retired_seeds: int,
    merged: int,
    compared: int,
    vetoed: int,
    combined: int = 0,
    synthesized: int = 0,
    combine_groups: int = 0,
    combine_failed: int = 0,
) -> dict[str, int]:
    """A pass's counts. ``kept`` is the facts checked that stay active; ``contradicted`` stays 0, because
    this pass never retires contradictions. ``synthesized`` counts the combined facts written and
    ``combined`` the facts they retired; ``combine_groups`` the groups tried and ``combine_failed``
    those declined or left as they were (see :mod:`azure.cosmos.agent_memory.services._combine`)."""
    return {
        "kept": checked - retired_seeds,
        "merged": merged,
        "contradicted": 0,
        "checked": checked,
        "compared": compared,
        "vetoed": vetoed,
        "combined": combined,
        "synthesized": synthesized,
        "combine_groups": combine_groups,
        "combine_failed": combine_failed,
    }
