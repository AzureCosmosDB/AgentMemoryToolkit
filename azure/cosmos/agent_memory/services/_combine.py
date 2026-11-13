"""Combining facts about one specific thing, shared by the sync and async dedup passes.

Dedup (:mod:`azure.cosmos.agent_memory.services._dedup`) retires a fact only when another states
everything it states, and it compares only neighbours at or above ``DEDUP_SIMILARITY_FLOOR``. Facts
about the same specific thing that each hold details the other lacks therefore all stay, and several
of them can fill a short search result. So the dedup pass on facts also:

1. **Compares further**: a fact's nearest neighbours down to ``COMBINE_SIMILARITY_FLOOR`` are judged
   in the same ``dedup_verify.prompty`` call when the two could be combined (:func:`combinable_pair`).
   Folds still need the dedup floor, so they are decided exactly as without combining.
2. **Groups** (:func:`plan_combine_groups`): pairs judged the same claim, contained or overlapping
   become groups of up to ``COMBINE_MAX_SOURCES``, the fact with the most such partners first. A fact
   joins one group; facts a fold touched in this pass wait for a later one.
3. **Writes and checks** (:func:`combine_group`): ``combine_memories.prompty`` picks the members about
   one specific thing (at least two) and writes one fact stating every detail of each.
   ``dedup_verify.prompty`` must then find that it contains each member it combines with nothing
   missing, and :func:`kept_specifics` that it keeps each member's numbers, names and negation, in at
   most ``COMBINE_MAX_CHARS`` characters. One retry lists what was missing; after that the members stay
   as they are.
4. **Applies** (:func:`combined_record`): the combined fact is created (tagged ``sys:combined``, with
   ``merged_from`` naming its members), then each member is retired into it
   (``supersede_reason="combined"``). Its ID derives from its user and members, so writing it twice (after
   a crash, or from a concurrent pass) cannot make two records.

A combined fact is never combined again, and dedup never retires it (it has absorbed others).
"""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Generator, Mapping, Optional, Sequence

from azure.cosmos.agent_memory._utils import compute_content_hash
from azure.cosmos.agent_memory.logging import describe_exception
from azure.cosmos.agent_memory.models import USER_ONLY_ASSERTIONS
from azure.cosmos.agent_memory.services._dedup import (
    CANDIDATE_CONTAINS,
    DEDUP_PROMPT,
    SAME_CLAIM,
    SEED_CONTAINS,
    ModelCall,
    Verdict,
    has_negation,
    is_agent_sourced,
    memory_json,
    memory_text,
    names_kept,
    numbers_kept,
    parse_verdicts,
    ranked_candidates,
    survivor_patch,
    verify_inputs,
)
from azure.cosmos.agent_memory.services._pipeline_helpers import ID_SEED_SEP, is_retryable_llm_error
from azure.cosmos.agent_memory.thresholds import (
    COMBINE_MAX_CHARS,
    COMBINE_MAX_SOURCES,
    COMBINE_SIMILARITY_FLOOR,
    DEDUP_SIMILARITY_FLOOR,
)

COMBINE_PROMPT = "combine_memories.prompty"
COMBINED_TAG = "sys:combined"
COMBINE_REASON = "combined"
COMBINABLE = frozenset({SAME_CLAIM, SEED_CONTAINS, CANDIDATE_CONTAINS, "partial_overlap"})
# The check's verdicts that mean the combined fact (its seed) holds a member in full.
_HOLDS_MEMBER = frozenset({SEED_CONTAINS, SAME_CLAIM})
_RETRY_NOTE = (
    "Your previous combined memory left out or changed these details. "
    "Write it again and include every one of them exactly:"
)


# ---------------------------------------------------------------- which facts and pairs
def may_combine(doc: Mapping[str, Any]) -> bool:
    """Whether a fact can be combined with others: a fact that is not a directive or correction (the
    procedure pass learns from those) and was not itself written from other facts."""
    if doc.get("type") != "fact" or (doc.get("assertion") or "") in USER_ONLY_ASSERTIONS:
        return False
    return COMBINED_TAG not in (doc.get("tags") or [])


def combinable_pair(a: Mapping[str, Any], b: Mapping[str, Any]) -> bool:
    """Whether two facts may be combined: each can be, and they were stated by the same speaker and
    asserted the same way (a fact extracted before assertions were recorded has none)."""
    return (
        may_combine(a)
        and may_combine(b)
        and is_agent_sourced(a) == is_agent_sourced(b)
        and (a.get("assertion") or "") == (b.get("assertion") or "")
    )


def comparable_candidates(
    seed: Mapping[str, Any], neighbors: Sequence[Mapping[str, Any]]
) -> list[tuple[dict[str, Any], float]]:
    """The neighbours a combining pass judges, most similar first: those dedup judges (at or above its
    floor), and those down to ``COMBINE_SIMILARITY_FLOOR`` that could be combined with the seed."""
    return ranked_candidates(
        seed,
        neighbors,
        lambda doc, score: (
            score >= DEDUP_SIMILARITY_FLOOR or (score >= COMBINE_SIMILARITY_FLOOR and combinable_pair(seed, doc))
        ),
    )


def fold_verdicts(checked: Sequence[tuple[dict[str, Any], Sequence[Verdict]]]) -> list[tuple[dict[str, Any], list]]:
    """The verdicts folds may act on: pairs at or above the dedup floor, as when nothing is combined."""
    return [
        (seed, [verdict for verdict in verdicts if verdict.similarity >= DEDUP_SIMILARITY_FLOOR])
        for seed, verdicts in checked
    ]


# ---------------------------------------------------------------- groups
def plan_combine_groups(
    checked: Sequence[tuple[dict[str, Any], Sequence[Verdict]]], skip: set[str]
) -> list[list[dict[str, Any]]]:
    """Groups of facts to combine from a pass's verdicts, leaving out the facts in ``skip``.

    Two facts are partners when their pair was judged the same claim, contained or overlapping and
    :func:`combinable_pair` allows it. The fact with the most partners leads a group (then the oldest,
    then the lowest ID) with its most similar partners not yet in a group, up to ``COMBINE_MAX_SOURCES``
    in all.
    """
    docs: dict[str, dict[str, Any]] = {}
    partners: dict[str, dict[str, float]] = defaultdict(dict)
    for seed, verdicts in checked:
        for verdict in verdicts:
            other = verdict.candidate
            if verdict.relation not in COMBINABLE or seed["id"] in skip or other["id"] in skip:
                continue
            if not combinable_pair(seed, other):
                continue
            docs[seed["id"]], docs[other["id"]] = seed, other
            partners[seed["id"]][other["id"]] = partners[other["id"]][seed["id"]] = verdict.similarity
    taken: set[str] = set()
    groups: list[list[dict[str, Any]]] = []
    for lead in sorted(partners, key=lambda key: (-len(partners[key]), str(docs[key].get("created_at") or ""), key)):
        if lead in taken:
            continue
        members = [lead]
        for other, _ in sorted(partners[lead].items(), key=lambda item: -item[1]):
            if len(members) == COMBINE_MAX_SOURCES:
                break
            if other not in taken:
                members.append(other)
        if len(members) > 1:
            taken.update(members)
            groups.append([docs[key] for key in members])
    return groups


def in_seed_order(groups: list[list[dict[str, Any]]], seeds: Sequence[Mapping[str, Any]]) -> list[list[dict[str, Any]]]:
    """``groups`` ordered by their first member among ``seeds`` (the order the pass checked them in), so a
    pass that stops early has tried the groups of the seeds its cursor moves past."""
    position = {seed["id"]: index for index, seed in enumerate(seeds)}
    return sorted(groups, key=lambda group: min(position.get(doc["id"], len(position)) for doc in group))


def settled_seeds(seeds: Sequence[dict[str, Any]], untried: Sequence[Sequence[Mapping[str, Any]]]) -> list[dict]:
    """The seeds before the first member of a group not yet tried: the cursor moves past these only, so a
    later pass checks the rest again."""
    waiting = {doc["id"] for group in untried for doc in group}
    settled: list[dict[str, Any]] = []
    for seed in seeds:
        if seed["id"] in waiting:
            break
        settled.append(seed)
    return settled


# ---------------------------------------------------------------- writing and checking one group
@dataclass(frozen=True)
class CombineDraft:
    """What the writer and the check made of one group: ``combined`` (apply ``members`` and ``text``),
    ``declined`` (the writer found fewer than two about one specific thing) or ``failed`` (the check
    failed twice, or the writer's answers were unusable)."""

    status: str
    members: tuple[dict[str, Any], ...] = ()
    text: str = ""
    missing: tuple[str, ...] = ()


def writer_inputs(docs: Sequence[Mapping[str, Any]], missing: Sequence[str] = ()) -> dict[str, str]:
    """``combine_memories.prompty`` inputs: the facts labeled ``M1..Mn`` (no IDs) and, on a retry, what the
    previous fact left out."""
    memories = json.dumps(
        [memory_json(doc, f"M{index}") for index, doc in enumerate(docs, start=1)], ensure_ascii=False, indent=1
    )
    feedback = "\n\n" + _RETRY_NOTE + "\n" + "\n".join(f"- {item}" for item in missing) if missing else ""
    return {"memories": memories, "feedback": feedback}


def _as_seed(text: str, members: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """The combined fact as the check's seed, said when and by whom its newest member was."""
    newest = max(members, key=lambda doc: str(doc.get("created_at") or ""))
    return {
        "id": "combined",
        "type": "fact",
        "content": text,
        "created_at": newest.get("created_at"),
        "metadata": newest.get("metadata"),
        "tags": newest.get("tags"),
    }


def kept_specifics(member: str, combined: str) -> bool:
    """Whether ``combined`` keeps every number and name ``member`` states, and a negation if ``member`` has one.

    Like dedup's :func:`~azure.cosmos.agent_memory.services._dedup.specifics_preserved`, but a combined fact
    may also hold another member's negation, so only a negation ``member`` states must survive.
    """
    return (
        numbers_kept(member, combined)
        and names_kept(member, combined)
        and (not has_negation(member) or has_negation(combined))
    )


def check_gaps(text: str, members: Sequence[Mapping[str, Any]], verdicts: Sequence[Verdict]) -> list[str]:
    """What ``text`` fails to keep: each member the check did not find held in full (with the details it
    lists), each member whose numbers, names or negation are missing, and the length when over
    ``COMBINE_MAX_CHARS``."""
    by_id = {verdict.candidate["id"]: verdict for verdict in verdicts}
    missing: list[str] = []
    for doc in members:
        verdict = by_id.get(doc["id"])
        stated = memory_text(doc)
        if verdict is None or verdict.relation not in _HOLDS_MEMBER or verdict.candidate_missing != ():
            listed = list((verdict.candidate_missing if verdict else None) or ())
            missing += listed or [f"everything stated in: {stated}"]
        if not kept_specifics(stated, text):
            missing.append(f"every number, name and negation in: {stated}")
    if len(text) > COMBINE_MAX_CHARS:
        missing.append(f"the whole memory in under {COMBINE_MAX_CHARS} characters")
    return missing


def combine_group(docs: Sequence[dict[str, Any]]) -> Generator[ModelCall, Any, CombineDraft]:
    """Write and check one group's combined fact, as a generator of the model calls it needs (the pipeline
    sends back each parsed answer, ``{}`` when unusable): the writer, then the check, and once more with
    what was missing. The retry offers only the members the writer chose."""
    offered = list(docs)
    missing: list[str] = []
    draft = CombineDraft("failed")
    for attempt in (1, 2):
        answer = yield ModelCall(COMBINE_PROMPT, writer_inputs(offered, missing))
        picked = answer.get("combine") if isinstance(answer, Mapping) else None
        if not isinstance(picked, list):
            continue
        labels = {f"M{index}": doc for index, doc in enumerate(offered, start=1)}
        members = [labels[label] for label in dict.fromkeys(x for x in picked if isinstance(x, str)) if label in labels]
        text = " ".join(str(answer.get("memory") or "").split())
        if len(members) < 2 or not text:
            return CombineDraft("declined" if attempt == 1 else "failed")
        pairs = [(doc, 1.0) for doc in members]
        check = yield ModelCall(DEDUP_PROMPT, verify_inputs(_as_seed(text, members), pairs))
        verdicts = parse_verdicts(check, pairs) if isinstance(check, Mapping) else []
        missing = check_gaps(text, members, verdicts)
        if not missing:
            return CombineDraft("combined", tuple(members), text)
        draft = CombineDraft("failed", tuple(members), text, tuple(missing))
        offered = members
    return draft


# ---------------------------------------------------------------- applying
def combined_record(members: Sequence[dict[str, Any]], text: str, *, lineage: Mapping[str, str]) -> dict[str, Any]:
    """The combined fact, without its embedding.

    It takes its newest member's fields (date, thread, category and speaker), every member's tags (and
    source turns, on facts that record them), the highest salience and confidence, and ``merged_from``
    naming the members (and what they had absorbed). Its ID derives from its user and members, not its
    text: a second pass that combines the same group, after a crash or beside a concurrent pass, finds
    the fact already there instead of making a second one.
    """
    newest = max(members, key=lambda doc: (str(doc.get("created_at") or ""), str(doc.get("id"))))
    record = survivor_patch(newest, [doc for doc in members if doc is not newest])
    for key in ("embedding", "superseded_by", "supersede_reason", "superseded_at"):
        record.pop(key, None)
    record["merged_from"] = sorted(set(record.get("merged_from") or []) | {doc["id"] for doc in members})
    record["content"] = text
    record["content_hash"] = compute_content_hash(text)
    record["tags"] = sorted(set(record.get("tags") or []) | {COMBINED_TAG})
    record.update(lineage)
    names = "|".join(sorted(doc["id"] for doc in members))
    seed = ID_SEED_SEP.join((str(record.get("user_id")), "combine", names))
    record["id"] = f"fact_{hashlib.sha256(seed.encode()).hexdigest()[:32]}"
    return record


@dataclass(frozen=True)
class CombineOutcome:
    """One group's model calls: ``combined``, ``declined`` or ``failed`` with the writer's ``draft``;
    ``deferred`` after a retryable model error (a later pass tries the group again); ``skipped`` after any
    other model error."""

    status: str
    draft: Optional[CombineDraft]
    calls: int
    error: Optional[str] = None


@dataclass(frozen=True)
class CombineWrite:
    """What writing a combined fact did: ``written`` (with the IDs of the members retired into it),
    ``lost`` (it cannot be built or validated, or its embedding fails for good: the members stay) or
    ``deferred`` (a retryable embedding error: a later pass tries the group again)."""

    status: str
    retired: tuple[str, ...] = ()


def _failed(exc: BaseException, calls: int) -> CombineOutcome:
    status = "deferred" if is_retryable_llm_error(exc) else "skipped"
    return CombineOutcome(status=status, draft=None, calls=calls, error=describe_exception(exc))


def run_combine_group(docs: Sequence[dict[str, Any]], call: Callable[[str, dict[str, str]], Any]) -> CombineOutcome:
    """Drive :func:`combine_group` with a synchronous ``call(prompt, inputs) -> parsed answer``."""
    steps = combine_group(docs)
    calls = 0
    try:
        request = next(steps)
        while True:
            try:
                answer = call(request.prompt, request.inputs)
            except Exception as exc:  # noqa: BLE001 - one bad call must not stop the pass
                steps.close()
                return _failed(exc, calls)
            calls += 1
            request = steps.send(answer)
    except StopIteration as stop:
        draft: CombineDraft = stop.value
        return CombineOutcome(status=draft.status, draft=draft, calls=calls)


async def arun_combine_group(
    docs: Sequence[dict[str, Any]], call: Callable[[str, dict[str, str]], Awaitable[Any]]
) -> CombineOutcome:
    """Drive :func:`combine_group` with an async ``call(prompt, inputs) -> parsed answer``."""
    steps = combine_group(docs)
    calls = 0
    try:
        request = next(steps)
        while True:
            try:
                answer = await call(request.prompt, request.inputs)
            except Exception as exc:  # noqa: BLE001 - one bad call must not stop the pass
                steps.close()
                return _failed(exc, calls)
            calls += 1
            request = steps.send(answer)
    except StopIteration as stop:
        draft: CombineDraft = stop.value
        return CombineOutcome(status=draft.status, draft=draft, calls=calls)


@dataclass
class CombineCounts:
    """A pass's combining: groups tried, combined facts written, groups declined or not written, and the
    facts the written ones retired."""

    groups: int = 0
    written: int = 0
    failed: int = 0
    retired_ids: set[str] = field(default_factory=set)
