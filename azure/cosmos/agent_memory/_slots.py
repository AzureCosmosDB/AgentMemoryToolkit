"""Typed result slots for search: one preference fact and one situational procedure.

A search with slots still returns ``top_k`` results. The ranked search fills the first
``top_k`` minus the slots; the user's best-matching preference fact not already shown fills
one slot, and their best-matching active situational procedure the other. A slot with no
candidate takes the next ranked result instead, so the count stays ``top_k``. For example, at
``top_k=5`` that is 3 ranked results, 1 preference and 1 procedure.

Slot candidates are ranked by vector similarity alone. Their queries reuse the search's own
arguments (user, thread and filters), so a slot can only return what the search itself could.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

PREFERENCE_CATEGORY = "preference"
# Where a slotted result names the slot it filled and its place in the results; ranked results carry none.
RETRIEVAL_METADATA_KEY = "retrieval"


@dataclass(frozen=True)
class SlotPlan:
    preference: bool = False
    procedure: bool = False

    @property
    def count(self) -> int:
        return int(self.preference) + int(self.procedure)


def plan_slots(top_k: int, *, preference_slot: bool, procedure_slot: bool, pool_types: Sequence[str]) -> SlotPlan:
    """The slots a search reserves: a preference slot only when it searches facts, a procedure slot only when it
    searches something besides procedures (a search for procedures alone ranks them), and none unless at least one
    ranked result remains beside them. ``pool_types`` are the memory types the ranked search covers, procedures
    excluded."""
    plan = SlotPlan(preference=preference_slot and "fact" in pool_types, procedure=procedure_slot and bool(pool_types))
    return plan if 0 < plan.count < top_k else SlotPlan()


def preference_search(search_kwargs: Mapping[str, Any], ranked_count: int) -> dict[str, Any]:
    """Store search arguments for the preference slot: enough candidates to skip those already shown."""
    return {
        **search_kwargs,
        "memory_types": ["fact"],
        "categories": [PREFERENCE_CATEGORY],
        "vector_only": True,
        "top_k": ranked_count + 1,
    }


def procedure_search(search_kwargs: Mapping[str, Any]) -> dict[str, Any]:
    """Store search arguments for the procedure slot: the nearest active situational procedure."""
    return {
        **search_kwargs,
        "memory_types": ["procedural"],
        "applies": "when",
        "active_procedures_only": True,
        "vector_only": True,
        "top_k": 1,
    }


def _key(doc: Mapping[str, Any]) -> tuple[str, str]:
    return str(doc.get("id") or "").strip(), str(doc.get("content") or "").strip()


def compose(
    ranked: Sequence[dict[str, Any]],
    *,
    top_k: int,
    plan: SlotPlan,
    preferences: Sequence[dict[str, Any]],
    procedures: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    """The ranked results with the planned slots filled, at most ``top_k`` in all.

    ``ranked`` holds up to ``top_k`` results so the spare ones can fill an empty slot. A
    candidate already shown (same ID or text) is skipped. Each slotted result's ``retrieval``
    metadata says which slot it filled (``preference``, ``procedure`` or ``backfill``) and its
    1-based ``rank`` in the results.
    """
    keep = top_k - plan.count
    results = [dict(doc) for doc in ranked[:keep]]
    spare = [dict(doc) for doc in ranked[keep:]]
    seen_ids = {key[0] for key in map(_key, results) if key[0]}
    seen_text = {key[1] for key in map(_key, results) if key[1]}

    def unseen(doc: Mapping[str, Any]) -> bool:
        doc_id, text = _key(doc)
        return not (doc_id and doc_id in seen_ids) and not (text and text in seen_text)

    def add(doc: dict[str, Any], slot: str) -> None:
        doc_id, text = _key(doc)
        seen_ids.add(doc_id)
        seen_text.add(text)
        doc[RETRIEVAL_METADATA_KEY] = {"rank": len(results) + 1, "slot": slot}
        results.append(doc)

    for wanted, candidates, slot in (
        (plan.preference, preferences, "preference"),
        (plan.procedure, procedures, "procedure"),
    ):
        if not wanted:
            continue
        pick = next((dict(doc) for doc in candidates if unseen(doc)), None)
        if pick is not None:
            add(pick, slot)
            continue
        backfill = next((doc for doc in spare if unseen(doc)), None)
        if backfill is not None:
            add(backfill, "backfill")
    return results
