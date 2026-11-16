"""Conflict links: facts that disagree stay active and point at each other.

The duplicate pass already asks the model how each new fact relates to its nearest neighbours (``dedup_verify``). Two of
its labels name a conflict: ``update`` (the same attribute with a different value at a later time) and ``inconsistent``
(the two cannot both be true). Retiring either side on that one call misleads answers whenever the call is wrong, so
nothing is retired. The pass links the pair instead, and ``search_cosmos`` returns each fact with the active facts it
conflicts with (``conflicts_with``), so the agent sees both sides and can ask the user which is right.

A link is one small document in the user's memories container, in the user's ``__conflicts__`` partition. It holds the
IDs of the two facts and the kind, and no text. Its ID derives from the pair, so judging a pair again rewrites the same
document. Reading ignores a link whose other fact is no longer active, so nothing needs cleaning up when a fact is
folded, combined or deleted. Facts written by the assistant (``sys:agent-fact``) are never linked: they record what the
agent said or recommended, not what the user holds true.

When a pass retires one of the facts of a pair into another (a fold or a combine), the link goes to the fact that took
its place, so a conflict survives its fact being combined.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Awaitable, Callable, Iterable, Mapping, Optional, Sequence

from azure.cosmos.agent_memory.logging import describe_exception, get_logger
from azure.cosmos.agent_memory.services._dedup import ACTIVE_FILTER, CONFLICT_RELATIONS, Verdict, is_agent_sourced
from azure.cosmos.agent_memory.services._pipeline_helpers import ID_SEED_SEP

logger = get_logger(__name__)

CONFLICT_LINK_TYPE = "conflict_link"
CONFLICT_LINK_THREAD_ID = "__conflicts__"
# The most conflicting facts shown with one fact: the newest ones.
MAX_CONFLICT_PARTNERS = 2


@dataclass(frozen=True)
class ConflictPair:
    """Two facts that disagree: the one recorded first, the one recorded later, and whether the pair reads as an
    ``update`` over time or as ``inconsistent``."""

    first: str
    later: str
    kind: str

    @property
    def key(self) -> tuple[str, str]:
        return (self.first, self.later) if self.first <= self.later else (self.later, self.first)


def _recorded(doc: Mapping[str, Any]) -> tuple[str, str]:
    return (str(doc.get("created_at") or ""), str(doc.get("id")))


def conflict_pairs(
    checked: Iterable[tuple[Mapping[str, Any], Sequence[Verdict]]],
    *,
    absorbed: Optional[Mapping[str, str]] = None,
) -> list[ConflictPair]:
    """The pairs a pass's verdicts call a conflict, each once.

    ``checked`` holds each seed with its verdicts. ``absorbed`` maps a fact the pass retired (into another fact, as a
    duplicate or as a combined member) to the fact that took its place: a pair is linked between the facts that are
    still active, and dropped when both sides ended up in the same fact. Pairs with an assistant-written fact are
    skipped.
    """
    absorbed = absorbed or {}

    def active(fact_id: str) -> str:
        for _ in range(3):
            if fact_id not in absorbed:
                break
            fact_id = absorbed[fact_id]
        return fact_id

    pairs: dict[tuple[str, str], ConflictPair] = {}
    for seed, verdicts in checked:
        for verdict in verdicts:
            other = verdict.candidate
            if verdict.relation not in CONFLICT_RELATIONS or seed["id"] == other["id"]:
                continue
            if is_agent_sourced(seed) or is_agent_sourced(other):
                continue
            first, later = sorted((seed, other), key=_recorded)
            first_id, later_id = active(first["id"]), active(later["id"])
            if first_id == later_id:
                continue
            pair = ConflictPair(first_id, later_id, verdict.relation)
            pairs.setdefault(pair.key, pair)
    return list(pairs.values())


def conflict_link_id(user_id: str, pair: ConflictPair) -> str:
    seed = ID_SEED_SEP.join((user_id, "conflict", *pair.key))
    return f"conflict_{hashlib.sha256(seed.encode()).hexdigest()[:32]}"


def conflict_link_body(user_id: str, pair: ConflictPair, *, now: Optional[datetime] = None) -> dict[str, Any]:
    """The document that links ``pair`` (written to the memories container, not through the memory store)."""
    return {
        "id": conflict_link_id(user_id, pair),
        "type": CONFLICT_LINK_TYPE,
        "user_id": user_id,
        "thread_id": CONFLICT_LINK_THREAD_ID,
        "first": pair.first,
        "later": pair.later,
        "kind": pair.kind,
        "created_at": (now or datetime.now(timezone.utc)).isoformat(),
    }


def links_query(user_id: str, fact_ids: Sequence[str]) -> tuple[str, list[dict[str, Any]]]:
    """The links that touch any of ``fact_ids``; run it in the partition ``[user_id, CONFLICT_LINK_THREAD_ID]``."""
    query = (
        "SELECT c.first, c.later, c.kind FROM c WHERE c.user_id = @user_id AND c.type = @type "
        "AND (ARRAY_CONTAINS(@ids, c.first) OR ARRAY_CONTAINS(@ids, c.later))"
    )
    parameters = [
        {"name": "@user_id", "value": user_id},
        {"name": "@type", "value": CONFLICT_LINK_TYPE},
        {"name": "@ids", "value": list(fact_ids)},
    ]
    return query, parameters


def partners_query(user_id: str, fact_ids: Sequence[str]) -> tuple[str, list[dict[str, Any]]]:
    """The active facts among ``fact_ids``, with what a result shows of a conflicting fact (run across the user's
    partitions)."""
    query = (
        "SELECT c.id, c.content, c.created_at FROM c WHERE c.user_id = @user_id AND c.type = 'fact' "
        f"AND {ACTIVE_FILTER} AND ARRAY_CONTAINS(@ids, c.id)"
    )
    parameters = [{"name": "@user_id", "value": user_id}, {"name": "@ids", "value": list(fact_ids)}]
    return query, parameters


def link_endpoints(links: Iterable[Mapping[str, Any]]) -> list[str]:
    """Every fact ID the ``links`` name, sorted and without repeats."""
    return sorted({fact_id for link in links for fact_id in (link.get("first"), link.get("later")) if fact_id})


def attach_conflicts(
    docs: Sequence[dict[str, Any]],
    links: Iterable[Mapping[str, Any]],
    partners: Mapping[str, Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """``docs`` with, on each fact that has any, ``conflicts_with``: at most ``MAX_CONFLICT_PARTNERS`` active facts it
    conflicts with, newest first, each as ``{"id", "content", "created_at", "kind"}``.

    ``partners`` holds the active facts by ID; a link to a fact that is not there (retired or deleted) is ignored.
    """
    linked: dict[str, list[tuple[str, Any]]] = {}
    for link in links:
        first, later, kind = link.get("first"), link.get("later"), link.get("kind")
        if first and later and first != later:
            linked.setdefault(first, []).append((later, kind))
            linked.setdefault(later, []).append((first, kind))
    out: list[dict[str, Any]] = []
    for doc in docs:
        entries: dict[str, dict[str, Any]] = {}
        for partner_id, kind in linked.get(doc.get("id"), ()):
            partner = partners.get(partner_id)
            if partner is not None and partner_id not in entries:
                entries[partner_id] = {
                    "id": partner_id,
                    "content": partner.get("content"),
                    "created_at": partner.get("created_at"),
                    "kind": kind,
                }
        shown = sorted(entries.values(), key=lambda e: (str(e["created_at"] or ""), e["id"]), reverse=True)
        out.append({**doc, "conflicts_with": shown[:MAX_CONFLICT_PARTNERS]} if shown else doc)
    return out


def _fact_ids_by_user(docs: Iterable[Mapping[str, Any]]) -> dict[str, list[str]]:
    ids_by_user: dict[str, list[str]] = {}
    for doc in docs:
        if doc.get("type") == "fact" and doc.get("id") and doc.get("user_id"):
            ids_by_user.setdefault(str(doc["user_id"]), []).append(str(doc["id"]))
    return ids_by_user


def with_conflicts(
    docs: Sequence[dict[str, Any]], query: Callable[..., Iterable[Mapping[str, Any]]]
) -> list[dict[str, Any]]:
    """``docs`` (search results) with the conflicts of each fact attached (``conflicts_with``).

    ``query(sql, parameters, *, partition_key=None, cross_partition=False)`` runs a query on the memories container and
    returns its rows. A search with no conflicts costs one query per user; with links, a second reads their partners.
    Conflicts are context: a failed lookup is logged and ``docs`` come back as they were.
    """
    ids_by_user = _fact_ids_by_user(docs)
    if not ids_by_user:
        return list(docs)
    try:
        links: list[Mapping[str, Any]] = []
        partners: dict[str, Mapping[str, Any]] = {}
        for user_id, fact_ids in ids_by_user.items():
            sql, parameters = links_query(user_id, fact_ids)
            found = list(query(sql, parameters, partition_key=[user_id, CONFLICT_LINK_THREAD_ID]))
            if not found:
                continue
            links.extend(found)
            sql, parameters = partners_query(user_id, link_endpoints(found))
            partners.update({row["id"]: row for row in query(sql, parameters, cross_partition=True)})
        return attach_conflicts(docs, links, partners)
    except Exception as exc:  # noqa: BLE001 - conflicts must never break a search
        logger.warning(
            "search: conflicts_with lookup failed (%s); returning results without it", describe_exception(exc)
        )
        return list(docs)


async def awith_conflicts(
    docs: Sequence[dict[str, Any]], query: Callable[..., Awaitable[Iterable[Mapping[str, Any]]]]
) -> list[dict[str, Any]]:
    """:func:`with_conflicts` for an async ``query``."""
    ids_by_user = _fact_ids_by_user(docs)
    if not ids_by_user:
        return list(docs)
    try:
        links: list[Mapping[str, Any]] = []
        partners: dict[str, Mapping[str, Any]] = {}
        for user_id, fact_ids in ids_by_user.items():
            sql, parameters = links_query(user_id, fact_ids)
            found = list(await query(sql, parameters, partition_key=[user_id, CONFLICT_LINK_THREAD_ID]))
            if not found:
                continue
            links.extend(found)
            sql, parameters = partners_query(user_id, link_endpoints(found))
            partners.update({row["id"]: row for row in await query(sql, parameters, cross_partition=True)})
        return attach_conflicts(docs, links, partners)
    except Exception as exc:  # noqa: BLE001 - conflicts must never break a search
        logger.warning(
            "search: conflicts_with lookup failed (%s); returning results without it", describe_exception(exc)
        )
        return list(docs)
