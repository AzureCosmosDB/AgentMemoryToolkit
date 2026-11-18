"""Conflict links: facts that disagree stay active and point at each other.

The duplicate pass already asks the model how each new fact relates to its nearest neighbours (``dedup_verify``). Two of
its labels name a conflict: ``update`` (the same attribute with a different value at a later time) and ``inconsistent``
(the two cannot both be true). Retiring either side on that one call misleads answers whenever the call is wrong, so
nothing is retired. The pass links the pair instead, and ``search_cosmos`` returns each fact with the active facts it
conflicts with (``conflicts_with``), so the agent sees both sides and can ask the user which is right.

A link is one small document in the owner's memories container, in the owner's partition, marked with the thread id
``__conflicts__``. It holds the IDs of the two facts and the kind, and no text. Its ID derives from the pair, so judging
a pair again rewrites the same document. Facts written by the assistant (``sys:agent-fact``) are never linked: they
record what the agent said or recommended, not what the user holds true.

A link names the two facts as they were when it was written, and a later pass can retire either into another fact (a
fold or a combine). The conflict follows the fact. When the pass that retires one of a pair is the pass that links it,
the link is written to the fact that took its place. When a later pass retires it, reading follows ``superseded_by``
from the retired fact to the fact that replaced it, and a fact that absorbed others (``merged_from``) shows the
conflicts of what it absorbed. A conflict is dropped once either side is deleted, or when both sides end up in one
fact, so nothing needs cleaning up.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Awaitable, Callable, Iterable, Mapping, Optional, Sequence

from azure.cosmos.agent_memory._partitioning import OWNER_FILTER, owner_parameters, partition_key_for
from azure.cosmos.agent_memory.logging import describe_exception, get_logger
from azure.cosmos.agent_memory.services._dedup import CONFLICT_RELATIONS, Verdict, is_agent_sourced
from azure.cosmos.agent_memory.services._pipeline_helpers import ID_SEED_SEP

logger = get_logger(__name__)

CONFLICT_LINK_TYPE = "conflict_link"
CONFLICT_LINK_THREAD_ID = "__conflicts__"
# The most conflicting facts shown with one fact: the newest ones.
MAX_CONFLICT_PARTNERS = 2
# How many times a retired fact is followed to the fact that replaced it.
MAX_REDIRECT_HOPS = 3
# The most fact IDs one search looks links up for, per user: the results' own IDs first, then the facts they absorbed.
MAX_LOOKUP_IDS = 200


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
        for _ in range(MAX_REDIRECT_HOPS):
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


def conflict_link_body(
    tenant_id: str, user_id: str, pair: ConflictPair, *, now: Optional[datetime] = None
) -> dict[str, Any]:
    """The document that links ``pair`` (written to the memories container, not through the memory store)."""
    return {
        "id": conflict_link_id(user_id, pair),
        "type": CONFLICT_LINK_TYPE,
        "tenant_id": tenant_id,
        "user_id": user_id,
        "thread_id": CONFLICT_LINK_THREAD_ID,
        "first": pair.first,
        "later": pair.later,
        "kind": pair.kind,
        "created_at": (now or datetime.now(timezone.utc)).isoformat(),
    }


def links_query(tenant_id: str, user_id: str, fact_ids: Sequence[str]) -> tuple[str, list[dict[str, Any]]]:
    """The links that touch any of ``fact_ids``; run it in the owner's partition."""
    query = (
        f"SELECT c.first, c.later, c.kind FROM c WHERE {OWNER_FILTER} AND c.type = @type "
        "AND (ARRAY_CONTAINS(@ids, c.first) OR ARRAY_CONTAINS(@ids, c.later))"
    )
    parameters = [
        *owner_parameters(tenant_id, user_id),
        {"name": "@type", "value": CONFLICT_LINK_TYPE},
        {"name": "@ids", "value": list(fact_ids)},
    ]
    return query, parameters


def facts_query(tenant_id: str, user_id: str, fact_ids: Sequence[str]) -> tuple[str, list[dict[str, Any]]]:
    """The facts among ``fact_ids``, active or not, with what a result shows of a conflicting fact and where a retired
    one went (``superseded_by``); run it in the owner's partition."""
    query = (
        "SELECT c.id, c.content, c.created_at, c.superseded_by FROM c "
        f"WHERE {OWNER_FILTER} AND c.type = 'fact' AND ARRAY_CONTAINS(@ids, c.id)"
    )
    parameters = [*owner_parameters(tenant_id, user_id), {"name": "@ids", "value": list(fact_ids)}]
    return query, parameters


def link_endpoints(links: Iterable[Mapping[str, Any]]) -> list[str]:
    """Every fact ID the ``links`` name, sorted and without repeats."""
    return sorted({fact_id for link in links for fact_id in (link.get("first"), link.get("later")) if fact_id})


class EndpointResolver:
    """The active fact that stands for each fact a link names.

    A link names facts as they were when it was written, and a later pass can retire one into another fact. The
    resolver follows ``superseded_by`` from a retired fact to the fact that replaced it, up to ``MAX_REDIRECT_HOPS``
    steps. It reads in rounds: take :meth:`next_ids`, read those facts, :meth:`add` the rows, and repeat until
    ``next_ids`` is empty. A fact that is gone, a chain that ends in one, and a chain that is too long or loops all
    stand for nothing.
    """

    def __init__(self, fact_ids: Iterable[str]) -> None:
        self._roots = list(dict.fromkeys(fact_ids))
        self._rows: dict[str, Mapping[str, Any]] = {}
        self._asked: set[str] = set()

    def _follow(self, root: str) -> tuple[Optional[Mapping[str, Any]], Optional[str]]:
        """The active fact ``root`` stands for, or else the ID where the chain stops because that fact is unread."""
        fact_id = root
        for _ in range(MAX_REDIRECT_HOPS + 1):
            row = self._rows.get(fact_id)
            if row is None:
                return None, fact_id
            replaced_by = row.get("superseded_by")
            if not replaced_by:
                return row, None
            fact_id = replaced_by
        return None, None

    def next_ids(self) -> list[str]:
        """The fact IDs to read next. Each is handed out once, whether or not the fact turns out to exist."""
        wanted: list[str] = []
        for root in self._roots:
            _, unread = self._follow(root)
            if unread is not None and unread not in self._asked and unread not in wanted:
                wanted.append(unread)
        self._asked.update(wanted)
        return wanted

    def add(self, rows: Iterable[Mapping[str, Any]]) -> None:
        for row in rows:
            if row.get("id"):
                self._rows[row["id"]] = row

    def resolved(self) -> dict[str, Mapping[str, Any]]:
        """Each fact ID that stands for an active fact, with that fact."""
        found: dict[str, Mapping[str, Any]] = {}
        for root in self._roots:
            row, _ = self._follow(root)
            if row is not None:
                found[root] = row
        return found


def _absorbed_ids(doc: Mapping[str, Any]) -> list[str]:
    """The facts ``doc`` absorbed (``merged_from``): duplicates folded into it, or the members of a combined fact."""
    merged = doc.get("merged_from")
    if not isinstance(merged, (list, tuple)):
        return []
    return [fact_id for fact_id in merged if isinstance(fact_id, str) and fact_id]


def attach_conflicts(
    docs: Sequence[dict[str, Any]],
    links: Iterable[Mapping[str, Any]],
    active: Mapping[str, Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """``docs`` with, on each fact that has any, ``conflicts_with``: at most ``MAX_CONFLICT_PARTNERS`` active facts it
    conflicts with, newest first, each as ``{"id", "content", "created_at", "kind"}``.

    ``active`` maps each fact ID a link names to the active fact that stands for it: the fact itself, or the fact it
    was retired into. A fact stands for itself and for the facts it absorbed, so it shows their conflicts too. A link
    whose far side stands for no active fact (it was deleted) is ignored, and so is one whose two sides stand for the
    same fact.
    """
    stands_for: dict[str, list[str]] = {}
    for doc in docs:
        if doc.get("type") == "fact" and doc.get("id"):
            for fact_id in (doc["id"], *_absorbed_ids(doc)):
                stands_for.setdefault(fact_id, []).append(doc["id"])
    entries: dict[str, dict[str, dict[str, Any]]] = {}
    for link in links:
        first, later, kind = link.get("first"), link.get("later"), link.get("kind")
        if not first or not later or first == later:
            continue
        for here, there in ((first, later), (later, first)):
            partner = active.get(there)
            if partner is None:
                continue
            for doc_id in stands_for.get(here, ()):
                if partner["id"] != doc_id:
                    entries.setdefault(doc_id, {}).setdefault(
                        partner["id"],
                        {
                            "id": partner["id"],
                            "content": partner.get("content"),
                            "created_at": partner.get("created_at"),
                            "kind": kind,
                        },
                    )
    out: list[dict[str, Any]] = []
    for doc in docs:
        found = entries.get(doc.get("id")) if doc.get("type") == "fact" else None
        shown = (
            sorted(found.values(), key=lambda e: (str(e["created_at"] or ""), e["id"]), reverse=True) if found else []
        )
        out.append({**doc, "conflicts_with": shown[:MAX_CONFLICT_PARTNERS]} if shown else doc)
    return out


def _lookup_ids_by_user(docs: Iterable[Mapping[str, Any]]) -> dict[str, list[str]]:
    """The fact IDs to look links up for, per user: each fact result's own ID, then (up to ``MAX_LOOKUP_IDS``) the facts
    it absorbed."""
    facts = [doc for doc in docs if doc.get("type") == "fact" and doc.get("id") and doc.get("user_id")]
    ids_by_user: dict[str, dict[str, None]] = {}
    for doc in facts:
        ids_by_user.setdefault(str(doc["user_id"]), {})[str(doc["id"])] = None
    for doc in facts:
        ids = ids_by_user[str(doc["user_id"])]
        for fact_id in _absorbed_ids(doc):
            if len(ids) >= MAX_LOOKUP_IDS:
                break
            ids.setdefault(fact_id, None)
    return {user_id: list(ids) for user_id, ids in ids_by_user.items()}


def with_conflicts(
    docs: Sequence[dict[str, Any]], query: Callable[..., Iterable[Mapping[str, Any]]], tenant_id: str
) -> list[dict[str, Any]]:
    """``docs`` (search results) with the conflicts of each fact attached (``conflicts_with``).

    ``query(sql, parameters, *, partition_key)`` runs a query on the memories container, in that partition, and returns
    its rows; every query reads one owner's partition, ``[tenant_id, user_id]``. A search with no conflicts costs one
    query per user; with links, one more reads the facts they name, and one more for each step a linked fact has since
    been retired through.
    Conflicts are context: a failed lookup is logged and ``docs`` come back as they were.
    """
    ids_by_user = _lookup_ids_by_user(docs)
    if not ids_by_user:
        return list(docs)
    try:
        links: list[Mapping[str, Any]] = []
        active: dict[str, Mapping[str, Any]] = {}
        for user_id, fact_ids in ids_by_user.items():
            partition_key = partition_key_for(tenant_id, user_id)
            sql, parameters = links_query(tenant_id, user_id, fact_ids)
            found = list(query(sql, parameters, partition_key=partition_key))
            if not found:
                continue
            links.extend(found)
            resolver = EndpointResolver(link_endpoints(found))
            while wanted := resolver.next_ids():
                sql, parameters = facts_query(tenant_id, user_id, wanted)
                resolver.add(query(sql, parameters, partition_key=partition_key))
            active.update(resolver.resolved())
        return attach_conflicts(docs, links, active)
    except Exception as exc:  # noqa: BLE001 - conflicts must never break a search
        logger.warning(
            "search: conflicts_with lookup failed (%s); returning results without it", describe_exception(exc)
        )
        return list(docs)


async def awith_conflicts(
    docs: Sequence[dict[str, Any]], query: Callable[..., Awaitable[Iterable[Mapping[str, Any]]]], tenant_id: str
) -> list[dict[str, Any]]:
    """:func:`with_conflicts` for an async ``query``."""
    ids_by_user = _lookup_ids_by_user(docs)
    if not ids_by_user:
        return list(docs)
    try:
        links: list[Mapping[str, Any]] = []
        active: dict[str, Mapping[str, Any]] = {}
        for user_id, fact_ids in ids_by_user.items():
            partition_key = partition_key_for(tenant_id, user_id)
            sql, parameters = links_query(tenant_id, user_id, fact_ids)
            found = list(await query(sql, parameters, partition_key=partition_key))
            if not found:
                continue
            links.extend(found)
            resolver = EndpointResolver(link_endpoints(found))
            while wanted := resolver.next_ids():
                sql, parameters = facts_query(tenant_id, user_id, wanted)
                resolver.add(await query(sql, parameters, partition_key=partition_key))
            active.update(resolver.resolved())
        return attach_conflicts(docs, links, active)
    except Exception as exc:  # noqa: BLE001 - conflicts must never break a search
        logger.warning(
            "search: conflicts_with lookup failed (%s); returning results without it", describe_exception(exc)
        )
        return list(docs)
