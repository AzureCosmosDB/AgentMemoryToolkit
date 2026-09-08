"""Data model for the Shared-Memory Governance Benchmark (SMGB).

SMGB evaluates *shared* agent memory: memory reused across agents, users,
teams, tenants, and organizations. Unlike single-user memory benchmarks
(LoCoMo, LongMemEval, DMR) that score only recall, every SMGB query is issued
by a *principal* (a user or agent with a tenant, roles, and scope memberships)
at a point in time, and ground truth partitions memories into what the
principal MUST retrieve (authorized + relevant -> utility) and MUST NOT
retrieve (unauthorized / wrong scope / superseded / deleted / cross-tenant ->
leakage).

The classes here are plain, dependency-free dataclasses. Labels are never
hand-assigned in a way the harness trusts blindly: :mod:`benchmarks.governance.policy`
derives the authorized/forbidden sets *deterministically* from the scope graph
and the event timeline, and the loader validates any hand annotations against
that policy. This is what makes the benchmark reproducible and system-agnostic:
a system under test only returns ranked memory ids, and the scorer compares
them to policy-derived labels.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Optional


def parse_time(value: str | datetime) -> datetime:
    """Parse an ISO-8601 timestamp into an aware ``datetime`` (UTC if naive).

    Accepts a trailing ``Z``. Naive datetimes are assumed to be UTC so that
    all comparisons in the policy layer are well defined.
    """

    if isinstance(value, datetime):
        dt = value
    else:
        text = value.strip()
        if text.endswith("Z"):
            text = text[:-1] + "+00:00"
        dt = datetime.fromisoformat(text)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


# Event types that mutate a memory's governance state over time.
EVENT_TYPES = frozenset({"promote", "supersede", "delete"})

# Query axes map to the governance failure modes SMGB scores. These mirror the
# four failure modes named in "Governed Shared Memory for Multi-Agent LLM
# Systems" (arXiv:2606.24535) plus explicit promotion and abstention.
QUERY_AXES = frozenset(
    {
        "utility",  # authorized recall: the principal should get entitled facts
        "leakage",  # the principal must not receive forbidden facts
        "isolation",  # cross-tenant memory must never be returned
        "promotion",  # promoted facts visible after T to scope members, not before
        "conflict",  # only the current version is retrievable in "current" mode
        "deletion",  # deleted facts (and copies) gone for all scope members
        "provenance",  # every shared fact traces to author/source/time
    }
)


@dataclass(frozen=True)
class Scope:
    """A visibility unit: user, agent, session, project, group, tenant, org.

    Membership is declared on :class:`Principal` (``scopes``); ``Scope`` carries
    the scope's kind and owning tenant for metadata and reporting. ``members``
    is optional and, when present, is cross-checked against principal
    declarations by the loader.
    """

    id: str
    kind: str
    tenant: str
    members: tuple[str, ...] = ()

    @staticmethod
    def from_dict(d: dict[str, Any]) -> "Scope":
        return Scope(
            id=d["id"],
            kind=d["kind"],
            tenant=d["tenant"],
            members=tuple(d.get("members", ()) or ()),
        )


@dataclass(frozen=True)
class Principal:
    """A user or agent issuing reads/writes, scoped by tenant + memberships."""

    id: str
    kind: str  # "user" or "agent"
    tenant: str
    roles: tuple[str, ...] = ()
    scopes: tuple[str, ...] = ()  # scope ids this principal is a member of

    @staticmethod
    def from_dict(d: dict[str, Any]) -> "Principal":
        return Principal(
            id=d["id"],
            kind=d.get("kind", "user"),
            tenant=d["tenant"],
            roles=tuple(d.get("roles", ()) or ()),
            scopes=tuple(d.get("scopes", ()) or ()),
        )


@dataclass(frozen=True)
class Provenance:
    """Where a shared memory came from — author, source message, confidence.

    ``derived_from`` lets a fact point at the memory ids it was derived from,
    so a scorer can reconstruct multi-hop provenance chains (cf. ArgusFleet's
    depth-N provenance reconstruction).
    """

    author: str
    source: Optional[str] = None
    confidence: Optional[float] = None
    derived_from: tuple[str, ...] = ()

    @staticmethod
    def from_dict(d: dict[str, Any]) -> "Provenance":
        return Provenance(
            author=d["author"],
            source=d.get("source"),
            confidence=d.get("confidence"),
            derived_from=tuple(d.get("derived_from", ()) or ()),
        )


@dataclass(frozen=True)
class Memory:
    """A single memory record with a birth time and an initial scope.

    Governance state (current scope set, superseded, deleted) is *not* stored
    here; it is computed as of a query time by the policy layer from the event
    timeline. This keeps the dataset a single source of truth.
    """

    id: str
    type: str  # turn | episode | fact | summary
    content: str
    tenant: str
    scope: str  # initial scope at creation
    created_at: datetime
    provenance: Provenance

    @staticmethod
    def from_dict(d: dict[str, Any]) -> "Memory":
        prov = d.get("provenance")
        if prov is None:
            # Allow a flat shorthand: author/source/confidence at top level.
            prov = {
                "author": d.get("author", "unknown"),
                "source": d.get("source"),
                "confidence": d.get("confidence"),
            }
        return Memory(
            id=d["id"],
            type=d.get("type", "fact"),
            content=d.get("content", ""),
            tenant=d["tenant"],
            scope=d["scope"],
            created_at=parse_time(d["created_at"]),
            provenance=Provenance.from_dict(prov),
        )


@dataclass(frozen=True)
class Event:
    """A governance mutation on the timeline: promote, supersede, or delete."""

    t: datetime
    type: str
    memory_id: str
    to_scope: Optional[str] = None  # promote target
    by: Optional[str] = None  # supersede: id of the replacement memory
    actor: Optional[str] = None  # principal who performed the action

    @staticmethod
    def from_dict(d: dict[str, Any]) -> "Event":
        etype = d["type"]
        if etype not in EVENT_TYPES:
            raise ValueError(f"unknown event type {etype!r}; expected {sorted(EVENT_TYPES)}")
        return Event(
            t=parse_time(d["t"]),
            type=etype,
            memory_id=d["memory_id"],
            to_scope=d.get("to_scope"),
            by=d.get("by"),
            actor=d.get("actor"),
        )


@dataclass(frozen=True)
class Query:
    """A read issued by a principal at ``as_of``, scored on one axis.

    ``relevant`` lists the memory ids that answer the question (the utility
    target *before* authorization is applied). ``temporal_mode`` is "current"
    (default) or "history": in history mode superseded versions are allowed.
    ``must_retrieve`` / ``must_not_retrieve`` are optional hand annotations the
    loader validates against the policy-derived labels.
    """

    id: str
    principal: str
    as_of: datetime
    axis: str
    question: str
    relevant: tuple[str, ...] = ()
    temporal_mode: str = "current"
    expected_answer: Optional[str] = None
    must_retrieve: Optional[tuple[str, ...]] = None
    must_not_retrieve: Optional[tuple[str, ...]] = None

    @staticmethod
    def from_dict(d: dict[str, Any]) -> "Query":
        axis = d.get("axis", "utility")
        if axis not in QUERY_AXES:
            raise ValueError(f"unknown query axis {axis!r}; expected {sorted(QUERY_AXES)}")
        mr = d.get("must_retrieve")
        mnr = d.get("must_not_retrieve")
        return Query(
            id=d["id"],
            principal=d["principal"],
            as_of=parse_time(d["as_of"]),
            axis=axis,
            question=d.get("question", ""),
            relevant=tuple(d.get("relevant", ()) or ()),
            temporal_mode=d.get("temporal_mode", "current"),
            expected_answer=d.get("expected_answer"),
            must_retrieve=tuple(mr) if mr is not None else None,
            must_not_retrieve=tuple(mnr) if mnr is not None else None,
        )


@dataclass
class Scenario:
    """A self-contained governance test: scopes, principals, memories, timeline, queries."""

    scenario_id: str
    tenants: tuple[str, ...]
    scopes: dict[str, Scope]
    principals: dict[str, Principal]
    memories: dict[str, Memory]
    events: list[Event]
    queries: list[Query]
    description: str = ""

    @staticmethod
    def from_dict(d: dict[str, Any]) -> "Scenario":
        scopes = {s["id"]: Scope.from_dict(s) for s in d.get("scopes", [])}
        principals = {p["id"]: Principal.from_dict(p) for p in d.get("principals", [])}
        memories = {m["id"]: Memory.from_dict(m) for m in d.get("memories", [])}
        events = sorted(
            (Event.from_dict(e) for e in d.get("events", [])),
            key=lambda e: e.t,
        )
        queries = [Query.from_dict(q) for q in d.get("queries", [])]
        tenants = tuple(d.get("tenants") or sorted({m.tenant for m in memories.values()}))
        return Scenario(
            scenario_id=d["scenario_id"],
            tenants=tenants,
            scopes=scopes,
            principals=principals,
            memories=memories,
            events=events,
            queries=queries,
            description=d.get("description", ""),
        )
