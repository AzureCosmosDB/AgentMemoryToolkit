"""Deterministic governance policy for SMGB.

This module is the crux of the benchmark. Given a scenario's scope graph and
event timeline, it computes — for any principal at any point in time — exactly
which memories are *authorized and current* (the allowed set) and which are
*forbidden* (everything else). These labels are derived, not hand-assigned, so
scoring never depends on an LLM judge and is fully reproducible.

Governance state as of ``as_of`` for one memory:

* **exists**  — ``created_at <= as_of``.
* **scopes**  — ``{initial scope} | {promote.to_scope : promote.t <= as_of}``.
  Promotion *adds* a broader scope (private -> shared); the author, who is a
  member of the broader scope, keeps visibility.
* **superseded** — some ``supersede`` event with ``t <= as_of`` targeted it.
* **deleted** — some ``delete`` event with ``t <= as_of`` targeted it.

A memory is **authorized** for a principal when it exists, is not deleted,
belongs to the principal's tenant (hard tenant isolation), and shares at least
one scope with the principal. It is **allowed** for a query when it is
authorized and — unless the query is in ``history`` temporal mode — not
superseded. Everything else in the scenario is **forbidden** for that query.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from .schema import Memory, Principal, Scenario


@dataclass(frozen=True)
class MemoryState:
    """Governance state of one memory as of a specific time."""

    exists: bool
    scopes: frozenset[str]
    superseded: bool
    deleted: bool


def memory_state(scenario: Scenario, memory_id: str, as_of: datetime) -> MemoryState:
    """Compute the governance state of ``memory_id`` as of ``as_of``."""

    mem = scenario.memories[memory_id]
    scopes = {mem.scope}
    superseded = False
    deleted = False
    for ev in scenario.events:  # events are pre-sorted by time in Scenario
        if ev.memory_id != memory_id or ev.t > as_of:
            continue
        if ev.type == "promote" and ev.to_scope:
            scopes.add(ev.to_scope)
        elif ev.type == "supersede":
            superseded = True
        elif ev.type == "delete":
            deleted = True
    return MemoryState(
        exists=mem.created_at <= as_of,
        scopes=frozenset(scopes),
        superseded=superseded,
        deleted=deleted,
    )


def is_authorized(
    scenario: Scenario,
    principal: Principal,
    memory: Memory,
    state: MemoryState,
) -> bool:
    """Whether ``principal`` may access ``memory`` given its ``state``.

    Enforces, in order: existence, deletion, tenant isolation (hard), and scope
    membership. Tenant isolation is checked before scope so a cross-tenant
    memory is never authorized even if scope ids happen to collide.
    """

    if not state.exists or state.deleted:
        return False
    if memory.tenant != principal.tenant:
        return False
    return bool(set(principal.scopes) & state.scopes)


@dataclass(frozen=True)
class QueryLabels:
    """Policy-derived labels for a single query.

    * ``allowed`` — authorized and current (per temporal mode). The system
      *should* be able to return these; ``must_retrieve`` is the relevant subset.
    * ``forbidden`` — every other memory in the scenario. Returning any of
      these is a leak.
    * ``must_retrieve`` — ``relevant & allowed`` (the utility target).
    * ``cross_tenant`` — forbidden memories from another tenant (isolation).
    * ``superseded_forbidden`` — forbidden only because they are stale in
      current mode (conflict / stale-propagation signal).
    """

    allowed: frozenset[str]
    forbidden: frozenset[str]
    must_retrieve: frozenset[str]
    cross_tenant: frozenset[str]
    superseded_forbidden: frozenset[str]


def compute_query_labels(scenario: Scenario, query) -> QueryLabels:
    """Derive the allowed / forbidden / must-retrieve sets for ``query``."""

    principal = scenario.principals[query.principal]
    history_mode = query.temporal_mode == "history"

    allowed: set[str] = set()
    cross_tenant: set[str] = set()
    superseded_forbidden: set[str] = set()

    for mem_id, mem in scenario.memories.items():
        state = memory_state(scenario, mem_id, query.as_of)
        authorized = is_authorized(scenario, principal, mem, state)
        current_ok = history_mode or not state.superseded
        if authorized and current_ok:
            allowed.add(mem_id)
            continue
        # Forbidden: record *why* for the axis-specific metrics.
        if state.exists and mem.tenant != principal.tenant:
            cross_tenant.add(mem_id)
        if authorized and state.superseded and not history_mode:
            superseded_forbidden.add(mem_id)

    all_ids = set(scenario.memories.keys())
    forbidden = all_ids - allowed
    must_retrieve = set(query.relevant) & allowed
    return QueryLabels(
        allowed=frozenset(allowed),
        forbidden=frozenset(forbidden),
        must_retrieve=frozenset(must_retrieve),
        cross_tenant=frozenset(cross_tenant),
        superseded_forbidden=frozenset(superseded_forbidden),
    )
