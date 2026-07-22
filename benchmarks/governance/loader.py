"""Loading and validation for SMGB scenarios.

Scenarios are stored as JSON Lines (one JSON object per line) so the dataset is
diff-friendly and easy to append to. The loader parses each line into a
:class:`~benchmarks.governance.schema.Scenario` and validates structural
integrity plus — crucially — that any hand-written ``must_retrieve`` /
``must_not_retrieve`` annotations agree with the labels the policy layer
derives. A mismatch is a dataset bug, surfaced early rather than silently
mis-scoring a system.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

from .policy import compute_query_labels
from .schema import QUERY_AXES, Scenario


class ValidationError(ValueError):
    """Raised when a scenario is structurally invalid or its labels disagree with policy."""


def load_scenarios(path: str | Path) -> list[Scenario]:
    """Load scenarios from a ``.jsonl`` file or a directory of ``.jsonl`` files."""
    p = Path(path)
    files = sorted(p.glob("*.jsonl")) if p.is_dir() else [p]
    scenarios: list[Scenario] = []
    for f in files:
        for lineno, line in enumerate(f.read_text(encoding="utf-8").splitlines(), start=1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            try:
                data = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValidationError(f"{f}:{lineno}: invalid JSON: {exc}") from exc
            scenarios.append(Scenario.from_dict(data))
    return scenarios


def validate_scenario(scenario: Scenario) -> list[str]:
    """Return a list of problems with ``scenario`` (empty list means valid)."""
    problems: list[str] = []
    sid = scenario.scenario_id

    # Referential integrity: every id referenced actually exists.
    for scope in scenario.scopes.values():
        if scope.tenant not in scenario.tenants:
            problems.append(f"{sid}: scope {scope.id!r} tenant {scope.tenant!r} not in tenants")
        for member in scope.members:
            if member not in scenario.principals:
                problems.append(f"{sid}: scope {scope.id!r} member {member!r} is not a principal")

    for principal in scenario.principals.values():
        if principal.tenant not in scenario.tenants:
            problems.append(
                f"{sid}: principal {principal.id!r} tenant {principal.tenant!r} not in tenants"
            )
        for scope_id in principal.scopes:
            if scope_id not in scenario.scopes:
                problems.append(
                    f"{sid}: principal {principal.id!r} references unknown scope {scope_id!r}"
                )
            elif scenario.scopes[scope_id].tenant != principal.tenant:
                problems.append(
                    f"{sid}: principal {principal.id!r} (tenant {principal.tenant!r}) references "
                    f"scope {scope_id!r} owned by tenant {scenario.scopes[scope_id].tenant!r}"
                )

    for mem in scenario.memories.values():
        if mem.scope not in scenario.scopes:
            problems.append(f"{sid}: memory {mem.id!r} initial scope {mem.scope!r} unknown")
        if mem.tenant not in scenario.tenants:
            problems.append(f"{sid}: memory {mem.id!r} tenant {mem.tenant!r} not in tenants")

    for ev in scenario.events:
        if ev.memory_id not in scenario.memories:
            problems.append(f"{sid}: event targets unknown memory {ev.memory_id!r}")
        if ev.type == "promote" and (not ev.to_scope or ev.to_scope not in scenario.scopes):
            problems.append(f"{sid}: promote event has unknown to_scope {ev.to_scope!r}")
        if ev.type == "supersede" and (not ev.by or ev.by not in scenario.memories):
            problems.append(f"{sid}: supersede event references unknown replacement {ev.by!r}")

    known_memories = set(scenario.memories)
    for q in scenario.queries:
        if q.principal not in scenario.principals:
            problems.append(f"{sid}: query {q.id!r} unknown principal {q.principal!r}")
            continue
        if q.axis not in QUERY_AXES:
            problems.append(f"{sid}: query {q.id!r} unknown axis {q.axis!r}")
        for rel in q.relevant:
            if rel not in scenario.memories:
                problems.append(f"{sid}: query {q.id!r} relevant id {rel!r} unknown")

        # The core check: hand annotations must agree with derived policy.
        labels = compute_query_labels(scenario, q)
        if q.must_retrieve is not None:
            hand = set(q.must_retrieve)
            unknown = hand - known_memories
            for mid in sorted(unknown):
                problems.append(f"{sid}: query {q.id!r} must_retrieve id {mid!r} unknown")
            # Compare only known ids: unknown ids can never appear in the
            # policy-derived set, so leaving them in would just double-report.
            if (hand - unknown) != set(labels.must_retrieve):
                problems.append(
                    f"{sid}: query {q.id!r} must_retrieve {sorted(hand - unknown)} != "
                    f"policy-derived {sorted(labels.must_retrieve)}"
                )
        if q.must_not_retrieve is not None:
            hand_not = set(q.must_not_retrieve)
            unknown = hand_not - known_memories
            for mid in sorted(unknown):
                problems.append(f"{sid}: query {q.id!r} must_not_retrieve id {mid!r} unknown")
            # Only real, known ids can be meaningfully "allowed"; excluding the
            # unknown ones avoids the misleading "actually allowed" message.
            stray = (hand_not - unknown) - set(labels.forbidden)
            if stray:
                problems.append(
                    f"{sid}: query {q.id!r} must_not_retrieve {sorted(stray)} are actually allowed"
                )

    return problems


def validate_all(scenarios: Iterable[Scenario]) -> list[str]:
    """Validate every scenario; return the combined problem list."""
    problems: list[str] = []
    for s in scenarios:
        problems.extend(validate_scenario(s))
    return problems
