"""Shared-Memory Governance Benchmark (SMGB).

A system-agnostic benchmark for evaluating *shared* agent memory — memory
reused across agents, users, teams, tenants, and organizations — on the
governance axes that single-user memory benchmarks (LoCoMo, LongMemEval, DMR)
do not test: authorization-filtered retrieval, private->shared promotion,
tenant isolation, conflict/supersession, scope-aware deletion, and provenance.

See ``benchmarks/governance/README.md`` and
``Docs/shared_memory_governance_benchmark.md`` for the design.
"""

from __future__ import annotations

from .loader import ValidationError, load_scenarios, validate_all, validate_scenario
from .policy import compute_query_labels, is_authorized, memory_state
from .schema import (
    Event,
    Memory,
    Principal,
    Provenance,
    Query,
    Scenario,
    Scope,
)
from .scorer import (
    REFERENCE_RUNNERS,
    Report,
    Run,
    merge_reports,
    naive_global_run,
    naive_shared_run,
    oracle_provenance,
    oracle_run,
    score_scenario,
)

__all__ = [
    "Event",
    "Memory",
    "Principal",
    "Provenance",
    "Query",
    "Scenario",
    "Scope",
    "compute_query_labels",
    "is_authorized",
    "memory_state",
    "load_scenarios",
    "validate_all",
    "validate_scenario",
    "ValidationError",
    "REFERENCE_RUNNERS",
    "Report",
    "Run",
    "merge_reports",
    "naive_global_run",
    "naive_shared_run",
    "oracle_provenance",
    "oracle_run",
    "score_scenario",
]
