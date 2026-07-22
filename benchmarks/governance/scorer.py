"""Scoring for the Shared-Memory Governance Benchmark (SMGB).

The scorer is **system-agnostic**: a system under test produces a *run* — for
each query, a ranked list of retrieved memory ids (and, optionally, the
provenance it claims for them). The scorer compares that run to the
policy-derived labels from :mod:`benchmarks.governance.policy` and reports:

* **utility** — ``recall@k`` over the authorized-and-relevant target set.
* **leakage** — fraction of queries that returned any forbidden memory, and the
  item-level leak count.
* **isolation** — cross-tenant memories returned (must be 0).
* **conflict / stale propagation** — superseded ("ghost") memories returned in
  current mode.
* **provenance** — for provenance-axis queries, whether the claimed author for
  each retrieved target matches ground truth.

Three reference runners are provided so the metrics can be validated offline
without a live memory service, and to anchor a leaderboard:

* :func:`oracle_run` — returns exactly the authorized target (perfect).
* :func:`naive_shared_run` — returns everything in the principal's tenant,
  ignoring scope and supersession (no authorization filter) -> high leak.
* :func:`naive_global_run` — returns everything across all tenants -> also
  breaks isolation.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

from .policy import compute_query_labels, memory_state
from .schema import Scenario

# A run maps query_id -> ranked list of retrieved memory ids.
Run = dict[str, list[str]]
# Optional provenance claims: query_id -> memory_id -> {"author": str, ...}.
ProvenanceClaims = dict[str, dict[str, dict[str, Any]]]


def _nan_to_none(value: float) -> Optional[float]:
    """Map ``float('nan')`` to ``None`` so summaries stay valid JSON.

    ``mean_recall`` / ``leak_rate`` are ``NaN`` when there is nothing to average
    (no utility targets or no queries). ``json.dumps`` would serialize that as
    the bare token ``NaN``, which is rejected by strict JSON parsers, so we emit
    ``null`` instead.
    """
    return None if math.isnan(value) else value


@dataclass
class QueryScore:
    query_id: str
    axis: str
    recall: Optional[float]  # None when there is no utility target (abstention)
    leaked: bool
    leak_count: int
    isolation_violations: int
    stale_leak: int
    provenance_correct: Optional[int] = None
    provenance_total: Optional[int] = None


@dataclass
class Report:
    system: str
    k: int
    per_query: list[QueryScore] = field(default_factory=list)

    # -- aggregates --------------------------------------------------------
    def _recalls(self) -> list[float]:
        return [q.recall for q in self.per_query if q.recall is not None]

    @property
    def mean_recall(self) -> float:
        vals = self._recalls()
        return sum(vals) / len(vals) if vals else float("nan")

    @property
    def leak_rate(self) -> float:
        """Fraction of queries that returned at least one forbidden memory."""
        if not self.per_query:
            return float("nan")
        return sum(1 for q in self.per_query if q.leaked) / len(self.per_query)

    @property
    def total_leaks(self) -> int:
        return sum(q.leak_count for q in self.per_query)

    @property
    def isolation_violations(self) -> int:
        return sum(q.isolation_violations for q in self.per_query)

    @property
    def stale_leaks(self) -> int:
        return sum(q.stale_leak for q in self.per_query)

    @property
    def provenance_accuracy(self) -> Optional[float]:
        correct = sum(q.provenance_correct or 0 for q in self.per_query if q.provenance_total)
        total = sum(q.provenance_total or 0 for q in self.per_query if q.provenance_total)
        return correct / total if total else None

    def by_axis(self) -> dict[str, dict[str, float]]:
        """Per-axis rollup of the headline metrics."""
        axes: dict[str, list[QueryScore]] = {}
        for q in self.per_query:
            axes.setdefault(q.axis, []).append(q)
        out: dict[str, dict[str, float]] = {}
        for axis, items in sorted(axes.items()):
            recalls = [q.recall for q in items if q.recall is not None]
            out[axis] = {
                "n": len(items),
                "mean_recall": (sum(recalls) / len(recalls)) if recalls else float("nan"),
                "leak_rate": sum(1 for q in items if q.leaked) / len(items),
                "isolation_violations": sum(q.isolation_violations for q in items),
                "stale_leaks": sum(q.stale_leak for q in items),
            }
        return out

    def summary(self) -> dict[str, Any]:
        return {
            "system": self.system,
            "k": self.k,
            "queries": len(self.per_query),
            "mean_recall": _nan_to_none(self.mean_recall),
            "leak_rate": _nan_to_none(self.leak_rate),
            "total_leaks": self.total_leaks,
            "isolation_violations": self.isolation_violations,
            "stale_leaks": self.stale_leaks,
            "provenance_accuracy": self.provenance_accuracy,
        }


def score_scenario(
    scenario: Scenario,
    run: Run,
    *,
    system: str = "system",
    k: int = 10,
    provenance: Optional[ProvenanceClaims] = None,
) -> Report:
    """Score a single scenario's ``run`` against policy-derived labels."""

    report = Report(system=system, k=k)
    for query in scenario.queries:
        labels = compute_query_labels(scenario, query)
        retrieved = list(run.get(query.id, []))[:k]
        retrieved_set = set(retrieved)

        if labels.must_retrieve:
            recall = len(retrieved_set & labels.must_retrieve) / len(labels.must_retrieve)
        else:
            recall = None  # abstention / pure-leakage probe: no utility target

        leak_items = retrieved_set & labels.forbidden
        prov_correct: Optional[int] = None
        prov_total: Optional[int] = None
        if query.axis == "provenance":
            prov_correct, prov_total = _score_provenance(
                scenario, query, labels, retrieved_set, provenance
            )

        report.per_query.append(
            QueryScore(
                query_id=query.id,
                axis=query.axis,
                recall=recall,
                leaked=bool(leak_items),
                leak_count=len(leak_items),
                isolation_violations=len(retrieved_set & labels.cross_tenant),
                stale_leak=len(retrieved_set & labels.superseded_forbidden),
                provenance_correct=prov_correct,
                provenance_total=prov_total,
            )
        )
    return report


def _score_provenance(
    scenario: Scenario,
    query,
    labels,
    retrieved_set: set[str],
    provenance: Optional[ProvenanceClaims],
) -> tuple[int, int]:
    """Fraction of retrieved targets whose claimed author matches ground truth."""
    targets = labels.must_retrieve & retrieved_set
    if not targets:
        return 0, len(labels.must_retrieve)
    claims = (provenance or {}).get(query.id, {})
    correct = 0
    for mem_id in targets:
        gold_author = scenario.memories[mem_id].provenance.author
        claimed = claims.get(mem_id, {}).get("author")
        if claimed is not None and claimed == gold_author:
            correct += 1
    return correct, len(labels.must_retrieve)


def merge_reports(reports: list[Report]) -> Report:
    """Combine per-scenario reports for one system into a single report."""
    if not reports:
        raise ValueError("no reports to merge")
    merged = Report(system=reports[0].system, k=reports[0].k)
    for r in reports:
        merged.per_query.extend(r.per_query)
    return merged


# -- reference runners -----------------------------------------------------


def oracle_run(scenario: Scenario, k: int = 10) -> Run:
    """Perfect system: return exactly the authorized-and-relevant target set."""
    run: Run = {}
    for query in scenario.queries:
        labels = compute_query_labels(scenario, query)
        # ``must_retrieve`` is a frozenset, whose iteration order is not stable
        # across runs/versions. Rank the targets by their declared ``relevant``
        # order instead so the run is deterministic — this matters once a query
        # has more than ``k`` targets or when consumers treat rank as meaningful.
        must = labels.must_retrieve
        seen: set[str] = set()
        ordered: list[str] = []
        for mem_id in query.relevant:
            if mem_id in must and mem_id not in seen:
                seen.add(mem_id)
                ordered.append(mem_id)
        run[query.id] = ordered[:k]
    return run


def oracle_provenance(scenario: Scenario) -> ProvenanceClaims:
    """Gold provenance claims matching :func:`oracle_run` (all authors correct)."""
    claims: ProvenanceClaims = {}
    for query in scenario.queries:
        labels = compute_query_labels(scenario, query)
        claims[query.id] = {
            mem_id: {"author": scenario.memories[mem_id].provenance.author}
            for mem_id in labels.must_retrieve
        }
    return claims


def _tenant_dump(scenario: Scenario, query, *, all_tenants: bool) -> list[str]:
    principal = scenario.principals[query.principal]
    out: list[str] = []
    for mem_id, mem in scenario.memories.items():
        state = memory_state(scenario, mem_id, query.as_of)
        if not state.exists or state.deleted:
            continue
        if not all_tenants and mem.tenant != principal.tenant:
            continue
        out.append(mem_id)
    return out


def naive_shared_run(scenario: Scenario, k: int = 10) -> Run:
    """No authorization filter: return everything in the principal's tenant.

    Ignores scope membership and supersession, so it recalls the target but
    leaks other-scope and stale memories. Tenant isolation is (accidentally)
    respected, so isolation violations stay 0 — isolating the leak signal.
    """
    return {
        q.id: _tenant_dump(scenario, q, all_tenants=False)[:k] for q in scenario.queries
    }


def naive_global_run(scenario: Scenario, k: int = 10) -> Run:
    """Worst case: return everything across all tenants -> breaks isolation."""
    return {
        q.id: _tenant_dump(scenario, q, all_tenants=True)[:k] for q in scenario.queries
    }


REFERENCE_RUNNERS: dict[str, Callable[[Scenario, int], Run]] = {
    "oracle": oracle_run,
    "naive_shared": naive_shared_run,
    "naive_global": naive_global_run,
}
