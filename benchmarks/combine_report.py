"""Reducer: join MemoryAgentBench accuracy output with the consistency leaderboard.

MemoryAgentBench writes one ``*_results.json`` per (agent, dataset) run under
``<output_dir>/<dataset>/`` containing ``agent_config``, ``dataset_config``,
``averaged_metrics`` (accuracy metrics are scaled x100; ``*_len`` / ``*_time``
keys are raw), ``data``, and ``time_cost``.

The consistency sweep (``benchmarks.consistency.sweep``) writes a
``leaderboard.csv`` with Delta/k staleness and session-anomaly rows per
``(level, concurrency)``.

This module produces a single combined report per run: each run's accuracy and
timing alongside a consistency summary aligned by the run's writer concurrency
(``len(writer_agents)``), plus the full consistency table for context. Outputs a
structured JSON and an optional flat per-run CSV.

CLI::

    python -m benchmarks.combine_report --mab outputs/agent_memory_toolkit \\
        --leaderboard leaderboard.csv --out combined_report.json --csv combined.csv
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import os
from datetime import datetime, timezone
from typing import Any, Optional, Sequence

# Leaderboard columns and their numeric parsers.
_INT_COLS = {
    "concurrency",
    "reads",
    "writes",
    "stale_reads",
    "k_max",
    "read_your_writes_violations",
    "monotonic_reads_violations",
}
_FLOAT_COLS = {"stale_read_rate", "delta_max", "delta_mean", "delta_p95", "k_mean"}


def _coerce_agent_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [v.strip() for v in value.split(",") if v.strip()]
    if isinstance(value, (list, tuple)):
        return [str(v).strip() for v in value if str(v).strip()]
    return [str(value)]


def _to_number(value: str):
    try:
        if value is None or value == "":
            return None
        if "." in value or "e" in value.lower():
            return float(value)
        return int(value)
    except (ValueError, AttributeError):
        return value


def discover_mab_results(path: str) -> list[str]:
    """Return ``*_results.json`` paths under ``path`` (file or directory)."""
    if os.path.isdir(path):
        return sorted(glob.glob(os.path.join(path, "**", "*_results.json"), recursive=True))
    return [path]


def load_mab_result(path: str) -> dict[str, Any]:
    """Parse one MemoryAgentBench result file into a run summary dict."""
    with open(path, "r", encoding="utf-8") as fh:
        doc = json.load(fh)

    agent_cfg = doc.get("agent_config", {}) or {}
    dataset_cfg = doc.get("dataset_config", {}) or {}
    averaged = doc.get("averaged_metrics", {}) or {}

    accuracy = {
        k: v for k, v in averaged.items() if "_len" not in k and "_time" not in k
    }
    timing = {k: v for k, v in averaged.items() if "_len" in k or "_time" in k}

    data = doc.get("data") or []
    if data:
        num_queries = len(data)
    else:
        metrics = doc.get("metrics") or {}
        num_queries = max((len(v) for v in metrics.values()), default=0)

    time_cost = doc.get("time_cost") or []
    wall_clock_s = time_cost[-1] if time_cost else None

    writer_agents = _coerce_agent_list(agent_cfg.get("memory_toolkit_writer_agents"))
    query_agents = _coerce_agent_list(agent_cfg.get("memory_toolkit_query_agents"))

    return {
        "run_id": agent_cfg.get("memory_toolkit_run_id", "default"),
        "agent_name": agent_cfg.get("agent_name", "unknown"),
        "model": agent_cfg.get("model", "unknown"),
        "dataset": dataset_cfg.get("dataset", "unknown"),
        "sub_dataset": dataset_cfg.get("sub_dataset", "unknown"),
        "store_mode": agent_cfg.get("memory_toolkit_store_mode", "turns_only"),
        "search_mode": agent_cfg.get("memory_toolkit_search_mode", "vector"),
        "writer_agents": writer_agents,
        "query_agents": query_agents,
        "retrieval_scope": "per_agent" if query_agents else "shared",
        "concurrency": len(writer_agents) or 1,
        "num_queries": num_queries,
        "wall_clock_s": wall_clock_s,
        "accuracy": accuracy,
        "timing": timing,
        "source_path": path,
    }


def load_leaderboard(path: str) -> list[dict[str, Any]]:
    """Read a consistency ``leaderboard.csv`` into typed row dicts."""
    rows: list[dict[str, Any]] = []
    with open(path, "r", encoding="utf-8", newline="") as fh:
        for raw in csv.DictReader(fh):
            row: dict[str, Any] = {}
            for key, value in raw.items():
                if key in _INT_COLS or key in _FLOAT_COLS:
                    row[key] = _to_number(value)
                else:
                    row[key] = value
            rows.append(row)
    return rows


def _consistency_summary(
    rows: Sequence[dict[str, Any]], concurrency: int, level: Optional[str]
) -> dict[str, Any]:
    """Summarize leaderboard rows for a run's concurrency (and optional level).

    Prefers an exact ``(level, concurrency)`` match; otherwise reports the
    worst-case across the candidate rows so a run is never silently dropped.
    """
    if not rows:
        return {"matched": False, "note": "no leaderboard rows"}

    candidates = rows
    if level is not None:
        candidates = [r for r in candidates if r.get("level") == level]
    conc_matches = [r for r in candidates if r.get("concurrency") == concurrency]

    if level is not None and len(conc_matches) == 1:
        row = conc_matches[0]
        return {
            "matched": True,
            "level": row.get("level"),
            "concurrency": row.get("concurrency"),
            "stale_read_rate": row.get("stale_read_rate"),
            "delta_p95": row.get("delta_p95"),
            "k_max": row.get("k_max"),
            "read_your_writes_violations": row.get("read_your_writes_violations"),
            "monotonic_reads_violations": row.get("monotonic_reads_violations"),
        }

    pool = conc_matches or candidates or rows
    return {
        "matched": False,
        "level": level or "worst_case",
        "concurrency": concurrency,
        "stale_read_rate": max((r.get("stale_read_rate") or 0.0) for r in pool),
        "delta_p95": max((r.get("delta_p95") or 0.0) for r in pool),
        "k_max": max((r.get("k_max") or 0) for r in pool),
        "read_your_writes_violations": sum(
            (r.get("read_your_writes_violations") or 0) for r in pool
        ),
        "monotonic_reads_violations": sum(
            (r.get("monotonic_reads_violations") or 0) for r in pool
        ),
        "note": "aggregated worst-case (no unique level+concurrency match)",
    }


def build_combined_report(
    runs: Sequence[dict[str, Any]],
    leaderboard_rows: Sequence[dict[str, Any]],
    *,
    run_id: Optional[str] = None,
    level: Optional[str] = None,
) -> dict[str, Any]:
    """Assemble the combined per-run report joining accuracy and consistency."""
    selected = [r for r in runs if run_id is None or r["run_id"] == run_id]
    enriched = []
    for run in selected:
        run = dict(run)
        run["consistency_summary"] = _consistency_summary(
            leaderboard_rows, run["concurrency"], level
        )
        enriched.append(run)

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "run_id_filter": run_id,
        "consistency_level": level,
        "runs": enriched,
        "consistency": list(leaderboard_rows),
    }


def _primary_accuracy(accuracy: dict[str, Any]) -> tuple[str, Any]:
    if not accuracy:
        return "", None
    metric = max(accuracy, key=lambda k: accuracy[k])
    return metric, accuracy[metric]


def flat_rows(report: dict[str, Any]) -> list[dict[str, Any]]:
    """Flatten the combined report into one row per run for CSV output."""
    out: list[dict[str, Any]] = []
    for run in report.get("runs", []):
        cons = run.get("consistency_summary", {})
        metric, value = _primary_accuracy(run.get("accuracy", {}))
        out.append(
            {
                "run_id": run["run_id"],
                "agent_name": run["agent_name"],
                "model": run["model"],
                "dataset": run["dataset"],
                "sub_dataset": run["sub_dataset"],
                "store_mode": run["store_mode"],
                "search_mode": run["search_mode"],
                "retrieval_scope": run["retrieval_scope"],
                "writer_agents": ";".join(run["writer_agents"]),
                "query_agents": ";".join(run["query_agents"]),
                "concurrency": run["concurrency"],
                "num_queries": run["num_queries"],
                "wall_clock_s": run["wall_clock_s"],
                "primary_accuracy_metric": metric,
                "primary_accuracy_value": value,
                "accuracy_json": json.dumps(run.get("accuracy", {}), sort_keys=True),
                "cons_level": cons.get("level"),
                "cons_concurrency": cons.get("concurrency"),
                "cons_stale_read_rate": cons.get("stale_read_rate"),
                "cons_delta_p95": cons.get("delta_p95"),
                "cons_k_max": cons.get("k_max"),
                "cons_read_your_writes_violations": cons.get("read_your_writes_violations"),
                "cons_monotonic_reads_violations": cons.get("monotonic_reads_violations"),
                "cons_matched": cons.get("matched"),
            }
        )
    return out


def write_json(report: dict[str, Any], path: str) -> str:
    with open(str(path), "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)
    return str(path)


def write_csv(rows: Sequence[dict[str, Any]], path: str) -> str:
    if not rows:
        with open(str(path), "w", encoding="utf-8", newline="") as fh:
            fh.write("")
        return str(path)
    with open(str(path), "w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    return str(path)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Join MemoryAgentBench accuracy results with the consistency leaderboard."
    )
    parser.add_argument(
        "--mab",
        required=True,
        help="MemoryAgentBench *_results.json file or a directory to scan recursively.",
    )
    parser.add_argument("--leaderboard", help="Path to a consistency leaderboard.csv.")
    parser.add_argument("--out", default="combined_report.json", help="Combined JSON output.")
    parser.add_argument("--csv", help="Optional flat per-run CSV output.")
    parser.add_argument("--run-id", help="Only include runs with this memory_toolkit_run_id.")
    parser.add_argument(
        "--level", help="Consistency level to align each run against (e.g. session)."
    )
    args = parser.parse_args(argv)

    runs = [load_mab_result(p) for p in discover_mab_results(args.mab)]
    leaderboard = load_leaderboard(args.leaderboard) if args.leaderboard else []

    report = build_combined_report(
        runs, leaderboard, run_id=args.run_id, level=args.level
    )
    out = write_json(report, args.out)
    print(f"Combined {len(report['runs'])} run(s) with {len(leaderboard)} leaderboard row(s) -> {out}")

    if args.csv:
        rows = flat_rows(report)
        csv_out = write_csv(rows, args.csv)
        print(f"Wrote {len(rows)} flat row(s) -> {csv_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
