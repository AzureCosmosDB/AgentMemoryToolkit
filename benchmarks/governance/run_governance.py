"""CLI: validate SMGB scenarios and score systems on the governance axes.

Examples
--------
Score the built-in reference runners (oracle + naive baselines) on the seed set::

    python -m benchmarks.governance.run_governance --reference

Validate the seed dataset only::

    python -m benchmarks.governance.run_governance --validate-only

Score an external system from a run file (JSON: ``{scenario_id: {query_id: [ids]}}``)::

    python -m benchmarks.governance.run_governance --run my_run.json --system my_system
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .loader import load_scenarios, validate_all
from .scorer import (
    REFERENCE_RUNNERS,
    merge_reports,
    oracle_provenance,
    score_scenario,
)

DEFAULT_DATA = Path(__file__).with_name("data") / "seed_scenarios.jsonl"


def _print_leaderboard(rows: list[dict]) -> None:
    cols = [
        ("system", 14, "s"),
        ("queries", 8, "d"),
        ("mean_recall", 12, ".3f"),
        ("leak_rate", 10, ".3f"),
        ("total_leaks", 12, "d"),
        ("isolation_violations", 21, "d"),
        ("stale_leaks", 12, "d"),
    ]
    header = "".join(f"{name:<{w}}" for name, w, _ in cols)
    print(header)
    print("-" * len(header))
    for r in rows:
        line = ""
        for name, w, fmt in cols:
            val = r.get(name)
            if val is None:
                cell = "n/a"
            elif fmt == "s":
                cell = str(val)
            else:
                cell = format(val, fmt)
            line += f"{cell:<{w}}"
        print(line)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Shared-Memory Governance Benchmark runner")
    ap.add_argument("--data", default=str(DEFAULT_DATA), help="scenario .jsonl file or directory")
    ap.add_argument("--reference", action="store_true", help="score the built-in reference runners")
    ap.add_argument("--run", help="JSON run file for an external system")
    ap.add_argument("--system", default="system", help="name for the --run system")
    ap.add_argument("--k", type=int, default=10, help="retrieval cutoff")
    ap.add_argument("--validate-only", action="store_true", help="validate scenarios and exit")
    ap.add_argument("--json-out", help="write the leaderboard summary to this JSON path")
    args = ap.parse_args(argv)

    scenarios = load_scenarios(args.data)
    problems = validate_all(scenarios)
    if problems:
        print(f"VALIDATION FAILED ({len(problems)} problem(s)):")
        for p in problems:
            print("  -", p)
        return 1
    print(f"Loaded {len(scenarios)} scenario(s); validation OK.")
    if args.validate_only:
        return 0

    rows: list[dict] = []

    if args.reference or not args.run:
        for name, runner in REFERENCE_RUNNERS.items():
            reports = []
            for s in scenarios:
                run = runner(s, args.k)
                prov = oracle_provenance(s) if name == "oracle" else None
                reports.append(score_scenario(s, run, system=name, k=args.k, provenance=prov))
            rows.append(merge_reports(reports).summary())

    if args.run:
        run_data = json.loads(Path(args.run).read_text(encoding="utf-8"))
        reports = []
        for s in scenarios:
            run = run_data.get(s.scenario_id, {})
            reports.append(score_scenario(s, run, system=args.system, k=args.k))
        rows.append(merge_reports(reports).summary())

    print()
    _print_leaderboard(rows)

    if args.json_out:
        Path(args.json_out).write_text(json.dumps(rows, indent=2), encoding="utf-8")
        print(f"\nWrote leaderboard to {args.json_out}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
