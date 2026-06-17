"""Single-command orchestrator: MAB accuracy + consistency sweep + combined report.

Runs the full multi-agent shared-memory evaluation pipeline end-to-end:
1. Executes MemoryAgentBench to measure task accuracy
2. Runs the consistency probe sweep to measure staleness/anomalies
3. Combines both into a single leaderboard report

CLI::

    python -m benchmarks.orchestrate --mab-agent-config path/to/agent.yaml \\
        --mab-dataset-config path/to/data.yaml --out-dir ./eval_run
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Optional, Sequence

from .combine_report import build_combined_report, discover_mab_results, flat_rows, load_leaderboard, load_mab_result, write_csv, write_json
from .consistency.sweep import run_sweep, simulated_store_factory, write_leaderboard_csv


def run_mab(
    agent_config: str,
    dataset_config: str,
    output_dir: str,
    mab_path: Optional[str] = None,
    force: bool = False,
) -> bool:
    """Execute MemoryAgentBench. Return True on success."""
    cmd = [
        sys.executable,
        "-m",
        "main",
        "--agent_config",
        agent_config,
        "--dataset_config",
        dataset_config,
    ]
    if force:
        cmd.append("--force")

    cwd = mab_path or os.getcwd()
    try:
        result = subprocess.run(cmd, cwd=cwd, check=False)
        return result.returncode == 0
    except Exception as exc:
        print(f"MAB run failed: {exc}", file=sys.stderr)
        return False


def run_consistency_sweep(
    out_dir: str,
    levels: Sequence[str] = ("strong", "bounded_staleness", "session", "consistent_prefix", "eventual"),
    concurrency_levels: Sequence[int] = (1, 2, 4),
    keys: int = 4,
    ops_per_agent: int = 100,
) -> str:
    """Execute the consistency probe sweep. Return leaderboard path."""
    factory = simulated_store_factory()
    rows = run_sweep(
        factory,
        levels=levels,
        concurrency_levels=concurrency_levels,
        keys=keys,
        ops_per_agent=ops_per_agent,
    )
    leaderboard_path = os.path.join(out_dir, "leaderboard.csv")
    write_leaderboard_csv(rows, leaderboard_path)
    print(f"Wrote consistency leaderboard: {leaderboard_path}")
    return leaderboard_path


def run_combine(
    mab_dir: str, leaderboard_path: str, out_dir: str, *, level: Optional[str] = None
) -> tuple[str, Optional[str]]:
    """Combine MAB results with leaderboard. Return (json_path, csv_path)."""
    runs = [load_mab_result(p) for p in discover_mab_results(mab_dir)]
    leaderboard = load_leaderboard(leaderboard_path) if os.path.exists(leaderboard_path) else []

    report = build_combined_report(runs, leaderboard, level=level)
    json_path = os.path.join(out_dir, "combined_report.json")
    write_json(report, json_path)
    print(f"Wrote combined report: {json_path}")

    csv_rows = flat_rows(report)
    csv_path = os.path.join(out_dir, "combined.csv")
    write_csv(csv_rows, csv_path)
    print(f"Wrote combined CSV: {csv_path}")
    return json_path, csv_path


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Orchestrate the full multi-agent shared-memory evaluation pipeline."
    )
    parser.add_argument(
        "--mab-agent-config",
        required=False,
        help="Path to MemoryAgentBench agent config YAML.",
    )
    parser.add_argument(
        "--mab-dataset-config",
        required=False,
        help="Path to MemoryAgentBench dataset config YAML.",
    )
    parser.add_argument(
        "--mab-path",
        help="Path to MemoryAgentBench root (default: current directory).",
    )
    parser.add_argument(
        "--mab-only", action="store_true", help="Run only MemoryAgentBench, skip consistency."
    )
    parser.add_argument(
        "--consistency-only",
        action="store_true",
        help="Run only consistency sweep, skip MAB.",
    )
    parser.add_argument(
        "--skip-combine", action="store_true", help="Skip the combine step."
    )
    parser.add_argument(
        "--mab-force", action="store_true", help="Force rerun of MAB even if results exist."
    )
    parser.add_argument("--out-dir", default="./eval_run", help="Output directory.")
    parser.add_argument(
        "--consistency-level",
        help="Consistency level to align each run against (e.g., session).",
    )
    parser.add_argument(
        "--consistency-ops",
        type=int,
        default=100,
        help="Consistency probe operations per agent.",
    )
    parser.add_argument(
        "--consistency-concurrency",
        default="1,2,4",
        help="Consistency probe agent counts (comma-separated).",
    )
    args = parser.parse_args(argv)

    # Validate: MAB configs required unless consistency-only
    if not args.consistency_only:
        if not args.mab_agent_config or not args.mab_dataset_config:
            parser.error("--mab-agent-config and --mab-dataset-config required unless --consistency-only is set")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {out_dir}")

    success = True

    if not args.consistency_only:
        print("\n=== Running MemoryAgentBench ===")
        # The MAB output is written to agent_config's output_dir; discover it.
        agent_cfg_path = args.mab_agent_config
        try:
            import yaml
            with open(agent_cfg_path, "r") as fh:
                agent_cfg = yaml.safe_load(fh)
            mab_output_dir = agent_cfg.get("output_dir", "./outputs")
        except Exception as exc:
            print(f"Failed to read agent config: {exc}", file=sys.stderr)
            mab_output_dir = "./outputs"

        if not run_mab(
            args.mab_agent_config,
            args.mab_dataset_config,
            mab_output_dir,
            mab_path=args.mab_path,
            force=args.mab_force,
        ):
            print("MemoryAgentBench run failed.", file=sys.stderr)
            success = False

    if not args.mab_only:
        print("\n=== Running Consistency Sweep ===")
        concurrency_levels = [
            int(x) for x in args.consistency_concurrency.split(",") if x.strip()
        ]
        leaderboard_path = run_consistency_sweep(
            str(out_dir),
            concurrency_levels=concurrency_levels,
            ops_per_agent=args.consistency_ops,
        )

        if success and not args.skip_combine:
            print("\n=== Combining Results ===")
            try:
                mab_output_dir = "./outputs"
                if not args.consistency_only:
                    agent_cfg_path = args.mab_agent_config
                    try:
                        import yaml
                        with open(agent_cfg_path, "r") as fh:
                            agent_cfg = yaml.safe_load(fh)
                        mab_output_dir = agent_cfg.get("output_dir", "./outputs")
                    except Exception:
                        pass
                run_combine(
                    mab_output_dir,
                    leaderboard_path,
                    str(out_dir),
                    level=args.consistency_level,
                )
            except Exception as exc:
                print(f"Combine step failed: {exc}", file=sys.stderr)
                success = False

    status = "[OK] Pipeline succeeded" if success else "[FAIL] Pipeline failed"
    print(f"\n{status}")
    return 0 if success else 1


if __name__ == "__main__":
    raise SystemExit(main())
