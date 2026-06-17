"""Orchestrator for running MemoryAgentBench against AgentMemoryToolkit.

Validates required environment variables, ensures the MemoryAgentBench clone
has the AgentMemoryToolkit dispatch patches applied, optionally disables
change-feed thresholds for deterministic baseline runs, then invokes
``main.py`` from the MemoryAgentBench checkout.

Example::

    python -m benchmarks.memoryagentbench.run_benchmark \\
        --memoryagentbench /path/to/MemoryAgentBench \\
        --agent-config benchmarks/memoryagentbench/configs/AgentMemoryToolkit_gpt-4o-mini.yaml \\
        --dataset-config configs/data_conf/Accurate_Retrieval/Ruler/QA/Ruler_qa1_197k.yaml \\
        --max-test-queries-ablation 3
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

from .patch_agent import patch as patch_agent_text

REQUIRED_BASELINE_ENV = ("COSMOS_DB_ENDPOINT", "COSMOS_DB_DATABASE", "COSMOS_DB_CONTAINER")
REQUIRED_PROCESSING_ENV = ("ADF_ENDPOINT",)


def _ensure_env(required: tuple[str, ...]) -> None:
    missing = [name for name in required if not os.environ.get(name)]
    if missing:
        raise SystemExit(
            "Missing required environment variables: " + ", ".join(missing)
        )


def _ensure_patched(memoryagentbench_root: Path) -> None:
    agent_py = memoryagentbench_root / "agent.py"
    if not agent_py.is_file():
        raise SystemExit(f"MemoryAgentBench agent.py not found at {agent_py}")
    original = agent_py.read_text(encoding="utf-8")
    patched, messages = patch_agent_text(original)
    for m in messages:
        print(f"  [patch] {m}")
    if patched != original:
        backup = agent_py.with_suffix(agent_py.suffix + ".bak")
        if not backup.exists():
            backup.write_text(original, encoding="utf-8")
        agent_py.write_text(patched, encoding="utf-8")
        print(f"  [patch] applied to {agent_py}")
    else:
        print("  [patch] no changes needed")


def _disable_change_feed(env: dict[str, str]) -> None:
    for k in ("THREAD_SUMMARY_EVERY_N", "FACT_EXTRACTION_EVERY_N", "USER_SUMMARY_EVERY_N"):
        env.setdefault(k, "0")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--memoryagentbench",
        type=Path,
        required=True,
        help="Path to a local clone of HUST-AI-HYZ/MemoryAgentBench.",
    )
    parser.add_argument(
        "--agent-config",
        required=True,
        help="Path to the agent_config YAML (passed to MemoryAgentBench main.py).",
    )
    parser.add_argument(
        "--dataset-config",
        required=True,
        help="Path to the dataset_config YAML (passed to MemoryAgentBench main.py).",
    )
    parser.add_argument(
        "--max-test-queries-ablation",
        type=int,
        default=None,
        help="MemoryAgentBench's ablation flag for capping queries.",
    )
    parser.add_argument(
        "--require-processing",
        action="store_true",
        help="Also require ADF_ENDPOINT (use for facts/summary modes).",
    )
    parser.add_argument(
        "--allow-change-feed",
        action="store_true",
        help="Do not force change-feed thresholds to 0.",
    )
    parser.add_argument(
        "--python",
        default=sys.executable,
        help="Python interpreter to use for the MemoryAgentBench run.",
    )
    parser.add_argument(
        "--extra",
        nargs=argparse.REMAINDER,
        help="Extra args forwarded to MemoryAgentBench main.py after a '--' separator.",
    )
    args = parser.parse_args(argv)

    _ensure_env(REQUIRED_BASELINE_ENV)
    if args.require_processing:
        _ensure_env(REQUIRED_PROCESSING_ENV)

    mab_root = args.memoryagentbench.expanduser().resolve()
    if not mab_root.is_dir():
        raise SystemExit(f"MemoryAgentBench root not found: {mab_root}")

    print(f"[run] MemoryAgentBench root: {mab_root}")
    print("[run] applying patch (idempotent)")
    _ensure_patched(mab_root)

    # Make our adapter importable from inside the MAB run (its CWD is the
    # MAB checkout). We do that by adding our repo to PYTHONPATH.
    repo_root = Path(__file__).resolve().parents[2]
    env = os.environ.copy()
    existing = env.get("PYTHONPATH", "")
    pieces = [str(repo_root)]
    if existing:
        pieces.append(existing)
    env["PYTHONPATH"] = os.pathsep.join(pieces)
    env.setdefault("PYTHONUNBUFFERED", "1")

    if not args.allow_change_feed:
        _disable_change_feed(env)
        print("[run] change-feed thresholds forced to 0 for deterministic timing")

    cmd = [
        args.python,
        str(mab_root / "main.py"),
        "--agent_config", args.agent_config,
        "--dataset_config", args.dataset_config,
    ]
    if args.max_test_queries_ablation is not None:
        cmd += ["--max_test_queries_ablation", str(args.max_test_queries_ablation)]
    if args.extra:
        # argparse REMAINDER includes a leading '--'; drop it if present.
        extras = list(args.extra)
        if extras and extras[0] == "--":
            extras = extras[1:]
        cmd += extras

    print("[run] command:", " ".join(shutil.which(c) or c if i == 0 else c for i, c in enumerate(cmd)))
    print(f"[run] cwd: {mab_root}")
    return subprocess.call(cmd, cwd=str(mab_root), env=env)


if __name__ == "__main__":
    raise SystemExit(main())
