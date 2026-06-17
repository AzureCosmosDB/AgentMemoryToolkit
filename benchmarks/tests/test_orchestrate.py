"""Unit tests for the orchestration of the full pipeline."""

import json
from pathlib import Path

from benchmarks.orchestrate import run_consistency_sweep, run_combine


def test_orchestrate_consistency_only(tmp_path):
    """Test consistency sweep + combine without MAB."""
    leaderboard_path = run_consistency_sweep(
        str(tmp_path),
        levels=("strong", "eventual"),
        concurrency_levels=(1,),
        keys=2,
        ops_per_agent=20,
    )
    assert Path(leaderboard_path).exists()
    lines = open(leaderboard_path, encoding="utf-8").read().strip().splitlines()
    assert len(lines) == 3  # header + 2 levels


def test_run_combine_with_empty_mab_dir(tmp_path):
    """Test combine gracefully handles missing MAB results."""
    lb_path = tmp_path / "leaderboard.csv"
    lb_path.write_text("level,concurrency,reads,writes,stale_reads,stale_read_rate,"
                       "delta_max,delta_mean,delta_p95,k_max,k_mean,"
                       "read_your_writes_violations,monotonic_reads_violations\n"
                       "strong,1,10,10,0,0.0,0.0,0.0,0.0,0,0.0,0,0\n")
    
    json_path, csv_path = run_combine(
        str(tmp_path), str(lb_path), str(tmp_path), level="strong"
    )
    
    # Should produce outputs even with 0 runs
    assert Path(json_path).exists()
    report = json.load(open(json_path))
    assert report["runs"] == []
    assert len(report["consistency"]) == 1
