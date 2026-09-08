"""Integrity tests for the checked-in SMGB seed dataset."""

from pathlib import Path

from benchmarks.governance import (
    load_scenarios,
    naive_global_run,
    naive_shared_run,
    oracle_provenance,
    oracle_run,
    score_scenario,
    validate_all,
)
from benchmarks.governance.scorer import merge_reports

SEED = Path(__file__).resolve().parents[1] / "governance" / "data" / "seed_scenarios.jsonl"


def _load():
    return load_scenarios(SEED)


def test_seed_file_exists_and_loads():
    scenarios = _load()
    assert len(scenarios) >= 6
    ids = {s.scenario_id for s in scenarios}
    # The two whiteboard anchors must be present.
    assert "acct_satya_steve" in ids
    assert "proj_scott_deploy" in ids


def test_seed_hand_labels_match_policy():
    problems = validate_all(_load())
    assert problems == [], "\n".join(problems)


def test_every_axis_is_represented():
    axes = {q.axis for s in _load() for q in s.queries}
    for expected in {"utility", "leakage", "isolation", "promotion", "conflict", "deletion", "provenance"}:
        assert expected in axes, f"seed set is missing axis {expected!r}"


def test_oracle_perfect_on_seed():
    scenarios = _load()
    reports = [
        score_scenario(s, oracle_run(s), system="oracle", provenance=oracle_provenance(s))
        for s in scenarios
    ]
    su = merge_reports(reports).summary()
    assert su["mean_recall"] == 1.0
    assert su["leak_rate"] == 0.0
    assert su["isolation_violations"] == 0
    assert su["provenance_accuracy"] == 1.0


def test_naive_baselines_are_discriminated():
    scenarios = _load()
    shared = merge_reports(
        [score_scenario(s, naive_shared_run(s), system="naive_shared") for s in scenarios]
    ).summary()
    glob = merge_reports(
        [score_scenario(s, naive_global_run(s), system="naive_global") for s in scenarios]
    ).summary()
    # No authorization filter -> the leakage axis fires.
    assert shared["leak_rate"] > 0.0
    assert shared["total_leaks"] > 0
    # No tenant filter -> the isolation axis fires (and it only fires here).
    assert shared["isolation_violations"] == 0
    assert glob["isolation_violations"] > 0
