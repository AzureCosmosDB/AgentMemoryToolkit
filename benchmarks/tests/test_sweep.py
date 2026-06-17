"""Unit tests for the consistency level x concurrency sweep driver."""

from benchmarks.consistency.sweep import (
    SweepRow,
    run_sweep,
    simulated_store_factory,
    write_leaderboard_csv,
)


def test_run_sweep_produces_one_row_per_cell():
    factory = simulated_store_factory(delays={"strong": 0.0, "eventual": 0.0})
    rows = run_sweep(
        factory,
        levels=["strong", "eventual"],
        concurrency_levels=[1, 2],
        keys=2,
        ops_per_agent=10,
        seed=1,
    )
    assert len(rows) == 4
    assert {(r.level, r.concurrency) for r in rows} == {
        ("strong", 1),
        ("strong", 2),
        ("eventual", 1),
        ("eventual", 2),
    }
    assert all(isinstance(r, SweepRow) for r in rows)


def test_strong_zero_delay_has_no_stale_reads():
    factory = simulated_store_factory(delays={"strong": 0.0})
    rows = run_sweep(
        factory,
        levels=["strong"],
        concurrency_levels=[2, 4],
        keys=3,
        ops_per_agent=40,
        seed=2,
    )
    for r in rows:
        assert r.stale_reads == 0
        assert r.stale_read_rate == 0.0
        assert r.read_your_writes_violations == 0


def test_write_leaderboard_csv(tmp_path):
    factory = simulated_store_factory(delays={"strong": 0.0})
    rows = run_sweep(
        factory,
        levels=["strong"],
        concurrency_levels=[1],
        keys=2,
        ops_per_agent=5,
        seed=0,
    )
    out = tmp_path / "leaderboard.csv"
    write_leaderboard_csv(rows, out)
    lines = out.read_text(encoding="utf-8").strip().splitlines()
    assert lines[0].startswith("level,concurrency,reads,writes,stale_reads")
    assert len(lines) == 1 + len(rows)
