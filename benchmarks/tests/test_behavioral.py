"""Unit tests for the behavioral lost-update benchmark."""

from benchmarks.behavioral import (
    AppendLog,
    CASCell,
    LockedCell,
    MutableCell,
    run_pattern,
)


def test_mutable_cell_basic():
    c = MutableCell()
    assert c.read() == 0
    c.write(5)
    assert c.value() == 5


def test_cas_cell_rejects_stale_expected_version():
    c = CASCell()
    _, ver = c.read()
    assert c.compare_and_set(ver, 1) is True  # version matches
    assert c.compare_and_set(ver, 2) is False  # stale version rejected
    assert c.value() == 1


def test_append_log_counts_all():
    log = AppendLog()
    for i in range(10):
        log.append(i)
    assert log.count() == 10


def test_locked_pattern_has_no_lost_updates():
    # Serialized read-modify-write must preserve every contribution.
    res = run_pattern("locked", concurrency=8, increments_per_agent=20, think_delay=0.0)
    assert res.observed == res.expected
    assert res.lost_updates == 0


def test_cas_pattern_has_no_lost_updates():
    # Optimistic concurrency retries until every contribution lands.
    res = run_pattern("cas", concurrency=8, increments_per_agent=20, think_delay=0.0)
    assert res.observed == res.expected
    assert res.lost_updates == 0


def test_append_pattern_has_no_lost_updates():
    # Append-only is inherently concurrency-safe.
    res = run_pattern("append", concurrency=8, increments_per_agent=20, think_delay=0.0)
    assert res.observed == res.expected
    assert res.lost_updates == 0


def test_mutable_pattern_loses_updates_under_contention():
    # Mutate-in-place read-modify-write drops contributions when agents race.
    # A think delay widens the race window to make the loss deterministic.
    res = run_pattern("mutable", concurrency=16, increments_per_agent=30, think_delay=0.0005)
    assert res.observed < res.expected
    assert res.lost_updates > 0
    assert res.lost_update_rate > 0.0


def test_single_agent_never_loses_updates():
    # With one agent there is no contention, so even mutate-in-place is correct.
    res = run_pattern("mutable", concurrency=1, increments_per_agent=50, think_delay=0.0001)
    assert res.observed == res.expected
    assert res.lost_updates == 0
