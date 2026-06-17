"""Unit tests for the client-centric consistency analyzer."""

from benchmarks.consistency.analyzer import analyze
from benchmarks.consistency.trace import TraceLog


def test_fresh_read_no_staleness():
    t = TraceLog()
    t.write("k", "a", 1, 0.0, 1.0)
    t.read("k", "b", 1, 2.0, 2.5)  # sees v1 after it committed
    r = analyze(t)
    assert r.stale_reads == 0
    assert r.delta_max == 0.0
    assert r.k_max == 0
    assert r.stale_read_rate == 0.0


def test_stale_read_detected_with_delta_and_k():
    t = TraceLog()
    t.write("k", "a", 1, 0.0, 1.0)  # v1 durable at t=1.0
    t.write("k", "a", 2, 1.5, 2.0)  # v2 durable at t=2.0
    t.read("k", "b", 1, 3.0, 3.2)   # at t=3.0 still sees v1 though v2 durable
    r = analyze(t)
    assert r.stale_reads == 1
    assert r.k_max == 1
    # delta = read.start(3.0) - oldest missed durable version (v2 @ 2.0) = 1.0
    assert abs(r.delta_max - 1.0) < 1e-9
    assert r.per_key["k"]["stale_reads"] == 1


def test_read_your_writes_violation():
    t = TraceLog()
    t.write("k", "a", 1, 0.0, 1.0)
    t.read("k", "a", None, 2.0, 2.1)  # agent a fails to see its own write
    r = analyze(t)
    assert r.read_your_writes_violations == 1
    assert r.read_misses == 1


def test_monotonic_reads_violation():
    t = TraceLog()
    t.write("k", "a", 1, 0.0, 1.0)
    t.write("k", "a", 2, 1.0, 2.0)
    t.read("k", "b", 2, 3.0, 3.1)  # sees v2
    t.read("k", "b", 1, 4.0, 4.1)  # later sees v1 -> goes backwards
    r = analyze(t)
    assert r.monotonic_reads_violations == 1


def test_empty_read_before_any_durable_write_not_stale():
    t = TraceLog()
    t.read("k", "a", None, 0.0, 0.1)  # nothing durable yet
    t.write("k", "a", 1, 1.0, 2.0)
    r = analyze(t)
    assert r.stale_reads == 0
    assert r.read_misses == 0


def test_report_to_dict_roundtrips_fields():
    t = TraceLog()
    t.write("k", "a", 1, 0.0, 1.0)
    t.read("k", "a", 1, 2.0, 2.1)
    d = analyze(t).to_dict()
    assert d["reads"] == 1 and d["writes"] == 1
    assert "delta_p95" in d and "monotonic_reads_violations" in d
