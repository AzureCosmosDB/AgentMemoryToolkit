"""Unit tests for the consistency probe and in-memory store adapter."""

from benchmarks.consistency.analyzer import analyze
from benchmarks.consistency.probe import InMemoryStoreAdapter, run_probe


def test_inmemory_visibility_delay():
    clk = {"t": 0.0}
    store = InMemoryStoreAdapter(visibility_delay=5.0, clock=lambda: clk["t"])
    store.write("k", 1, "a")
    assert store.read_latest("k", "b") is None  # not visible during the delay
    clk["t"] = 6.0
    assert store.read_latest("k", "b") == 1


def test_inmemory_no_delay_immediately_visible():
    store = InMemoryStoreAdapter(visibility_delay=0.0)
    store.write("k", 1, "a")
    store.write("k", 2, "a")
    assert store.read_latest("k", "b") == 2


def test_run_probe_wiring_and_counts():
    store = InMemoryStoreAdapter(visibility_delay=0.0)
    trace = run_probe(
        store,
        agents=["a", "b"],
        keys=["k1", "k2"],
        ops_per_agent=15,
        read_ratio=0.5,
        seed=7,
    )
    assert len(trace) == 30
    rep = analyze(trace)
    assert rep.reads + rep.writes == 30
    assert 0.0 <= rep.stale_read_rate <= 1.0
