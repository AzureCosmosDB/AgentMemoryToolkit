from __future__ import annotations

from agent_memory_toolkit.thresholds import DEFAULT_TTL_BY_TYPE, default_ttl_for


def test_default_ttl_table_values() -> None:
    assert DEFAULT_TTL_BY_TYPE == {
        "turn": 2_592_000,
        "episodic": 7_776_000,
        "thread_summary": -1,
        "user_summary": -1,
        "fact": -1,
        "procedural": -1,
    }


def test_default_ttl_for_expiring_types() -> None:
    assert default_ttl_for("turn") == 2_592_000
    assert default_ttl_for("episodic") == 7_776_000


def test_default_ttl_for_never_and_unknown_types() -> None:
    assert default_ttl_for("thread_summary") is None
    assert default_ttl_for("user_summary") is None
    assert default_ttl_for("fact") is None
    assert default_ttl_for("procedural") is None
    assert default_ttl_for("unknown") is None
