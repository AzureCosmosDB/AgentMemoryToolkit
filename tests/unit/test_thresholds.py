from __future__ import annotations

import pytest

from azure.cosmos.agent_memory.thresholds import (
    DEFAULT_ENABLE_TURN_EMBEDDINGS,
    DEFAULT_TTL_BY_TYPE,
    default_ttl_for,
    get_enable_turn_embeddings,
)


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


def test_enable_turn_embeddings_defaults_to_false(monkeypatch) -> None:
    monkeypatch.delenv("ENABLE_TURN_EMBEDDINGS", raising=False)
    assert DEFAULT_ENABLE_TURN_EMBEDDINGS is False
    assert get_enable_turn_embeddings() is False


@pytest.mark.parametrize("raw", ["true", "True", "1", "yes", "on"])
def test_enable_turn_embeddings_truthy_values(monkeypatch, raw) -> None:
    monkeypatch.setenv("ENABLE_TURN_EMBEDDINGS", raw)
    assert get_enable_turn_embeddings() is True


@pytest.mark.parametrize("raw", ["false", "False", "0", "no", "off"])
def test_enable_turn_embeddings_falsy_values(monkeypatch, raw) -> None:
    monkeypatch.setenv("ENABLE_TURN_EMBEDDINGS", raw)
    assert get_enable_turn_embeddings() is False
