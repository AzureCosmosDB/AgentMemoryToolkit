from __future__ import annotations

import pytest

from azure.cosmos.agent_memory import thresholds
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


@pytest.mark.parametrize(
    ("env_name", "getter_name", "expected"),
    [
        ("FACT_EXTRACTION_EVERY_N", "get_fact_extraction_every_n", 1),
        ("THREAD_SUMMARY_EVERY_N", "get_thread_summary_every_n", 10),
        ("USER_SUMMARY_EVERY_N", "get_user_summary_every_n", 20),
        ("DEDUP_EVERY_N", "get_dedup_every_n", 5),
        ("DEDUP_POOL_SIZE", "get_dedup_pool_size", 50),
        ("PROCEDURAL_SYNTHESIS_AUTO", "get_procedural_synthesis_auto", True),
    ],
)
def test_env_config_getters_defaults(
    monkeypatch: pytest.MonkeyPatch,
    env_name: str,
    getter_name: str,
    expected: object,
) -> None:
    monkeypatch.delenv(env_name, raising=False)

    assert getattr(thresholds, getter_name)() == expected


@pytest.mark.parametrize(
    ("env_name", "getter_name", "raw", "expected"),
    [
        ("FACT_EXTRACTION_EVERY_N", "get_fact_extraction_every_n", "2", 2),
        ("THREAD_SUMMARY_EVERY_N", "get_thread_summary_every_n", "11", 11),
        ("USER_SUMMARY_EVERY_N", "get_user_summary_every_n", "21", 21),
        ("DEDUP_EVERY_N", "get_dedup_every_n", "3", 3),
        ("DEDUP_POOL_SIZE", "get_dedup_pool_size", "75", 75),
        ("PROCEDURAL_SYNTHESIS_AUTO", "get_procedural_synthesis_auto", "false", False),
    ],
)
def test_env_config_getters_parse_env(
    monkeypatch: pytest.MonkeyPatch,
    env_name: str,
    getter_name: str,
    raw: str,
    expected: object,
) -> None:
    monkeypatch.setenv(env_name, raw)

    assert getattr(thresholds, getter_name)() == expected


@pytest.mark.parametrize(
    ("env_name", "getter_name", "expected"),
    [
        ("FACT_EXTRACTION_EVERY_N", "get_fact_extraction_every_n", 1),
        ("THREAD_SUMMARY_EVERY_N", "get_thread_summary_every_n", 10),
        ("USER_SUMMARY_EVERY_N", "get_user_summary_every_n", 20),
        ("DEDUP_EVERY_N", "get_dedup_every_n", 5),
        ("DEDUP_POOL_SIZE", "get_dedup_pool_size", 50),
    ],
)
def test_int_getters_reject_negative(
    monkeypatch: pytest.MonkeyPatch,
    env_name: str,
    getter_name: str,
    expected: int,
) -> None:
    monkeypatch.setenv(env_name, "-1")

    assert getattr(thresholds, getter_name)() == expected


@pytest.mark.parametrize(
    ("env_name", "getter_name", "expected"),
    [
        ("FACT_EXTRACTION_EVERY_N", "get_fact_extraction_every_n", 1),
        ("THREAD_SUMMARY_EVERY_N", "get_thread_summary_every_n", 10),
        ("USER_SUMMARY_EVERY_N", "get_user_summary_every_n", 20),
        ("DEDUP_EVERY_N", "get_dedup_every_n", 5),
        ("DEDUP_POOL_SIZE", "get_dedup_pool_size", 50),
    ],
)
def test_int_getters_invalid_use_default(
    monkeypatch: pytest.MonkeyPatch,
    env_name: str,
    getter_name: str,
    expected: int,
) -> None:
    monkeypatch.setenv(env_name, "bogus")

    assert getattr(thresholds, getter_name)() == expected


def test_dedup_pool_size_clamps_high(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DEDUP_POOL_SIZE", "501")

    assert thresholds.get_dedup_pool_size() == 500


def test_dedup_pool_size_rejects_zero(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DEDUP_POOL_SIZE", "0")

    assert thresholds.get_dedup_pool_size() == 50


def test_procedural_synthesis_auto_invalid_uses_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PROCEDURAL_SYNTHESIS_AUTO", "bogus")

    assert thresholds.get_procedural_synthesis_auto() is True


def test_processor_owner_defaults_to_none(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("MEMORY_PROCESSOR_OWNER", raising=False)

    assert thresholds.get_processor_owner() is None


@pytest.mark.parametrize("raw", ["inprocess", "durable", "INPROCESS", "DURABLE"])
def test_processor_owner_parse_env(monkeypatch: pytest.MonkeyPatch, raw: str) -> None:
    monkeypatch.setenv("MEMORY_PROCESSOR_OWNER", raw)

    assert thresholds.get_processor_owner() == raw.lower()


def test_processor_owner_invalid_uses_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MEMORY_PROCESSOR_OWNER", "bogus")

    assert thresholds.get_processor_owner() is None


def test_internalized_getters_return_fixed_constants_and_ignore_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("EXTRACTION_BATCH_MAX_TOKENS", "999")
    monkeypatch.setenv("DEDUP_SIM_HIGH", "0.50")

    assert thresholds.get_extraction_batch_max_tokens() == 7000
    assert thresholds.get_dedup_sim_high() == 0.97


def test_dedup_vector_enabled_defaults_false_and_reads_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("DEDUP_VECTOR_ENABLED", raising=False)
    assert thresholds.get_dedup_vector_enabled() is False

    monkeypatch.setenv("DEDUP_VECTOR_ENABLED", "true")
    assert thresholds.get_dedup_vector_enabled() is True

    monkeypatch.setenv("DEDUP_VECTOR_ENABLED", "false")
    assert thresholds.get_dedup_vector_enabled() is False
