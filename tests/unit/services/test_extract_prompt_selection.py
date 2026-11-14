"""Tests for the env-selectable fact-extraction prompt.

The v3 extractor is the shipped default; an env override can select v2 or v1, and an
unknown value falls back to the v3 default (safe allowlist).
"""

from __future__ import annotations

import pytest

from azure.cosmos.agent_memory.services._pipeline_helpers import extract_memories_prompt_file


def test_default_is_v3(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("AMT_EXTRACT_MEMORIES_PROMPT", raising=False)
    assert extract_memories_prompt_file() == "extract_memories-v3.prompty"


def test_env_override_selects_v1(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AMT_EXTRACT_MEMORIES_PROMPT", "extract_memories.prompty")
    assert extract_memories_prompt_file() == "extract_memories.prompty"


def test_env_override_selects_v2(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AMT_EXTRACT_MEMORIES_PROMPT", "extract_memories-v2.prompty")
    assert extract_memories_prompt_file() == "extract_memories-v2.prompty"


def test_unknown_value_falls_back_to_v3(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AMT_EXTRACT_MEMORIES_PROMPT", "bogus.prompty")
    assert extract_memories_prompt_file() == "extract_memories-v3.prompty"
