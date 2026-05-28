"""Unit tests for :class:`PromptyLoader`."""

from __future__ import annotations

from pathlib import Path

from agent_memory_toolkit.services._pipeline_helpers import (
    DEFAULT_PROMPT_VERSION,
    PromptyLoader,
    _read_prompty_version,
)


def test_prepare_returns_messages_and_params() -> None:
    loader = PromptyLoader()
    messages, params = loader.prepare(
        "summarize.prompty",
        inputs={"transcript": "[user]: hi"},
    )
    assert isinstance(messages, list) and messages
    assert isinstance(params, dict)


def test_prompt_version_reads_front_matter(tmp_path: Path) -> None:
    f = tmp_path / "t.prompty"
    f.write_text(
        "---\nname: t\nversion: v3\nmodel:\n  apiType: chat\n---\nsystem:\nhi\n",
        encoding="utf-8",
    )
    assert _read_prompty_version(f) == "v3"


def test_prompt_version_defaults_when_missing(tmp_path: Path) -> None:
    f = tmp_path / "t.prompty"
    f.write_text(
        "---\nname: t\nmodel:\n  apiType: chat\n---\nsystem:\nhi\n",
        encoding="utf-8",
    )
    assert _read_prompty_version(f) == DEFAULT_PROMPT_VERSION


def test_loader_prompt_version_is_cached(tmp_path: Path) -> None:
    f = tmp_path / "cached.prompty"
    f.write_text(
        "---\nname: cached\nversion: v7\nmodel:\n  apiType: chat\n---\nsystem:\nhi\n",
        encoding="utf-8",
    )
    loader = PromptyLoader(prompts_dir=str(tmp_path))
    assert loader.prompt_version("cached.prompty") == "v7"
    f.write_text(
        "---\nname: cached\nversion: v999\nmodel:\n  apiType: chat\n---\nsystem:\nhi\n",
        encoding="utf-8",
    )
    assert loader.prompt_version("cached.prompty") == "v7"


def test_all_shipped_prompts_declare_version() -> None:
    loader = PromptyLoader()
    for filename in (
        "extract_memories.prompty",
        "dedup.prompty",
        "summarize.prompty",
        "summarize_update.prompty",
        "user_summary.prompty",
        "user_summary_update.prompty",
        "synthesize_procedural.prompty",
    ):
        assert loader.prompt_version(filename) == "v1"
