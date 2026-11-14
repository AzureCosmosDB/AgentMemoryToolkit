"""Unit tests for :class:`PromptyLoader`."""

from __future__ import annotations

from pathlib import Path

from azure.cosmos.agent_memory.services._pipeline_helpers import (
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
    # extract_memories bumped to v4 when episodic extraction was removed
    # (facts-only); the rest remain v1. Every shipped prompt must declare
    # *some* version.
    expected = {
        "extract_memories.prompty": "v4",
        "extract_episode.prompty": "v1",
        "dedup.prompty": "v1",
        "summarize.prompty": "v1",
        "summarize_update.prompty": "v1",
        "user_summary.prompty": "v1",
        "user_summary_update.prompty": "v1",
        "extract_memories-v3.prompty": "v5-contextual",
        "extract_procedure.prompty": "v6",
    }
    for filename, version in expected.items():
        assert loader.prompt_version(filename) == version


def test_extract_memories_v3_renders_its_context_inputs_and_defaults_them() -> None:
    loader = PromptyLoader()

    bare, _ = loader.prepare("extract_memories-v3.prompty", inputs={"transcript": "[user]: Turn a: hi"})
    full, params = loader.prepare(
        "extract_memories-v3.prompty",
        inputs={
            "transcript": "[user]: Turn a: hi",
            "previous_turns": "[user]: earlier",
            "thread_summary": "A summary.",
            "known_memories": "- (2026-03-02 · user) Known.",
        },
    )

    assert (
        "### Thread summary\n(none)\n\n### Earlier turns\n(none)\n\n### Stored memories\n(none)" in bare[-1]["content"]
    )
    assert "### Thread summary\nA summary.\n\n### Earlier turns\n[user]: earlier" in full[-1]["content"]
    assert "- (2026-03-02 · user) Known.\n\n## New lines\n[user]: Turn a: hi" in full[-1]["content"]
    assert params["max_completion_tokens"] == 16384 and params["seed"] == 42
