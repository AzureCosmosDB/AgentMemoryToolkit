"""Tests for InProcessProcessor — verifies pipeline delegation order."""

from __future__ import annotations

from unittest.mock import MagicMock

from agent_memory_toolkit.processors import InProcessProcessor, ProcessThreadResult


def test_process_thread_calls_summarize_extract_dedup_in_order():
    pipeline = MagicMock()
    pipeline.generate_thread_summary.return_value = {"id": "summary_u_t", "type": "summary"}
    pipeline.extract_memories.return_value = {"facts": 2, "episodic": 1, "procedural": 0}
    pipeline.deduplicate_facts.return_value = {"merged": 2, "superseded": 1, "kept": 5}

    proc = InProcessProcessor(pipeline=pipeline)
    result = proc.process_thread(user_id="u1", thread_id="t1", turns=[])

    # Order of calls: summary -> extract -> dedup
    method_order = [c[0] for c in pipeline.method_calls]
    assert method_order == [
        "generate_thread_summary",
        "extract_memories",
        "deduplicate_facts",
    ]
    pipeline.generate_thread_summary.assert_called_once_with("u1", "t1")
    pipeline.extract_memories.assert_called_once_with("u1", "t1")
    pipeline.deduplicate_facts.assert_called_once_with("u1")

    assert isinstance(result, ProcessThreadResult)
    assert result.thread_summary == {"id": "summary_u_t", "type": "summary"}
    assert result.deduplicated_count == 3
    assert result.elapsed_ms >= 0


def test_process_thread_handles_non_dict_summary():
    pipeline = MagicMock()
    pipeline.generate_thread_summary.return_value = None
    pipeline.extract_memories.return_value = {"facts": 0}
    pipeline.deduplicate_facts.return_value = {}

    proc = InProcessProcessor(pipeline=pipeline)
    result = proc.process_thread(user_id="u1", thread_id="t1", turns=[])
    assert result.thread_summary is None
    assert result.deduplicated_count == 0


def test_extract_dedup_count_ignores_legacy_keys():
    """Legacy ``{"deduplicated": N}`` payloads should yield 0, not N.

    The pipeline's contract is ``{"kept", "merged", "superseded"}``;
    legacy keys are not honored so stale callers can't fake a count.
    """
    assert InProcessProcessor._extract_dedup_count({"deduplicated": 7}) == 0
    assert InProcessProcessor._extract_dedup_count({"merged": 3, "superseded": 2}) == 5
    assert InProcessProcessor._extract_dedup_count({"merged": 4}) == 4
    assert InProcessProcessor._extract_dedup_count({"superseded": 1}) == 1
    assert InProcessProcessor._extract_dedup_count({}) == 0
    assert InProcessProcessor._extract_dedup_count(None) == 0


def test_generate_user_summary_passes_thread_ids():
    pipeline = MagicMock()
    pipeline.generate_user_summary.return_value = {"id": "user_summary", "type": "user_summary"}

    proc = InProcessProcessor(pipeline=pipeline)
    summaries = [{"thread_id": "t1"}, {"thread_id": "t2"}, {"thread_id": ""}]
    res = proc.generate_user_summary(user_id="u1", thread_summaries=summaries)
    pipeline.generate_user_summary.assert_called_once_with("u1", ["t1", "t2"])
    assert res.summary == {"id": "user_summary", "type": "user_summary"}


def test_generate_user_summary_no_summaries():
    pipeline = MagicMock()
    pipeline.generate_user_summary.return_value = None
    proc = InProcessProcessor(pipeline=pipeline)
    res = proc.generate_user_summary(user_id="u1", thread_summaries=[])
    pipeline.generate_user_summary.assert_called_once_with("u1", None)
    assert res.summary is None


def test_close_is_noop():
    proc = InProcessProcessor(pipeline=MagicMock())
    assert proc.close() is None


def test_constructor_builds_pipeline_from_components():
    container = MagicMock()
    llm = MagicMock()
    embeddings = MagicMock()

    proc = InProcessProcessor(cosmos_container=container, chat_client=llm, embeddings_client=embeddings)
    # The pipeline should be a real ProcessingPipeline bound to those components
    assert proc._pipeline is not None
    assert proc._pipeline._container is container
    assert proc._pipeline._llm is llm
    assert proc._pipeline._embeddings is embeddings
