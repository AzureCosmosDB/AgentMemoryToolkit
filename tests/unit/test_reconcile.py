"""Tests for ``PipelineService.reconcile_memories`` and what the extract and supersede paths around it write.

Covers:
* argument validation
* ``reconcile_memories`` as one pass: it hands its arguments and ``deadline`` to ``consolidate_memories``,
  returns its counts unchanged and logs them (``reconcile.outcome``)
* ``_mark_superseded`` writes ``supersede_reason`` + ``superseded_at``
* exact-dedup short-circuit at extract time
* ``_normalize_for_hash`` + ``_content_hash`` helper stability

The pipeline is constructed via ``PipelineService.__new__`` and patched in
place to avoid requiring a real Cosmos / LLM / embeddings stack. ``test_conflict_links.py`` and
``test_reconcile_passes.py`` run the passes themselves.
"""

from __future__ import annotations

import hashlib
import json
import logging
from unittest.mock import MagicMock

import pytest

from azure.cosmos.agent_memory._utils import _normalize_for_hash, compute_content_hash
from azure.cosmos.agent_memory.exceptions import ValidationError
from azure.cosmos.agent_memory.services._dedup import dedup_result
from azure.cosmos.agent_memory.services.pipeline import PipelineService

NO_DEDUP = dedup_result(checked=0, retired_seeds=0, merged=0, compared=0, vetoed=0)


def _make_pipeline() -> PipelineService:
    p = PipelineService.__new__(PipelineService)
    p.consolidate_memories = MagicMock(return_value=dict(NO_DEDUP))
    p._container = MagicMock()
    p._memories_container = p._container
    p._turns_container = p._container
    p._summaries_container = p._container
    return p


# ---------------------------------------------------------------------------
# Helper-function unit tests
# ---------------------------------------------------------------------------


class TestNormalizeAndHash:
    def test_normalize_for_hash_lowercases_and_collapses_whitespace(self):
        assert _normalize_for_hash("Hello   World") == "hello world"
        assert _normalize_for_hash("  Hello\tWorld\n") == "hello world"
        assert _normalize_for_hash("HELLO") == "hello"

    def test_normalize_for_hash_handles_empty(self):
        assert _normalize_for_hash("") == ""
        assert _normalize_for_hash("   ") == ""

    def test_content_hash_stable_across_paraphrase_whitespace_case(self):
        h1 = compute_content_hash("User likes coffee")
        h2 = compute_content_hash("user   LIKES coffee")
        h3 = compute_content_hash("user likes coffee")
        assert h1 == h2 == h3
        # 32 hex chars
        assert len(h1) == 32
        assert all(c in "0123456789abcdef" for c in h1)

    def test_content_hash_distinguishes_distinct_contents(self):
        assert compute_content_hash("a") != compute_content_hash("b")


# ---------------------------------------------------------------------------
# _mark_superseded
# ---------------------------------------------------------------------------


class TestMarkSupersededReason:
    def _build(self) -> PipelineService:
        p = PipelineService.__new__(PipelineService)
        p._container = MagicMock()
        p._memories_container = p._container
        p._turns_container = p._container
        p._summaries_container = p._container
        return p

    @pytest.mark.parametrize("reason", ["duplicate", "combined", "procedure"])
    def test_writes_the_reason_it_is_given_and_when(self, reason):
        p = self._build()
        old = {"id": "f1", "_etag": "e1", "content": "x"}
        ok = p._mark_superseded(old, "f2", reason=reason)
        assert ok is True
        body = p._container.replace_item.call_args.kwargs["body"]
        assert body["superseded_by"] == "f2"
        assert body["supersede_reason"] == reason
        assert "superseded_at" in body and body["superseded_at"]


# ---------------------------------------------------------------------------
# reconcile_memories
# ---------------------------------------------------------------------------


class TestReconcileMemories:
    def test_validates_user_id(self):
        p = _make_pipeline()
        with pytest.raises(ValidationError):
            p.reconcile_memories("")

    def test_validates_n(self):
        p = _make_pipeline()
        with pytest.raises(ValidationError):
            p.reconcile_memories("u1", n=0)
        with pytest.raises(ValidationError):
            p.reconcile_memories("u1", n=-3)
        with pytest.raises(ValidationError, match="<= 500"):
            p.reconcile_memories("u1", n=501)

    def test_it_is_one_pass_and_returns_that_passs_counts_unchanged(self):
        p = _make_pipeline()
        counts = dedup_result(
            checked=4,
            retired_seeds=1,
            merged=1,
            compared=3,
            vetoed=0,
            conflicts=2,
            combined=2,
            synthesized=1,
            combine_groups=1,
            combine_failed=0,
        )
        p.consolidate_memories.return_value = dict(counts)
        p._run_prompty = MagicMock()

        result = p.reconcile_memories("u1", n=7, deadline=123.0)

        assert result == counts and "contradicted" not in result
        p.consolidate_memories.assert_called_once_with("u1", 7, memory_type="fact", deadline=123.0)
        p._container.query_items.assert_not_called()  # no second read of the user's facts
        p._run_prompty.assert_not_called()  # and no second model call


class TestExactDedupShortCircuit:
    def _build(self) -> PipelineService:
        p = PipelineService.__new__(PipelineService)
        p._embeddings = MagicMock()
        p._embeddings.generate.return_value = [0.1] * 8
        p._container = MagicMock()
        p._memories_container = p._container
        p._turns_container = p._container
        p._summaries_container = p._container
        p._chat = MagicMock()
        p._upsert_memory = MagicMock(side_effect=lambda doc: doc)
        p._create_memory = MagicMock(side_effect=lambda doc: doc)
        p._mark_superseded = MagicMock(return_value=True)
        return p

    def test_extract_skips_in_batch_duplicate_facts(self):
        p = self._build()
        dup_text = "User likes coffee"
        # extract_memories pulls turns directly from the container.
        turns = [
            {
                "id": "turn-1",
                "role": "user",
                "content": "I like coffee",
                "type": "turn",
                "created_at": "2024-01-01T00:00:00+00:00",
            }
        ]
        p._container.query_items.return_value = iter(turns)
        # Stub the LLM extraction to emit the SAME fact text twice in one batch;
        # the second is an in-batch exact duplicate and must be skipped. There is
        # no store-side preload query - cross-run duplicates are handled by the
        # deterministic-id create (409) at write time instead.
        p._run_prompty = MagicMock(
            return_value=json.dumps(
                {
                    "facts": [
                        {
                            "text": dup_text,
                            "confidence": 0.9,
                            "salience": 0.6,
                            "action": "ADD",
                            "tags": ["sys:fact"],
                        },
                        {
                            "text": dup_text,
                            "confidence": 0.9,
                            "salience": 0.6,
                            "action": "ADD",
                            "tags": ["sys:fact"],
                        },
                    ],
                    "procedural": [],
                    "episodic": [],
                }
            )
        )

        out = p.extract_memories("u1", "t1")

        assert out["exact_dedup_skipped"] >= 1
        # Exactly one fact survives the in-batch dedup.
        assert out["fact_count"] == 1

    def test_extract_writes_content_hash_on_new_facts(self):
        from azure.cosmos.agent_memory._utils import compute_content_hash

        p = self._build()
        turns = [
            {
                "id": "turn-1",
                "role": "user",
                "content": "I love tea",
                "type": "turn",
                "created_at": "2024-01-01T00:00:00+00:00",
            }
        ]
        p._container.query_items.return_value = iter(turns)
        p._run_prompty = MagicMock(
            return_value=json.dumps(
                {
                    "facts": [
                        {
                            "text": "User loves tea",
                            "confidence": 0.9,
                            "salience": 0.6,
                            "action": "ADD",
                            "tags": ["sys:fact"],
                        }
                    ],
                    "procedural": [],
                    "episodic": [],
                }
            )
        )

        p.extract_memories("u1", "t1")

        fact_docs = [c.args[0] for c in p._create_memory.call_args_list if c.args[0].get("type") == "fact"]
        assert len(fact_docs) == 1
        assert fact_docs[0]["content_hash"] == compute_content_hash("User loves tea")


class TestExactDedupCrossTypeIsolation:
    """Existing non-fact hashes must not silently drop an extracted fact."""

    def _build(self) -> PipelineService:
        p = PipelineService.__new__(PipelineService)
        p._embeddings = MagicMock()
        p._embeddings.generate.return_value = [0.1] * 8
        p._embeddings.generate_batch.return_value = [[0.1] * 8]
        p._container = MagicMock()
        p._memories_container = p._container
        p._turns_container = p._container
        p._summaries_container = p._container
        p._upsert_memory = MagicMock(side_effect=lambda doc: doc)
        p._create_memory = MagicMock(side_effect=lambda doc: doc)
        p._mark_superseded = MagicMock(return_value=True)
        return p

    def test_fact_not_dropped_when_only_procedural_has_same_hash(self):
        # With no store-side preload, extraction never cross-checks other types,
        # so a procedural doc sharing a fact's content_hash cannot affect the fact
        # bucket. A normal fact ADD is created and nothing is exact-dedup skipped.
        p = self._build()
        text = "Always reply in Spanish"
        p._container.query_items.return_value = iter(
            [
                {
                    "id": "turn-1",
                    "role": "user",
                    "content": "x",
                    "type": "turn",
                    "created_at": "2024-01-01T00:00:00+00:00",
                }
            ]
        )
        p._run_prompty = MagicMock(
            return_value=json.dumps(
                {
                    "facts": [
                        {
                            "text": text,
                            "confidence": 0.9,
                            "salience": 0.6,
                            "action": "ADD",
                            "tags": ["sys:fact"],
                        }
                    ],
                    "procedural": [],
                    "episodic": [],
                    "unclassified": [],
                }
            )
        )
        out = p.extract_memories("u1", "t1")
        assert out["exact_dedup_skipped"] == 0
        fact_docs = [c.args[0] for c in p._create_memory.call_args_list if c.args[0].get("type") == "fact"]
        assert len(fact_docs) == 1
        assert fact_docs[0]["content"] == text


class TestExtractEarlyReturnShape:
    """The no-memories early-return must include every key the success
    path returns; otherwise callers using ``result["exact_dedup_skipped"]``
    KeyError on empty threads."""

    def test_empty_thread_returns_full_dict_shape(self):
        p = PipelineService.__new__(PipelineService)
        p._container = MagicMock()
        p._memories_container = p._container
        p._turns_container = p._container
        p._summaries_container = p._container
        p._container.query_items.return_value = iter([])  # no items
        out = p.extract_memories("u1", "t-empty")
        for key in (
            "fact_count",
            "episodic_count",
            "updated_count",
            "exact_dedup_skipped",
        ):
            assert key in out, f"missing key: {key}"
            assert out[key] == 0


class TestDedupPoolSizeThreshold:
    def test_pool_size_default(self, monkeypatch):
        from azure.cosmos.agent_memory.thresholds import DEFAULT_DEDUP_POOL_SIZE, get_dedup_pool_size

        monkeypatch.delenv("DEDUP_POOL_SIZE", raising=False)
        assert get_dedup_pool_size() == DEFAULT_DEDUP_POOL_SIZE

    def test_pool_size_override(self, monkeypatch):
        from azure.cosmos.agent_memory.thresholds import get_dedup_pool_size

        monkeypatch.setenv("DEDUP_POOL_SIZE", "100")
        assert get_dedup_pool_size() == 100

    def test_pool_size_clamped_to_500(self, monkeypatch):
        from azure.cosmos.agent_memory.thresholds import get_dedup_pool_size

        monkeypatch.setenv("DEDUP_POOL_SIZE", "9999")
        assert get_dedup_pool_size() == 500

    def test_pool_size_zero_falls_back_to_default(self, monkeypatch):
        from azure.cosmos.agent_memory.thresholds import DEFAULT_DEDUP_POOL_SIZE, get_dedup_pool_size

        monkeypatch.setenv("DEDUP_POOL_SIZE", "0")
        assert get_dedup_pool_size() == DEFAULT_DEDUP_POOL_SIZE


class TestExtractUpdateSelfCollapseGuard:
    """Procedural synthesis self-collapse guard: when synthesis would emit a
    proc id identical to the existing one, treat as a no-op. (Fact extract-time
    UPDATE was removed - duplicates are folded and conflicts linked by reconcile, not extract-tagged.)"""

    def _build(self) -> PipelineService:
        p = PipelineService.__new__(PipelineService)
        p._embeddings = MagicMock()
        p._embeddings.generate.return_value = [[0.1] * 8]
        p._upsert_memory = MagicMock(side_effect=lambda doc: doc)
        p._mark_superseded = MagicMock(return_value=True)
        p._container = MagicMock()
        p._memories_container = p._container
        p._turns_container = p._container
        p._summaries_container = p._container
        p._chat = MagicMock()
        return p

    def test_procedural_update_with_self_referential_id_is_skipped(self):
        from azure.cosmos.agent_memory._utils import compute_content_hash
        from azure.cosmos.agent_memory.services.pipeline import _ID_SEED_SEP

        p = self._build()
        text = "Greet the user casually"
        seed = _ID_SEED_SEP.join(("u1", compute_content_hash(text)))
        det_id = f"proc_{hashlib.sha256(seed.encode()).hexdigest()[:32]}"

        turns = [
            {
                "id": "turn-1",
                "role": "user",
                "content": "be casual",
                "type": "turn",
                "created_at": "2024-01-01T00:00:00+00:00",
            }
        ]
        p._container.query_items = MagicMock(side_effect=[iter(turns), iter([])])
        p._run_prompty = MagicMock(
            return_value=json.dumps(
                {
                    "facts": [],
                    "procedural": [
                        {
                            "instruction": text,
                            "confidence": 0.9,
                            "salience": 0.6,
                            "action": "UPDATE",
                            "supersedes_id": det_id,
                            "tags": ["sys:procedural"],
                            "trigger": "any greeting",
                            "category": "communication",
                        }
                    ],
                    "episodic": [],
                }
            )
        )

        p.extract_memories("u1", "t1")

        assert p._mark_superseded.call_count == 0
        proc_upserts = [c for c in p._upsert_memory.call_args_list if c.args[0].get("type") == "procedural"]
        assert proc_upserts == []


class TestReconcileOutcomeTelemetry:
    """``reconcile.outcome`` structured log line is emitted by ``reconcile_memories``
    with timing, the pass's counts and the prompt lineage."""

    @staticmethod
    def _outcome_records(caplog) -> list:
        return [
            r
            for r in caplog.records
            if r.name == "azure.cosmos.agent_memory.pipeline" and r.getMessage() == "reconcile.outcome"
        ]

    def test_reconcile_emits_outcome_log_line_with_every_count_the_pass_returned(self, caplog):
        p = _make_pipeline()
        counts = dedup_result(
            checked=4,
            retired_seeds=1,
            merged=1,
            compared=3,
            vetoed=2,
            conflicts=2,
            combined=2,
            synthesized=1,
            combine_groups=2,
            combine_failed=1,
        )
        p.consolidate_memories.return_value = dict(counts)

        with caplog.at_level(logging.INFO, logger="azure.cosmos.agent_memory.pipeline"):
            result = p.reconcile_memories("u1")

        records = self._outcome_records(caplog)
        assert len(records) == 1, f"expected exactly one reconcile.outcome record, got {len(records)}"
        rec = records[0]
        assert rec.operation == "reconcile_memories"
        assert rec.user_id == "u1"
        assert (rec.facts_checked, rec.candidates_considered, rec.pairs_compared) == (4, 4, 3)
        for key in (
            "kept",
            "merged",
            "conflicts",
            "vetoed",
            "combined",
            "synthesized",
            "combine_groups",
            "combine_failed",
        ):
            assert getattr(rec, key) == result[key], key
        assert not hasattr(rec, "contradicted")
        assert isinstance(rec.duration_ms, float)
        assert rec.duration_ms >= 0.0
        assert rec.prompt_id == "dedup_verify.prompty"
        assert rec.prompt_version == "v1"

    def test_reconcile_emits_outcome_log_line_when_nothing_was_checked(self, caplog):
        p = _make_pipeline()

        with caplog.at_level(logging.INFO, logger="azure.cosmos.agent_memory.pipeline"):
            result = p.reconcile_memories("u1")

        assert result == NO_DEDUP
        records = self._outcome_records(caplog)
        assert len(records) == 1
        rec = records[0]
        assert (rec.candidates_considered, rec.facts_checked, rec.kept, rec.merged, rec.conflicts) == (0, 0, 0, 0, 0)
        assert rec.user_id == "u1"
        assert rec.operation == "reconcile_memories"
        assert isinstance(rec.duration_ms, float)
        assert rec.prompt_id == "dedup_verify.prompty"
