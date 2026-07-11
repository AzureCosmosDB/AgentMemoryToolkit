"""Tests for ``ProcessingPipeline.reconcile_memories`` (P0 dedup + conflict pass).

Covers:
* duplicate-only path
* contradiction-only path
* mixed pool with dangling-id resolution (a contradiction loser also a dup source)
* dangling collapse to no-op (winner and loser both absorbed into same merged doc)
* empty pool / single-fact no-op
* ``n`` cap honored
* ``_mark_superseded`` writes ``supersede_reason`` + ``superseded_at``
* exact-dedup short-circuit at extract time
* ``_normalize_for_hash`` + ``_content_hash`` helper stability

The pipeline is constructed via ``ProcessingPipeline.__new__`` and patched in
place to avoid requiring a real Cosmos / LLM / embeddings stack.
"""

from __future__ import annotations

import hashlib
import json
import logging
from unittest.mock import MagicMock

import pytest

from azure.cosmos.agent_memory._utils import _normalize_for_hash, compute_content_hash
from azure.cosmos.agent_memory.exceptions import ValidationError
from azure.cosmos.agent_memory.services.pipeline import PipelineService


@pytest.fixture(autouse=True)
def _pin_legacy_dedup_paths(monkeypatch):
    """Disable write-time in-place folding so the extract-path tests here
    exercise the plain ADD path deterministically."""
    monkeypatch.setattr(
        "azure.cosmos.agent_memory.thresholds.get_dedup_vector_enabled",
        lambda: False,
    )


def _make_pipeline() -> PipelineService:
    p = PipelineService.__new__(PipelineService)
    p._embeddings = MagicMock()
    p._embeddings.generate.return_value = [0.1] * 8
    p._upsert_memory = MagicMock(side_effect=lambda doc: doc)
    p._mark_superseded = MagicMock(return_value=True)
    p._container = MagicMock()
    p._memories_container = p._container
    p._turns_container = p._container
    p._summaries_container = p._container
    p._chat = MagicMock()
    return p


def _fact(fid: str, content: str, **extra) -> dict:
    base = {
        "id": fid,
        "user_id": "u1",
        "thread_id": extra.get("thread_id", "t1"),
        "type": "fact",
        "content": content,
        "confidence": extra.get("confidence", 0.8),
        "salience": extra.get("salience", 0.5),
        "tags": extra.get("tags", ["sys:fact"]),
        "source_memory_ids": extra.get("source_memory_ids", []),
        "created_at": extra.get("created_at", "2024-01-01T00:00:00+00:00"),
    }
    return base


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

    def test_writes_reason_duplicate_and_at(self):
        p = self._build()
        old = {"id": "f1", "_etag": "e1", "content": "x"}
        ok = p._mark_superseded(old, "f2", reason="duplicate")
        assert ok is True
        body = p._container.replace_item.call_args.kwargs["body"]
        assert body["superseded_by"] == "f2"
        assert body["supersede_reason"] == "duplicate"
        assert "superseded_at" in body and body["superseded_at"]

    def test_writes_reason_contradiction_and_at(self):
        p = self._build()
        old = {"id": "f1", "_etag": "e1", "content": "x"}
        ok = p._mark_superseded(old, "f2", reason="contradict")
        assert ok is True
        body = p._container.replace_item.call_args.kwargs["body"]
        assert body["supersede_reason"] == "contradict"


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

    def test_empty_pool(self):
        p = _make_pipeline()
        p._container.query_items.return_value = iter([])
        p._run_prompty = MagicMock()
        result = p.reconcile_memories("u1")
        assert result == {"kept": 0, "merged": 0, "contradicted": 0}
        p._run_prompty.assert_not_called()

    def test_single_fact_no_op(self):
        p = _make_pipeline()
        p._container.query_items.return_value = iter([_fact("f1", "User likes coffee")])
        p._run_prompty = MagicMock()
        result = p.reconcile_memories("u1")
        assert result == {"kept": 1, "merged": 0, "contradicted": 0}
        p._run_prompty.assert_not_called()

    def test_only_contradictions(self):
        p = _make_pipeline()
        facts = [
            _fact("f1", "User is vegetarian", created_at="2024-01-01T00:00:00+00:00"),
            _fact("f2", "User loves a good ribeye steak", created_at="2024-01-09T00:00:00+00:00"),
        ]
        p._container.query_items.return_value = iter(facts)
        p._run_prompty = MagicMock(
            return_value=json.dumps(
                {
                    "duplicate_groups": [],
                    "contradicted_pairs": [{"winner_id": "f2", "loser_id": "f1", "reason": "more recent"}],
                    "kept_ids": ["f2"],
                }
            )
        )

        result = p.reconcile_memories("u1")

        assert result == {"kept": 1, "merged": 0, "contradicted": 1}
        # No new doc upserted (contradiction never creates a merged doc)
        p._upsert_memory.assert_not_called()
        assert p._mark_superseded.call_count == 1
        call = p._mark_superseded.call_args
        assert call.args[0]["id"] == "f1"
        assert call.args[1] == "f2"
        assert call.kwargs["reason"] == "contradict"

    def test_n_cap_honored(self):
        """Custom ``n`` is interpolated into the SQL query's TOP clause."""
        p = _make_pipeline()
        captured_query: dict = {}

        def q(query, parameters=None, **kwargs):
            captured_query["sql"] = query
            return iter([])

        p._container.query_items.side_effect = q
        p._run_prompty = MagicMock()

        p.reconcile_memories("u1", n=7)

        assert "TOP 7" in captured_query["sql"]


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

    def test_extract_skips_when_content_hash_matches_existing(self):
        from azure.cosmos.agent_memory._utils import compute_content_hash

        p = self._build()
        existing_text = "User likes coffee"
        existing = [
            {
                "id": "fact_existing",
                "type": "fact",
                "content": existing_text,
                "content_hash": compute_content_hash(existing_text),
                "thread_id": "t1",
                "tags": ["sys:fact"],
            }
        ]
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
        p._load_existing_memories = MagicMock(return_value=existing)
        # Stub the LLM extraction to emit a duplicate fact (same text).
        p._run_prompty = MagicMock(
            return_value=json.dumps(
                {
                    "facts": [
                        {
                            "text": existing_text,
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

        out = p.extract_memories("u1", "t1")

        assert out["exact_dedup_skipped"] >= 1
        assert out["fact_count"] == 0
        # No new fact upserted (the only ADD got short-circuited).
        assert all(call.args[0].get("type") != "fact" for call in p._upsert_memory.call_args_list)

    def test_extract_writes_content_hash_on_new_facts(self):
        from azure.cosmos.agent_memory._utils import compute_content_hash

        p = self._build()
        p._load_existing_memories = MagicMock(return_value=[])
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
        p = self._build()
        text = "Always reply in Spanish"
        # Existing PROCEDURAL with that text - must NOT poison the FACT bucket.
        existing = [
            {
                "id": "proc_existing",
                "type": "procedural",
                "content": text,
                "content_hash": compute_content_hash(text),
                "thread_id": "__procedural__",
                "tags": ["sys:procedural"],
            }
        ]
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
        p._load_existing_memories = MagicMock(return_value=existing)
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


class TestReconcileSupersedeRaceCounting:
    """When ``_mark_superseded`` returns False (lost ETag race), the source
    must NOT be added to ``source_to_merged_id`` or counted as consumed -
    otherwise contradictions get redirected to a doc that doesn't claim
    the source, and ``kept`` undercounts."""

    def test_failed_supersede_does_not_consume_source(self):
        p = PipelineService.__new__(PipelineService)
        p._container = MagicMock()
        p._memories_container = p._container
        p._turns_container = p._container
        p._summaries_container = p._container
        facts = [
            _fact("f1", "alpha"),
            _fact("f2", "alpha-restated"),
            _fact("f3", "beta"),
        ]
        p._container.query_items.return_value = iter(facts)
        p._run_prompty = MagicMock(
            return_value=json.dumps(
                {
                    "duplicate_groups": [
                        {
                            "merged_content": "alpha (consolidated)",
                            "source_ids": ["f1", "f2"],
                            "confidence": 0.9,
                            "salience": 0.7,
                        }
                    ],
                    "contradicted_pairs": [],
                    "kept_ids": ["f3"],
                }
            )
        )
        p._upsert_memory = MagicMock()
        p._embeddings = MagicMock()
        p._embeddings.generate.return_value = [0.0]
        # Both supersede attempts lose the race.
        p._mark_superseded = MagicMock(return_value=False)
        result = p.reconcile_memories("u1")
        # Sources stay active: kept counts ALL three originals.
        assert result == {"kept": 3, "merged": 0, "contradicted": 0}


class TestReconcileWinnerValidation:
    """Hallucinated ``winner_id`` must be refused - never write a dangling
    ``superseded_by`` that breaks the audit trail."""

    def test_hallucinated_winner_id_skipped(self):
        p = PipelineService.__new__(PipelineService)
        p._container = MagicMock()
        p._memories_container = p._container
        p._turns_container = p._container
        p._summaries_container = p._container
        facts = [
            _fact("f1", "user is vegetarian"),
            _fact("f2", "user loves ribeye"),
        ]
        p._container.query_items.return_value = iter(facts)
        p._run_prompty = MagicMock(
            return_value=json.dumps(
                {
                    "duplicate_groups": [],
                    "contradicted_pairs": [
                        {
                            "winner_id": "fact_does_not_exist",  # hallucinated
                            "loser_id": "f1",
                            "reason": "x",
                        }
                    ],
                    "kept_ids": ["f1", "f2"],
                }
            )
        )
        p._upsert_memory = MagicMock()
        p._mark_superseded = MagicMock(return_value=True)
        p._embeddings = MagicMock()
        result = p.reconcile_memories("u1")
        # Refuse to write a dangling superseded_by pointer.
        p._mark_superseded.assert_not_called()
        assert result == {"kept": 2, "merged": 0, "contradicted": 0}


class TestReconcileFactsTextEscapesContent:
    """Content with ``"`` or ``|`` must not break the prompt grammar."""

    def test_special_chars_in_content_are_json_escaped(self):
        p = PipelineService.__new__(PipelineService)
        p._container = MagicMock()
        p._memories_container = p._container
        p._turns_container = p._container
        p._summaries_container = p._container
        facts = [
            _fact("f1", 'She said "hi" | weird'),
            _fact("f2", "normal text"),
        ]
        p._container.query_items.return_value = iter(facts)
        captured: dict[str, str] = {}

        def _capture(name, inputs):
            captured["facts_text"] = inputs["facts_text"]
            return json.dumps({"duplicate_groups": [], "contradicted_pairs": [], "kept_ids": ["f1", "f2"]})

        p._run_prompty = MagicMock(side_effect=_capture)
        p._upsert_memory = MagicMock()
        p._mark_superseded = MagicMock(return_value=True)
        p._embeddings = MagicMock()
        p.reconcile_memories("u1")
        # The embedded `"` must be escaped (\\") - not raw - and the
        # Content: field must remain JSON-quoted so the LLM can parse it
        # as a single string even though the original text contained ``|``.
        text = captured["facts_text"]
        line_one = text.splitlines()[0]
        assert '\\"hi\\"' in line_one
        # Quoted content block survives intact.
        assert 'Content: "She said \\"hi\\" | weird"' in line_one
        # The id, confidence, salience, created fields all still parseable
        # (4 well-defined separators after the json-quoted content block).
        assert line_one.startswith("1. ID: f1 | Content: ")
        assert " | Confidence: 0.8 | Salience: 0.5 | Created:" in line_one


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


class TestReconcileContradictionWinnerNotInKeptIds:
    """RD#5+#13: contradiction winners are absent from kept_ids - must NOT trigger warning."""

    def test_clean_contradiction_does_not_warn_about_kept_mismatch(self, caplog):
        import logging

        p = _make_pipeline()
        p._container.query_items.return_value = iter([_fact("w1", "A is true"), _fact("l1", "A is false")])
        p._run_prompty = MagicMock(
            return_value=json.dumps(
                {
                    "duplicate_groups": [],
                    "contradicted_pairs": [{"winner_id": "w1", "loser_id": "l1"}],
                    # LLM correctly omits w1 from kept_ids (it lives under contradicted_pairs).
                    "kept_ids": [],
                }
            )
        )
        with caplog.at_level(logging.WARNING, logger="azure.cosmos.agent_memory.pipeline"):
            result = p.reconcile_memories("u1")
        assert result["contradicted"] == 1
        # No "kept_ids mismatch" warnings on a clean LLM response.
        warns = [r for r in caplog.records if "kept_ids mismatch" in r.getMessage()]
        assert warns == []


class TestReconcileNullCheckUsesIsNull:
    """PR#1: query uses IS_NULL(c.superseded_by), not the broken `= null`."""

    def test_query_uses_is_null(self):
        p = _make_pipeline()
        p._container.query_items.return_value = iter([])
        p._run_prompty = MagicMock(return_value=json.dumps({}))
        p.reconcile_memories("u1")
        # query_items is called with the SQL string in the `query` kwarg.
        call = p._container.query_items.call_args
        sql = (call.kwargs.get("query") or call.args[0]) if call else ""
        assert "IS_NULL(c.superseded_by)" in sql
        assert "c.superseded_by = null" not in sql


class TestFactsTextHandlesNullConfidence:
    """Pool facts with ``confidence=None`` / ``salience=None`` (legacy
    docs from before these fields existed) must render as ``N/A`` in the
    prompt body, never as the literal string ``None``."""

    def test_none_fields_render_as_na_in_facts_text(self):
        p = _make_pipeline()
        captured_prompt: dict = {}

        def capture_prompty(name, inputs):
            captured_prompt["facts_text"] = inputs.get("facts_text", "")
            return json.dumps(
                {
                    "duplicate_groups": [],
                    "contradicted_pairs": [],
                    "kept_ids": ["f-null-1", "f-null-2"],
                }
            )

        p._run_prompty = MagicMock(side_effect=capture_prompty)
        legacy1 = _fact("f-null-1", "Legacy fact with no confidence", confidence=None, salience=None)
        legacy1["created_at"] = None
        legacy2 = _fact("f-null-2", "Another legacy fact", confidence=None, salience=None)
        legacy2["created_at"] = None
        p._container.query_items.return_value = iter([legacy1, legacy2])
        p.reconcile_memories("u1")
        text = captured_prompt["facts_text"]
        assert "Confidence: N/A" in text
        assert "Salience: N/A" in text
        assert "Created: N/A" in text
        assert "None" not in text


class TestExtractUpdateSelfCollapseGuard:
    """Procedural synthesis self-collapse guard: when synthesis would emit a
    proc id identical to the existing one, treat as a no-op. (Fact extract-time
    UPDATE was removed - facts/contradictions are reconciled, not extract-tagged.)"""

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
        p._load_existing_memories = MagicMock(return_value=[])
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
    """``reconcile.outcome`` structured log line is emitted on every exit path
    of ``reconcile_memories`` with timing + counts + prompt lineage."""

    @staticmethod
    def _outcome_records(caplog) -> list:
        return [
            r
            for r in caplog.records
            if r.name == "azure.cosmos.agent_memory.pipeline" and r.getMessage() == "reconcile.outcome"
        ]

    def test_reconcile_emits_outcome_log_line_on_success(self, caplog):
        p = _make_pipeline()
        facts = [
            _fact("f1", "User likes aisle seats", confidence=0.9, salience=0.7),
            _fact("f2", "User prefers aisle seats on flights", confidence=0.85, salience=0.65),
        ]
        p._container.query_items.return_value = iter(facts)
        p._run_prompty = MagicMock(
            return_value=json.dumps(
                {
                    "duplicate_groups": [
                        {
                            "merged_content": "User prefers aisle seats on flights",
                            "source_ids": ["f1", "f2"],
                            "confidence": 0.9,
                            "salience": 0.7,
                        }
                    ],
                    "contradicted_pairs": [],
                    "kept_ids": [],
                }
            )
        )
        p._upsert_memory = MagicMock(side_effect=lambda doc: doc)
        p._mark_superseded = MagicMock(return_value=True)

        with caplog.at_level(logging.INFO, logger="azure.cosmos.agent_memory.pipeline"):
            result = p.reconcile_memories("u1")

        records = self._outcome_records(caplog)
        assert len(records) == 1, f"expected exactly one reconcile.outcome record, got {len(records)}"
        rec = records[0]
        assert rec.operation == "reconcile_memories"
        assert rec.user_id == "u1"
        assert isinstance(rec.kept, int)
        assert isinstance(rec.merged, int)
        assert isinstance(rec.contradicted, int)
        assert rec.kept == result["kept"]
        assert rec.merged == result["merged"]
        assert rec.contradicted == result["contradicted"]
        assert rec.candidates_considered == len(facts)
        assert isinstance(rec.duration_ms, float)
        assert rec.duration_ms > 0.0
        assert rec.prompt_id == "dedup.prompty"
        assert rec.prompt_version == "v1"

    def test_reconcile_emits_outcome_log_line_on_zero_candidates(self, caplog):
        p = _make_pipeline()
        p._container.query_items.return_value = iter([])
        p._run_prompty = MagicMock()

        with caplog.at_level(logging.INFO, logger="azure.cosmos.agent_memory.pipeline"):
            result = p.reconcile_memories("u1")

        p._run_prompty.assert_not_called()
        assert result == {"kept": 0, "merged": 0, "contradicted": 0}

        records = self._outcome_records(caplog)
        assert len(records) == 1
        rec = records[0]
        assert rec.candidates_considered == 0
        assert rec.kept == 0
        assert rec.merged == 0
        assert rec.contradicted == 0
        assert rec.user_id == "u1"
        assert rec.operation == "reconcile_memories"
        assert isinstance(rec.duration_ms, float)
        assert rec.duration_ms > 0.0
        assert rec.prompt_id == "dedup.prompty"
        assert rec.prompt_version == "v1"

    def test_reconcile_duration_ms_is_positive_float(self, caplog):
        p = _make_pipeline()
        facts = [_fact("only", "User is left-handed")]
        p._container.query_items.return_value = iter(facts)
        p._run_prompty = MagicMock()

        with caplog.at_level(logging.INFO, logger="azure.cosmos.agent_memory.pipeline"):
            p.reconcile_memories("u1")

        records = self._outcome_records(caplog)
        assert len(records) == 1
        rec = records[0]
        assert isinstance(rec.duration_ms, float)
        assert rec.duration_ms > 0.0
        assert rec.candidates_considered == 1
