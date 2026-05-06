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

import json
from unittest.mock import MagicMock

import pytest

from agent_memory_toolkit import pipeline as pipeline_mod
from agent_memory_toolkit.exceptions import ValidationError
from agent_memory_toolkit.pipeline import ProcessingPipeline


def _make_pipeline() -> ProcessingPipeline:
    p = ProcessingPipeline.__new__(ProcessingPipeline)
    p._embeddings = MagicMock()
    p._embeddings.generate.return_value = [0.1] * 8
    p._upsert_memory = MagicMock(side_effect=lambda doc: doc)
    p._mark_superseded = MagicMock(return_value=True)
    p._container = MagicMock()
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
        assert pipeline_mod._normalize_for_hash("Hello   World") == "hello world"
        assert pipeline_mod._normalize_for_hash("  Hello\tWorld\n") == "hello world"
        assert pipeline_mod._normalize_for_hash("HELLO") == "hello"

    def test_normalize_for_hash_handles_empty(self):
        assert pipeline_mod._normalize_for_hash("") == ""
        assert pipeline_mod._normalize_for_hash("   ") == ""

    def test_content_hash_stable_across_paraphrase_whitespace_case(self):
        h1 = pipeline_mod._content_hash("User likes coffee")
        h2 = pipeline_mod._content_hash("user   LIKES coffee")
        h3 = pipeline_mod._content_hash("user likes coffee")
        assert h1 == h2 == h3
        # 32 hex chars
        assert len(h1) == 32
        all(c in "0123456789abcdef" for c in h1)

    def test_content_hash_distinguishes_distinct_contents(self):
        assert pipeline_mod._content_hash("a") != pipeline_mod._content_hash("b")


# ---------------------------------------------------------------------------
# _mark_superseded
# ---------------------------------------------------------------------------


class TestMarkSupersededReason:
    def _build(self) -> ProcessingPipeline:
        p = ProcessingPipeline.__new__(ProcessingPipeline)
        p._container = MagicMock()
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
        ok = p._mark_superseded(old, "f2", reason="contradiction")
        assert ok is True
        body = p._container.replace_item.call_args.kwargs["body"]
        assert body["supersede_reason"] == "contradiction"


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

    def test_only_duplicates(self):
        p = _make_pipeline()
        facts = [
            _fact("f1", "User prefers aisle seats on flights"),
            _fact("f2", "User likes aisle seats when flying"),
        ]
        p._container.query_items.return_value = iter(facts)
        p._run_prompty = MagicMock(
            return_value=json.dumps(
                {
                    "duplicate_groups": [
                        {
                            "merged_content": "User prefers aisle seats on flights",
                            "source_ids": ["f1", "f2"],
                            "confidence": 0.95,
                            "salience": 0.7,
                        }
                    ],
                    "contradicted_pairs": [],
                    "kept_ids": [],
                }
            )
        )

        result = p.reconcile_memories("u1")

        assert result == {"kept": 0, "merged": 2, "contradicted": 0}
        # merged doc upserted
        assert p._upsert_memory.call_count == 1
        merged_doc = p._upsert_memory.call_args.args[0]
        assert merged_doc["content"] == "User prefers aisle seats on flights"
        assert "f1" in merged_doc["supersedes_ids"] and "f2" in merged_doc["supersedes_ids"]
        # both sources marked superseded with reason=duplicate
        assert p._mark_superseded.call_count == 2
        for call in p._mark_superseded.call_args_list:
            assert call.kwargs["reason"] == "duplicate"

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
        assert call.kwargs["reason"] == "contradiction"

    def test_mixed_pool_with_dangling_resolution(self):
        """Loser of a contradiction is also a duplicate source.

        Pipeline must redirect the contradiction's ``loser_id`` through
        ``source_to_merged_id`` and supersede the *merged* doc, not the
        original (already-merged) source.
        """
        p = _make_pipeline()
        facts = [
            _fact("f1", "User prefers aisle seats on flights"),
            _fact("f2", "User likes aisle seats when flying"),
            _fact("f3", "User loves the window seat"),
        ]
        p._container.query_items.return_value = iter(facts)
        # Capture the merged doc fetched via the dangling-resolution query.
        # Configure side_effect to return facts first, then merged-doc lookup.
        first_call = iter(facts)
        merged_doc_holder: dict = {}

        def query_items_side_effect(query, parameters=None, **kwargs):
            # Pool query
            if "TOP" in query:
                return first_call
            # Dangling-id resolver query: returns the merged doc by id
            if merged_doc_holder.get("doc"):
                return iter([merged_doc_holder["doc"]])
            return iter([])

        p._container.query_items.side_effect = query_items_side_effect

        def upsert(doc):
            merged_doc_holder["doc"] = dict(doc)  # snapshot for resolver
            merged_doc_holder["doc"]["_etag"] = "merged-etag"
            return doc

        p._upsert_memory.side_effect = upsert

        p._run_prompty = MagicMock(
            return_value=json.dumps(
                {
                    "duplicate_groups": [
                        {
                            "merged_content": "User prefers aisle seats on flights",
                            "source_ids": ["f1", "f2"],
                            "confidence": 0.95,
                            "salience": 0.7,
                        }
                    ],
                    "contradicted_pairs": [
                        # Loser f2 was just merged into the merged doc;
                        # winner f3 must contradict the *merged* doc.
                        {"winner_id": "f3", "loser_id": "f2", "reason": "contradicts merged"}
                    ],
                    "kept_ids": ["f3"],
                }
            )
        )

        result = p.reconcile_memories("u1")

        assert result["merged"] == 2  # f1, f2 marked dup
        assert result["contradicted"] == 1  # merged doc marked contradiction
        # mark_superseded calls: f1 (dup), f2 (dup), then merged doc (contradiction)
        assert p._mark_superseded.call_count == 3
        last = p._mark_superseded.call_args_list[-1]
        # The third call should target the merged doc (fetched via resolver)
        # and use the merged record's id as the new winner id only if winner
        # also collapsed; here winner=f3 stays as-is.
        assert last.kwargs["reason"] == "contradiction"
        # winner remains f3 (was not in any dup group)
        assert last.args[1] == "f3"

    def test_dangling_collapses_to_no_op(self):
        """Both winner and loser absorbed into the same merged group → skip."""
        p = _make_pipeline()
        facts = [
            _fact("f1", "User likes coffee"),
            _fact("f2", "User likes coffee in the morning"),
        ]
        p._container.query_items.return_value = iter(facts)
        p._run_prompty = MagicMock(
            return_value=json.dumps(
                {
                    "duplicate_groups": [
                        {
                            "merged_content": "User likes coffee in the morning",
                            "source_ids": ["f1", "f2"],
                            "confidence": 0.9,
                            "salience": 0.6,
                        }
                    ],
                    "contradicted_pairs": [
                        # Both f1 and f2 collapse to the same merged id → skip
                        {"winner_id": "f1", "loser_id": "f2", "reason": "irrelevant"}
                    ],
                    "kept_ids": [],
                }
            )
        )

        result = p.reconcile_memories("u1")

        assert result["merged"] == 2
        assert result["contradicted"] == 0
        # No contradiction supersede call beyond the two duplicate marks
        assert p._mark_superseded.call_count == 2
        for call in p._mark_superseded.call_args_list:
            assert call.kwargs["reason"] == "duplicate"

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


# ---------------------------------------------------------------------------
# Exact-dedup short-circuit at extract time (Change 5)
# ---------------------------------------------------------------------------


class TestExactDedupShortCircuit:
    def _build(self) -> ProcessingPipeline:
        p = ProcessingPipeline.__new__(ProcessingPipeline)
        p._embeddings = MagicMock()
        p._embeddings.generate.return_value = [0.1] * 8
        p._container = MagicMock()
        p._chat = MagicMock()
        p._upsert_memory = MagicMock(side_effect=lambda doc: doc)
        p._mark_superseded = MagicMock(return_value=True)
        return p

    def test_extract_skips_when_content_hash_matches_existing(self):
        from agent_memory_toolkit.pipeline import _content_hash

        p = self._build()
        existing_text = "User likes coffee"
        existing = [
            {
                "id": "fact_existing",
                "type": "fact",
                "content": existing_text,
                "content_hash": _content_hash(existing_text),
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
                    "procedurals": [],
                    "episodics": [],
                }
            )
        )

        out = p.extract_memories("u1", "t1")

        assert out["exact_dedup_skipped"] >= 1
        assert out["facts_count"] == 0
        # No new fact upserted (the only ADD got short-circuited).
        assert all(call.args[0].get("type") != "fact" for call in p._upsert_memory.call_args_list)

    def test_extract_writes_content_hash_on_new_facts(self):
        from agent_memory_toolkit.pipeline import _content_hash

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
                    "procedurals": [],
                    "episodics": [],
                }
            )
        )

        p.extract_memories("u1", "t1")

        fact_docs = [c.args[0] for c in p._upsert_memory.call_args_list if c.args[0].get("type") == "fact"]
        assert len(fact_docs) == 1
        assert fact_docs[0]["content_hash"] == _content_hash("User loves tea")
