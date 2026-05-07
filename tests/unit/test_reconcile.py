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
from unittest.mock import MagicMock

import pytest

from agent_memory_toolkit._utils import _normalize_for_hash, compute_content_hash
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
        with pytest.raises(ValidationError, match="<= 500"):
            p.reconcile_memories("u1", n=501)

    def test_merge_falls_back_to_max_source_confidence_salience_when_llm_omits(self):
        """If the LLM omits or zero-fills confidence/salience on a duplicate
        group, the merged record must inherit max(source_*) so it doesn't
        silently drop below ``min_confidence`` / ``min_salience`` filters."""
        p = _make_pipeline()
        facts = [
            _fact("f1", "User likes aisle seats", confidence=0.92, salience=0.7),
            _fact("f2", "User prefers aisle seats on flights", confidence=0.85, salience=0.65),
        ]
        p._container.query_items.return_value = iter(facts)
        p._run_prompty = MagicMock(
            return_value=json.dumps(
                {
                    "duplicate_groups": [
                        {
                            "merged_content": "User prefers aisle seats",
                            "source_ids": ["f1", "f2"],
                            # confidence/salience deliberately omitted
                        }
                    ],
                    "contradicted_pairs": [],
                    "kept_ids": [],
                }
            )
        )
        p._upsert_memory = MagicMock()
        p._mark_superseded = MagicMock(return_value=True)
        p._embeddings = MagicMock()
        p._embeddings.generate.return_value = [0.0]
        p.reconcile_memories("u1")
        merged_doc = p._upsert_memory.call_args.args[0]
        assert merged_doc["confidence"] == 0.92  # max of 0.92, 0.85
        assert merged_doc["salience"] == 0.7  # max of 0.7, 0.65

    def test_merge_treats_zero_confidence_as_omitted(self):
        """gpt-4o-mini sometimes echoes the literal placeholder. A zero
        confidence/salience on the LLM output must trigger the same
        max(source_*) fallback as omission."""
        p = _make_pipeline()
        facts = [
            _fact("f1", "User likes coffee", confidence=0.9, salience=0.8),
            _fact("f2", "User loves coffee in the mornings", confidence=0.95, salience=0.85),
        ]
        p._container.query_items.return_value = iter(facts)
        p._run_prompty = MagicMock(
            return_value=json.dumps(
                {
                    "duplicate_groups": [
                        {
                            "merged_content": "User loves coffee in the mornings",
                            "source_ids": ["f1", "f2"],
                            "confidence": 0.0,
                            "salience": 0.0,
                        }
                    ],
                    "contradicted_pairs": [],
                    "kept_ids": [],
                }
            )
        )
        p._upsert_memory = MagicMock()
        p._mark_superseded = MagicMock(return_value=True)
        p._embeddings = MagicMock()
        p._embeddings.generate.return_value = [0.0]
        p.reconcile_memories("u1")
        merged_doc = p._upsert_memory.call_args.args[0]
        assert merged_doc["confidence"] == 0.95
        assert merged_doc["salience"] == 0.85

    def test_merged_doc_thread_id_picked_from_newest_source_by_ts(self):
        """Merged record's thread_id (partition key) must come from the
        source with the highest Cosmos ``_ts``, independent of the order
        the LLM lists ``source_ids`` in."""
        p = _make_pipeline()
        # f-old has lower _ts; f-new has higher. LLM lists them in the
        # "wrong" order (old first) — pipeline must still pick f-new's
        # thread_id for the merged doc.
        f_old = _fact("f-old", "User likes coffee", thread_id="thread-old")
        f_old["_ts"] = 100
        f_new = _fact("f-new", "User loves coffee", thread_id="thread-new")
        f_new["_ts"] = 999
        p._container.query_items.return_value = iter([f_new, f_old])
        p._run_prompty = MagicMock(
            return_value=json.dumps(
                {
                    "duplicate_groups": [
                        {
                            "merged_content": "User loves coffee",
                            "source_ids": ["f-old", "f-new"],
                            "confidence": 0.9,
                            "salience": 0.7,
                        }
                    ],
                    "contradicted_pairs": [],
                    "kept_ids": [],
                }
            )
        )
        p._upsert_memory = MagicMock()
        p._mark_superseded = MagicMock(return_value=True)
        p._embeddings = MagicMock()
        p._embeddings.generate.return_value = [0.0]
        p.reconcile_memories("u1")
        merged_doc = p._upsert_memory.call_args.args[0]
        assert merged_doc["thread_id"] == "thread-new"

    def test_synthetic_partition_is_user_scoped_when_no_thread_id(self):
        """If every source somehow lacks a thread_id, the synthetic
        fallback must be scoped per-user to avoid cross-tenant collisions."""
        p = _make_pipeline()
        f1 = _fact("f1", "alpha", thread_id="")
        f2 = _fact("f2", "alpha-restated", thread_id="")
        p._container.query_items.return_value = iter([f1, f2])
        p._run_prompty = MagicMock(
            return_value=json.dumps(
                {
                    "duplicate_groups": [
                        {
                            "merged_content": "alpha",
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
        p._upsert_memory = MagicMock()
        p._mark_superseded = MagicMock(return_value=True)
        p._embeddings = MagicMock()
        p._embeddings.generate.return_value = [0.0]
        p.reconcile_memories("user-abc")
        merged_doc = p._upsert_memory.call_args.args[0]
        assert merged_doc["thread_id"] == "__reconciled__:user-abc"

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

        # The dangling-loser redirect resolves through the in-memory
        # ``merged_docs_by_id`` cache populated when the merged doc is
        # upserted — no second Cosmos query is issued. The upsert
        # response carries the ``_etag`` that flows through the cache.
        def upsert(doc):
            snap = dict(doc)
            snap["_etag"] = "merged-etag"
            return snap

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
        from agent_memory_toolkit._utils import compute_content_hash

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
        assert out["facts_count"] == 0
        # No new fact upserted (the only ADD got short-circuited).
        assert all(call.args[0].get("type") != "fact" for call in p._upsert_memory.call_args_list)

    def test_extract_writes_content_hash_on_new_facts(self):
        from agent_memory_toolkit._utils import compute_content_hash

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

        fact_docs = [c.args[0] for c in p._upsert_memory.call_args_list if c.args[0].get("type") == "fact"]
        assert len(fact_docs) == 1
        assert fact_docs[0]["content_hash"] == compute_content_hash("User loves tea")


# ---------------------------------------------------------------------------
# Additional regression tests for round-16 review fixes.
# ---------------------------------------------------------------------------


class TestExactDedupCrossTypeIsolation:
    """Hash buckets must be type-scoped: a procedural with the same
    normalized text as an LLM-extracted fact (or vice versa) must NOT
    silently drop the new memory."""

    def _build(self) -> ProcessingPipeline:
        p = ProcessingPipeline.__new__(ProcessingPipeline)
        p._embeddings = MagicMock()
        p._embeddings.generate.return_value = [0.1] * 8
        p._embeddings.generate_batch.return_value = [[0.1] * 8]
        p._container = MagicMock()
        p._upsert_memory = MagicMock(side_effect=lambda doc: doc)
        p._mark_superseded = MagicMock(return_value=True)
        return p

    def test_fact_not_dropped_when_only_procedural_has_same_hash(self):
        p = self._build()
        text = "Always reply in Spanish"
        # Existing PROCEDURAL with that text — must NOT poison the FACT bucket.
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
        fact_docs = [c.args[0] for c in p._upsert_memory.call_args_list if c.args[0].get("type") == "fact"]
        assert len(fact_docs) == 1
        assert fact_docs[0]["content"] == text

    def test_procedural_not_dropped_when_only_fact_has_same_hash(self):
        p = self._build()
        text = "Use Celsius for temperatures"
        existing = [
            {
                "id": "fact_existing",
                "type": "fact",
                "content": text,
                "content_hash": compute_content_hash(text),
                "thread_id": "t1",
                "tags": ["sys:fact"],
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
                    "facts": [],
                    "procedural": [
                        {
                            "instruction": text,
                            "confidence": 0.9,
                            "salience": 0.6,
                            "action": "ADD",
                            "tags": [],
                        }
                    ],
                    "episodic": [],
                    "unclassified": [],
                }
            )
        )
        out = p.extract_memories("u1", "t1")
        assert out["exact_dedup_skipped"] == 0
        proc_docs = [c.args[0] for c in p._upsert_memory.call_args_list if c.args[0].get("type") == "procedural"]
        assert len(proc_docs) == 1

    def test_procedural_short_circuits_on_existing_procedural_hash(self):
        p = self._build()
        text = "Be terse in code reviews"
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
                    "facts": [],
                    "procedural": [
                        {
                            "instruction": text,
                            "confidence": 0.9,
                            "salience": 0.6,
                            "action": "ADD",
                            "tags": [],
                        }
                    ],
                    "episodic": [],
                    "unclassified": [],
                }
            )
        )
        out = p.extract_memories("u1", "t1")
        assert out["exact_dedup_skipped"] >= 1
        proc_docs = [c.args[0] for c in p._upsert_memory.call_args_list if c.args[0].get("type") == "procedural"]
        assert len(proc_docs) == 0


class TestExtractEarlyReturnShape:
    """The no-memories early-return must include every key the success
    path returns; otherwise callers using ``result["exact_dedup_skipped"]``
    KeyError on empty threads."""

    def test_empty_thread_returns_full_dict_shape(self):
        p = ProcessingPipeline.__new__(ProcessingPipeline)
        p._container = MagicMock()
        p._container.query_items.return_value = iter([])  # no items
        out = p.extract_memories("u1", "t-empty")
        for key in (
            "facts_count",
            "procedural_count",
            "episodic_count",
            "unclassified_count",
            "updated_count",
            "exact_dedup_skipped",
        ):
            assert key in out, f"missing key: {key}"
            assert out[key] == 0


class TestReconcileEmbeddingFailureAborts:
    """If embedding generation fails for the merged content, the duplicate
    group must be aborted entirely — no upsert, no supersede — so we don't
    create a search-index hole."""

    def test_embedding_failure_skips_upsert_and_supersede(self):
        p = ProcessingPipeline.__new__(ProcessingPipeline)
        p._container = MagicMock()
        facts = [
            _fact("f1", "alpha"),
            _fact("f2", "alpha-restated"),
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
                    "kept_ids": [],
                }
            )
        )
        p._upsert_memory = MagicMock()
        p._mark_superseded = MagicMock(return_value=True)
        p._embeddings = MagicMock()
        p._embeddings.generate.side_effect = RuntimeError("rate limit")
        result = p.reconcile_memories("u1")
        # Embedding failed → abort: nothing upserted, nothing superseded.
        p._upsert_memory.assert_not_called()
        p._mark_superseded.assert_not_called()
        assert result == {"kept": 2, "merged": 0, "contradicted": 0}


class TestReconcileSupersedeRaceCounting:
    """When ``_mark_superseded`` returns False (lost ETag race), the source
    must NOT be added to ``source_to_merged_id`` or counted as consumed —
    otherwise contradictions get redirected to a doc that doesn't claim
    the source, and ``kept`` undercounts."""

    def test_failed_supersede_does_not_consume_source(self):
        p = ProcessingPipeline.__new__(ProcessingPipeline)
        p._container = MagicMock()
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
    """Hallucinated ``winner_id`` must be refused — never write a dangling
    ``superseded_by`` that breaks the audit trail."""

    def test_hallucinated_winner_id_skipped(self):
        p = ProcessingPipeline.__new__(ProcessingPipeline)
        p._container = MagicMock()
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

    def test_resolved_winner_via_merge_redirect_is_accepted(self):
        """If winner_id refers to a fact that was just absorbed into a
        duplicate group, the merged_id must satisfy the validation."""
        p = ProcessingPipeline.__new__(ProcessingPipeline)
        p._container = MagicMock()
        facts = [
            _fact("f1", "alpha"),
            _fact("f2", "alpha-paraphrased"),
            _fact("f3", "contradicts alpha"),
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
                    "contradicted_pairs": [
                        # winner_id=f1 was absorbed into the merged group;
                        # redirect must resolve and validate cleanly.
                        {"winner_id": "f1", "loser_id": "f3", "reason": "x"},
                    ],
                    "kept_ids": [],
                }
            )
        )
        p._upsert_memory = MagicMock()
        p._mark_superseded = MagicMock(return_value=True)
        p._embeddings = MagicMock()
        p._embeddings.generate.return_value = [0.0]
        result = p.reconcile_memories("u1")
        assert result["contradicted"] == 1


class TestReconcileBoolNotNumeric:
    """``True`` and ``False`` are instances of ``int`` in Python — they must
    NOT be treated as numeric LLM-supplied confidence/salience."""

    def test_bool_confidence_falls_back_to_max_source(self):
        p = ProcessingPipeline.__new__(ProcessingPipeline)
        p._container = MagicMock()
        facts = [
            _fact("f1", "alpha", confidence=0.7, salience=0.5),
            _fact("f2", "alpha-restated", confidence=0.85, salience=0.6),
        ]
        p._container.query_items.return_value = iter(facts)
        p._run_prompty = MagicMock(
            return_value=json.dumps(
                {
                    "duplicate_groups": [
                        {
                            "merged_content": "alpha",
                            "source_ids": ["f1", "f2"],
                            "confidence": True,  # JSON boolean — must be ignored
                            "salience": False,
                        }
                    ],
                    "contradicted_pairs": [],
                    "kept_ids": [],
                }
            )
        )
        p._upsert_memory = MagicMock()
        p._mark_superseded = MagicMock(return_value=True)
        p._embeddings = MagicMock()
        p._embeddings.generate.return_value = [0.0]
        p.reconcile_memories("u1")
        merged_doc = p._upsert_memory.call_args.args[0]
        # NOT 1.0 (True coerced) and NOT 0.0 (False coerced); fallback to max source.
        assert merged_doc["confidence"] == 0.85
        assert merged_doc["salience"] == 0.6


class TestReconcileFactsTextEscapesContent:
    """Content with ``"`` or ``|`` must not break the prompt grammar."""

    def test_special_chars_in_content_are_json_escaped(self):
        p = ProcessingPipeline.__new__(ProcessingPipeline)
        p._container = MagicMock()
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
        # The embedded `"` must be escaped (\\") — not raw — and the
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
        from agent_memory_toolkit.thresholds import DEFAULT_DEDUP_POOL_SIZE, get_dedup_pool_size

        monkeypatch.delenv("DEDUP_POOL_SIZE", raising=False)
        assert get_dedup_pool_size() == DEFAULT_DEDUP_POOL_SIZE

    def test_pool_size_override(self, monkeypatch):
        from agent_memory_toolkit.thresholds import get_dedup_pool_size

        monkeypatch.setenv("DEDUP_POOL_SIZE", "100")
        assert get_dedup_pool_size() == 100

    def test_pool_size_clamped_to_500(self, monkeypatch):
        from agent_memory_toolkit.thresholds import get_dedup_pool_size

        monkeypatch.setenv("DEDUP_POOL_SIZE", "9999")
        assert get_dedup_pool_size() == 500

    def test_pool_size_zero_falls_back_to_default(self, monkeypatch):
        from agent_memory_toolkit.thresholds import DEFAULT_DEDUP_POOL_SIZE, get_dedup_pool_size

        monkeypatch.setenv("DEDUP_POOL_SIZE", "0")
        assert get_dedup_pool_size() == DEFAULT_DEDUP_POOL_SIZE


# ---------------------------------------------------------------------------
# Round 17 fixes
# ---------------------------------------------------------------------------


class TestReconcileMergedIdDeterministic:
    """RD#1: merged_id is deterministic on (user, content_hash) so cycles are idempotent."""

    def test_same_merged_content_yields_same_id_across_runs(self):
        import hashlib

        p = _make_pipeline()
        upserts: list[dict] = []
        p._upsert_memory = MagicMock(side_effect=lambda doc: upserts.append(doc) or doc)
        p._mark_superseded = MagicMock(return_value=True)
        p._container.query_items.return_value = iter(
            [_fact("a1", "User likes coffee"), _fact("a2", "User enjoys coffee")]
        )
        p._run_prompty = MagicMock(
            return_value=json.dumps(
                {
                    "duplicate_groups": [
                        {
                            "merged_content": "User likes coffee",
                            "source_ids": ["a1", "a2"],
                        }
                    ],
                    "contradicted_pairs": [],
                    "kept_ids": [],
                }
            )
        )
        p.reconcile_memories("u1")
        first_id = upserts[0]["id"]
        # Predict id from public formula:
        ch = compute_content_hash("User likes coffee")
        from agent_memory_toolkit.pipeline import _ID_SEED_SEP

        seed = _ID_SEED_SEP.join(("u1", "merged", ch))
        expected = "fact_" + hashlib.sha256(seed.encode()).hexdigest()[:32]
        assert first_id == expected
        # Second run: different source ids, identical canonical merged
        # content → identical merged id (idempotent upsert).
        upserts.clear()
        p._container.query_items.return_value = iter(
            [_fact("b1", "User likes coffee"), _fact("b2", "user LIKES coffee")]
        )
        p._run_prompty = MagicMock(
            return_value=json.dumps(
                {
                    "duplicate_groups": [
                        {
                            "merged_content": "User likes coffee",
                            "source_ids": ["b1", "b2"],
                        }
                    ],
                    "contradicted_pairs": [],
                    "kept_ids": [],
                }
            )
        )
        p.reconcile_memories("u1")
        assert upserts[0]["id"] == first_id


class TestReconcileSupersedesIdsFiltered:
    """RD#4: hallucinated source_ids are scrubbed from supersedes_ids."""

    def test_hallucinated_source_ids_filtered(self):
        p = _make_pipeline()
        upserts: list[dict] = []
        p._upsert_memory = MagicMock(side_effect=lambda doc: upserts.append(doc) or doc)
        p._mark_superseded = MagicMock(return_value=True)
        p._container.query_items.return_value = iter([_fact("real1", "X"), _fact("real2", "Y")])
        p._run_prompty = MagicMock(
            return_value=json.dumps(
                {
                    "duplicate_groups": [
                        {
                            "merged_content": "X and Y",
                            "source_ids": ["real1", "real2", "ghost_id_404"],
                        }
                    ],
                    "contradicted_pairs": [],
                    "kept_ids": [],
                }
            )
        )
        p.reconcile_memories("u1")
        assert upserts, "merged doc should have been upserted"
        assert "ghost_id_404" not in upserts[0]["supersedes_ids"]
        assert set(upserts[0]["supersedes_ids"]) == {"real1", "real2"}


class TestReconcileTransitiveSupersedes:
    """RD#1 follow-on: prior chain hops survive into the new merged record."""

    def test_prior_supersedes_ids_preserved_in_chain(self):
        p = _make_pipeline()
        upserts: list[dict] = []
        p._upsert_memory = MagicMock(side_effect=lambda doc: upserts.append(doc) or doc)
        p._mark_superseded = MagicMock(return_value=True)
        # f1 was itself a previously-merged record carrying its own provenance.
        f1 = _fact("f1", "X v1")
        f1["supersedes_ids"] = ["older_a", "older_b"]
        f2 = _fact("f2", "X v2")
        p._container.query_items.return_value = iter([f1, f2])
        p._run_prompty = MagicMock(
            return_value=json.dumps(
                {
                    "duplicate_groups": [{"merged_content": "X canonical", "source_ids": ["f1", "f2"]}],
                    "contradicted_pairs": [],
                    "kept_ids": [],
                }
            )
        )
        p.reconcile_memories("u1")
        sup = upserts[0]["supersedes_ids"]
        assert "f1" in sup and "f2" in sup
        assert "older_a" in sup and "older_b" in sup


class TestReconcileMergedMetadata:
    """RD#3: merged docs carry a positive merged_via signal."""

    def test_merged_doc_has_metadata(self):
        p = _make_pipeline()
        upserts: list[dict] = []
        p._upsert_memory = MagicMock(side_effect=lambda doc: upserts.append(doc) or doc)
        p._mark_superseded = MagicMock(return_value=True)
        p._container.query_items.return_value = iter([_fact("x1", "A"), _fact("x2", "B")])
        p._run_prompty = MagicMock(
            return_value=json.dumps(
                {
                    "duplicate_groups": [{"merged_content": "A and B", "source_ids": ["x1", "x2"]}],
                    "contradicted_pairs": [],
                    "kept_ids": [],
                }
            )
        )
        p.reconcile_memories("u1")
        meta = upserts[0].get("metadata") or {}
        assert meta.get("merged_via") == "reconcile"
        assert meta.get("merged_from_count") == 2


class TestReconcileNoOrphanDeleteOnRace:
    """When ALL supersede attempts lose the ETag race, the merged doc must
    NOT be deleted: deleting it would orphan any sources whose
    ``superseded_by`` was already pointed at this deterministic merged id
    by the concurrent-reconcile winner — those sources would become
    invisible to default reads (filter ``superseded_by IS NULL``) and to
    the reconcile pool, causing permanent data loss. The merged doc is
    idempotent (deterministic id) so leaving it in place is consistent."""

    def test_orphan_merged_doc_is_not_deleted_when_no_supersede_succeeds(self):
        p = _make_pipeline()
        p._upsert_memory = MagicMock(side_effect=lambda doc: doc)
        # Every supersede attempt loses the race → without the fix, the
        # merged doc would be hard-deleted. With the fix, the merged doc
        # stays as-is and the loss path is logged at INFO.
        p._mark_superseded = MagicMock(return_value=False)
        p._container.query_items.return_value = iter([_fact("o1", "A"), _fact("o2", "B")])
        p._run_prompty = MagicMock(
            return_value=json.dumps(
                {
                    "duplicate_groups": [{"merged_content": "A&B", "source_ids": ["o1", "o2"]}],
                    "contradicted_pairs": [],
                    "kept_ids": [],
                }
            )
        )
        result = p.reconcile_memories("u1")
        # delete_item must NOT be called — orphan-delete path was the
        # data-loss bug fixed in this round.
        assert not p._container.delete_item.called
        # No facts merged (no supersede succeeded).
        assert result["merged"] == 0


class TestReconcileContradictionWinnerNotInKeptIds:
    """RD#5+#13: contradiction winners are absent from kept_ids — must NOT trigger warning."""

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
        with caplog.at_level(logging.WARNING, logger="agent_memory_toolkit.pipeline"):
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


# ---------------------------------------------------------------------------
# Round-18 regression tests: out-of-range LLM numbers, etag flow on merged
# docs, and ``confidence=None`` rendering in facts_text.
# ---------------------------------------------------------------------------


class TestReconcileClampsConfidenceAndSalience:
    """LLM emitting values outside (0, 1] (e.g. 1.05 from a model that
    confused percent with [0,1]) must NOT propagate to MemoryRecord — the
    Pydantic validator would reject and the blanket except in reconcile
    would silently drop the entire merge group. Out-of-range values must
    fall back to ``max(source.*)``."""

    def test_out_of_range_confidence_falls_back_to_source_max(self):
        p = _make_pipeline()
        upserts: list[dict] = []
        p._upsert_memory = MagicMock(side_effect=lambda doc: upserts.append(doc) or doc)
        p._container.query_items.return_value = iter(
            [
                _fact("f1", "User likes coffee", confidence=0.7, salience=0.6),
                _fact("f2", "User enjoys coffee", confidence=0.85, salience=0.8),
            ]
        )
        p._run_prompty = MagicMock(
            return_value=json.dumps(
                {
                    "duplicate_groups": [
                        {
                            "merged_content": "User likes coffee",
                            "source_ids": ["f1", "f2"],
                            "confidence": 1.05,
                            "salience": 1.5,
                        }
                    ],
                    "contradicted_pairs": [],
                    "kept_ids": [],
                }
            )
        )
        result = p.reconcile_memories("u1")
        assert result["merged"] == 2
        assert len(upserts) == 1
        assert upserts[0]["confidence"] == pytest.approx(0.85)
        assert upserts[0]["salience"] == pytest.approx(0.8)


class TestReconcileEtagFlowsThroughMergedDocCache:
    """``_upsert_memory`` must return the response (which carries the
    fresh ``_etag``) so the in-memory ``merged_docs_by_id`` cache can
    feed it into a downstream supersede on the contradiction-redirect
    path. Without it, ``_mark_superseded`` falls through to
    ``upsert_item`` with no concurrency protection.

    This test exercises the full path: a duplicate group folds f1 + f2
    into a fresh merged doc M, then a contradiction names f2 as the
    loser. The pipeline must redirect the contradiction to M and call
    ``_mark_superseded(M, ...)`` with M carrying the etag returned from
    the upsert.
    """

    def test_supersede_on_merged_doc_receives_doc_with_etag(self):
        p = _make_pipeline()

        def upsert_response(doc):
            response = dict(doc)
            response["_etag"] = "etag-from-cosmos"
            return response

        p._upsert_memory = MagicMock(side_effect=upsert_response)
        p._mark_superseded = MagicMock(return_value=True)
        p._container.query_items.return_value = iter(
            [
                _fact("f1", "User likes tea"),
                _fact("f2", "User enjoys tea"),
                _fact("f3", "User hates all hot drinks"),
            ]
        )
        p._run_prompty = MagicMock(
            return_value=json.dumps(
                {
                    "duplicate_groups": [{"merged_content": "User likes tea", "source_ids": ["f1", "f2"]}],
                    "contradicted_pairs": [{"winner_id": "f3", "loser_id": "f2", "reason": "contradicts merged"}],
                    "kept_ids": ["f3"],
                }
            )
        )
        p.reconcile_memories("u1")

        # The third call to _mark_superseded targets the merged doc
        # (loser f2 was redirected through merged_docs_by_id). That
        # doc must carry the _etag returned by _upsert_memory — proving
        # the response actually flowed through the cache and not just
        # the locally-built dict.
        assert p._mark_superseded.call_count == 3
        contradiction_call = p._mark_superseded.call_args_list[-1]
        merged_doc_passed = contradiction_call.args[0]
        assert merged_doc_passed.get("_etag") == "etag-from-cosmos"


class TestReconcileSkipsSingleSourceDuplicateGroup:
    """A `duplicate_group` with only one valid `source_id` is a no-op
    masquerading as a merge — it would supersede a single fact with a
    near-identical clone (extra row, no signal) and could redirect a
    later contradiction's loser_id onto a merged doc that represents
    nothing real. Skip such groups."""

    def test_single_source_duplicate_group_does_not_create_merged_doc(self):
        p = _make_pipeline()
        p._upsert_memory = MagicMock(side_effect=lambda d: dict(d, _etag="e"))
        p._mark_superseded = MagicMock(return_value=True)
        p._container.query_items.return_value = iter([_fact("f1", "User likes tea"), _fact("f2", "User likes coffee")])
        p._run_prompty = MagicMock(
            return_value=json.dumps(
                {
                    "duplicate_groups": [
                        {"merged_content": "User likes tea", "source_ids": ["f1"]},
                    ],
                    "contradicted_pairs": [],
                    "kept_ids": ["f1", "f2"],
                }
            )
        )

        out = p.reconcile_memories("u1")

        # No merged record created; no source superseded as a duplicate.
        merged_upserts = [c for c in p._upsert_memory.call_args_list if c.args[0].get("type") == "fact"]
        assert merged_upserts == []
        assert p._mark_superseded.call_count == 0
        assert out["merged"] == 0


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


class TestExtractUpdateSupersedeReason:
    """Extract-time UPDATE actions stamp ``supersede_reason="update"``,
    distinct from reconcile-time ``"duplicate"`` (paraphrase merge) and
    ``"contradiction"`` (semantic conflict). The extract prompt defines
    UPDATE as "contradicts or refines an existing memory" — labelling
    these as ``"duplicate"`` makes audit trails ambiguous."""

    def _build(self) -> ProcessingPipeline:
        p = ProcessingPipeline.__new__(ProcessingPipeline)
        p._embeddings = MagicMock()
        p._embeddings.generate.return_value = [[0.1] * 8]
        p._upsert_memory = MagicMock(side_effect=lambda doc: doc)
        p._mark_superseded = MagicMock(return_value=True)
        p._container = MagicMock()
        p._chat = MagicMock()
        p._load_existing_memories = MagicMock(
            return_value=[
                {
                    "id": "fact_old",
                    "type": "fact",
                    "content": "User likes coffee",
                    "content_hash": "h_old",
                }
            ]
        )
        p._container.read_item = MagicMock(
            return_value={"id": "fact_old", "type": "fact", "content": "User likes coffee"}
        )
        p._container.query_items.return_value = iter(
            [
                {
                    "id": "turn-1",
                    "role": "user",
                    "content": "I love tea now",
                    "type": "turn",
                    "created_at": "2024-01-01T00:00:00+00:00",
                }
            ]
        )
        return p

    def test_fact_update_uses_reason_update_not_duplicate(self):
        p = self._build()
        p._run_prompty = MagicMock(
            return_value=json.dumps(
                {
                    "facts": [
                        {
                            "text": "User now prefers tea over coffee",
                            "confidence": 0.9,
                            "salience": 0.7,
                            "action": "UPDATE",
                            "supersedes_id": "fact_old",
                            "tags": ["sys:fact"],
                        }
                    ],
                    "procedural": [],
                    "episodic": [],
                }
            )
        )
        p.extract_memories("u1", "t1")
        assert p._mark_superseded.called
        call_kwargs = p._mark_superseded.call_args.kwargs
        assert call_kwargs.get("reason") == "update"

    def test_procedural_update_uses_reason_update_not_duplicate(self):
        p = self._build()
        p._load_existing_memories = MagicMock(
            return_value=[
                {
                    "id": "proc_old",
                    "type": "procedural",
                    "content": "Always greet the user formally",
                    "content_hash": "h_proc_old",
                }
            ]
        )
        # First query_items → turns; second → lookup of proc_old by id.
        turns = [
            {
                "id": "turn-1",
                "role": "user",
                "content": "be casual",
                "type": "turn",
                "created_at": "2024-01-01T00:00:00+00:00",
            }
        ]
        old_proc = [
            {
                "id": "proc_old",
                "type": "procedural",
                "content": "Always greet the user formally",
            }
        ]
        p._container.query_items = MagicMock(side_effect=[iter(turns), iter(old_proc)])
        p._run_prompty = MagicMock(
            return_value=json.dumps(
                {
                    "facts": [],
                    "procedural": [
                        {
                            "instruction": "Greet the user casually with their first name",
                            "confidence": 0.9,
                            "salience": 0.8,
                            "action": "UPDATE",
                            "supersedes_id": "proc_old",
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
        assert p._mark_superseded.called
        call_kwargs = p._mark_superseded.call_args.kwargs
        assert call_kwargs.get("reason") == "update"


class TestExtractUpdateSelfCollapseGuard:
    """When an LLM emits ``UPDATE`` whose new content hashes to the same
    deterministic id as the target (paraphrase-equivalent text), the
    upsert would overwrite the audit metadata that ``_mark_superseded``
    just stamped on the target. Treat as a no-op."""

    def _build(self) -> ProcessingPipeline:
        p = ProcessingPipeline.__new__(ProcessingPipeline)
        p._embeddings = MagicMock()
        p._embeddings.generate.return_value = [[0.1] * 8]
        p._upsert_memory = MagicMock(side_effect=lambda doc: doc)
        p._mark_superseded = MagicMock(return_value=True)
        p._container = MagicMock()
        p._chat = MagicMock()
        p._load_existing_memories = MagicMock(return_value=[])
        return p

    def test_fact_update_with_self_referential_id_is_skipped(self):
        from agent_memory_toolkit._utils import compute_content_hash
        from agent_memory_toolkit.pipeline import _ID_SEED_SEP

        p = self._build()
        text = "User likes tea"
        seed = _ID_SEED_SEP.join(("u1", "t1", compute_content_hash(text)))
        det_id = f"fact_{hashlib.sha256(seed.encode()).hexdigest()[:32]}"

        p._container.query_items.return_value = iter(
            [
                {
                    "id": "turn-1",
                    "role": "user",
                    "content": "tea",
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
                            "action": "UPDATE",
                            "supersedes_id": det_id,
                            "tags": ["sys:fact"],
                        }
                    ],
                    "procedural": [],
                    "episodic": [],
                }
            )
        )

        out = p.extract_memories("u1", "t1")

        assert p._mark_superseded.call_count == 0
        fact_upserts = [c for c in p._upsert_memory.call_args_list if c.args[0].get("type") == "fact"]
        assert fact_upserts == []
        assert out["facts_count"] == 0

    def test_procedural_update_with_self_referential_id_is_skipped(self):
        from agent_memory_toolkit._utils import compute_content_hash
        from agent_memory_toolkit.pipeline import _ID_SEED_SEP

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

        out = p.extract_memories("u1", "t1")

        assert p._mark_superseded.call_count == 0
        proc_upserts = [c for c in p._upsert_memory.call_args_list if c.args[0].get("type") == "procedural"]
        assert proc_upserts == []
        assert out["procedural_count"] == 0
