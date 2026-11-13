"""``reconcile_memories`` runs the dedup pass first, then the contradiction pass on a re-read pool.

The dedup pass (``consolidate_memories``) retires duplicates; the contradiction pass then asks the
dedup prompt about the facts that remain. These tests run both for real against a fake container
and check their order, what each sees, the merged result and the deadline.
"""

from __future__ import annotations

import json
import logging
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from azure.cosmos.exceptions import CosmosResourceNotFoundError

from azure.cosmos.agent_memory.services import pipeline as pipeline_module
from azure.cosmos.agent_memory.services._dedup import dedup_result
from azure.cosmos.agent_memory.services.pipeline import PipelineService

NEAR = [1.0, 0.0, 0.0]


@pytest.fixture(autouse=True)
def _dedup_only(monkeypatch):
    monkeypatch.setattr("azure.cosmos.agent_memory.thresholds.COMBINE_ENABLED", False)


def _fact(fid: str, content: str, *, vector=None, ts: int = 1, created_at: str = "2024-01-01T00:00:00+00:00") -> dict:
    doc = {
        "id": fid,
        "user_id": "u1",
        "thread_id": "t1",
        "type": "fact",
        "content": content,
        "confidence": 0.8,
        "salience": 0.5,
        "tags": ["sys:fact"],
        "metadata": {"source": "user"},
        "created_at": created_at,
        "_ts": ts,
        "_etag": f"etag-{fid}",
    }
    if vector is not None:
        doc["embedding"] = vector
    return doc


class _Store:
    """A memories container holding facts; retiring one hides it from later queries."""

    def __init__(self, facts: list[dict]):
        self.facts = facts
        self.retired: dict[str, tuple[str, str]] = {}
        self.queries: list[str] = []

    def active(self) -> list[dict]:
        return [doc for doc in self.facts if doc["id"] not in self.retired]

    def query_items(self, query, parameters, **kwargs):
        self.queries.append(query)
        if "VectorDistance" in query:
            seed_id = {p["name"]: p["value"] for p in parameters}["@seed_id"]
            return iter([doc for doc in self.active() if doc["id"] != seed_id and "embedding" in doc])
        if "ORDER BY c._ts ASC" in query:
            return iter(self.active())
        return iter(sorted(self.active(), key=lambda doc: doc["created_at"], reverse=True))

    def mark_superseded(self, doc, superseder_id, *, reason):
        self.retired[doc["id"]] = (superseder_id, reason)
        return True


def _make_pipeline(store: _Store, replies: list[dict]) -> PipelineService:
    p = PipelineService.__new__(PipelineService)
    p._container = MagicMock()
    p._container.query_items.side_effect = store.query_items
    p._container.read_item.side_effect = CosmosResourceNotFoundError(message="no dedup cursor yet")
    p._memories_container = p._container
    p._turns_container = p._container
    p._summaries_container = p._container
    p._mark_superseded = MagicMock(
        side_effect=lambda doc, superseder, *, reason: store.mark_superseded(doc, superseder, reason=reason)
    )
    p._run_prompty = MagicMock(side_effect=[json.dumps(reply) for reply in replies])
    return p


def _dedup_reply(relation: str = "same_claim") -> dict:
    return {
        "candidates": [
            {
                "label": "C1",
                "relation": relation,
                "seed_details_missing_from_candidate": [],
                "candidate_details_missing_from_seed": [],
            }
        ]
    }


def _store() -> _Store:
    return _Store(
        [
            _fact("f1", "User likes coffee", vector=NEAR, ts=1, created_at="2024-01-01T00:00:00+00:00"),
            _fact("f2", "The user likes coffee", vector=NEAR, ts=2, created_at="2024-02-01T00:00:00+00:00"),
            _fact("f3", "User is vegetarian", ts=3, created_at="2024-03-01T00:00:00+00:00"),
            _fact("f4", "User eats steak", ts=4, created_at="2024-04-01T00:00:00+00:00"),
        ]
    )


CONTRADICTION = {"contradicted_pairs": [{"winner_id": "f4", "loser_id": "f3", "reason": "newer"}], "kept_ids": []}


class TestDedupThenContradictions:
    def test_duplicates_are_folded_first_and_the_contradiction_pass_reads_what_is_left(self):
        store = _store()
        p = _make_pipeline(store, [_dedup_reply(), CONTRADICTION])

        result = p.reconcile_memories("u1")

        assert [call.args[0] for call in p._run_prompty.call_args_list] == ["dedup_verify.prompty", "dedup.prompty"]
        pool = p._run_prompty.call_args.kwargs["inputs"]["facts_text"]
        assert "ID: f1" not in pool and all(f"ID: {fid} |" in pool for fid in ("f2", "f3", "f4"))
        assert store.retired == {"f1": ("f2", "duplicate"), "f3": ("f4", "contradict")}
        assert (result["merged"], result["contradicted"], result["kept"]) == (1, 1, 2)
        assert (result["checked"], result["compared"], result["vetoed"]) == (4, 1, 0)

    def test_the_dedup_pass_writes_its_cursor_before_the_contradiction_pool_is_read(self):
        store = _store()
        p = _make_pipeline(store, [_dedup_reply(), CONTRADICTION])

        p.reconcile_memories("u1")

        body = p._container.create_item.call_args.kwargs["body"]
        assert (body["id"], body["thread_id"], body["last_ts"]) == ("dedup_cursor_u1_fact", "__dedup__", 4)
        assert "ORDER BY c.created_at DESC" in store.queries[-1]

    def test_a_pass_with_nothing_to_fold_still_resolves_contradictions(self):
        store = _store()
        p = _make_pipeline(store, [_dedup_reply("related"), CONTRADICTION])

        result = p.reconcile_memories("u1")

        assert store.retired == {"f3": ("f4", "contradict")}
        assert (result["merged"], result["contradicted"], result["kept"]) == (0, 1, 3)

    def test_the_result_keeps_its_keys_and_adds_the_dedup_counts(self):
        store = _store()
        p = _make_pipeline(store, [_dedup_reply(), CONTRADICTION])

        result = p.reconcile_memories("u1")

        assert set(result) == set(dedup_result(checked=0, retired_seeds=0, merged=0, compared=0, vetoed=0))
        assert {"kept", "merged", "contradicted"} <= set(result)

    @pytest.mark.parametrize("memory_type", ["episodic", "procedural"])
    def test_other_memory_types_are_no_ops_with_the_full_result_shape(self, memory_type):
        p = _make_pipeline(_store(), [])
        p.consolidate_memories = MagicMock()

        result = p.reconcile_memories("u1", memory_type=memory_type)

        assert result == dedup_result(checked=0, retired_seeds=0, merged=0, compared=0, vetoed=0)
        p.consolidate_memories.assert_not_called()
        p._container.query_items.assert_not_called()

    def test_the_deadline_is_handed_to_the_dedup_pass(self):
        p = _make_pipeline(_store(), [])
        p.consolidate_memories = MagicMock(
            return_value=dedup_result(checked=0, retired_seeds=0, merged=0, compared=0, vetoed=0)
        )
        p._reconcile_contradictions = MagicMock(return_value={"kept": 0, "merged": 0, "contradicted": 0})
        p._active_memories_for_reconcile = MagicMock(return_value=[])

        p.reconcile_memories("u1", n=9, deadline=float("inf"))

        assert p.consolidate_memories.call_args.args == ("u1", 9)
        assert p.consolidate_memories.call_args.kwargs == {"memory_type": "fact", "deadline": float("inf")}


class TestReconcileDeadline:
    def test_a_deadline_passed_during_the_dedup_pass_skips_the_contradiction_pass(self, monkeypatch, caplog):
        store = _store()
        clock = {"now": 0.0}
        monkeypatch.setattr(pipeline_module, "time", SimpleNamespace(monotonic=lambda: clock["now"]))
        p = _make_pipeline(store, [])

        def slow_check(filename, inputs):
            del filename, inputs
            clock["now"] = 100.0
            return json.dumps(_dedup_reply())

        p._run_prompty = MagicMock(side_effect=slow_check)

        with caplog.at_level(logging.INFO, logger="azure.cosmos.agent_memory.pipeline"):
            result = p.reconcile_memories("u1", deadline=50.0)

        assert [call.args[0] for call in p._run_prompty.call_args_list] == ["dedup_verify.prompty"]
        assert store.retired == {"f1": ("f2", "duplicate")}
        assert (result["merged"], result["contradicted"]) == (1, 0)
        assert result["kept"] == result["checked"] - 1
        outcome = [r for r in caplog.records if r.getMessage() == "reconcile.outcome"]
        assert len(outcome) == 1 and outcome[0].candidates_considered == 0

    def test_without_a_deadline_both_passes_always_run(self):
        store = _store()
        p = _make_pipeline(store, [_dedup_reply(), CONTRADICTION])

        result = p.reconcile_memories("u1", deadline=None)

        assert (result["merged"], result["contradicted"]) == (1, 1)


class TestReconcileOutcomeTelemetryIncludesDedupCounts:
    def test_the_outcome_record_carries_the_dedup_counts(self, caplog):
        store = _store()
        p = _make_pipeline(store, [_dedup_reply(), CONTRADICTION])

        with caplog.at_level(logging.INFO, logger="azure.cosmos.agent_memory.pipeline"):
            result = p.reconcile_memories("u1")

        [record] = [r for r in caplog.records if r.getMessage() == "reconcile.outcome"]
        assert record.facts_checked == result["checked"] == 4
        assert record.pairs_compared == result["compared"] == 1
        for key in ("kept", "merged", "contradicted", "vetoed", "combined", "synthesized"):
            assert getattr(record, key) == result[key]
        assert (record.combine_groups, record.combine_failed) == (0, 0)
        assert record.candidates_considered == 3
