"""Integration tests for CosmosMemoryClient against live Azure services.

These tests exercise the end-to-end in-process flow: writing turns to
Cosmos DB, running the ProcessingPipeline (summarisation, fact / procedural /
episodic extraction, deduplication) inline, and reading back results via
Cosmos DB queries and vector / hybrid search.

The Azure Function host is **not** required — the same ProcessingPipeline
that the change-feed trigger invokes is also exposed directly on
``CosmosMemoryClient`` (``extract_memories``, ``generate_thread_summary``,
``generate_user_summary``, ``reconcile``).

Enable by setting::

    AGENT_MEMORY_RUN_INTEGRATION=true

Auth: ``COSMOS_DB_KEY`` is used when present (relief while Cosmos control-plane
RBAC is still in private preview); otherwise ``DefaultAzureCredential``.
"""

from __future__ import annotations

import time
import uuid

import pytest

from agent_memory_toolkit import CosmosMemoryClient
from tests.conftest import INTEGRATION_ENABLED

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        not INTEGRATION_ENABLED,
        reason="Set AGENT_MEMORY_RUN_INTEGRATION=true",
    ),
]


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def agent_memory(
    cosmos_endpoint,
    cosmos_key,
    cosmos_database,
    cosmos_container,
    ai_foundry_endpoint,
    ai_foundry_api_key,
    embedding_deployment_name,
    embedding_dimensions,
    chat_deployment_name,
):
    """A live CosmosMemoryClient shared by every test in this module."""
    if not cosmos_endpoint or not ai_foundry_endpoint:
        pytest.skip("COSMOS_DB_ENDPOINT / AI_FOUNDRY_ENDPOINT not set")

    return CosmosMemoryClient(
        cosmos_endpoint=cosmos_endpoint,
        cosmos_key=cosmos_key or None,
        cosmos_database=cosmos_database,
        cosmos_container=cosmos_container,
        ai_foundry_endpoint=ai_foundry_endpoint,
        ai_foundry_api_key=ai_foundry_api_key or None,
        embedding_deployment_name=embedding_deployment_name,
        embedding_dimensions=embedding_dimensions,
        chat_deployment_name=chat_deployment_name,
    )


def _add_turns(
    mem: CosmosMemoryClient,
    user_id: str,
    thread_id: str,
    turns: list[tuple[str, str]],
) -> None:
    for role, content in turns:
        mem.add_cosmos(
            user_id=user_id,
            role=role,
            content=content,
            memory_type="turn",
            thread_id=thread_id,
        )


def _cleanup(mem: CosmosMemoryClient, user_id: str) -> None:
    """Best-effort delete of every memory belonging to *user_id*."""
    try:
        for m in mem.get_memories(user_id=user_id, include_superseded=True):
            try:
                mem.delete_cosmos(
                    memory_id=m["id"],
                    thread_id=m.get("thread_id", ""),
                    user_id=user_id,
                )
            except Exception:
                pass
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestThreadSummary:
    def test_generate_thread_summary(self, agent_memory, unique_user_id, unique_thread_id):
        try:
            _add_turns(
                agent_memory,
                unique_user_id,
                unique_thread_id,
                [
                    ("user", "What are some good restaurants in Paris?"),
                    ("agent", "Le Comptoir du Panthéon is a classic bistro in the 5th arrondissement."),
                    ("user", "What kind of cuisine do they serve?"),
                    ("agent", "Traditional French bistro fare — confit de canard, steak frites, etc."),
                ],
            )
            time.sleep(1)

            doc = agent_memory.generate_thread_summary(
                user_id=unique_user_id,
                thread_id=unique_thread_id,
            )
            assert doc.get("id"), f"Expected summary doc with id, got {doc}"
            assert doc.get("type") == "summary"
            assert doc.get("content"), "Summary content must not be empty"

            summaries = agent_memory.get_memories(
                user_id=unique_user_id,
                thread_id=unique_thread_id,
                memory_type="summary",
            )
            assert len(summaries) >= 1
        finally:
            _cleanup(agent_memory, unique_user_id)


class TestExtractMemories:
    def test_extract_facts_episodic_procedural(self, agent_memory, unique_user_id, unique_thread_id):
        try:
            _add_turns(
                agent_memory,
                unique_user_id,
                unique_thread_id,
                [
                    ("user", "I live in Seattle and I work at Microsoft as a software engineer."),
                    ("agent", "That's great! Seattle is a wonderful city for tech professionals."),
                    ("user", "I prefer Python over JavaScript for backend work."),
                    ("user", "I had a great trip to Japan last spring during cherry blossom season."),
                    ("user", "When asked about deployment, always check the resource group first."),
                ],
            )
            time.sleep(1)

            stats = agent_memory.extract_memories(
                user_id=unique_user_id,
                thread_id=unique_thread_id,
            )
            assert isinstance(stats, dict)
            total = stats.get("facts_count", 0) + stats.get("procedural_count", 0) + stats.get("episodic_count", 0)
            assert total >= 1, f"Expected at least one memory extracted, got {stats}"

            facts = agent_memory.get_memories(user_id=unique_user_id, memory_type="fact")
            assert len(facts) >= 1, "Expected at least one fact (Seattle / Microsoft / Python)"

            for f in facts:
                assert isinstance(f.get("tags", []), list)
                salience = f.get("salience")
                assert salience is None or 0.0 <= float(salience) <= 1.0
        finally:
            _cleanup(agent_memory, unique_user_id)


class TestUserSummary:
    def test_multi_thread_user_summary(self, agent_memory, unique_user_id):
        t1 = str(uuid.uuid4())
        t2 = str(uuid.uuid4())
        try:
            _add_turns(
                agent_memory,
                unique_user_id,
                t1,
                [
                    ("user", "I love Italian cooking, especially pasta."),
                    ("agent", "Italian cuisine is wonderful! Do you make fresh pasta?"),
                    ("user", "Yes, I make homemade fettuccine every weekend."),
                ],
            )
            _add_turns(
                agent_memory,
                unique_user_id,
                t2,
                [
                    ("user", "I go running every morning before work."),
                    ("agent", "Running is a great habit. How far do you usually run?"),
                    ("user", "About 5 kilometres each day."),
                ],
            )
            time.sleep(1)

            doc = agent_memory.generate_user_summary(
                user_id=unique_user_id,
                thread_ids=[t1, t2],
            )
            assert doc.get("id"), f"Expected user_summary doc with id, got {doc}"
            assert doc.get("type") == "user_summary"

            summaries = agent_memory.get_user_summary(unique_user_id)
            assert len(summaries) >= 1
            combined = " ".join(s.get("content", "") for s in summaries).lower()
            assert any(t in combined for t in ("pasta", "italian", "cooking", "fettuccine"))
            assert any(t in combined for t in ("running", "run", "morning"))
        finally:
            _cleanup(agent_memory, unique_user_id)


class TestSearchAfterExtraction:
    def test_vector_and_hybrid_search(self, agent_memory, unique_user_id, unique_thread_id):
        try:
            _add_turns(
                agent_memory,
                unique_user_id,
                unique_thread_id,
                [
                    ("user", "I have a golden retriever named Buddy."),
                    ("agent", "Golden retrievers are great family dogs! How old is Buddy?"),
                    ("user", "Buddy is 3 years old and loves playing fetch at the park."),
                ],
            )
            time.sleep(1)

            agent_memory.extract_memories(user_id=unique_user_id, thread_id=unique_thread_id)
            agent_memory.generate_thread_summary(user_id=unique_user_id, thread_id=unique_thread_id)
            time.sleep(2)

            vec = agent_memory.search_cosmos(
                search_terms="golden retriever dog",
                user_id=unique_user_id,
                top_k=5,
            )
            assert len(vec) >= 1, "Vector search should return at least 1 result"

            hyb = agent_memory.search_cosmos(
                search_terms="Buddy the dog park",
                user_id=unique_user_id,
                hybrid_search=True,
                top_k=5,
            )
            assert len(hyb) >= 1, "Hybrid search should return at least 1 result"
        finally:
            _cleanup(agent_memory, unique_user_id)


class TestTaggingAndSalience:
    def test_add_remove_tags_and_salience_filter(self, agent_memory, unique_user_id, unique_thread_id):
        try:
            agent_memory.add_cosmos(
                user_id=unique_user_id,
                role="user",
                content="The user prefers dark mode UI and uses VS Code.",
                memory_type="fact",
                thread_id=unique_thread_id,
                tags=["preference", "ide"],
                salience=0.8,
            )
            tagged = agent_memory.get_memories(
                user_id=unique_user_id,
                tags=["preference"],
            )
            assert len(tagged) == 1
            mid = tagged[0]["id"]

            agent_memory.add_tags(
                memory_id=mid,
                user_id=unique_user_id,
                thread_id=unique_thread_id,
                tags=["ui"],
            )
            agent_memory.remove_tags(
                memory_id=mid,
                user_id=unique_user_id,
                thread_id=unique_thread_id,
                tags=["ide"],
            )

            refreshed = agent_memory.get_memories(
                user_id=unique_user_id,
                tags=["ui"],
            )
            assert any(m["id"] == mid for m in refreshed)
            stored = next(m for m in refreshed if m["id"] == mid)
            assert "ui" in (stored.get("tags") or [])
            assert "ide" not in (stored.get("tags") or [])

            results = agent_memory.search_cosmos(
                search_terms="dark mode",
                user_id=unique_user_id,
                min_salience=0.5,
                top_k=5,
            )
            assert any(r["id"] == mid for r in results)
        finally:
            _cleanup(agent_memory, unique_user_id)


class TestReconciliation:
    def test_dedup_near_duplicate_facts(self, agent_memory, unique_user_id, unique_thread_id):
        try:
            for content in [
                "The user lives in Seattle.",
                "User lives in Seattle, WA.",
                "The user resides in Seattle.",
                "The user works at Microsoft as an engineer.",
            ]:
                agent_memory.add_cosmos(
                    user_id=unique_user_id,
                    role="user",
                    content=content,
                    memory_type="fact",
                    thread_id=unique_thread_id,
                    salience=0.7,
                )

            before = agent_memory.get_memories(
                user_id=unique_user_id,
                memory_type="fact",
            )
            assert len(before) >= 4

            stats = agent_memory.reconcile(
                user_id=unique_user_id,
            )
            assert isinstance(stats, dict)
            assert "kept" in stats and "merged" in stats and "contradicted" in stats
            assert stats["merged"] + stats["contradicted"] >= 1, (
                f"Expected at least one near-duplicate to be merged/contradicted, got {stats}"
            )

            active = [
                m
                for m in agent_memory.get_memories(user_id=unique_user_id, memory_type="fact")
                if not m.get("superseded_by")
            ]
            assert len(active) < len(before)
        finally:
            _cleanup(agent_memory, unique_user_id)

    def test_reconcile_resolves_contradiction(self, agent_memory, unique_user_id, unique_thread_id):
        try:
            contradictory_facts = [
                "User is strictly vegetarian and never eats meat.",
                "User loves a good ribeye steak.",
                "User often orders the bone-in pork chop at steakhouses.",
            ]
            for content in contradictory_facts:
                agent_memory.add_cosmos(
                    user_id=unique_user_id,
                    role="user",
                    content=content,
                    memory_type="fact",
                    thread_id=unique_thread_id,
                    salience=0.7,
                )

            before = agent_memory.get_memories(
                user_id=unique_user_id,
                memory_type="fact",
            )
            assert len(before) == len(contradictory_facts)

            stats = agent_memory.reconcile(user_id=unique_user_id)
            assert isinstance(stats, dict)
            assert stats.get("contradicted", 0) >= 1, f"Expected at least one contradiction to be resolved, got {stats}"

            all_facts = agent_memory.get_memories(
                user_id=unique_user_id,
                memory_type="fact",
                include_superseded=True,
            )
            contradicted = [m for m in all_facts if m.get("supersede_reason") == "contradiction"]
            assert len(contradicted) >= 1, "Expected at least one record marked supersede_reason=contradiction"
            sample = contradicted[0]
            assert isinstance(sample.get("superseded_at"), str) and len(sample["superseded_at"]) > 0
            assert isinstance(sample.get("superseded_by"), str) and len(sample["superseded_by"]) > 0

            active = [m for m in all_facts if not m.get("superseded_by")]
            assert len(active) < len(before), (
                f"Active fact count should shrink after contradiction resolution; "
                f"before={len(before)} active={len(active)}"
            )
        finally:
            _cleanup(agent_memory, unique_user_id)

    def test_extract_content_hash_short_circuit(self, agent_memory, unique_user_id, unique_thread_id):
        try:
            agent_memory.add_cosmos(
                user_id=unique_user_id,
                role="user",
                content="My favorite color is teal.",
                thread_id=unique_thread_id,
            )
            agent_memory.add_cosmos(
                user_id=unique_user_id,
                role="agent",
                content="Got it, teal is a great color.",
                thread_id=unique_thread_id,
            )
            time.sleep(1)

            agent_memory.process_now(user_id=unique_user_id, thread_id=unique_thread_id)
            facts_after_first = [
                m
                for m in agent_memory.get_memories(user_id=unique_user_id, memory_type="fact")
                if not m.get("superseded_by")
            ]
            assert len(facts_after_first) >= 1, "First extraction should produce at least one fact"
            assert all(len(m.get("content_hash") or "") == 32 for m in facts_after_first), (
                "All extracted facts should carry a content_hash of length 32"
            )

            agent_memory.process_now(user_id=unique_user_id, thread_id=unique_thread_id)
            facts_after_second = [
                m
                for m in agent_memory.get_memories(user_id=unique_user_id, memory_type="fact")
                if not m.get("superseded_by")
            ]
            assert len(facts_after_second) <= len(facts_after_first), (
                f"Re-extraction should not grow active fact count; "
                f"first={len(facts_after_first)} second={len(facts_after_second)}"
            )
        finally:
            _cleanup(agent_memory, unique_user_id)

    def test_reconcile_writes_supersede_metadata(self, agent_memory, unique_user_id, unique_thread_id):
        try:
            paraphrases = [
                "The user lives in Seattle and works at Microsoft as a data engineer.",
                "User resides in Seattle, WA, employed by Microsoft as a data engineer.",
                "The user is based in Seattle and is a data engineer at Microsoft.",
                "User lives in Seattle; works at Microsoft on the data engineering team.",
                "The user works as a data engineer at Microsoft in Seattle.",
            ]
            for content in paraphrases:
                agent_memory.add_cosmos(
                    user_id=unique_user_id,
                    role="user",
                    content=content,
                    memory_type="fact",
                    thread_id=unique_thread_id,
                    salience=0.7,
                )

            stats = agent_memory.reconcile(user_id=unique_user_id)
            assert isinstance(stats, dict)
            assert stats.get("merged", 0) + stats.get("contradicted", 0) >= 1, (
                f"Expected at least one merge/contradiction across paraphrases, got {stats}"
            )

            all_facts = agent_memory.get_memories(
                user_id=unique_user_id,
                memory_type="fact",
                include_superseded=True,
            )
            losers = [m for m in all_facts if m.get("supersede_reason") is not None]
            assert len(losers) >= 1, "Expected at least one superseded record"

            sample_loser = losers[0]
            assert sample_loser["supersede_reason"] in {"duplicate", "contradiction"}
            assert isinstance(sample_loser["superseded_at"], str) and len(sample_loser["superseded_at"]) > 0
            assert isinstance(sample_loser["superseded_by"], str) and len(sample_loser["superseded_by"]) > 0

            survivor_id = sample_loser["superseded_by"]
            live = [m for m in all_facts if not m.get("superseded_by")]
            assert any(m["id"] == survivor_id for m in live), "supersede_by must point at a live record"
        finally:
            _cleanup(agent_memory, unique_user_id)
