"""Integration tests for the full CosmosMemoryClient pipeline against live Azure services.

These tests exercise the end-to-end flow: writing turns to Cosmos DB, triggering
Azure Durable Functions for summarisation / fact-extraction, and reading back the
results via Cosmos DB queries and vector search.

Enable by setting the environment variable::

    AGENT_MEMORY_RUN_INTEGRATION=true
"""

import time
import uuid

import pytest

from agent_memory_toolkit import CosmosMemoryClient
from tests.conftest import INTEGRATION_ENABLED

# ---------------------------------------------------------------------------
# Module-level markers – every test in this file is an integration test and
# will be skipped automatically when the flag is not set.
# ---------------------------------------------------------------------------

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        not INTEGRATION_ENABLED,
        reason="Set AGENT_MEMORY_RUN_INTEGRATION=true",
    ),
]

# Durable Functions can take a while; generous defaults keep CI green.
_POLL_INTERVAL = 3.0
_TIMEOUT = 180.0


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def agent_memory(
    cosmos_endpoint,
    cosmos_database,
    cosmos_container,
    ai_foundry_endpoint,
    ai_foundry_api_key,
    adf_endpoint,
    adf_key,
    embedding_model,
    embedding_dimensions,
):
    """Create and connect a CosmosMemoryClient instance shared across this module."""
    mem = CosmosMemoryClient(
        cosmos_endpoint=cosmos_endpoint,
        cosmos_database=cosmos_database,
        cosmos_container=cosmos_container,
        ai_foundry_endpoint=ai_foundry_endpoint,
        ai_foundry_api_key=ai_foundry_api_key,
        adf_endpoint=adf_endpoint,
        adf_key=adf_key,
        embedding_model=embedding_model,
        embedding_dimensions=embedding_dimensions,
    )
    return mem


def _add_turns(agent_memory: CosmosMemoryClient, user_id: str, thread_id: str, turns: list[tuple[str, str]]):
    """Helper – add a sequence of (role, content) turns to Cosmos."""
    for role, content in turns:
        agent_memory.add_cosmos(
            user_id=user_id,
            role=role,
            content=content,
            memory_type="turn",
            thread_id=thread_id,
        )


def _cleanup_memories(agent_memory: CosmosMemoryClient, user_id: str, thread_id: str | None = None):
    """Best-effort cleanup of all memories for a user (optionally scoped to thread)."""
    try:
        kwargs: dict = {"user_id": user_id}
        if thread_id:
            kwargs["thread_id"] = thread_id
        memories = agent_memory.get_memories(**kwargs)
        for mem in memories:
            try:
                agent_memory.delete_cosmos(
                    memory_id=mem["id"],
                    thread_id=mem.get("thread_id", ""),
                    user_id=user_id,
                )
            except Exception:
                pass  # best-effort
    except Exception:
        pass


def _cleanup_user(agent_memory: CosmosMemoryClient, user_id: str):
    """Remove all data for *user_id* regardless of thread."""
    _cleanup_memories(agent_memory, user_id)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestAddTurnsAndGenerateThreadSummary:
    """Scenario 1: add turns → generate thread summary → verify."""

    def test_add_turns_and_generate_thread_summary(
        self, agent_memory, unique_user_id, unique_thread_id
    ):
        user_id = unique_user_id
        thread_id = unique_thread_id
        try:
            # -- Arrange: insert 3 conversation turns --
            _add_turns(
                agent_memory,
                user_id,
                thread_id,
                [
                    ("user", "What are some good restaurants in Paris?"),
                    ("agent", "Le Comptoir du Panthéon is a classic bistro in the 5th arrondissement."),
                    ("user", "What kind of cuisine do they serve?"),
                ],
            )
            time.sleep(2)

            # -- Act: trigger the durable function --
            result = agent_memory.generate_thread_summary(
                user_id=user_id,
                thread_id=thread_id,
                poll_interval=_POLL_INTERVAL,
                timeout=_TIMEOUT,
            )

            # -- Assert: orchestration completed --
            assert result.get("runtimeStatus") == "Completed", (
                f"Expected runtimeStatus='Completed', got {result.get('runtimeStatus')}"
            )

            time.sleep(2)

            # -- Assert: summary persisted in Cosmos --
            summaries = agent_memory.get_memories(
                user_id=user_id,
                thread_id=thread_id,
                memory_type="summary",
            )
            assert len(summaries) >= 1, "Expected at least 1 summary memory"
            assert summaries[0].get("content"), "Summary content must not be empty"
        finally:
            _cleanup_memories(agent_memory, user_id, thread_id)


class TestAddTurnsAndExtractFacts:
    """Scenario 2: add turns with factual info → extract facts → verify."""

    def test_add_turns_and_extract_facts(
        self, agent_memory, unique_user_id, unique_thread_id
    ):
        user_id = unique_user_id
        thread_id = unique_thread_id
        try:
            _add_turns(
                agent_memory,
                user_id,
                thread_id,
                [
                    ("user", "I live in Seattle and I work at Microsoft as a software engineer."),
                    ("agent", "That's great! Seattle is a wonderful city for tech professionals."),
                    ("user", "I prefer Python over JavaScript for backend work."),
                ],
            )
            time.sleep(2)

            result = agent_memory.extract_facts(
                user_id=user_id,
                thread_id=thread_id,
                poll_interval=_POLL_INTERVAL,
                timeout=_TIMEOUT,
            )

            assert result.get("runtimeStatus") == "Completed", (
                f"Expected runtimeStatus='Completed', got {result.get('runtimeStatus')}"
            )

            time.sleep(2)

            facts = agent_memory.get_memories(
                user_id=user_id,
                memory_type="fact",
            )
            assert len(facts) >= 1, (
                "Expected at least 1 extracted fact about Seattle / Microsoft"
            )
        finally:
            _cleanup_user(agent_memory, user_id)


class TestMultipleThreadsGenerateUserSummary:
    """Scenario 3: two threads → generate user summary → verify."""

    def test_multiple_threads_generate_user_summary(
        self, agent_memory, unique_user_id
    ):
        user_id = unique_user_id
        t1 = str(uuid.uuid4())
        t2 = str(uuid.uuid4())
        try:
            # Thread 1: cooking topic
            _add_turns(
                agent_memory, user_id, t1,
                [
                    ("user", "I love Italian cooking, especially pasta."),
                    ("agent", "Italian cuisine is wonderful! Do you make fresh pasta?"),
                    ("user", "Yes, I make homemade fettuccine every weekend."),
                ],
            )
            # Thread 2: fitness topic
            _add_turns(
                agent_memory, user_id, t2,
                [
                    ("user", "I go running every morning before work."),
                    ("agent", "Running is a great habit. How far do you usually run?"),
                    ("user", "About 5 kilometres each day."),
                ],
            )
            time.sleep(2)

            result = agent_memory.generate_user_summary(
                user_id=user_id,
                thread_ids=[t1, t2],
                poll_interval=_POLL_INTERVAL,
                timeout=_TIMEOUT,
            )

            assert result.get("runtimeStatus") == "Completed", (
                f"Expected runtimeStatus='Completed', got {result.get('runtimeStatus')}"
            )

            time.sleep(2)

            user_summaries = agent_memory.get_user_summary(user_id)
            assert len(user_summaries) >= 1, "Expected at least 1 user_summary"
        finally:
            _cleanup_memories(agent_memory, user_id, t1)
            _cleanup_memories(agent_memory, user_id, t2)
            _cleanup_user(agent_memory, user_id)


class TestIncrementalSummaryUpdate:
    """Scenario 4: add turns → summarise → add more turns → re-summarise → verify update."""

    def test_incremental_summary_update(
        self, agent_memory, unique_user_id, unique_thread_id
    ):
        user_id = unique_user_id
        thread_id = unique_thread_id
        try:
            # -- First batch of turns --
            _add_turns(
                agent_memory, user_id, thread_id,
                [
                    ("user", "Tell me about the Eiffel Tower."),
                    ("agent", "The Eiffel Tower is a wrought-iron lattice tower in Paris, France."),
                ],
            )
            time.sleep(2)

            first_result = agent_memory.generate_thread_summary(
                user_id=user_id,
                thread_id=thread_id,
                poll_interval=_POLL_INTERVAL,
                timeout=_TIMEOUT,
            )
            assert first_result.get("runtimeStatus") == "Completed"
            time.sleep(2)

            first_summaries = agent_memory.get_memories(
                user_id=user_id, thread_id=thread_id, memory_type="summary",
            )
            assert len(first_summaries) >= 1, "First summary should exist"
            first_content = first_summaries[0].get("content", "")

            # -- Second batch of turns --
            _add_turns(
                agent_memory, user_id, thread_id,
                [
                    ("user", "How tall is it exactly?"),
                    ("agent", "The Eiffel Tower is approximately 330 metres tall, including antennas."),
                ],
            )
            time.sleep(2)

            second_result = agent_memory.generate_thread_summary(
                user_id=user_id,
                thread_id=thread_id,
                poll_interval=_POLL_INTERVAL,
                timeout=_TIMEOUT,
            )
            assert second_result.get("runtimeStatus") == "Completed"
            time.sleep(2)

            updated_summaries = agent_memory.get_memories(
                user_id=user_id, thread_id=thread_id, memory_type="summary",
            )
            assert len(updated_summaries) >= 1, "Updated summary should exist"
            updated_content = updated_summaries[0].get("content", "")

            # The summary should have changed or include references to the new info.
            # Use updated_at comparison or content change as evidence.
            first_updated = first_summaries[0].get("updated_at", "")
            second_updated = updated_summaries[0].get("updated_at", "")
            content_changed = updated_content != first_content
            timestamp_changed = second_updated != first_updated
            assert content_changed or timestamp_changed, (
                "Expected the summary to be updated after adding new turns"
            )
        finally:
            _cleanup_memories(agent_memory, user_id, thread_id)


class TestIncrementalUserSummary:
    """Scenario 5: thread A → user summary → thread B → user summary → verify both topics."""

    def test_incremental_user_summary(self, agent_memory, unique_user_id):
        user_id = unique_user_id
        t1 = str(uuid.uuid4())
        t2 = str(uuid.uuid4())
        try:
            # Thread 1: gardening
            _add_turns(
                agent_memory, user_id, t1,
                [
                    ("user", "I grow tomatoes and basil in my backyard garden."),
                    ("agent", "Home gardening is rewarding! Do you compost as well?"),
                ],
            )
            time.sleep(2)

            r1 = agent_memory.generate_user_summary(
                user_id=user_id,
                thread_ids=[t1],
                poll_interval=_POLL_INTERVAL,
                timeout=_TIMEOUT,
            )
            assert r1.get("runtimeStatus") == "Completed"
            time.sleep(2)

            # Thread 2: photography
            _add_turns(
                agent_memory, user_id, t2,
                [
                    ("user", "I love landscape photography, especially during golden hour."),
                    ("agent", "Golden hour light is stunning! What camera do you use?"),
                ],
            )
            time.sleep(2)

            r2 = agent_memory.generate_user_summary(
                user_id=user_id,
                thread_ids=[t1, t2],
                poll_interval=_POLL_INTERVAL,
                timeout=_TIMEOUT,
            )
            assert r2.get("runtimeStatus") == "Completed"
            time.sleep(2)

            user_summaries = agent_memory.get_user_summary(user_id)
            assert len(user_summaries) >= 1, "Expected at least 1 user_summary"
            combined_content = " ".join(
                s.get("content", "") for s in user_summaries
            ).lower()
            # The final summary should reference both topics.
            assert "garden" in combined_content or "tomato" in combined_content or "basil" in combined_content, (
                "User summary should mention gardening topic"
            )
            assert "photo" in combined_content or "camera" in combined_content or "golden" in combined_content, (
                "User summary should mention photography topic"
            )
        finally:
            _cleanup_memories(agent_memory, user_id, t1)
            _cleanup_memories(agent_memory, user_id, t2)
            _cleanup_user(agent_memory, user_id)


class TestSearchSummariesAndFacts:
    """Scenario 6: add turns → extract facts → generate summary → search."""

    def test_search_summaries_and_facts(
        self, agent_memory, unique_user_id, unique_thread_id
    ):
        user_id = unique_user_id
        thread_id = unique_thread_id
        try:
            _add_turns(
                agent_memory, user_id, thread_id,
                [
                    ("user", "I have a golden retriever named Buddy."),
                    ("agent", "Golden retrievers are great family dogs! How old is Buddy?"),
                    ("user", "Buddy is 3 years old and loves playing fetch at the park."),
                ],
            )
            time.sleep(2)

            # Extract facts
            fact_result = agent_memory.extract_facts(
                user_id=user_id,
                thread_id=thread_id,
                poll_interval=_POLL_INTERVAL,
                timeout=_TIMEOUT,
            )
            assert fact_result.get("runtimeStatus") == "Completed"

            # Generate thread summary
            summary_result = agent_memory.generate_thread_summary(
                user_id=user_id,
                thread_id=thread_id,
                poll_interval=_POLL_INTERVAL,
                timeout=_TIMEOUT,
            )
            assert summary_result.get("runtimeStatus") == "Completed"
            time.sleep(3)

            # -- Vector search --
            vector_results = agent_memory.search_cosmos(
                search_terms="golden retriever dog",
                user_id=user_id,
                top_k=5,
            )
            assert len(vector_results) >= 1, (
                "Vector search for 'golden retriever dog' should return at least 1 result"
            )

            # -- Hybrid search --
            hybrid_results = agent_memory.search_cosmos(
                search_terms="Buddy the dog park",
                user_id=user_id,
                hybrid_search=True,
                top_k=5,
            )
            assert len(hybrid_results) >= 1, (
                "Hybrid search for 'Buddy the dog park' should return at least 1 result"
            )
        finally:
            _cleanup_user(agent_memory, user_id)
