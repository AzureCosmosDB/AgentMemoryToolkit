"""Live episodic-memory integration test.

This module deliberately builds its Cosmos client with Microsoft Entra ID
(``DefaultAzureCredential``) rather than a key because the live integration
account disables local auth.
"""

from __future__ import annotations

import time

import pytest

from azure.cosmos.agent_memory import CosmosMemoryClient
from tests.conftest import INTEGRATION_ENABLED

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        not INTEGRATION_ENABLED,
        reason="Set AGENT_MEMORY_RUN_INTEGRATION=true",
    ),
]

VALID_OUTCOMES = {"successful", "partially_successful", "failed", "abandoned", "unknown"}


@pytest.fixture(scope="module")
def episodic_memory(
    cosmos_endpoint,
    cosmos_database,
    cosmos_container,
    ai_foundry_endpoint,
    ai_foundry_api_key,
    embedding_deployment_name,
    embedding_dimensions,
    chat_deployment_name,
):
    """Live client using AAD for Cosmos and existing containers only."""
    if not cosmos_endpoint or not ai_foundry_endpoint:
        pytest.skip("COSMOS_DB_ENDPOINT / AI_FOUNDRY_ENDPOINT not set")

    mem = CosmosMemoryClient(
        cosmos_database=cosmos_database,
        cosmos_container=cosmos_container,
        ai_foundry_endpoint=ai_foundry_endpoint,
        ai_foundry_api_key=ai_foundry_api_key or None,
        embedding_deployment_name=embedding_deployment_name,
        embedding_dimensions=embedding_dimensions,
        chat_deployment_name=chat_deployment_name,
        use_default_credential=True,
        cadence_thresholds={
            "FACT_EXTRACTION_EVERY_N": 1_000_000,
            "EPISODE_EVAL_EVERY_N": 1_000_000,
            "THREAD_SUMMARY_EVERY_N": 1_000_000,
            "USER_SUMMARY_EVERY_N": 1_000_000,
            "DEDUP_EVERY_N": 1_000_000,
        },
    )
    mem._maybe_auto_trigger = lambda turn_counts: None  # type: ignore[method-assign]
    mem.connect_cosmos(
        endpoint=cosmos_endpoint,
        database=cosmos_database,
        container=cosmos_container,
    )
    try:
        yield mem
    finally:
        mem.close()


def _delete_user_records(mem: CosmosMemoryClient, user_id: str) -> None:
    query = "SELECT c.id, c.thread_id FROM c WHERE c.user_id = @user_id"
    params = [{"name": "@user_id", "value": user_id}]
    for container in (
        mem._turns_container_client,
        mem._memories_container_client,
        mem._summaries_container_client,
    ):
        try:
            docs = list(
                container.query_items(
                    query=query,
                    parameters=params,
                    enable_cross_partition_query=True,
                )
            )
        except Exception:
            continue
        for doc in docs:
            try:
                container.delete_item(
                    item=doc["id"],
                    partition_key=[user_id, doc.get("thread_id", "")],
                )
            except Exception:
                pass


def _write_turns(mem: CosmosMemoryClient, user_id: str, thread_id: str, turns: list[tuple[str, str]]) -> set[str]:
    turn_ids = set()
    for role, content in turns:
        turn_ids.add(
            mem.upsert_memory(
                user_id=user_id,
                role=role,
                content=content,
                memory_type="turn",
                thread_id=thread_id,
            )
        )
    return turn_ids


def _write_hiking_thread(mem: CosmosMemoryClient, user_id: str, thread_id: str) -> set[str]:
    turns = [
        (
            "user",
            "On Friday 2026-07-17, Priya, Marco, and I planned a weekend hiking trip to "
            "Mount Rainier's Skyline Trail for Saturday morning.",
        ),
        (
            "agent",
            "That sounds like a clear plan: Saturday morning on Skyline Trail with Priya and Marco.",
        ),
        (
            "user",
            "We left Seattle at 6:30 AM on Saturday, stopped in Ashford for coffee, and reached "
            "Paradise before the main parking lot filled.",
        ),
        (
            "agent",
            "Getting to Paradise early likely helped the group start the hike before the crowds.",
        ),
        (
            "user",
            "Near Panorama Point, Marco slipped on a wet rock and scraped his knee, so Priya used "
            "the small first-aid kit I had packed.",
        ),
        (
            "user",
            "After a short rest, Marco felt okay, and we continued slowly to see the wildflowers "
            "and the Nisqually Glacier views.",
        ),
        (
            "agent",
            "The first-aid kit turned the mishap into a manageable pause rather than ending the hike.",
        ),
        (
            "user",
            "We got back to Seattle by 7 PM Saturday, tired but happy, and decided next time we "
            "would bring trekking poles for the steeper wet sections.",
        ),
    ]
    return _write_turns(mem, user_id, thread_id, turns)


def _write_export_fix_thread(mem: CosmosMemoryClient, user_id: str, thread_id: str) -> set[str]:
    """A thread in which the user and the assistant fix a recurring failure with a method that works.

    The hiking trip's lessons are general advice from one outing, which the procedure pass rightly leaves
    out. This is what it is meant to learn from an episode: a multi-step method specific to the user's
    environment, carried out and successful.
    """
    turns = [
        (
            "user",
            "Our nightly billing export in the finance-etl repo failed again with a lock timeout "
            "on the staging Postgres database.",
        ),
        (
            "agent",
            "A lock timeout usually means a stale session is holding locks. Let's look for "
            "idle-in-transaction sessions in pg_stat_activity.",
        ),
        (
            "user",
            "I ran SELECT pid, now() - xact_start AS age FROM pg_stat_activity WHERE state = "
            "'idle in transaction' and found three sessions older than 10 minutes.",
        ),
        (
            "agent",
            "Terminate those three with pg_terminate_backend(pid), then rerun the export with make export-billing.",
        ),
        (
            "user",
            "Done: I terminated the three stale sessions with pg_terminate_backend and reran "
            "make export-billing, and the export finished in 4 minutes.",
        ),
        (
            "agent",
            "Clearing the stale sessions fixed the run, so idle-in-transaction sessions caused the lock timeout.",
        ),
        (
            "user",
            "Good. Whenever the billing export fails on a lock timeout, we should first list the "
            "idle-in-transaction sessions older than 10 minutes, terminate them, and only then rerun "
            "make export-billing.",
        ),
        (
            "agent",
            "Understood: list the stale sessions, terminate them, then rerun make export-billing.",
        ),
    ]
    return _write_turns(mem, user_id, thread_id, turns)


def test_live_episodic_extraction_and_blended_search(
    episodic_memory,
    unique_user_id,
    unique_thread_id,
):
    try:
        turn_ids = _write_hiking_thread(episodic_memory, unique_user_id, unique_thread_id)
        time.sleep(1)

        stats = episodic_memory.extract_episodes(unique_user_id, unique_thread_id, flush=True)
        assert stats.get("episodes", 0) >= 1, f"Expected at least one extracted episode, got {stats}"

        episodes = episodic_memory.get_episodes(unique_user_id)
        assert len(episodes) >= 1
        episode = episodes[0]
        assert episode.get("content")
        assert episode.get("title")
        assert episode.get("started_at")

        events = episode.get("events") or []
        assert events, f"Expected at least one grounded event, got {episode}"
        assert any(set(event.get("source_turn_ids") or []) & turn_ids for event in events), (
            f"Expected event source_turn_ids to reference written turns {turn_ids}, got {events}"
        )

        outcome = episode.get("outcome")
        assert outcome is None or outcome.get("status") in VALID_OUTCOMES

        results = episodic_memory.search_cosmos(
            search_terms="the hiking trip",
            user_id=unique_user_id,
            include_episodes=True,
        )
        assert any(result.get("type") == "episodic" for result in results), results
    finally:
        _delete_user_records(episodic_memory, unique_user_id)


def test_live_episodic_lessons_feed_procedural_synthesis(
    episodic_memory,
    unique_user_id,
    unique_thread_id,
):
    """Episodes carry first-class ``lessons`` that must flow into procedural synthesis.

    Uses a fresh user with no extracted facts, so the only possible source for the
    procedure pass is episodic lessons - this isolates the episodic->procedural
    seam end-to-end against live Cosmos (real ``IS_DEFINED(c.lessons)`` filtering).
    The thread is a multi-step fix that worked, which is what the pass learns from an
    episode: a trip's lessons are general advice that it correctly leaves out.

    ``synthesize_procedural`` returns counts, never the procedures, and a run reads
    only the sources written since the last one. So: the first run must read the
    episodes (``synthesized``, not ``unchanged``) and write at least one candidate
    procedure that cites them (and only them), and a second run has nothing new
    left to read.
    """
    try:
        _write_export_fix_thread(episodic_memory, unique_user_id, unique_thread_id)
        time.sleep(1)

        stats = episodic_memory.extract_episodes(unique_user_id, unique_thread_id, flush=True)
        assert stats.get("episodes", 0) >= 1, f"Expected at least one extracted episode, got {stats}"

        episodes = episodic_memory.get_episodes(unique_user_id)
        lesson_bearing = {
            ep["id"]
            for ep in episodes
            if isinstance(ep.get("lessons"), list)
            and any(isinstance(lesson, str) and lesson.strip() for lesson in ep.get("lessons", []))
        }
        assert lesson_bearing, (
            "Expected extract_episodes to write at least one episode with first-class lessons, "
            f"got {[ep.get('lessons') for ep in episodes]}"
        )

        result = episodic_memory.synthesize_procedural(unique_user_id)
        assert result.get("status") == "synthesized", (
            f"Expected the lesson-bearing episodes to be read and answered by the model, got {result}"
        )
        counts = {"procedures_created", "procedures_updated", "procedures_reinforced", "procedures_skipped"}
        assert counts <= set(result), result
        assert result.get("facts_retired") == 0, "no fact was extracted, so none can be retired into a procedure"
        assert result["procedures_created"] >= 1, (
            f"Expected the model to learn at least one procedure from the episodes' lessons, got {result}"
        )

        procedures = episodic_memory.get_procedural_memories(unique_user_id)
        assert len(procedures) >= result["procedures_created"], (result, procedures)
        assert any(set(procedure.get("source_episodic_ids") or []) & lesson_bearing for procedure in procedures), (
            f"Expected a stored procedure to cite the lesson-bearing episodes {lesson_bearing}, got {procedures}"
        )
        for procedure in procedures:
            assert set(procedure.get("source_episodic_ids") or []) <= lesson_bearing, (
                f"Expected a procedure to cite only lesson-bearing episodes {lesson_bearing}, got {procedure}"
            )
            assert not procedure.get("source_fact_ids"), procedure
            # An episode can neither teach nor correct, and these all fall on one date, so none has met its bar.
            assert procedure.get("status") == "candidate", procedure

        again = episodic_memory.synthesize_procedural(unique_user_id)
        assert again.get("status") == "unchanged", (
            f"Expected every lesson-bearing episode to be consumed by the first run, got {again}"
        )
    finally:
        _delete_user_records(episodic_memory, unique_user_id)
