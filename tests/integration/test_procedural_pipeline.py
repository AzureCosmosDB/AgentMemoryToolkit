"""Live procedural-memory integration test.

This module deliberately builds its Cosmos client with Microsoft Entra ID
(``DefaultAzureCredential``) rather than a key because the live integration
account disables local auth.
"""

from __future__ import annotations

import os
import time

import pytest

from azure.cosmos.agent_memory import CosmosMemoryClient
from azure.cosmos.agent_memory.services._procedures import procedure_status

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        os.getenv("AGENT_MEMORY_RUN_INTEGRATION") != "true",
        reason="Set AGENT_MEMORY_RUN_INTEGRATION=true",
    ),
]


@pytest.fixture(scope="module")
def procedural_memory(
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


def _write_procedural_seed_thread(mem: CosmosMemoryClient, user_id: str, thread_id: str) -> set[str]:
    turns = [
        (
            "user",
            "This is a durable requirement for future work: always confirm with me before deleting "
            "files, database records, cloud resources, or any other destructive or irreversible data.",
        ),
        (
            "agent",
            "Understood. I will ask for explicit confirmation before destructive or irreversible deletions.",
        ),
        (
            "user",
            "Last week I debugged a Cosmos DB hybrid search failure where ORDER BY VectorDistance "
            "started failing in an integration test.",
        ),
        (
            "agent",
            "What finally resolved the Cosmos DB hybrid search failure?",
        ),
        (
            "user",
            "The useful lesson was to inspect the vector and full-text indexing policy, fix the "
            "missing vector path, wait for the policy to propagate, and rerun the focused integration test.",
        ),
        (
            "agent",
            "That is a reusable recovery strategy for Cosmos DB hybrid search ORDER BY failures.",
        ),
    ]
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


def _wait_for_procedure_retrieval(
    mem: CosmosMemoryClient,
    user_id: str,
    search_terms: str,
    active_procedure_ids: set[str],
    *,
    timeout: float = 20.0,
) -> list[dict]:
    deadline = time.time() + timeout
    last_results: list[dict] = []
    while time.time() < deadline:
        try:
            last_results = mem.retrieve_procedures(user_id, search_terms, top_k=5)
            if any(result.get("id") in active_procedure_ids for result in last_results):
                return last_results
        except Exception:
            pass
        time.sleep(1)
    return last_results


def _assert_context_excludes_candidates(context: str, candidates: list[dict]) -> None:
    for candidate in candidates:
        for field in ("name", "summary", "retrieval_text", "content"):
            value = candidate.get(field)
            if isinstance(value, str) and value.strip():
                assert value not in context, f"Candidate {field} leaked into procedural context: {value!r}"


def test_live_procedural_synthesis_retrieval_and_context(
    procedural_memory,
    unique_user_id,
    unique_thread_id,
):
    try:
        _write_procedural_seed_thread(procedural_memory, unique_user_id, unique_thread_id)
        time.sleep(1)

        memory_stats = procedural_memory.extract_memories(unique_user_id, unique_thread_id)
        facts = procedural_memory.get_memories(user_id=unique_user_id, memory_types=["fact"])
        behavioral_facts = [
            fact
            for fact in facts
            if fact.get("metadata", {}).get("category") in {"preference", "requirement"}
            or float(fact.get("salience") or 0.0) >= 0.8
        ]
        assert memory_stats.get("fact_count", 0) >= 1 or behavioral_facts, (
            f"Expected at least one behavioral fact extracted from the deletion requirement, got {memory_stats}"
        )

        episode_stats = procedural_memory.extract_episodes(unique_user_id, unique_thread_id, flush=True)
        assert episode_stats.get("episodes", 0) >= 1, (
            f"Expected at least one extracted episode with a debugging lesson, got {episode_stats}"
        )
        episodes = procedural_memory.get_episodes(unique_user_id)
        lesson_bearing = [
            episode
            for episode in episodes
            if any(isinstance(lesson, str) and lesson.strip() for lesson in (episode.get("lessons") or []))
        ]
        assert lesson_bearing, f"Expected at least one episode with first-class lessons, got {episodes}"

        result = procedural_memory.synthesize_procedural(unique_user_id)
        assert result.get("status") == "synthesized", result
        assert result.get("procedures_created", 0) >= 1, result
        counts = {"procedures_updated", "procedures_reinforced", "procedures_skipped", "facts_retired"}
        assert counts <= set(result), result

        procedures = procedural_memory.get_procedural_memories(unique_user_id)
        assert len(procedures) >= result.get("procedures_created", 0)
        for procedure in procedures:
            assert procedure.get("type") == "procedural"
            assert procedure.get("name")
            assert procedure.get("procedure_kind") in {"instruction", "habit", "skill"}
            assert procedure.get("applies") in {"always", "when"}
            assert procedure.get("status") in {"active", "candidate"}
            assert procedure.get("retrieval_text")
            # The status follows from the evidence gathered; the model never sets it.
            assert procedure.get("status") == procedure_status(procedure), procedure

        active = [procedure for procedure in procedures if procedure.get("status") == "active"]
        candidates = [procedure for procedure in procedures if procedure.get("status") == "candidate"]
        assert active, (
            "Expected the user's directive to always confirm before deleting to become an active procedure, "
            f"got {procedures}"
        )

        fact_only = [
            procedure
            for procedure in procedures
            if procedure.get("source_fact_ids") and not procedure.get("source_episodic_ids")
        ]
        episode_only = [
            procedure
            for procedure in procedures
            if procedure.get("source_episodic_ids") and not procedure.get("source_fact_ids")
        ]
        if fact_only:
            assert any(procedure.get("status") == "active" for procedure in fact_only), fact_only
        if episode_only:
            assert all(procedure.get("status") == "candidate" for procedure in episode_only), episode_only

        # A run reads only what is new, so with nothing written since, another makes no model call.
        again = procedural_memory.synthesize_procedural(unique_user_id)
        assert again.get("status") == "unchanged", again

        task = "delete destructive irreversible confirmation before deleting data"
        active_ids = {procedure["id"] for procedure in active}
        retrieved = _wait_for_procedure_retrieval(procedural_memory, unique_user_id, task, active_ids)
        assert retrieved, "Expected retrieve_procedures to return an active procedure"
        assert retrieved[0].get("id") in active_ids, retrieved

        # Search shows a procedure only while it is active.
        searched = procedural_memory.search_cosmos(
            search_terms=task, user_id=unique_user_id, memory_types=["procedural"], top_k=10
        )
        assert {hit["id"] for hit in searched} <= active_ids, searched

        context = procedural_memory.build_procedural_context(unique_user_id, task=task)
        assert context.strip()
        assert any(
            (procedure.get("name") and procedure["name"] in context)
            or (procedure.get("summary") and procedure["summary"] in context)
            for procedure in active
        ), context
        _assert_context_excludes_candidates(context, candidates)

        prompt = procedural_memory.get_procedural_prompt(unique_user_id)
        assert prompt is None or prompt.strip()
        if prompt:
            _assert_context_excludes_candidates(prompt, candidates)
    finally:
        _delete_user_records(procedural_memory, unique_user_id)
