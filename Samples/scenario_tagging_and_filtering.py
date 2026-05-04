"""Scenario: Tagging memories and filtering retrieval by tags.

This sample showcases the tagging + filtering surface as a first-class
feature. Tags are arbitrary string labels attached to a memory at write
time; at read time you can filter retrieval with three composable predicates:

* ``tags=[...]``        — AND filter: every listed tag must be present
* ``any_tags=[...]``    — OR filter: at least one listed tag must be present
* ``exclude_tags=[...]`` — NOT filter: none of these tags may be present

Tags are stored in Cosmos as a JSON array on the memory document. They work
across every memory type (``turn``, ``fact``, ``episodic``, ``procedural``,
``summary``, ``user_summary``) and compose freely with semantic / hybrid
search and the other structural filters (``user_id``, ``thread_id``,
``memory_type`` …).

Required environment variables:
    COSMOS_DB_ENDPOINT   – Azure Cosmos DB endpoint URL
    AI_FOUNDRY_ENDPOINT  – Azure AI Foundry endpoint URL (for embeddings)
    AI_FOUNDRY_EMBEDDING_DEPLOYMENT_NAME – embedding model deployment name
"""
from __future__ import annotations

import os
import uuid

from dotenv import load_dotenv

load_dotenv()

from agent_memory_toolkit import CosmosMemoryClient


def print_section(title: str) -> None:
    print(f"\n{'=' * 60}\n  {title}\n{'=' * 60}")


def print_results(results: list) -> None:
    if not results:
        print("  (no results)")
        return
    for i, r in enumerate(results, 1):
        tags = r.get("tags", [])
        mt = r.get("memory_type") or r.get("type", "n/a")
        content = r.get("content", "")[:80]
        print(f"  {i}. [{mt:>10}] tags={tags}  {content}")


def main() -> None:
    user_id = "alice"
    thread_id = f"thread-{uuid.uuid4().hex[:8]}"

    client = CosmosMemoryClient(
        cosmos_endpoint=os.environ["COSMOS_DB_ENDPOINT"],
        cosmos_key=os.environ.get("COSMOS_DB_KEY"),
        ai_foundry_endpoint=os.environ["AI_FOUNDRY_ENDPOINT"],
        embedding_deployment_name=os.environ.get("AI_FOUNDRY_EMBEDDING_DEPLOYMENT_NAME", "text-embedding-3-large"),
    )

    # ------------------------------------------------------------------
    # 1. Write memories of different types with varied tag sets
    # ------------------------------------------------------------------
    print_section("Seeding memories with tags")
    seeds = [
        # (memory_type, content, tags)
        ("fact",       "User prefers Python over Java for backend work",
         ["preference", "language", "work"]),
        ("fact",       "User is allergic to peanuts",
         ["preference", "diet", "health", "important"]),
        ("fact",       "User lives in Seattle, WA",
         ["profile", "location"]),
        ("episodic",   "Last Friday the user shipped a Cosmos-backed search service",
         ["work", "milestone"]),
        ("episodic",   "User attended PyCon 2024 in Pittsburgh",
         ["work", "travel", "conference"]),
        ("procedural", "When the user says 'deploy', run `azd up` and tail the logs",
         ["workflow", "ops"]),
        ("procedural", "Always confirm before purging Cosmos containers",
         ["workflow", "ops", "important"]),
    ]
    for mem_type, content, tags in seeds:
        client.add_cosmos(
            user_id=user_id,
            role="system",
            content=content,
            memory_type=mem_type,
            thread_id=thread_id,
            tags=tags,
        )
        print(f"  + {mem_type:>10}  tags={tags}")

    # ------------------------------------------------------------------
    # 2. AND filter: tags=[...] — every listed tag must be present
    #    Use get_memories() for filter-only retrieval (no vector search).
    # ------------------------------------------------------------------
    print_section("AND filter — tags=['workflow', 'important']")
    results = client.get_memories(
        user_id=user_id,
        tags=["workflow", "important"],
    )
    print_results(results)
    # Expected: only the "Always confirm before purging" procedural memory.

    # ------------------------------------------------------------------
    # 3. OR filter: any_tags=[...] — any listed tag may match
    # ------------------------------------------------------------------
    print_section("OR filter — any_tags=['diet', 'travel']")
    results = client.get_memories(
        user_id=user_id,
        any_tags=["diet", "travel"],
    )
    print_results(results)
    # Expected: the peanut allergy fact and the PyCon episodic memory.

    # ------------------------------------------------------------------
    # 4. NOT filter: exclude_tags=[...] — drop anything carrying these tags
    # ------------------------------------------------------------------
    print_section("NOT filter — exclude_tags=['ops']  (memory_type='procedural')")
    results = client.get_memories(
        user_id=user_id,
        memory_type="procedural",
        exclude_tags=["ops"],
    )
    print_results(results)
    # Expected: empty — both procedural memories carry the 'ops' tag.

    # ------------------------------------------------------------------
    # 5. Combine tag filters with semantic search
    #    search_cosmos() runs vector similarity against search_terms and
    #    composes naturally with structural / tag filters.
    # ------------------------------------------------------------------
    print_section("Semantic + tag filter — search_terms='shipping a service' tags=['work']")
    results = client.search_cosmos(
        search_terms="shipping a service",
        user_id=user_id,
        tags=["work"],
        top_k=5,
    )
    print_results(results)
    # Expected: ranked by vector similarity but constrained to memories
    # tagged 'work'.

    # ------------------------------------------------------------------
    # 6. Compose AND + NOT filters across types
    # ------------------------------------------------------------------
    print_section("AND + NOT — tags=['preference']  exclude_tags=['health']")
    results = client.get_memories(
        user_id=user_id,
        tags=["preference"],
        exclude_tags=["health"],
    )
    print_results(results)
    # Expected: only the "prefers Python" fact (peanut allergy is excluded by 'health').


if __name__ == "__main__":
    main()
