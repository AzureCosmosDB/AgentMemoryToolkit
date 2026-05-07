"""Advanced memory lifecycle — create → process → archive (delete raw turns).

Demonstrates the typical long-term-memory flow:
  1. Add raw conversation turns
  2. Extract structured memories (facts / procedural / episodic)
  3. Generate a thread summary
  4. Delete raw turns — keeping only the compact derived memories

Uses the in-process ProcessingPipeline (same code as the Azure Function
change-feed trigger). No Function deployment required.

Required env vars (.env supported):

    COSMOS_DB_ENDPOINT, AI_FOUNDRY_ENDPOINT, AI_FOUNDRY_EMBEDDING_DEPLOYMENT_NAME, AI_FOUNDRY_CHAT_DEPLOYMENT_NAME
    COSMOS_DB_KEY (optional fallback)
"""

from __future__ import annotations

import os
import sys
import uuid

from dotenv import load_dotenv

from agent_memory_toolkit import CosmosMemoryClient

load_dotenv()


def _header(step: int, title: str) -> None:
    print(f"\n{'=' * 60}\n  STEP {step}: {title}\n{'=' * 60}")


def _print_memories(mem: CosmosMemoryClient, user_id: str, thread_id: str) -> None:
    items = mem.get_memories(user_id=user_id, thread_id=thread_id)
    by_type: dict[str, int] = {}
    for m in items:
        by_type[m.get("type", "?")] = by_type.get(m.get("type", "?"), 0) + 1
    print(f"  total memories: {len(items)} :: {by_type}")
    for m in items:
        tags = ", ".join(m.get("tags") or [])
        print(f"   • [{m.get('type', '?'):11}] {m['content'][:80]}  [tags={tags}]")


def main() -> None:
    required = ["COSMOS_DB_ENDPOINT", "AI_FOUNDRY_ENDPOINT"]
    missing = [v for v in required if not os.environ.get(v)]
    if missing:
        print(f"ERROR: missing env vars: {', '.join(missing)}")
        sys.exit(1)

    mem = CosmosMemoryClient(
        cosmos_endpoint=os.environ["COSMOS_DB_ENDPOINT"],
        cosmos_key=os.environ.get("COSMOS_DB_KEY") or None,
        cosmos_database=os.environ.get("COSMOS_DB_DATABASE", "ai_memory"),
        cosmos_container=os.environ.get("COSMOS_DB_CONTAINER", "memories"),
        ai_foundry_endpoint=os.environ["AI_FOUNDRY_ENDPOINT"],
        ai_foundry_api_key=os.environ.get("AI_FOUNDRY_API_KEY") or None,
        embedding_deployment_name=os.environ.get("AI_FOUNDRY_EMBEDDING_DEPLOYMENT_NAME", "text-embedding-3-large"),
        chat_deployment_name=os.environ.get("AI_FOUNDRY_CHAT_DEPLOYMENT_NAME", "gpt-4o-mini"),
    )
    print("✅ Connected to Cosmos DB")

    user_id = f"demo-user-{uuid.uuid4().hex[:6]}"
    thread_id = str(uuid.uuid4())

    _header(1, "Add raw conversation turns")
    for role, content in [
        ("user", "I'm building a recommendation engine for an online bookstore."),
        ("agent", "Great! Are you using collaborative filtering or content-based?"),
        ("user", "Hybrid. I want to use embeddings on book descriptions and reviews."),
        ("agent", "Cosmos DB for NoSQL with the vector index works well for that."),
        ("user", "I prefer Python and want to use FastAPI for the API layer."),
        ("agent", "FastAPI is a great choice — fast, type-safe, async-native."),
        ("user", "Last quarter I tried doing this with Pinecone and the costs blew up."),
    ]:
        mem.add_cosmos(user_id=user_id, role=role, content=content, thread_id=thread_id)
    _print_memories(mem, user_id, thread_id)

    _header(2, "Extract structured memories (facts / procedural / episodic)")
    stats = mem.extract_memories(user_id=user_id, thread_id=thread_id)
    print(f"  extraction stats: {stats}")

    _header(3, "Generate thread summary")
    summary_doc = mem.generate_thread_summary(user_id=user_id, thread_id=thread_id)
    print(f"  summary id: {summary_doc['id']}")
    print(f"  summary   : {summary_doc['content'][:200]}…")

    _header(4, "Inventory before archiving raw turns")
    _print_memories(mem, user_id, thread_id)

    _header(5, "Archive: delete raw turns, keep derived memories")
    deleted = 0
    for m in mem.get_memories(user_id=user_id, thread_id=thread_id, memory_types=["turn"]):
        mem.delete_cosmos(memory_id=m["id"], thread_id=thread_id, user_id=user_id)
        deleted += 1
    print(f"  deleted {deleted} raw turn(s)")

    _header(6, "Final inventory — only compact long-term memory remains")
    _print_memories(mem, user_id, thread_id)

    _header(7, "Search still works against the archived knowledge")
    for q in [
        "what is the user building",
        "what programming language does the user prefer",
        "what previous experience does the user have with vector databases",
    ]:
        print(f'\n  query: "{q}"')
        for r in mem.search_cosmos(search_terms=q, user_id=user_id, top_k=3):
            print(f"    → [{r.get('type')}] {r['content'][:80]}")

    print("\nDone.")


if __name__ == "__main__":
    main()
