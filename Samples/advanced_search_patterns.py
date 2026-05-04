"""
Advanced Search Patterns — Agent Memory Toolkit

Demonstrates vector, hybrid, and filtered search patterns using Cosmos DB
with AI Foundry embeddings.

Required environment variables:
    COSMOS_DB_ENDPOINT  – Azure Cosmos DB endpoint URL
    AI_FOUNDRY_ENDPOINT – Azure AI Foundry endpoint URL (for embeddings)
"""

import os

from dotenv import load_dotenv
load_dotenv()
import uuid

from agent_memory_toolkit import CosmosMemoryClient

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def print_header(title: str) -> None:
    """Print a formatted section header."""
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print(f"{'=' * 60}")


def print_results(results: list) -> None:
    """Print search results in a readable format."""
    if not results:
        print("  (no results)")
        return
    for i, result in enumerate(results, 1):
        content = result.get("content", "") if isinstance(result, dict) else str(result)
        role = result.get("role", "n/a") if isinstance(result, dict) else "n/a"
        mem_type = result.get("type", "n/a") if isinstance(result, dict) else "n/a"
        salience = result.get("salience") if isinstance(result, dict) else None
        suffix = f"  [salience: {salience:.2f}]" if isinstance(salience, (int, float)) else ""
        print(f"  {i}. [{role}] ({mem_type}) {content[:100]}{suffix}")


# ---------------------------------------------------------------------------
# Seed data
# ---------------------------------------------------------------------------

def seed_memories(mem: CosmosMemoryClient, user_id: str, thread_id: str) -> None:
    """Populate the store with sample memories for searching."""
    entries = [
        {"role": "user", "content": "I love hiking in the Pacific Northwest", "memory_type": "turn"},
        {
            "role": "agent",
            "content": "The PNW has amazing trails like the Wonderland Trail and the PCT!",
            "memory_type": "turn",
        },
        {"role": "user", "content": "My favorite food is sushi, especially salmon nigiri", "memory_type": "fact"},
        {"role": "user", "content": "I usually run 5 miles every morning before work", "memory_type": "turn"},
        {
            "role": "agent",
            "content": "Running is a great way to stay fit! Do you prefer road or trail running?",
            "memory_type": "turn",
        },
        {"role": "user", "content": "I work as a software engineer at a startup in Seattle", "memory_type": "fact"},
        {"role": "user", "content": "My preferred programming language is Python", "memory_type": "fact"},
        {
            "role": "agent",
            "content": "Python is very popular for AI/ML workloads. What frameworks do you use?",
            "memory_type": "turn",
        },
    ]

    print("Seeding memories …")
    for entry in entries:
        mem.add_cosmos(
            user_id=user_id,
            role=entry["role"],
            content=entry["content"],
            memory_type=entry["memory_type"],
            thread_id=thread_id,
        )
    print(f"  ✓ {len(entries)} memories added (thread {thread_id[:8]}…)\n")


# ---------------------------------------------------------------------------
# Search patterns
# ---------------------------------------------------------------------------

def vector_search(mem: CosmosMemoryClient, user_id: str) -> None:
    """Pattern 1 — Pure vector (semantic similarity) search."""
    print_header("1. Vector Search (semantic similarity)")
    print("  Query: 'outdoor activities'")
    print("  Finds semantically related memories even without exact keyword matches.\n")

    results = mem.search_cosmos(
        search_terms="outdoor activities",
        user_id=user_id,
        top_k=5,
    )
    print_results(results)


def hybrid_search(mem: CosmosMemoryClient, user_id: str) -> None:
    """Pattern 2 — Hybrid search (vector + full-text)."""
    print_header("2. Hybrid Search (vector + full-text)")
    print("  Query: 'hiking trails Pacific Northwest'")
    print("  Combines embedding similarity with BM25 keyword matching.\n")

    results = mem.search_cosmos(
        search_terms="hiking trails Pacific Northwest",
        user_id=user_id,
        hybrid_search=True,
        top_k=5,
    )
    print_results(results)


def filtered_by_role(mem: CosmosMemoryClient, user_id: str) -> None:
    """Pattern 3 — Filtered search: only user messages."""
    print_header("3. Filtered Search — by role ('user')")
    print("  Query: 'preferences'")
    print("  Restricts results to a specific conversation role.\n")

    results = mem.search_cosmos(
        search_terms="preferences",
        user_id=user_id,
        role="user",
        top_k=3,
    )
    print_results(results)


def filtered_by_memory_type(mem: CosmosMemoryClient, user_id: str) -> None:
    """Pattern 4 — Filtered search: only 'fact' memories."""
    print_header("4. Filtered Search — by memory_type ('fact')")
    print("  Query: 'food preferences'")
    print("  Narrows results to a specific memory category.\n")

    results = mem.search_cosmos(
        search_terms="food preferences",
        user_id=user_id,
        memory_type="fact",
        top_k=3,
    )
    print_results(results)


def filtered_by_thread(mem: CosmosMemoryClient, user_id: str, thread_id: str) -> None:
    """Pattern 5 — Filtered search: scoped to a single thread."""
    print_header("5. Filtered Search — by thread_id")
    print(f"  Query: 'activities'  |  thread: {thread_id[:8]}…")
    print("  Limits results to a specific conversation thread.\n")

    results = mem.search_cosmos(
        search_terms="activities",
        user_id=user_id,
        thread_id=thread_id,
        top_k=3,
    )
    print_results(results)


def top_k_tuning(mem: CosmosMemoryClient, user_id: str) -> None:
    """Pattern 6 — top-k tuning comparison."""
    print_header("6. Top-K Tuning Comparison")
    print("  Query: 'hobbies and interests'")
    print("  Demonstrates how top_k affects the breadth of results.\n")

    for k in (1, 3, 5):
        results = mem.search_cosmos(
            search_terms="hobbies and interests",
            user_id=user_id,
            top_k=k,
        )
        print(f"  --- top_k={k} → {len(results)} result(s) ---")
        print_results(results)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    cosmos_endpoint = os.environ.get("COSMOS_DB_ENDPOINT")
    ai_foundry_endpoint = os.environ.get("AI_FOUNDRY_ENDPOINT")

    if not cosmos_endpoint or not ai_foundry_endpoint:
        raise SystemExit(
            "Error: Set COSMOS_DB_ENDPOINT and AI_FOUNDRY_ENDPOINT environment variables."
        )

    mem = CosmosMemoryClient(
        cosmos_endpoint=cosmos_endpoint,
        cosmos_database=os.environ.get("COSMOS_DB_DATABASE", "ai_memory"),
        cosmos_container=os.environ.get("COSMOS_DB_CONTAINER", "memories"),
        cosmos_key=os.environ.get("COSMOS_DB_KEY"),
        ai_foundry_endpoint=ai_foundry_endpoint,
        ai_foundry_api_key=os.environ.get("AI_FOUNDRY_API_KEY"),
        embedding_deployment_name=os.environ.get("AI_FOUNDRY_EMBEDDING_DEPLOYMENT_NAME", "text-embedding-3-large"),
        chat_deployment_name=os.environ.get("AI_FOUNDRY_CHAT_DEPLOYMENT_NAME", "gpt-4o-mini"),
    )
    user_id = "search-demo-user"
    thread_id = str(uuid.uuid4())

    seed_memories(mem, user_id, thread_id)

    vector_search(mem, user_id)
    hybrid_search(mem, user_id)
    filtered_by_role(mem, user_id)
    filtered_by_memory_type(mem, user_id)
    filtered_by_thread(mem, user_id, thread_id)
    top_k_tuning(mem, user_id)

    print(f"\n{'=' * 60}")
    print("  All search patterns complete.")
    print(f"{'=' * 60}\n")


if __name__ == "__main__":
    main()
