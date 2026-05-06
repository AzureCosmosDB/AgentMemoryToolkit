"""Scenario: Multi-turn chat with memory persistence across sessions.

Demonstrates how an agent can:
1. Carry on a multi-turn conversation, storing each turn in Cosmos DB.
2. Retrieve the full conversation thread later (simulating a reconnect).
3. Search past conversations by topic to recall relevant context.

Requirements:
    - Azure Cosmos DB account with vector-search enabled.
    - Azure AI Foundry endpoint for embeddings.
    - Environment variables:
        COSMOS_DB_ENDPOINT  – Cosmos DB endpoint URL
        AI_FOUNDRY_ENDPOINT – AI Foundry endpoint URL
"""

from __future__ import annotations

import os

from dotenv import load_dotenv
load_dotenv()
import time
import uuid

from agent_memory_toolkit import CosmosMemoryClient

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

DIVIDER = "-" * 60


def banner(title: str) -> None:
    """Print a section banner."""
    print(f"\n{DIVIDER}")
    print(f"  {title}")
    print(DIVIDER)


def print_turn(role: str, content: str) -> None:
    """Pretty-print a single conversation turn."""
    label = "🧑 User " if role == "user" else "🤖 Agent"
    print(f"  {label}: {content}")


def print_thread(thread: list[dict]) -> None:
    """Print every turn in a retrieved thread."""
    if not thread:
        print("  (empty thread)")
        return
    for turn in thread:
        print_turn(turn["role"], turn["content"])


def print_search_results(results: list[dict]) -> None:
    """Print search results with role and content."""
    if not results:
        print("  No results found.")
        return
    for i, r in enumerate(results, 1):
        print(f"  [{i}] ({r['role']}) {r['content']}")

# ---------------------------------------------------------------------------
# Simulated conversations
# ---------------------------------------------------------------------------

SESSION_1_TURNS = [
    ("user", "Hi! I'm planning a trip to Tokyo next month."),
    (
        "agent",
        "That sounds exciting! Tokyo is wonderful in spring. "
        "Would you like recommendations for food, sightseeing, or accommodation?",
    ),
    ("user", "I'd love food recommendations. I'm vegetarian."),
    (
        "agent",
        "Great choice! Try Ain Soph in Shinjuku for plant-based ramen, "
        "and T's TanTan at Tokyo Station for vegan tantanmen.",
    ),
    ("user", "Also, what's the best way to get around the city?"),
    (
        "agent",
        "A Suica or Pasmo IC card is the easiest option. It works on trains, "
        "subways, and buses. The Tokyo Metro day pass is also great value.",
    ),
]

SESSION_2_TURNS = [
    ("user", "I've been having trouble sleeping lately. Any tips?"),
    (
        "agent",
        "Keep a consistent sleep schedule, avoid screens 1 hour before bed, "
        "keep your room cool (around 65°F / 18°C), and try deep breathing.",
    ),
    ("user", "Does caffeine really affect sleep that much?"),
    (
        "agent",
        "Yes — caffeine has a half-life of about 5-6 hours, so a coffee at "
        "3 PM still has half its caffeine at 9 PM. Limit caffeine to mornings.",
    ),
]


def run_session(
    mem: CosmosMemoryClient,
    user_id: str,
    thread_id: str,
    turns: list[tuple[str, str]],
    session_label: str,
) -> None:
    """Simulate a conversation session by storing each turn."""
    banner(f"Session: {session_label}")
    print(f"  thread_id = {thread_id}")
    print()

    for role, content in turns:
        mem.add_cosmos(
            user_id=user_id,
            role=role,
            content=content,
            thread_id=thread_id,
        )
        print_turn(role, content)

    print(f"\n  ✅ Stored {len(turns)} turns in Cosmos DB.")

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    cosmos_endpoint = os.environ.get("COSMOS_DB_ENDPOINT")
    ai_foundry_endpoint = os.environ.get("AI_FOUNDRY_ENDPOINT")

    if not cosmos_endpoint or not ai_foundry_endpoint:
        raise SystemExit(
            "Please set COSMOS_DB_ENDPOINT and AI_FOUNDRY_ENDPOINT environment variables."
        )

    # --- Initialise CosmosMemoryClient and connect to Cosmos DB ---
    banner("Initialising CosmosMemoryClient")
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
    mem.connect_cosmos()
    print("  Connected to Cosmos DB ✔")

    user_id = "demo-user-chat"
    thread_1 = str(uuid.uuid4())
    thread_2 = str(uuid.uuid4())

    # ── Session 1: Travel planning ────────────────────────────
    run_session(mem, user_id, thread_1, SESSION_1_TURNS, "Travel Planning")

    # ── Session 2: Sleep & health ─────────────────────────────
    run_session(mem, user_id, thread_2, SESSION_2_TURNS, "Sleep & Health")

    # Small pause so timestamps differ for ordering
    time.sleep(1)

    # ── Simulate reconnect: retrieve full thread ──────────────
    banner("Reconnect — Retrieve travel thread")
    print(f"  Fetching thread_id = {thread_1}\n")
    thread = mem.get_thread(thread_id=thread_1)
    print_thread(thread)
    print(f"\n  Retrieved {len(thread)} turns from Cosmos DB.")

    # ── Cross-session search: recall relevant context ─────────
    banner("Cross-session search — 'vegetarian food recommendations'")
    results = mem.search_cosmos(
        search_terms="vegetarian food recommendations",
        user_id=user_id,
        top_k=3,
    )
    print_search_results(results)

    banner("Cross-session search — 'caffeine and sleep'")
    results = mem.search_cosmos(
        search_terms="caffeine and sleep",
        user_id=user_id,
        top_k=3,
    )
    print_search_results(results)

    # ── Show how a new session can build on old context ───────
    banner("New session — using recalled context")
    thread_3 = str(uuid.uuid4())
    print(f"  thread_id = {thread_3}\n")

    # Simulate: agent searches memory before responding
    print("  Agent searches memory for 'Tokyo trip' ...")
    recalled = mem.search_cosmos(
        search_terms="Tokyo trip",
        user_id=user_id,
        top_k=2,
    )
    if recalled:
        print(f"  Found {len(recalled)} relevant memories:\n")
        print_search_results(recalled)
    else:
        print("  No prior context found.")

    print()

    # Continue the conversation with context-aware response
    new_turns = [
        ("user", "Can you remind me of the Tokyo restaurant suggestions?"),
        (
            "agent",
            "Of course! Last time we discussed Ain Soph in Shinjuku for plant-based "
            "ramen and T's TanTan at Tokyo Station for vegan tantanmen. Would you like "
            "more options?",
        ),
    ]
    for role, content in new_turns:
        mem.add_cosmos(
            user_id=user_id,
            role=role,
            content=content,
            thread_id=thread_3,
        )
        print_turn(role, content)

    print(f"\n  ✅ New session stored {len(new_turns)} turns, informed by past context.")

    # ── Summary ───────────────────────────────────────────────
    banner("Done")
    print("  This sample demonstrated:")
    print("    • Storing multi-turn conversations with add_cosmos")
    print("    • Retrieving a full thread with get_thread")
    print("    • Searching across sessions with search_cosmos")
    print("    • Using recalled context to inform new sessions")
    print()


if __name__ == "__main__":
    main()
