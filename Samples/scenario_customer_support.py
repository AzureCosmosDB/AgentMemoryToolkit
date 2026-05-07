"""Customer-support scenario — build a unified customer profile across tickets.

Walks through three support tickets for the same customer; after each
ticket we extract memories and update the user profile, so the agent can
greet returning customers with personalised context.

In-process pipeline — no Azure Function deployment required.

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


def _ticket_id() -> str:
    return f"ticket-{uuid.uuid4().hex[:8]}"


def _banner(title: str) -> None:
    print(f"\n{'=' * 60}\n  {title}\n{'=' * 60}\n")


def _add_dialogue(mem: CosmosMemoryClient, user_id: str, ticket: str, dialogue: list[tuple[str, str]]) -> None:
    for role, content in dialogue:
        mem.add_cosmos(user_id=user_id, role=role, content=content, thread_id=ticket)
        print(f"  [{role:>5}] {content[:90]}")


def _process_ticket(mem: CosmosMemoryClient, user_id: str, ticket: str) -> None:
    print("\n  → extracting memories…")
    print("  ", mem.extract_memories(user_id=user_id, thread_id=ticket))
    print("  → summarising thread…")
    mem.generate_thread_summary(user_id=user_id, thread_id=ticket)


def _run_ticket_1(mem: CosmosMemoryClient, user_id: str, ticket: str) -> None:
    _banner(f"Ticket 1 ({ticket}) — initial complaint")
    _add_dialogue(mem, user_id, ticket, [
        ("user", "Hi, I'm having issues with my Surface Pro 9. The battery drains in 3 hours."),
        ("agent", "I'm sorry to hear that. Could you share when you bought the device?"),
        ("user", "I bought it in March 2024 from the Microsoft Store online."),
        ("agent", "Thanks. I'll check warranty status. Could you also confirm your email?"),
        ("user", "It's alex.chen@example.com. The order number was MS-78234."),
        ("agent", "Verified — the device is under warranty. I've initiated a battery diagnostic. Please run the Surface app diagnostic."),
        ("user", "OK, will do. By the way, I work as a software engineer so I rely on this device daily."),
    ])
    _process_ticket(mem, user_id, ticket)


def _run_ticket_2(mem: CosmosMemoryClient, user_id: str, ticket: str) -> None:
    _banner(f"Ticket 2 ({ticket}) — accessory question")
    _add_dialogue(mem, user_id, ticket, [
        ("user", "Hello again — Alex from the previous battery ticket. The Surface app says battery is degraded."),
        ("agent", "Welcome back, Alex! Yes, that confirms the diagnostic. We'll ship a replacement battery service unit."),
        ("user", "Great. While we're talking, can you recommend a USB-C dock that works with the Surface Pro 9?"),
        ("agent", "The Surface Dock 2 or the Anker 778 are excellent choices for multi-monitor setups."),
        ("user", "Multi-monitor is exactly what I need — I run 3 displays for development work."),
    ])
    _process_ticket(mem, user_id, ticket)


def _run_ticket_3_greeting(mem: CosmosMemoryClient, user_id: str, ticket: str) -> None:
    _banner(f"Ticket 3 ({ticket}) — personalised greeting using profile")

    # Build context from the user profile + extracted facts to personalise the greeting.
    profile = mem.get_user_summary(user_id)
    facts = mem.get_memories(user_id=user_id, memory_types=["fact"])
    print("Profile content (truncated):")
    if profile:
        print(f"  {profile['content'][:200]}…")
    print("\nKnown facts:")
    for f in facts[:8]:
        print(f"  • {f['content']}")

    _add_dialogue(mem, user_id, ticket, [
        ("user", "Hi, I have a quick question about my Microsoft 365 subscription."),
        # An agent built on top of the toolkit could now look up the profile + facts
        # before composing its greeting (Alex / engineer / Surface Pro 9 owner).
        ("agent", "Welcome back, Alex! Happy to help with your Microsoft 365 subscription. What's on your mind?"),
    ])


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

    user_id = f"customer-{uuid.uuid4().hex[:6]}"
    print(f"Customer ID: {user_id}")

    t1, t2, t3 = _ticket_id(), _ticket_id(), _ticket_id()

    _run_ticket_1(mem, user_id, t1)
    _run_ticket_2(mem, user_id, t2)

    _banner("Synthesise unified customer profile across the first 2 tickets")
    profile_doc = mem.generate_user_summary(user_id=user_id, thread_ids=[t1, t2])
    print(profile_doc["content"][:400], "…")

    _run_ticket_3_greeting(mem, user_id, t3)

    print("\nDone.")


if __name__ == "__main__":
    main()
