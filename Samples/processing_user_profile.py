"""Cross-thread user profile generation with Agent Memory Toolkit.

Demonstrates how to build and incrementally update a user profile that
synthesises interests across multiple conversation threads.

Prerequisites
-------------
* Azure Cosmos DB account with a ``memories`` container.
* Azure Durable Functions (ADF) endpoint for processing.
* Environment variables:
    COSMOS_DB_ENDPOINT  – Cosmos DB account endpoint URL
    AI_FOUNDRY_ENDPOINT – Azure OpenAI / AI Foundry endpoint URL
    ADF_ENDPOINT        – Azure Durable Functions base URL
    ADF_KEY             – (optional) function-level auth key
"""

import json
import os

from dotenv import load_dotenv
load_dotenv()
import uuid

from agent_memory_toolkit import CosmosMemoryClient


def _print_section(title: str) -> None:
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print("=" * 60)


def main() -> None:
    # ── Configuration from environment variables ─────────────────
    cosmos_endpoint = os.environ["COSMOS_DB_ENDPOINT"]
    ai_foundry_endpoint = os.environ["AI_FOUNDRY_ENDPOINT"]
    adf_endpoint = os.environ["ADF_ENDPOINT"]
    adf_key = os.environ.get("ADF_KEY")

    mem = CosmosMemoryClient(
        cosmos_endpoint=cosmos_endpoint,
        ai_foundry_endpoint=ai_foundry_endpoint,
        adf_endpoint=adf_endpoint,
        adf_key=adf_key,
    )
    mem.connect_cosmos()

    user_id = f"demo-user-{uuid.uuid4().hex[:8]}"

    # ── Thread 1: cooking conversation ───────────────────────────
    _print_section("Thread 1 – Cooking")
    thread_1 = str(uuid.uuid4())
    mem.add_cosmos(
        user_id=user_id,
        role="user",
        content="What's a good pasta recipe for a weeknight dinner?",
        thread_id=thread_1,
    )
    mem.add_cosmos(
        user_id=user_id,
        role="agent",
        content=(
            "Try a classic spaghetti aglio e olio — garlic, olive oil, "
            "chilli flakes, and parsley. Ready in 20 minutes!"
        ),
        thread_id=thread_1,
    )
    mem.add_cosmos(
        user_id=user_id,
        role="user",
        content="I love Italian food. Any dessert suggestions?",
        thread_id=thread_1,
    )
    mem.add_cosmos(
        user_id=user_id,
        role="agent",
        content="Panna cotta is simple and delicious — just cream, sugar, and vanilla.",
        thread_id=thread_1,
    )
    print(f"Thread 1 ID: {thread_1}")

    # ── Thread 2: travel conversation ────────────────────────────
    _print_section("Thread 2 – Travel")
    thread_2 = str(uuid.uuid4())
    mem.add_cosmos(
        user_id=user_id,
        role="user",
        content="Plan a trip to Tokyo for someone who loves street food.",
        thread_id=thread_2,
    )
    mem.add_cosmos(
        user_id=user_id,
        role="agent",
        content=(
            "Here's a 5-day Tokyo itinerary focused on street food: "
            "Tsukiji Outer Market, Takeshita Street crêpes, Yakitori Alley in Yurakucho…"
        ),
        thread_id=thread_2,
    )
    mem.add_cosmos(
        user_id=user_id,
        role="user",
        content="I'd also like to visit some temples and gardens.",
        thread_id=thread_2,
    )
    mem.add_cosmos(
        user_id=user_id,
        role="agent",
        content="Senso-ji and Meiji Shrine are must-sees. Shinjuku Gyoen is a beautiful garden.",
        thread_id=thread_2,
    )
    print(f"Thread 2 ID: {thread_2}")

    # ── Generate cross-thread user profile ───────────────────────
    _print_section("Generating cross-thread user profile (threads 1 & 2)")
    result = mem.generate_user_summary(
        user_id=user_id,
        thread_ids=[thread_1, thread_2],
    )
    print("Processing result:")
    print(json.dumps(result, indent=2, default=str))

    # ── Retrieve the generated profile ───────────────────────────
    _print_section("Retrieving user profile")
    profiles = mem.get_user_summary(user_id=user_id)
    for i, profile in enumerate(profiles):
        print(f"\n--- Profile {i + 1} ---")
        print(json.dumps(profile, indent=2, default=str))

    # ── Thread 3: programming conversation ───────────────────────
    _print_section("Thread 3 – Programming (new interest)")
    thread_3 = str(uuid.uuid4())
    mem.add_cosmos(
        user_id=user_id,
        role="user",
        content="I need help debugging a Python async race condition.",
        thread_id=thread_3,
    )
    mem.add_cosmos(
        user_id=user_id,
        role="agent",
        content=(
            "Race conditions in asyncio usually stem from shared mutable state. "
            "Use asyncio.Lock or redesign with message passing."
        ),
        thread_id=thread_3,
    )
    print(f"Thread 3 ID: {thread_3}")

    # ── Incrementally update the profile with the new thread ─────
    _print_section("Updating user profile incrementally (threads 1, 2 & 3)")
    result_updated = mem.generate_user_summary(
        user_id=user_id,
        thread_ids=[thread_1, thread_2, thread_3],
    )
    print("Updated processing result:")
    print(json.dumps(result_updated, indent=2, default=str))

    # ── Retrieve the updated profile ─────────────────────────────
    _print_section("Retrieving updated user profile")
    updated_profiles = mem.get_user_summary(user_id=user_id)
    for i, profile in enumerate(updated_profiles):
        print(f"\n--- Profile {i + 1} ---")
        print(json.dumps(profile, indent=2, default=str))

    print(f"\nDone. User '{user_id}' profile built from 3 threads.")


if __name__ == "__main__":
    main()
