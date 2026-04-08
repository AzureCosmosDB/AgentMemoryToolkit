"""Demonstrate the thread-summarization workflow.

Requirements
------------
* Azure Cosmos DB account (with the ``ai_memory`` database / ``memories`` container)
* Azure Durable Functions (ADF) endpoint for memory processing

Set the following environment variables before running:

    COSMOS_DB_ENDPOINT   – Cosmos DB account URL
    AI_FOUNDRY_ENDPOINT  – Azure OpenAI / AI Foundry endpoint
    ADF_ENDPOINT         – Azure Durable Functions base URL
    ADF_KEY              – (optional) function-level auth key
"""

from __future__ import annotations

import os

from dotenv import load_dotenv
load_dotenv()
import sys
import uuid

from agent_memory_toolkit import CosmosMemoryClient


def main() -> None:
    # ------------------------------------------------------------------
    # 1. Initialise CosmosMemoryClient with Cosmos DB + ADF
    # ------------------------------------------------------------------
    required_vars = ["COSMOS_DB_ENDPOINT", "AI_FOUNDRY_ENDPOINT", "ADF_ENDPOINT"]
    missing = [v for v in required_vars if not os.environ.get(v)]
    if missing:
        print(f"ERROR: Missing required environment variables: {', '.join(missing)}")
        sys.exit(1)

    mem = CosmosMemoryClient(
        cosmos_endpoint=os.environ["COSMOS_DB_ENDPOINT"],
        ai_foundry_endpoint=os.environ["AI_FOUNDRY_ENDPOINT"],
        adf_endpoint=os.environ["ADF_ENDPOINT"],
        adf_key=os.environ.get("ADF_KEY"),
    )
    mem.connect_cosmos()
    print("✅ Connected to Cosmos DB\n")

    user_id = "demo-user"
    thread_id = str(uuid.uuid4())
    print(f"Thread ID: {thread_id}\n")

    # ------------------------------------------------------------------
    # 2. Add initial conversation turns
    # ------------------------------------------------------------------
    print("=" * 60)
    print("STEP 1 – Add initial conversation turns")
    print("=" * 60)

    turns = [
        ("user", "I'm planning a trip to Japan next spring. Any suggestions?"),
        ("agent", "Spring is a wonderful time to visit Japan! Cherry blossom "
                  "season typically runs from late March to mid-April. I'd "
                  "recommend visiting Tokyo, Kyoto, and Osaka."),
        ("user", "What about budget? I'm thinking around $3,000 for two weeks."),
        ("agent", "$3,000 for two weeks is doable if you stay in hostels or "
                  "budget hotels, use a Japan Rail Pass for transport, and eat "
                  "at local restaurants. Budget roughly $150/day for "
                  "accommodation, food, and activities."),
        ("user", "Great! Should I book flights now or wait?"),
        ("agent", "Booking 3-4 months ahead usually gives the best fares for "
                  "spring travel to Japan. Set up a price alert and book when "
                  "you see a fare you're comfortable with."),
    ]

    for role, content in turns:
        mem.add_cosmos(
            user_id=user_id,
            role=role,
            content=content,
            thread_id=thread_id,
        )
        print(f"  [{role}] {content[:70]}...")

    print(f"\n✅ Added {len(turns)} turns to thread\n")

    # ------------------------------------------------------------------
    # 3. Trigger summarization via Azure Durable Functions
    # ------------------------------------------------------------------
    print("=" * 60)
    print("STEP 2 – Generate thread summary (Durable Function)")
    print("=" * 60)

    result = mem.generate_thread_summary(user_id=user_id, thread_id=thread_id)

    print(f"  Runtime status : {result.get('runtimeStatus')}")
    print(f"  Instance ID    : {result.get('instance_id', 'N/A')}")
    print(f"  Custom status  : {result.get('customStatus', 'N/A')}")
    if result.get("output"):
        print(f"  Output preview : {str(result['output'])[:200]}")
    print()

    # ------------------------------------------------------------------
    # 4. Read back the summary from Cosmos DB
    # ------------------------------------------------------------------
    print("=" * 60)
    print("STEP 3 – Read summary from Cosmos DB")
    print("=" * 60)

    summaries = mem.get_memories(
        user_id=user_id,
        thread_id=thread_id,
        memory_type="summary",
    )

    if summaries:
        for i, s in enumerate(summaries, 1):
            print(f"\n  Summary #{i}:")
            print(f"  Content: {s.get('content', '')[:300]}")
            print(f"  Created: {s.get('created_at', 'N/A')}")
    else:
        print("  ⚠️  No summaries found (the function may still be processing).")
    print()

    # ------------------------------------------------------------------
    # 5. Add more turns and re-summarize (incremental update)
    # ------------------------------------------------------------------
    print("=" * 60)
    print("STEP 4 – Add more turns & update summary incrementally")
    print("=" * 60)

    extra_turns = [
        ("user", "I also want to visit Hiroshima. Is it worth a day trip?"),
        ("agent", "Absolutely! Hiroshima is about 2 hours from Osaka by "
                  "Shinkansen. Visit the Peace Memorial Park and Museum, "
                  "then take the ferry to Miyajima Island to see the "
                  "iconic floating torii gate."),
        ("user", "Perfect, I'll add that to my itinerary. Thanks!"),
        ("agent", "You're welcome! Your two-week itinerary now covers "
                  "Tokyo, Kyoto, Osaka, and a day trip to Hiroshima. "
                  "That's a fantastic route for a first visit to Japan."),
    ]

    for role, content in extra_turns:
        mem.add_cosmos(
            user_id=user_id,
            role=role,
            content=content,
            thread_id=thread_id,
        )
        print(f"  [{role}] {content[:70]}...")

    print(f"\n✅ Added {len(extra_turns)} more turns\n")

    print("  Triggering incremental re-summarization...")
    result2 = mem.generate_thread_summary(user_id=user_id, thread_id=thread_id)

    print(f"  Runtime status : {result2.get('runtimeStatus')}")
    print(f"  Instance ID    : {result2.get('instance_id', 'N/A')}")
    if result2.get("output"):
        print(f"  Output preview : {str(result2['output'])[:200]}")
    print()

    # ------------------------------------------------------------------
    # 6. Read the updated summary
    # ------------------------------------------------------------------
    print("=" * 60)
    print("STEP 5 – Read updated summary")
    print("=" * 60)

    updated_summaries = mem.get_memories(
        user_id=user_id,
        thread_id=thread_id,
        memory_type="summary",
    )

    if updated_summaries:
        for i, s in enumerate(updated_summaries, 1):
            print(f"\n  Summary #{i}:")
            print(f"  Content: {s.get('content', '')[:300]}")
            print(f"  Updated: {s.get('updated_at', s.get('created_at', 'N/A'))}")
    else:
        print("  ⚠️  No summaries found.")

    print("\n" + "=" * 60)
    print("Done! Thread summary workflow complete.")
    print("=" * 60)


if __name__ == "__main__":
    main()
