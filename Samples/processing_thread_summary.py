"""Demonstrate per-thread summarisation (incremental updates included).

``CosmosMemoryClient.generate_thread_summary(...)`` runs the same in-process
ProcessingPipeline that the change-feed Azure Function uses — no Function
deployment is required for this sample.

Required env vars (.env supported):

    COSMOS_DB_ENDPOINT, AI_FOUNDRY_ENDPOINT, AI_FOUNDRY_EMBEDDING_DEPLOYMENT_NAME, AI_FOUNDRY_CHAT_DEPLOYMENT_NAME
    COSMOS_DB_KEY (optional fallback)
"""

from __future__ import annotations

import json
import os
import sys
import uuid

from dotenv import load_dotenv

from agent_memory_toolkit import CosmosMemoryClient

load_dotenv()


def _banner(title: str) -> None:
    print(f"\n{'=' * 60}\n  {title}\n{'=' * 60}")


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
    print(f"User ID:   {user_id}")
    print(f"Thread ID: {thread_id}")

    _banner("STEP 1 – initial conversation")
    initial = [
        ("user", "I'm planning a trip to Japan next spring. Any suggestions?"),
        ("agent", "Spring is a wonderful time to visit Japan! Cherry blossom season runs late March to early April."),
        ("user", "I'd love to see Kyoto and Tokyo. How long should I stay?"),
        ("agent", "10–14 days lets you spend ~5 days in each city plus some day trips."),
    ]
    for role, content in initial:
        mem.add_cosmos(user_id=user_id, role=role, content=content, thread_id=thread_id)
        print(f"  [{role:>5}] {content[:80]}")

    _banner("STEP 2 – generate first summary")
    doc1 = mem.generate_thread_summary(user_id=user_id, thread_id=thread_id)
    print(f"summary id : {doc1['id']}")
    print(f"source_count: {doc1.get('metadata', {}).get('source_count')}")
    print("structured summary:")
    print(json.dumps(doc1.get("metadata", {}).get("structured_summary"), indent=2))

    _banner("STEP 3 – more turns (incremental update path)")
    follow_up = [
        ("user", "What about food? I'm a vegetarian."),
        ("agent", "Japan has wonderful vegetarian options — try shojin-ryori (Buddhist temple cuisine), tofu specialties, and tempura vegetables."),
        ("user", "Are there any vegetarian restaurants you'd recommend in Kyoto?"),
        ("agent", "Shigetsu inside Tenryu-ji temple is famous for its shojin-ryori meals."),
    ]
    for role, content in follow_up:
        mem.add_cosmos(user_id=user_id, role=role, content=content, thread_id=thread_id)
        print(f"  [{role:>5}] {content[:80]}")

    doc2 = mem.generate_thread_summary(user_id=user_id, thread_id=thread_id)
    print(f"\nupdated summary id     : {doc2['id']}")
    print(f"updated source_count   : {doc2.get('metadata', {}).get('source_count')}")
    print(f"incremental_update flag: {doc2.get('metadata', {}).get('incremental_update')}")

    _banner("STEP 4 – read summaries from Cosmos")
    for s in mem.get_memories(user_id=user_id, thread_id=thread_id, memory_types=["summary"]):
        print(f"  • {s['id']} :: {s['content'][:120]}…")

    print("\nDone.")


if __name__ == "__main__":
    main()
