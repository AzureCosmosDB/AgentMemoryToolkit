"""Cross-thread user profile generation.

Runs ``CosmosMemoryClient.generate_user_summary(...)`` which synthesises
a single user profile from per-thread summaries and extracted facts —
again, all in-process via the shared ProcessingPipeline.

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
    print(f"User ID: {user_id}")

    # ---------- thread 1: cooking ----------
    _banner("Thread 1 – cooking")
    t1 = str(uuid.uuid4())
    for role, content in [
        ("user", "What's a good pasta recipe for a weeknight dinner?"),
        ("agent", "Try aglio e olio — pasta, garlic, olive oil, chilli flakes, parsley."),
        ("user", "Sounds great. I love simple Italian food, especially fresh ingredients."),
        ("agent", "Italian cuisine emphasises quality ingredients prepared simply."),
    ]:
        mem.add_cosmos(user_id=user_id, role=role, content=content, thread_id=t1)

    mem.extract_memories(user_id=user_id, thread_id=t1)
    mem.generate_thread_summary(user_id=user_id, thread_id=t1)

    # ---------- thread 2: travel ----------
    _banner("Thread 2 – travel")
    t2 = str(uuid.uuid4())
    for role, content in [
        ("user", "I want to visit Italy next year — Rome, Florence, Tuscany."),
        ("agent", "Great itinerary! Tuscany is amazing in autumn — wine harvest season."),
        ("user", "Perfect — I love wine, especially Chianti and Brunello."),
        ("agent", "Brunello di Montalcino producers offer wonderful cellar tours."),
    ]:
        mem.add_cosmos(user_id=user_id, role=role, content=content, thread_id=t2)

    mem.extract_memories(user_id=user_id, thread_id=t2)
    mem.generate_thread_summary(user_id=user_id, thread_id=t2)

    # ---------- thread 3: technology ----------
    _banner("Thread 3 – technology")
    t3 = str(uuid.uuid4())
    for role, content in [
        ("user", "What's the best way to deploy a Python FastAPI app on Azure?"),
        ("agent", "Azure Container Apps is a great fit for FastAPI."),
        ("user", "Cool. I'm a Python engineer building AI tooling."),
        ("agent", "Azure has excellent AI services — AI Foundry, AI Search, Cosmos DB for vectors."),
    ]:
        mem.add_cosmos(user_id=user_id, role=role, content=content, thread_id=t3)

    mem.extract_memories(user_id=user_id, thread_id=t3)
    mem.generate_thread_summary(user_id=user_id, thread_id=t3)

    # ---------- generate user summary across all threads ----------
    _banner("User profile — synthesis across all 3 threads")
    doc = mem.generate_user_summary(user_id=user_id, thread_ids=[t1, t2, t3])
    print(f"id           : {doc['id']}")
    print(f"source_count : {doc.get('metadata', {}).get('source_count')}")
    print("structured profile:")
    print(json.dumps(doc.get("metadata", {}).get("structured_summary"), indent=2))

    _banner("Read profile back from Cosmos")
    summary = mem.get_user_summary(user_id)
    if summary:
        print(f"  • {summary['content'][:200]}…")
    else:
        print("  (no summary stored yet)")

    print("\nDone.")


if __name__ == "__main__":
    main()
