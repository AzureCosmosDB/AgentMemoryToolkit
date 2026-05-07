"""Demonstrate fact / procedural / episodic memory extraction.

The Agent Memory Toolkit runs the extraction pipeline **in-process** via
``CosmosMemoryClient.extract_memories(...)``. The same pipeline also runs
inside the Azure Functions change-feed trigger, so this script exercises
identical code without needing a deployed Function.

Required environment variables (.env supported via python-dotenv):

    COSMOS_DB_ENDPOINT   – Cosmos DB account URL
    COSMOS_DB_KEY           – (optional) account key fallback while
                           Cosmos control-plane RBAC is in private preview
    AI_FOUNDRY_ENDPOINT  – Azure AI Foundry / Azure OpenAI endpoint
    AI_FOUNDRY_EMBEDDING_DEPLOYMENT_NAME      – embedding deployment name
    AI_FOUNDRY_CHAT_DEPLOYMENT_NAME            – chat completion deployment name
"""

from __future__ import annotations

import json
import os
import sys
import uuid

from dotenv import load_dotenv

from agent_memory_toolkit import CosmosMemoryClient

load_dotenv()


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
    print("Connected to Cosmos DB.\n")

    user_id = f"demo-user-{uuid.uuid4().hex[:6]}"
    thread_id = str(uuid.uuid4())
    print(f"User ID:   {user_id}")
    print(f"Thread ID: {thread_id}\n")

    conversations = [
        ("user", "I live in Seattle and work at Microsoft as a software engineer."),
        ("agent", "Got it! You're based in Seattle working at Microsoft as a software engineer."),
        ("user", "My favourite programming language is Python and I've been using it for 8 years."),
        ("agent", "8 years of Python experience is impressive!"),
        ("user", "I'm currently working on a project involving large language models and RAG."),
        ("agent", "That's a great area! LLMs combined with RAG can unlock powerful applications."),
        ("user", "Last spring I went hiking on Mount Rainier with my dog — best trip ever."),
        ("user", "When debugging an LLM, always check the prompt first then the response format."),
    ]

    print("Adding conversation turns…")
    for role, content in conversations:
        mem.add_cosmos(user_id=user_id, role=role, content=content, thread_id=thread_id)
        print(f"  [{role:>5}] {content[:80]}")
    print()

    print("Running extraction pipeline (facts / procedural / episodic)…")
    stats = mem.extract_memories(user_id=user_id, thread_id=thread_id)
    print(f"Extraction stats: {json.dumps(stats, indent=2)}\n")

    for kind in ("fact", "procedural", "episodic"):
        items = mem.get_memories(user_id=user_id, memory_types=[kind])
        print(f"{kind.upper()}S ({len(items)}):")
        for it in items:
            tags = ", ".join(it.get("tags") or [])
            sal = it.get("salience")
            print(f"  • {it['content'][:90]}  [salience={sal} tags={tags}]")
        print()

    print("Semantic search examples:")
    for q in [
        "where does the user work",
        "what programming languages does the user know",
        "what outdoor activities does the user enjoy",
    ]:
        print(f'  query: "{q}"')
        for r in mem.search_cosmos(search_terms=q, user_id=user_id, top_k=3):
            print(f"    → [{r.get('type')}] {r['content'][:80]}")
        print()

    print("Done.")


if __name__ == "__main__":
    main()
