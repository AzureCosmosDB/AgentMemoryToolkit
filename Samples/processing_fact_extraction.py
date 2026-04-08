"""
Fact Extraction Workflow – Agent Memory Toolkit

Demonstrates the end-to-end fact extraction pipeline:
  1. Connect to Cosmos DB
  2. Add conversation turns containing factual statements
  3. Extract facts via Azure Durable Functions (ADF)
  4. Read the extracted facts back from Cosmos DB
  5. Search facts semantically using vector search

Requirements:
  - Azure Cosmos DB configured with the Agent Memory Toolkit schema
  - Azure Durable Functions (ADF) deployed or running locally
  - Azure AI Foundry endpoint for embeddings

Environment variables (set in shell or .env file):
  COSMOS_DB_ENDPOINT   – Cosmos DB account endpoint
  COSMOS_DB_DATABASE   – (optional) database name override
  COSMOS_DB_CONTAINER  – (optional) container name override
  AI_FOUNDRY_ENDPOINT  – Azure AI Foundry endpoint for embedding models
  EMBEDDING_MODEL      – (optional) embedding model name, default text-embedding-3-large
  ADF_ENDPOINT         – Azure Durable Functions endpoint (or http://localhost:7071/api)
  ADF_KEY              – (optional) function-level auth key for ADF
"""

import json
import os
import uuid

from dotenv import load_dotenv

from agent_memory_toolkit import CosmosMemoryClient


def main() -> None:
    # ── Load configuration ────────────────────────────────────────────
    load_dotenv()

    cosmos_endpoint = os.environ["COSMOS_DB_ENDPOINT"]
    ai_foundry_endpoint = os.environ["AI_FOUNDRY_ENDPOINT"]
    adf_endpoint = os.environ.get("ADF_ENDPOINT", "http://localhost:7071/api")
    adf_key = os.environ.get("ADF_KEY", "")

    memory = CosmosMemoryClient(
        cosmos_endpoint=cosmos_endpoint,
        cosmos_database=os.getenv("COSMOS_DB_DATABASE"),
        cosmos_container=os.getenv("COSMOS_DB_CONTAINER"),
        ai_foundry_endpoint=ai_foundry_endpoint,
        embedding_model=os.getenv("EMBEDDING_MODEL", "text-embedding-3-large"),
        adf_endpoint=adf_endpoint,
        adf_key=adf_key,
    )

    # ── 1. Connect to Cosmos DB ───────────────────────────────────────
    print("Connecting to Cosmos DB …")
    memory.connect_cosmos()
    print("Connected.\n")

    # ── 2. Add conversation turns with factual content ────────────────
    user_id = "demo-user"
    thread_id = str(uuid.uuid4())
    print(f"Thread ID: {thread_id}\n")

    conversations = [
        ("user", "I live in Seattle and work at Microsoft as a software engineer."),
        ("agent", "Got it! You're based in Seattle working at Microsoft as a software engineer."),
        ("user", "My favorite programming language is Python and I've been using it for 8 years."),
        ("agent", "Nice — 8 years of Python experience is impressive!"),
        (
            "user",
            "I'm currently working on a project involving large language models and RAG.",
        ),
        (
            "agent",
            "That's a great area! LLMs combined with RAG can unlock powerful applications.",
        ),
    ]

    print("Adding conversation turns …")
    for role, content in conversations:
        memory.add_cosmos(
            user_id=user_id,
            role=role,
            content=content,
            thread_id=thread_id,
        )
        print(f"  [{role:>5}] {content[:80]}")
    print()

    # ── 3. Extract facts via Azure Durable Functions ──────────────────
    print("Extracting facts (calling ADF) …")
    result = memory.extract_facts(user_id=user_id, thread_id=thread_id)
    print(f"Extraction result:\n{json.dumps(result, indent=2)}\n")

    # ── 4. Read extracted facts from Cosmos DB ────────────────────────
    print("Reading extracted facts from Cosmos DB …")
    facts = memory.get_memories(user_id=user_id, memory_type="fact")
    print(f"Found {len(facts)} fact(s):\n")
    for fact in facts:
        print(f"  • [{fact['id'][:8]}…] {fact['content']}")
    print()

    # ── 5. Semantic search over extracted facts ───────────────────────
    queries = [
        "where does the user work",
        "programming languages the user knows",
        "what project is the user working on",
    ]

    for query in queries:
        print(f'Searching: "{query}"')
        results = memory.search_cosmos(
            search_terms=query,
            user_id=user_id,
            memory_type="fact",
            top_k=3,
        )
        if results:
            for r in results:
                print(f"  → {r['content'][:100]}")
        else:
            print("  (no results)")
        print()

    print("Done.")


if __name__ == "__main__":
    main()
