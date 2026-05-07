"""RAG pipeline augmented with user-specific memory context.

This sample demonstrates how to combine Retrieval-Augmented Generation
(RAG) with the Agent Memory Toolkit so that responses are personalised
based on previously stored user facts and conversation summaries.

Flow
----
1. Seed user facts and conversation history into Cosmos DB.
2. A new user query arrives.
3. Search memory for relevant context (``search_cosmos``).
4. Retrieve the user profile summary (``get_user_summary``).
5. Build an augmented prompt that merges the RAG document context with
   the memory context.
6. Show how the memory context personalises the final answer.

Prerequisites
-------------
* Azure Cosmos DB (NoSQL) account with vector-search enabled.
* Azure AI Foundry endpoint for embeddings.
* Environment variables:
    - ``COSMOS_DB_ENDPOINT``
    - ``AI_FOUNDRY_ENDPOINT``
"""

from __future__ import annotations

import os

from dotenv import load_dotenv
load_dotenv()
import textwrap
import uuid
from typing import Any

from agent_memory_toolkit import CosmosMemoryClient

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

SEPARATOR = "=" * 60


def _print_section(title: str) -> None:
    print(f"\n{SEPARATOR}")
    print(f"  {title}")
    print(SEPARATOR)


def _format_memories(memories: list[dict[str, Any]]) -> str:
    """Turn a list of memory dicts into a readable block for prompt injection."""
    if not memories:
        return "(no relevant memories found)"
    lines: list[str] = []
    for m in memories:
        role = m.get("role", "unknown")
        content = m.get("content", "")
        mtype = m.get("memory_type", "")
        lines.append(f"[{mtype}|{role}] {content}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Simulated RAG retrieval (replace with your real retriever)
# ---------------------------------------------------------------------------

_SIMULATED_DOCUMENTS = [
    {
        "title": "Getting Started with Web Frameworks",
        "text": (
            "Popular web frameworks include Django and Flask for Python, "
            "Express for Node.js, and Spring Boot for Java. Each has "
            "trade-offs in performance, ecosystem size, and learning curve."
        ),
    },
    {
        "title": "Choosing a Database for Your Project",
        "text": (
            "Relational databases like PostgreSQL and MySQL are great for "
            "structured data. NoSQL options such as Azure Cosmos DB and "
            "MongoDB shine for flexible schemas and global distribution."
        ),
    },
]


def simulated_rag_retrieve(query: str) -> list[dict[str, Any]]:
    """Return hard-coded documents as if they came from a vector store."""
    return _SIMULATED_DOCUMENTS


# ---------------------------------------------------------------------------
# Core demo
# ---------------------------------------------------------------------------


def run_demo() -> None:
    # ---- configuration from environment ----
    cosmos_endpoint = os.environ["COSMOS_DB_ENDPOINT"]
    ai_foundry_endpoint = os.environ["AI_FOUNDRY_ENDPOINT"]

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

    user_id = "demo-user-rag"
    thread_id = f"rag-session-{uuid.uuid4().hex[:8]}"

    # ------------------------------------------------------------------
    # Step 1 — Seed user facts and conversation history
    # ------------------------------------------------------------------
    _print_section("Step 1: Seeding user facts & conversation history")

    facts = [
        "User prefers Python over Java for backend development",
        "User is building a SaaS product on Azure",
        "User has experience with Flask but wants to try FastAPI",
        "User prefers NoSQL databases for their current project",
    ]
    for fact in facts:
        mem.add_cosmos(
            user_id=user_id,
            role="system",
            content=fact,
            memory_type="fact",
            thread_id=thread_id,
        )
        print(f"  ✓ Stored fact: {fact}")

    # Add a couple of prior conversation turns for richer context
    conversation = [
        ("user", "I'm starting a new microservice and need a web framework."),
        ("agent", "Based on your Python preference, I'd suggest FastAPI or Flask."),
        ("user", "I also need a database — something flexible for evolving schemas."),
        ("agent", "Azure Cosmos DB with its NoSQL API would be a great fit."),
    ]
    for role, content in conversation:
        mem.add_cosmos(
            user_id=user_id,
            role=role,
            content=content,
            memory_type="turn",
            thread_id=thread_id,
        )
    print(f"  ✓ Stored {len(conversation)} conversation turns")

    # ------------------------------------------------------------------
    # Step 2 — New query arrives
    # ------------------------------------------------------------------
    _print_section("Step 2: New user query")

    user_query = "What web framework and database should I use for my new project?"
    print(f"  User asks: \"{user_query}\"")

    # ------------------------------------------------------------------
    # Step 3 — Search memory for relevant context
    # ------------------------------------------------------------------
    _print_section("Step 3: Searching memory for relevant context")

    memory_results = mem.search_cosmos(
        search_terms="web framework database preference technology stack",
        user_id=user_id,
        top_k=5,
    )
    print(f"  Found {len(memory_results)} relevant memories:")
    for m in memory_results:
        print(f"    • [{m.get('memory_type')}] {m.get('content', '')[:80]}")

    # ------------------------------------------------------------------
    # Step 3b — Retrieve user summary (if one exists)
    # ------------------------------------------------------------------
    _print_section("Step 3b: Retrieving user summary")

    summaries = mem.get_user_summary(user_id=user_id)
    if summaries:
        print("  Found user summary:")
        print(f"    • {summaries.get('content', '')[:120]}")
    else:
        print("  No summary available yet (generate one via generate_thread_summary).")

    # ------------------------------------------------------------------
    # Step 4 — Build augmented prompt
    # ------------------------------------------------------------------
    _print_section("Step 4: Building augmented prompt")

    # 4a. RAG retrieval
    rag_docs = simulated_rag_retrieve(user_query)
    rag_context = "\n\n".join(
        f"### {doc['title']}\n{doc['text']}" for doc in rag_docs
    )

    # 4b. Memory context
    memory_context = _format_memories(memory_results)

    # 4c. User summary context
    summary_context = (
        summaries.get("content", "") if summaries else "No summary available."
    )

    augmented_prompt = textwrap.dedent(f"""\
        You are a helpful developer assistant. Use the following context to
        answer the user's question with personalised recommendations.

        === User Profile Summary ===
        {summary_context}

        === Relevant Memories (facts & prior conversation) ===
        {memory_context}

        === Retrieved Documents (RAG) ===
        {rag_context}

        === User Question ===
        {user_query}

        Provide a concise, personalised recommendation.
    """)

    print(augmented_prompt)

    # ------------------------------------------------------------------
    # Step 5 — Show how memory personalises the response
    # ------------------------------------------------------------------
    _print_section("Step 5: Personalised response (simulated)")

    simulated_response = textwrap.dedent("""\
        Based on your preferences and history:

        **Web Framework → FastAPI**
        You already know Flask and expressed interest in trying FastAPI.
        It offers async support, automatic OpenAPI docs, and great
        performance — ideal for the Azure-hosted SaaS microservice
        you're building.

        **Database → Azure Cosmos DB (NoSQL API)**
        Given your preference for NoSQL and your Azure-based stack,
        Cosmos DB provides flexible schemas, global distribution,
        and seamless Azure integration.

        This recommendation is personalised using your stored preferences
        (Python over Java, NoSQL preference) and your prior conversation
        about framework and database choices.
    """)

    print(simulated_response)

    # ------------------------------------------------------------------
    # Cleanup — remove seeded data so the sample is re-runnable
    # ------------------------------------------------------------------
    _print_section("Cleanup")

    stored = mem.get_memories(user_id=user_id, thread_id=thread_id)
    for item in stored:
        mem.delete_cosmos(
            memory_id=item["id"],
            thread_id=item.get("thread_id", thread_id),
            user_id=user_id,
        )
    print(f"  Deleted {len(stored)} seeded memories.")
    print("\nDone ✓")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    run_demo()
