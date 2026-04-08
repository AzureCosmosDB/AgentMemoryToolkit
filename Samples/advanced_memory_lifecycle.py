"""
Advanced Memory Lifecycle — Agent Memory Toolkit

Demonstrates the full memory lifecycle: create → use → summarize → archive.

After processing, raw conversation turns are deleted while derived memories
(summaries and extracted facts) are kept — producing a compact, long-term
memory store.

Requirements
------------
* Azure Cosmos DB (with the Agent Memory Toolkit schema)
* Azure Durable Functions (ADF) for summarization & fact extraction
* Azure AI Foundry endpoint for embeddings

Environment variables:
    COSMOS_DB_ENDPOINT   – Cosmos DB account URL
    AI_FOUNDRY_ENDPOINT  – Azure AI Foundry endpoint
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

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def print_header(step: int, title: str) -> None:
    """Print a formatted section header."""
    print(f"\n{'=' * 60}")
    print(f"  STEP {step} – {title}")
    print(f"{'=' * 60}")


def print_memories(memories: list, label: str = "Memories") -> None:
    """Print a list of memory dicts in a readable format."""
    print(f"\n  {label} ({len(memories)} item(s)):")
    if not memories:
        print("    (none)")
        return
    for i, m in enumerate(memories, 1):
        mem_type = m.get("memory_type", "n/a")
        content = m.get("content", "")
        preview = (content[:100] + "…") if len(content) > 100 else content
        print(f"    {i}. [{mem_type}] {preview}")


# ---------------------------------------------------------------------------
# Main lifecycle
# ---------------------------------------------------------------------------

def main() -> None:
    # ── Validate configuration ────────────────────────────────────────
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
    print("✅ Connected to Cosmos DB")

    user_id = "lifecycle-user"
    thread_id = str(uuid.uuid4())
    print(f"   User ID   : {user_id}")
    print(f"   Thread ID : {thread_id}")

    # ------------------------------------------------------------------
    # 1. Create conversation turns
    # ------------------------------------------------------------------
    print_header(1, "Create conversation turns")

    turns = [
        ("user",  "I'm researching machine learning frameworks. What do you recommend?"),
        ("agent", "It depends on your use case. PyTorch is great for research, "
                  "while TensorFlow excels in production deployments. JAX is "
                  "gaining traction for high-performance numerical computing."),
        ("user",  "I mostly do NLP work — transformer fine-tuning and RAG pipelines."),
        ("agent", "For NLP and transformer work, PyTorch with Hugging Face "
                  "Transformers is the most popular stack. You can pair it "
                  "with LangChain or Semantic Kernel for RAG pipelines."),
        ("user",  "I also need something that works well on Azure. "
                  "My team uses Azure Machine Learning."),
        ("agent", "Azure ML has first-class support for PyTorch, and "
                  "Hugging Face models can be deployed directly via managed "
                  "endpoints. You can also leverage Azure AI Foundry for "
                  "model cataloging and evaluation."),
    ]

    for role, content in turns:
        mem.add_cosmos(
            user_id=user_id,
            role=role,
            content=content,
            thread_id=thread_id,
        )
        tag = "👤" if role == "user" else "🤖"
        print(f"  {tag} {content[:72]}…" if len(content) > 72 else f"  {tag} {content}")

    print(f"\n  ✅ Added {len(turns)} turns to thread")

    # ------------------------------------------------------------------
    # 2. Retrieve the thread
    # ------------------------------------------------------------------
    print_header(2, "Retrieve the thread")

    thread = mem.get_thread(thread_id=thread_id)
    print(f"  Retrieved {len(thread)} messages from thread {thread_id[:8]}…")
    for msg in thread:
        role = msg.get("role", "?")
        content = msg.get("content", "")
        preview = (content[:80] + "…") if len(content) > 80 else content
        print(f"    [{role}] {preview}")

    # ------------------------------------------------------------------
    # 3. Generate a thread summary
    # ------------------------------------------------------------------
    print_header(3, "Generate thread summary")

    summary_result = mem.generate_thread_summary(
        user_id=user_id, thread_id=thread_id,
    )

    print(f"  Runtime status : {summary_result.get('runtimeStatus')}")
    print(f"  Instance ID    : {summary_result.get('instance_id', 'N/A')}")
    if summary_result.get("output"):
        print(f"  Output preview : {str(summary_result['output'])[:200]}")

    # Verify the summary was persisted
    summaries = mem.get_memories(
        user_id=user_id, thread_id=thread_id, memory_type="summary",
    )
    print_memories(summaries, label="Summaries")

    # ------------------------------------------------------------------
    # 4. Extract facts from the conversation
    # ------------------------------------------------------------------
    print_header(4, "Extract facts")

    facts_result = mem.extract_facts(
        user_id=user_id, thread_id=thread_id,
    )

    print(f"  Runtime status : {facts_result.get('runtimeStatus')}")
    print(f"  Instance ID    : {facts_result.get('instance_id', 'N/A')}")
    if facts_result.get("output"):
        print(f"  Output preview : {str(facts_result['output'])[:200]}")

    # Verify the extracted facts
    facts = mem.get_memories(
        user_id=user_id, thread_id=thread_id, memory_type="fact",
    )
    print_memories(facts, label="Extracted facts")

    # ------------------------------------------------------------------
    # 5. Search for the processed memories
    # ------------------------------------------------------------------
    print_header(5, "Search processed memories")

    queries = ["machine learning frameworks", "NLP transformers", "Azure deployment"]
    for query in queries:
        results = mem.search_cosmos(search_terms=query, user_id=user_id)
        print(f"\n  🔍 '{query}' → {len(results)} result(s)")
        for r in results[:3]:
            content = getattr(r, "content", None) or r.get("content", str(r))
            mem_type = getattr(r, "memory_type", None) or r.get("memory_type", "n/a")
            preview = (content[:90] + "…") if len(content) > 90 else content
            print(f"     [{mem_type}] {preview}")

    # ------------------------------------------------------------------
    # 6. Archive — delete raw turns, keep summaries & facts
    # ------------------------------------------------------------------
    print_header(6, "Archive — delete raw turns")

    raw_turns = mem.get_memories(
        user_id=user_id, thread_id=thread_id, memory_type="turn",
    )
    print(f"  Found {len(raw_turns)} raw turn(s) to delete")

    for turn in raw_turns:
        mem.delete_cosmos(
            memory_id=turn["id"],
            thread_id=thread_id,
            user_id=user_id,
        )
    print(f"  ✅ Deleted {len(raw_turns)} raw turn(s)")

    # Confirm only derived memories remain
    remaining = mem.get_memories(user_id=user_id, thread_id=thread_id)
    print_memories(remaining, label="Remaining memories (summaries + facts)")

    # ------------------------------------------------------------------
    # Done
    # ------------------------------------------------------------------
    print(f"\n{'=' * 60}")
    print("  ✅ Memory lifecycle complete!")
    print(f"     Thread    : {thread_id}")
    print(f"     Turns     : {len(turns)} created → {len(raw_turns)} archived")
    print(f"     Summaries : {len(summaries)}")
    print(f"     Facts     : {len(facts)}")
    print(f"     Remaining : {len(remaining)} derived memory item(s)")
    print(f"{'=' * 60}\n")


if __name__ == "__main__":
    main()
