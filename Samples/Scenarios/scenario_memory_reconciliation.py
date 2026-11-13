"""Scenario: Memory reconciliation - duplicate folding and contradiction resolution.

Demonstrates how `reconcile` cleans up a user's facts in two passes:

1. Seed paraphrased facts about a user (the dedup pass retires a fact into another
   only when nothing it states would be lost; it may instead combine facts into one new
   fact).
2. Seed contradicting facts about the same subject (the contradiction pass retires
   the loser).
3. Run reconcile() and print the stats: {kept, merged, contradicted} plus the dedup
   pass's checked, compared, vetoed, combined, synthesized, combine_groups and
   combine_failed counts.
4. Show the live state - folded duplicates and the contradiction loser are hidden.
5. Show the audit trail (include_superseded=True) - retired records carry
   supersede_reason ("duplicate", "combined" or "contradict"), superseded_at, and
   superseded_by pointing at the survivor.

Requirements:
    - Azure Cosmos DB account with vector-search enabled.
    - Azure AI Foundry endpoint with chat + embeddings deployments.
    - Environment variables (.env supported via python-dotenv):
        COSMOS_DB_ENDPOINT
        COSMOS_DB_KEY                          (optional, falls back to DefaultAzureCredential)
        AI_FOUNDRY_ENDPOINT
        AI_FOUNDRY_EMBEDDING_DEPLOYMENT_NAME
        AI_FOUNDRY_CHAT_DEPLOYMENT_NAME
"""

from __future__ import annotations

import os
import sys
import uuid

from dotenv import load_dotenv

from azure.cosmos.agent_memory import CosmosMemoryClient, MemoryNotFoundError

load_dotenv()

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

DIVIDER = "-" * 60


def banner(title: str) -> None:
    """Print a section banner."""
    print(f"\n{DIVIDER}")
    print(f"  {title}")
    print(DIVIDER)


def print_facts(facts: list[dict]) -> None:
    """Pretty-print fact records (id + content)."""
    if not facts:
        print("  (none)")
        return
    for f in facts:
        print(f"  id={f['id']}  content: {f.get('content', '')}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

PARAPHRASED_FACTS = [
    "User prefers aisle seats on flights.",
    "User likes aisle seats when flying.",
    "User always picks aisle when booking flights.",
]

CONTRADICTING_FACTS = [
    "User is strictly vegetarian and avoids all meat.",
    "User loves a good ribeye steak.",
]


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
        cosmos_container=os.environ.get("COSMOS_DB_MEMORIES_CONTAINER", "memories"),
        ai_foundry_endpoint=os.environ["AI_FOUNDRY_ENDPOINT"],
        ai_foundry_api_key=os.environ.get("AI_FOUNDRY_API_KEY") or None,
        embedding_deployment_name=os.environ.get("AI_FOUNDRY_EMBEDDING_DEPLOYMENT_NAME", "text-embedding-3-large"),
        chat_deployment_name=os.environ.get("AI_FOUNDRY_CHAT_DEPLOYMENT_NAME", "gpt-4o-mini"),
        use_default_credential=True,
    )
    print("Connected to Cosmos DB.")

    unique_user_id = f"reconcile-demo-{uuid.uuid4().hex[:8]}"
    unique_thread_id = f"reconcile-demo-thread-{uuid.uuid4().hex[:8]}"
    print(f"User ID:   {unique_user_id}")
    print(f"Thread ID: {unique_thread_id}")

    try:
        banner("1. Seeding paraphrased facts (duplicates)")
        for content in PARAPHRASED_FACTS:
            mem.upsert_memory(
                user_id=unique_user_id,
                role="user",
                content=content,
                memory_type="fact",
                thread_id=unique_thread_id,
                salience=0.7,
            )
            print(f"  + {content}")

        banner("2. Seeding contradicting facts")
        for content in CONTRADICTING_FACTS:
            mem.upsert_memory(
                user_id=unique_user_id,
                role="user",
                content=content,
                memory_type="fact",
                thread_id=unique_thread_id,
                salience=0.7,
            )
            print(f"  + {content}")

        banner("3. Active facts before reconcile")
        before = mem.get_memories(user_id=unique_user_id, memory_types=["fact"])
        print_facts(before)

        banner("4. Running reconcile (dedup pass, then contradiction pass)")
        stats = mem.reconcile(user_id=unique_user_id)
        print(f"  stats: {dict(stats)}")

        banner("5. Active facts after reconcile (duplicates folded, contradictions resolved)")
        after = mem.get_memories(user_id=unique_user_id, memory_types=["fact"])
        print_facts(after)

        banner("6. Audit trail (soft-deleted records)")
        all_facts = mem.get_memories(
            user_id=unique_user_id,
            memory_types=["fact"],
            include_superseded=True,
        )
        soft_deleted = [f for f in all_facts if f.get("supersede_reason")]
        if not soft_deleted:
            print("  (no soft-deleted records)")
        for f in soft_deleted:
            print(
                f"  id={f['id']}  reason={f.get('supersede_reason')}  "
                f"superseded_at={f.get('superseded_at')}  "
                f"superseded_by={f.get('superseded_by')}"
            )
            print(f"  content: {f.get('content', '')}")

    finally:
        banner("7. Cleanup")
        try:
            all_records = mem.get_memories(
                user_id=unique_user_id,
                include_superseded=True,
            )
            deleted = 0
            for rec in all_records:
                try:
                    mem.delete_memory(
                        memory_id=rec["id"],
                        user_id=unique_user_id,
                        thread_id=rec.get("thread_id", unique_thread_id),
                        memory_type=rec["type"],
                    )
                    deleted += 1
                except MemoryNotFoundError:
                    continue  # already deleted with the memory it was folded into
                except Exception as exc:  # pragma: no cover - best effort cleanup
                    print(f"  WARN: failed to delete {rec.get('id')}: {exc}")
            print(f"  Deleted {deleted} record(s) for user {unique_user_id}")
        except Exception as exc:  # pragma: no cover - best effort cleanup
            print(f"  WARN: cleanup failed: {exc}")


if __name__ == "__main__":
    main()
