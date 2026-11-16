"""Scenario: Memory reconciliation - duplicate folding and conflict links.

Demonstrates how `reconcile` cleans up a user's facts in one pass:

1. Seed paraphrased facts about a user (the pass retires a fact into another only when
   nothing it states would be lost; it may instead combine facts into one new fact).
2. Seed two facts that give different values of one attribute (a project deadline).
3. Run reconcile() and print the stats: {kept, merged, conflicts} plus the checked,
   compared, vetoed, combined, synthesized, combine_groups and combine_failed counts.
4. Show the live state - folded duplicates are hidden, but BOTH conflicting facts stay
   active: reconcile links a conflict instead of retiring either side.
5. Search with search_cosmos(user_id=...) - a fact that conflicts with another carries
   conflicts_with (up to 2 of the newest active facts, each with id, content, created_at and
   kind). Show your agent both sides and have it ask the user which is right.
6. Show the audit trail (include_superseded=True) - retired records carry
   supersede_reason ("duplicate" or "combined"), superseded_at, and superseded_by pointing
   at the survivor. The conflicting facts are not among them.

The model judges every pair, so a run can differ from this outline: a pair it judges
compatible is neither retired nor linked.

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
from datetime import datetime, timedelta, timezone

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


def print_hits(hits: list[dict]) -> None:
    """Print search hits and, under each, the active facts it conflicts with."""
    if not hits:
        print("  (none)")
        return
    for h in hits:
        print(f"  id={h['id']}  content: {h.get('content', '')}")
        for other in h.get("conflicts_with", []):
            print(f"      conflicts with ({other['kind']}): {other['content']}  [{other['created_at']}]")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

PARAPHRASED_FACTS = [
    "User prefers aisle seats on flights.",
    "User likes aisle seats when flying.",
    "User always picks aisle when booking flights.",
]

# Two values for one attribute, recorded 30 days apart.
CONFLICTING_FACTS = [
    ("The project deadline is March 1.", 30),
    ("The project deadline is March 15.", 0),
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

        banner("2. Seeding conflicting facts (one attribute, two values)")
        now = datetime.now(timezone.utc)
        for content, days_ago in CONFLICTING_FACTS:
            mem.upsert_memory(
                user_id=unique_user_id,
                role="user",
                content=content,
                memory_type="fact",
                thread_id=unique_thread_id,
                salience=0.7,
                created_at=now - timedelta(days=days_ago),
            )
            print(f"  + {content}  ({days_ago} day(s) ago)")

        banner("3. Active facts before reconcile")
        before = mem.get_memories(user_id=unique_user_id, memory_types=["fact"])
        print_facts(before)

        banner("4. Running reconcile (one pass: fold duplicates, combine facts, link conflicts)")
        stats = mem.reconcile(user_id=unique_user_id)
        print(f"  stats: {dict(stats)}")
        print(f"  duplicates retired: {stats['merged']}  conflicts linked: {stats['conflicts']}")

        banner("5. Active facts after reconcile (duplicates folded, both conflicting facts kept)")
        after = mem.get_memories(user_id=unique_user_id, memory_types=["fact"])
        print_facts(after)

        banner("6. Search shows the conflict (conflicts_with)")
        # With user_id, a fact that conflicts with others carries conflicts_with (up to 2 of the newest
        # active ones). Show both sides to the user and ask which is right.
        hits = mem.search_cosmos("project deadline", user_id=unique_user_id, top_k=5)
        print_hits(hits)
        if not any(h.get("conflicts_with") for h in hits):
            print("  (no conflicts linked: the model judged the facts compatible)")

        banner("7. Audit trail (soft-deleted records)")
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
        banner("8. Cleanup")
        # Conflict links hold only fact ids and are ignored once either fact is gone, so they need no cleanup.
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
