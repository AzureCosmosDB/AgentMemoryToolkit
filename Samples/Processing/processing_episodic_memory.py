"""Demonstrate episodic memory extraction and retrieval.

Episodic memory captures bounded user experiences as structured episodes:
what happened, who participated, when it happened, how it ended, and what
lessons were learned. Episodes are boundary-based: the toolkit segments the
turn stream into coherent experiences automatically (at an idle time-gap, a
topic shift, or a max-size cap) - the caller never signals "session end". This
sample writes a short single-topic conversation and calls
``CosmosMemoryClient.extract_episodes(..., flush=True)`` to finalize the open
segment at the end of the conversation, prints the resulting episode shape,
demonstrates blended retrieval with episodic results, and then cleans up all
created records.

Required environment variables (.env supported via python-dotenv):

    COSMOS_DB_ENDPOINT
    COSMOS_DB_DATABASE
    COSMOS_DB_MEMORIES_CONTAINER
    AI_FOUNDRY_ENDPOINT
    AI_FOUNDRY_API_KEY
    AI_FOUNDRY_CHAT_DEPLOYMENT_NAME
    AI_FOUNDRY_EMBEDDING_DEPLOYMENT_NAME
    AI_FOUNDRY_EMBEDDING_DIMENSIONS

Cosmos DB authentication uses DefaultAzureCredential. Do not pass a Cosmos DB
key when local auth is disabled on the account.
"""

from __future__ import annotations

import json
import os
import sys
import uuid

from dotenv import load_dotenv

from azure.cosmos.agent_memory import CosmosMemoryClient

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


def print_outcome(outcome: dict | None) -> None:
    """Pretty-print an episode outcome."""
    if not outcome:
        print("    outcome: none")
        return
    status = outcome.get("status") or "unknown"
    description = outcome.get("description") or ""
    print(f"    outcome: {status} - {description}")


def print_episodes(episodes: list[dict]) -> None:
    """Pretty-print episodic memory records."""
    if not episodes:
        print("  (none)")
        return

    for episode in episodes:
        print(f"  title: {episode.get('title', '')}")
        print(f"    summary: {episode.get('content', '')}")
        print(f"    started_at: {episode.get('started_at')}")
        print(f"    ended_at: {episode.get('ended_at')}")
        participants = ", ".join(episode.get("participants") or [])
        print(f"    participants: {participants or '(none)'}")
        print("    events:")
        for event in sorted(episode.get("events") or [], key=lambda item: item.get("sequence", 0)):
            print(f"      {event.get('sequence')}. {event.get('description', '')}")
        print_outcome(episode.get("outcome"))
        lessons = episode.get("lessons") or []
        print("    lessons:")
        if lessons:
            for lesson in lessons:
                print(f"      - {lesson}")
        else:
            print("      - none")


def print_search_results(results: list[dict]) -> None:
    """Pretty-print blended search results."""
    if not results:
        print("  (none)")
        return

    for result in results:
        content = str(result.get("content") or "").replace("\n", " ")
        print(f"  [{result.get('type', 'unknown')}] {content}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

CONVERSATION = [
    (
        "user",
        "On Friday afternoon, I drove from Seattle to Leavenworth with Maya and Jordan for a weekend hike.",
    ),
    (
        "agent",
        "That sounds like a fun start. What trail did you choose for Saturday?",
    ),
    (
        "user",
        "We chose the Colchuck Lake trail for Saturday morning and planned to start by 6:30 AM.",
    ),
    (
        "agent",
        "An early start should help with parking and give you cooler hiking weather.",
    ),
    (
        "user",
        "Jordan forgot his trekking poles, so we stopped at a gear shop before heading to the trailhead.",
    ),
    (
        "user",
        "The hike was harder than expected after the boulder field, but Maya paced us and kept everyone steady.",
    ),
    (
        "agent",
        "It sounds like teamwork helped the group handle the hard section.",
    ),
    (
        "user",
        "We reached the lake by noon, ate lunch by the water, and decided the trip was a success.",
    ),
]


def main() -> None:
    required = [
        "COSMOS_DB_ENDPOINT",
        "COSMOS_DB_DATABASE",
        "COSMOS_DB_MEMORIES_CONTAINER",
        "AI_FOUNDRY_ENDPOINT",
        "AI_FOUNDRY_CHAT_DEPLOYMENT_NAME",
        "AI_FOUNDRY_EMBEDDING_DEPLOYMENT_NAME",
        "AI_FOUNDRY_EMBEDDING_DIMENSIONS",
    ]
    missing = [v for v in required if not os.environ.get(v)]
    if missing:
        print(f"ERROR: missing env vars: {', '.join(missing)}")
        sys.exit(1)

    mem = CosmosMemoryClient(
        cosmos_endpoint=os.environ["COSMOS_DB_ENDPOINT"],
        cosmos_database=os.environ["COSMOS_DB_DATABASE"],
        cosmos_container=os.environ["COSMOS_DB_MEMORIES_CONTAINER"],
        ai_foundry_endpoint=os.environ["AI_FOUNDRY_ENDPOINT"],
        ai_foundry_api_key=os.environ.get("AI_FOUNDRY_API_KEY") or None,
        embedding_deployment_name=os.environ["AI_FOUNDRY_EMBEDDING_DEPLOYMENT_NAME"],
        embedding_dimensions=int(os.environ["AI_FOUNDRY_EMBEDDING_DIMENSIONS"]),
        chat_deployment_name=os.environ["AI_FOUNDRY_CHAT_DEPLOYMENT_NAME"],
        use_default_credential=True,
        cadence_thresholds={
            "FACT_EXTRACTION_EVERY_N": 0,
            "THREAD_SUMMARY_EVERY_N": 0,
            "EPISODE_EVAL_EVERY_N": 0,
            "USER_SUMMARY_EVERY_N": 0,
        },
    )
    print("Connected to Cosmos DB with DefaultAzureCredential.")

    user_id = f"episodic-demo-{uuid.uuid4().hex[:8]}"
    thread_id = f"episodic-demo-thread-{uuid.uuid4().hex[:8]}"
    print(f"User ID:   {user_id}")
    print(f"Thread ID: {thread_id}")

    try:
        banner("1. Adding conversation turns")
        for role, content in CONVERSATION:
            mem.upsert_memory(user_id=user_id, role=role, content=content, thread_id=thread_id)
            print(f"  [{role:>5}] {content}")

        banner("2. Finalizing episodes (flush the open segment)")
        # In production, boundary evaluation runs automatically every
        # EPISODE_EVAL_EVERY_N turns and closes an episode at each detected
        # boundary (idle time-gap, topic shift, or max-size cap). Here the whole
        # conversation is one bounded experience with no gap, so we flush=True to
        # drain the trailing open segment at the end of the chat.
        stats = mem.extract_episodes(user_id, thread_id, flush=True)
        print(f"  stats: {json.dumps(stats, indent=2)}")

        banner("3. Retrieved episodes")
        episodes = mem.get_episodes(user_id)
        print_episodes(episodes)

        banner("4. Blended search (facts + episodes in one combined query)")
        # With include_episodes, facts and episodes are returned by a single
        # ranked query sharing one top_k budget (no separate episodic query).
        results = mem.search_cosmos(
            search_terms="Colchuck Lake hiking outcome",
            user_id=user_id,
            include_episodes=True,
        )
        print_search_results(results)

    finally:
        banner("5. Cleanup")
        deleted = 0
        try:
            memory_records = mem.get_memories(user_id=user_id, include_superseded=True)
            thread_records = mem.get_thread(thread_id=thread_id, user_id=user_id, include_superseded=True)
            summary_records = mem.get_thread_summary(user_id=user_id, thread_id=thread_id)
            user_summary = mem.get_user_summary(user_id)
            all_records = [*memory_records, *thread_records, *summary_records]
            if user_summary:
                all_records.append(user_summary)

            seen_ids: set[str] = set()
            for record in all_records:
                memory_id = record.get("id")
                if not memory_id or memory_id in seen_ids:
                    continue
                seen_ids.add(memory_id)
                try:
                    mem.delete_memory(
                        memory_id=memory_id,
                        user_id=user_id,
                        thread_id=record.get("thread_id", thread_id),
                        memory_type=record["type"],
                    )
                    deleted += 1
                except Exception as exc:  # pragma: no cover - best effort cleanup
                    print(f"  WARN: failed to delete {memory_id}: {exc}")
            print(f"  Deleted {deleted} record(s) for user {user_id}")
        except Exception as exc:  # pragma: no cover - best effort cleanup
            print(f"  WARN: cleanup failed: {exc}")
        finally:
            mem.close()


if __name__ == "__main__":
    main()
