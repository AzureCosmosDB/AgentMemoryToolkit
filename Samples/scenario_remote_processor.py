"""Scenario: Remote processor — processing happens in the sibling Azure Function app.

This sample shows how to wire :class:`DurableFunctionProcessor` so the SDK only
writes raw turns to Cosmos DB, and a sibling Azure Function app (deployed via
``azd up`` with the full profile) picks them up from the Cosmos change feed and
produces summaries / facts / user profiles asynchronously.

Prerequisites
-------------
* ``COSMOS_DB_ENDPOINT`` (and ``COSMOS_DB_KEY`` *or* a logged-in identity) set.
* The sibling function app deployed and running, configured against the same
  Cosmos account / database / container, with at least one threshold > 0
  (e.g. ``THREAD_SUMMARY_EVERY_N=4``).

Behavior notes
--------------
* :meth:`CosmosMemoryClient.process_now` is a debug-logged no-op when using the
  durable processor — the function app owns processing.
* :meth:`CosmosMemoryClient.process_now_and_wait` *polls* Cosmos for the resulting
  summary; that costs RUs so it's opt-in and intended for demos / tests.
"""
from __future__ import annotations

import os

from agent_memory_toolkit import CosmosMemoryClient, DurableFunctionProcessor


def main() -> None:
    client = CosmosMemoryClient(
        cosmos_endpoint=os.environ["COSMOS_DB_ENDPOINT"],
        cosmos_key=os.environ.get("COSMOS_DB_KEY"),
        cosmos_database=os.environ.get("COSMOS_DB_DATABASE", "ai_memory"),
        cosmos_container=os.environ.get("COSMOS_DB_CONTAINER", "memories"),
        # Hand processing off to the sibling Azure Function app.
        processor=DurableFunctionProcessor(),
    )

    user_id = "alice"
    thread_id = "thread-001"

    # Write turns just like normal — the SDK never invokes the LLM here.
    transcript = [
        ("user", "Hi! I love Cosmos DB."),
        ("agent", "Cosmos DB is fantastic for low-latency global apps."),
        ("user", "Can it do vector search?"),
        ("agent", "Yes — DiskANN indexes power semantic search natively."),
        ("user", "Great. What about hierarchical partition keys?"),
        ("agent", "HPK lets you co-locate related items for efficient queries."),
    ]
    for role, content in transcript:
        client.add_cosmos(
            user_id=user_id,
            thread_id=thread_id,
            role=role,
            content=content,
            memory_type="turn",
        )
        print(f"  wrote {role:>9}: {content}")

    # No-op locally; the change-feed trigger in the function app does the work.
    client.process_now(user_id=user_id, thread_id=thread_id)

    # Optional: block until the function app has produced a summary for this
    # thread. Polls Cosmos every 0.5s until ``timeout`` — RU-costly, demo only.
    print("\nWaiting for the function app to produce a summary...")
    ok = client.process_now_and_wait(user_id=user_id, thread_id=thread_id, timeout=60.0)
    print(f"Summary available: {ok}")


if __name__ == "__main__":
    main()
