"""Scenario: Tuning the per-orchestrator counter thresholds (durable mode).

This sample demonstrates the *operational* surface of the
:class:`DurableFunctionProcessor` deployment: the SDK never invokes an LLM
locally — instead, the sibling Azure Function app reads the Cosmos change
feed and decides which orchestrator(s) to fire based on per-(user,thread)
counters.

Three orchestrators run independently, each gated on its own threshold:

+--------------------------+---------+------------------------------------+
| App setting              | Default | Effect (set 0 to disable)          |
+--------------------------+---------+------------------------------------+
| THREAD_SUMMARY_EVERY_N   |   4     | Summarize a thread every N turns   |
| FACT_EXTRACTION_EVERY_N  |   4     | Extract facts/episodic/procedural  |
| USER_SUMMARY_EVERY_N     |  20     | Roll up user profile every N turns |
+--------------------------+---------+------------------------------------+

This sample writes a transcript long enough to cross the thread-level
thresholds and uses :meth:`process_now_and_wait` to *poll* Cosmos until the
function app has produced a summary. ``process_now_and_wait`` is intended for
demos/tests; production code should not poll Cosmos for orchestration
state.

Prerequisites
-------------
* ``COSMOS_DB_ENDPOINT`` set, plus ``COSMOS_DB_KEY`` *or* a logged-in identity.
* The sibling function app deployed (``azd up``) and running.
* App settings tuned, e.g.::

      azd env set THREAD_SUMMARY_EVERY_N 4
      azd env set FACT_EXTRACTION_EVERY_N 4
      azd env set USER_SUMMARY_EVERY_N 20
      azd deploy

  Set any value to ``0`` to disable that orchestrator entirely (useful for
  cost control or for staged rollout).
"""
from __future__ import annotations

import os
import time
import uuid

from agent_memory_toolkit import CosmosMemoryClient, DurableFunctionProcessor


def main() -> None:
    user_id = "alice"
    thread_id = f"thread-{uuid.uuid4().hex[:8]}"

    client = CosmosMemoryClient(
        cosmos_endpoint=os.environ["COSMOS_DB_ENDPOINT"],
        cosmos_key=os.environ.get("COSMOS_DB_KEY"),
        cosmos_database=os.environ.get("COSMOS_DB_DATABASE", "ai_memory"),
        cosmos_container=os.environ.get("COSMOS_DB_CONTAINER", "memories"),
        # Hand processing off to the sibling Azure Function app.
        processor=DurableFunctionProcessor(),
    )

    # ------------------------------------------------------------------
    # Write a 5-turn transcript. With THREAD_SUMMARY_EVERY_N=4 the
    # function app will fire the thread-summary orchestrator on turn 4.
    # ------------------------------------------------------------------
    transcript = [
        ("user",      "I'm planning a hiking trip to Olympic National Park."),
        ("agent", "Great choice! Hoh Rainforest and Hurricane Ridge are must-sees."),
        ("user",      "I'd like to camp 2 nights. Any permit guidance?"),
        ("agent", "You'll need a wilderness permit from recreation.gov for the Hoh."),
        ("user",      "Thanks — also, I'm vegetarian, please remember that."),
    ]
    print(f"Writing {len(transcript)} turns to Cosmos (thread={thread_id})...")
    for role, content in transcript:
        client.add_cosmos(
            user_id=user_id,
            thread_id=thread_id,
            role=role,
            content=content,
            memory_type="turn",
        )

    # ------------------------------------------------------------------
    # In durable mode .process_now() is a no-op locally — the function app's
    # change-feed trigger has already begun work asynchronously. Use
    # process_now_and_wait() to *poll* for the summary doc; it's RU-costly so
    # use it only in demos/tests.
    # ------------------------------------------------------------------
    print("\nWaiting (poll) for thread summary to be written by function app...")
    t0 = time.monotonic()
    ok = client.process_now_and_wait(
        user_id=user_id,
        thread_id=thread_id,
        timeout=90.0,
    )
    elapsed = time.monotonic() - t0
    print(f"  summary available after {elapsed:.1f}s: {ok}")

    if not ok:
        print(
            "\nNo summary detected within timeout. Check that the function app "
            "is running and that THREAD_SUMMARY_EVERY_N <= number of turns "
            "written above."
        )
        return

    # ------------------------------------------------------------------
    # Inspect what the function app produced.
    # ------------------------------------------------------------------
    summaries = client.get_memories(
        user_id=user_id,
        thread_id=thread_id,
        memory_type="summary",
    )
    print(f"\n  thread summaries persisted: {len(summaries)}")
    for s in summaries[:3]:
        print(f"    - {s.get('content', '')[:120]}")

    facts = client.get_memories(
        user_id=user_id,
        memory_type="fact",
    )
    print(f"\n  facts extracted: {len(facts)}")
    for f in facts[:5]:
        print(f"    - {f.get('content', '')[:120]}")

    print(
        "\nTip: to disable an orchestrator entirely (e.g. for cost control "
        "during a rollout):\n"
        "    azd env set FACT_EXTRACTION_EVERY_N 0 && azd deploy"
    )


if __name__ == "__main__":
    main()
