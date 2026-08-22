"""Demonstrate procedural memory extraction, retrieval, and prompt projection.

Procedural memory is a skill and policy library. Instead of storing one
mutable prompt blob, the toolkit stores many atomic ``ProceduralRecord`` items:
behavioral policies, workflows, decision rules, tool-usage notes, and recovery
strategies. Each procedure carries provenance, activation conditions, scope,
status, and executable steps where appropriate.

The personalized system prompt is compiled on demand by
``CosmosMemoryClient.build_procedural_context(...)``. That prompt is a
deterministic projection of ACTIVE procedures only. Candidate procedures remain
in the library for inspection and later promotion, but they are not injected.

This sample seeds both kinds of sources:

1. An explicit user instruction: "Always ask for confirmation before deleting
   cloud resources." Explicit user instructions are trusted provenance, so the
   synthesized behavioral policy should become ACTIVE.
2. A realistic debugging episode about a Cosmos DB ORDER BY query. Lessons
   distilled from episodes are useful skills, but episode-distilled procedures
   stay CANDIDATE under the provenance gate and are excluded from prompt
   injection until promoted by trusted policy or review.

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
from typing import Any

from dotenv import load_dotenv

from azure.cosmos.agent_memory import CosmosMemoryClient

load_dotenv()

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

DIVIDER = "-" * 72


def banner(title: str) -> None:
    """Print a section banner."""
    print(f"\n{DIVIDER}")
    print(f"  {title}")
    print(DIVIDER)


def _short(value: Any, width: int = 120) -> str:
    """Return a single-line string trimmed for readable terminal output."""
    text = str(value or "").replace("\n", " ").strip()
    return text if len(text) <= width else f"{text[: width - 3]}..."


def print_procedure(proc: dict[str, Any], *, index: int | None = None) -> None:
    """Pretty-print the fields that make procedural memory understandable."""
    prefix = f"  {index}." if index is not None else "  -"
    print(f"{prefix} {proc.get('name') or '(unnamed)'}")
    print(f"      kind:        {proc.get('procedure_kind')}")
    print(f"      scope:       {proc.get('scope_type') or '(none)'}:{proc.get('scope_value') or '*'}")
    print(f"      status:      {proc.get('status')}")
    print(f"      source_kind: {proc.get('source_kind')}")
    print(f"      summary:     {_short(proc.get('summary') or proc.get('content'))}")
    conditions = proc.get("activation_conditions") or []
    if conditions:
        print("      activates:")
        for condition in conditions[:3]:
            print(f"        - {_short(condition, 96)}")
    steps = proc.get("steps") or []
    if steps:
        print("      steps:")
        for step in sorted(steps, key=lambda item: int(item.get("sequence") or 0))[:4]:
            instruction = step.get("instruction") if isinstance(step, dict) else step
            print(f"        {step.get('sequence', '-')}. {_short(instruction, 96)}")


def print_procedures(title: str, procedures: list[dict[str, Any]]) -> None:
    """Print a titled procedure list."""
    print(title)
    if not procedures:
        print("  (none)")
        return
    for index, procedure in enumerate(procedures, start=1):
        print_procedure(procedure, index=index)


# ---------------------------------------------------------------------------
# Demo data
# ---------------------------------------------------------------------------

# Thread 1 is a direct instruction from the user. The procedural synthesizer can
# turn this into a behavioral_policy. Because the source is an explicit user
# instruction, provenance gating should mark it ACTIVE.
POLICY_TURNS = [
    (
        "user",
        "Always ask for confirmation before deleting cloud resources, including "
        "Cosmos DB accounts, databases, containers, or resource groups.",
    ),
    (
        "agent",
        "Understood. I will ask for confirmation before deleting any cloud resource.",
    ),
]

# Thread 2 is an ordinary debugging conversation. Episodic extraction should
# produce an episode with a lesson, and procedural synthesis can distill that
# lesson into a reusable Cosmos DB recovery_strategy or workflow. Because this
# is episode-distilled rather than an explicit policy, it remains CANDIDATE and
# is not injected into the compiled system prompt.
DEBUGGING_TURNS = [
    (
        "user",
        "My Cosmos DB query is failing: SELECT * FROM c WHERE c.customerId = "
        "@customerId ORDER BY c.createdAt DESC. The portal says the ORDER BY "
        "query needs a matching composite index.",
    ),
    (
        "agent",
        "First confirm the query filters by the partition key, then inspect the "
        "indexing policy for a composite index on customerId ASC and createdAt DESC.",
    ),
    (
        "user",
        "The partition key is /customerId. The container only has the default "
        "range index; there is no composite index configured.",
    ),
    (
        "agent",
        "Add a composite index matching the equality filter and ORDER BY sort, "
        "wait for indexing to finish, and retry with a small TOP literal while testing.",
    ),
    (
        "user",
        "After adding the composite index and waiting for transformation progress, "
        "the query succeeded. Lesson learned: for Cosmos DB ORDER BY failures, "
        "check partition scope and composite indexes before changing application code.",
    ),
]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


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
        print("Set the variables listed in the module docstring, then rerun this sample.")
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

    user_id = f"procedural-demo-{uuid.uuid4().hex[:8]}"
    policy_thread_id = f"procedural-policy-{uuid.uuid4().hex[:8]}"
    debug_thread_id = f"procedural-debug-{uuid.uuid4().hex[:8]}"
    print(f"User ID:          {user_id}")
    print(f"Policy thread ID: {policy_thread_id}")
    print(f"Debug thread ID:  {debug_thread_id}")

    try:
        banner("1. Seed explicit instruction turns")
        for role, content in POLICY_TURNS:
            mem.upsert_memory(user_id=user_id, role=role, content=content, thread_id=policy_thread_id)
            print(f"  [{role:>5}] {_short(content)}")

        banner("2. Extract facts from the explicit instruction")
        # Fact extraction supplies behavioral facts to synthesize_procedural.
        # The direct "Always ask..." instruction should be categorized as a
        # user requirement or high-salience preference by the extraction prompt.
        fact_stats = mem.extract_memories(user_id=user_id, thread_id=policy_thread_id)
        print(f"  stats: {json.dumps(fact_stats, indent=2)}")

        banner("3. Seed a Cosmos DB debugging episode")
        for role, content in DEBUGGING_TURNS:
            mem.upsert_memory(user_id=user_id, role=role, content=content, thread_id=debug_thread_id)
            print(f"  [{role:>5}] {_short(content)}")

        banner("4. Finalize the episode and expose its lesson")
        # In production, episode boundaries are evaluated automatically by the
        # processing backend. This script knows the demo conversation is over,
        # so flush=True drains the trailing open segment now.
        episode_stats = mem.extract_episodes(user_id, debug_thread_id, flush=True)
        print(f"  stats: {json.dumps(episode_stats, indent=2)}")
        episodes = mem.get_episodes(user_id, thread_id=debug_thread_id)
        for episode in episodes:
            print(f"  episode: {_short(episode.get('title'))}")
            for lesson in episode.get("lessons") or []:
                print(f"    lesson: {_short(lesson)}")

        banner("5. Synthesize the procedural skill and policy library")
        synthesis_stats = mem.synthesize_procedural(user_id)
        print(f"  stats: {json.dumps(synthesis_stats, indent=2)}")
        print(f"  procedures created: {synthesis_stats.get('procedures_created', 0)}")

        banner("6. Inspect the procedural memory library")
        procedures = mem.get_procedural_memories(user_id)
        active = [p for p in procedures if p.get("status") == "active"]
        candidates = [p for p in procedures if p.get("status") == "candidate"]
        print_procedures("ACTIVE procedures from trusted provenance:", active)
        print_procedures("CANDIDATE procedures retained for review, not injection:", candidates)

        banner("7. Context-aware retrieval")
        # The default retrieval API is injection-safe: status defaults to
        # "active", so candidates are excluded just like they are excluded from
        # prompt projection.
        active_matches = mem.retrieve_procedures(user_id, "cosmos db order by query failing")
        print_procedures("Default active-only retrieval:", active_matches)

        # For library inspection or review workflows, pass status=None to rank
        # both active and candidate procedures. This should surface the Cosmos DB
        # debugging skill while the delete-confirmation policy ranks poorly or is
        # absent for this coding task.
        all_matches = mem.retrieve_procedures(user_id, "cosmos db order by query failing", status=None)
        print_procedures("Review retrieval including candidates:", all_matches)

        banner("8. Compile deterministic prompt projections")
        # With no task, only always-on global/user behavioral policies are folded
        # into the prompt. Candidate skills are deliberately excluded.
        global_context = mem.build_procedural_context(user_id)
        print("Global policies only:")
        print(global_context or "(no active global policies)")

        # With a task, ACTIVE task procedures relevant to that task can be folded
        # in after the global policies. Episode-distilled candidates still stay
        # out of the prompt until promoted by trusted provenance.
        task_context = mem.build_procedural_context(user_id, task="fix a failing cosmos db query")
        print("\nPolicies plus relevant active task skills:")
        print(task_context or "(no active policies or task skills)")

        banner("9. Closing notes")
        print(
            "Procedural memory keeps a two-layer model: atomic procedure "
            "records form the skill/policy library, and build_procedural_context "
            "compiles the deterministic prompt projection on demand. Provenance "
            "gating keeps explicit user instructions, observed preferences, and "
            "organization policy active, while episode-distilled, document, and "
            "inferred procedures remain candidates. Episodic extraction, one "
            "source of candidate procedures, also runs under the Durable Functions "
            "backend in deployed processing setups."
        )

    finally:
        banner("10. Cleanup")
        deleted = 0
        try:
            memory_records = mem.get_memories(user_id=user_id, include_superseded=True)
            policy_turns = mem.get_thread(thread_id=policy_thread_id, user_id=user_id, include_superseded=True)
            debug_turns = mem.get_thread(thread_id=debug_thread_id, user_id=user_id, include_superseded=True)
            procedural_records = mem.get_procedural_memories(user_id=user_id, include_superseded=True)
            thread_summaries = [
                *mem.get_thread_summary(user_id=user_id, thread_id=policy_thread_id),
                *mem.get_thread_summary(user_id=user_id, thread_id=debug_thread_id),
            ]
            user_summary = mem.get_user_summary(user_id)
            all_records = [*memory_records, *policy_turns, *debug_turns, *procedural_records, *thread_summaries]
            if user_summary:
                all_records.append(user_summary)

            seen_ids: set[str] = set()
            for record in all_records:
                memory_id = record.get("id")
                if not memory_id or memory_id in seen_ids:
                    continue
                seen_ids.add(memory_id)
                record_type = record.get("type")
                if not record_type:
                    continue
                try:
                    mem.delete_memory(
                        memory_id=memory_id,
                        user_id=user_id,
                        thread_id=record.get("thread_id") or debug_thread_id,
                        memory_type=record_type,
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
