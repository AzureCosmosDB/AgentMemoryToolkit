"""
Multi-Agent Collaboration — Agent Memory Toolkit

Demonstrates two agents (planner + researcher) sharing a single Cosmos DB
thread so each can read the other's contributions via get_thread.

Agent identity is tracked through metadata={"agent_id": "..."} on every
add_cosmos call.

Workflow:
    1. User posts a complex question.
    2. Planner breaks it into sub-tasks and stores the plan.
    3. Researcher reads the plan, performs research, stores findings.
    4. Planner reads the researcher's findings and generates a final answer.

Required environment variables (or .env file):
    COSMOS_DB_ENDPOINT   – Azure Cosmos DB endpoint URL
    COSMOS_DB_DATABASE   – database name  (default: "ai_memory")
    COSMOS_DB_CONTAINER  – container name (default: "memories")
"""

import os
import uuid

from dotenv import load_dotenv

from agent_memory_toolkit import CosmosMemoryClient

load_dotenv()

# Agent identifiers (stored in metadata, not a first-class parameter)
PLANNER = "planner"
RESEARCHER = "researcher"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def print_header(title: str) -> None:
    """Print a formatted section header."""
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print(f"{'=' * 60}")


def print_thread(thread: list[dict], label: str = "Thread") -> None:
    """Pretty-print a thread, highlighting agent_id from metadata."""
    print(f"\n  — {label} ({len(thread)} messages) —")
    for i, msg in enumerate(thread, 1):
        role = msg.get("role", "?")
        agent = msg.get("metadata", {}).get("agent_id", "n/a")
        content = msg.get("content", "")
        mem_type = msg.get("memory_type", "turn")
        tag = f"[{role}|agent={agent}|type={mem_type}]"
        # Truncate long content for readability
        preview = (content[:100] + "…") if len(content) > 100 else content
        print(f"  {i}. {tag} {preview}")


def filter_by_agent(thread: list[dict], agent_id: str) -> list[dict]:
    """Return only messages written by a specific agent."""
    return [
        msg for msg in thread
        if msg.get("metadata", {}).get("agent_id") == agent_id
    ]


# ---------------------------------------------------------------------------
# Scenario steps
# ---------------------------------------------------------------------------

def step1_user_question(
    mem: CosmosMemoryClient, user_id: str, thread_id: str,
) -> None:
    """User asks a complex, multi-part question."""
    print_header("Step 1 — User posts a complex question")

    mem.add_cosmos(
        user_id=user_id,
        role="user",
        content=(
            "Compare the environmental impact of electric vehicles vs hydrogen "
            "fuel-cell vehicles.  Cover manufacturing, energy sources, and "
            "end-of-life recycling.  Give a recommendation for urban fleets."
        ),
        memory_type="turn",
        thread_id=thread_id,
    )
    print("  ✓ User question stored in thread.")


def step2_planner_creates_plan(
    mem: CosmosMemoryClient, user_id: str, thread_id: str,
) -> None:
    """Planner reads the user question and stores a structured plan."""
    print_header("Step 2 — Planner reads question & creates plan")

    # Planner retrieves the thread to see the user's question
    thread = mem.get_thread(thread_id=thread_id, user_id=user_id)
    user_msg = next(
        (m for m in thread if m.get("role") == "user"), None,
    )
    print(f"  Planner sees user question: {user_msg['content'][:80]}…")

    # Planner stores the decomposed plan
    plan = (
        "PLAN:\n"
        "  1. Research manufacturing impact (battery vs fuel-cell production)\n"
        "  2. Research energy-source lifecycle (grid electricity vs hydrogen)\n"
        "  3. Research end-of-life recycling challenges for both\n"
        "  4. Synthesise findings into a recommendation for urban fleets"
    )
    mem.add_cosmos(
        user_id=user_id,
        role="agent",
        content=plan,
        memory_type="turn",
        metadata={"agent_id": PLANNER, "step": "decomposition", "category": "plan"},
        thread_id=thread_id,
    )
    print("  ✓ Planner stored plan with 4 sub-tasks.")


def step3_researcher_performs_research(
    mem: CosmosMemoryClient, user_id: str, thread_id: str,
) -> None:
    """Researcher reads the planner's plan, then stores research findings."""
    print_header("Step 3 — Researcher reads plan & stores findings")

    # Researcher reads the full thread to find the plan
    thread = mem.get_thread(thread_id=thread_id, user_id=user_id)
    planner_msgs = filter_by_agent(thread, PLANNER)
    print(f"  Researcher found {len(planner_msgs)} message(s) from Planner.")

    plan_msg = next(
        (m for m in planner_msgs if m.get("memory_type") == "plan"), None,
    )
    if plan_msg:
        print(f"  Plan preview: {plan_msg['content'][:80]}…")

    # Researcher stores findings for each sub-task
    findings = [
        {
            "content": (
                "FINDING 1 — Manufacturing: EV battery production emits ~8 tonnes "
                "CO₂ per 60 kWh pack. Fuel-cell stack production is lower per unit "
                "but hydrogen tanks require energy-intensive carbon-fibre."
            ),
            "subtask": "manufacturing",
        },
        {
            "content": (
                "FINDING 2 — Energy source: EVs charged on renewable grids achieve "
                "near-zero operational emissions. Green hydrogen (electrolysis) is "
                "promising but currently 3× less energy-efficient than direct "
                "grid-to-battery charging."
            ),
            "subtask": "energy_source",
        },
        {
            "content": (
                "FINDING 3 — End-of-life: Li-ion battery recycling recovers ~95% "
                "of cobalt and nickel via hydrometallurgy. Fuel-cell membranes "
                "(PEM) contain platinum, which is recoverable but recycling "
                "infrastructure is nascent."
            ),
            "subtask": "recycling",
        },
    ]

    for f in findings:
        mem.add_cosmos(
            user_id=user_id,
            role="agent",
            content=f["content"],
            memory_type="turn",
            metadata={"agent_id": RESEARCHER, "subtask": f["subtask"], "category": "research"},
            thread_id=thread_id,
        )
    print(f"  ✓ Researcher stored {len(findings)} findings.")


def step4_planner_synthesises_answer(
    mem: CosmosMemoryClient, user_id: str, thread_id: str,
) -> None:
    """Planner reads the researcher's findings and generates a final answer."""
    print_header("Step 4 — Planner reads findings & produces final answer")

    # Planner retrieves the full thread
    thread = mem.get_thread(thread_id=thread_id, user_id=user_id)

    # Filter to researcher's contributions only
    research_msgs = filter_by_agent(thread, RESEARCHER)
    print(f"  Planner found {len(research_msgs)} research finding(s).")
    for r in research_msgs:
        subtask = r.get("metadata", {}).get("subtask", "?")
        print(f"    • subtask={subtask}: {r['content'][:60]}…")

    # Planner produces a synthesised recommendation
    recommendation = (
        "RECOMMENDATION: For urban fleets, battery-electric vehicles are the "
        "stronger choice today.  Manufacturing emissions are offset within "
        "2–3 years of operation on a renewable grid, energy efficiency is 3× "
        "higher than hydrogen, and battery recycling infrastructure is more "
        "mature.  Hydrogen fuel-cell vehicles may become competitive once "
        "green-hydrogen costs fall and recycling capacity scales."
    )
    mem.add_cosmos(
        user_id=user_id,
        role="agent",
        content=recommendation,
        memory_type="turn",
        metadata={"agent_id": PLANNER, "step": "synthesis", "category": "recommendation"},
        thread_id=thread_id,
    )
    print("  ✓ Planner stored final recommendation.")


def step5_review_shared_thread(
    mem: CosmosMemoryClient, user_id: str, thread_id: str,
) -> None:
    """Display the complete thread and per-agent filtered views."""
    print_header("Step 5 — Review the shared thread")

    full_thread = mem.get_thread(thread_id=thread_id, user_id=user_id)
    print_thread(full_thread, label="Full shared thread")

    # Filtered views
    planner_only = filter_by_agent(full_thread, PLANNER)
    researcher_only = filter_by_agent(full_thread, RESEARCHER)
    print_thread(planner_only, label=f"Filtered — {PLANNER} only")
    print_thread(researcher_only, label=f"Filtered — {RESEARCHER} only")

    # You can also filter by memory_type (e.g. only "turn" or "summary")
    turns = mem.get_thread(
        thread_id=thread_id, user_id=user_id, memory_type="turn",
    )
    print_thread(turns, label="Filtered by memory_type='turn'")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    cosmos_endpoint = os.environ.get("COSMOS_DB_ENDPOINT")
    if not cosmos_endpoint:
        raise SystemExit(
            "Error: Set COSMOS_DB_ENDPOINT environment variable."
        )

    mem = CosmosMemoryClient(
        cosmos_endpoint=cosmos_endpoint,
        cosmos_database=os.environ.get("COSMOS_DB_DATABASE", "ai_memory"),
        cosmos_container=os.environ.get("COSMOS_DB_CONTAINER", "memories"),
        cosmos_key=os.environ.get("COSMOS_DB_KEY"),
        ai_foundry_endpoint=os.environ.get("AI_FOUNDRY_ENDPOINT"),
        ai_foundry_api_key=os.environ.get("AI_FOUNDRY_API_KEY"),
        embedding_deployment_name=os.environ.get("AI_FOUNDRY_EMBEDDING_DEPLOYMENT_NAME", "text-embedding-3-large"),
        chat_deployment_name=os.environ.get("AI_FOUNDRY_CHAT_DEPLOYMENT_NAME", "gpt-4o-mini"),
        use_default_credential=True,
    )
    mem.connect_cosmos()
    print("Connected to Cosmos DB")

    user_id = "multi-agent-demo-user"
    thread_id = str(uuid.uuid4())
    print(f"Shared thread: {thread_id}")

    # Execute the multi-agent workflow
    step1_user_question(mem, user_id, thread_id)
    step2_planner_creates_plan(mem, user_id, thread_id)
    step3_researcher_performs_research(mem, user_id, thread_id)
    step4_planner_synthesises_answer(mem, user_id, thread_id)
    step5_review_shared_thread(mem, user_id, thread_id)

    print(f"\n{'=' * 60}")
    print("  Multi-agent scenario complete.")
    print(f"{'=' * 60}\n")


if __name__ == "__main__":
    main()
