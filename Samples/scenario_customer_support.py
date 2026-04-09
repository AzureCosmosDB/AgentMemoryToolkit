"""Customer-support scenario — build user profiles over multiple tickets.

Requires:
    • Azure Cosmos DB (message storage)
    • Azure Durable Functions / ADF (fact extraction & summary generation)

Environment variables:
    COSMOS_DB_ENDPOINT   – Cosmos DB account endpoint URL
    AI_FOUNDRY_ENDPOINT  – Azure OpenAI / AI Foundry endpoint for embeddings
    ADF_ENDPOINT         – Base URL for the Azure Durable Functions API
    ADF_KEY              – (optional) function-level key for the Azure Function

The script walks through three support tickets for a single customer,
demonstrating how extracted facts and a generated user summary enable
increasingly personalized interactions.
"""

import os

from dotenv import load_dotenv
load_dotenv()
import uuid

from agent_memory_toolkit import CosmosMemoryClient

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ticket_id() -> str:
    """Return a unique ticket / thread ID."""
    return f"ticket-{uuid.uuid4().hex[:8]}"


def _banner(title: str) -> None:
    """Print a section banner."""
    width = 60
    print(f"\n{'=' * width}")
    print(f"  {title}")
    print(f"{'=' * width}\n")


# ---------------------------------------------------------------------------
# Simulated support conversations
# ---------------------------------------------------------------------------

USER_ID = "cust-1"


def _run_ticket_1(mem: CosmosMemoryClient, ticket_id: str) -> None:
    """Ticket 1 — billing issue (overcharge on monthly plan)."""
    _banner(f"Ticket 1: Billing Issue  [{ticket_id}]")

    turns = [
        ("user",  "Hi, I noticed an extra $15 charge on my last invoice. "
                  "I'm on the Basic monthly plan — can you look into this?"),
        ("agent", "Hello! I'm sorry about the unexpected charge. Let me pull "
                  "up your billing history right away."),
        ("user",  "Thanks. I didn't authorize any add-ons or upgrades."),
        ("agent", "I can see the $15 charge was caused by a temporary promo "
                  "add-on that wasn't removed properly. I've refunded the "
                  "amount — you should see the credit within 3-5 business "
                  "days. Is there anything else I can help with?"),
        ("user",  "No, that's it. Thanks for the quick help!"),
        ("agent", "You're welcome! Don't hesitate to reach out if you need "
                  "anything. Have a great day!"),
    ]

    for role, content in turns:
        print(f"  [{role.upper()}] {content}")
        mem.add_cosmos(
            user_id=USER_ID, role=role, content=content, thread_id=ticket_id,
        )


def _run_ticket_2(mem: CosmosMemoryClient, ticket_id: str) -> None:
    """Ticket 2 — product feature question (data-export)."""
    _banner(f"Ticket 2: Feature Question  [{ticket_id}]")

    turns = [
        ("user",  "Is there a way to export my dashboard data to CSV? "
                  "I need to share weekly reports with my team."),
        ("agent", "Great question! You can export any dashboard by clicking "
                  "the '⋮' menu in the top-right corner and selecting "
                  "'Export → CSV'. Would you like a walkthrough?"),
        ("user",  "That would be helpful. Also, can I schedule automatic "
                  "exports?"),
        ("agent", "Scheduled exports are available on the Pro plan. Since "
                  "you're on Basic, you'd need to upgrade. I can send you "
                  "a comparison link if you're interested."),
        ("user",  "Yes please, send the link. I'll discuss with my manager."),
        ("agent", "Here's the link: https://example.com/plans. If you have "
                  "any questions about the upgrade, feel free to ask!"),
    ]

    for role, content in turns:
        print(f"  [{role.upper()}] {content}")
        mem.add_cosmos(
            user_id=USER_ID, role=role, content=content, thread_id=ticket_id,
        )


def _run_ticket_3_greeting(mem: CosmosMemoryClient, ticket_id: str) -> None:
    """Ticket 3 — personalized greeting using the accumulated profile."""
    _banner(f"Ticket 3: Personalized Greeting  [{ticket_id}]")

    # Retrieve the user profile built from previous tickets
    profile = mem.get_user_summary(user_id=USER_ID)

    if profile:
        summary_text = profile[0].get("content", "")
        print("  [PROFILE LOADED]")
        print(f"  {summary_text}\n")
    else:
        summary_text = ""
        print("  [PROFILE] No user summary available yet.\n")

    # The agent can now craft a personalized opening
    greeting = (
        "Welcome back! I see you're on our Basic plan and recently had a "
        "billing adjustment. I also noticed you were exploring the Pro plan "
        "for scheduled CSV exports — happy to help with anything related to "
        "that, or whatever else you need today!"
    )

    user_msg = "Hi, I have a quick question about integrations."
    print(f"  [USER]  {user_msg}")
    mem.add_cosmos(
        user_id=USER_ID, role="user", content=user_msg, thread_id=ticket_id,
    )

    print(f"  [AGENT] {greeting}")
    mem.add_cosmos(
        user_id=USER_ID, role="agent", content=greeting, thread_id=ticket_id,
    )

    print("\n  ✅  The agent used the stored profile to personalize its "
          "greeting — no need to ask the customer to repeat context from "
          "earlier tickets.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    # ---- Setup --------------------------------------------------------------
    mem = CosmosMemoryClient(
        cosmos_endpoint=os.environ["COSMOS_DB_ENDPOINT"],
        ai_foundry_endpoint=os.environ["AI_FOUNDRY_ENDPOINT"],
        adf_endpoint=os.environ["ADF_ENDPOINT"],
        adf_key=os.environ.get("ADF_KEY"),
    )
    mem.connect_cosmos()
    print("Connected to Cosmos DB and processing backend.\n")

    # ---- Ticket 1: Billing issue -------------------------------------------
    ticket_1_id = _ticket_id()
    _run_ticket_1(mem, ticket_1_id)

    print("\n  → Extracting facts from Ticket 1 …")
    facts_1 = mem.extract_facts(user_id=USER_ID, thread_id=ticket_1_id)
    print(f"  ✓ Facts extracted: {facts_1}\n")

    # ---- Ticket 2: Feature question ----------------------------------------
    ticket_2_id = _ticket_id()
    _run_ticket_2(mem, ticket_2_id)

    print("\n  → Extracting facts from Ticket 2 …")
    facts_2 = mem.extract_facts(user_id=USER_ID, thread_id=ticket_2_id)
    print(f"  ✓ Facts extracted: {facts_2}\n")

    # ---- Generate cross-ticket user summary --------------------------------
    _banner("Generating User Summary")
    print(f"  Combining facts from tickets: {ticket_1_id}, {ticket_2_id}")
    summary_result = mem.generate_user_summary(
        user_id=USER_ID, thread_ids=[ticket_1_id, ticket_2_id],
    )
    print(f"  ✓ Summary generated: {summary_result}\n")

    # Retrieve and display the profile
    profile = mem.get_user_summary(user_id=USER_ID)
    if profile:
        _banner("Stored User Profile")
        print(f"  {profile[0].get('content', '(empty)')}\n")

    # ---- Ticket 3: Show personalized greeting ------------------------------
    ticket_3_id = _ticket_id()
    _run_ticket_3_greeting(mem, ticket_3_id)


if __name__ == "__main__":
    main()
