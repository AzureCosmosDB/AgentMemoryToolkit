"""Azure Durable Functions app for Agent Memory operations.

Activities:
  - load_memories: Fetch memories from Cosmos DB by thread_id
  - generate_embeddings: Embed text via Azure AI Foundry
  - store_results: Upsert a memory document into Cosmos DB
  - generate_thread_summary: Generate or incrementally update a thread summary
  - extract_facts: Extract facts from memories using an AI Foundry LLM
  - generate_user_summary: Generate a cross-thread user profile

The orchestrator chains these activities in sequence.
"""

import azure.functions as func
import azure.durable_functions as df

from activities import bp as activities_bp

df_app = df.DFApp(http_auth_level=func.AuthLevel.FUNCTION)
df_app.register_functions(activities_bp)


# =====================================================================
# Orchestrator
# =====================================================================

@df_app.orchestration_trigger(context_name="context")
def memory_orchestrator(context: df.DurableOrchestrationContext):
    """Orchestrate a full memory-processing pipeline.

    Input payload::

        {
            "thread_id": "...",
            "user_id": "...",
            "content": "...",           # new content to embed & store
            "role": "user",             # optional, default "user"
            "memory_type": "turn",       # optional
            "metadata": {},             # optional
            "thread_summary": true,     # optional – trigger thread summary
            "thread_summary_only": false, # optional – skip embed/store
            "extract_facts": true,      # optional – trigger fact extraction
            "extract_facts_only": false,# optional – skip embed/store
            "user_summary": true,       # optional – trigger user summary
            "user_summary_only": false, # optional – skip embed/store
            "thread_ids": null,         # optional – limit user summary to
                                        #   specific threads
            "recent_k": null            # optional – limit memories to
                                        #   the most recent k for summary/facts
        }
    """
    payload = context.get_input()
    thread_summary_only = payload.get("thread_summary_only", False)
    extract_facts_only = payload.get("extract_facts_only", False)
    user_summary_only = payload.get("user_summary_only", False)

    if thread_summary_only:
        # --- Thread-summary-only path ---
        summary_doc = yield context.call_activity(
            "generate_thread_summary",
            {
                "user_id": payload["user_id"],
                "thread_id": payload["thread_id"],
                "recent_k": payload.get("recent_k"),
            },
        )
        return summary_doc

    if extract_facts_only:
        # --- Extract-facts-only path ---
        facts_doc = yield context.call_activity(
            "extract_facts",
            {
                "user_id": payload["user_id"],
                "thread_id": payload["thread_id"],
                "recent_k": payload.get("recent_k"),
            },
        )
        return facts_doc

    if user_summary_only:
        # --- User-summary-only path ---
        user_summary_doc = yield context.call_activity(
            "generate_user_summary",
            {
                "user_id": payload["user_id"],
                "thread_ids": payload.get("thread_ids"),
                "recent_k": payload.get("recent_k"),
            },
        )
        return user_summary_doc

    # --- Full pipeline path ---

    # 1. Load existing memories for the thread
    memories = yield context.call_activity(
        "load_memories",
        {"thread_id": payload.get("thread_id")},
    )

    # 2. Generate embeddings for the new content
    embedding = yield context.call_activity(
        "generate_embeddings",
        {"text": payload.get("content")},
    )

    # 3. Store the new memory (with its embedding) in Cosmos DB
    store_input = {
        "user_id": payload.get("user_id"),
        "thread_id": payload.get("thread_id"),
        "role": payload.get("role", "user"),
        "content": payload.get("content"),
        "memory_type": payload.get("memory_type", "turn"),
        "metadata": payload.get("metadata", {}),
        "embedding": embedding,
    }
    yield context.call_activity("store_results", store_input)

    # 4. Optionally summarize the thread
    summary = None
    if payload.get("thread_summary"):
        summary = yield context.call_activity(
            "generate_thread_summary",
            {
                "user_id": payload["user_id"],
                "thread_id": payload["thread_id"],
                "recent_k": payload.get("recent_k"),
            },
        )

    # 5. Optionally extract facts from the thread
    facts = None
    if payload.get("extract_facts"):
        facts_doc = yield context.call_activity(
            "extract_facts",
            {
                "user_id": payload["user_id"],
                "thread_id": payload["thread_id"],
                "recent_k": payload.get("recent_k"),
            },
        )
        facts = facts_doc

    # 6. Optionally generate a user summary
    user_summary = None
    if payload.get("user_summary"):
        user_summary = yield context.call_activity(
            "generate_user_summary",
            {
                "user_id": payload["user_id"],
                "thread_ids": payload.get("thread_ids"),
                "recent_k": payload.get("recent_k"),
            },
        )

    return {
        "thread_id": payload.get("thread_id"),
        "memories_loaded": len(memories) if memories else 0,
        "stored": True,
        "summary": summary,
        "facts": facts,
        "user_summary": user_summary,
    }


# =====================================================================
# HTTP starter
# =====================================================================

@df_app.route(route="orchestrators/{functionName}")
@df_app.durable_client_input(client_name="client")
async def http_start(req: func.HttpRequest, client):
    """HTTP trigger that starts the durable orchestration."""
    function_name = req.route_params.get("functionName", "memory_orchestrator")
    payload = req.get_json()
    instance_id = await client.start_new(function_name, client_input=payload)

    return client.create_check_status_response(req, instance_id)
