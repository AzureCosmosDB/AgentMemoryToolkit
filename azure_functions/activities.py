"""Activity functions for the Agent Memory durable orchestration."""

import json
import os
import uuid
from datetime import datetime, timezone

import azure.durable_functions as df
from azure.cosmos import CosmosClient
from azure.identity import DefaultAzureCredential

bp = df.Blueprint()

# ---------------------------------------------------------------------------
# Shared helpers – lazily initialised singletons
# ---------------------------------------------------------------------------

_cosmos_container = None
_credential = None


def _get_credential():
    """Return a shared DefaultAzureCredential (Entra ID / MI)."""
    global _credential
    if _credential is None:
        _credential = DefaultAzureCredential()
    return _credential


def _get_cosmos_container():
    """Return the Cosmos DB container client, connecting on first call."""
    global _cosmos_container
    if _cosmos_container is None:
        endpoint = os.environ["COSMOS_DB_ENDPOINT"]
        database = os.environ["COSMOS_DB_DATABASE"]
        container = os.environ["COSMOS_DB_CONTAINER"]
        client = CosmosClient(endpoint, credential=_get_credential())
        db = client.get_database_client(database)
        _cosmos_container = db.get_container_client(container)
    return _cosmos_container


_embeddings_client = None


def _get_embeddings_client():
    """Return a cached AzureOpenAI client for embeddings."""
    global _embeddings_client
    if _embeddings_client is None:
        from openai import AzureOpenAI

        endpoint = os.environ["AI_FOUNDRY_ENDPOINT"]
        api_key = os.environ.get("AI_FOUNDRY_API_KEY")

        if api_key:
            _embeddings_client = AzureOpenAI(
                api_version="2024-12-01-preview",
                azure_endpoint=endpoint,
                api_key=api_key,
            )
        else:
            from azure.identity import get_bearer_token_provider

            token_provider = get_bearer_token_provider(
                _get_credential(),
                "https://cognitiveservices.azure.com/.default",
            )
            _embeddings_client = AzureOpenAI(
                api_version="2024-12-01-preview",
                azure_endpoint=endpoint,
                azure_ad_token_provider=token_provider,
            )
    return _embeddings_client


_chat_client = None


def _get_chat_client():
    """Return a cached AzureOpenAI client for chat completions."""
    global _chat_client
    if _chat_client is None:
        from openai import AzureOpenAI

        endpoint = os.environ["AI_FOUNDRY_ENDPOINT"]
        api_key = os.environ.get("AI_FOUNDRY_API_KEY")

        if api_key:
            _chat_client = AzureOpenAI(
                api_version="2024-12-01-preview",
                azure_endpoint=endpoint,
                api_key=api_key,
            )
        else:
            from azure.identity import get_bearer_token_provider

            token_provider = get_bearer_token_provider(
                _get_credential(),
                "https://cognitiveservices.azure.com/.default",
            )
            _chat_client = AzureOpenAI(
                api_version="2024-12-01-preview",
                azure_endpoint=endpoint,
                azure_ad_token_provider=token_provider,
            )
    return _chat_client


# =====================================================================
# Activity: load_memories
# =====================================================================

@bp.activity_trigger(input_name="payload")
def load_memories(payload: dict) -> list:
    """Load all memories for a given thread_id from Cosmos DB.

    Input::
        {"thread_id": "..."}

    Returns a list of memory dicts.
    """
    thread_id = payload["thread_id"]
    container = _get_cosmos_container()

    query = "SELECT * FROM c WHERE c.thread_id = @thread_id"
    parameters = [{"name": "@thread_id", "value": thread_id}]

    items = container.query_items(
        query=query,
        parameters=parameters,
        enable_cross_partition_query=True,
    )
    return list(items)


# =====================================================================
# Activity: generate_embeddings
# =====================================================================

@bp.activity_trigger(input_name="payload")
def generate_embeddings(payload: dict) -> list:
    """Generate a vector embedding for the given text.

    Input::
        {"text": "some content to embed"}

    Returns a list of floats (the embedding vector).
    """
    text = payload["text"]
    model = os.environ.get("AI_FOUNDRY_EMBEDDING_MODEL", "text-embedding-3-large")

    dimensions = int(os.environ.get("EMBEDDING_DIMENSION", "1536"))
    client = _get_embeddings_client()
    response = client.embeddings.create(
        input=[text],
        model=model,
        dimensions=dimensions,
    )
    return response.data[0].embedding


# =====================================================================
# Activity: store_results
# =====================================================================

@bp.activity_trigger(input_name="payload")
def store_results(payload: dict) -> dict:
    """Store (upsert) a memory document in Cosmos DB.

    Input::
        {
            "user_id": "...",
            "thread_id": "...",
            "role": "user",
            "content": "...",
            "memory_type": "turn",
            "metadata": {},
            "embedding": [0.1, ...]
        }

    Returns the stored document.
    """
    container = _get_cosmos_container()

    doc = {
        "id": str(uuid.uuid4()),
        "user_id": payload["user_id"],
        "thread_id": payload["thread_id"],
        "role": payload.get("role", "user"),
        "type": payload.get("memory_type", "turn"),
        "content": payload["content"],
        "metadata": payload.get("metadata", {}),
        "embedding": payload["embedding"],
        "created_at": datetime.now(timezone.utc).isoformat(),
    }

    container.upsert_item(body=doc)
    return doc


# =====================================================================
# Activity: generate_thread_summary
# =====================================================================

@bp.activity_trigger(input_name="payload")
def generate_thread_summary(payload: dict) -> dict:
    """Generate or incrementally update a thread summary using an AI Foundry LLM.

    If a summary already exists for the thread, only memories created
    *after* the existing summary are fetched. The LLM then receives
    the old summary together with the new messages and produces an
    updated summary. The document is upserted with a deterministic ID
    so there is at most one active summary per thread.

    Input::
        {
            "user_id": "...",
            "thread_id": "...",
            "recent_k": 10          # optional – per-thread recency limit
        }
    """
    from azure.cosmos.exceptions import CosmosResourceNotFoundError

    user_id = payload["user_id"]
    thread_id = payload["thread_id"]
    recent_k = payload.get("recent_k")
    model = os.environ.get("LLM_MODEL", "gpt-5-nano")
    container = _get_cosmos_container()

    # ---- 1. Check for an existing thread summary ----
    existing_summary = None
    summary_id = f"summary_{user_id}_{thread_id}"
    try:
        existing_summary = container.read_item(
            item=summary_id,
            partition_key=[user_id, thread_id],
        )
    except CosmosResourceNotFoundError:
        pass  # first time – full generation

    # ---- 2. Query memories (time-filtered if updating) ----
    query = (
        "SELECT * FROM c "
        "WHERE c.user_id = @user_id AND c.thread_id = @thread_id "
        "AND c.type != 'summary'"
    )
    parameters: list[dict] = [
        {"name": "@user_id", "value": user_id},
        {"name": "@thread_id", "value": thread_id},
    ]

    if existing_summary:
        since = existing_summary["created_at"]
        query += " AND c.created_at > @since"
        parameters.append({"name": "@since", "value": since})

    items = list(container.query_items(
        query=query,
        parameters=parameters,
        enable_cross_partition_query=True,
    ))

    # If updating and there are no new memories, return the existing summary
    if existing_summary and not items:
        return existing_summary

    if not existing_summary and not items:
        raise ValueError(
            f"No memories found for user_id={user_id!r}, thread_id={thread_id!r}"
        )

    # ---- 3. Sort and trim ----
    items.sort(key=lambda m: m.get("created_at", ""), reverse=True)
    if recent_k is not None:
        items = items[:recent_k]
    items.reverse()  # chronological order

    # ---- 4. Build transcript ----
    transcript_lines = []
    for m in items:
        role = m.get("role", "unknown")
        content = m.get("content", "")
        metadata = m.get("metadata", {})
        meta_str = f" [metadata: {json.dumps(metadata)}]" if metadata else ""
        transcript_lines.append(f"[{role}]: {content}{meta_str}")
    transcript = "\n".join(transcript_lines)

    # ---- 5. Call LLM (full or incremental prompt) ----
    if existing_summary:
        prompt_file = "summarize_update.md"
        user_message = (
            f"## Existing Summary\n\n{existing_summary['content']}\n\n"
            f"## New Messages\n\n{transcript}"
        )
    else:
        prompt_file = "summarize.md"
        user_message = transcript

    prompt_path = os.path.join(os.path.dirname(__file__), "prompts", prompt_file)
    with open(prompt_path, "r", encoding="utf-8") as f:
        system_prompt = f.read()

    client = _get_chat_client()
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ],
    )
    summary_text = response.choices[0].message.content

    # ---- 6. Generate embedding ----
    embedding_model = os.environ.get("AI_FOUNDRY_EMBEDDING_MODEL", "text-embedding-3-large")
    dimensions = int(os.environ.get("EMBEDDING_DIMENSION", "1536"))
    emb_client = _get_embeddings_client()
    emb_response = emb_client.embeddings.create(
        input=[summary_text],
        model=embedding_model,
        dimensions=dimensions,
    )
    summary_embedding = emb_response.data[0].embedding

    # ---- 7. Upsert summary (deterministic ID, accumulate counts) ----
    if existing_summary:
        old_source_count = existing_summary.get("metadata", {}).get("source_count", 0)
        total_source_count = old_source_count + len(items)
    else:
        total_source_count = len(items)

    summary_doc = {
        "id": summary_id,
        "user_id": user_id,
        "thread_id": thread_id,
        "role": "system",
        "type": "summary",
        "content": summary_text,
        "embedding": summary_embedding,
        "metadata": {
            "source_count": total_source_count,
            "recent_k": recent_k,
            "incremental_update": existing_summary is not None,
        },
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    container.upsert_item(body=summary_doc)

    return summary_doc


# =====================================================================
# Activity: extract_facts
# =====================================================================

@bp.activity_trigger(input_name="payload")
def extract_facts(payload: dict) -> dict:
    """Extract facts from a user's thread memories using an AI Foundry LLM.

    Input::
        {
            "user_id": "...",
            "thread_id": "...",
            "recent_k": 10          # optional – keep only the most recent k
        }

    Steps:
      1. Query Cosmos DB for memories matching user_id + thread_id.
      2. Sort by created_at descending; if recent_k is set, keep only the
         most recent k documents.
      3. Extract content and metadata, send to the LLM for fact extraction.
      4. Insert the facts back into Cosmos DB as a memory of type "fact".
      5. Return the stored fact document.
    """
    user_id = payload["user_id"]
    thread_id = payload["thread_id"]
    recent_k = payload.get("recent_k")
    model = os.environ.get("AI_FOUNDRY_LLM", "gpt-5-nano")

    # ---- 1. Query Cosmos DB ----
    container = _get_cosmos_container()
    query = (
        "SELECT * FROM c "
        "WHERE c.user_id = @user_id AND c.thread_id = @thread_id"
    )
    parameters = [
        {"name": "@user_id", "value": user_id},
        {"name": "@thread_id", "value": thread_id},
    ]
    items = list(container.query_items(
        query=query,
        parameters=parameters,
        enable_cross_partition_query=True,
    ))

    # ---- 2. Sort and trim ----
    items.sort(key=lambda m: m.get("created_at", ""), reverse=True)
    if recent_k is not None:
        items = items[:recent_k]
    items.reverse()

    if not items:
        raise ValueError(
            f"No memories found for user_id={user_id!r}, thread_id={thread_id!r}"
        )

    # ---- 3. Build transcript and call LLM ----
    transcript_lines = []
    for m in items:
        role = m.get("role", "unknown")
        content = m.get("content", "")
        metadata = m.get("metadata", {})
        meta_str = f" [metadata: {json.dumps(metadata)}]" if metadata else ""
        transcript_lines.append(f"[{role}]: {content}{meta_str}")
    transcript = "\n".join(transcript_lines)

    prompt_path = os.path.join(os.path.dirname(__file__), "prompts", "facts.md")
    with open(prompt_path, "r", encoding="utf-8") as f:
        system_prompt = f.read()

    client = _get_chat_client()
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": transcript},
        ],
    )
    facts_text = response.choices[0].message.content

    # ---- 4. Parse individual facts (one per line) ----
    fact_lines = []
    for line in facts_text.splitlines():
        stripped = line.strip().lstrip("-").strip()
        if stripped:
            fact_lines.append(stripped)

    if not fact_lines:
        fact_lines = [facts_text.strip()]

    # ---- 5. Generate embeddings and store each fact ----
    embedding_model = os.environ.get("AI_FOUNDRY_EMBEDDING_MODEL", "text-embedding-3-large")
    dimensions = int(os.environ.get("EMBEDDING_DIMENSION", "1536"))
    emb_client = _get_embeddings_client()
    now = datetime.now(timezone.utc).isoformat()
    facts_docs = []

    for fact in fact_lines:
        emb_response = emb_client.embeddings.create(
            input=[fact],
            model=embedding_model,
            dimensions=dimensions,
        )
        fact_doc = {
            "id": str(uuid.uuid4()),
            "user_id": user_id,
            "thread_id": thread_id,
            "role": "system",
            "type": "fact",
            "content": fact,
            "embedding": emb_response.data[0].embedding,
            "metadata": {
                "source_count": len(items),
                "recent_k": recent_k,
            },
            "created_at": now,
        }
        container.upsert_item(body=fact_doc)
        facts_docs.append(fact_doc)

    return facts_docs


# =====================================================================
# Activity: generate_user_summary
# =====================================================================

@bp.activity_trigger(input_name="payload")
def generate_user_summary(payload: dict) -> dict:
    """Generate or incrementally update a cross-thread user summary.

    If a user summary already exists, only memories created *after* the
    existing summary are fetched. The LLM then receives the old profile
    together with the new conversation data and produces an updated
    profile. Thread IDs and memory counts are accumulated across runs.

    Input::
        {
            "user_id": "...",
            "thread_ids": ["..."],  # optional – limit to specific threads
            "recent_k": 10          # optional – per-thread recency limit
        }
    """
    from collections import defaultdict
    from azure.cosmos.exceptions import CosmosResourceNotFoundError

    user_id = payload["user_id"]
    thread_ids = payload.get("thread_ids")
    recent_k = payload.get("recent_k")
    model = os.environ.get("AI_FOUNDRY_LLM", "gpt-5-nano")
    container = _get_cosmos_container()

    # ---- 1. Check for an existing user summary ----
    existing_summary = None
    try:
        existing_summary = container.read_item(
            item=f"user_summary_{user_id}",
            partition_key=[user_id, "__user_summary__"],
        )
    except CosmosResourceNotFoundError:
        pass  # first time – full generation

    # ---- 2. Query memories (time-filtered if updating) ----
    query = (
        "SELECT * FROM c "
        "WHERE c.user_id = @user_id AND c.type != 'user_summary'"
    )
    parameters: list[dict] = [
        {"name": "@user_id", "value": user_id},
    ]

    if existing_summary:
        since = existing_summary["created_at"]
        query += " AND c.created_at > @since"
        parameters.append({"name": "@since", "value": since})

    if thread_ids:
        placeholders = ", ".join(
            f"@tid{i}" for i in range(len(thread_ids))
        )
        query += f" AND c.thread_id IN ({placeholders})"
        for i, tid in enumerate(thread_ids):
            parameters.append({"name": f"@tid{i}", "value": tid})

    items = list(container.query_items(
        query=query,
        parameters=parameters,
        enable_cross_partition_query=True,
    ))

    # If updating and there are no new memories, return the existing summary
    if existing_summary and not items:
        return existing_summary

    if not existing_summary and not items:
        raise ValueError(
            f"No memories found for user_id={user_id!r}"
        )

    # ---- 3. Sort and apply per-thread recent_k trimming ----
    items.sort(key=lambda m: m.get("created_at", ""), reverse=True)

    if recent_k is not None:
        by_thread: dict[str, list] = defaultdict(list)
        for m in items:
            by_thread[m.get("thread_id", "")].append(m)
        trimmed = []
        for thread_items in by_thread.values():
            trimmed.extend(thread_items[:recent_k])
        trimmed.sort(key=lambda m: m.get("created_at", ""))
        items = trimmed
    else:
        items.reverse()  # chronological order

    # ---- 4. Build transcript grouped by thread ----
    threads: dict[str, list] = defaultdict(list)
    for m in items:
        threads[m.get("thread_id", "")].append(m)

    transcript_parts = []
    for tid, thread_items in threads.items():
        transcript_parts.append(f"=== Thread {tid} ===")
        for m in thread_items:
            role = m.get("role", "unknown")
            content = m.get("content", "")
            metadata = m.get("metadata", {})
            meta_str = f" [metadata: {json.dumps(metadata)}]" if metadata else ""
            transcript_parts.append(f"[{role}]: {content}{meta_str}")
        transcript_parts.append("")
    transcript = "\n".join(transcript_parts)

    # ---- 5. Call LLM (full or incremental prompt) ----
    if existing_summary:
        prompt_file = "user_summary_update.md"
        user_message = (
            f"## Existing Profile\n\n{existing_summary['content']}\n\n"
            f"## New Conversations\n\n{transcript}"
        )
    else:
        prompt_file = "user_summary.md"
        user_message = transcript

    prompt_path = os.path.join(os.path.dirname(__file__), "prompts", prompt_file)
    with open(prompt_path, "r", encoding="utf-8") as f:
        system_prompt = f.read()

    client = _get_chat_client()
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ],
    )
    summary_text = response.choices[0].message.content

    # ---- 6. Generate embedding ----
    embedding_model = os.environ.get("AI_FOUNDRY_EMBEDDING_MODEL", "text-embedding-3-large")
    dimensions = int(os.environ.get("EMBEDDING_DIMENSION", "1536"))
    emb_client = _get_embeddings_client()
    emb_response = emb_client.embeddings.create(
        input=[summary_text],
        model=embedding_model,
        dimensions=dimensions,
    )
    summary_embedding = emb_response.data[0].embedding

    # ---- 7. Upsert user summary (accumulate thread IDs and counts) ----
    new_thread_ids = set(threads.keys())
    if existing_summary:
        old_thread_ids = set(existing_summary.get("metadata", {}).get("thread_ids", []))
        all_thread_ids = sorted(old_thread_ids | new_thread_ids)
        old_memory_count = existing_summary.get("metadata", {}).get("source_memory_count", 0)
        total_memory_count = old_memory_count + len(items)
    else:
        all_thread_ids = sorted(new_thread_ids)
        total_memory_count = len(items)

    summary_doc = {
        "id": f"user_summary_{user_id}",
        "user_id": user_id,
        "thread_id": "__user_summary__",
        "role": "system",
        "type": "user_summary",
        "content": summary_text,
        "embedding": summary_embedding,
        "metadata": {
            "source_thread_count": len(all_thread_ids),
            "source_memory_count": total_memory_count,
            "thread_ids": all_thread_ids,
            "recent_k": recent_k,
            "incremental_update": existing_summary is not None,
        },
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    container.upsert_item(body=summary_doc)

    return summary_doc
