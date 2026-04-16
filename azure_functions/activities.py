"""Activity functions for the Agent Memory durable orchestration.

Provides Azure Durable Functions activities for memory processing:
load, embed, store, summarize threads, extract facts, and build user profiles.
"""

import json
import logging
import os
import time
import uuid
from collections import defaultdict
from datetime import datetime, timezone

import azure.durable_functions as df
from azure.cosmos.aio import CosmosClient
from azure.cosmos.exceptions import CosmosHttpResponseError, CosmosResourceNotFoundError
from azure.identity.aio import DefaultAzureCredential

logger = logging.getLogger(__name__)

bp = df.Blueprint()


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _get_llm_model() -> str:
    """Return the configured LLM model name.

    Checks ``LLM_MODEL`` first, then ``AI_FOUNDRY_LLM``, then defaults
    to ``"gpt-4o"``.
    """
    return os.environ.get("LLM_MODEL") or os.environ.get("AI_FOUNDRY_LLM") or "gpt-4o"


def _get_embedding_model() -> str:
    return (
        os.environ.get("AI_FOUNDRY_EMBEDDING_MODEL")
        or os.environ.get("EMBEDDING_MODEL")
        or "text-embedding-3-large"
    )


def _get_embedding_dimensions() -> int:
    return int(os.environ.get("EMBEDDING_DIMENSION", "1536"))


def _build_transcript(items: list[dict], *, group_by_thread: bool = False) -> str:
    """Build a formatted transcript from memory documents.

    Args:
        items: Memory dicts with ``role``, ``content``, and optional ``metadata``.
        group_by_thread: If *True*, group messages under ``=== Thread <id> ===``
            headers.  Otherwise produce a flat list.

    Returns:
        A newline-joined transcript string.
    """
    if not group_by_thread:
        lines: list[str] = []
        for m in items:
            role = m.get("role", "unknown")
            content = m.get("content", "")
            metadata = m.get("metadata", {})
            meta_str = f" [metadata: {json.dumps(metadata)}]" if metadata else ""
            lines.append(f"[{role}]: {content}{meta_str}")
        return "\n".join(lines)

    threads: dict[str, list[dict]] = defaultdict(list)
    for m in items:
        threads[m.get("thread_id", "")].append(m)

    parts: list[str] = []
    for tid, thread_items in threads.items():
        parts.append(f"=== Thread {tid} ===")
        for m in thread_items:
            role = m.get("role", "unknown")
            content = m.get("content", "")
            metadata = m.get("metadata", {})
            meta_str = f" [metadata: {json.dumps(metadata)}]" if metadata else ""
            parts.append(f"[{role}]: {content}{meta_str}")
        parts.append("")
    return "\n".join(parts)


def _validate_required(payload: dict, *keys: str, activity: str = "") -> None:
    """Raise ``ValueError`` if any *keys* are missing or ``None`` in *payload*."""
    missing = [k for k in keys if payload.get(k) is None]
    if missing:
        raise ValueError(
            f"{activity + ': ' if activity else ''}missing required field(s): {', '.join(missing)}"
        )


def _load_prompt(filename: str) -> str:
    """Read a prompt file from the ``prompts/`` directory."""
    path = os.path.join(os.path.dirname(__file__), "prompts", filename)
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


# ---------------------------------------------------------------------------
# Lazily initialised singletons
# ---------------------------------------------------------------------------

_cosmos_client = None
_cosmos_database_client = None
_cosmos_container = None
_cosmos_counter_container = None
_credential = None
_openai_client = None


def _get_credential():
    """Return a shared ``DefaultAzureCredential``."""
    global _credential
    if _credential is None:
        _credential = DefaultAzureCredential()
    return _credential


async def _get_cosmos_database_client():
    """Return the Cosmos DB database client, connecting on first call."""
    global _cosmos_client, _cosmos_database_client
    if _cosmos_database_client is None:
        endpoint = os.environ.get("COSMOS_DB__accountEndpoint") or os.environ.get("COSMOS_DB_ENDPOINT")
        if not endpoint:
            raise ValueError(
                "Cosmos DB endpoint not configured. "
                "Set COSMOS_DB__accountEndpoint (required for change feed trigger) "
                "or COSMOS_DB_ENDPOINT."
            )
        database = os.environ["COSMOS_DB_DATABASE"]
        logger.info(
            "Connecting to Cosmos DB endpoint=%s database=%s",
            f"...{endpoint[-8:]}", database,
        )
        _cosmos_client = CosmosClient(endpoint, credential=_get_credential())
        _cosmos_database_client = _cosmos_client.get_database_client(database)
    return _cosmos_database_client


async def _get_cosmos_container():
    """Return the main memories container client, connecting on first call."""
    global _cosmos_container
    if _cosmos_container is None:
        container = os.environ["COSMOS_DB_CONTAINER"]
        logger.info("Using Cosmos memories container=%s", container)
        db_client = await _get_cosmos_database_client()
        _cosmos_container = db_client.get_container_client(container)
    return _cosmos_container


async def _get_cosmos_counter_container():
    """Return the dedicated counter container client, connecting on first call."""
    global _cosmos_counter_container
    if _cosmos_counter_container is None:
        container = os.environ.get("COSMOS_DB_COUNTERS_CONTAINER", "counter")
        logger.info("Using Cosmos counter container=%s", container)
        db_client = await _get_cosmos_database_client()
        _cosmos_counter_container = db_client.get_container_client(container)
    return _cosmos_counter_container


def _get_openai_client():
    """Return a cached ``AzureOpenAI`` client (used for both embeddings and chat)."""
    global _openai_client
    if _openai_client is None:
        from openai import AzureOpenAI

        endpoint = os.environ["AI_FOUNDRY_ENDPOINT"]
        api_key = os.environ.get("AI_FOUNDRY_API_KEY")
        api_version = os.environ.get("AI_FOUNDRY_API_VERSION", "2024-12-01-preview")

        logger.info(
            "Initializing AzureOpenAI client endpoint=%s auth=%s",
            f"...{endpoint[-8:]}", "api_key" if api_key else "entra_id",
        )

        if api_key:
            _openai_client = AzureOpenAI(
                api_version=api_version,
                azure_endpoint=endpoint,
                api_key=api_key,
            )
        else:
            from azure.identity import DefaultAzureCredential as SyncDefaultAzureCredential
            from azure.identity import get_bearer_token_provider

            sync_credential = SyncDefaultAzureCredential()
            token_provider = get_bearer_token_provider(
                sync_credential,
                "https://cognitiveservices.azure.com/.default",
            )
            _openai_client = AzureOpenAI(
                api_version=api_version,
                azure_endpoint=endpoint,
                azure_ad_token_provider=token_provider,
            )
    return _openai_client


# ---------------------------------------------------------------------------
# Resilient LLM / embedding wrappers
# ---------------------------------------------------------------------------

_RETRYABLE_STATUS_CODES = (429, 500, 503)


def _call_llm_with_retry(
    client,
    model: str,
    messages: list[dict],
    *,
    max_retries: int = 3,
    base_delay: float = 2.0,
):
    """Call chat completions with exponential backoff for transient errors."""
    import openai

    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(model=model, messages=messages)
            usage = getattr(response, "usage", None)
            if usage:
                logger.info(
                    "LLM response model=%s prompt_tokens=%s completion_tokens=%s total_tokens=%s",
                    model, usage.prompt_tokens, usage.completion_tokens, usage.total_tokens,
                )
            return response
        except openai.RateLimitError as exc:
            if attempt < max_retries - 1:
                delay = base_delay * (2 ** attempt)
                logger.warning(
                    "LLM rate-limited (attempt %d/%d), retrying in %.1fs: %s",
                    attempt + 1, max_retries, delay, exc,
                )
                time.sleep(delay)
                continue
            logger.warning("LLM rate-limited after %d attempts, re-raising", max_retries)
            raise
        except openai.APIError as exc:
            status = getattr(exc, "status_code", None)
            if status in _RETRYABLE_STATUS_CODES and attempt < max_retries - 1:
                delay = base_delay * (2 ** attempt)
                logger.warning(
                    "LLM API error %s (attempt %d/%d), retrying in %.1fs: %s",
                    status, attempt + 1, max_retries, delay, exc,
                )
                time.sleep(delay)
                continue
            logger.error("LLM API error (status=%s): %s", status, exc, exc_info=True)
            raise
        except Exception as exc:
            logger.error("LLM call failed unexpectedly: %s", exc, exc_info=True)
            raise


def _generate_embedding(
    text: str,
    *,
    max_retries: int = 3,
    base_delay: float = 2.0,
) -> list[float]:
    """Generate an embedding for *text* with retry logic.

    Uses the shared ``AzureOpenAI`` client, embedding model, and
    dimensions from environment variables.  Retries on rate-limit and
    transient API errors with exponential backoff.
    """
    import openai

    model = _get_embedding_model()
    dimensions = _get_embedding_dimensions()
    client = _get_openai_client()

    for attempt in range(max_retries):
        try:
            response = client.embeddings.create(
                input=[text], model=model, dimensions=dimensions,
            )
            return response.data[0].embedding
        except openai.RateLimitError as exc:
            if attempt < max_retries - 1:
                delay = base_delay * (2 ** attempt)
                logger.warning(
                    "Embedding rate-limited (attempt %d/%d), retrying in %.1fs: %s",
                    attempt + 1, max_retries, delay, exc,
                )
                time.sleep(delay)
                continue
            logger.warning("Embedding rate-limited after %d attempts, re-raising", max_retries)
            raise
        except openai.APIError as exc:
            status = getattr(exc, "status_code", None)
            if status in _RETRYABLE_STATUS_CODES and attempt < max_retries - 1:
                delay = base_delay * (2 ** attempt)
                logger.warning(
                    "Embedding API error %s (attempt %d/%d), retrying in %.1fs: %s",
                    status, attempt + 1, max_retries, delay, exc,
                )
                time.sleep(delay)
                continue
            logger.error("Embedding API error (status=%s): %s", status, exc, exc_info=True)
            raise
        except Exception as exc:
            logger.error("Embedding call failed model=%s: %s", model, exc, exc_info=True)
            raise

    # Should not be reached, but satisfy type checkers
    raise RuntimeError("Embedding generation failed after all retries")


def _generate_embeddings_batch(
    texts: list[str],
    *,
    max_retries: int = 3,
    base_delay: float = 2.0,
) -> list[list[float]]:
    """Generate embeddings for multiple texts in a single API call.

    Uses the OpenAI batch embedding API to reduce latency and rate-limit
    pressure compared to sequential per-text calls.
    """
    import openai

    if not texts:
        return []

    model = _get_embedding_model()
    dimensions = _get_embedding_dimensions()
    client = _get_openai_client()

    for attempt in range(max_retries):
        try:
            response = client.embeddings.create(
                input=texts, model=model, dimensions=dimensions,
            )
            # Sort by index to preserve input order
            sorted_data = sorted(response.data, key=lambda d: d.index)
            return [d.embedding for d in sorted_data]
        except openai.RateLimitError as exc:
            if attempt < max_retries - 1:
                delay = base_delay * (2 ** attempt)
                logger.warning(
                    "Batch embedding rate-limited (attempt %d/%d), retrying in %.1fs: %s",
                    attempt + 1, max_retries, delay, exc,
                )
                time.sleep(delay)
                continue
            logger.warning("Batch embedding rate-limited after %d attempts, re-raising", max_retries)
            raise
        except openai.APIError as exc:
            status = getattr(exc, "status_code", None)
            if status in _RETRYABLE_STATUS_CODES and attempt < max_retries - 1:
                delay = base_delay * (2 ** attempt)
                logger.warning(
                    "Batch embedding API error %s (attempt %d/%d), retrying in %.1fs: %s",
                    status, attempt + 1, max_retries, delay, exc,
                )
                time.sleep(delay)
                continue
            logger.error("Batch embedding API error (status=%s): %s", status, exc, exc_info=True)
            raise
        except Exception as exc:
            logger.error("Batch embedding call failed model=%s: %s", model, exc, exc_info=True)
            raise

    raise RuntimeError("Batch embedding generation failed after all retries")


# =====================================================================
# Activity: load_memories
# =====================================================================


@bp.activity_trigger(input_name="payload")
async def load_memories(payload: dict) -> list:
    """Load all memories for a given thread_id from Cosmos DB.

    Input::
        {"thread_id": "..."}

    Returns a list of memory dicts.
    """
    _validate_required(payload, "thread_id", activity="load_memories")
    thread_id = payload["thread_id"]
    logger.info("load_memories started thread_id=%s", thread_id)

    try:
        container = await _get_cosmos_container()
        query = "SELECT * FROM c WHERE c.thread_id = @thread_id"
        parameters = [{"name": "@thread_id", "value": thread_id}]

        results = [item async for item in container.query_items(
            query=query,
            parameters=parameters,
        )]
    except CosmosHttpResponseError as exc:
        logger.error("load_memories Cosmos query failed: %s", exc, exc_info=True)
        raise

    if not results:
        logger.warning("load_memories returned 0 results thread_id=%s", thread_id)
    else:
        logger.info("load_memories completed count=%d thread_id=%s", len(results), thread_id)
    return results


# =====================================================================
# Activity: generate_embeddings
# =====================================================================


@bp.activity_trigger(input_name="payload")
async def generate_embeddings(payload: dict) -> list:
    """Generate a vector embedding for the given text.

    Input::
        {"text": "some content to embed"}

    Returns a list of floats (the embedding vector).
    """
    _validate_required(payload, "text", activity="generate_embeddings")
    text = payload["text"]

    model = _get_embedding_model()
    dimensions = _get_embedding_dimensions()
    logger.info("generate_embeddings started model=%s dimensions=%d", model, dimensions)

    embedding = _generate_embedding(text)
    logger.info("generate_embeddings completed vector_length=%d", len(embedding))
    return embedding


# =====================================================================
# Activity: store_results
# =====================================================================


@bp.activity_trigger(input_name="payload")
async def store_results(payload: dict) -> dict:
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
    _validate_required(payload, "user_id", "thread_id", "content", "embedding", activity="store_results")
    logger.info("store_results started input_keys=%s", list(payload.keys()))

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

    try:
        container = await _get_cosmos_container()
        await container.upsert_item(body=doc)
    except CosmosHttpResponseError as exc:
        logger.error("store_results upsert failed id=%s: %s", doc["id"], exc, exc_info=True)
        raise

    logger.info("store_results upserted id=%s type=%s", doc["id"], doc["type"])
    return doc


# =====================================================================
# Activity: generate_thread_summary
# =====================================================================


@bp.activity_trigger(input_name="payload")
async def generate_thread_summary(payload: dict) -> dict:
    """Generate or incrementally update a thread summary using an LLM.

    If a summary already exists for the thread, only memories created
    *after* the existing summary are fetched.  The LLM then receives
    the old summary together with the new messages and produces an
    updated summary.  The document is upserted with a deterministic ID
    so there is at most one active summary per thread.

    Input::
        {
            "user_id": "...",
            "thread_id": "...",
            "recent_k": 10          # optional -- per-thread recency limit
        }
    """
    _validate_required(payload, "user_id", "thread_id", activity="generate_thread_summary")

    user_id = payload["user_id"]
    thread_id = payload["thread_id"]
    recent_k = payload.get("recent_k")
    model = _get_llm_model()
    logger.info(
        "generate_thread_summary started user_id=%s thread_id=%s model=%s",
        user_id, thread_id, model,
    )

    try:
        container = await _get_cosmos_container()
    except CosmosHttpResponseError as exc:
        logger.error("generate_thread_summary Cosmos connection failed: %s", exc, exc_info=True)
        raise

    # ---- 1. Check for an existing thread summary ----
    existing_summary = None
    summary_id = f"summary_{user_id}_{thread_id}"
    try:
        existing_summary = await container.read_item(
            item=summary_id,
            partition_key=[user_id, thread_id],
        )
    except CosmosResourceNotFoundError:
        pass  # first time -- full generation
    except CosmosHttpResponseError as exc:
        logger.error("generate_thread_summary read existing summary failed: %s", exc, exc_info=True)
        raise

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

    try:
        items = [item async for item in container.query_items(
            query=query,
            parameters=parameters,
        )]
    except CosmosHttpResponseError as exc:
        logger.error("generate_thread_summary query failed: %s", exc, exc_info=True)
        raise

    logger.debug("generate_thread_summary query returned %d memories", len(items))

    if existing_summary and not items:
        logger.info("generate_thread_summary no new memories, returning existing summary")
        return existing_summary

    if not existing_summary and not items:
        logger.warning("generate_thread_summary no memories found user_id=%s thread_id=%s", user_id, thread_id)
        raise ValueError(
            f"No memories found for user_id={user_id!r}, thread_id={thread_id!r}"
        )

    # ---- 3. Sort and trim ----
    items.sort(key=lambda m: m.get("created_at", ""), reverse=True)
    if recent_k is not None:
        items = items[:recent_k]
    items.reverse()  # chronological order

    # ---- 4. Build transcript ----
    transcript = _build_transcript(items)

    # ---- 5. Call LLM (full or incremental prompt) ----
    if existing_summary:
        system_prompt = _load_prompt("summarize_update.md")
        user_message = (
            f"## Existing Summary\n\n{existing_summary['content']}\n\n"
            f"## New Messages\n\n{transcript}"
        )
    else:
        system_prompt = _load_prompt("summarize.md")
        user_message = transcript

    logger.info("generate_thread_summary calling LLM model=%s", model)
    client = _get_openai_client()
    response = _call_llm_with_retry(
        client, model,
        [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ],
    )
    summary_text = response.choices[0].message.content

    # ---- 6. Generate embedding ----
    logger.info("generate_thread_summary generating embedding")
    summary_embedding = _generate_embedding(summary_text)

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

    try:
        await container.upsert_item(body=summary_doc)
    except CosmosHttpResponseError as exc:
        logger.error("generate_thread_summary upsert failed: %s", exc, exc_info=True)
        raise

    logger.info("generate_thread_summary completed id=%s source_count=%d", summary_id, total_source_count)
    return summary_doc


# =====================================================================
# Activity: extract_facts
# =====================================================================


@bp.activity_trigger(input_name="payload")
async def extract_facts(payload: dict) -> dict:
    """Extract facts from a user's thread memories using an LLM.

    Input::
        {
            "user_id": "...",
            "thread_id": "...",
            "recent_k": 10          # optional -- keep only the most recent k
        }

    Steps:
      1. Query Cosmos DB for memories matching user_id + thread_id.
      2. Sort by created_at descending; if recent_k is set, keep only the
         most recent k documents.
      3. Extract content and metadata, send to the LLM for fact extraction.
      4. Insert the facts back into Cosmos DB as a memory of type ``"fact"``.
      5. Return the stored fact documents.
    """
    _validate_required(payload, "user_id", "thread_id", activity="extract_facts")

    user_id = payload["user_id"]
    thread_id = payload["thread_id"]
    recent_k = payload.get("recent_k")
    model = _get_llm_model()
    logger.info("extract_facts started user_id=%s thread_id=%s model=%s", user_id, thread_id, model)

    # ---- 1. Query Cosmos DB ----
    try:
        container = await _get_cosmos_container()
        query = (
            "SELECT * FROM c "
            "WHERE c.user_id = @user_id AND c.thread_id = @thread_id"
        )
        parameters = [
            {"name": "@user_id", "value": user_id},
            {"name": "@thread_id", "value": thread_id},
        ]
        items = [item async for item in container.query_items(
            query=query,
            parameters=parameters,
        )]
    except CosmosHttpResponseError as exc:
        logger.error("extract_facts Cosmos query failed: %s", exc, exc_info=True)
        raise

    # ---- 2. Sort and trim ----
    items.sort(key=lambda m: m.get("created_at", ""), reverse=True)
    if recent_k is not None:
        items = items[:recent_k]
    items.reverse()

    if not items:
        logger.warning("extract_facts no memories found user_id=%s thread_id=%s", user_id, thread_id)
        raise ValueError(
            f"No memories found for user_id={user_id!r}, thread_id={thread_id!r}"
        )

    # ---- 3. Build transcript and call LLM ----
    transcript = _build_transcript(items)
    system_prompt = _load_prompt("facts.md")

    logger.info("extract_facts calling LLM model=%s", model)
    client = _get_openai_client()
    response = _call_llm_with_retry(
        client, model,
        [
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

    # ---- 5. Generate embeddings in batch and store each fact ----
    logger.info("extract_facts generating embeddings for %d facts", len(fact_lines))
    now = datetime.now(timezone.utc).isoformat()
    fact_embeddings = _generate_embeddings_batch(fact_lines)
    facts_docs = []

    for fact, fact_embedding in zip(fact_lines, fact_embeddings):

        fact_doc = {
            "id": str(uuid.uuid4()),
            "user_id": user_id,
            "thread_id": thread_id,
            "role": "system",
            "type": "fact",
            "content": fact,
            "embedding": fact_embedding,
            "metadata": {
                "source_count": len(items),
                "recent_k": recent_k,
            },
            "created_at": now,
        }

        try:
            await container.upsert_item(body=fact_doc)
        except CosmosHttpResponseError as exc:
            logger.error("extract_facts upsert failed fact_id=%s: %s", fact_doc["id"], exc, exc_info=True)
            raise

        facts_docs.append(fact_doc)

    logger.info("extract_facts completed facts_count=%d", len(facts_docs))
    return facts_docs


# =====================================================================
# Activity: generate_user_summary
# =====================================================================


@bp.activity_trigger(input_name="payload")
async def generate_user_summary(payload: dict) -> dict:
    """Generate or incrementally update a cross-thread user summary.

    If a user summary already exists, only memories created *after* the
    existing summary are fetched.  The LLM then receives the old profile
    together with the new conversation data and produces an updated
    profile.  Thread IDs and memory counts are accumulated across runs.

    Input::
        {
            "user_id": "...",
            "thread_ids": ["..."],  # optional -- limit to specific threads
            "recent_k": 10          # optional -- per-thread recency limit
        }
    """
    _validate_required(payload, "user_id", activity="generate_user_summary")

    user_id = payload["user_id"]
    thread_ids = payload.get("thread_ids")
    recent_k = payload.get("recent_k")
    model = _get_llm_model()
    logger.info("generate_user_summary started user_id=%s model=%s", user_id, model)

    try:
        container = await _get_cosmos_container()
    except CosmosHttpResponseError as exc:
        logger.error("generate_user_summary Cosmos connection failed: %s", exc, exc_info=True)
        raise

    # ---- 1. Check for an existing user summary ----
    existing_summary = None
    try:
        existing_summary = await container.read_item(
            item=f"user_summary_{user_id}",
            partition_key=[user_id, "__user_summary__"],
        )
    except CosmosResourceNotFoundError:
        pass  # first time -- full generation
    except CosmosHttpResponseError as exc:
        logger.error("generate_user_summary read existing summary failed: %s", exc, exc_info=True)
        raise

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
        placeholders = ", ".join(f"@tid{i}" for i in range(len(thread_ids)))
        query += f" AND c.thread_id IN ({placeholders})"
        for i, tid in enumerate(thread_ids):
            parameters.append({"name": f"@tid{i}", "value": tid})

    try:
        items = [item async for item in container.query_items(
            query=query,
            parameters=parameters,
        )]
    except CosmosHttpResponseError as exc:
        logger.error("generate_user_summary query failed: %s", exc, exc_info=True)
        raise

    logger.debug("generate_user_summary query returned %d memories", len(items))

    if existing_summary and not items:
        logger.info("generate_user_summary no new memories, returning existing summary")
        return existing_summary

    if not existing_summary and not items:
        logger.warning("generate_user_summary no memories found user_id=%s", user_id)
        raise ValueError(f"No memories found for user_id={user_id!r}")

    # ---- 3. Sort and apply per-thread recent_k trimming ----
    items.sort(key=lambda m: m.get("created_at", ""), reverse=True)

    if recent_k is not None:
        by_thread: dict[str, list] = defaultdict(list)
        for m in items:
            by_thread[m.get("thread_id", "")].append(m)
        trimmed: list[dict] = []
        for thread_items in by_thread.values():
            trimmed.extend(thread_items[:recent_k])
        trimmed.sort(key=lambda m: m.get("created_at", ""))
        items = trimmed
    else:
        items.reverse()  # chronological order

    # ---- 4. Build transcript grouped by thread ----
    transcript = _build_transcript(items, group_by_thread=True)

    # Collect thread IDs from items for metadata
    new_thread_ids = {m.get("thread_id", "") for m in items}

    # ---- 5. Call LLM (full or incremental prompt) ----
    if existing_summary:
        system_prompt = _load_prompt("user_summary_update.md")
        user_message = (
            f"## Existing Profile\n\n{existing_summary['content']}\n\n"
            f"## New Conversations\n\n{transcript}"
        )
    else:
        system_prompt = _load_prompt("user_summary.md")
        user_message = transcript

    logger.info("generate_user_summary calling LLM model=%s", model)
    client = _get_openai_client()
    response = _call_llm_with_retry(
        client, model,
        [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ],
    )
    summary_text = response.choices[0].message.content

    # ---- 6. Generate embedding ----
    logger.info("generate_user_summary generating embedding")
    summary_embedding = _generate_embedding(summary_text)

    # ---- 7. Upsert user summary (accumulate thread IDs and counts) ----
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

    try:
        await container.upsert_item(body=summary_doc)
    except CosmosHttpResponseError as exc:
        logger.error("generate_user_summary upsert failed: %s", exc, exc_info=True)
        raise

    logger.info(
        "generate_user_summary completed thread_count=%d memory_count=%d",
        len(all_thread_ids), total_memory_count,
    )
    return summary_doc
