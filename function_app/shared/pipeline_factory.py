"""Lazy PipelineService factory (MI auth, sync clients).

The activities reuse :class:`azure.cosmos.agent_memory.services.pipeline.PipelineService`
verbatim — no business logic is duplicated in the function app.
"""

from __future__ import annotations

import os
from typing import Any

from . import config
from .cosmos_clients import (
    get_memories_container,
    get_summaries_container,
    get_turns_container,
)

_pipeline: Any | None = None


def _read_transcript_metadata_keys() -> tuple[str, ...] | None:
    """Parse ``AGENT_MEMORY_TRANSCRIPT_METADATA_KEYS`` (comma-separated allow-list).

    Mirrors the ``transcript_metadata_keys`` ctor kwarg on
    :class:`CosmosMemoryClient` so the Durable-Functions backend produces
    the same prompt content as the in-process backend.
    """
    raw = os.environ.get("AGENT_MEMORY_TRANSCRIPT_METADATA_KEYS", "").strip()
    if not raw:
        return None
    keys = tuple(part.strip() for part in raw.split(",") if part.strip())
    return keys or None


def get_pipeline():
    """Return the cached :class:`PipelineService` for this worker."""
    global _pipeline
    if _pipeline is not None:
        return _pipeline

    from azure.identity import DefaultAzureCredential

    from azure.cosmos.agent_memory._container_routing import ContainerKey
    from azure.cosmos.agent_memory._utils import _resolve_embedding_dimensions
    from azure.cosmos.agent_memory.chat import ChatClient
    from azure.cosmos.agent_memory.embeddings import EmbeddingsClient
    from azure.cosmos.agent_memory.services.pipeline import PipelineService
    from azure.cosmos.agent_memory.store import MemoryStore

    credential = DefaultAzureCredential()
    memories_container = get_memories_container()
    turns_container = get_turns_container()
    summaries_container = get_summaries_container()
    ai_endpoint = config.get_ai_foundry_endpoint()

    embedding_dimensions = _resolve_embedding_dimensions(None)

    chat = ChatClient(
        endpoint=ai_endpoint,
        credential=credential,
        model=config.get_chat_deployment_name(),
    )
    embeddings = EmbeddingsClient(
        endpoint=ai_endpoint,
        credential=credential,
        model=config.get_embedding_deployment_name(),
        dimensions=embedding_dimensions,
    )

    containers = {
        ContainerKey.TURNS: turns_container,
        ContainerKey.MEMORIES: memories_container,
        ContainerKey.SUMMARIES: summaries_container,
    }
    store = MemoryStore(containers=containers, embeddings_client=embeddings)
    _pipeline = PipelineService(
        store,
        chat,
        embeddings,
        containers=containers,
        transcript_metadata_keys=_read_transcript_metadata_keys(),
    )
    return _pipeline
