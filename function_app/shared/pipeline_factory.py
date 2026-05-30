"""Lazy PipelineService factory (MI auth, sync clients).

The activities reuse :class:`agent_memory_toolkit.services.pipeline.PipelineService`
verbatim — no business logic is duplicated in the function app.
"""

from __future__ import annotations

from typing import Any

from . import config
from .cosmos_clients import (
    get_memories_container,
    get_summaries_container,
    get_turns_container,
)

_pipeline: Any | None = None


def get_pipeline():
    """Return the cached :class:`PipelineService` for this worker."""
    global _pipeline
    if _pipeline is not None:
        return _pipeline

    from azure.identity import DefaultAzureCredential

    from agent_memory_toolkit._container_routing import ContainerKey
    from agent_memory_toolkit._utils import _resolve_embedding_dimensions
    from agent_memory_toolkit.chat import ChatClient
    from agent_memory_toolkit.embeddings import EmbeddingsClient
    from agent_memory_toolkit.services.pipeline import PipelineService
    from agent_memory_toolkit.store import MemoryStore

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
    _pipeline = PipelineService(store, chat, embeddings, containers=containers)
    return _pipeline
