"""Lazy PipelineService factory (MI auth, sync clients).

The activities reuse :class:`azure.cosmos.agent_memory.services.pipeline.PipelineService`
verbatim - no business logic is duplicated in the function app. A pipeline is bound to one tenant, so the
factory keeps one per tenant; they share the containers and the chat and embeddings clients.
"""

from __future__ import annotations

import os
from typing import Any

from azure.cosmos.agent_memory._partitioning import DEFAULT_TENANT_ID, resolve_tenant_id

from . import config
from .cosmos_clients import (
    get_memories_container,
    get_summaries_container,
    get_turns_container,
)

_pipelines: dict[str, Any] = {}
# (chat client, embeddings client, containers), built once and shared by every tenant's pipeline.
_shared: tuple[Any, Any, dict] | None = None


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


def _shared_clients() -> tuple[Any, Any, dict]:
    """Return the chat client, embeddings client and containers shared by every tenant's pipeline."""
    global _shared
    if _shared is not None:
        return _shared

    from azure.identity import DefaultAzureCredential

    from azure.cosmos.agent_memory._container_routing import ContainerKey
    from azure.cosmos.agent_memory._utils import _resolve_embedding_dimensions
    from azure.cosmos.agent_memory.chat import ChatClient
    from azure.cosmos.agent_memory.embeddings import EmbeddingsClient

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
    _shared = (chat, embeddings, containers)
    return _shared


def get_pipeline(tenant_id: str = DEFAULT_TENANT_ID):
    """Return the cached :class:`PipelineService` of ``tenant_id`` for this worker."""
    tenant_id = resolve_tenant_id(tenant_id)
    pipeline = _pipelines.get(tenant_id)
    if pipeline is not None:
        return pipeline

    from azure.cosmos.agent_memory.services.pipeline import PipelineService
    from azure.cosmos.agent_memory.store import MemoryStore

    chat, embeddings, containers = _shared_clients()
    store = MemoryStore(
        containers=containers,
        embeddings_client=embeddings,
        tenant_id=tenant_id,
    )
    pipeline = PipelineService(
        store,
        chat,
        embeddings,
        containers=containers,
        transcript_metadata_keys=_read_transcript_metadata_keys(),
        tenant_id=tenant_id,
    )
    _pipelines[tenant_id] = pipeline
    return pipeline
