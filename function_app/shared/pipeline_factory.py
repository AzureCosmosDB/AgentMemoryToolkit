"""Lazy ProcessingPipeline factory (MI auth, sync clients).

The activities reuse :class:`agent_memory_toolkit.pipeline.ProcessingPipeline`
verbatim — no business logic is duplicated in the function app.
"""

from __future__ import annotations

from typing import Any

from . import config
from .cosmos_clients import get_memories_container

_pipeline: Any | None = None


def get_pipeline():
    """Return the cached :class:`ProcessingPipeline` for this worker."""
    global _pipeline
    if _pipeline is not None:
        return _pipeline

    from azure.identity import DefaultAzureCredential

    from agent_memory_toolkit.embeddings import EmbeddingsClient
    from agent_memory_toolkit.chat import ChatClient
    from agent_memory_toolkit.pipeline import ProcessingPipeline

    credential = DefaultAzureCredential()
    container = get_memories_container()
    ai_endpoint = config.get_ai_foundry_endpoint()

    llm = ChatClient(
        endpoint=ai_endpoint,
        credential=credential,
        model=config.get_chat_deployment_name(),
    )
    embeddings = EmbeddingsClient(
        endpoint=ai_endpoint,
        credential=credential,
        model=config.get_embedding_deployment_name(),
    )

    _pipeline = ProcessingPipeline(container, llm, embeddings)
    return _pipeline
