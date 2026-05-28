"""Async embedding client for the Agent Memory Toolkit.

Provides :class:`AsyncEmbeddingsClient` that lazily initialises an
``openai.AsyncAzureOpenAI`` connection and generates embeddings via the
OpenAI API.
"""

from __future__ import annotations

import asyncio
from typing import Any

from agent_memory_toolkit.chat import resolve_api_version
from agent_memory_toolkit.exceptions import ConfigurationError
from agent_memory_toolkit.logging import get_logger

logger = get_logger(__name__)

_TOKEN_SCOPE = "https://cognitiveservices.azure.com/.default"

AOAI_EMBEDDING_BATCH_SIZE = 16


class AsyncEmbeddingsClient:
    """Async embedding client backed by Azure OpenAI.

    Supports the async context-manager protocol::

        async with AsyncEmbeddingsClient(endpoint=..., credential=cred) as client:
            vec = await client.generate("hello")
    """

    def __init__(
        self,
        endpoint: str | None = None,
        credential: Any = None,
        api_key: str | None = None,
        model: str = "text-embedding-3-large",
        dimensions: int | None = None,
        api_version: str | None = None,
    ) -> None:
        self._endpoint = endpoint
        self._credential = credential
        self._api_key = api_key
        self._model = model
        self._dimensions = dimensions
        self._api_version = resolve_api_version(api_version)
        self._client: Any = None

    async def __aenter__(self) -> AsyncEmbeddingsClient:
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        await self.close()

    async def close(self) -> None:
        """Close the underlying HTTP client, if one has been created."""
        if self._client is not None:
            await self._client.close()
            self._client = None

    def _ensure_client(self) -> Any:
        """Lazily create the ``AsyncAzureOpenAI`` client on first use."""
        if self._client is not None:
            return self._client

        if not self._endpoint:
            raise ConfigurationError("An embedding endpoint is required", parameter="endpoint")

        from openai import AsyncAzureOpenAI

        if self._api_key:
            self._client = AsyncAzureOpenAI(
                api_version=self._api_version,
                azure_endpoint=self._endpoint,
                api_key=self._api_key,
            )
        else:
            if self._credential is None:
                raise ConfigurationError(
                    "Either api_key or a TokenCredential is required for embeddings",
                    parameter="credential",
                )
            from azure.identity.aio import get_bearer_token_provider

            token_provider = get_bearer_token_provider(self._credential, _TOKEN_SCOPE)
            self._client = AsyncAzureOpenAI(
                api_version=self._api_version,
                azure_endpoint=self._endpoint,
                azure_ad_token_provider=token_provider,
            )

        return self._client

    def _build_kwargs(self, input_: str | list[str]) -> dict[str, Any]:
        texts = [input_] if isinstance(input_, str) else input_
        logger.debug(
            "Embedding request: model=%s, dimensions=%s, texts=%d",
            self._model,
            self._dimensions,
            len(texts),
        )
        kwargs: dict[str, Any] = {"input": texts, "model": self._model}
        if self._dimensions:
            kwargs["dimensions"] = self._dimensions
        return kwargs

    async def generate(self, text: str) -> list[float]:
        """Generate an embedding vector for *text*.

        Raises
        ------
        ConfigurationError
            If the endpoint or credentials are missing.
        openai.OpenAIError
            Propagated from the SDK on API failure.
        """
        client = self._ensure_client()
        kwargs = self._build_kwargs(text)
        response = await client.embeddings.create(**kwargs)
        return response.data[0].embedding

    async def generate_batch(
        self,
        texts: list[str],
        *,
        batch_size: int = AOAI_EMBEDDING_BATCH_SIZE,
    ) -> list[list[float]]:
        """Generate embeddings for multiple texts.

        Returns a list of embedding vectors **in the same order** as *texts*.

        Parameters
        ----------
        texts:
            Texts to embed. An empty list returns ``[]`` with no API call.
        batch_size:
            Maximum number of inputs per ``embeddings.create()`` call.
            Defaults to :data:`AOAI_EMBEDDING_BATCH_SIZE` (16) to stay under
            the observed AOAI per-request input cap.

        Raises
        ------
        ConfigurationError
            If the endpoint or credentials are missing.
        openai.OpenAIError
            Propagated from the SDK on API failure (no retry — see module
            docstring).
        """
        if not texts:
            return []

        if batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {batch_size}")

        logger.info(
            "Generating embeddings for batch of %d texts (batch_size=%d)",
            len(texts),
            batch_size,
        )
        client = self._ensure_client()

        async def _run_chunk(chunk: list[str]) -> list[list[float]]:
            kwargs = self._build_kwargs(chunk)
            response = await client.embeddings.create(**kwargs)
            sorted_data = sorted(response.data, key=lambda d: d.index)
            return [item.embedding for item in sorted_data]

        chunks = [texts[start : start + batch_size] for start in range(0, len(texts), batch_size)]
        chunk_results = await asyncio.gather(*(_run_chunk(chunk) for chunk in chunks))
        return [emb for chunk_result in chunk_results for emb in chunk_result]
