"""Async embedding client for the Agent Memory Toolkit.

Provides :class:`AsyncEmbeddingsClient` that lazily initialises an
``openai.AsyncAzureOpenAI`` connection and generates embeddings via the
OpenAI API.
"""

from __future__ import annotations

import logging
from typing import Any

from agent_memory_toolkit.exceptions import ConfigurationError, EmbeddingError

logger = logging.getLogger(__name__)

_TOKEN_SCOPE = "https://cognitiveservices.azure.com/.default"


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
        api_version: str = "2024-12-01-preview",
    ) -> None:
        self._endpoint = endpoint
        self._credential = credential
        self._api_key = api_key
        self._model = model
        self._dimensions = dimensions
        self._api_version = api_version
        self._client: Any = None  # openai.AsyncAzureOpenAI (lazy)

    # -- async context manager ----------------------------------------------

    async def __aenter__(self) -> AsyncEmbeddingsClient:
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        await self.close()

    async def close(self) -> None:
        """Close the underlying HTTP client, if one has been created."""
        if self._client is not None:
            await self._client.close()
            self._client = None

    # -- internal helpers ---------------------------------------------------

    def _ensure_client(self) -> Any:
        """Lazily create the ``AsyncAzureOpenAI`` client on first use."""
        if self._client is not None:
            return self._client

        if not self._endpoint:
            raise ConfigurationError(
                "An embedding endpoint is required", parameter="endpoint"
            )

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

            token_provider = get_bearer_token_provider(
                self._credential, _TOKEN_SCOPE
            )
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

    # -- public API ---------------------------------------------------------

    async def generate(self, text: str) -> list[float]:
        """Generate an embedding vector for *text*.

        Raises
        ------
        ConfigurationError
            If the endpoint or credentials are missing.
        EmbeddingError
            If the OpenAI API call fails.
        """
        client = self._ensure_client()
        kwargs = self._build_kwargs(text)
        try:
            response = await client.embeddings.create(**kwargs)
        except Exception as exc:
            raise EmbeddingError(f"Embedding generation failed: {exc}") from exc
        return response.data[0].embedding

    async def generate_batch(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for multiple texts in a single API call.

        Returns a list of embedding vectors **in the same order** as *texts*.

        Raises
        ------
        ConfigurationError
            If the endpoint or credentials are missing.
        EmbeddingError
            If the OpenAI API call fails.
        """
        if not texts:
            return []

        logger.info("Generating embeddings for batch of %d texts", len(texts))
        client = self._ensure_client()
        kwargs = self._build_kwargs(texts)
        try:
            response = await client.embeddings.create(**kwargs)
        except Exception as exc:
            raise EmbeddingError(
                f"Batch embedding generation failed: {exc}"
            ) from exc

        sorted_data = sorted(response.data, key=lambda d: d.index)
        return [item.embedding for item in sorted_data]
