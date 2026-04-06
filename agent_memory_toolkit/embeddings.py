"""Synchronous embedding client for the Agent Memory Toolkit.

Provides :class:`EmbeddingsClient` that lazily initialises an Azure OpenAI
connection and generates embeddings via the OpenAI API.
"""

from __future__ import annotations

import logging
from typing import Any

from .exceptions import ConfigurationError, EmbeddingError

logger = logging.getLogger(__name__)

_TOKEN_SCOPE = "https://cognitiveservices.azure.com/.default"


# ---------------------------------------------------------------------------
# Sync client
# ---------------------------------------------------------------------------


class EmbeddingsClient:
    """Synchronous embedding client backed by Azure OpenAI.

    Parameters
    ----------
    endpoint:
        Azure OpenAI resource endpoint URL.
    credential:
        Optional Azure ``TokenCredential``.  Used when *api_key* is not set
        to obtain bearer tokens for the OpenAI service.
    api_key:
        Optional API key for the Azure OpenAI resource.
    model:
        Deployment / model name.  Defaults to ``"text-embedding-3-large"``.
    dimensions:
        Optional embedding dimensions override.
    api_version:
        Azure OpenAI API version.  Defaults to ``"2024-12-01-preview"``.
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
        self._client: Any = None  # openai.AzureOpenAI (lazy)

    # -- internal helpers ---------------------------------------------------

    def _ensure_client(self) -> Any:
        """Lazily create the ``AzureOpenAI`` client on first use."""
        if self._client is not None:
            return self._client

        if not self._endpoint:
            raise ConfigurationError(
                "An embedding endpoint is required", parameter="endpoint"
            )

        from openai import AzureOpenAI

        if self._api_key:
            self._client = AzureOpenAI(
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
            from azure.identity import get_bearer_token_provider

            token_provider = get_bearer_token_provider(self._credential, _TOKEN_SCOPE)
            self._client = AzureOpenAI(
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

    def generate(self, text: str) -> list[float]:
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
            response = client.embeddings.create(**kwargs)
        except Exception as exc:
            raise EmbeddingError(f"Embedding generation failed: {exc}") from exc
        return response.data[0].embedding

    def generate_batch(self, texts: list[str]) -> list[list[float]]:
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
            response = client.embeddings.create(**kwargs)
        except Exception as exc:
            raise EmbeddingError(
                f"Batch embedding generation failed: {exc}"
            ) from exc

        # The API returns results with an ``index`` field; sort to guarantee
        # the caller receives embeddings in the same order as the input.
        sorted_data = sorted(response.data, key=lambda d: d.index)
        return [item.embedding for item in sorted_data]
