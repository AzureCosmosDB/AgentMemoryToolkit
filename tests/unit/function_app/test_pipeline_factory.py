"""Unit tests for ``function_app/shared/pipeline_factory.py``.

The factory lazily constructs a ``PipelineService`` from environment variables.
We mock ``DefaultAzureCredential``, the Cosmos container helper, and the SDK
clients so no Azure resource is touched.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from shared import pipeline_factory


@pytest.fixture(autouse=True)
def _reset_pipeline_cache():
    """Each test must observe a fresh build."""
    pipeline_factory._pipeline = None
    yield
    pipeline_factory._pipeline = None


@pytest.fixture(autouse=True)
def _required_env(monkeypatch):
    """Set the env vars the factory always needs."""
    monkeypatch.setenv("COSMOS_DB_ENDPOINT", "https://fake-cosmos.documents.azure.com:443/")
    monkeypatch.setenv("AI_FOUNDRY_ENDPOINT", "https://fake-foundry.openai.azure.com/")
    yield


@pytest.fixture
def mocks(monkeypatch):
    """Patch every external dependency used by ``get_pipeline``.

    Returns a namespace exposing each MagicMock so tests can assert on calls.
    """
    container = MagicMock(name="ContainerProxy")
    turns_container = MagicMock(name="TurnsContainerProxy")
    summaries_container = MagicMock(name="SummariesContainerProxy")
    credential = MagicMock(name="DefaultAzureCredential")
    chat_instance = MagicMock(name="ChatClient_instance")
    embed_instance = MagicMock(name="EmbeddingsClient_instance")
    store_instance = MagicMock(name="MemoryStore_instance")
    pipeline_instance = MagicMock(name="PipelineService_instance")

    chat_ctor = MagicMock(name="ChatClient_ctor", return_value=chat_instance)
    embed_ctor = MagicMock(name="EmbeddingsClient_ctor", return_value=embed_instance)
    store_ctor = MagicMock(name="MemoryStore_ctor", return_value=store_instance)
    pipeline_ctor = MagicMock(name="PipelineService_ctor", return_value=pipeline_instance)
    credential_ctor = MagicMock(name="DefaultAzureCredential_ctor", return_value=credential)

    monkeypatch.setattr(pipeline_factory, "get_memories_container", lambda: container)
    monkeypatch.setattr(pipeline_factory, "get_turns_container", lambda: turns_container)
    monkeypatch.setattr(pipeline_factory, "get_summaries_container", lambda: summaries_container)

    patches = [
        patch("azure.identity.DefaultAzureCredential", credential_ctor),
        patch("azure.cosmos.agent_memory.chat.ChatClient", chat_ctor),
        patch("azure.cosmos.agent_memory.embeddings.EmbeddingsClient", embed_ctor),
        patch("azure.cosmos.agent_memory.store.MemoryStore", store_ctor),
        patch("azure.cosmos.agent_memory.services.pipeline.PipelineService", pipeline_ctor),
    ]
    for p in patches:
        p.start()
    yield MagicMock(
        container=container,
        turns_container=turns_container,
        summaries_container=summaries_container,
        credential=credential,
        credential_ctor=credential_ctor,
        chat_ctor=chat_ctor,
        chat_instance=chat_instance,
        embed_ctor=embed_ctor,
        embed_instance=embed_instance,
        store_ctor=store_ctor,
        store_instance=store_instance,
        pipeline_ctor=pipeline_ctor,
        pipeline_instance=pipeline_instance,
    )
    for p in patches:
        p.stop()


# ---------------------------------------------------------------------------
# Build path
# ---------------------------------------------------------------------------


def test_builds_pipeline_from_complete_env(mocks):
    result = pipeline_factory.get_pipeline()

    assert result is mocks.pipeline_instance
    from azure.cosmos.agent_memory._container_routing import ContainerKey

    expected_containers = {
        ContainerKey.TURNS: mocks.turns_container,
        ContainerKey.MEMORIES: mocks.container,
        ContainerKey.SUMMARIES: mocks.summaries_container,
    }
    mocks.store_ctor.assert_called_once_with(
        containers=expected_containers,
        embeddings_client=mocks.embed_instance,
    )
    mocks.pipeline_ctor.assert_called_once_with(
        mocks.store_instance,
        mocks.chat_instance,
        mocks.embed_instance,
        containers=expected_containers,
        transcript_metadata_keys=None,
    )


def test_uses_chat_deployment_name_env_var(monkeypatch, mocks):
    monkeypatch.setenv("AI_FOUNDRY_CHAT_DEPLOYMENT_NAME", "gpt-4o-mega")
    pipeline_factory.get_pipeline()

    chat_kwargs = mocks.chat_ctor.call_args.kwargs
    assert chat_kwargs["model"] == "gpt-4o-mega"
    assert chat_kwargs["endpoint"] == "https://fake-foundry.openai.azure.com/"


def test_uses_embedding_deployment_name_env_var(monkeypatch, mocks):
    monkeypatch.setenv("AI_FOUNDRY_EMBEDDING_DEPLOYMENT_NAME", "text-embedding-tiny")
    pipeline_factory.get_pipeline()

    embed_kwargs = mocks.embed_ctor.call_args.kwargs
    assert embed_kwargs["model"] == "text-embedding-tiny"


def test_default_chat_and_embedding_models_when_env_unset(monkeypatch, mocks):
    monkeypatch.delenv("AI_FOUNDRY_CHAT_DEPLOYMENT_NAME", raising=False)
    monkeypatch.delenv("AI_FOUNDRY_EMBEDDING_DEPLOYMENT_NAME", raising=False)

    pipeline_factory.get_pipeline()

    assert mocks.chat_ctor.call_args.kwargs["model"] == "gpt-4o-mini"
    assert mocks.embed_ctor.call_args.kwargs["model"] == "text-embedding-3-large"


def test_passes_credential_to_both_clients(mocks):
    pipeline_factory.get_pipeline()

    assert mocks.chat_ctor.call_args.kwargs["credential"] is mocks.credential
    assert mocks.embed_ctor.call_args.kwargs["credential"] is mocks.credential


def test_transcript_metadata_keys_env_threads_into_pipeline(monkeypatch, mocks):
    monkeypatch.setenv(
        "AGENT_MEMORY_TRANSCRIPT_METADATA_KEYS",
        " agent_id , timestamp ,  , model_id",
    )
    pipeline_factory.get_pipeline()

    kwargs = mocks.pipeline_ctor.call_args.kwargs
    assert kwargs["transcript_metadata_keys"] == ("agent_id", "timestamp", "model_id")


def test_transcript_metadata_keys_env_unset_yields_none(monkeypatch, mocks):
    monkeypatch.delenv("AGENT_MEMORY_TRANSCRIPT_METADATA_KEYS", raising=False)
    pipeline_factory.get_pipeline()

    assert mocks.pipeline_ctor.call_args.kwargs["transcript_metadata_keys"] is None


def test_transcript_metadata_keys_env_empty_string_yields_none(monkeypatch, mocks):
    monkeypatch.setenv("AGENT_MEMORY_TRANSCRIPT_METADATA_KEYS", "   ")
    pipeline_factory.get_pipeline()

    assert mocks.pipeline_ctor.call_args.kwargs["transcript_metadata_keys"] is None


# ---------------------------------------------------------------------------
# Caching / idempotence
# ---------------------------------------------------------------------------


def test_pipeline_is_cached_across_calls(mocks):
    a = pipeline_factory.get_pipeline()
    b = pipeline_factory.get_pipeline()

    assert a is b
    assert mocks.pipeline_ctor.call_count == 1
    assert mocks.chat_ctor.call_count == 1
    assert mocks.embed_ctor.call_count == 1


# ---------------------------------------------------------------------------
# Missing-env error paths (these run BEFORE any client is constructed, so
# we don't need the heavy ``mocks`` fixture).
# ---------------------------------------------------------------------------


def test_missing_ai_foundry_endpoint_raises(monkeypatch):
    monkeypatch.delenv("AI_FOUNDRY_ENDPOINT", raising=False)
    # We must still patch the credential constructor so we can reach the
    # config check rather than failing on Azure-Identity import wiring.
    with (
        patch("azure.identity.DefaultAzureCredential", MagicMock()),
        patch.object(pipeline_factory, "get_memories_container", return_value=MagicMock()),
        patch.object(pipeline_factory, "get_turns_container", return_value=MagicMock()),
        patch.object(pipeline_factory, "get_summaries_container", return_value=MagicMock()),
    ):
        with pytest.raises(RuntimeError, match="AI_FOUNDRY_ENDPOINT"):
            pipeline_factory.get_pipeline()
