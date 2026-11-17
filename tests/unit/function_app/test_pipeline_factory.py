"""Unit tests for ``function_app/shared/pipeline_factory.py``.

The factory lazily constructs one ``PipelineService`` per tenant from environment variables.
We mock ``DefaultAzureCredential``, the Cosmos container helper, and the SDK
clients so no Azure resource is touched.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from shared import pipeline_factory

from azure.cosmos.agent_memory.exceptions import ValidationError


@pytest.fixture(autouse=True)
def _reset_pipeline_cache():
    """Each test must observe a fresh build."""
    pipeline_factory._pipelines.clear()
    pipeline_factory._shared = None
    yield
    pipeline_factory._pipelines.clear()
    pipeline_factory._shared = None


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
    result = pipeline_factory.get_pipeline("acme")

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
        tenant_id="acme",
    )
    mocks.pipeline_ctor.assert_called_once_with(
        mocks.store_instance,
        mocks.chat_instance,
        mocks.embed_instance,
        containers=expected_containers,
        transcript_metadata_keys=None,
        tenant_id="acme",
    )


def test_the_tenant_defaults_to_the_default_tenant(mocks):
    pipeline_factory.get_pipeline()

    assert mocks.store_ctor.call_args.kwargs["tenant_id"] == "default"
    assert mocks.pipeline_ctor.call_args.kwargs["tenant_id"] == "default"


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
    a = pipeline_factory.get_pipeline("acme")
    b = pipeline_factory.get_pipeline("acme")

    assert a is b
    assert mocks.pipeline_ctor.call_count == 1
    assert mocks.store_ctor.call_count == 1
    assert mocks.chat_ctor.call_count == 1
    assert mocks.embed_ctor.call_count == 1


def test_each_tenant_gets_its_own_cached_pipeline_bound_to_its_own_store(mocks):
    stores: dict[str, MagicMock] = {}
    pipelines: dict[str, MagicMock] = {}

    def new_store(**kwargs):
        stores[kwargs["tenant_id"]] = MagicMock(name=f"store-{kwargs['tenant_id']}")
        return stores[kwargs["tenant_id"]]

    def new_pipeline(store, *_args, **kwargs):
        pipelines[kwargs["tenant_id"]] = MagicMock(name=f"pipeline-{kwargs['tenant_id']}", store=store)
        return pipelines[kwargs["tenant_id"]]

    mocks.store_ctor.side_effect = new_store
    mocks.pipeline_ctor.side_effect = new_pipeline

    acme = pipeline_factory.get_pipeline("acme")
    globex = pipeline_factory.get_pipeline("globex")

    assert acme is pipelines["acme"] and globex is pipelines["globex"] and acme is not globex
    assert acme.store is stores["acme"] and globex.store is stores["globex"]
    assert pipeline_factory.get_pipeline("acme") is acme
    assert pipeline_factory.get_pipeline("globex") is globex
    assert mocks.pipeline_ctor.call_count == 2 and mocks.store_ctor.call_count == 2


def test_tenants_share_the_containers_and_the_chat_and_embeddings_clients(mocks):
    pipeline_factory.get_pipeline("acme")
    pipeline_factory.get_pipeline("globex")

    assert mocks.credential_ctor.call_count == 1
    assert mocks.chat_ctor.call_count == 1
    assert mocks.embed_ctor.call_count == 1
    (acme_store, globex_store) = mocks.store_ctor.call_args_list
    (acme, globex) = mocks.pipeline_ctor.call_args_list
    assert acme_store.kwargs["containers"] is globex_store.kwargs["containers"] is acme.kwargs["containers"]
    assert acme.kwargs["containers"] is globex.kwargs["containers"]
    assert acme.args[1:] == globex.args[1:] == (mocks.chat_instance, mocks.embed_instance)


@pytest.mark.parametrize("tenant_id", ["", "   ", "acme/eu", "acme?eu", "acme#eu", "acme\\eu", "x" * 300])
def test_an_invalid_tenant_is_refused_before_anything_is_built(mocks, tenant_id):
    with pytest.raises(ValidationError):
        pipeline_factory.get_pipeline(tenant_id)

    mocks.credential_ctor.assert_not_called()
    mocks.store_ctor.assert_not_called()
    mocks.pipeline_ctor.assert_not_called()
    assert pipeline_factory._pipelines == {}


def test_the_store_and_pipeline_it_builds_are_bound_to_the_tenant(monkeypatch):
    """The real ``MemoryStore`` and ``PipelineService`` take the tenant the factory passes (only Azure is faked)."""
    for getter in ("get_memories_container", "get_turns_container", "get_summaries_container"):
        monkeypatch.setattr(pipeline_factory, getter, lambda: MagicMock())
    with (
        patch("azure.identity.DefaultAzureCredential"),
        patch("azure.cosmos.agent_memory.chat.ChatClient"),
        patch("azure.cosmos.agent_memory.embeddings.EmbeddingsClient"),
    ):
        acme = pipeline_factory.get_pipeline("acme")
        globex = pipeline_factory.get_pipeline("globex")

    assert (acme._tenant_id, acme._store.tenant_id) == ("acme", "acme")
    assert (globex._tenant_id, globex._store.tenant_id) == ("globex", "globex")


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
