from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from azure.cosmos.agent_memory._base import _BaseMemoryClient
from azure.cosmos.agent_memory.exceptions import ConfigurationError, CosmosNotConnectedError


class DummyClient(_BaseMemoryClient):
    def __init__(self, **kwargs):
        defaults = {
            "cosmos_endpoint": None,
            "cosmos_credential": None,
            "cosmos_key": None,
            "cosmos_database": None,
            "cosmos_container": None,
            "cosmos_counter_container": None,
            "cosmos_lease_container": None,
            "cosmos_throughput_mode": None,
            "cosmos_autoscale_max_ru": None,
            "ai_foundry_endpoint": None,
            "ai_foundry_credential": None,
            "ai_foundry_api_key": None,
            "embedding_deployment_name": "text-embedding-3-large",
            "embedding_dimensions": None,
            "chat_deployment_name": "gpt-4o-mini",
            "use_default_credential": False,
        }
        defaults.update(kwargs)
        self._init_base_config(**defaults)
        self.closed = False

    def close(self) -> None:
        self.closed = True


def test_base_config_defaults_and_cosmos_key_fallback():
    client = DummyClient(cosmos_key="key")

    assert client.local_memory == []
    assert client._cosmos_database == "ai_memory"
    assert client._cosmos_container == "memories"
    assert client._cosmos_credential == "key"
    assert client._embedding_dimensions == 1536


def test_base_config_creates_default_credentials_when_requested():
    credential = MagicMock()
    module = MagicMock()
    module.DefaultAzureCredential.return_value = credential

    with patch.dict("sys.modules", {"azure.identity": module}):
        client = DummyClient(use_default_credential=True)

    assert module.DefaultAzureCredential.call_count == 2
    assert client._cosmos_credential is credential
    assert client._ai_foundry_credential is credential
    assert client._owns_cosmos_credential is True
    assert client._owns_ai_foundry_credential is True


def test_base_config_validates_throughput_mode():
    with pytest.raises(ConfigurationError, match="expected 'serverless' or 'autoscale'"):
        DummyClient(cosmos_throughput_mode="invalid")


def test_base_config_normalizes_ai_foundry_project_endpoint():
    client = DummyClient(ai_foundry_endpoint="https://my-res.services.ai.azure.com/api/projects/my-project")
    assert client._ai_foundry_endpoint == "https://my-res.services.ai.azure.com"


def test_base_config_leaves_plain_ai_foundry_endpoint_untouched():
    client = DummyClient(ai_foundry_endpoint="https://my-res.services.ai.azure.com")
    assert client._ai_foundry_endpoint == "https://my-res.services.ai.azure.com"


def test_require_cosmos_guard_and_context_manager():
    client = DummyClient()

    with pytest.raises(CosmosNotConnectedError):
        client._require_cosmos()

    client._memories_container_client = object()
    client._require_cosmos()

    with client as entered:
        assert entered is client
    assert client.closed is True


def test_warn_on_embedding_dim_mismatch_logs_warning(caplog):
    client = DummyClient(embedding_dimensions=128)
    container = MagicMock()
    container.read.return_value = {"vectorEmbeddingPolicy": {"vectorEmbeddings": [{"dimensions": 256}]}}

    client._warn_on_embedding_dim_mismatch(container)

    assert "Embedding dimension mismatch" in caplog.text
