"""Unit tests for shared helpers in agent_memory_toolkit._utils."""

from agent_memory_toolkit._utils import _build_container_kwargs


def test_build_container_kwargs_includes_required_fields_and_extras():
    partition_key = object()
    throughput = object()

    kwargs = _build_container_kwargs(
        container_id="memories",
        partition_key=partition_key,
        offer_throughput=throughput,
        indexing_policy={"includedPaths": [{"path": "/*"}]},
        full_text_policy={"defaultLanguage": "en-US"},
    )

    assert kwargs["id"] == "memories"
    assert kwargs["partition_key"] is partition_key
    assert kwargs["offer_throughput"] is throughput
    assert kwargs["indexing_policy"] == {"includedPaths": [{"path": "/*"}]}
    assert kwargs["full_text_policy"] == {"defaultLanguage": "en-US"}


def test_build_container_kwargs_omits_offer_throughput_when_none():
    kwargs = _build_container_kwargs(
        container_id="leases",
        partition_key="/id",
        offer_throughput=None,
    )

    assert kwargs == {
        "id": "leases",
        "partition_key": "/id",
    }
