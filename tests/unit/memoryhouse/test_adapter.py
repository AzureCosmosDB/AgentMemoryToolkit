from __future__ import annotations

import copy

import pytest

from azure.cosmos.agent_memory.memoryhouse import (
    AMT_MEMORY_CONTENT_TYPES,
    AMTMemoryEnvelopeAdapter,
    MemoryACL,
    MemoryEnvelopeIntegrityError,
    MemoryEnvelopeValidationError,
    MemoryOwner,
    UnsupportedMemoryContentTypeError,
    build_envelope_id,
    content_type_for_memory_type,
    memory_type_for_content_type,
    parse_envelope_id,
)
from azure.cosmos.agent_memory.models import (
    EpisodicRecord,
    FactRecord,
    ProceduralRecord,
    ThreadSummaryRecord,
    TurnRecord,
    UserSummaryRecord,
)

_HEX32 = "a" * 32


def _records():
    """Build representative instances of every supported AMT record subtype."""
    return [
        TurnRecord(id="turn/1", user_id="u1", thread_id="t1", role="user", content="Hello"),
        ThreadSummaryRecord(
            id="summary_" + _HEX32,
            user_id="u1",
            thread_id="t1",
            content="Summary",
            prompt_id="summarize.prompty",
        ),
        UserSummaryRecord(
            id="user_summary_" + _HEX32,
            user_id="u1",
            thread_id="__user_summary__",
            content="User summary",
            metadata={"thread_ids": ["t1"]},
            prompt_id="user_summary.prompty",
        ),
        FactRecord(
            id="fact_" + _HEX32,
            user_id="u1",
            thread_id="t1",
            content="User prefers concise responses.",
            metadata={"category": "preference"},
            content_hash=_HEX32,
            prompt_id="extract_memories.prompty",
            confidence=0.9,
            embedding=[0.1, 0.2],
        ),
        EpisodicRecord(
            id="ep_" + _HEX32,
            user_id="u1",
            thread_id="t1",
            content="The deployment succeeded.",
            title="Deployment",
            events=[{"sequence": 1, "description": "Deployed the service."}],
            content_hash=_HEX32,
            prompt_id="extract_episode.prompty",
        ),
        ProceduralRecord(
            id="proc_u1_1",
            user_id="u1",
            thread_id="t1",
            content="Validate before deploying.",
            name="Safe deployment",
            summary="Validate changes before deployment.",
            retrieval_text="deployment validation",
            procedure_kind="behavioral_policy",
            prompt_id="extract_procedure.prompty",
        ),
    ]


def _acl() -> MemoryACL:
    return MemoryACL(
        read=["user:u1"],
        write=["user:u1"],
        annotate=["service:enricher"],
        forget=["user:u1"],
    )


@pytest.mark.parametrize(
    "store_id,memory_id",
    [
        ("default", "fact_1"),
        ("west/us", "turn/1"),
        ("store with spaces", "memory % value"),
    ],
)
def test_envelope_id_round_trip(store_id: str, memory_id: str):
    envelope_id = build_envelope_id(store_id, memory_id)
    assert parse_envelope_id(envelope_id) == (store_id, memory_id)


@pytest.mark.parametrize("envelope_id", ["", "other/a/b", "amt/a", "amt/a/b/c", "amt/a/%ZZ", "amt/a/b c"])
def test_parse_envelope_id_rejects_invalid_values(envelope_id: str):
    with pytest.raises(MemoryEnvelopeValidationError):
        parse_envelope_id(envelope_id)


@pytest.mark.parametrize("memory_type", sorted(AMT_MEMORY_CONTENT_TYPES))
def test_content_type_round_trip(memory_type: str):
    content_type = content_type_for_memory_type(memory_type)
    assert memory_type_for_content_type(content_type) == memory_type
    media_type, version = content_type.split(";")
    assert memory_type_for_content_type(f"  {media_type} ; {version}  ") == memory_type


def test_content_type_rejects_unknown_type_and_version():
    with pytest.raises(UnsupportedMemoryContentTypeError):
        content_type_for_memory_type("unknown")
    with pytest.raises(UnsupportedMemoryContentTypeError):
        memory_type_for_content_type("application/vnd.microsoft.amt.fact+json;version=2")


@pytest.mark.parametrize("record", _records(), ids=lambda record: record.to_doc()["type"])
def test_wrap_unwrap_is_lossless_for_every_record_type(record):
    # This equality is the central compatibility guarantee: governance may be
    # added around a record, but AMT sees exactly the same document after
    # unwrapping.
    adapter = AMTMemoryEnvelopeAdapter()
    annotations = {"classification": {"domain": "payments"}}
    envelope = adapter.wrap(
        record,
        store_id="default",
        owner=MemoryOwner(type="user", id="u1"),
        acl=_acl(),
        created_by="agent-1",
        annotations=annotations,
    )

    assert envelope.scope == "personal"
    assert envelope.annotations == annotations
    assert set(envelope.annotations) == {"classification"}
    assert envelope.payload == record.to_doc()
    assert envelope.content_type == AMT_MEMORY_CONTENT_TYPES[record.to_doc()["type"]]
    assert adapter.unwrap(envelope).to_doc() == record.to_doc()


def test_wrap_deep_copies_payload_and_annotations():
    # Both directions contain nested mutable values. The adapter must isolate
    # them so callers cannot accidentally mutate source records or input
    # annotations through the returned envelope.
    adapter = AMTMemoryEnvelopeAdapter()
    record = _records()[3]
    annotations = {"classification": {"keywords": ["preference"]}}
    expected_payload = copy.deepcopy(record.to_doc())

    envelope = adapter.wrap(
        record,
        store_id="default",
        owner={"type": "team", "id": "payments"},
        acl=_acl(),
        created_by="agent-1",
        annotations=annotations,
    )
    envelope.payload["content"] = "changed"
    envelope.annotations["classification"]["keywords"].append("changed")

    assert record.to_doc() == expected_payload
    assert annotations == {"classification": {"keywords": ["preference"]}}
    assert envelope.scope == "team"


def test_unwrap_rejects_content_type_payload_mismatch():
    adapter = AMTMemoryEnvelopeAdapter()
    envelope = adapter.wrap(
        _records()[3],
        store_id="default",
        owner={"type": "user", "id": "u1"},
        acl=_acl(),
        created_by="agent-1",
    )
    doc = envelope.model_dump(mode="json", by_alias=True)
    doc["contentType"] = AMT_MEMORY_CONTENT_TYPES["episodic"]

    with pytest.raises(MemoryEnvelopeIntegrityError, match="payload.type"):
        adapter.unwrap(doc)


def test_unwrap_rejects_identity_mismatch():
    adapter = AMTMemoryEnvelopeAdapter()
    envelope = adapter.wrap(
        _records()[0],
        store_id="default",
        owner={"type": "organization", "id": "contoso"},
        acl=_acl(),
        created_by="agent-1",
    )
    doc = envelope.model_dump(mode="json", by_alias=True)
    doc["provenance"]["sourceId"] = "different"

    with pytest.raises(MemoryEnvelopeIntegrityError, match="sourceId"):
        adapter.unwrap(doc)
