# MemoryHouse Envelope Implementation Plan

## Purpose

Implement the wrapping design in
[`memoryhouse-amt-envelope.md`](memoryhouse-amt-envelope.md) without
changing the existing Agent Memory Toolkit (AMT) record hierarchy or Cosmos DB
serving behavior.

The implementation will serialize a complete `MemoryRecordBase` document into
a governed MemoryHouse envelope and reconstruct the typed AMT record from that
payload. Integration with a concrete MemoryHouse service will be isolated
behind protocols so the core adapter can be developed and tested before a
service SDK is selected.

## Desired Outcome

Applications can:

1. Wrap any supported AMT record in a versioned MemoryHouse envelope.
2. Validate and unwrap the envelope into the correct AMT record subtype.
3. Apply explicit personal, team, or organization scope through `owner`.
4. Apply ACL policy independently from scope at the integration boundary.
5. Write, look up, scan, annotate, and forget envelopes through an abstract
   MemoryHouse client.
6. Replicate envelopes and deletion tombstones reliably between MemoryHouse
   and AMT serving stores.
7. Adopt MemoryHouse as the canonical store without breaking existing AMT
   retrieval APIs.

## Guiding Constraints

- `record.to_doc()` is the authoritative serialized AMT payload.
- Wrapping must not mutate the source record or payload.
- Unwrapping must use `MemoryRecordBase.from_doc()` and preserve subtype
  validation.
- Owner and ACL values must be explicit inputs or supplied by a trusted policy.
- `owner.type` is the canonical scope level; no separate serialized `scope`
  field is introduced.
- `owner.id` identifies the user, team, or organization that owns the memory.
- `payload.user_id` remains an AMT subject and partition field, not a
  MemoryHouse scope or authorization field.
- Annotations must not duplicate AMT payload fields.
- AMT subtype discovery uses subtype-specific content types.
- Unsupported content types and versions must fail explicitly.
- The first release must not alter existing `CosmosMemoryClient` behavior.
- Sync and async integrations must provide equivalent semantics.
- No MemoryHouse dependency should be added until an official client contract
  is selected.

## Proposed Package Structure

```text
azure/cosmos/agent_memory/memoryhouse/
├── __init__.py
├── models.py
├── adapter.py
├── protocols.py
├── service.py
├── aio/
│   ├── __init__.py
│   └── service.py
└── projection.py

tests/unit/memoryhouse/
├── test_models.py
├── test_adapter.py
├── test_service.py
└── test_projection.py

tests/unit/aio/memoryhouse/
└── test_service.py
```

The initial implementation should include only `models.py`, `adapter.py`, and
`protocols.py`. Service and projection modules should be added in later phases.

## Core Types

### MemoryEnvelope

Define Pydantic wire models matching the reviewed MemoryHouse contract:

```python
class MemoryOwner(BaseModel):
    type: Literal["user", "team", "organization"]
    id: str

    @property
    def scope(self) -> Literal["personal", "team", "organization"]:
        return {
            "user": "personal",
            "team": "team",
            "organization": "organization",
        }[self.type]


class MemoryACL(BaseModel):
    read: list[str]
    write: list[str]
    annotate: list[str]
    forget: list[str]


class MemoryProvenance(BaseModel):
    application: str
    agent_id: str | None = Field(alias="agentId")
    created_by: str = Field(alias="createdBy")
    created_at: str = Field(alias="createdAt")
    source_id: str = Field(alias="sourceId")
    prompt_id: str | None = Field(default=None, alias="promptId")
    prompt_version: str | None = Field(default=None, alias="promptVersion")


class MemoryEnvelope(BaseModel):
    id: str
    content_type: str = Field(alias="contentType")
    owner: MemoryOwner
    provenance: MemoryProvenance
    acl: MemoryACL
    annotations: dict[str, Any]
    payload: dict[str, Any]
```

Model requirements:

- Emit camel-case wire names where required.
- Reject unknown owner types.
- Reject a separate serialized `scope` field.
- Expose scope only as a non-serialized convenience property derived from
  `owner.type`.
- Reject empty IDs and ACL principals.
- Require every ACL operation, even when its principal list is empty.
- Preserve unknown annotation namespaces.
- Treat the payload as a JSON object without interpreting it in the envelope
  model.
- Use `extra="forbid"` for the standardized envelope and governance objects.

If MemoryHouse publishes an official model package before implementation,
prefer those models and keep AMT-specific types limited to adapter inputs and
content-type mapping.

## Adapter Design

### Constants

```python
AMT_MEMORY_CONTENT_TYPES = {
    "turn": "application/vnd.microsoft.amt.turn+json;version=1",
    "fact": "application/vnd.microsoft.amt.fact+json;version=1",
    "episodic": "application/vnd.microsoft.amt.episodic+json;version=1",
    "procedural": "application/vnd.microsoft.amt.procedural+json;version=1",
    "thread_summary": (
        "application/vnd.microsoft.amt.thread-summary+json;version=1"
    ),
    "user_summary": (
        "application/vnd.microsoft.amt.user-summary+json;version=1"
    ),
}
AMT_APPLICATION_NAME = "agent-memory-toolkit"
```

Content-type parsing should normalize insignificant whitespace but require an
exact supported subtype and version.

### Wrapping

`AMTMemoryEnvelopeAdapter.wrap()` will:

1. Validate `store_id` as a safe path segment.
2. Serialize the AMT record with `record.to_doc()`.
3. Build the stable ID `amt/{store_id}/{record.id}`.
4. Require `owner`, `acl`, and `created_by`, or resolve them through an
   injected trusted policy.
5. Build provenance from the record and explicit caller context.
6. Select the content type from the validated record type.
7. Copy optional caller-provided enrichment annotations without adding AMT
   payload fields.
8. Return a validated `MemoryEnvelope`.

The adapter must deep-copy the payload and annotations so later mutations do
not modify the source AMT record or previously returned envelopes.

### Unwrapping

`AMTMemoryEnvelopeAdapter.unwrap()` will:

1. Validate the envelope content type and version.
2. Require a mapping payload.
3. Verify `provenance.sourceId` matches `payload.id`.
4. Verify the namespaced envelope ID ends with the expected encoded AMT ID.
5. Verify the content-type subtype agrees with `payload.type`.
6. Reconstruct the record with `MemoryRecordBase.from_doc(payload)`.
7. Return the typed `MemoryRecordBase` subtype.

### Identifier Encoding

Use percent encoding for `store_id` and AMT IDs rather than accepting arbitrary
slashes:

```text
amt/{encoded-store-id}/{encoded-amt-id}
```

Provide `build_envelope_id()` and `parse_envelope_id()` helpers with round-trip
tests. Reject empty values and malformed prefixes.

## MemoryHouse Client Protocol

Define transport-neutral sync and async protocols:

```python
class MemoryHouseClientProtocol(Protocol):
    def lookup(self, memory_id: str) -> MemoryEnvelope: ...
    def scan(self, query: MemoryScanQuery) -> list[MemoryEnvelope]: ...
    def write(self, envelope: MemoryEnvelope) -> MemoryEnvelope: ...
    def annotate(
        self,
        memory_id: str,
        annotations: dict[str, Any],
    ) -> MemoryEnvelope: ...
    def forget(self, memory_id: str) -> None: ...
```

The async protocol exposes equivalent coroutine methods. Authentication and
authorization remain the responsibility of the MemoryHouse implementation;
the AMT integration must not provide a bypass or local authorization fallback.

## Integration Service

Add an opt-in `MemoryHouseService` after the core adapter is stable. It should:

- Wrap and write AMT records.
- Look up and unwrap AMT envelopes.
- Scan envelopes and return typed AMT records with their envelopes.
- Preserve enrichment annotations when replacing an AMT payload.
- Expose annotation mutation for independently produced enrichment.
- Translate MemoryHouse not-found, conflict, authorization, and transport
  errors into explicit AMT integration exceptions.

Do not add these methods directly to `CosmosMemoryClient` in the first release.
Keeping the integration composable avoids coupling the established AMT public
API to an unfinalized MemoryHouse service contract.

## Replication Strategy

### Phase A: Adapter-Only

AMT and MemoryHouse applications call the adapter explicitly. No automatic
replication or existing-client behavior changes occur.

### Phase B: AMT-Authoritative Mirror

For an initial production pilot, keep AMT Cosmos containers authoritative and
use their change feeds to:

1. Read inserted or updated AMT documents.
2. Parse them with `MemoryRecordBase.from_doc()`.
3. Resolve owner scope and ACL through deployment policy.
4. Wrap and write them to MemoryHouse.
5. Record retry state and dead-letter terminal failures.

This avoids unreliable synchronous dual writes and requires no change to AMT's
current write path.

Deletes require explicit tombstones because Cosmos change feed behavior alone
may not provide all routing and governance information needed by the target.
Add a durable deletion event before enabling mirrored forget behavior.

### Phase C: MemoryHouse-Authoritative

After MemoryHouse write, audit, and availability requirements are proven:

1. Route new governed writes through `MemoryHouseService`.
2. Publish a durable projection event in the same logical write workflow.
3. Project AMT payloads into the existing three Cosmos serving containers.
4. Continue serving AMT retrieval APIs from Cosmos.
5. Reconcile envelope and serving-store state periodically.

The authority mode must be explicit configuration:

```text
disabled
amt_mirror
memoryhouse_canonical
```

Startup validation must reject conflicting writer configurations that could
create replication loops.

## Projection Rules

The projector will:

- Accept only supported AMT content types.
- Unwrap and validate the typed record before writing it to Cosmos.
- Verify the content-type subtype matches the validated payload `type`.
- Route by the validated payload `type`.
- Use the payload's `user_id` and `thread_id` as Cosmos partition keys.
- Upsert through existing `MemoryStore.upsert_memory()` behavior.
- Preserve the complete payload, including AMT lifecycle fields.
- Use idempotent envelope version or ETag checkpoints.
- Ignore replayed events that are not newer than the recorded checkpoint.

Forgetting publishes a tombstone containing:

```json
{
  "envelopeId": "amt/default/fact_123",
  "sourceId": "fact_123",
  "memoryType": "fact",
  "userId": "u1",
  "threadId": "thread-42"
}
```

The projector validates this data against its last known envelope before
deleting the AMT serving document.

## Error Model

Add integration-specific exceptions:

- `MemoryEnvelopeValidationError`
- `UnsupportedMemoryContentTypeError`
- `MemoryEnvelopeIntegrityError`
- `MemoryHouseAuthorizationError`
- `MemoryHouseConflictError`
- `MemoryHouseTransportError`
- `MemoryProjectionError`

Do not convert authorization or transport failures into empty results.
Projection failures must retain the envelope ID, operation, and retryability
classification without logging payload content.

## Testing Strategy

### Unit Tests

- Envelope and governance model validation.
- Camel-case serialization and parsing.
- Wrap/unwrap round trips for all six AMT record types.
- Preservation of every payload field, including embeddings and subtype data.
- Stable and reversible envelope IDs.
- Empty annotations on wrapping unless enrichment is explicitly supplied.
- No AMT payload fields copied into annotations.
- Content-type and version rejection.
- Content-type subtype and payload-type mismatch detection.
- Provenance and payload integrity mismatch detection.
- Source-record immutability after wrapping.
- Sync and async service protocol behavior using fakes.

### Property and Compatibility Tests

- For every representative AMT document:

  ```python
  unwrapped.to_doc() == original.to_doc()
  ```

- Existing AMT model, store, retrieval, and pipeline tests remain unchanged.
- Historical AMT documents accepted by `MemoryRecordBase.from_doc()` also
  survive a wrap/unwrap round trip.

### Integration Tests

- Write and lookup against a MemoryHouse test implementation.
- AMT Cosmos change-feed mirror with retry and replay.
- MemoryHouse-to-Cosmos projection routing for every memory type.
- Annotation update without payload mutation.
- Forget propagation and idempotent tombstone replay.
- Conflict behavior under concurrent payload and annotation updates.

## Observability

Emit structured telemetry for:

- `memoryhouse.wrap`
- `memoryhouse.unwrap`
- `memoryhouse.write`
- `memoryhouse.lookup`
- `memoryhouse.annotate`
- `memoryhouse.forget`
- `memoryhouse.project`
- `memoryhouse.projection_retry`
- `memoryhouse.projection_dead_letter`

Include envelope ID, AMT memory type, operation duration, result, and retry
count. Do not log payload content, embeddings, ACL contents, or owner IDs unless
the deployment's telemetry policy explicitly permits them.

## Documentation and Public API

When the adapter is released:

- Export adapter and envelope types from
  `azure.cosmos.agent_memory.memoryhouse`, not initially from the package root.
- Add API documentation and wrapping examples.
- Document identity and ACL policy requirements.
- Document supported content-type versions.
- Document authority modes before enabling replication.
- Add a changelog entry describing the feature as opt-in.

## Delivery Phases

### Phase 1: Contracts and Lossless Adapter

Deliver models, adapter, exceptions, identifier helpers, and exhaustive unit
tests. This phase has no external service or Cosmos behavior changes.

**Exit criteria:** all supported AMT record types round-trip losslessly and
invalid governance or content types fail explicitly.

### Phase 2: Transport-Neutral Service Layer

Deliver sync and async protocols, service implementations, fake clients, and
operation-level tests.

**Exit criteria:** applications can use all five MemoryHouse operations through
an injected client without coupling AMT to a specific SDK.

### Phase 3: AMT-Authoritative Mirror Pilot

Deliver change-feed wrapping, ACL policy resolution, retries, dead-letter
handling, and reconciliation reporting.

**Exit criteria:** AMT writes are mirrored reliably and replay produces no
duplicates or divergent IDs.

### Phase 4: MemoryHouse-Canonical Projection

Deliver durable projection events, Cosmos projection, forget tombstones,
authority-mode validation, and operational runbooks.

**Exit criteria:** MemoryHouse can be the canonical governed store while all
existing AMT retrieval APIs continue to operate from consistent Cosmos
projections.

## Open Decisions

The following decisions must be resolved before Phase 2 or Phase 3:

1. Whether MemoryHouse provides an official Python SDK and canonical models.
2. Exact scan-query and optimistic-concurrency contracts.
3. Identity principal syntax and the service responsible for ACL resolution.
4. The source of explicit team and organization scope during AMT mirroring.
5. Audit-event ownership and retention requirements.
6. Event or outbox technology used for canonical-store projection.
7. Whether payload embeddings are accepted by MemoryHouse storage policy.
8. Required consistency and recovery objectives for forget propagation.
