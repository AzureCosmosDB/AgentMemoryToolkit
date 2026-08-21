# Wrapping Agent Memory Toolkit Records in a MemoryHouse Envelope

## Overview

Agent Memory Toolkit (AMT) and MemoryHouse standardize different layers of a
memory system:

- AMT defines typed memory documents and product-specific retrieval fields.
- MemoryHouse defines a governed envelope around an opaque producer-owned
  payload.

The recommended integration is therefore to store the complete serialized AMT
document as the MemoryHouse payload. MemoryHouse fields provide ownership,
access control, provenance, discovery, enrichment, and audit without changing
the AMT representation.

```text
MemoryHouse memory
├── ID
├── ContentType
├── Owner
├── Provenance
├── ACL
├── Annotations
└── Payload
    └── Complete AMT document
```

This approach preserves AMT's typed records and allows existing AMT consumers
to reconstruct `TurnRecord`, `FactRecord`, `EpisodicRecord`,
`ProceduralRecord`, and summary records without losing information.

## Design Goals

- Preserve the AMT document without rewriting or normalizing its content.
- Add MemoryHouse governance without adding governance fields to every AMT
  record type.
- Keep AMT's Cosmos containers and indexes available as serving projections.
- Allow MemoryHouse to discover AMT memories without understanding every AMT
  subtype.
- Support future AMT schema versions through a versioned content type.
- Ensure MemoryHouse annotations can evolve without mutating the original AMT
  payload.

## Non-Goals

- Replacing AMT's typed Pydantic models with a generic memory model.
- Making MemoryHouse interpret episodic or procedural payload structures.
- Moving AMT vector, full-text, ranking, or prompt-building behavior into the
  MemoryHouse contract.
- Copying AMT payload fields into annotations for discovery.

## Canonical Wrapper

An AMT record is serialized with `MemoryRecordBase.to_doc()` and stored
unchanged under `payload`.

```json
{
  "id": "amt/default/fact_123",
  "contentType": "application/vnd.microsoft.amt.fact+json;version=1",
  "owner": {
    "type": "user",
    "id": "u1"
  },
  "provenance": {
    "application": "agent-memory-toolkit",
    "agentId": "architecture-agent",
    "createdBy": "architecture-agent",
    "createdAt": "2026-08-20T20:00:00Z",
    "sourceId": "fact_123"
  },
  "acl": {
    "read": ["user:u1"],
    "write": ["user:u1"],
    "annotate": ["service:memory-enrichers"],
    "forget": ["user:u1"]
  },
  "annotations": {},
  "payload": {
    "id": "fact_123",
    "user_id": "u1",
    "thread_id": "thread-42",
    "role": "system",
    "type": "fact",
    "content": "The user prefers concise reviews.",
    "metadata": {
      "category": "preference"
    },
    "salience": 0.8,
    "confidence": 0.95,
    "content_hash": "0123456789abcdef0123456789abcdef",
    "prompt_id": "extract_memories.prompty",
    "prompt_version": "v2",
    "created_at": "2026-08-20T20:00:00Z"
  }
}
```

## Field Mapping

| AMT field | MemoryHouse location | Notes |
| --- | --- | --- |
| Complete `record.to_doc()` result | `payload` | Authoritative AMT artifact |
| `id` | `id` and `provenance.sourceId` | Envelope ID is namespaced; payload ID is unchanged |
| `user_id` | `owner.id` for personal scope | Team and organization scope require explicit writer context |
| `type` | `contentType` subtype and `payload.type` | Content type supports subtype discovery; payload remains authoritative |
| `agent_id` | `provenance.agentId` | May be absent on some AMT records |
| `created_at` | `provenance.createdAt` | Payload value remains authoritative for AMT |
| `prompt_id`, `prompt_version` | `provenance` extensions | Records the generating prompt when available |
| `thread_id`, `role`, `tags` | `payload` only | Product indexes may extract them outside the canonical envelope |
| `salience`, `confidence` | `payload` only | Do not duplicate producer ranking signals in annotations |
| Supersession fields | `payload` only | AMT remains authoritative for its lifecycle state |
| `embedding` | Payload or external reference annotation | MemoryHouse does not prescribe vector storage |
| Permissions | `acl` | Must be explicitly supplied or derived by trusted policy |

Only fields required by the MemoryHouse governance contract should be repeated
outside the payload. AMT-specific retrieval fields stay exclusively in the
payload or in product-owned serving indexes.

## Identifier Strategy

MemoryHouse requires a unified ID space, while AMT IDs are generated within an
AMT deployment. Use a stable namespace:

```text
amt/{store-id}/{amt-id}
```

Examples:

```text
amt/default/fact_123
amt/support-prod/ep_456
amt/copilot-westus/proc_789
```

The same AMT record must always produce the same envelope ID. Changing a
record's content does not create a new envelope ID unless AMT itself creates a
new record.

## Content Type and Versioning

Use subtype-specific vendor MIME types that identify the producer, AMT record
type, and schema version:

```text
application/vnd.microsoft.amt.turn+json;version=1
application/vnd.microsoft.amt.fact+json;version=1
application/vnd.microsoft.amt.episodic+json;version=1
application/vnd.microsoft.amt.procedural+json;version=1
application/vnd.microsoft.amt.thread-summary+json;version=1
application/vnd.microsoft.amt.user-summary+json;version=1
```

The version describes the serialized AMT payload contract. The subtype in the
content type must agree with `payload.type`; consumers reject a mismatch.

Consumers must reject unsupported versions rather than attempting a
best-effort parse. A future breaking AMT representation should introduce a new
content-type version and a corresponding adapter.

## Adapter Contract

Wrapping and unwrapping should be implemented outside the AMT record models so
MemoryHouse concerns do not become part of every AMT subtype.

```python
class AMTMemoryEnvelopeAdapter:
    def wrap(
        self,
        record: MemoryRecordBase,
        *,
        store_id: str,
        owner: MemoryOwner,
        acl: MemoryACL,
        created_by: str,
    ) -> MemoryEnvelope:
        ...

    def unwrap(self, envelope: MemoryEnvelope) -> MemoryRecordBase:
        ...
```

`wrap()` should:

1. Serialize the record using `record.to_doc()`.
2. Generate the stable namespaced envelope ID.
3. Require an explicit owner and ACL.
4. Select the subtype-specific content type from the validated AMT record type.
5. Copy only caller-provided enrichment annotations.
6. Store the complete serialized document under `payload`.

`unwrap()` should:

1. Verify the caller has already passed MemoryHouse authorization.
2. Validate the content type and supported version.
3. Require a JSON object payload.
4. Verify the content-type subtype agrees with `payload.type`.
5. Reconstruct the typed record with `MemoryRecordBase.from_doc(payload)`.
6. Surface validation failures rather than silently returning an untyped
   object.

## Ownership and ACL Policy

AMT's `user_id` identifies the user associated with a memory, but it is not an
authorization policy. The wrapper must not assume that possession of a
`user_id` grants access.

### Scope Representation

The `owner` object is the canonical scope representation. Do not add a separate
serialized `scope` field because it would duplicate `owner.type` and could
become inconsistent with the owner identity.

| Memory scope | `owner.type` | `owner.id` |
| --- | --- | --- |
| Personal | `user` | User identity |
| Team | `team` | Team identity |
| Organization | `organization` | Organization identity |

Examples:

```json
{
  "owner": {
    "type": "user",
    "id": "u1"
  }
}
```

```json
{
  "owner": {
    "type": "team",
    "id": "payments"
  }
}
```

```json
{
  "owner": {
    "type": "organization",
    "id": "contoso"
  }
}
```

The ownership fields have distinct responsibilities:

- `owner.type` defines the scope level.
- `owner.id` identifies that scope.
- `provenance.createdBy` identifies the user or service that created the
  memory.
- `acl` defines which callers may read, write, annotate, or forget the memory.
- `payload.user_id` retains AMT subject and partition semantics; it does not
  define MemoryHouse scope or authorization.

For existing AMT memories, a trusted policy can create a personal scope:

```text
AMT user_id=u1
    ↓
owner={type: user, id: u1}
```

Team and organization scopes require explicit context from the writer because
they cannot be inferred from an AMT document. ACL principals should use a
consistent namespace such as `user:`, `team:`, `organization:`, and `service:`.

An SDK may expose a derived convenience property without adding it to the wire
format:

```python
@property
def scope(self) -> Literal["personal", "team", "organization"]:
    return {
        "user": "personal",
        "team": "team",
        "organization": "organization",
    }[self.owner.type]
```

## Annotation Policy

Annotations must not repeat AMT payload fields. A newly wrapped AMT memory
normally starts with an empty annotation map:

```json
{
  "annotations": {}
}
```

MemoryHouse enrichment pipelines should write their own named annotations:

```json
{
  "annotations": {
    "classification": {
      "domain": "payments",
      "sensitivity": "internal"
    },
    "search": {
      "embeddingReference": "embeddings/mem_123"
    },
    "summary": {
      "text": "User review-format preference."
    }
  }
}
```

Valid annotations are independently produced assertions such as
classification, generated summaries, or references to external indexes. AMT
fields such as `type`, `thread_id`, `role`, `tags`, `salience`, `confidence`,
and supersession state stay in the payload.

Enrichers must not rewrite the payload. If the AMT record changes, the producer
replaces the payload while preserving enrichment annotations owned by other
writers.

## Operation Mapping

| MemoryHouse operation | AMT wrapper behavior |
| --- | --- |
| Lookup | Authorize, return the envelope, and optionally unwrap the AMT record |
| Scan | Filter on owner, content type, provenance, ACL, and annotations |
| Write | Wrap `record.to_doc()` and create or replace the envelope |
| Annotate | Modify named annotations without modifying the payload |
| Forget | Delete the envelope and propagate deletion to serving projections when configured |

MemoryHouse lookup and scan events should be recorded by the MemoryHouse audit
layer. AMT logging is not a substitute for authorization or access audit.

## Storage and Serving Architecture

MemoryHouse should hold the governed canonical artifact. AMT's Cosmos
containers can remain product-specific serving projections optimized for
conversation replay, vector search, summaries, and procedural context.

```text
AMT producer
    │
    ▼
AMTMemoryEnvelopeAdapter.wrap()
    │
    ▼
MemoryHouse canonical envelope
    │
    ▼ change feed or outbox projector
AMT Cosmos serving containers and indexes
```

An asynchronous projector is preferred over uncoordinated synchronous dual
writes. It provides replay, observable failures, and a clear source of truth.
The projector validates that the content-type subtype agrees with
`payload.type`, then routes using the payload's `type`, `user_id`, and
`thread_id`.

During an incremental migration, the existing AMT store may temporarily remain
authoritative. In that mode, changes should be captured through an outbox or
Cosmos change feed and wrapped into MemoryHouse. The deployment must explicitly
declare which store is authoritative to avoid update loops.

## Updates, Supersession, and Forgetting

An in-place AMT update replaces the payload under the same MemoryHouse ID.
MemoryHouse enrichment annotations should be preserved unless their producer
explicitly replaces them.

AMT supersession remains exclusively in the payload. It is distinct from
MemoryHouse `Forget`:

- Supersession preserves history and marks one AMT memory as replaced.
- Forget removes the governed artifact according to authorization and
  retention policy.

If AMT Cosmos containers are retained as serving projections, forgetting must
publish a tombstone containing the AMT ID and routing fields so the projector
can delete the corresponding document.

## Failure Handling

- Reject writes without an owner or ACL.
- Reject unsupported content-type versions.
- Reject malformed payloads before projecting them into AMT.
- Do not report a successful write when either the canonical MemoryHouse write
  or its durable outbox record fails.
- Send projection failures to a retryable dead-letter path with the envelope
  ID and operation.
- Use version or ETag checks to prevent concurrent updates from overwriting a
  newer payload or annotation.

## Recommendation

Adopt the wrapper as an integration boundary rather than modifying AMT's
existing record hierarchy. The complete AMT document remains a lossless,
producer-owned payload, while MemoryHouse supplies the governance contract.
This lets AMT continue to evolve its memory types and serving strategy without
requiring changes to the MemoryHouse core.
