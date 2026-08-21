# MemoryHouse Envelope Implementation Tasks

## Task Conventions

- Tasks are ordered by dependency.
- **P0** is required for the lossless adapter milestone.
- **P1** is required for service integration.
- **P2** is required for production replication.
- A task is complete only when its acceptance criteria and tests pass.

## Phase 1: Contracts and Lossless Adapter

### MH-001 — Confirm the MemoryHouse wire contract

**Priority:** P0

**Work**

- Confirm required fields, JSON casing, owner values, owner-as-scope semantics,
  ACL operations, annotation merge semantics, and content-type parsing rules.
- Determine whether an official MemoryHouse Python model package exists.
- Record any deviations from
  [`memoryhouse-amt-envelope.md`](memoryhouse-amt-envelope.md).

**Acceptance criteria**

- The approved wire contract is documented.
- The implementation team knows whether to use official or local Pydantic
  models.
- No unresolved field-name or required-field ambiguity remains.

### MH-002 — Add the MemoryHouse integration package

**Priority:** P0
**Depends on:** MH-001

**Files**

- `azure/cosmos/agent_memory/memoryhouse/__init__.py`
- `azure/cosmos/agent_memory/memoryhouse/models.py`
- `azure/cosmos/agent_memory/memoryhouse/exceptions.py`

**Work**

- Create the package.
- Add or import `MemoryEnvelope`, `MemoryOwner`, `MemoryACL`, and
  `MemoryProvenance`.
- Add AMT-specific integration exceptions.
- Configure camel-case aliases and strict governance validation.
- Add a non-serialized `scope` convenience property derived from `owner.type`.

**Acceptance criteria**

- Valid canonical examples parse and serialize to the approved wire shape.
- Missing owner, provenance, ACL, annotations, or payload fields are rejected.
- A serialized `scope` field is rejected.
- `owner.type` maps `user`, `team`, and `organization` to personal, team, and
  organization scope.
- Unknown standardized envelope fields are rejected.
- Unknown annotation namespaces are preserved.

### MH-003 — Implement envelope ID helpers

**Priority:** P0
**Depends on:** MH-002

**Files**

- `azure/cosmos/agent_memory/memoryhouse/adapter.py`
- `tests/unit/memoryhouse/test_adapter.py`

**Work**

- Implement `build_envelope_id(store_id, memory_id)`.
- Implement `parse_envelope_id(envelope_id)`.
- Percent-encode path segments.
- Reject empty IDs, malformed escapes, extra segments, and invalid prefixes.

**Acceptance criteria**

- IDs follow `amt/{encoded-store-id}/{encoded-amt-id}`.
- Every valid ID round-trips without information loss.
- Equivalent inputs always produce the same ID.

### MH-004 — Implement AMT subtype content types

**Priority:** P0
**Depends on:** MH-002

**Files**

- `azure/cosmos/agent_memory/memoryhouse/adapter.py`

**Work**

- Define a versioned content type for every AMT record type.
- Implement content-type generation and parsing.
- Normalize AMT names such as `thread_summary` to MIME-safe subtype names.
- Reject unknown subtypes and versions.

**Acceptance criteria**

- Every supported AMT record type maps to exactly one content type.
- Content-type parsing returns the corresponding AMT record type.
- The mapping round-trips for all six record types.
- No AMT payload field is copied into annotations.

### MH-005 — Implement `AMTMemoryEnvelopeAdapter.wrap`

**Priority:** P0
**Depends on:** MH-003, MH-004

**Files**

- `azure/cosmos/agent_memory/memoryhouse/adapter.py`
- `tests/unit/memoryhouse/test_adapter.py`

**Work**

- Define versioned content-type constants.
- Serialize with `record.to_doc()`.
- Build the envelope ID, owner, ACL, provenance, annotations, and payload.
- Require explicit `created_by`.
- Require explicit owner context unless a trusted personal-scope policy is
  injected.
- Deep-copy mutable values.

**Acceptance criteria**

- Every supported AMT record subtype can be wrapped.
- The payload equals the source record's `to_doc()` result.
- Wrapping does not mutate the AMT record or caller annotations.
- Wrapping does not serialize a separate `scope` field.
- Prompt and source metadata are copied into provenance when present.

### MH-006 — Implement `AMTMemoryEnvelopeAdapter.unwrap`

**Priority:** P0
**Depends on:** MH-005

**Files**

- `azure/cosmos/agent_memory/memoryhouse/adapter.py`
- `tests/unit/memoryhouse/test_adapter.py`

**Work**

- Validate media type and version.
- Verify envelope ID, provenance source ID, and payload ID.
- Verify the content-type subtype matches `payload.type`.
- Reconstruct records through `MemoryRecordBase.from_doc()`.
- Support explicit strict integrity checking.

**Acceptance criteria**

- Unwrapping returns the correct AMT subtype.
- Unsupported types and versions raise explicit errors.
- Malformed payloads and integrity mismatches are rejected.
- The payload remains authoritative when diagnostic mismatch reporting is used.

### MH-007 — Add round-trip tests for all record types

**Priority:** P0
**Depends on:** MH-006

**Files**

- `tests/unit/memoryhouse/test_adapter.py`
- `tests/unit/memoryhouse/test_models.py`

**Work**

- Cover turn, thread summary, user summary, fact, episodic, and procedural
  records.
- Include optional fields, embeddings, lineage, supersession, tags, and subtype
  structures.
- Test historical document shapes already supported by AMT.

**Acceptance criteria**

For every fixture:

```python
adapter.unwrap(adapter.wrap(record, ...)).to_doc() == record.to_doc()
```

- No existing AMT tests need behavior changes.
- Negative tests cover every adapter exception.

### MH-008 — Publish the adapter API and documentation

**Priority:** P0
**Depends on:** MH-007

**Files**

- `azure/cosmos/agent_memory/memoryhouse/__init__.py`
- `Docs/public_api.md`
- `README.md`
- `CHANGELOG.md`

**Work**

- Export integration types from the `memoryhouse` subpackage.
- Add a minimal wrapping and unwrapping example.
- Document that authorization occurs before unwrapping.
- Document supported content-type versions.

**Acceptance criteria**

- Users can import the adapter from
  `azure.cosmos.agent_memory.memoryhouse`.
- Documentation does not imply that `user_id` is an ACL.
- The feature is documented as opt-in.

## Phase 2: MemoryHouse Service Integration

### MH-009 — Define sync and async client protocols

**Priority:** P1
**Depends on:** MH-001, MH-008

**Files**

- `azure/cosmos/agent_memory/memoryhouse/protocols.py`

**Work**

- Define lookup, scan, write, annotate, and forget contracts.
- Define scan filters and pagination representation.
- Define optimistic-concurrency inputs and outputs.
- Define equivalent async protocols.

**Acceptance criteria**

- Protocols can be satisfied by fakes without importing a concrete SDK.
- Sync and async operations have equivalent result and error semantics.
- Pagination and conflict behavior are unambiguous.

### MH-010 — Add fake MemoryHouse clients

**Priority:** P1
**Depends on:** MH-009

**Files**

- `tests/unit/memoryhouse/fakes.py`

**Work**

- Implement deterministic in-memory sync and async fakes.
- Enforce ID uniqueness, version conflicts, annotation merges, and forget.
- Record calls for assertions.

**Acceptance criteria**

- Fakes satisfy the protocols.
- Tests can simulate authorization, not-found, conflict, and transport errors.

### MH-011 — Implement the sync MemoryHouse service

**Priority:** P1
**Depends on:** MH-010

**Files**

- `azure/cosmos/agent_memory/memoryhouse/service.py`
- `tests/unit/memoryhouse/test_service.py`

**Work**

- Implement wrapped write, typed lookup, scan, annotate, and forget.
- Preserve enrichment annotations during producer payload updates.
- Keep annotations limited to independently produced enrichment.
- Map concrete-client failures to integration exceptions.

**Acceptance criteria**

- All five operations work through the fake client.
- Authorization and transport errors are never returned as empty results.
- Payload replacement does not erase enrichment annotations.

### MH-012 — Implement the async MemoryHouse service

**Priority:** P1
**Depends on:** MH-011

**Files**

- `azure/cosmos/agent_memory/memoryhouse/aio/__init__.py`
- `azure/cosmos/agent_memory/memoryhouse/aio/service.py`
- `tests/unit/aio/memoryhouse/test_service.py`

**Work**

- Implement native async operations.
- Match sync validation, error mapping, and annotation behavior.
- Do not use thread-pool wrappers for service calls.

**Acceptance criteria**

- Async tests cover parity with every sync operation and failure mode.
- No blocking transport calls run on the event loop.

### MH-013 — Integrate the selected MemoryHouse SDK

**Priority:** P1
**Depends on:** MH-009, official SDK decision

**Work**

- Add an optional dependency only if required.
- Implement sync and async protocol adapters.
- Configure authentication externally.
- Add contract tests against a test service or emulator.

**Acceptance criteria**

- The SDK adapter passes the same contract suite as the fake clients.
- Credentials are not stored in envelopes, logs, or configuration files.
- Package users who do not use MemoryHouse do not need the optional dependency.

## Phase 3: AMT-Authoritative Mirror

### MH-014 — Define scope, owner, and ACL resolution policy

**Priority:** P2
**Depends on:** MH-001

**Work**

- Define the trusted interface that maps an AMT document and deployment context
  to `MemoryOwner` and `MemoryACL`.
- Define `owner.type` as the canonical personal, team, or organization scope.
- Define user, team, organization, and service principal syntax.
- Define failure behavior when policy cannot resolve governance.

**Acceptance criteria**

- Existing AMT records may map `user_id` to personal scope through trusted
  policy.
- Team and organization scope never relies on inference from `user_id`.
- Creator identity is recorded in provenance and is not conflated with scope.
- Unresolved ownership or ACL policy blocks mirroring and produces an
  actionable failure.
- Policy decisions are testable without a live identity provider.

### MH-015 — Design the change-feed mirror checkpoint

**Priority:** P2
**Depends on:** MH-013, MH-014

**Work**

- Define event identity, ordering, retry state, and idempotency keys.
- Account for AMT's three Cosmos containers.
- Define dead-letter records without copying memory payload content.
- Define operational replay controls.

**Acceptance criteria**

- Replaying an event cannot create a second envelope ID.
- Updates cannot overwrite a newer mirrored envelope.
- Operators can identify and retry failed envelope IDs.

### MH-016 — Implement AMT-to-MemoryHouse mirroring

**Priority:** P2
**Depends on:** MH-015

**Files**

- New Function App trigger or dedicated projector package
- Unit and integration tests

**Work**

- Consume changes from turns, memories, and summaries containers.
- Parse documents with `MemoryRecordBase.from_doc()`.
- Resolve governance, wrap, and write envelopes.
- Add bounded retries and dead-letter handling.

**Acceptance criteria**

- Every AMT memory type is mirrored.
- Replay is idempotent.
- Invalid AMT documents are dead-lettered with a safe diagnostic.
- Existing AMT write latency is unaffected.

### MH-017 — Add mirror reconciliation

**Priority:** P2
**Depends on:** MH-016

**Work**

- Scan AMT documents and expected envelope IDs.
- Detect missing, stale, and malformed mirrored envelopes.
- Support report-only and repair modes.

**Acceptance criteria**

- Report-only mode performs no writes.
- Repair mode is idempotent and uses optimistic concurrency.
- Metrics expose lag, failures, and divergence counts.

### MH-018 — Implement delete and forget tombstones

**Priority:** P2
**Depends on:** MH-015

**Work**

- Define a durable tombstone schema with AMT routing fields.
- Emit tombstones before deleting authoritative AMT documents where required.
- Propagate authorized forget operations.
- Make tombstone replay idempotent.

**Acceptance criteria**

- Forget reaches all configured stores.
- Replayed tombstones do not fail if the target is already absent.
- Deletion routing is validated against the last known envelope.

## Phase 4: MemoryHouse-Canonical Projection

### MH-019 — Add authority-mode configuration

**Priority:** P2
**Depends on:** MH-016

**Work**

- Add `disabled`, `amt_mirror`, and `memoryhouse_canonical` modes.
- Validate configuration at startup.
- Prevent AMT mirror and canonical projection from forming an update loop.

**Acceptance criteria**

- Invalid or conflicting configurations fail startup.
- The default remains `disabled`.
- Existing deployments retain current behavior without configuration changes.

### MH-020 — Define durable projection events

**Priority:** P2
**Depends on:** MH-019

**Work**

- Define upsert and forget event schemas.
- Include envelope version or ETag and AMT routing data.
- Select the outbox or event transport.
- Define retry and dead-letter behavior.

**Acceptance criteria**

- A successful canonical write always has a durable projection event.
- Events are safe to replay and contain no credentials.
- Ordering and stale-event rejection rules are documented.

### MH-021 — Implement MemoryHouse-to-AMT projection

**Priority:** P2
**Depends on:** MH-020

**Files**

- `azure/cosmos/agent_memory/memoryhouse/projection.py`
- Projector host and tests

**Work**

- Validate and unwrap envelopes.
- Route records using the validated payload type.
- Upsert through existing `MemoryStore` behavior.
- Process forget tombstones.
- Checkpoint envelope versions.

**Acceptance criteria**

- All six AMT types reach the correct Cosmos container.
- Partition keys come from the validated payload.
- Replayed and stale events do not corrupt serving state.
- Existing retrieval APIs return projected records unchanged.

### MH-022 — Add canonical-mode client integration

**Priority:** P2
**Depends on:** MH-021

**Work**

- Decide whether to add a new governed client or opt-in methods on existing
  clients.
- Route canonical writes through `MemoryHouseService`.
- Preserve existing direct-Cosmos methods for compatibility unless explicitly
  deprecated.
- Surface projection status separately from canonical write success.

**Acceptance criteria**

- No existing API changes behavior unless canonical mode is explicitly enabled.
- Canonical write and projection status are distinguishable.
- Sync and async clients provide equivalent behavior.

### MH-023 — Add end-to-end reliability tests

**Priority:** P2
**Depends on:** MH-022

**Work**

- Test write, update, annotate, lookup, scan, supersede, and forget.
- Test process crashes between canonical write and projection.
- Test retries, duplicate events, stale events, and unavailable targets.
- Test recovery from dead-letter and reconciliation repair.

**Acceptance criteria**

- No acknowledged canonical write is permanently lost from projection.
- Replay does not create duplicates.
- Forget is eventually reflected in every configured serving store.
- Existing AMT retrieval and processing pipelines continue to pass.

### MH-024 — Add operational documentation and rollout gates

**Priority:** P2
**Depends on:** MH-023

**Files**

- `Docs/operations.md`
- `Docs/troubleshooting.md`
- `Docs/MemoryHouseEnvelope/`

**Work**

- Document configuration, identity, monitoring, replay, dead-letter recovery,
  reconciliation, and rollback.
- Define pilot and production rollout metrics.
- Define procedures for switching authority modes.

**Acceptance criteria**

- Operators can diagnose mirror lag and failed projections.
- Rollback does not require deleting canonical memories.
- Production enablement requires passing documented reliability gates.

## Cross-Cutting Validation

Run the smallest relevant checks during each task and the complete suite at
phase boundaries:

```bash
ruff check azure/cosmos/agent_memory/memoryhouse tests/unit/memoryhouse
pytest -q tests/unit/memoryhouse tests/unit/aio/memoryhouse
pytest -q tests/unit
```

Integration phases must also run their targeted live-service tests using the
project's existing `integration`, `slow`, and `e2e` markers.
