# Durable orchestrator activity chains

W15 splits LLM extraction from persistence so Durable retries after a Cosmos write failure do not re-run the LLM activity.

```text
ExtractMemoriesOrchestrator
  em_Extract  (load recent turns + LLM + parse; no embeddings/writes)
  em_Persist  (embeddings + deterministic create_item; 409 = already persisted)
  em_ReconcileMemories (optional; single activity for GA)

ThreadSummaryOrchestrator
  ts_Extract
  ts_PersistSummary

UserSummaryOrchestrator
  us_Extract
  us_PersistUserSummary

SynthesizeProceduralOrchestrator
  sp_SynthesizeProcedural (single activity for GA)
```

Fact and episodic IDs are deterministic from user, thread, and normalized content. Thread and user summaries keep their deterministic summary IDs.
