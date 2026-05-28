# Operations

Runtime knobs for an Agent-Memory-Toolkit deployment. Most ops levers live in `.env` / Function-app App Settings — change them, restart the consumer, and you're done. Deployment-time knobs (Bicep params bound to `azd env set ...`) live in [`infra/README.md`](../infra/README.md).

## Memory lifecycle (TTL)

| Type | Default TTL | Source |
|---|---:|---|
| turn | 30 d | container default (memories_turns) |
| episodic | 90 d | per-doc ttl (memories container) |
| thread_summary | never | container default (memories, -1) |
| user_summary | never | container default |
| fact | never | container default; supersession handles aging |
| procedural | never | container default; supersession handles aging |

Override per write:

    client.add_memory(text, type="turn", ttl=60)   # expires in 60 seconds

Override per container at provision time:

    azd env set MEMORIES_TURNS_DEFAULT_TTL 86400   # 1 day

## Counter-based trigger configuration

Function-app threshold knobs (`THREAD_SUMMARY_EVERY_N`, `FACT_EXTRACTION_EVERY_N`, `DEDUP_EVERY_N`, `USER_SUMMARY_EVERY_N`, `MAX_BATCH_SIZE`, `MEMORY_PROCESSOR_OWNER`) are documented in [`infra/README.md` → Counter-based trigger configuration](../infra/README.md#counter-based-trigger-configuration-function-app-only). Change them with `azd env set ...` then `azd up`.

