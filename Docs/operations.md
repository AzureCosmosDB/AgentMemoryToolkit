# Operations

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
