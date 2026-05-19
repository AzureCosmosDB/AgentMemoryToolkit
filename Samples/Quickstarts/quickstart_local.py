"""Quickstart: local-only CosmosMemoryClient demo (no cloud credentials needed).

Run with:
    python Samples/Quickstarts/quickstart_local.py
"""

from agent_memory_toolkit import CosmosMemoryClient


def main():
    mem = CosmosMemoryClient(use_default_credential=False)

    # --- Add memories ---
    mem.add_local(user_id="user1", role="user", content="I prefer dark mode.", memory_type="fact", thread_id="t1")
    mem.add_local(
        user_id="user1", role="agent", content="Noted! I'll remember that.",
        memory_type="turn", thread_id="t1",
    )
    mem.add_local(
        user_id="user2", role="user", content="My favorite language is Python.",
        memory_type="fact", thread_id="t2",
    )
    print("✅ Added 3 memories\n")

    # --- Get all ---
    all_mems = mem.get_local()
    print(f"All memories ({len(all_mems)}):")
    for m in all_mems:
        print(f"  [{m['role']}] {m['content']}")

    # --- Filtered get ---
    user1_facts = mem.get_local(user_id="user1", role="user")
    print(f"\nFiltered (user1, role=user): {len(user1_facts)} result(s)")
    for m in user1_facts:
        print(f"  [{m['type']}] {m['content']}")

    # --- Update ---
    target_id = all_mems[0]["id"]
    mem.update_local(memory_id=target_id, content="I prefer light mode now.")
    updated = mem.get_local(user_id="user1", role="user")
    print(f"\n✏️  Updated memory {target_id}:")
    print(f"  New content: {updated[0]['content']}")

    # --- Delete ---
    mem.delete_local(memory_id=target_id)
    remaining = mem.get_local()
    print(f"\n🗑️  Deleted memory {target_id}. Remaining: {len(remaining)}")


if __name__ == "__main__":
    main()
