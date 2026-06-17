#!/usr/bin/env python
"""
Real Cosmos DB consistency test.
Run: python benchmarks/test_cosmos_consistency.py
"""

import os
import sys
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from azure.cosmos.agent_memory import CosmosMemoryClient
except ImportError:
    from agent_memory_toolkit import CosmosMemoryClient

from benchmarks.consistency.probe import CosmosStoreAdapter, run_probe
from benchmarks.consistency.analyzer import analyze
from benchmarks.consistency.sweep import write_leaderboard_csv


def create_cosmos_client(endpoint: str, master_key: str):
    """Create a CosmosMemoryClient for real Cosmos DB."""
    client = CosmosMemoryClient(
        cosmos_endpoint=endpoint,
        cosmos_key=master_key,
        cosmos_database="agent_memory_db",
        cosmos_container="memory",
        use_default_credential=False,
        ai_foundry_endpoint=None,  # Skip AI Foundry for this test
    )
    return client


def cosmos_store_factory(endpoint: str, master_key: str):
    """Create a CosmosStoreAdapter for real Cosmos DB."""
    client = create_cosmos_client(endpoint, master_key)
    
    # Shared partition keys for multi-agent scenario
    user_id = "test_user"
    thread_id = "test_thread"
    
    adapter = CosmosStoreAdapter(client, user_id=user_id, thread_id=thread_id)
    return adapter


def main():
    print("=== Real Cosmos DB Consistency Test ===\n")
    
    endpoint = os.getenv("COSMOS_ENDPOINT", "https://agentmemorytest.documents.azure.com:443/")
    master_key = os.getenv("COSMOS_MASTER_KEY", "")
    
    if not master_key:
        print("Error: COSMOS_MASTER_KEY environment variable not set")
        print("Set it to your Cosmos primary key and try again")
        return 1
    
    print(f"Endpoint: {endpoint}")
    print(f"Database: agent_memory_db (or auto-created)")
    print(f"Container: memory (or auto-created)\n")
    
    try:
        # Test 1: Single agent, session consistency
        print("Test 1: Single agent, session consistency")
        store1 = cosmos_store_factory(endpoint, master_key)
        agents = ["agent1"]
        keys = ["key1", "key2"]
        
        trace = run_probe(
            store=store1,
            agents=agents,
            keys=keys,
            ops_per_agent=20,
        )
        
        report = analyze(trace)
        print(f"  Reads: {report.reads}, Writes: {report.writes}")
        print(f"  Stale reads: {report.stale_reads} ({report.stale_read_rate:.1%})")
        print(f"  Max staleness: delta_max={report.delta_max:.6f}s, k_max={report.k_max} versions")
        print(f"  RYW violations: {report.read_your_writes_violations}\n")
        
        # Test 2: Multi-agent, eventual consistency
        print("Test 2: Multi-agent (3 agents), eventual consistency")
        store2 = cosmos_store_factory(endpoint, master_key)
        agents = ["agent1", "agent2", "agent3"]
        keys = ["key1", "key2"]
        
        trace = run_probe(
            store=store2,
            agents=agents,
            keys=keys,
            ops_per_agent=20,
        )
        
        report = analyze(trace)
        print(f"  Reads: {report.reads}, Writes: {report.writes}")
        print(f"  Stale reads: {report.stale_reads} ({report.stale_read_rate:.1%})")
        print(f"  Max staleness: delta_max={report.delta_max:.6f}s, k_max={report.k_max} versions")
        print(f"  RYW violations: {report.read_your_writes_violations}\n")
        
        print("[OK] Real Cosmos DB consistency tests passed!")
        return 0
        
    except Exception as exc:
        print(f"[FAIL] Test failed: {exc}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
