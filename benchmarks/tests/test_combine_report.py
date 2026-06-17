"""Unit tests for the MAB-accuracy x consistency-leaderboard reducer."""

import json

from benchmarks.combine_report import (
    build_combined_report,
    flat_rows,
    load_leaderboard,
    load_mab_result,
    write_csv,
    write_json,
)


def _write_mab_result(path, *, run_id, writer_agents, query_agents, accuracy):
    doc = {
        "agent_config": {
            "agent_name": "agent_memory_toolkit",
            "model": "gpt-4o-mini",
            "memory_toolkit_run_id": run_id,
            "memory_toolkit_store_mode": "facts_only",
            "memory_toolkit_search_mode": "hybrid",
            "memory_toolkit_writer_agents": writer_agents,
            "memory_toolkit_query_agents": query_agents,
        },
        "dataset_config": {"dataset": "Ruler", "sub_dataset": "qa_1"},
        "data": [{"query": "q1"}, {"query": "q2"}],
        "metrics": {"accuracy": [1.0, 0.0]},
        "time_cost": [1.5, 3.0],
        "averaged_metrics": {**accuracy, "query_time_len": 0.42, "input_len": 1200},
    }
    path.write_text(json.dumps(doc), encoding="utf-8")
    return str(path)


def _write_leaderboard(path):
    path.write_text(
        "level,concurrency,reads,writes,stale_reads,stale_read_rate,delta_max,"
        "delta_mean,delta_p95,k_max,k_mean,read_your_writes_violations,"
        "monotonic_reads_violations\n"
        "strong,1,10,10,0,0.0,0.0,0.0,0.0,0,0.0,0,0\n"
        "strong,2,20,20,0,0.0,0.0,0.0,0.0,0,0.0,0,0\n"
        "eventual,1,10,10,5,0.5,0.003,0.001,0.0025,4,2.0,3,1\n"
        "eventual,2,20,20,12,0.6,0.004,0.0015,0.0031,9,4.0,7,2\n",
        encoding="utf-8",
    )
    return str(path)


def test_load_mab_result_separates_accuracy_and_timing(tmp_path):
    p = _write_mab_result(
        tmp_path / "r_results.json",
        run_id="exp1",
        writer_agents=["a", "b"],
        query_agents=[],
        accuracy={"accuracy": 80.0},
    )
    run = load_mab_result(p)
    assert run["run_id"] == "exp1"
    assert run["accuracy"] == {"accuracy": 80.0}
    assert run["timing"] == {"query_time_len": 0.42, "input_len": 1200}
    assert run["concurrency"] == 2  # len(writer_agents)
    assert run["retrieval_scope"] == "shared"
    assert run["num_queries"] == 2
    assert run["wall_clock_s"] == 3.0


def test_consistency_summary_exact_match_with_level(tmp_path):
    run = load_mab_result(
        _write_mab_result(
            tmp_path / "r_results.json",
            run_id="exp1",
            writer_agents=["a", "b"],
            query_agents=["a"],
            accuracy={"accuracy": 70.0},
        )
    )
    leaderboard = load_leaderboard(_write_leaderboard(tmp_path / "lb.csv"))
    report = build_combined_report([run], leaderboard, level="eventual")
    cons = report["runs"][0]["consistency_summary"]
    assert cons["matched"] is True
    assert cons["level"] == "eventual"
    assert cons["concurrency"] == 2
    assert cons["k_max"] == 9
    assert cons["stale_read_rate"] == 0.6


def test_consistency_summary_worst_case_without_level(tmp_path):
    run = load_mab_result(
        _write_mab_result(
            tmp_path / "r_results.json",
            run_id="exp1",
            writer_agents=["a", "b"],
            query_agents=[],
            accuracy={"accuracy": 70.0},
        )
    )
    leaderboard = load_leaderboard(_write_leaderboard(tmp_path / "lb.csv"))
    report = build_combined_report([run], leaderboard)  # no level
    cons = report["runs"][0]["consistency_summary"]
    assert cons["matched"] is False
    # concurrency==2 rows: strong(k0) and eventual(k9) -> worst-case k_max=9
    assert cons["k_max"] == 9


def test_run_id_filter(tmp_path):
    run_a = load_mab_result(
        _write_mab_result(
            tmp_path / "a_results.json",
            run_id="A",
            writer_agents=[],
            query_agents=[],
            accuracy={"accuracy": 50.0},
        )
    )
    run_b = load_mab_result(
        _write_mab_result(
            tmp_path / "b_results.json",
            run_id="B",
            writer_agents=[],
            query_agents=[],
            accuracy={"accuracy": 90.0},
        )
    )
    report = build_combined_report([run_a, run_b], [], run_id="B")
    assert len(report["runs"]) == 1
    assert report["runs"][0]["run_id"] == "B"


def test_flat_rows_and_csv_roundtrip(tmp_path):
    run = load_mab_result(
        _write_mab_result(
            tmp_path / "r_results.json",
            run_id="exp1",
            writer_agents=["a", "b"],
            query_agents=["a"],
            accuracy={"accuracy": 70.0, "sub_em": 65.0},
        )
    )
    leaderboard = load_leaderboard(_write_leaderboard(tmp_path / "lb.csv"))
    report = build_combined_report([run], leaderboard, level="eventual")

    rows = flat_rows(report)
    assert len(rows) == 1
    row = rows[0]
    assert row["primary_accuracy_metric"] == "accuracy"  # highest-valued metric
    assert row["primary_accuracy_value"] == 70.0
    assert row["cons_k_max"] == 9
    assert row["writer_agents"] == "a;b"

    json_out = write_json(report, tmp_path / "combined.json")
    csv_out = write_csv(rows, tmp_path / "combined.csv")
    assert json.loads(open(json_out, encoding="utf-8").read())["runs"]
    lines = open(csv_out, encoding="utf-8").read().strip().splitlines()
    assert lines[0].startswith("run_id,agent_name,model,dataset,sub_dataset")
    assert len(lines) == 2
