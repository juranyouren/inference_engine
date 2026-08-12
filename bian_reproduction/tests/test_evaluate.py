# -*- coding: utf-8 -*-

import json
from pathlib import Path

import pytest

from bian_reproduction.evaluate import (
    build_evaluation_report,
    calculate_metrics,
    discover_prediction_files,
    evaluate_payloads,
)


def test_calculate_metrics_uses_strict_ranking_and_skipped_misses() -> None:
    results = [
        {"groundtruth": "A", "rank": 1, "parse_strategy": "scored_ranking"},
        {"groundtruth": "B", "rank": 3, "parse_strategy": "ranking_only"},
        {"groundtruth": "C", "rank": 1, "parse_strategy": "candidate_order_fallback"},
        {"groundtruth": "D", "rank": None, "parse_strategy": "unparsed"},
    ]
    metrics = calculate_metrics(
        results,
        timing={"elapsed_seconds": 10.0, "model_init_seconds": 2.0},
        skipped_as_miss_count=1,
    )

    assert metrics["evaluated_count"] == 5
    assert metrics["invalid_prediction_count"] == 2
    assert metrics["top1"] == pytest.approx(1 / 5)
    assert metrics["top3"] == pytest.approx(2 / 5)
    assert metrics["top5"] == pytest.approx(2 / 5)
    assert metrics["mrr"] == pytest.approx((1 + 1 / 3) / 5)
    assert metrics["average_seconds_per_case"] == 2.0


def test_report_aggregates_alarm_types_and_timing(tmp_path: Path) -> None:
    payloads = [
        {
            "meta": {
                "alarm_type": "场景A",
                "timing": {"elapsed_seconds": 2.0, "model_init_seconds": 1.0},
            },
            "summary": {"skipped_count": 0},
            "results": [
                {"groundtruth": "A", "rank": 1, "parse_strategy": "scored_ranking"}
            ],
        },
        {
            "meta": {
                "alarm_type": "场景B",
                "timing": {"elapsed_seconds": 4.0, "model_init_seconds": 0.0},
            },
            "summary": {"skipped_count": 1},
            "results": [
                {"groundtruth": "B", "rank": 2, "parse_strategy": "ranking_only"}
            ],
        },
    ]

    report = build_evaluation_report(payloads)
    overall = next(row for row in report["metrics"] if row["scope"] == "overall")
    assert overall["evaluated_count"] == 3
    assert overall["top1"] == pytest.approx(1 / 3)
    assert overall["top3"] == pytest.approx(2 / 3)
    assert overall["mrr"] == pytest.approx(0.5)
    assert overall["elapsed_seconds"] == 6.0
    assert overall["average_seconds_per_case"] == 2.0

    evaluate_payloads(payloads, tmp_path / "evaluation")
    assert (tmp_path / "evaluation" / "metrics.json").is_file()
    assert (tmp_path / "evaluation" / "metrics.csv").is_file()


def test_discover_prediction_files_accepts_result_directory(tmp_path: Path) -> None:
    run_dir = tmp_path / "idx_0_0"
    run_dir.mkdir()
    prediction_file = run_dir / "predictions.json"
    prediction_file.write_text(json.dumps({"results": []}), encoding="utf-8")

    assert discover_prediction_files([str(run_dir)]) == [prediction_file.resolve()]
