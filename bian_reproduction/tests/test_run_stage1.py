# -*- coding: utf-8 -*-

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Sequence

from bian_reproduction.run_stage1 import build_parser, run_alarm_types


class RecordingRunner:
    def __init__(self) -> None:
        self.calls: List[Dict[str, Any]] = []

    def run(
        self,
        data_dir: str,
        alarm_type: str,
        indices: Sequence[int],
        output_dir: str,
    ) -> Dict[str, Any]:
        self.calls.append({
            "data_dir": data_dir,
            "alarm_type": alarm_type,
            "indices": list(indices),
            "output_dir": output_dir,
        })
        return {
            "meta": {
                "alarm_type": alarm_type,
                "output_dir": str(Path(output_dir) / alarm_type),
                "timing": {"elapsed_seconds": 1.0},
            },
            "summary": {
                "processed_count": len(indices),
                "skipped_count": 0,
                "processed_indices": list(indices),
                "skipped_indices": [],
            },
            "results": [
                {
                    "groundtruth": "根因A",
                    "rank": 1,
                    "parse_strategy": "scored_ranking",
                }
                for _index in indices
            ],
        }


def _write_label(data_dir: Path, alarm_type: str, index: int) -> None:
    data_dir.mkdir(parents=True, exist_ok=True)
    (data_dir / f"{alarm_type}_{index}_label_sample.json").write_text(
        json.dumps({"root_cause": {"category": "根因A"}}, ensure_ascii=False),
        encoding="utf-8",
    )


def test_parser_accepts_multiple_alarm_types() -> None:
    args = build_parser().parse_args([
        "--input-root", "/tmp/input",
        "--alarm-types", "类别A", "类别B", "类别C", "类别D",
    ])

    assert args.alarm_types == ["类别A", "类别B", "类别C", "类别D"]


def test_run_alarm_types_reuses_runner_and_writes_overall_metrics(
    tmp_path: Path,
) -> None:
    alarm_types = ["类别A", "类别B", "类别C", "类别D"]
    input_root = tmp_path / "input"
    for alarm_type in alarm_types:
        _write_label(input_root / alarm_type, alarm_type, 0)
        _write_label(input_root / alarm_type, alarm_type, 1)

    output_dir = tmp_path / "output"
    args = argparse.Namespace(
        input_root=str(input_root),
        alarm_types=alarm_types,
        output_dir=str(output_dir),
        model_path="unused",
        visible_devices="",
        batch_size=4,
        max_new_tokens=128,
        start=0,
        end=2,
    )
    runner = RecordingRunner()

    payloads, aggregate_dir = run_alarm_types(args, runner=runner)

    assert len(payloads) == 4
    assert [call["alarm_type"] for call in runner.calls] == alarm_types
    assert all(call["indices"] == [0, 1] for call in runner.calls)
    assert (aggregate_dir / "metrics.json").is_file()
    assert (aggregate_dir / "metrics.csv").is_file()
    report = json.loads(
        (aggregate_dir / "metrics.json").read_text(encoding="utf-8")
    )
    overall = next(row for row in report["metrics"] if row["scope"] == "overall")
    assert overall["evaluated_count"] == 8
    assert overall["top1"] == 1.0
