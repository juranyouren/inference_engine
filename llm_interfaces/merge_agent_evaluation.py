"""从各分片的 case 排名汇总两种 Agent 的评测结果。"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

from .evaluate import _metrics, _write_report
from .io_utils import safe_category_name


def merge_shards(output_dir, shard_count, wall_elapsed_seconds):
    if shard_count < 1:
        raise ValueError("分片总数必须大于 0")
    runs = set()
    records = defaultdict(dict)
    elapsed = defaultdict(float)
    excluded_without_sop = 0
    for index in range(shard_count):
        shard_dir = output_dir / f"shard_{index}"
        report = json.loads((shard_dir / "metrics.json").read_text(encoding="utf-8"))
        if (report["shard_index"], report["shard_count"]) != (index, shard_count):
            raise ValueError(f"{shard_dir} 的分片编号或分片总数不匹配")
        runs.add((report["mode"], report["label_root"], report.get("limit_per_mode")))
        excluded_without_sop += report["excluded_without_sop"]
        for row in report["metrics"]:
            mode, category = row["mode"], row["alarm_type"]
            if category is None:
                continue
            trace = shard_dir / "traces" / safe_category_name(category) / mode / "results.json"
            results = json.loads(trace.read_text(encoding="utf-8"))["results"]
            if len(results) != row["attempted_count"]:
                raise ValueError(f"{trace} 的 case 数与分片报告不一致")
            group = records[mode, category]
            for case in results:
                case_index = case["case_idx"]
                if case_index % shard_count != index or case_index in group:
                    raise ValueError(f"{category} 的 case {case_index} 分片归属错误或重复")
                group[case_index] = {
                    field: case[field] for field in ("status", "groundtruth", "rank_v1", "parse_strategy")
                }
            elapsed[mode, category] += row["elapsed_seconds"]

    if len(runs) != 1:
        raise ValueError("各分片的数据集路径、评测模式或数量上限不一致")
    run_mode, label_root, limit_per_mode = runs.pop()
    if not records:
        raise ValueError("各分片均无可评测样本。若 SOP 在 JSON 顶层，可加 --import-top-level-sop")

    rows = []
    for mode in (("competition", "cooperation") if run_mode == "both" else (run_mode,)):
        all_cases = []
        total_elapsed = 0.0
        for candidate_mode, category in sorted(records):
            if candidate_mode != mode:
                continue
            cases = list(records[mode, category].values())
            duration = elapsed[mode, category]
            rows.append(_metrics(mode, category, cases, duration))
            all_cases.extend(cases)
            total_elapsed += duration
        rows.append(_metrics(mode, None, all_cases, total_elapsed))

    _write_report(output_dir, {
        "mode": run_mode, "label_root": label_root, "shard_count": shard_count,
        "limit_per_mode": limit_per_mode,
        "excluded_without_sop": excluded_without_sop, "wall_elapsed_seconds": wall_elapsed_seconds,
    }, rows)
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--shard-count", type=int, required=True)
    parser.add_argument("--wall-seconds", type=int, required=True)
    args = parser.parse_args()
    try:
        rows = merge_shards(args.output_dir, args.shard_count, args.wall_seconds)
    except (OSError, KeyError, TypeError, ValueError) as exc:
        parser.error(str(exc))
    print(f"汇总结果已写入 {args.output_dir.resolve()}")
    return 1 if any(row["failed_count"] for row in rows) else 0


if __name__ == "__main__":
    raise SystemExit(main())
