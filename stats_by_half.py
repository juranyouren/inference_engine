# -*- coding: utf-8 -*-
"""按每类前一百 / 后一百分别统计指标。

从实验输出目录读取 predictions.json，将每个 scenario 的 case 按 case_idx
升序排列后分为前一半（前 100 条）和后一半（后 100 条），分别计算 Top-1/3/5、
MRR、unparsed 等指标，输出 JSON 和 CSV。
"""

import argparse
import csv
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

METHODS = ("tree", "competition", "cooperation")


# ---------------------------------------------------------------------------
# 工具函数
# ---------------------------------------------------------------------------

def load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(data: Any, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def calculate_metrics(
    results: Sequence[Dict[str, Any]],
    elapsed_seconds: float = 0.0,
) -> Dict[str, Any]:
    """与 run_three_method_experiment.calculate_metrics 保持一致的指标计算。"""
    labeled = [item for item in results if item.get("groundtruth")]
    ranks = [item.get("rank") for item in labeled]
    total = len(labeled)

    def top_k(k: int) -> Optional[float]:
        if not total:
            return None
        return sum(rank is not None and rank <= k for rank in ranks) / total

    return {
        "evaluated_count": total,
        "top1": top_k(1),
        "top3": top_k(3),
        "top5": top_k(5),
        "mrr": (
            sum(1.0 / rank if rank else 0.0 for rank in ranks) / total
            if total else None
        ),
        "unparsed_count": sum(
            item.get("parse_strategy") == "unparsed" for item in labeled
        ),
    }


# ---------------------------------------------------------------------------
# 核心逻辑
# ---------------------------------------------------------------------------

def split_half(results: List[Dict[str, Any]]):
    """按 case_idx 升序排列后切分为前一半和后一半。"""
    sorted_results = sorted(results, key=lambda item: int(item.get("case_idx", 0)))
    mid = len(sorted_results) // 2
    return sorted_results[:mid], sorted_results[mid:]


def evaluate_half(
    output_dir: str,
    methods: Sequence[str],
) -> Dict[str, Any]:
    rows: List[Dict[str, Any]] = []

    for method in methods:
        prediction_file = os.path.join(output_dir, method, "predictions.json")
        if not os.path.exists(prediction_file):
            print(f"[WARN] 跳过缺失文件: {prediction_file}")
            continue

        payload = load_json(prediction_file)
        results = payload.get("results", [])

        scenario_names = sorted({item.get("scenario_name") for item in results})
        for scenario_name in scenario_names:
            scenario_results = [
                item for item in results
                if item.get("scenario_name") == scenario_name
            ]
            first_half, second_half = split_half(scenario_results)

            for tag, half_results in [("first_100", first_half), ("last_100", second_half)]:
                metrics = calculate_metrics(half_results)
                rows.append({
                    "method": method,
                    "scope": scenario_name,
                    "half": tag,
                    **metrics,
                })

        # 全局分半
        all_sorted = sorted(results, key=lambda item: (
            str(item.get("scenario_name")),
            int(item.get("case_idx", 0)),
        ))
        # 按场景交错 -> 先按 scenario 分组再各自分半后合并
        all_first: List[Dict[str, Any]] = []
        all_second: List[Dict[str, Any]] = []
        for scenario_name in scenario_names:
            sc_results = [
                item for item in results
                if item.get("scenario_name") == scenario_name
            ]
            first, second = split_half(sc_results)
            all_first.extend(first)
            all_second.extend(second)

        for tag, half_results in [("first_100", all_first), ("last_100", all_second)]:
            rows.append({
                "method": method,
                "scope": "overall",
                "half": tag,
                **calculate_metrics(half_results),
            })

    return {"metrics": rows}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="按每类前一百 / 后一百分别统计指标",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="实验输出目录（与 run_three_method_experiment 的 --output-dir 相同）",
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        choices=METHODS,
        default=list(METHODS),
        help="要统计的方法（默认三种全跑）",
    )
    parser.add_argument(
        "--out",
        default=None,
        help="输出目录（默认: <output-dir>/evaluation/half_stats）",
    )
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    out_dir = args.out or os.path.join(args.output_dir, "evaluation", "half_stats")

    report = evaluate_half(args.output_dir, args.methods)
    rows = report["metrics"]

    if not rows:
        print("无数据可统计。")
        return

    # JSON
    save_json(report, os.path.join(out_dir, "half_metrics.json"))
    print(f"JSON: {os.path.join(out_dir, 'half_metrics.json')}")

    # CSV
    csv_path = os.path.join(out_dir, "half_metrics.csv")
    os.makedirs(out_dir, exist_ok=True)
    fieldnames = [
        "method", "scope", "half", "evaluated_count",
        "top1", "top3", "top5", "mrr", "unparsed_count",
    ]
    with open(csv_path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"CSV: {csv_path}")

    # 简要打印
    print()
    for row in rows:
        print(
            f"[{row['method']}][{row['scope']}][{row['half']}] "
            f"n={row['evaluated_count']}  "
            f"top1={_fmt(row['top1'])}  top3={_fmt(row['top3'])}  "
            f"top5={_fmt(row['top5'])}  mrr={_fmt(row['mrr'])}  "
            f"unparsed={row['unparsed_count']}"
        )


def _fmt(val: Optional[float]) -> str:
    return f"{val:.4f}" if val is not None else "N/A"


if __name__ == "__main__":
    main()
