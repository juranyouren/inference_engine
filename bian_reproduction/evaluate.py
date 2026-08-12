# -*- coding: utf-8 -*-

# 单个结果目录：
# python -m bian_reproduction.evaluate \
#   --predictions /path/to/idx_0_49/predictions.json
#
# 汇总多个结果目录：
# python -m bian_reproduction.evaluate \
#   --predictions /path/to/run_a /path/to/run_b \
#   --output-dir /path/to/evaluation

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence


VALID_RANKING_STRATEGIES = {"scored_ranking", "ranking_only"}
TIMING_FIELDS = (
    "model_init_seconds",
    "alert_summary_seconds",
    "anomaly_analysis_seconds",
    "joint_scoring_seconds",
    "model_inference_seconds",
    "elapsed_seconds",
    "wall_clock_seconds",
)


def _number(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _positive_rank(result: Dict[str, Any]) -> Optional[int]:
    if result.get("parse_strategy") not in VALID_RANKING_STRATEGIES:
        return None
    try:
        rank = int(result.get("rank"))
    except (TypeError, ValueError):
        return None
    return rank if rank > 0 else None


def _sum_timing(payloads: Iterable[Dict[str, Any]]) -> Dict[str, float]:
    totals = {field: 0.0 for field in TIMING_FIELDS}
    for payload in payloads:
        timing = payload.get("meta", {}).get("timing", {})
        if not isinstance(timing, dict):
            continue
        for field in TIMING_FIELDS:
            totals[field] += _number(timing.get(field))
    return totals


def calculate_metrics(
    results: Sequence[Dict[str, Any]],
    timing: Optional[Dict[str, Any]] = None,
    skipped_as_miss_count: int = 0,
) -> Dict[str, Any]:
    """Calculate strict Top-K/MRR metrics for labeled cases.

    Only rankings parsed from a real model ranking are accepted. Candidate-order
    fallback, malformed output, and skipped inference cases contribute zero.
    """
    labeled = [item for item in results if str(item.get("groundtruth", "")).strip()]
    ranks = [_positive_rank(item) for item in labeled]
    skipped = max(0, int(skipped_as_miss_count or 0))
    total = len(labeled) + skipped

    def top_k(k: int) -> Optional[float]:
        if not total:
            return None
        return sum(rank is not None and rank <= k for rank in ranks) / total

    timing_values = {
        field: _number((timing or {}).get(field))
        for field in TIMING_FIELDS
    }
    elapsed = timing_values["elapsed_seconds"]
    return {
        "evaluated_count": total,
        "prediction_count": len(labeled),
        "skipped_as_miss_count": skipped,
        "invalid_prediction_count": sum(rank is None for rank in ranks),
        "top1": top_k(1),
        "top3": top_k(3),
        "top5": top_k(5),
        "mrr": (
            sum(1.0 / rank if rank is not None else 0.0 for rank in ranks) / total
            if total else None
        ),
        **timing_values,
        "average_seconds_per_case": elapsed / total if total else None,
    }


def build_evaluation_report(
    payloads: Sequence[Dict[str, Any]],
) -> Dict[str, Any]:
    alarm_types = sorted({
        str(payload.get("meta", {}).get("alarm_type", "")).strip()
        for payload in payloads
        if str(payload.get("meta", {}).get("alarm_type", "")).strip()
    })
    rows: List[Dict[str, Any]] = []

    def append_scope(scope: str, selected_payloads: Sequence[Dict[str, Any]]) -> None:
        results = [
            result
            for payload in selected_payloads
            for result in payload.get("results", [])
            if isinstance(result, dict)
        ]
        skipped = sum(
            int(payload.get("summary", {}).get("skipped_count", 0) or 0)
            for payload in selected_payloads
        )
        rows.append({
            "scope": scope,
            **calculate_metrics(
                results,
                timing=_sum_timing(selected_payloads),
                skipped_as_miss_count=skipped,
            ),
        })

    for alarm_type in alarm_types:
        append_scope(
            alarm_type,
            [
                payload for payload in payloads
                if payload.get("meta", {}).get("alarm_type") == alarm_type
            ],
        )
    append_scope("overall", payloads)

    return {
        "metric_definition": {
            "groundtruth": "label['root_cause']['category']",
            "rank": "groundtruth 在模型候选排序中的位置，从 1 开始",
            "top_k": "rank <= K 的 case 比例；解析失败、候选顺序回退和跳过 case 按未命中计",
            "mrr": "真实根因排名倒数的平均值；未命中按 0 计",
            "elapsed_seconds": "第一阶段端到端运行时间，不含一次性模型初始化",
            "average_seconds_per_case": "elapsed_seconds / evaluated_count",
            "model_init_seconds": "一次性模型初始化时间，单独报告",
        },
        "metrics": rows,
    }


def write_evaluation_report(
    report: Dict[str, Any],
    output_dir: Path,
) -> Dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "metrics.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    rows = report.get("metrics", [])
    with (output_dir / "metrics.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=list(rows[0].keys()) if rows else ["scope"],
        )
        writer.writeheader()
        writer.writerows(rows)
    return report


def evaluate_payloads(
    payloads: Sequence[Dict[str, Any]],
    output_dir: Path,
) -> Dict[str, Any]:
    return write_evaluation_report(
        build_evaluation_report(payloads),
        output_dir,
    )


def discover_prediction_files(inputs: Sequence[str]) -> List[Path]:
    paths: List[Path] = []
    for value in inputs:
        path = Path(value).resolve()
        if path.is_file():
            paths.append(path)
        elif (path / "predictions.json").is_file():
            paths.append(path / "predictions.json")
        elif path.is_dir():
            paths.extend(sorted(path.rglob("predictions.json")))
        else:
            raise FileNotFoundError(f"结果路径不存在: {path}")
    unique = list(dict.fromkeys(paths))
    if not unique:
        raise FileNotFoundError("没有找到 predictions.json")
    return unique


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate standalone BiAn Stage 1 predictions"
    )
    parser.add_argument(
        "--predictions",
        nargs="+",
        required=True,
        help="predictions.json、单次结果目录或包含多个结果的目录",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="评价输出目录；单个结果默认写入该结果目录的 evaluation",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    prediction_files = discover_prediction_files(args.predictions)
    payloads = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in prediction_files
    ]
    if args.output_dir:
        output_dir = Path(args.output_dir).resolve()
    elif len(prediction_files) == 1:
        output_dir = prediction_files[0].parent / "evaluation"
    else:
        output_dir = Path.cwd() / "bian_evaluation"

    report = evaluate_payloads(payloads, output_dir)
    overall = next(
        row for row in report["metrics"] if row["scope"] == "overall"
    )
    print(json.dumps(overall, ensure_ascii=False, indent=2))
    print(f"评价输出目录: {output_dir}")


if __name__ == "__main__":
    main()
