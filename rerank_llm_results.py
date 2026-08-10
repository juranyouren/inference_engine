# -*- coding: utf-8 -*-
"""Rerank saved Competition/Cooperation results with the Tree val rules.

This entrypoint never loads an LLM.  It reads the normalized prediction files
written by ``run_three_method_experiment.py``, maps ``groundtruth`` to the
``rc`` field expected by the shared val scorer, reloads the corresponding
cases, and writes a new result directory containing both raw and val rankings.
"""

import ast
import json
import os

# Must be set before importing modules that may initialize torch_npu/vLLM.
os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")

import argparse
import copy
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

import config  # noqa: E402
from inference.common import (  # noqa: E402
    IndexedCaseReader,
    ensure_dir,
    load_json,
    save_csv,
    save_infer_outputs,
    save_json,
)
from inference.validation import rerank_results_with_val  # noqa: E402


def _as_list(value: Any) -> List[str]:
    if isinstance(value, (list, tuple)):
        return [str(item).strip() for item in value if str(item).strip()]
    if not isinstance(value, str) or not value.strip():
        return []
    text = value.strip()
    for parser in (json.loads, ast.literal_eval):
        try:
            parsed = parser(text)
        except (ValueError, SyntaxError, TypeError, json.JSONDecodeError):
            continue
        if isinstance(parsed, (list, tuple)):
            return [str(item).strip() for item in parsed if str(item).strip()]
    return []


def _root_from_label_path(label_path: Optional[str]) -> Optional[str]:
    if not label_path:
        return None
    return str(Path(label_path).resolve().parent)


def _scenario_for_results(
    scenario_name: str,
    items: Sequence[Dict[str, Any]],
    label_root: str,
) -> Dict[str, Any]:
    first = items[0] if items else {}
    alarm_type = str(first.get("alarm_type") or scenario_name)
    data_dir = str(Path(label_root).resolve() / scenario_name)
    if not Path(data_dir).is_dir():
        fallback = _root_from_label_path(first.get("label_file_path"))
        if fallback:
            data_dir = fallback
    return {
        "name": scenario_name,
        "alarm_type": alarm_type,
        "data_dir": data_dir,
    }


def _normalize_result(
    item: Dict[str, Any],
    scenario: Dict[str, Any],
) -> Dict[str, Any]:
    normalized = copy.deepcopy(item)
    if normalized.get("case_idx") is None:
        raise ValueError(f"LLM result 缺少 case_idx: {item}")
    normalized["scenario_name"] = scenario["name"]
    normalized["alarm_type"] = str(
        normalized.get("alarm_type") or scenario["alarm_type"]
    )
    normalized["pred_rc"] = _as_list(normalized.get("pred_rc", []))
    groundtruth = normalized.get("groundtruth", normalized.get("rc"))
    if groundtruth is None:
        groundtruth = ""
    normalized["groundtruth"] = str(groundtruth).strip()
    normalized["rc"] = normalized["groundtruth"]
    prediction = normalized["pred_rc"]
    normalized["pred_top1_rc"] = prediction[0] if prediction else None
    normalized["rank"] = (
        prediction.index(normalized["groundtruth"]) + 1
        if normalized["groundtruth"] in prediction
        else None
    )
    normalized["is_correct"] = normalized["rank"] == 1
    return normalized


def _calculate_metrics(
    results: Sequence[Dict[str, Any]],
    elapsed_seconds: float,
    rank_key: str,
) -> Dict[str, Any]:
    labeled = [item for item in results if item.get("groundtruth")]
    ranks = [item.get(rank_key) for item in labeled]
    total = len(labeled)

    def top_k(k: int) -> Optional[float]:
        if not total:
            return None
        return sum(
            rank is not None and rank <= k
            for rank in ranks
        ) / total

    return {
        "evaluated_count": total,
        "top1": top_k(1),
        "top3": top_k(3),
        "top5": top_k(5),
        "mrr": (
            sum(1.0 / rank if rank else 0.0 for rank in ranks) / total
            if total else None
        ),
        "elapsed_seconds": elapsed_seconds,
        "average_seconds_per_case": (
            elapsed_seconds / total if total else None
        ),
        "unparsed_count": sum(
            item.get("parse_strategy") == "unparsed"
            for item in labeled
        ),
    }


def _source_elapsed_seconds(
    source_payload: Dict[str, Any],
    scenario_name: Optional[str] = None,
) -> float:
    runs = source_payload.get("meta", {}).get("runs", [])
    if isinstance(runs, list):
        return sum(
            float(run.get("elapsed_seconds", 0.0))
            for run in runs
            if scenario_name is None
            or run.get("scenario_name") == scenario_name
        )
    return float(source_payload.get("meta", {}).get("elapsed_seconds", 0.0))


def _build_evaluation_report(
    method: str,
    results: Sequence[Dict[str, Any]],
    scenario_names: Sequence[str],
    source_payload: Dict[str, Any],
    rerank_elapsed_by_scenario: Dict[str, float],
) -> Dict[str, Any]:
    rows: List[Dict[str, Any]] = []

    def append_rows(scope: str, scoped_results: Sequence[Dict[str, Any]]) -> None:
        model_elapsed = _source_elapsed_seconds(
            source_payload,
            None if scope == "overall" else scope,
        )
        rerank_elapsed = sum(
            rerank_elapsed_by_scenario.values()
        ) if scope == "overall" else rerank_elapsed_by_scenario.get(scope, 0.0)

        for ranking, rank_key, elapsed in (
            ("raw", "rank", model_elapsed),
            ("val", "rank_val", model_elapsed + rerank_elapsed),
        ):
            metrics = _calculate_metrics(
                scoped_results,
                elapsed_seconds=elapsed,
                rank_key=rank_key,
            )
            rows.append({
                "method": method,
                "scope": scope,
                "ranking": ranking,
                "model_elapsed_seconds": model_elapsed,
                "rerank_elapsed_seconds": rerank_elapsed,
                "end_to_end_elapsed_seconds": elapsed,
                **metrics,
            })

    for scenario_name in scenario_names:
        append_rows(
            scenario_name,
            [
                item for item in results
                if item.get("scenario_name") == scenario_name
            ],
        )
    append_rows("overall", results)

    return {
        "metric_definition": {
            "raw": "LLM 原始候选列表，使用 rank",
            "val": "规则重排后的候选列表，使用 rank_val",
            "top_k": "truth rank <= k 的比例",
            "mrr": "truth rank 倒数的平均值，未命中按 0 计",
            "model_time": "已有结果中的 LLM 运行耗时",
            "rerank_time": "本次 val 重排耗时",
            "average_time": "end_to_end_elapsed_seconds / evaluated_count",
        },
        "metrics": rows,
    }


def _load_label_root(experiment_dir: str, explicit_root: Optional[str]) -> str:
    if explicit_root:
        return explicit_root
    config_path = os.path.join(experiment_dir, "experiment_config.json")
    if os.path.exists(config_path):
        saved = load_json(config_path)
        for key in ("agentdigest_label_root", "anomalydetect_label_root"):
            value = saved.get(key)
            if value:
                return str(value)
    return str(getattr(
        config,
        "AGENTDIGEST_LABEL_ROOT",
        getattr(config, "ANOMALYDETECT_LABEL_ROOT"),
    ))


def rerank_method(
    experiment_dir: str,
    method: str,
    output_dir: str,
    label_root: str,
    scenario_filter: Optional[str] = None,
    rule_root: Optional[str] = None,
    template_dir: Optional[str] = None,
    kpi_mode: Optional[str] = None,
    neutral_score: Optional[float] = None,
    target_size: Optional[int] = None,
    random_seed: Optional[int] = None,
) -> Dict[str, Any]:
    source_path = os.path.join(experiment_dir, method, "predictions.json")
    if not os.path.exists(source_path):
        raise FileNotFoundError(f"未找到 {method} 的结果文件: {source_path}")

    source_payload = load_json(source_path)
    source_results = source_payload.get("results", [])
    if not isinstance(source_results, list):
        raise TypeError(f"{source_path} 的 results 必须是 list")

    grouped: Dict[str, List[Dict[str, Any]]] = {}
    for raw_item in source_results:
        scenario_name = str(
            raw_item.get("scenario_name")
            or raw_item.get("alarm_type")
            or "unknown"
        )
        if scenario_filter and scenario_name != scenario_filter:
            continue
        grouped.setdefault(scenario_name, []).append(raw_item)

    merged_results: List[Dict[str, Any]] = []
    validations: List[Dict[str, Any]] = []
    rerank_elapsed_by_scenario: Dict[str, float] = {}
    for scenario_name in sorted(grouped):
        raw_items = grouped[scenario_name]
        scenario = _scenario_for_results(scenario_name, raw_items, label_root)
        normalized = [
            _normalize_result(item, scenario)
            for item in raw_items
        ]
        indices = [int(item["case_idx"]) for item in normalized]
        reader = IndexedCaseReader(scenario["data_dir"], scenario["alarm_type"])
        cases = reader.load_cases(indices)
        rerank_started = time.perf_counter()
        reranked, validation = rerank_results_with_val(
            scenario=scenario,
            cases=cases,
            indices=indices,
            results=normalized,
            rule_root=rule_root,
            template_dir=template_dir,
            kpi_mode=kpi_mode,
            neutral_score=neutral_score,
            target_size=target_size,
            random_seed=random_seed,
        )
        rerank_elapsed = time.perf_counter() - rerank_started
        rerank_elapsed_by_scenario[scenario_name] = rerank_elapsed
        validation["rerank_elapsed_seconds"] = rerank_elapsed
        merged_results.extend(reranked)
        validation["scenario_name"] = scenario_name
        validations.append(validation)

    payload = copy.deepcopy(source_payload)
    payload["meta"] = copy.deepcopy(source_payload.get("meta", {}))
    payload["meta"].update({
        "method": method,
        "validation_type": "root_cause_rerank",
        "validation_enabled": True,
        "source_prediction_file": source_path,
        "label_root": label_root,
        "rule_root": rule_root,
        "template_dir": template_dir,
        "kpi_mode": kpi_mode,
        "neutral_score": neutral_score,
        "target_size": target_size,
        "random_seed": random_seed,
        "validation_by_scenario": validations,
    })
    evaluation = _build_evaluation_report(
        method=method,
        results=merged_results,
        scenario_names=sorted(grouped),
        source_payload=source_payload,
        rerank_elapsed_by_scenario=rerank_elapsed_by_scenario,
    )
    payload["summary"] = {
        "processed_count": len(merged_results),
        "processed_indices": [item.get("case_idx") for item in merged_results],
    }
    payload["results"] = merged_results
    payload["evaluation"] = evaluation

    method_output_dir = os.path.join(output_dir, method)
    ensure_dir(method_output_dir)
    save_infer_outputs(payload, method_output_dir, "all")
    save_json(
        {
            "source": source_path,
            "method": method,
            "validation_by_scenario": validations,
            "evaluation": evaluation,
        },
        os.path.join(method_output_dir, "validation.json"),
    )
    evaluation_dir = os.path.join(output_dir, "evaluation")
    ensure_dir(evaluation_dir)
    save_json(evaluation, os.path.join(evaluation_dir, f"{method}_metrics.json"))
    save_csv(evaluation["metrics"], os.path.join(evaluation_dir, f"{method}_metrics.csv"))
    return payload


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="对已有 Competition/Cooperation 结果调用 Tree val 规则重排"
    )
    parser.add_argument("--experiment-dir", required=True)
    parser.add_argument(
        "--methods",
        nargs="+",
        choices=["competition", "cooperation"],
        default=["competition", "cooperation"],
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="输出目录；默认写到 experiment-dir/llm_reranked",
    )
    parser.add_argument(
        "--label-root",
        "--agentdigest-label-root",
        dest="label_root",
        default=None,
        help="LLM case/label 类别根目录；默认从 experiment_config 或 config 读取",
    )
    parser.add_argument("--scenario", default=None)
    parser.add_argument("--tree-val-rule-root", default=None)
    parser.add_argument("--tree-val-template-dir", default=None)
    parser.add_argument("--tree-val-kpi-mode", choices=["mix", "alarm"], default=None)
    parser.add_argument("--tree-val-neutral-score", type=float, default=None)
    parser.add_argument("--tree-val-target-size", type=int, default=None)
    parser.add_argument("--tree-val-random-seed", type=int, default=None)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    experiment_dir = str(Path(args.experiment_dir).resolve())
    output_dir = args.output_dir or os.path.join(experiment_dir, "llm_reranked")
    label_root = _load_label_root(experiment_dir, args.label_root)
    ensure_dir(output_dir)

    reports: List[Dict[str, Any]] = []
    for method in args.methods:
        payload = rerank_method(
            experiment_dir=experiment_dir,
            method=method,
            output_dir=output_dir,
            label_root=label_root,
            scenario_filter=args.scenario,
            rule_root=args.tree_val_rule_root,
            template_dir=args.tree_val_template_dir,
            kpi_mode=args.tree_val_kpi_mode,
            neutral_score=args.tree_val_neutral_score,
            target_size=args.tree_val_target_size,
            random_seed=args.tree_val_random_seed,
        )
        reports.extend(payload.get("evaluation", {}).get("metrics", []))
        print(f"[Done] {method}: {os.path.join(output_dir, method)}")

    combined_report = {
        "metric_definition": {
            "raw": "LLM 原始候选列表，使用 rank",
            "val": "规则重排后的候选列表，使用 rank_val",
            "top_k": "truth rank <= k 的比例",
            "mrr": "truth rank 倒数的平均值，未命中按 0 计",
            "average_time": "end_to_end_elapsed_seconds / evaluated_count",
        },
        "metrics": reports,
    }
    evaluation_dir = os.path.join(output_dir, "evaluation")
    ensure_dir(evaluation_dir)
    save_json(combined_report, os.path.join(evaluation_dir, "metrics.json"))
    save_csv(reports, os.path.join(evaluation_dir, "metrics.csv"))
    print(json.dumps(combined_report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
