# -*- coding: utf-8 -*-
"""Rerun only LLM cases whose saved responses still cannot be parsed."""

import argparse
import json
import multiprocessing as mp
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")

import config

if hasattr(config, "ASCEND_RT_VISIBLE_DEVICES"):
    os.environ["ASCEND_RT_VISIBLE_DEVICES"] = str(config.ASCEND_RT_VISIBLE_DEVICES)

from inference.common import IndexedCaseReader, load_json, save_infer_outputs, save_json
from reparse_and_score_llm_results import (
    _candidate_categories,
    _groundtruth,
    parse_competition_case,
    parse_cooperation_case,
)
from run_three_method_experiment import ensure_valid_working_directory, evaluate_saved_results


METHODS = ("competition", "cooperation")


def select_failed_results(
    results: Sequence[Dict[str, Any]],
    include_text_fallback: bool = False,
) -> List[Dict[str, Any]]:
    failed = []
    for item in results:
        strategy = str(item.get("parse_strategy", ""))
        is_failed = not item.get("pred_rc") or strategy in {
            "unparsed",
            "missing_scenario_output_dir",
            "rerun_inference_failed",
        }
        if include_text_fallback and "candidate_text_fallback" in strategy:
            is_failed = True
        if is_failed:
            failed.append(item)
    return failed


def merge_results(
    baseline: Sequence[Dict[str, Any]],
    replacements: Sequence[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    replacement_map = {
        (item.get("scenario_name"), int(item["case_idx"])): item
        for item in replacements
    }
    return [
        replacement_map.get(
            (item.get("scenario_name"), int(item["case_idx"])),
            item,
        )
        for item in baseline
    ]


def _baseline_file(experiment_dir: str, method: str) -> str:
    candidates = [
        os.path.join(experiment_dir, "reparsed", method, "predictions.json"),
        os.path.join(experiment_dir, method, "predictions.json"),
    ]
    for path in candidates:
        if os.path.isfile(path):
            return path
    raise FileNotFoundError(f"找不到 {method} predictions.json: {candidates}")


def _build_plan(failed: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    grouped: Dict[Tuple[str, str], List[int]] = {}
    for item in failed:
        scenario = str(item.get("scenario_name") or item.get("alarm_type"))
        alarm_type = str(item.get("alarm_type") or scenario)
        grouped.setdefault((scenario, alarm_type), []).append(int(item["case_idx"]))
    return [
        {
            "scenario_name": scenario,
            "alarm_type": alarm_type,
            "indices": sorted(set(indices)),
        }
        for (scenario, alarm_type), indices in sorted(grouped.items())
    ]


def _rerun_worker(
    args: argparse.Namespace,
    method: str,
    plan: List[Dict[str, Any]],
) -> None:
    from inference.llm import (
        _lazy_import_llm_modules,
        llm_competition_one_scenario,
        llm_cooperation_one_scenario,
    )

    (
        RCAGenerator,
        load_alarm_data_meta,
        load_alarm_data_reasoner_cooperation,
        load_alarm_data_verifier,
    ) = _lazy_import_llm_modules()

    model_started = time.perf_counter()
    generator = RCAGenerator(getattr(config, "MODEL_PATH"))
    model_init_seconds = time.perf_counter() - model_started
    method_dir = os.path.join(args.rerun_output_dir, method)
    raw_dir = os.path.join(method_dir, "runs")
    runs = []

    for group in plan:
        scenario = {
            "name": group["scenario_name"],
            "alarm_type": group["alarm_type"],
            "data_dir": os.path.join(args.label_root, group["scenario_name"]),
        }
        started = time.perf_counter()
        if method == "competition":
            payload = llm_competition_one_scenario(
                scenario=scenario,
                indices=group["indices"],
                rca_generator=generator,
                load_alarm_data_verifier=load_alarm_data_verifier,
                output_dir=raw_dir,
                output_format="json",
                batch_size=args.batch_size,
            )
        else:
            payload = llm_cooperation_one_scenario(
                scenario=scenario,
                indices=group["indices"],
                rca_generator=generator,
                load_alarm_data_meta=load_alarm_data_meta,
                load_alarm_data_reasoner_cooperation=load_alarm_data_reasoner_cooperation,
                output_dir=raw_dir,
                output_format="json",
                batch_size=args.batch_size,
                max_rounds=args.max_rounds,
            )
        elapsed = time.perf_counter() - started
        runs.append({
            "scenario_name": group["scenario_name"],
            "alarm_type": group["alarm_type"],
            "requested_indices": group["indices"],
            "processed_indices": payload.get("summary", {}).get("processed_indices", []),
            "skipped_indices": payload.get("summary", {}).get("skipped_indices", []),
            "processed_count": payload.get("summary", {}).get("processed_count", 0),
            "elapsed_seconds": elapsed,
            "output_dir": payload.get("meta", {}).get("output_dir"),
        })

    save_json({
        "method": method,
        "model_init_seconds": model_init_seconds,
        "runs": runs,
    }, os.path.join(method_dir, "rerun_manifest.json"))


def _label_for_item(item: Dict[str, Any], label_root: str):
    label_path = item.get("label_file_path")
    if label_path and os.path.isfile(label_path):
        return load_json(label_path), label_path
    scenario = str(item.get("scenario_name") or item.get("alarm_type"))
    reader = IndexedCaseReader(os.path.join(label_root, scenario), scenario)
    _case_path, label_path = reader.find_case_and_label_path(int(item["case_idx"]))
    return load_json(label_path), label_path


def _parse_rerun_results(
    method: str,
    failed: Sequence[Dict[str, Any]],
    manifest: Dict[str, Any],
    label_root: str,
) -> List[Dict[str, Any]]:
    run_map = {
        run["scenario_name"]: run
        for run in manifest.get("runs", [])
    }
    results = []
    for base_item in failed:
        scenario = str(base_item.get("scenario_name") or base_item.get("alarm_type"))
        alarm_type = str(base_item.get("alarm_type") or scenario)
        case_idx = int(base_item["case_idx"])
        label_data, label_path = _label_for_item(base_item, label_root)
        candidates = _candidate_categories(label_data)
        run = run_map.get(scenario)
        if not run or not run.get("output_dir"):
            ranking, strategy, attempts = [], "rerun_inference_failed", []
        elif method == "competition":
            ranking, strategy, attempts = parse_competition_case(
                Path(run["output_dir"]),
                alarm_type,
                case_idx,
                candidates,
            )
        else:
            ranking, strategy, attempts = parse_cooperation_case(
                Path(run["output_dir"]),
                case_idx,
                candidates,
            )
        truth = _groundtruth(label_data)
        rank = ranking.index(truth) + 1 if truth in ranking else None
        item = dict(base_item)
        item.update({
            "label_file_path": label_path,
            "groundtruth": truth,
            "pred_top1_rc": ranking[0] if ranking else None,
            "pred_rc": ranking,
            "rank": rank,
            "is_correct": rank == 1,
            "parse_strategy": f"rerun:{strategy}" if strategy != "rerun_inference_failed" else strategy,
            "parse_attempts": attempts,
            "rerun": True,
        })
        results.append(item)
    return results


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="只重新推理重新解析后仍失败的 Competition/Cooperation case。",
    )
    parser.add_argument(
        "--experiment-dir",
        default=os.path.join(
            getattr(config, "PREDICT_RES_DIR"),
            "experiments",
            "three_methods",
        ),
    )
    parser.add_argument(
        "--label-root",
        default=getattr(config, "ANOMALYDETECT_LABEL_ROOT"),
    )
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=list(METHODS))
    parser.add_argument("--batch-size", type=int, default=getattr(config, "BATCH_SIZE", 16))
    parser.add_argument(
        "--max-rounds",
        type=int,
        default=getattr(config, "LLM_COOPERATION_MAX_ROUNDS", 5),
    )
    parser.add_argument(
        "--rerun-text-fallback",
        action="store_true",
        help="把仅通过 candidate_text_fallback 解析出的低置信度 case 也重跑。",
    )
    parser.add_argument("--rerun-output-dir", default=None)
    return parser


def main() -> None:
    ensure_valid_working_directory()
    args = build_arg_parser().parse_args()
    args.rerun_output_dir = args.rerun_output_dir or os.path.join(
        args.experiment_dir,
        "rerun_failed",
    )

    baselines = {}
    failed_by_method = {}
    plans = {}
    total_failed = 0
    for method in args.methods:
        source_file = _baseline_file(args.experiment_dir, method)
        baselines[method] = load_json(source_file)
        failed = select_failed_results(
            baselines[method].get("results", []),
            include_text_fallback=args.rerun_text_fallback,
        )
        failed_by_method[method] = failed
        plans[method] = _build_plan(failed)
        total_failed += len(failed)
        print(f"[{method}] 待重跑: {len(failed)}")

    save_json({
        "total_failed": total_failed,
        "plans": plans,
    }, os.path.join(args.rerun_output_dir, "rerun_plan.json"))

    # 先把 baseline 全量复制到 rerun_output_dir，为每个 method 建立完整基础
    for method in args.methods:
        method_dir = os.path.join(args.rerun_output_dir, method)
        save_json(baselines[method], os.path.join(method_dir, "baseline_backup.json"))
        save_infer_outputs(baselines[method], method_dir, "all")
        print(f"[{method}] 基线已复制到: {method_dir}")

    if total_failed == 0:
        print("没有需要重跑的 case，rerun_output_dir 即为完整结果。")
        report = evaluate_saved_results(args.rerun_output_dir, args.methods)
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return

    ctx = mp.get_context("spawn")
    for method in args.methods:
        if not plans[method]:
            continue
        print(f"[Rerun] 启动独立进程: {method}")
        process = ctx.Process(
            target=_rerun_worker,
            args=(args, method, plans[method]),
            name=f"rerun_failed_{method}",
        )
        process.start()
        process.join()
        if process.exitcode != 0:
            raise RuntimeError(f"{method} 失败 case 重跑异常，exitcode={process.exitcode}")

    for method in args.methods:
        method_dir = os.path.join(args.rerun_output_dir, method)
        manifest_path = os.path.join(method_dir, "rerun_manifest.json")
        baseline = baselines[method]

        if plans[method]:
            manifest = load_json(manifest_path)
            rerun_results = _parse_rerun_results(
                method,
                failed_by_method[method],
                manifest,
                args.label_root,
            )
        else:
            manifest = {"runs": []}
            rerun_results = []

        # 用 baseline 打底，填入 rerun 结果，保存为完整结果
        merged_results = merge_results(baseline.get("results", []), rerun_results)
        merged_payload = {
            "meta": {
                **baseline.get("meta", {}),
                "rerun_failed_only": True,
                "runs": baseline.get("meta", {}).get("runs", []) + manifest.get("runs", []),
            },
            "summary": {
                "processed_count": len(merged_results),
                "rerun_count": len(rerun_results),
                "requested_count": len(failed_by_method[method]),
                "rerun_parsed_count": sum(
                    bool(item.get("pred_rc")) for item in rerun_results
                ),
                "rerun_unparsed_count": sum(
                    not item.get("pred_rc") for item in rerun_results
                ),
                "final_unparsed_count": sum(
                    not item.get("pred_rc") for item in merged_results
                ),
            },
            "results": merged_results,
        }
        save_infer_outputs(merged_payload, method_dir, "all")
        print(
            f"[{method}] 完整结果: {os.path.join(method_dir, 'predictions.json')}  "
            f"(共 {len(merged_results)} 条，rerun {len(rerun_results)} 条)"
        )

    report = evaluate_saved_results(args.rerun_output_dir, args.methods)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"完整结果（含 rerun）: {args.rerun_output_dir}")


if __name__ == "__main__":
    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass
    sys.exit(main())
