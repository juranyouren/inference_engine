# -*- coding: utf-8 -*-
"""Run and evaluate the Tree, Competition, and Cooperation benchmarks.

This is intentionally a separate entrypoint.  The production incremental
entrypoint and its state file are not modified by benchmark runs.
"""

import os

# Must be set before importing anything that can initialize vLLM/torch_npu.
os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")

import argparse
import csv
import json
import multiprocessing as mp
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

import config  # noqa: E402


def configure_runtime_environment() -> None:
    """Apply the same Ascend runtime environment as the production entrypoint."""
    if hasattr(config, "ASCEND_RT_VISIBLE_DEVICES"):
        os.environ.setdefault(
            "ASCEND_RT_VISIBLE_DEVICES",
            str(config.ASCEND_RT_VISIBLE_DEVICES),
        )


# Spawned children import this module again, so this runs before any lazy
# Selector/Refiner/RCAGenerator import can initialize torch_npu or vLLM.
configure_runtime_environment()

from inference.common import (  # noqa: E402
    IndexedCaseReader,
    discover_label_indices,
    ensure_dir,
    load_json,
    save_infer_outputs,
    save_json,
    scan_label_root,
)
from inference.result_parsing import (  # noqa: E402
    extract_ranked_categories,
    parse_competition_case,
    parse_cooperation_case,
)


METHODS = ("tree", "competition", "cooperation")
TEST_CASES_PER_CATEGORY = 5


def ensure_valid_working_directory() -> Path:
    """Recover when the shell's cwd was deleted/replaced during deployment."""
    try:
        return Path(os.getcwd()).resolve()
    except FileNotFoundError:
        os.chdir(CURRENT_DIR)
        print(f"[Runtime] 当前工作目录已失效，自动切换到: {CURRENT_DIR}")
        return CURRENT_DIR


def _root_cause_category(label_data: Dict[str, Any]) -> str:
    """Read the benchmark truth from label['root_cause']['category']."""
    root_cause = label_data.get("root_cause", {})
    if isinstance(root_cause, dict):
        value = root_cause.get("category", "")
        return str(value).strip() if value is not None else ""
    # Retain compatibility with old labels that stored the category directly.
    return str(root_cause).strip() if root_cause is not None else ""


def _candidate_categories(label_data: Dict[str, Any]) -> List[str]:
    categories: List[str] = []
    for item in label_data.get("root_cause_candidates", []) or []:
        if isinstance(item, dict):
            value = item.get("category", "")
        else:
            value = item
        value = str(value).strip() if value is not None else ""
        if value and value not in categories:
            categories.append(value)
    return categories


def _deduplicate(values: Iterable[str]) -> List[str]:
    result: List[str] = []
    for value in values:
        value = str(value).strip()
        if value and value not in result:
            result.append(value)
    return result


def _rank_result(
    case_idx: int,
    scenario: Dict[str, Any],
    label_path: str,
    prediction: Sequence[str],
    response_path: Optional[str],
    parse_strategy: str,
    parse_attempts: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    label_data = load_json(label_path)
    groundtruth = _root_cause_category(label_data)
    pred_rc = _deduplicate(prediction)
    rank = pred_rc.index(groundtruth) + 1 if groundtruth in pred_rc else None
    return {
        "case_idx": case_idx,
        "scenario_name": scenario["name"],
        "alarm_type": scenario["alarm_type"],
        "label_file_path": label_path,
        "groundtruth": groundtruth,
        "pred_top1_rc": pred_rc[0] if pred_rc else None,
        "pred_rc": pred_rc,
        "rank": rank,
        "is_correct": rank == 1,
        "response_file_path": response_path,
        "parse_strategy": parse_strategy,
        "parse_attempts": parse_attempts or [],
    }


def _validate_scenarios(
    root: str,
    scenario_arg: str,
    expected_categories: int,
    cases_per_category: int,
    start_index: int,
    end_index: int,
) -> List[Dict[str, Any]]:
    scenarios = scan_label_root(root, scenario_arg)
    if scenario_arg == "all" and expected_categories > 0:
        if len(scenarios) != expected_categories:
            raise ValueError(
                f"{root} 应有 {expected_categories} 个类别目录，实际发现 {len(scenarios)} 个: "
                f"{[item['name'] for item in scenarios]}"
            )
    for scenario in scenarios:
        discovered_indices = discover_label_indices(
            scenario["data_dir"],
            scenario["alarm_type"],
        )
        indices = [
            idx for idx in discovered_indices
            if start_index <= idx <= end_index
        ]
        if cases_per_category > 0 and len(indices) != cases_per_category:
            raise ValueError(
                f"{scenario['data_dir']} 在闭区间 [{start_index}, {end_index}] "
                f"应有 {cases_per_category} 个 case，实际发现 {len(indices)} 个 label index"
            )
        expected_indices = list(range(start_index, end_index + 1))
        if indices != expected_indices:
            missing = sorted(set(expected_indices) - set(indices))
            raise ValueError(
                f"{scenario['data_dir']} 的实验索引必须连续覆盖 "
                f"[{start_index}, {end_index}]，缺失: {missing}"
            )
        scenario["indices"] = indices
    return scenarios


def _normalize_tree_result(
    item: Dict[str, Any],
    scenario: Dict[str, Any],
    stage: int,
    train_indices: List[int],
) -> Dict[str, Any]:
    label_path = item.get("label_file_path")
    label_data = load_json(label_path)
    groundtruth = _root_cause_category(label_data)
    prediction = _deduplicate(item.get("pred_rc", []))
    rank = prediction.index(groundtruth) + 1 if groundtruth in prediction else None
    normalized = dict(item)
    normalized.update({
        "scenario_name": scenario["name"],
        "groundtruth": groundtruth,
        "pred_top1_rc": prediction[0] if prediction else None,
        "pred_rc": prediction,
        "rank": rank,
        "is_correct": rank == 1,
        "tree_stage": stage,
        "train_indices": train_indices,
    })
    return normalized


def build_tree_stage_specs(
    scenario: Dict[str, Any],
    args: argparse.Namespace,
) -> List[Tuple[int, List[int], List[int]]]:
    """Build train/infer index sets for normal and five-case test runs."""
    indices = scenario["indices"]
    if args.test:
        train_n = int(getattr(config, "TRAIN_N", 50))
        all_indices = discover_label_indices(
            scenario["data_dir"],
            scenario["alarm_type"],
        )
        train_candidates = [idx for idx in all_indices if idx < min(indices)]
        train_indices = train_candidates[-train_n:]
        if len(train_indices) < train_n:
            raise ValueError(
                f"{scenario['name']} 测试模式训练数据不足："
                f"需要 {train_n} 条早于 {min(indices)} 的数据，"
                f"实际只有 {len(train_indices)} 条"
            )
        return [(1, train_indices, list(indices))]

    if len(indices) < args.block_size * 2:
        raise ValueError(f"{scenario['name']} 至少需要两个 block 才能做 Tree 实验")

    specs = []
    for stage, start in enumerate(
        range(args.block_size, len(indices), args.block_size),
        start=1,
    ):
        infer_indices = indices[start:start + args.block_size]
        if infer_indices:
            specs.append((stage, indices[:start], infer_indices))
    return specs


def _run_tree(args: argparse.Namespace) -> None:
    # Lazy imports keep Tree/Selector/Refiner model initialization in this
    # dedicated process and make the three artifact groups explicit.
    from inference.selection import (
        build_tree_summary_for_refiner,
        generate_selection_by_selector,
        refine_selection_by_tree_summary,
    )
    from inference.tree import run_tree_once

    scenarios = _validate_scenarios(
        args.anomalydetect_label_root,
        args.scenario,
        args.expected_categories,
        args.cases_per_category,
        args.start_index,
        args.end_index,
    )
    all_results: List[Dict[str, Any]] = []
    runs: List[Dict[str, Any]] = []
    method_dir = os.path.join(args.output_dir, "tree")

    for scenario in scenarios:
        for stage, train_indices, infer_indices in build_tree_stage_specs(
            scenario,
            args,
        ):

            reader = IndexedCaseReader(
                scenario["data_dir"],
                scenario["alarm_type"],
            )
            train_cases = reader.load_cases(train_indices)
            infer_cases = reader.load_cases(infer_indices)
            stage_name = f"stage_{stage}_infer_{min(infer_indices)}_{max(infer_indices)}"

            selector_dir = os.path.join(
                method_dir,
                "selector",
                scenario["name"],
                stage_name,
            )
            refiner_dir = os.path.join(
                method_dir,
                "refiner",
                scenario["name"],
                stage_name,
            )
            final_tree_dir = os.path.join(
                method_dir,
                "tree",
                scenario["name"],
                stage_name,
            )

            selector_started = time.perf_counter()
            selection = generate_selection_by_selector(scenario, selector_dir)
            selector_seconds = time.perf_counter() - selector_started

            refiner_started = time.perf_counter()
            for round_id in range(args.refiner_rounds):
                round_dir = os.path.join(refiner_dir, f"round_{round_id}")
                interim_tree_dir = os.path.join(round_dir, "tree_input")
                interim_payload = run_tree_once(
                    scenario=scenario,
                    train_cases=train_cases,
                    infer_cases=infer_cases,
                    train_indices=train_indices,
                    infer_indices=infer_indices,
                    selection=selection,
                    output_dir=interim_tree_dir,
                    output_format="json",
                    tag=f"before_refiner_round_{round_id}",
                    enable_val=args.tree_val,
                    val_depths=args.tree_val_depths,
                )
                tree_summary = build_tree_summary_for_refiner(interim_payload)
                save_json(tree_summary, os.path.join(round_dir, "tree_summary.json"))
                selection = refine_selection_by_tree_summary(
                    scenario=scenario,
                    previous_selection=selection,
                    tree_summary=tree_summary,
                    output_dir=refiner_dir,
                    round_id=round_id,
                )
            refiner_seconds = time.perf_counter() - refiner_started

            ensure_dir(final_tree_dir)
            save_json(selection, os.path.join(final_tree_dir, "selection_final.json"))
            tree_started = time.perf_counter()
            payload = run_tree_once(
                scenario=scenario,
                train_cases=train_cases,
                infer_cases=infer_cases,
                train_indices=train_indices,
                infer_indices=infer_indices,
                selection=selection,
                output_dir=final_tree_dir,
                output_format=args.output_format,
                tag="final_after_refiner",
                enable_val=args.tree_val,
                val_depths=args.tree_val_depths,
            )
            tree_seconds = time.perf_counter() - tree_started
            elapsed = selector_seconds + refiner_seconds + tree_seconds
            results = [
                _normalize_tree_result(item, scenario, stage, train_indices)
                for item in payload.get("results", [])
            ]
            all_results.extend(results)
            runs.append({
                "scenario_name": scenario["name"],
                "stage": stage,
                "train_indices": train_indices,
                "infer_indices": infer_indices,
                "processed_count": len(results),
                "elapsed_seconds": elapsed,
                "average_seconds_per_case": elapsed / len(results) if results else None,
                "selector_seconds": selector_seconds,
                "refiner_seconds": refiner_seconds,
                "tree_seconds": tree_seconds,
                "tree_val_enabled": args.tree_val,
                "tree_validation": payload.get("validation"),
                "selector_output_dir": selector_dir,
                "refiner_output_dir": refiner_dir,
                "tree_output_dir": final_tree_dir,
            })

    payload = {
        "meta": {
            "method": "tree",
            "pipeline": "selector_then_refiner_then_tree",
            "timing_scope": "Selector + Refiner (including its Tree summary runs) + final Tree; process startup excluded",
            "block_size": args.block_size,
            "selection_source": "selector_refiner",
            "refiner_rounds": args.refiner_rounds,
            "tree_val_enabled": args.tree_val,
            "tree_val_depths": args.tree_val_depths,
            "test_mode": args.test,
            "runs": runs,
        },
        "summary": {"processed_count": len(all_results)},
        "results": all_results,
    }
    save_infer_outputs(payload, method_dir, "all")


def _llm_results_for_payload(
    payload: Dict[str, Any],
    scenario: Dict[str, Any],
    method: str,
) -> List[Dict[str, Any]]:
    output_dir = payload["meta"]["output_dir"]
    reader = IndexedCaseReader(scenario["data_dir"], scenario["alarm_type"])
    results: List[Dict[str, Any]] = []

    for idx in payload.get("summary", {}).get("processed_indices", []):
        _case_path, label_path = reader.find_case_and_label_path(idx)
        label_data = load_json(label_path)
        candidates = _candidate_categories(label_data)
        if method == "competition":
            prediction, strategy, attempts = parse_competition_case(
                Path(output_dir),
                scenario["alarm_type"],
                idx,
                candidates,
            )
        else:
            prediction, strategy, attempts = parse_cooperation_case(
                Path(output_dir),
                idx,
                candidates,
            )
        response_path = attempts[0]["path"] if attempts else None
        results.append(_rank_result(
            idx,
            scenario,
            label_path,
            prediction,
            response_path,
            strategy,
            attempts,
        ))
    return results


def _run_llm_method(args: argparse.Namespace, method: str) -> None:
    from inference.llm import (
        _lazy_import_llm_modules,
        llm_competition_one_scenario,
        llm_cooperation_one_scenario,
    )

    scenarios = _validate_scenarios(
        args.anomalydetect_label_root,
        args.scenario,
        args.expected_categories,
        args.cases_per_category,
        args.start_index,
        args.end_index,
    )
    method_dir = os.path.join(args.output_dir, method)
    raw_dir = os.path.join(method_dir, "runs")

    model_started = time.perf_counter()
    (
        RCAGenerator,
        load_alarm_data_meta,
        load_alarm_data_reasoner_cooperation,
        load_alarm_data_verifier,
    ) = _lazy_import_llm_modules()
    generator = RCAGenerator(getattr(config, "MODEL_PATH"))
    model_init_seconds = time.perf_counter() - model_started

    all_results: List[Dict[str, Any]] = []
    runs: List[Dict[str, Any]] = []
    for scenario in scenarios:
        started = time.perf_counter()
        if method == "competition":
            payload = llm_competition_one_scenario(
                scenario=scenario,
                indices=scenario["indices"],
                rca_generator=generator,
                load_alarm_data_verifier=load_alarm_data_verifier,
                output_dir=raw_dir,
                output_format=args.output_format,
                batch_size=args.batch_size,
            )
        else:
            payload = llm_cooperation_one_scenario(
                scenario=scenario,
                indices=scenario["indices"],
                rca_generator=generator,
                load_alarm_data_meta=load_alarm_data_meta,
                load_alarm_data_reasoner_cooperation=load_alarm_data_reasoner_cooperation,
                output_dir=raw_dir,
                output_format=args.output_format,
                batch_size=args.batch_size,
                max_rounds=args.max_rounds,
            )
        elapsed = time.perf_counter() - started
        results = _llm_results_for_payload(payload, scenario, method)
        all_results.extend(results)
        runs.append({
            "scenario_name": scenario["name"],
            "processed_count": len(results),
            "skipped_indices": payload.get("summary", {}).get("skipped_indices", []),
            "elapsed_seconds": elapsed,
            "average_seconds_per_case": elapsed / len(results) if results else None,
            "output_dir": payload.get("meta", {}).get("output_dir"),
        })

    combined = {
        "meta": {
            "method": method,
            "forced_llm_mode": method,
            "test_mode": args.test,
            "model_init_seconds": model_init_seconds,
            "timing_scope": "full per-scenario LLM pipeline; one-time model initialization excluded",
            "runs": runs,
        },
        "summary": {"processed_count": len(all_results)},
        "results": all_results,
    }
    save_infer_outputs(combined, method_dir, "all")


def _method_worker(args: argparse.Namespace, method: str) -> None:
    if method == "tree":
        _run_tree(args)
    elif method in ("competition", "cooperation"):
        _run_llm_method(args, method)
    else:
        raise ValueError(f"未知实验方法: {method}")


def calculate_metrics(
    results: Sequence[Dict[str, Any]],
    elapsed_seconds: float,
) -> Dict[str, Any]:
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
        "elapsed_seconds": elapsed_seconds,
        "average_seconds_per_case": elapsed_seconds / total if total else None,
        "unparsed_count": sum(
            item.get("parse_strategy") == "unparsed" for item in labeled
        ),
    }


def evaluate_saved_results(output_dir: str, methods: Sequence[str]) -> Dict[str, Any]:
    rows: List[Dict[str, Any]] = []
    for method in methods:
        prediction_file = os.path.join(output_dir, method, "predictions.json")
        if not os.path.exists(prediction_file):
            raise FileNotFoundError(f"缺少实验结果: {prediction_file}")
        payload = load_json(prediction_file)
        results = payload.get("results", [])
        runs = payload.get("meta", {}).get("runs", [])

        scenario_names = sorted({item.get("scenario_name") for item in results})
        for scenario_name in scenario_names:
            scenario_results = [
                item for item in results if item.get("scenario_name") == scenario_name
            ]
            elapsed = sum(
                float(run.get("elapsed_seconds", 0.0))
                for run in runs
                if run.get("scenario_name") == scenario_name
            )
            rows.append({
                "method": method,
                "scope": scenario_name,
                **calculate_metrics(scenario_results, elapsed),
            })

        rows.append({
            "method": method,
            "scope": "overall",
            **calculate_metrics(
                results,
                sum(float(run.get("elapsed_seconds", 0.0)) for run in runs),
            ),
        })

    evaluation_dir = os.path.join(output_dir, "evaluation")
    ensure_dir(evaluation_dir)
    report = {
        "metric_definition": {
            "truth": "label['root_cause']['category']",
            "top_k": "fraction whose truth rank is <= k; missing/unparsed predictions are misses",
            "mrr": "mean reciprocal truth rank; missing/unparsed predictions contribute 0",
            "average_time": "summed run wall time / evaluated cases; model initialization excluded",
        },
        "metrics": rows,
    }
    save_json(report, os.path.join(evaluation_dir, "metrics.json"))
    csv_path = os.path.join(evaluation_dir, "metrics.csv")
    with open(csv_path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else [])
        if rows:
            writer.writeheader()
            writer.writerows(rows)
    return report


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="分别运行 Tree / Competition / Cooperation 并统一评测。",
    )
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=list(METHODS))
    parser.add_argument("--scenario", default="all")
    parser.add_argument(
        "--anomalydetect-label-root",
        default=getattr(config, "ANOMALYDETECT_LABEL_ROOT"),
    )
    parser.add_argument(
        "--output-dir",
        default=os.path.join(
            getattr(config, "PREDICT_RES_DIR"),
            "experiments",
            "three_methods",
        ),
    )
    parser.add_argument("--expected-categories", type=int, default=4)
    parser.add_argument("--cases-per-category", type=int, default=200)
    parser.add_argument("--start-index", type=int, default=51)
    parser.add_argument("--end-index", type=int, default=250)
    parser.add_argument("--block-size", type=int, default=50)
    parser.add_argument(
        "--test",
        action="store_true",
        help=(
            "快速测试：每个类别只评测从 start-index 开始的 5 个 case；"
            "Tree 使用这些 case 之前的 TRAIN_N 条数据训练。"
        ),
    )
    parser.add_argument(
        "--selection-source",
        choices=["selector_refiner"],
        default="selector_refiner",
        help="Tree 实验固定开启 Selector 和 Refiner。",
    )
    parser.add_argument(
        "--refiner-rounds",
        type=int,
        default=getattr(config, "REFINER_ROUNDS", 1),
    )
    tree_val_group = parser.add_mutually_exclusive_group()
    tree_val_group.add_argument(
        "--tree-val",
        dest="tree_val",
        action="store_true",
        help="Tree 在每个 infer block 上比较候选 max_depth 并选择验证准确率最高者。",
    )
    tree_val_group.add_argument(
        "--no-tree-val",
        dest="tree_val",
        action="store_false",
        help="Tree 固定使用 config.MAX_DEPTH。",
    )
    parser.set_defaults(
        tree_val=getattr(config, "TREE_VAL_ENABLED", False),
    )
    parser.add_argument(
        "--tree-val-depths",
        type=int,
        nargs="+",
        default=None,
        metavar="DEPTH",
        help="Tree val 候选深度；默认读取 config.TREE_VAL_MAX_DEPTH_CANDIDATES。",
    )
    parser.add_argument("--batch-size", type=int, default=getattr(config, "BATCH_SIZE", 16))
    parser.add_argument(
        "--max-rounds",
        type=int,
        default=getattr(config, "LLM_COOPERATION_MAX_ROUNDS", 5),
    )
    parser.add_argument("--output-format", choices=["json", "jsonl", "csv", "all"], default="json")
    parser.add_argument(
        "--evaluate-only",
        action="store_true",
        help="不运行模型，只重新评测 output-dir 中已有结果。",
    )
    return parser


def apply_test_mode(args: argparse.Namespace, default_output_dir: str) -> None:
    if not args.test:
        return

    args.cases_per_category = TEST_CASES_PER_CATEGORY
    args.end_index = args.start_index + TEST_CASES_PER_CATEGORY - 1
    if os.path.abspath(args.output_dir) == os.path.abspath(default_output_dir):
        args.output_dir = os.path.join(default_output_dir, "test_5")


def main() -> None:
    # multiprocessing spawn calls os.getcwd() while preparing each child.
    # An absolute script path alone cannot recover from a deleted shell cwd.
    ensure_valid_working_directory()
    parser = build_arg_parser()
    args = parser.parse_args()
    apply_test_mode(args, parser.get_default("output_dir"))
    if args.block_size <= 0:
        raise ValueError("block-size 必须 > 0")
    if args.start_index > args.end_index:
        raise ValueError("start-index 不能大于 end-index")
    window_size = args.end_index - args.start_index + 1
    if args.cases_per_category != window_size:
        raise ValueError(
            "cases-per-category 必须等于闭区间大小: "
            f"{args.start_index}..{args.end_index} 共 {window_size} 条"
        )
    if "tree" in args.methods and args.refiner_rounds <= 0:
        raise ValueError("Tree 实验必须开启 Refiner，refiner-rounds 必须 > 0")
    if (
        not args.test
        and args.cases_per_category > 0
        and args.cases_per_category % args.block_size
    ):
        raise ValueError("cases-per-category 必须能被 block-size 整除")
    ensure_dir(args.output_dir)
    save_json(vars(args), os.path.join(args.output_dir, "experiment_config.json"))

    if not args.evaluate_only:
        # All three methods intentionally share the exact same source window.
        _validate_scenarios(
            args.anomalydetect_label_root,
            args.scenario,
            args.expected_categories,
            args.cases_per_category,
            args.start_index,
            args.end_index,
        )

        ctx = mp.get_context("spawn")
        for method in args.methods:
            print(f"[Experiment] 启动独立进程: {method}")
            process = ctx.Process(
                target=_method_worker,
                args=(args, method),
                name=f"experiment_{method}",
            )
            process.start()
            process.join()
            if process.exitcode != 0:
                raise RuntimeError(f"{method} 实验失败，exitcode={process.exitcode}")

    report = evaluate_saved_results(args.output_dir, args.methods)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"评测完成: {os.path.join(args.output_dir, 'evaluation', 'metrics.json')}")


if __name__ == "__main__":
    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass
    main()
