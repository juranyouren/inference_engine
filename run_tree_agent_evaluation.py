"""按固定数据切分，运行 Tree、Competition 和 Cooperation。

用法：
    python run_tree_agent_evaluation.py --dataset-root /path/to/anomalydetect_label --mode all
    python run_tree_agent_evaluation.py \
      --dataset-root /path/to/anomalydetect_label \
      --mode tree \
      --selection-path /path/to/anomalydetect_label \
      --output-dir /path/to/results
    python run_tree_agent_evaluation.py --dataset-root /path/to/labels --mode competition
    python run_tree_agent_evaluation.py --dataset-root /path/to/labels --mode cooperation

--dataset-root 下每个直接子目录视为一个告警类别；也兼容直接包含 label 文件的单类别目录。
每类按 label index 升序取前 50 条训练 Tree，紧接着的 150 条作为测试窗口。
Competition 处理测试窗口中有 semantic_labels.sop 的样本。SOP 在 JSON 顶层时，
可传 --import-top-level-sop。Cooperation 使用同一测试窗口，并从模型输入中移除 SOP。
真实 root_cause 供 Tree 训练和结果评测使用。
"""

from __future__ import annotations

import argparse
import csv
import json
import multiprocessing as mp
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Sequence

# Must be set before importing modules that may initialize vLLM/torch_npu.
os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")

import config
from inference.common import (
    IndexedCaseReader,
    discover_label_index_map,
    ensure_dir,
    load_json,
    save_json,
)
from inference.metrics import calculate_metrics
from inference.selection import load_selection
from llm_interfaces.data_adapter import read_sop, with_top_level_sop, without_sop
from llm_interfaces.io_utils import safe_category_name, save_batch_results


MODES = ("all", "tree", "competition", "cooperation")


def _load_label(path: Path, index: int) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        label = json.load(file)
    if not isinstance(label, dict):
        raise TypeError(f"label 必须是 JSON object: {path}")
    label["_label_file_path"] = str(path.resolve())
    label.setdefault("case_id", str(index))
    return label


def _discover_datasets(dataset_root: Path) -> List[Dict[str, Any]]:
    """扫描类别子目录；若根目录自身含 label 文件，则按单类别处理。"""
    if not dataset_root.is_dir():
        raise NotADirectoryError(f"数据集目录不存在或不是文件夹: {dataset_root}")

    scenarios = []
    for folder in sorted(path for path in dataset_root.iterdir() if path.is_dir()):
        alarm_type = folder.name
        label_map = discover_label_index_map(str(folder), alarm_type)
        if label_map:
            scenarios.append({
                "name": alarm_type,
                "alarm_type": alarm_type,
                "data_dir": str(folder.resolve()),
                "label_map": label_map,
            })

    if not scenarios:
        alarm_type = dataset_root.name
        label_map = discover_label_index_map(str(dataset_root), alarm_type)
        if label_map:
            scenarios.append({
                "name": alarm_type,
                "alarm_type": alarm_type,
                "data_dir": str(dataset_root.resolve()),
                "label_map": label_map,
            })
    if not scenarios:
        raise ValueError(f"目录及其直接子目录中没有找到可识别的 label 文件: {dataset_root}")
    return scenarios


def _split_scenario(scenario: Dict[str, Any], train_count: int, test_count: int):
    indices = list(scenario["label_map"])
    required = train_count + test_count
    if len(indices) < required:
        raise ValueError(
            f"类别 {scenario['name']} 数据不足：需要 {required} 个 label，"
            f"实际找到 {len(indices)} 个"
        )
    return indices[:train_count], indices[train_count:required]


def _metric_row(
    method: str,
    results: Sequence[Dict[str, Any]],
    elapsed: float,
    *,
    rank_key: str,
    selected_count: int,
    skipped_count: int = 0,
    category: str | None = None,
) -> Dict[str, Any]:
    return {
        "method": method,
        "category": category,
        "selected_count": selected_count,
        "failed_count": sum(item.get("status") == "failed" for item in results),
        "skipped_count": skipped_count,
        **calculate_metrics(results, elapsed, rank_key=rank_key),
    }


def _write_metrics(output_dir: Path, rows: List[Dict[str, Any]]) -> None:
    save_json({"metrics": rows}, str(output_dir / "metrics.json"))
    if rows:
        with (output_dir / "metrics.csv").open("w", encoding="utf-8-sig", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


def _write_mode_report(output_dir: Path, mode: str, metadata, rows) -> None:
    save_json({**metadata, "mode": mode, "metrics": rows}, str(output_dir / "evaluation.json"))
    _write_metrics(output_dir, rows)


def _configure_devices(raw_devices: str | None) -> None:
    devices = raw_devices or os.environ.get("ASCEND_RT_VISIBLE_DEVICES")
    if not devices:
        devices = str(getattr(config, "ASCEND_RT_VISIBLE_DEVICES", ""))
    if not devices:
        return
    parts = [part.strip() for part in devices.split(",")]
    if any(not part.isdigit() for part in parts) or len(set(parts)) != len(parts):
        raise ValueError("Ascend 卡号格式无效，请使用如 0,1 的格式且不要重复")
    os.environ["ASCEND_RT_VISIBLE_DEVICES"] = ",".join(parts)


def run_tree(
    scenarios: List[Dict[str, Any]],
    args: argparse.Namespace,
    output_dir: Path,
    split_info: Dict[str, Dict[str, List[int]]],
) -> List[Dict[str, Any]]:
    from inference.tree import run_tree_once

    rows = []
    category_results = {}
    total_results = []
    total_elapsed = 0.0
    selection_root = args.selection_path.resolve() if args.selection_path else args.dataset_root.resolve()

    for scenario in scenarios:
        train_indices = split_info[scenario["name"]]["train_indices"]
        test_indices = split_info[scenario["name"]]["test_indices"]
        reader = IndexedCaseReader(scenario["data_dir"], scenario["alarm_type"])
        train_cases = reader.load_cases(train_indices)
        test_cases = reader.load_cases(test_indices)
        selection = load_selection(scenario, str(selection_root))
        category_output = output_dir / safe_category_name(scenario["name"])
        started = time.perf_counter()
        result = run_tree_once(
            scenario=scenario,
            train_cases=train_cases,
            infer_cases=test_cases,
            train_indices=train_indices,
            infer_indices=test_indices,
            selection=selection,
            output_dir=str(category_output),
            output_format="all",
            tag="fixed_split",
        )
        elapsed = time.perf_counter() - started
        results = result.get("results", [])
        category_results[scenario["name"]] = results
        total_results.extend(results)
        total_elapsed += elapsed
        rows.append(_metric_row(
            "tree", results, elapsed, rank_key="rank",
            selected_count=len(test_indices), category=scenario["name"],
        ))

    rows.append(_metric_row(
        "tree", total_results, total_elapsed, rank_key="rank",
        selected_count=len(total_results), category="ALL",
    ))
    save_json({"results": category_results}, str(output_dir / "predictions.json"))
    return rows


def _run_tree_report(scenarios, args, output_dir, split_info, metadata) -> None:
    rows = run_tree(scenarios, args, output_dir, split_info)
    _write_mode_report(output_dir, "tree", metadata, rows)


def run_agents(
    scenarios: List[Dict[str, Any]],
    args: argparse.Namespace,
    output_root: Path,
    split_info: Dict[str, Dict[str, List[int]]],
    methods: Sequence[str],
    metadata: Dict[str, Any],
) -> Dict[str, List[Dict[str, Any]]]:
    _configure_devices(args.ascend_devices)
    from llm_interfaces.competition import CompetitionInference
    from llm_interfaces.cooperation import CooperationInference
    from llm_interfaces.llm_engine import LLMEngine

    engine = LLMEngine.get_instance(
        model_path=args.model_path,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
    )
    inferences = {}
    if "competition" in methods:
        inferences["competition"] = CompetitionInference(engine, batch_size=args.batch_size)
    if "cooperation" in methods:
        inferences["cooperation"] = CooperationInference(
            engine, batch_size=args.batch_size, invoke_limit=args.invoke_limit,
        )

    reports = {}
    for method in methods:
        output_dir = output_root / method
        ensure_dir(str(output_dir))
        rows = []
        all_results = []
        total_elapsed = 0.0
        trace_dir = output_dir / "traces"
        for scenario in scenarios:
            test_indices = split_info[scenario["name"]]["test_indices"]
            label_map = scenario["label_map"]
            labels = {
                index: _load_label(Path(label_map[index]), index)
                for index in test_indices
            }
            if method == "competition":
                if args.import_top_level_sop:
                    labels = {index: with_top_level_sop(label) for index, label in labels.items()}
                selected = [(index, label) for index, label in labels.items() if read_sop(label)]
            else:
                selected = list(labels.items())

            indices = [index for index, _label in selected]
            inputs = [
                without_sop(label) if method == "cooperation" else label
                for _index, label in selected
            ]
            started = time.perf_counter()
            results = inferences[method].infer_results(
                inputs,
                scenario["alarm_type"],
                str(trace_dir),
                case_indices=indices,
            ) if inputs else []
            elapsed = time.perf_counter() - started
            total_elapsed += elapsed
            all_results.extend(results)
            rows.append(_metric_row(
                method,
                results,
                elapsed,
                rank_key="rank_v1",
                selected_count=len(indices),
                skipped_count=len(test_indices) - len(indices),
                category=scenario["name"],
            ))

        rows.append(_metric_row(
            method, all_results, total_elapsed, rank_key="rank_v1",
            selected_count=len(all_results),
            skipped_count=sum(row["skipped_count"] for row in rows), category="ALL",
        ))
        save_batch_results(mode=method, output_dir=trace_dir / method, results=all_results)
        _write_mode_report(output_dir, method, metadata, rows)
        reports[method] = rows
    return reports


def run(args: argparse.Namespace) -> Dict[str, Any]:
    scenarios = _discover_datasets(args.dataset_root.resolve())
    split_info = {}
    for scenario in scenarios:
        train_indices, test_indices = _split_scenario(
            scenario, args.train_count, args.test_count,
        )
        split_info[scenario["name"]] = {
            "train_indices": train_indices,
            "test_indices": test_indices,
        }

    output_root = args.output_dir.resolve()
    ensure_dir(str(output_root))
    methods = MODES[1:] if args.mode == "all" else (args.mode,)
    metadata = {
        "dataset_root": str(args.dataset_root.resolve()),
        "train_count": args.train_count,
        "test_count": args.test_count,
        "selection_path": str(args.selection_path.resolve()) if args.selection_path else None,
        "import_top_level_sop": args.import_top_level_sop,
        "splits": split_info,
    }
    metrics_by_mode = {}
    if "tree" in methods:
        tree_dir = output_root / "tree"
        tree_args = (scenarios, args, tree_dir, split_info, metadata)
        if args.mode == "all":
            worker = mp.get_context("spawn").Process(target=_run_tree_report, args=tree_args)
            worker.start()
            worker.join()
            if worker.exitcode != 0:
                raise RuntimeError(f"Tree 评测失败，退出码 {worker.exitcode}")
        else:
            _run_tree_report(*tree_args)
        metrics_by_mode["tree"] = load_json(str(tree_dir / "evaluation.json"))["metrics"]
    llm_methods = [method for method in methods if method != "tree"]
    if llm_methods:
        metrics_by_mode.update(run_agents(scenarios, args, output_root, split_info, llm_methods, metadata))

    report = {**metadata, "mode": args.mode, "metrics_by_mode": metrics_by_mode}
    if args.mode == "all":
        save_json(report, str(output_root / "evaluation.json"))
        _write_metrics(output_root, [row for rows in metrics_by_mode.values() for row in rows])
    return report


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True,
                        help="数据集根目录，自动扫描其中所有类别子目录")
    parser.add_argument("--mode", choices=MODES, required=True,
                        help="运行 all，或单独运行 tree、competition、cooperation")
    parser.add_argument("--output-dir", type=Path, default=Path("res/separate_evaluation"))
    parser.add_argument("--train-count", type=int, default=50)
    parser.add_argument("--test-count", type=int, default=150)
    parser.add_argument("--selection-path", type=Path,
                        help="Tree selection.json 或包含每类 selection.json 的目录")
    parser.add_argument("--model-path", default=config.MODEL_PATH)
    parser.add_argument("--ascend-devices")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument("--max-model-len", type=int, default=16384)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--invoke-limit", type=int, default=10)
    parser.add_argument("--import-top-level-sop", action="store_true",
                        help="在内存中将顶层 SOP 导入 semantic_labels.sop")
    return parser


def main() -> int:
    parser = build_arg_parser()
    args = parser.parse_args()
    if args.train_count <= 0 or args.test_count <= 0:
        parser.error("--train-count 和 --test-count 必须大于 0")
    try:
        report = run(args)
    except (OSError, TypeError, ValueError, RuntimeError) as exc:
        parser.error(str(exc))
    report_dir = args.output_dir if args.mode == "all" else args.output_dir / args.mode
    print(f"评测完成：{report_dir.resolve()}")
    failed = False
    for rows in report["metrics_by_mode"].values():
        for row in rows:
            print(json.dumps(row, ensure_ascii=False))
            failed |= bool(row["failed_count"])
    return int(failed)


if __name__ == "__main__":
    raise SystemExit(main())
