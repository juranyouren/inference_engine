"""在同一共享模型上评测合作型和竞争型，不使用增量状态。"""

from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path
from time import perf_counter

import config
from inference.common import discover_label_index_map
from inference.metrics import calculate_metrics

from .data_adapter import read_sop, without_sop
from .io_utils import safe_category_name, save_json
from .main import configure_ascend_devices, load_json_case


def load_groups(root, limit, import_top_level_sop):
    if not root.is_dir():
        raise ValueError(f"找不到样例或 label 目录: {root}")
    groups = {}
    for path in sorted(root.glob("*.json")):
        groups.setdefault(path.stem, []).append(path)
    for folder in sorted(path for path in root.iterdir() if path.is_dir()):
        indexed = discover_label_index_map(str(folder), folder.name)
        if indexed:
            groups.setdefault(folder.name, []).extend(Path(indexed[index]) for index in sorted(indexed))

    loaded = {}
    for alarm_type, paths in groups.items():
        loaded[alarm_type] = []
        for path in paths[:limit]:
            label = load_json_case(path)
            labels = label.get("semantic_labels")
            if import_top_level_sop and isinstance(labels, dict) and not read_sop(label):
                label = {**label, "semantic_labels": {**labels, "sop": label.get("sop")}}
            loaded[alarm_type].append(label)
    return loaded


def _metrics(mode, alarm_type, results, elapsed):
    return {
        "mode": mode, "alarm_type": alarm_type, "attempted_count": len(results),
        "failed_count": sum(item["status"] == "failed" for item in results),
        **calculate_metrics(results, elapsed, rank_key="rank_v1"),
    }


def run(args):
    groups = load_groups(args.label_root, args.limit_per_category, args.import_top_level_sop)
    if not groups:
        raise ValueError("没有找到可评测的 label 文件")
    without_sop_count = sum(not read_sop(label) for labels in groups.values() for label in labels)
    if args.mode != "cooperation":
        groups = {
            category: [label for label in labels if read_sop(label)]
            for category, labels in groups.items()
        }
        groups = {category: labels for category, labels in groups.items() if labels}
        if not groups:
            raise ValueError("没有包含 semantic_labels.sop 的 case。保存的样例可用 --import-top-level-sop")

    os.environ.setdefault("ASCEND_RT_VISIBLE_DEVICES", str(config.ASCEND_RT_VISIBLE_DEVICES))
    configure_ascend_devices(args)
    from .competition import CompetitionInference
    from .cooperation import CooperationInference
    from .llm_engine import LLMEngine

    engine = LLMEngine.get_instance(model_path=args.model_path, max_model_len=args.max_model_len)
    modes = {}
    if args.mode in ("both", "competition"):
        modes["competition"] = CompetitionInference(engine, batch_size=args.batch_size)
    if args.mode in ("both", "cooperation"):
        modes["cooperation"] = CooperationInference(
            engine, batch_size=args.batch_size, invoke_limit=args.invoke_limit,
        )

    rows = []
    for mode, inference in modes.items():
        all_results = []
        total_elapsed = 0.0
        for alarm_type, labels in groups.items():
            inputs = [without_sop(label) for label in labels] if mode == "cooperation" else labels
            trace_root = args.output_dir / "traces" / safe_category_name(alarm_type)
            start = perf_counter()
            results = inference.infer_results(inputs, alarm_type, trace_root)
            elapsed = perf_counter() - start
            rows.append(_metrics(mode, alarm_type, results, elapsed))
            all_results.extend(results)
            total_elapsed += elapsed
        rows.append(_metrics(mode, None, all_results, total_elapsed))

    save_json({
        "mode": args.mode, "label_root": str(args.label_root.resolve()),
        "excluded_without_sop": without_sop_count if args.mode != "cooperation" else 0,
        "metrics": rows,
    }, args.output_dir / "metrics.json")
    with (args.output_dir / "metrics.csv").open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return rows


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--label-root", type=Path, default=Path(config.AGENTDIGEST_LABEL_ROOT),
        help="数据集根目录，默认使用 config.AGENTDIGEST_LABEL_ROOT",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("res/agent_evaluation"))
    parser.add_argument("--mode", choices=("both", "competition", "cooperation"), default="both")
    parser.add_argument("--model-path", default=config.MODEL_PATH)
    parser.add_argument("--ascend-devices")
    parser.add_argument("--max-model-len", type=int, default=16384)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--invoke-limit", type=int, default=10)
    parser.add_argument("--limit-per-category", type=int)
    parser.add_argument("--import-top-level-sop", action="store_true",
                        help="将样例顶层 SOP 显式复制到内存中的 semantic_labels.sop")
    args = parser.parse_args(argv)
    if args.limit_per_category is not None and args.limit_per_category < 1:
        parser.error("--limit-per-category 必须大于 0")
    try:
        rows = run(args)
    except (OSError, TypeError, ValueError, RuntimeError) as exc:
        parser.error(str(exc))
    print(f"评测结果已写入 {args.output_dir.resolve()}")
    return 1 if any(row["failed_count"] for row in rows) else 0


if __name__ == "__main__":
    raise SystemExit(main())
