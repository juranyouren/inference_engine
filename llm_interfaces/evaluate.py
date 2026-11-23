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

from .data_adapter import read_sop, with_top_level_sop, without_sop
from .io_utils import safe_category_name, save_json
from .main import configure_ascend_devices, load_json_case


def load_groups(
    root, limit, import_top_level_sop, *, require_sop=False, limit_per_mode=None,
    shard_count=1, shard_index=0,
):
    if shard_count < 1 or not 0 <= shard_index < shard_count:
        raise ValueError("分片编号必须从 0 开始且小于分片总数")
    if not root.is_dir():
        raise ValueError(f"找不到样例或 label 目录: {root}")
    groups = {}
    for path in sorted(root.glob("*.json")):
        groups.setdefault(path.stem, []).append(path)
    for folder in sorted(path for path in root.iterdir() if path.is_dir()):
        indexed = discover_label_index_map(str(folder), folder.name)
        if indexed:
            groups.setdefault(folder.name, []).extend(Path(indexed[index]) for index in sorted(indexed))

    loaded = {alarm_type: [] for alarm_type in groups}
    excluded_without_sop = 0
    selected_count = 0
    for alarm_type, paths in sorted(groups.items()):
        for case_index, path in enumerate(paths[:limit]):
            if selected_count == limit_per_mode:
                return loaded, excluded_without_sop
            assigned = case_index % shard_count == shard_index
            if not assigned and limit_per_mode is None:
                continue
            label = load_json_case(path)
            if import_top_level_sop:
                label = with_top_level_sop(label)
            if require_sop and not read_sop(label):
                excluded_without_sop += int(assigned)
                continue
            selected_count += 1
            if assigned:
                label["_label_file_path"] = str(path.resolve())
                loaded[alarm_type].append((case_index, label))
    return loaded, excluded_without_sop


def _metrics(mode, alarm_type, results, elapsed):
    return {
        "mode": mode, "alarm_type": alarm_type, "attempted_count": len(results),
        "failed_count": sum(item["status"] == "failed" for item in results),
        **calculate_metrics(results, elapsed, rank_key="rank_v1"),
    }


def _write_report(output_dir, metadata, rows):
    save_json({**metadata, "metrics": rows}, output_dir / "metrics.json")
    with (output_dir / "metrics.csv").open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _load_agents(args, mode_names):
    os.environ.setdefault("ASCEND_RT_VISIBLE_DEVICES", str(config.ASCEND_RT_VISIBLE_DEVICES))
    configure_ascend_devices(args)
    from .competition import CompetitionInference
    from .cooperation import CooperationInference
    from .llm_engine import LLMEngine

    engine = LLMEngine.get_instance(model_path=args.model_path, max_model_len=args.max_model_len)
    modes = {}
    if "competition" in mode_names:
        modes["competition"] = CompetitionInference(engine, batch_size=args.batch_size)
    if "cooperation" in mode_names:
        modes["cooperation"] = CooperationInference(
            engine, batch_size=args.batch_size, invoke_limit=args.invoke_limit,
        )
    return modes


def run(args):
    shard_count = getattr(args, "shard_count", 1)
    shard_index = getattr(args, "shard_index", 0)
    limit_per_mode = getattr(args, "limit_per_mode", None)
    # 全量双模式各取适用样本；限量对照实验沿用两种模式共享的 SOP 样本集。
    full_both = args.mode == "both" and args.limit_per_category is None and limit_per_mode is None
    groups, without_sop_count = load_groups(
        args.label_root, args.limit_per_category, args.import_top_level_sop,
        require_sop=args.mode != "cooperation" and not full_both, limit_per_mode=limit_per_mode,
        shard_count=shard_count, shard_index=shard_index,
    )
    if not groups:
        raise ValueError("没有找到可评测的 label 文件")

    groups = {category: cases for category, cases in groups.items() if cases}
    if not groups and shard_count == 1:
        raise ValueError("没有包含 semantic_labels.sop 的 case。保存的样例可用 --import-top-level-sop")

    mode_names = ("competition", "cooperation") if args.mode == "both" else (args.mode,)
    modes = _load_agents(args, mode_names) if groups else {}

    rows = []
    first_category = getattr(args, "first_category", None)
    jobs = [(mode, category) for mode in mode_names for category in groups]
    jobs.sort(key=lambda job: job[1] != first_category)
    results_by_mode = {mode: [] for mode in mode_names}
    elapsed_by_mode = {mode: 0.0 for mode in mode_names}
    for mode, alarm_type in jobs:
        cases = groups[alarm_type]
        if full_both and mode == "competition":
            eligible = [(index, label) for index, label in cases if read_sop(label)]
            without_sop_count += len(cases) - len(eligible)
            if not eligible:
                continue
            cases = eligible
        indices, labels = zip(*cases)
        labels = list(labels)
        inputs = [without_sop(label) for label in labels] if mode == "cooperation" else labels
        trace_root = args.output_dir / "traces" / safe_category_name(alarm_type)
        start = perf_counter()
        results = modes[mode].infer_results(
            inputs, alarm_type, trace_root, case_indices=indices if shard_count > 1 or full_both else None,
        )
        elapsed = perf_counter() - start
        rows.append(_metrics(mode, alarm_type, results, elapsed))
        results_by_mode[mode].extend(
            {key: row.get(key) for key in ("status", "groundtruth", "rank_v1", "parse_strategy")}
            for row in results
        )
        elapsed_by_mode[mode] += elapsed
    for mode in mode_names:
        rows.append(_metrics(mode, None, results_by_mode[mode], elapsed_by_mode[mode]))
    rows.sort(key=lambda row: mode_names.index(row["mode"]))


    _write_report(args.output_dir, {
        "mode": args.mode, "label_root": str(args.label_root.resolve()),
        "excluded_without_sop": without_sop_count,
        "limit_per_mode": limit_per_mode,
        "shard_count": shard_count, "shard_index": shard_index,
    }, rows)
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
    parser.add_argument("--first-category", help="优先运行该告警类别的所有已选模式")
    parser.add_argument("--limit-per-category", type=int)
    parser.add_argument(
        "--limit-per-mode", type=int, help="每种模式最多评测前 N 个符合条件的 case，跨类别及分片累计",
    )
    parser.add_argument("--shard-count", type=int, default=1, help="每类样本轮流分给几个分片")
    parser.add_argument("--shard-index", type=int, default=0, help="本进程的分片编号，从 0 开始")
    parser.add_argument("--import-top-level-sop", action="store_true",
                        help="将样例顶层 SOP 显式复制到内存中的 semantic_labels.sop")
    args = parser.parse_args(argv)
    for option, limit in (("--limit-per-category", args.limit_per_category),
                          ("--limit-per-mode", args.limit_per_mode)):
        if limit is not None and limit < 1:
            parser.error(f"{option} 必须大于 0")
    try:
        rows = run(args)
    except (OSError, TypeError, ValueError, RuntimeError) as exc:
        parser.error(str(exc))
    print(f"评测结果已写入 {args.output_dir.resolve()}")
    return 1 if any(row["failed_count"] for row in rows) else 0


if __name__ == "__main__":
    raise SystemExit(main())
