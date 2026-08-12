# -*- coding: utf-8 -*-

# 运行示例：
# python -m bian_reproduction.run_stage1 \
#   --input-root /path/to/input_root \
#   --alarm-types "BGP邻接状态改变" "网络设备掉线" \
#                 "接口状态DOWN" "入方向带宽利用率过高" \
#   --output-dir /path/to/bian_output \
#   --model-path /usr/share/large_language_models/DeepSeek-R1-Distill-Qwen-32B \
#   --visible-devices 6,7 \
#   --start 0 \
#   --end 1

import argparse
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from bian_reproduction.evaluate import evaluate_payloads
from bian_reproduction.stage1 import BianStage1Runner, discover_label_files


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Standalone BiAn Stage 1: alert summary -> anomaly analysis -> joint scoring"
    )
    parser.add_argument("--input-root", required=True, help="包含告警类别子目录的根目录")
    parser.add_argument(
        "--alarm-type",
        "--alarm-types",
        dest="alarm_types",
        nargs="+",
        required=True,
        help="一个或多个告警类别，也就是 input-root 下的子目录名",
    )
    parser.add_argument(
        "--output-dir",
        default="./bian_output",
        help="独立输出根目录，默认 ./bian_output",
    )
    parser.add_argument(
        "--model-path",
        default="/usr/share/large_language_models/DeepSeek-R1-Distill-Qwen-32B",
    )
    parser.add_argument("--visible-devices", default="6,7")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    parser.add_argument("--start", type=int, default=None)
    parser.add_argument("--end", type=int, default=None, help="半开区间，不包含 end")
    return parser


def _select_indices(
    label_map: Dict[int, Path],
    start: Optional[int],
    end: Optional[int],
) -> List[int]:
    all_indices = sorted(label_map)
    selected_start = start if start is not None else min(all_indices)
    selected_end = end if end is not None else max(all_indices) + 1
    indices = [idx for idx in all_indices if selected_start <= idx < selected_end]
    if not indices:
        raise ValueError(
            f"索引窗口内没有数据: start={selected_start}, end={selected_end}"
        )
    return indices


def _unique_alarm_types(values: Sequence[str]) -> List[str]:
    return list(dict.fromkeys(value.strip() for value in values if value.strip()))


def run_alarm_types(
    args: argparse.Namespace,
    runner: Optional[BianStage1Runner] = None,
) -> Tuple[List[Dict[str, Any]], Path]:
    input_root = Path(args.input_root).resolve()
    alarm_types = _unique_alarm_types(args.alarm_types)
    if not alarm_types:
        raise ValueError("至少需要传入一个告警类别")

    # Validate every category before loading the model.
    run_specs = []
    for alarm_type in alarm_types:
        data_dir = input_root / alarm_type
        label_map = discover_label_files(data_dir, alarm_type)
        if not label_map:
            raise FileNotFoundError(f"未找到 label 文件: {data_dir}")
        run_specs.append((alarm_type, data_dir, _select_indices(
            label_map,
            args.start,
            args.end,
        )))

    os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
    if runner is None:
        runner = BianStage1Runner(
            model_path=args.model_path,
            visible_devices=args.visible_devices,
            batch_size=args.batch_size,
            max_new_tokens=args.max_new_tokens,
        )

    payloads = []
    for alarm_type, data_dir, indices in run_specs:
        payload = runner.run(
            data_dir=str(data_dir),
            alarm_type=alarm_type,
            indices=indices,
            output_dir=args.output_dir,
        )
        payloads.append(payload)
        print(f"[{alarm_type}] 处理完成: {payload['summary']}")
        print(f"[{alarm_type}] 输出目录: {payload['meta']['output_dir']}")

    aggregate_dir = (
        Path(args.output_dir).resolve()
        / "llm_infer_bian_stage1"
        / "evaluation"
    )
    evaluate_payloads(payloads, aggregate_dir)
    return payloads, aggregate_dir


def main() -> None:
    args = build_parser().parse_args()
    payloads, aggregate_dir = run_alarm_types(args)
    if len(payloads) > 1:
        print(f"多类别汇总评价目录: {aggregate_dir}")


if __name__ == "__main__":
    main()
