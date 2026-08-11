# -*- coding: utf-8 -*-

# 运行示例：
# python -m bian_reproduction.run_stage1 \
#   --input-root /path/to/input_root \
#   --alarm-type "BGP邻接状态改变" \
#   --output-dir /path/to/bian_output \
#   --model-path /usr/share/large_language_models/DeepSeek-R1-Distill-Qwen-32B \
#   --visible-devices 6,7 \
#   --start 0 \
#   --end 1

import argparse
import os
from pathlib import Path

from bian_reproduction.stage1 import BianStage1Runner, discover_label_files


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Standalone BiAn Stage 1: alert summary -> anomaly analysis -> joint scoring"
    )
    parser.add_argument("--input-root", required=True, help="包含告警类别子目录的根目录")
    parser.add_argument("--alarm-type", required=True, help="告警类别，也就是 input-root 下的子目录名")
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


def main() -> None:
    args = build_parser().parse_args()
    data_dir = Path(args.input_root).resolve() / args.alarm_type
    label_map = discover_label_files(data_dir, args.alarm_type)
    if not label_map:
        raise FileNotFoundError(f"未找到 label 文件: {data_dir}")

    all_indices = sorted(label_map)
    start = args.start if args.start is not None else min(all_indices)
    end = args.end if args.end is not None else max(all_indices) + 1
    indices = [idx for idx in all_indices if start <= idx < end]
    if not indices:
        raise ValueError(f"索引窗口内没有数据: start={start}, end={end}")

    os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
    runner = BianStage1Runner(
        model_path=args.model_path,
        visible_devices=args.visible_devices,
        batch_size=args.batch_size,
        max_new_tokens=args.max_new_tokens,
    )
    payload = runner.run(
        data_dir=str(data_dir),
        alarm_type=args.alarm_type,
        indices=indices,
        output_dir=args.output_dir,
    )
    print(f"处理完成: {payload['summary']}")
    print(f"输出目录: {payload['meta']['output_dir']}")


if __name__ == "__main__":
    main()
