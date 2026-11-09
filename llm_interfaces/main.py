"""串行测试 Competition 与 Cooperation 接口，各推理一个 case。

示例：
    python -m llm_interfaces.main \
        --model-path /path/to/model \
        --competition-json /path/to/competition_label.json \
        --cooperation-json /path/to/cooperation_label.json \
        --output-dir /path/to/output

两个模式直接在当前 main 进程中串行执行，不创建子进程。
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

# 必须在导入可能初始化 vLLM/torch_npu 的模块之前设置。
os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")


ModeJob = Tuple[str, str, Optional[str]]


def load_json_case(path: str | Path) -> Dict[str, Any]:
    file_path = Path(path)
    with file_path.open("r", encoding="utf-8") as file:
        data = json.load(file)
    if not isinstance(data, dict):
        raise TypeError(
            f"测试输入必须是单个 JSON 对象: {file_path}，"
            f"实际为 {type(data).__name__}"
        )
    return data


def configure_ascend_devices(args: argparse.Namespace) -> None:
    """在 LLMEngine.get_instance() 之前配置可见 NPU。"""
    raw_devices = args.ascend_devices
    if raw_devices is None:
        raw_devices = os.environ.get("ASCEND_RT_VISIBLE_DEVICES")
    if raw_devices is None or not raw_devices.strip():
        return

    device_ids = [item.strip() for item in raw_devices.split(",")]
    if any(not item or not item.isdigit() for item in device_ids):
        raise ValueError(
            "--ascend-devices 格式错误，请使用英文逗号分隔的卡号，"
            "例如 0,1 或 2,3,4,5"
        )
    if len(set(device_ids)) != len(device_ids):
        raise ValueError("--ascend-devices 不能包含重复卡号")

    normalized = ",".join(device_ids)
    os.environ["ASCEND_RT_VISIBLE_DEVICES"] = normalized
    print(
        "Ascend NPU 配置: "
        f"ASCEND_RT_VISIBLE_DEVICES={normalized}, "
        f"tensor_parallel_size={len(device_ids)}"
    )


def _infer_one_case(
    *,
    mode: str,
    inference,
    label: Dict[str, Any],
    alarm_type: str,
    output_dir: str,
) -> None:
    results = inference.infer_results([label], alarm_type, output_dir)
    if not results:
        raise RuntimeError(f"{mode} 单 case 推理未返回结果")
    if results[0]["status"] != "success":
        raise RuntimeError(f"{mode} 单 case 推理失败: {results[0]['error']}")

    print(
        f"[{mode}] 推理完成，case_idx={results[0].get('case_idx')}，"
        f"predicted_list={results[0].get('predicted_list')}，"
        f"rank={results[0].get('rank')}"
    )


def selected_jobs(args: argparse.Namespace) -> List[ModeJob]:
    jobs: List[ModeJob] = []
    if args.mode in ("both", "competition"):
        if not args.competition_json:
            raise ValueError(
                "mode=both/competition 时必须提供 --competition-json"
            )
        if not args.competition_alarm_type:
            raise ValueError(
                "mode=both/competition 时必须提供 --competition-alarm-type"
            )
        jobs.append((
            "competition",
            args.competition_json,
            args.competition_alarm_type,
        ))
    if args.mode in ("both", "cooperation"):
        if not args.cooperation_json:
            raise ValueError(
                "mode=both/cooperation 时必须提供 --cooperation-json"
            )
        if not args.cooperation_alarm_type:
            raise ValueError(
                "mode=both/cooperation 时必须提供 --cooperation-alarm-type"
            )
        jobs.append((
            "cooperation",
            args.cooperation_json,
            args.cooperation_alarm_type,
        ))
    return jobs


def run(args: argparse.Namespace) -> None:
    configure_ascend_devices(args)
    jobs = selected_jobs(args)
    output_root = Path(args.output_dir).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    from .competition import CompetitionInference
    from .cooperation import CooperationInference
    from .llm_engine import LLMEngine

    engine = LLMEngine.get_instance(
        model_path=args.model_path,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
    )
    inference_by_mode = {}
    if args.mode in ("both", "competition"):
        inference_by_mode["competition"] = CompetitionInference(engine)
    if args.mode in ("both", "cooperation"):
        inference_by_mode["cooperation"] = CooperationInference(engine)

    for mode, json_path, alarm_type in jobs:
        label = load_json_case(json_path)
        print(f"开始 {mode} 单 case 测试: {json_path}")
        _infer_one_case(
            mode=mode,
            inference=inference_by_mode[mode],
            label=label,
            alarm_type=alarm_type,
            output_dir=str(output_root),
        )
        print(f"{mode} 单 case 测试完成。")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "测试 Competition/Cooperation 接口，各推理一个 JSON case；"
            "两个模式在当前 main 进程中直接串行运行。"
        )
    )
    parser.add_argument(
        "--mode",
        choices=("both", "competition", "cooperation"),
        default="both",
    )
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--competition-json")
    parser.add_argument("--cooperation-json")
    parser.add_argument("--competition-alarm-type")
    parser.add_argument("--cooperation-alarm-type")
    parser.add_argument(
        "--output-dir",
        default="res",
        help="结果根目录，默认 res。",
    )
    parser.add_argument(
        "--ascend-devices",
        help=(
            "当前 main 进程可见的 Ascend NPU 卡号，例如 0 或 0,1；"
            "未设置时沿用环境变量 ASCEND_RT_VISIBLE_DEVICES。"
        ),
    )
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument("--max-model-len", type=int, default=16384)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    try:
        run(args)
    except (OSError, TypeError, ValueError, RuntimeError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
