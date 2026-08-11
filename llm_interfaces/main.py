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
import gc
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
    """在模型初始化前配置可见 NPU，并自动确定张量并行卡数。"""
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
    visible_count = len(device_ids)

    if args.tensor_parallel_size is None:
        args.tensor_parallel_size = visible_count
    elif args.tensor_parallel_size != visible_count:
        raise ValueError(
            "--tensor-parallel-size 必须等于可见 NPU 数量: "
            f"visible={visible_count}, tensor_parallel_size="
            f"{args.tensor_parallel_size}"
        )
    os.environ["ASCEND_RT_VISIBLE_DEVICES"] = normalized
    print(
        "Ascend NPU 配置: "
        f"ASCEND_RT_VISIBLE_DEVICES={normalized}, "
        f"tensor_parallel_size={args.tensor_parallel_size}"
    )


def _infer_one_case(
    *,
    mode: str,
    model_path: str,
    label: Dict[str, Any],
    alarm_type: Optional[str],
    output_dir: str,
    batch_size: int,
    max_rounds: int,
    tensor_parallel_size: Optional[int],
    gpu_memory_utilization: float,
    max_model_len: int,
    max_tokens: int,
    prompt_safety_tokens: int,
) -> None:
    common_options = {
        "alarm_type": alarm_type,
        "batch_size": batch_size,
        "tensor_parallel_size": tensor_parallel_size,
        "gpu_memory_utilization": gpu_memory_utilization,
        "max_model_len": max_model_len,
        "max_tokens": max_tokens,
        "prompt_safety_tokens": prompt_safety_tokens,
    }

    if mode == "competition":
        from .competition import CompetitionInference

        engine = CompetitionInference(model_path, **common_options)
    elif mode == "cooperation":
        from .cooperation import CooperationInference

        engine = CooperationInference(
            model_path,
            max_rounds=max_rounds,
            **common_options,
        )
    else:  # pragma: no cover - 只允许由参数解析后的 mode 调用
        raise ValueError(f"不支持的推理模式: {mode}")

    results = engine.infer([label], output_dir)
    if not results or results[0].get("status") != "success":
        error = results[0].get("error") if results else "接口未返回结果"
        raise RuntimeError(f"{mode} 单 case 推理失败: {error}")

    print(
        f"[{mode}] 推理成功，case_id={results[0].get('case_id')}，"
        f"结果目录={results[0].get('output_dir')}"
    )
    # 当前模式结束后删除 Python 侧引用，再初始化下一个类。
    del engine
    gc.collect()


def selected_jobs(args: argparse.Namespace) -> List[ModeJob]:
    jobs: List[ModeJob] = []
    if args.mode in ("both", "competition"):
        if not args.competition_json:
            raise ValueError(
                "mode=both/competition 时必须提供 --competition-json"
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

    for mode, json_path, alarm_type in jobs:
        label = load_json_case(json_path)
        print(f"开始 {mode} 单 case 测试: {json_path}")
        _infer_one_case(
            mode=mode,
            model_path=args.model_path,
            label=label,
            alarm_type=alarm_type,
            output_dir=str(output_root),
            batch_size=args.batch_size,
            max_rounds=args.max_rounds,
            tensor_parallel_size=args.tensor_parallel_size,
            gpu_memory_utilization=args.gpu_memory_utilization,
            max_model_len=args.max_model_len,
            max_tokens=args.max_tokens,
            prompt_safety_tokens=args.prompt_safety_tokens,
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
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-rounds", type=int, default=5)
    parser.add_argument("--tensor-parallel-size", type=int, default=None)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument("--max-model-len", type=int, default=16384)
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--prompt-safety-tokens", type=int, default=512)
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
