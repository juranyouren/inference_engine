# infer_by_index.py
# -*- coding: utf-8 -*-

import os

# Ascend NPU + vLLM multiprocessing must use spawn.
# This block must stay before importing any module that may import torch_npu/vllm.
os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")

import multiprocessing as mp

try:
    mp.set_start_method("spawn", force=True)
except RuntimeError:
    pass

import argparse
import sys
from pathlib import Path

CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

import config  # noqa: E402

PROJECT_ROOT = Path(
    getattr(config, "PROJECT_ROOT", str(CURRENT_DIR))
).resolve()

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

if hasattr(config, "ASCEND_RT_VISIBLE_DEVICES"):
    os.environ["ASCEND_RT_VISIBLE_DEVICES"] = str(
        config.ASCEND_RT_VISIBLE_DEVICES
    )

from llm_pipeline import llm_infer  # noqa: E402
from tree_pipeline import tree_infer  # noqa: E402


# ============================================================
# 10. main
# ============================================================


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "扫描两个 label 根目录的类别子目录，"
            "按状态文件只处理新增数据。默认同时执行 tree_infer 和 llm_infer。"
        )
    )

    parser.add_argument(
        "--infer-type",
        choices=["all", "tree_infer", "llm_infer"],
        default=getattr(config, "DEFAULT_INFER_TYPE", "all"),
        help="默认 all：同时执行 tree_infer 和 llm_infer。",
    )

    parser.add_argument(
        "--scenario",
        default="all",
        help="默认 all：处理根目录下所有类别子目录；也可指定单个子目录名。",
    )

    # 两个输入根目录
    parser.add_argument(
        "--agentdigest-label-root",
        default=getattr(
            config,
            "AGENTDIGEST_LABEL_ROOT",
            "/home/sbp/deployment/case_pool/AgentDigest_label",
        ),
        help="LLM 推理数据根目录。",
    )

    parser.add_argument(
        "--anomalydetect-label-root",
        default=getattr(
            config,
            "ANOMALYDETECT_LABEL_ROOT",
            "/home/sbp/deployment/case_pool/anomalydetect_label",
        ),
        help="Tree 推理数据根目录。",
    )

    # 增量状态
    parser.add_argument(
        "--state-file",
        default=getattr(
            config,
            "INFER_STATE_FILE",
            "/home/sbp/deployment/case_pool/predict_result/state/infer_state.json",
        ),
        help="记录每个文件夹、每种推理上次处理到的 index。",
    )

    parser.add_argument(
        "--reset-state",
        action="store_true",
        help="本次忽略已有 checkpoint，从 STATE_INITIAL_LAST_INDEX 后重新扫描。",
    )

    parser.add_argument(
        "--max-new-cases",
        type=int,
        default=getattr(config, "MAX_NEW_CASES", None),
        help="每个类别本次最多处理多少条新增数据；默认不限制。",
    )

    # 手动调试窗口；使用后不更新 state。
    parser.add_argument(
        "--start",
        type=int,
        default=None,
        help="调试窗口起点；传入 start/end 后不更新增量状态。",
    )

    parser.add_argument(
        "--end",
        type=int,
        default=None,
        help="调试窗口终点（不含）；传入 start/end 后不更新增量状态。",
    )

    # Tree
    parser.add_argument(
        "--train-n",
        type=int,
        default=None,
        help="Tree 使用当前新增数据之前最近多少条样本训练。",
    )

    parser.add_argument(
        "--selection-path",
        default=None,
        help="selection.json、selector/res.jsonl 或包含它们的目录。",
    )

    parser.add_argument(
        "--selection-source",
        choices=["file", "selector", "selector_refiner"],
        default=getattr(config, "SELECTION_SOURCE", "file"),
        help=(
            "file=读取已有 selection；"
            "selector=现场生成；"
            "selector_refiner=Selector -> Tree summary -> Refiner。"
        ),
    )

    parser.add_argument(
        "--refiner-rounds",
        type=int,
        default=getattr(config, "REFINER_ROUNDS", 1),
        help="selector_refiner 的 Refiner 轮数。",
    )

    # LLM
    parser.add_argument(
        "--llm-mode",
        choices=["auto", "competition", "cooperation"],
        default=getattr(config, "DEFAULT_LLM_MODE", "auto"),
        help=(
            "默认 auto：semantic_labels['sop'] 非空走 competition，"
            "为空走 cooperation。Competition=多采样+Verifier；"
            "Cooperation=多轮 Meta/Reasoner。"
        ),
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="LLM batch size；默认使用 config.BATCH_SIZE。",
    )

    parser.add_argument(
        "--max-rounds",
        type=int,
        default=getattr(config, "LLM_COOPERATION_MAX_ROUNDS", 5),
        help="Cooperation 最大 Meta/Reasoner 迭代轮数。",
    )

    # 输出
    parser.add_argument(
        "--output-dir",
        default=None,
        help="自定义输出根目录；默认使用 config.PREDICT_RES_DIR。",
    )

    parser.add_argument(
        "--output-format",
        choices=["json", "jsonl", "csv", "all"],
        default="all",
    )

    parser.add_argument(
        "--strict",
        action="store_true",
        help="任一类别失败时立即抛出异常；默认记录错误后继续其他类别。",
    )

    return parser


def _run_tree_process(args: argparse.Namespace) -> None:
    tree_infer(args)


def _run_llm_process(args: argparse.Namespace) -> None:
    llm_infer(args)


def run_all_in_separate_processes(args: argparse.Namespace) -> None:
    """
    Tree 与 LLM 使用两个独立 spawn 进程，避免 Selector/Refiner 与 RCAGenerator
    在同一个已初始化 NPU 的 Python 进程中互相影响。
    """
    ctx = mp.get_context("spawn")

    print("=" * 100)
    print("[all] 启动 tree_infer 独立进程")
    print("=" * 100)
    tree_process = ctx.Process(
        target=_run_tree_process,
        args=(args,),
        name="tree_infer_process",
    )
    tree_process.start()
    tree_process.join()
    if tree_process.exitcode != 0:
        raise RuntimeError(f"tree_infer 进程失败，exitcode={tree_process.exitcode}")

    print("=" * 100)
    print("[all] 启动 llm_infer 独立进程")
    print("=" * 100)
    llm_process = ctx.Process(
        target=_run_llm_process,
        args=(args,),
        name="llm_infer_process",
    )
    llm_process.start()
    llm_process.join()
    if llm_process.exitcode != 0:
        raise RuntimeError(f"llm_infer 进程失败，exitcode={llm_process.exitcode}")


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    print("=" * 100)
    print("inference_engine 增量推理启动")
    print(f"infer_type               : {args.infer_type}")
    print(f"scenario                 : {args.scenario}")
    print(f"AgentDigest_label root   : {args.agentdigest_label_root}")
    print(f"anomalydetect_label root : {args.anomalydetect_label_root}")
    print(f"state_file               : {args.state_file}")
    print("=" * 100)

    if args.infer_type == "tree_infer":
        tree_infer(args)
    elif args.infer_type == "llm_infer":
        llm_infer(args)
    elif args.infer_type == "all":
        run_all_in_separate_processes(args)
    else:
        raise ValueError(f"未知 infer_type: {args.infer_type}")

    print("所有增量推理任务完成。")


if __name__ == "__main__":
    main()
