#!/usr/bin/env bash
# 用法：
#   bash run_all_inference_modes.sh DATASET_ROOT OUTPUT_DIR [Python参数...]
#
# 示例：
#   bash run_all_inference_modes.sh \
#     /home/sbp/deploy/case_pool/anomalydetect_label_0 \
#     /home/sbp/deployment/case_pool/predict_result/fixed_split \
#     --ascend-devices 6,7 --import-top-level-sop
#
# DATASET_ROOT 下每个类别子文件夹都会参与运行；每类前 50 个 label 训练 Tree，
# 紧接着的 150 个 label 用作测试。Tree 在独立进程中先运行，随后两个 Agent 共用一次 LLM 初始化。
# Tree 每次从当前类别首个训练样本调用 Selector 重新生成 selection，结果写入 OUTPUT_DIR/tree/类别/。
# Tree 进程内各类别共用一次 Selector 模型初始化。
# 读取已有配置可传 --selection-source file --selection-path /path/to/selections。
# 兼容旧用法中的第三个 SELECTION_PATH 位置参数，仅在 --selection-source file 时使用。
# 数据集 SOP 在 JSON 顶层时传 --import-top-level-sop。
# 两种 Agent 按类别和阶段跨 case 组批。用 --batch-size N 设置每次提交的 prompt 上限，默认 8。

set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "缺少参数。查看脚本顶部的用法说明。" >&2
  exit 2
fi

DATASET_ROOT=$1
OUTPUT_DIR=$2
shift 2

selection_args=()
if [[ $# -gt 0 && $1 != --* ]]; then
  selection_args=(--selection-path "$1")
  shift
fi

PYTHON_BIN=${PYTHON_BIN:-python}
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)

exec "$PYTHON_BIN" "$SCRIPT_DIR/run_tree_agent_evaluation.py" \
  --dataset-root "$DATASET_ROOT" \
  --output-dir "$OUTPUT_DIR" \
  "${selection_args[@]}" \
  "$@" --mode all
