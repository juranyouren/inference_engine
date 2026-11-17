#!/usr/bin/env bash
# 用法：
#   bash run_all_inference_modes.sh DATASET_ROOT OUTPUT_DIR [SELECTION_PATH] [Python参数...]
#
# 示例：
#   bash run_all_inference_modes.sh \
#     /home/sbp/deploy/case_pool/anomalydetect_label_0 \
#     /home/sbp/deployment/case_pool/predict_result/fixed_split \
#     /path/to/selections \
#     --ascend-devices 6,7 --import-top-level-sop
#
# DATASET_ROOT 下每个类别子文件夹都会参与运行；每类前 50 个 label 训练 Tree，
# 紧接着的 150 个 label 用作测试。Tree 在独立进程中先运行，随后两个 Agent 共用一次 LLM 初始化。
# Tree selection 默认从 DATASET_ROOT/类别名/selection.json 查找；也可传 selection 目录或文件。
# 数据集 SOP 在 JSON 顶层时传 --import-top-level-sop。

set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "缺少参数。查看脚本顶部的用法说明。" >&2
  exit 2
fi

DATASET_ROOT=$1
OUTPUT_DIR=$2
shift 2

SELECTION_PATH=$DATASET_ROOT
if [[ $# -gt 0 && $1 != --* ]]; then
  SELECTION_PATH=$1
  shift
fi

PYTHON_BIN=${PYTHON_BIN:-python}
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)

exec "$PYTHON_BIN" "$SCRIPT_DIR/run_tree_agent_evaluation.py" \
  --dataset-root "$DATASET_ROOT" \
  --output-dir "$OUTPUT_DIR" \
  --selection-path "$SELECTION_PATH" \
  "$@" --mode all
