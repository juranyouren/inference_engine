#!/usr/bin/env bash
# 用法：bash run_tree_only.sh DATASET_ROOT OUTPUT_DIR [Python 参数...]
# 示例（在仓库根目录运行）：
#   bash run_tree_only.sh /home/sbp/deploy/case_pool/anomalydetect_label_0 \
#     res/tree_rounds --ascend-devices 6,7
# 默认每类三轮：前 50、100、150 条分别重新生成 selection 和训练 Tree，各预测随后 50 条。
# --tree-block-size 可调整每轮条数，最后一轮使用剩余样本。默认导入顶层 SOP。
# 每轮原始排名和 selection 保存在 OUTPUT_DIR/tree/<类别>/round_01 等目录。
# 可用 --model-path 指定 Selector 模型，或用 PYTHON_BIN 指定 Python 解释器。
set -euo pipefail

if (( $# < 2 )); then
    echo '用法：bash run_tree_only.sh DATASET_ROOT OUTPUT_DIR [Python 参数...]' >&2
    exit 2
fi
dataset_root=$1
output_dir=$2
shift 2
script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
exec "${PYTHON_BIN:-python}" "$script_dir/run_tree_agent_evaluation.py" \
    --dataset-root "$dataset_root" --output-dir "$output_dir" --import-top-level-sop \
    "$@" --mode tree --selection-source selector
