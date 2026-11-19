#!/usr/bin/env bash
# 用法：bash run_tree_only.sh DATASET_ROOT OUTPUT_DIR [Python 参数...]
# 示例（在仓库根目录运行）：
#   bash run_tree_only.sh /home/sbp/deploy/case_pool/anomalydetect_label_0 \
#     res/tree_rerun --ascend-devices 6,7
# 每类按数字索引取前 50 条训练、随后 150 条测试。默认导入顶层 SOP。
# 根据训练集重新生成 selection，只评测原始 Tree 排名。结果在 OUTPUT_DIR/tree/。
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
