#!/usr/bin/env bash
# 用法（在仓库根目录运行）：
#   bash run_competition_8cards.sh
#   bash run_competition_8cards.sh --batch-size 16 --model-path /path/to/model
# 默认数据集：/home/sbp/deploy/case_pool/anomalydetect_label_0
# 可用 DATASET_ROOT、OUTPUT_DIR、PYTHON_BIN 覆盖路径。默认输出为 res/competition_8cards。
# 自动发现四个告警类别，按目录名排序，依次固定使用 0,1 / 2,3 / 4,5 / 6,7。
# 仅运行 Competition；沿用每类前 50 条留作训练、随后 150 条作为测试窗口的口径。
# 顶层 SOP 自动导入；每类日志在 OUTPUT_DIR/<类别>/run.log，指标在其 competition/metrics.csv。
set -euo pipefail

repo_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
dataset_root=${DATASET_ROOT:-/home/sbp/deploy/case_pool/anomalydetect_label_0}
output_dir=${OUTPUT_DIR:-res/competition_8cards}
python_bin=${PYTHON_BIN:-python}

# 与评测入口用同一规则发现有 label 的目录，忽略其他子目录。
categories=()
while IFS= read -r -d '' name; do
    categories+=("$name")
done < <("$python_bin" - "$repo_dir" "$dataset_root" <<'PY'
import sys
from pathlib import Path

sys.path.insert(0, sys.argv[1])
from inference.common import discover_label_index_map

for folder in sorted(Path(sys.argv[2]).iterdir()):
    if folder.is_dir() and discover_label_index_map(str(folder), folder.name):
        sys.stdout.buffer.write(folder.name.encode("utf-8") + b"\0")
PY
)
if (( ${#categories[@]} != 4 )); then
    printf '在 %s 中发现 %s 类告警，固定八卡入口需要四类。\n' "$dataset_root" "${#categories[@]}" >&2
    exit 2
fi

pairs=(0,1 2,3 4,5 6,7)
pids=()
cleanup() {
    for pid in "${pids[@]}"; do
        kill "$pid" 2>/dev/null || true
    done
    wait || true
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

for i in "${!categories[@]}"; do
    name=${categories[i]}
    pair=${pairs[i]}
    dest="$output_dir/$name"
    mkdir -p -- "$dest"
    printf 'Competition  %s  卡 %s  日志 %s/run.log\n' "$name" "$pair" "$dest"
    ASCEND_RT_VISIBLE_DEVICES="$pair" VLLM_WORKER_MULTIPROC_METHOD=spawn \
        "$python_bin" -u "$repo_dir/run_tree_agent_evaluation.py" "$@" \
        --dataset-root "$dataset_root/$name" --output-dir "$dest" \
        --mode competition --import-top-level-sop --ascend-devices "$pair" \
        > "$dest/run.log" 2>&1 &
    pids[i]=$!
done

status=0
for i in "${!pids[@]}"; do
    if ! wait "${pids[i]}"; then
        printf '%s 运行失败，详见 %s/%s/run.log\n' "${categories[i]}" "$output_dir" "${categories[i]}" >&2
        status=1
    fi
    unset 'pids[i]'
done
exit "$status"
