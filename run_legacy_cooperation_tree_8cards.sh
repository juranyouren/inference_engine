#!/usr/bin/env bash
# 在仓库根目录执行：
#   bash run_legacy_cooperation_tree_8cards.sh --check-only
#   nohup bash run_legacy_cooperation_tree_8cards.sh > legacy_launch.log 2>&1 &
# 正式运行会再次预检，通过后等待 10800 秒，再用四组双卡运行 Cooperation 和 Tree。
# 数据根目录默认 /home/sbp/deployment/case_pool/RCA_Units；支持直接含类别目录，
# 或分别含 AgentDigest_label 和 anomalydetect_label。可用 COOP_ROOT / TREE_ROOT 指定实际子目录。
# 可设置 DATASET_ROOT、TREE_TEMPLATE_DIR、MODEL_PATH、OUTPUT_DIR、PYTHON_BIN。
# 输出为 OUTPUT_DIR/<类别>/<模式>/，每个进程的日志为 OUTPUT_DIR/<类别>/<模式>.log。
set -euo pipefail
if (( $# > 1 )) || [[ -n ${1:-} && $1 != --check-only ]]; then
    echo '用法：bash run_legacy_cooperation_tree_8cards.sh [--check-only]' >&2
    exit 2
fi
repo=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
data=${DATASET_ROOT:-/home/sbp/deployment/case_pool/RCA_Units}
coop=${COOP_ROOT:-$data}
tree=${TREE_ROOT:-$data}
if [[ -z ${COOP_ROOT:-} && -d "$data/AgentDigest_label" ]]; then coop="$data/AgentDigest_label"; fi
if [[ -z ${TREE_ROOT:-} && -d "$data/anomalydetect_label" ]]; then tree="$data/anomalydetect_label"; fi
python_bin=${PYTHON_BIN:-python}
model=${MODEL_PATH:-/usr/share/large_language_models/DeepSeek-R1-Distill-Qwen-32B}
templates=${TREE_TEMPLATE_DIR:-/home/sbp/huangzeshun/RCA_Units/template}
coop_template="$repo/llm_interfaces/semantic_labels_template.json"
out=${OUTPUT_DIR:-res/legacy_cooperation_tree_8cards}
mkdir -p -- "$out"
manifest=$(mktemp "$out/.categories.XXXXXX")
pids=()
cleanup() {
    for pid in "${pids[@]}"; do kill "$pid" 2>/dev/null || true; done
    wait || true
    rm -f -- "$manifest"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
export VLLM_WORKER_MULTIPROC_METHOD=spawn
printf '预检使用 %s\nCooperation 数据 %s\nTree 数据 %s\nCooperation 模板 %s\nTree 模板 %s\n模型 %s\n' \
    "$python_bin" "$coop" "$tree" "$coop_template" "$templates" "$model"
"$python_bin" -u "$repo/check_legacy_evaluation.py" --coop-root "$coop" --tree-root "$tree" \
    --coop-template "$coop_template" --tree-templates "$templates" --model "$model" \
    --output "$out" --manifest "$manifest"
names=()
while IFS= read -r -d '' name; do names+=("$name"); done < "$manifest"
echo "预检通过，共 ${#names[@]} 类"
if [[ ${1:-} == --check-only ]]; then exit 0; fi
printf '%s 开始等待 3 小时\n' "$(date)"
sleep 10800 &
pids=("$!")
wait "${pids[0]}"
pids=()
pairs=(0,1 2,3 4,5 6,7)
status=0
for mode in cooperation tree; do
    for i in "${!names[@]}"; do
        name=${names[i]}
        dest="$out/$name"
        root=$coop
        options=(--cooperation-template "$coop_template")
        if [[ $mode == tree ]]; then
            root=$tree
            options=(--tree-template-dir "$templates" --tree-block-size 50)
        fi
        printf '%s %s %s 卡 %s\n' "$(date)" "$mode" "$name" "${pairs[i]}"
        ASCEND_RT_VISIBLE_DEVICES="${pairs[i]}" "$python_bin" -u "$repo/run_tree_agent_evaluation.py" \
            --mode "$mode" --dataset-root "$root/$name" --output-dir "$dest" \
            --ascend-devices "${pairs[i]}" --model-path "$model" "${options[@]}" > "$dest/$mode.log" 2>&1 &
        pids[i]=$!
    done
    for i in "${!pids[@]}"; do
        if ! wait "${pids[i]}"; then
            echo "失败：$out/${names[i]}/$mode.log" >&2
            status=1
        fi
        unset 'pids[i]'
    done
done
exit "$status"
