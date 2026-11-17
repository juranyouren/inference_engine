#!/usr/bin/env bash
# 在仓库根目录运行。默认读取 /home/sbp/deploy/case_pool/anomalydetect_label_0 的顶层 SOP。
# bash run_agent_evaluation_parallel.sh --limit-per-mode 10 --output-dir res/agent_evaluation_parallel
# 默认卡号为 0,1,2,3,4,5,6,7，可用 --ascend-devices 或 ASCEND_RT_VISIBLE_DEVICES 指定卡号。
# 每两张卡运行一个分片，单出的卡空闲。每个分片的日志和结果在 shard_N 目录，汇总在输出根目录。
# 支持原评测入口的参数，可用 --label-root 指定其他数据集路径。
# 每种模式只跑前 N 个符合条件的 case 时，加 --limit-per-mode N。各分片共享这个上限。
# 如需指定 Python 解释器，可设置 RCA_PYTHON。

set -euo pipefail

repo_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
cd -- "$repo_dir"
python_bin=${RCA_PYTHON:-python}
devices=${ASCEND_RT_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}
output_dir=res/agent_evaluation_parallel
evaluation_args=(--label-root /home/sbp/deploy/case_pool/anomalydetect_label_0 --import-top-level-sop)

fail() {
    printf '%s\n' "$1" >&2
    exit 2
}

while (($#)); do
    case "$1" in
        -h|--help)
            printf '%s\n' '用法  bash run_agent_evaluation_parallel.sh --ascend-devices 0,1,2,3 --label-root /path/to/label_root'
            exec "$python_bin" run_agent_evaluation.py --help
            ;;
        --ascend-devices)
            (($# >= 2)) || fail '--ascend-devices 需要逗号分隔的卡号'
            devices=$2
            shift 2
            ;;
        --ascend-devices=*)
            devices=${1#*=}
            shift
            ;;
        --output-dir)
            (($# >= 2)) || fail '--output-dir 需要输出目录'
            output_dir=$2
            shift 2
            ;;
        --output-dir=*)
            output_dir=${1#*=}
            shift
            ;;
        --shard-count|--shard-count=*|--shard-index|--shard-index=*)
            fail '并行入口根据卡号自动设置分片参数'
            ;;
        *)
            evaluation_args+=("$1")
            shift
            ;;
    esac
done

devices=${devices%$'\r'}
[[ "$devices" != *, ]] || fail '卡号列表末尾多了一个逗号'
IFS=, read -r -a cards <<< "$devices"
declare -A used_cards=()
for index in "${!cards[@]}"; do
    [[ "${cards[index]}" =~ ^[[:space:]]*(0|[1-9][0-9]*)[[:space:]]*$ ]] || fail '请提供有效的数字卡号'
    card=${BASH_REMATCH[1]}
    [[ -z ${used_cards[$card]:-} ]] || fail "卡号 $card 重复"
    used_cards[$card]=1
    cards[index]=$card
done

shard_count=$((${#cards[@]} / 2))
((shard_count >= 1)) || fail '每个分片至少需要两张卡'
if ((${#cards[@]} % 2)); then
    printf '卡 %s 空闲\n' "${cards[shard_count * 2]}"
fi

mkdir -p -- "$output_dir"
shopt -s nullglob
existing_shards=("$output_dir"/shard_*)
if ((${#existing_shards[@]})) || [[ -e "$output_dir/metrics.json" || -e "$output_dir/metrics.csv" ]]; then
    fail "输出目录已有评测结果，请指定新目录  $output_dir"
fi

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

started=$SECONDS
for ((shard = 0; shard < shard_count; shard++)); do
    pair="${cards[shard * 2]},${cards[shard * 2 + 1]}"
    shard_dir="$output_dir/shard_$shard"
    mkdir -- "$shard_dir"
    printf '分片 %s 使用卡 %s，日志写入 %s/run.log\n' "$shard" "$pair" "$shard_dir"
    "$python_bin" -u run_agent_evaluation.py "${evaluation_args[@]}" \
        --ascend-devices "$pair" --output-dir "$shard_dir" \
        --shard-count "$shard_count" --shard-index "$shard" > "$shard_dir/run.log" 2>&1 &
    pids[shard]=$!
done

failed=0
for shard in "${!pids[@]}"; do
    if ! wait "${pids[shard]}"; then
        printf '分片 %s 失败，见 %s/shard_%s/run.log\n' "$shard" "$output_dir" "$shard" >&2
        failed=1
    fi
    unset "pids[$shard]"
done

"$python_bin" -m llm_interfaces.merge_agent_evaluation --output-dir "$output_dir" \
    --shard-count "$shard_count" --wall-seconds "$((SECONDS - started))"
exit "$failed"
