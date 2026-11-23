#!/usr/bin/env bash
# 在仓库根目录运行：bash run_rca_labels_all.sh
# 后台运行：nohup bash run_rca_labels_all.sh > rca_labels_all.log 2>&1 &
# 可选：bash run_rca_labels_all.sh 输出目录
# 使用 run_agent_evaluation_parallel.sh 的模板与批处理入口。读取 RCA_Labels 的全部类别，
# cooperation 跑全部 case，competition 跑其中有 SOP 的 case。8 张卡，每两张一组。
# 四个分片并行。各分片先跑网络设备掉线的 competition 和 cooperation，再跑其他类别。
# 每个分片的日志位于输出目录下的 shard_N/run.log，汇总结果见 metrics.csv。
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
if (( $# > 1 )); then
    echo "用法：bash run_rca_labels_all.sh [输出目录]" >&2
    exit 2
fi
exec bash run_agent_evaluation_parallel.sh \
    --label-root /home/sbp/deployment/case_pool/RCA_Labels \
    --first-category "网络设备掉线" \
    --mode both --ascend-devices 0,1,2,3,4,5,6,7 \
    --output-dir "${1:-res/rca_labels_all}"
