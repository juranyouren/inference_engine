# BiAn 第一阶段独立复现

本目录独立实现论文 BiAn 的第一阶段，不接入原有 `infer_by_index.py`、`inference/llm.py` 或 `llm_inference/generator.py`。

流程为：

```text
告警摘要 → 异常分析 → 联合评分
```

当前输入数据是项目现有的根因候选格式，因此第一阶段对 `root_cause_candidates` 排序；论文原始方法中的候选设备字段后续可以替换进同一接口。

## 输入目录

```text
input_root/
├── 告警类别A/
│   ├── 告警类别A_0_label_x.json
│   └── 告警类别A_1_label_x.json
└── 告警类别B/
    └── 告警类别B_0_label_x.json
```

输入 JSON 至少包含：

```json
{
  "semantic_labels": {
    "anomaly_logs": {},
    "anomaly_kpi": {},
    "other_info": {}
  },
  "sop": [],
  "root_cause_candidates": [
    {"category": "根因A", "description": "..."}
  ],
  "root_cause": {"category": "根因A"},
  "alarm_time": 0
}
```

## 运行

```bash
python -m bian_reproduction.run_stage1 \
  --input-root /path/to/input_root \
  --alarm-type "告警类别" \
  --output-dir /path/to/bian_output \
  --start 0 \
  --end 1
```

一次传入四种类别（同一个模型只加载一次）：

```bash
python -m bian_reproduction.run_stage1 \
  --input-root /path/to/input_root \
  --alarm-types "BGP邻接状态改变" "网络设备掉线" \
                "接口状态DOWN" "入方向带宽利用率过高" \
  --output-dir /path/to/bian_output \
  --start 0 \
  --end 50
```

`--start/--end` 会分别应用于每个类别，区间为 `[start, end)`。类别参数没有硬编码，也可以传入一个、两个或更多类别；`--alarm-type` 和 `--alarm-types` 两种写法等价。

输出位于：

```text
bian_output/llm_infer_bian_stage1/告警类别/idx_0_0/
├── predictions.json
├── predictions.jsonl
├── predictions.csv
├── meta.json
├── evaluation/
│   ├── metrics.json
│   └── metrics.csv
└── 0/
    ├── alert_summary/
    ├── anomaly_analysis/
    ├── joint_scoring/
    └── prediction.json
```

多类别运行还会生成跨类别汇总：

```text
bian_output/llm_infer_bian_stage1/evaluation/
├── metrics.json
└── metrics.csv
```

运行第一阶段后会自动生成评价结果。也可以对已有结果单独评价：

```bash
python -m bian_reproduction.evaluate \
  --predictions /path/to/idx_0_49/predictions.json
```

汇总多个运行结果：

```bash
python -m bian_reproduction.evaluate \
  --predictions /path/to/run_a /path/to/run_b \
  --output-dir /path/to/evaluation
```

评价指标包括 Top-1、Top-3、Top-5、MRR、三个推理阶段耗时、端到端总耗时和平均每 case 耗时：

```text
Top@K = 真实根因排名不超过 K 的 case 数 / 有标签 case 总数
MRR   = sum(1 / 真实根因排名；未命中为 0) / 有标签 case 总数
Time  = elapsed_seconds / evaluated_count
```

模型加载时间单独记录，不计入端到端平均推理时间。无法解析、排序不完整、候选顺序回退以及推理跳过的 case 均按未命中计算。
