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
└── 告警类别/
    ├── 告警类别_0_label_x.json
    └── 告警类别_1_label_x.json
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

输出位于：

```text
bian_output/llm_infer_bian_stage1/告警类别/idx_0_0/
├── predictions.json
├── predictions.jsonl
├── predictions.csv
├── meta.json
└── 0/
    ├── alert_summary/
    ├── anomaly_analysis/
    ├── joint_scoring/
    └── prediction.json
```

