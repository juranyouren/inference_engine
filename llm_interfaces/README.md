# LLM 推理接口

本目录是独立的 Python 包，项目内依赖均位于本目录。它提供两个入口类：

- `CompetitionInference`：三个 Reasoner 完成多采样推理，再由 Verifier 输出最终结果。
- `CooperationInference`：Meta 与 Reasoner 多轮协作，支持早停。

导入模块不会加载模型。实例化所选择的类时加载一次模型，同一实例后续多次调用
`infer()` 都会复用该模型。不要同时实例化两个模式；调用方根据本次任务选择其中一个。

## 输入

`infer()` 接收已经由 `json.load()` 读取的 `list[dict]`。SOP 严格从
`item["semantic_labels"]["sop"]` 读取：

- Competition 要求 SOP 非空。
- Cooperation 要求 SOP 为空。

每条 JSON 可以包含 `alarm_type`；如果原始 label 中没有该字段，请在类初始化时统一
传入 `alarm_type`。`case_id`、`id`、`index` 或 `idx` 可用于标识结果；均不存在时使用
列表下标。

## Competition 示例

```python
import json

from llm_interfaces.competition import CompetitionInference

with open("label.json", "r", encoding="utf-8") as file:
    label = json.load(file)

engine = CompetitionInference(
    model_path="/path/to/model",
    alarm_type="BGP邻接状态改变",
    batch_size=8,
)
results = engine.infer([label], "/path/to/output")
```

## Cooperation 示例

```python
from llm_interfaces.cooperation import CooperationInference

engine = CooperationInference(
    model_path="/path/to/model",
    alarm_type="BGP邻接状态改变",
    batch_size=8,
    max_rounds=5,
)
results = engine.infer(label_list, "/path/to/output")
```

每个 case 的 prompt、原始响应和 `result.json` 会保存到
`output/case_XXXXXX/`；整个批次的汇总保存在 `output/results.json`。
