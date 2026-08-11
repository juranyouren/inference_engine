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

## 两个接口各测试一个 case

下面的入口会读取两个 JSON 文件，并分别以 `[json_dict]` 调用两个接口。两个模式
使用串行的 spawn 子进程：Competition 进程完全退出后才启动 Cooperation，避免
两个模型同时驻留在 NPU 内存中。

```bash
python -m llm_interfaces.main \
  --model-path /path/to/model \
  --competition-json /path/to/competition_label.json \
  --cooperation-json /path/to/cooperation_label.json \
  --competition-alarm-type BGP邻接状态改变 \
  --cooperation-alarm-type BGP邻接状态改变 \
  --output-dir /path/to/test_output
```

如果 JSON 本身包含 `alarm_type`，对应的 `--*-alarm-type` 参数可以省略。也可以用
`--mode competition` 或 `--mode cooperation` 单独测试一种接口。
