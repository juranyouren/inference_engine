# LLM 推理接口

本目录是独立的 Python 包，项目内依赖均位于本目录。它提供两个入口类：

- `CompetitionInference`：三个 Reasoner 完成多采样推理，再由 Verifier 输出最终结果。
- `CooperationInference`：Meta 与 Reasoner 多轮协作，支持早停。

模型由 `LLMEngine.get_instance()` 第一次调用时加载。Competition、Cooperation、
Selector 和 Refiner 都可以接收同一个 engine 实例，整个 main 进程只加载一次模型。

## 输入

`infer()` 接收已经由 `json.load()` 读取的 `list[dict]`。SOP 严格从
`item["semantic_labels"]["sop"]` 读取：

- Competition 要求 SOP 非空。
- Cooperation 要求 SOP 为空。

告警类别作为 `infer(input_list, alarm_type, output_dir)` 的第二个参数显式传入，JSON
中的 `alarm_type` 不参与本次推理类别选择。`case_idx` 直接使用输入列表下标。

## Competition 示例

```python
import json

from llm_interfaces.competition import CompetitionInference
from llm_interfaces.llm_engine import LLMEngine

with open("label.json", "r", encoding="utf-8") as file:
    label = json.load(file)

engine = LLMEngine.get_instance(
    model_path="/path/to/model",
    gpu_memory_utilization=0.9,
    max_model_len=16384,
)
competition = CompetitionInference(engine)
results = competition.infer(
    [label],
    "BGP邻接状态改变",
    "/path/to/output",
)
```

## Cooperation 示例

```python
from llm_interfaces.cooperation import CooperationInference

cooperation = CooperationInference(engine)
results = cooperation.infer(
    label_list,
    "BGP邻接状态改变",
    "/path/to/output",
)
```

两个 `infer()` 都返回 `list[dict]`，每个字典只包含与 CSV 一致的九个字段，必有
`predicted_list` 和 `rank`。

每个 case 的 prompt、原始响应和 `result.json` 会保存到
`output/<mode>/<category>/case_XXXXXX/`；模式级汇总保存在
`output/<mode>/results.json`。

每个类别还会生成 `output/<mode>/<category>/results.csv`，列顺序为：

```text
case_idx,timestamp,incident_type,predicted_list,groundtruth,rank,ttm,predicted_list_v1,rank_v1
```

其中 `case_idx` 直接使用输入列表下标；`predicted_list` 是完整模型排序的前 5 项，
`predicted_list_v1` 是完整排序。`groundtruth` 读取 `root_cause.category`，`ttm`
读取输入 JSON 的同名字段，缺失时留空。

## 两个接口各测试一个 case

下面的入口会读取两个 JSON 文件，并分别以 `[json_dict]` 调用两个接口。两个模式
直接在 main 进程中串行执行，不创建子进程。main 先调用一次
`LLMEngine.get_instance()`，再把同一个 engine 传给两个推理类。

```bash
python -m llm_interfaces.main \
  --model-path /path/to/model \
  --competition-json /path/to/competition_label.json \
  --cooperation-json /path/to/cooperation_label.json \
  --competition-alarm-type BGP邻接状态改变 \
  --cooperation-alarm-type BGP邻接状态改变 \
  --ascend-devices 0,1 \
  --output-dir res
```

`--competition-alarm-type` 和 `--cooperation-alarm-type` 是显式告警类别，测试对应
模式时必须提供。也可以用 `--mode competition` 或 `--mode cooperation` 单独测试。

`--ascend-devices 0,1` 会在模型初始化前设置
`ASCEND_RT_VISIBLE_DEVICES=0,1`，并自动使用 `tensor_parallel_size=2`。指定四张卡时可传
`--ascend-devices 0,1,2,3`，`LLMEngine` 会根据环境变量中的卡号数量将张量并行度设为 4。

`--output-dir` 默认就是 `res`，可以省略。最终 CSV 路径为
`res/<mode>/<category>/results.csv`。

两个推理类只是共享同一个 `LLMEngine`，不会再次初始化或加载模型。
