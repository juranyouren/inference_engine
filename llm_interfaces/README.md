# LLM 推理接口

本目录是独立的 Python 包，项目内依赖均位于本目录。它提供两个入口类：

- `CompetitionInference` 为每个候选根因独立填写 SOP 路径，Verifier 用原始证据复核并排序。[竞争型演示](competition_demo.md)
- `CooperationInference` 由 Orchestrator 分发数据，三位专家初评并复评共享意见，Aggregator 汇总。[合作型演示](cooperation_demo.md)

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
`predicted_list_v1` 是完整排序。`groundtruth` 优先读取 `semantic_labels.root_cause`，兼容顶层 `root_cause`。真实根因只用于评分，模型仍可使用显式提供的候选根因。`ttm` 读取输入 JSON 的同名字段，缺失时留空。

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

## 两个模式的评测

在仓库根目录运行以下命令，替换 `--label-root` 后的数据集路径即可做配对评测。

```bash
python run_agent_evaluation.py --label-root /home/sbp/deployment/case_pool/AgentDigest_label --output-dir res/agent_evaluation
```

数据集根目录中的类别会动态发现。`--mode` 可选 `both`、`competition` 或 `cooperation`。`both` 只选择有 SOP 的 case，对同一 case 分别运行两种模式。合作型使用隐藏 SOP 的副本。若数据集的 SOP 位于 JSON 顶层，可加 `--import-top-level-sop`，在本次评测的内存副本中导入。

结果目录包含 `metrics.json`、`metrics.csv` 和 `traces/`。指标包括 Top-1/3/5、MRR、耗时及失败数，模型推理失败的有标注样本按未命中计分。

## 竞争型预处理审阅稿

[网络设备掉线 SOP](sop_templates/network_device_offline.sop.json)定义了入口、6 个决策节点、证据字段和 5 个根因。可编辑这份结构化 SOP，然后在仓库根目录重新生成模板。

```bash
python -m llm_interfaces.sop_preprocess llm_interfaces/sop_templates/network_device_offline.sop.json llm_interfaces/sop_templates/network_device_offline.templates.json
```

[生成的模板](sop_templates/network_device_offline.templates.json)包含共享的 `decision_templates`，以及 `root_cause_prompts` 下每个根因的名称、提示词、所有到达路径和结论模板。模型返回的 `decisions` 以决策 ID 为键，填写原始证据的路径和值、条件判断、置信度和理由。路径里的 `required_condition_met` 是分支要求，节点输出里的 `condition_met` 是实际判断。证据不足填 `null`。`conclusion` 只填写 `supported`、`confidence` 和 `reason`。

来源是用户提供的第 4 条 label 的顶层 `sop`。下面按原文保留分支。Procedure 1 判定非 L1，或 Procedure 2 判定服务器状态不正常，都进入 Procedure 4。Procedure 4 未取得运行时间时进入 Procedure 6，取得运行时间且 Procedure 5 命中重启条件时也进入 Procedure 6。

| 根因 | 终段决策 | 完整路径数 |
| --- | --- | ---: |
| 服务器未运行 | Step 1 是 → Step 2 是 | 1 |
| 设备未掉线 | Step 4 是 → Step 5 否 | 2 |
| 设备重启（变更引起） | Step 6 是 | 4 |
| 设备丢包率=100% | Step 6 否 → Step 9 是 | 4 |
| 设备丢包率<100% | Step 6 否 → Step 9 否 | 4 |

Step 5 的重启阈值是严格小于 `360000ms`（6 分钟），差值方向仍待确认。Step 9 的窗口是告警前后各 1 分钟。原文有两处条件和结论冲突，已保留在审阅稿中。

- Step 2 的「服务器状态正常」走 Yes，进入「服务器未运行」。
- Step 9 的「丢包率超过 100%」走 Yes，进入「丢包率=100%」。样例采样最高为 `100%`，样例的 `script_cots` 写的是 `>=100%` 并走到该结论。

附件四条 label 的 `sop` 都在 JSON 顶层，现有运行时分流读取 `semantic_labels.sop`。字段位置差异和待核对的证据字段见 `source.review_notes`。
