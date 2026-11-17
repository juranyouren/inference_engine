# 合作型 Agent 如何协作

`Orchestrator → 三位专家初评 → 共享意见池 → 专家复评 → Aggregator`

下面是简化示例。所有 Agent 共用同一个模型实例。

## 1. Orchestrator 为当前告警设计三位专家

Orchestrator 收到告警类别及其描述、全部候选根因及其简要描述，以及按当前样本 `semantic_labels` 生成的字段目录。告警描述可通过 label 或 `semantic_labels` 的 `incident_type_description` 或 `alarm_type_description` 提供。路径支持递归到对象和数组内的叶子字段。`available` 标记字段是否有数据。以下省略了目录中的其他字段。

```json
{
  "incident_type": "BGP邻接状态改变",
  "incident_type_description": "BGP 对等会话的邻接状态发生变化，需区分链路、接口、对端设备和协议异常。",
  "root_cause_candidates": [
    {"category":"链路瞬断","description":"链路短时中断后恢复"},
    {"category":"配置错误","description":"配置变化导致邻接异常"}
  ],
  "semantic_labels_template": {
    "log": [{"key":"/anomaly_logs","desc":"告警窗口内的日志","available":true}],
    "time_series": [{"key":"/anomaly_kpi/offline_loss_rate/abnormal_timing_segments/*/value/*","desc":"丢包率片段的单个采样值","available":true}],
    "metadata": [{"key":"/other_info/command_echo_info/BGP_protocol_status","desc":"BGP 会话状态","available":true}]
  }
}
```

Orchestrator 为每类遥测输出一个 `persona` 及其关注点。`key` 是相对于 `semantic_labels` 的字段路径，`*` 匹配数组各项，`desc` 说明具体检查什么。程序按这些路径提取原始值，保留数组顺序，再和 persona 一起发给对应专家。

```json
{
  "log": {"persona":"邻接事件专家","focus":[{"key":"/anomaly_logs","desc":"比对接口 DOWN 和 UP 事件与告警的时间关系"}]},
  "time_series": {"persona":"丢包时序专家","focus":[{"key":"/anomaly_kpi/offline_loss_rate/abnormal_timing_segments/*/value/*","desc":"核对采样值是否突升后回落"}]},
  "metadata": {"persona":"BGP 状态专家","focus":[{"key":"/other_info/command_echo_info/BGP_protocol_status","desc":"核对邻接是否已经恢复"}]}
}
```

## 2. 三位专家独立初评

三份初评进入意见池。每位专家输出 `reasoning_chain` 和 `root_cause`，程序补充角色和轮次。此处的 `root_cause` 是专家给出的预测。

实际样本可能只有部分遥测。所选字段没有可用数据的专家会被跳过，其他专家可在初评和复评中返回 `NO COMMENT`。Aggregator 使用已有意见，并收到缺失遥测或未形成意见的专家名单。若没有任何可用意见，`result.json` 会记录 `insufficient_evidence`。

```json
[
  {"agent":"log","round":0,"reasoning_chain":["告警前后接口先 DOWN 后 UP，支持短时中断"],"root_cause":"链路瞬断"},
  {"agent":"time_series","round":0,"reasoning_chain":["丢包率突升后回落，异常未持续"],"root_cause":"链路瞬断"},
  {"agent":"metadata","round":0,"reasoning_chain":["当前 BGP 已建立，说明会话已恢复"],"root_cause":"链路瞬断"}
]
```

## 3. 专家复评共享意见池

同轮的专家收到同一份意见池快照，以及自己的 persona 和原始证据。例如时序专家看到日志专家的时间点后提交一条完整的更新意见。

```json
{
  "reasoning_chain": ["丢包尖峰与日志中的接口中断相符，两种证据共同支持短时链路异常"],
  "root_cause": "链路瞬断"
}
```

同轮结束后，时序专家的新意见替换自己的上一条意见。返回 `NO COMMENT` 的专家保留原意见。因此池中仍是三条，每位专家一条。如果首轮复评三位专家都返回 `NO COMMENT`，直接汇总，总计 8 次调用。如果首轮有人提交新意见，默认 10 次上限下还可再复评两位专家，并预留最后一次给 Aggregator。

## 4. Aggregator 汇总

输入是最终意见池、告警上下文和候选根因。假设候选为「链路瞬断」和「配置错误」，Aggregator 可以给出以下排序。

```json
{
  "ranking": ["链路瞬断", "配置错误"],
  "reason": "接口 DOWN/UP 与短时丢包相互印证。BGP 随后恢复，现有意见未提供配置错误的证据。"
}
```

每个 case 的 `result.json` 记录三位专家的 persona、调用次数、停止原因、最终意见池和排序。历轮原始响应保存在各自的 `raw_responses.txt`，实际输入保存在 `prompt.txt`。
