# 合作型 Agent 如何协作

`Orchestrator → 三位专家初评 → 共享意见池 → 专家复评 → Aggregator`

下面是简化示例。每个阶段的输入和输出都使用同一个模型实例生成。

## 1. Orchestrator 把 case 分给三位专家

输入是告警类别、告警时间和 `semantic_labels`。候选根因另行传给专家和 Aggregator。

```json
{
  "alarm_type": "BGP邻接状态改变",
  "alarm_time": 100,
  "semantic_labels": {
    "anomaly_logs": ["t=99 接口 DOWN", "t=101 接口 UP"],
    "anomaly_kpi": {"offline_loss_rate": [0, 100, 0]},
    "other_info": {"BGP_protocol_status": "Established"}
  }
}
```

Orchestrator 输出字段分组。程序从 case 取原值并附上各自的角色 Prompt。

```json
{
  "log_data": ["anomaly_logs"],
  "time_series_data": ["anomaly_kpi"],
  "metadata": ["other_info"]
}
```

## 2. 三位专家独立初评

三份初评进入意见池。每位专家输出 `reasoning_chain` 和 `root_cause`，程序补充角色和轮次。

```json
[
  {"agent":"log","round":0,"reasoning_chain":["告警前后接口先 DOWN 后 UP，支持短时中断"],"root_cause":"链路瞬断"},
  {"agent":"time_series","round":0,"reasoning_chain":["丢包率突升后回落，异常未持续"],"root_cause":"链路瞬断"},
  {"agent":"metadata","round":0,"reasoning_chain":["当前 BGP 已建立，说明会话已恢复"],"root_cause":"链路瞬断"}
]
```

## 3. 专家复评共享意见池

同轮的专家收到同一份意见池快照，以及自己分到的原始证据。例如时序专家看到日志专家的时间点后提交一条新意见。

```json
{
  "reasoning_chain": ["丢包尖峰与日志中的接口中断相符，两种证据共同支持短时链路异常"],
  "root_cause": "链路瞬断"
}
```

日志和元数据专家可以各自返回 `NO COMMENT`。同轮结束后只把新意见加入池。如果首轮复评三位专家都返回 `NO COMMENT`，直接汇总，总计 8 次调用。如果首轮有人提交新意见，默认 10 次上限下还可再复评两位专家，并预留最后一次给 Aggregator。

## 4. Aggregator 汇总

输入是最终意见池、告警上下文和候选根因。假设候选为「链路瞬断」和「配置错误」，Aggregator 可以给出以下排序。

```json
{
  "ranking": ["链路瞬断", "配置错误"],
  "reason": "接口 DOWN/UP 与短时丢包相互印证。BGP 随后恢复，现有意见未提供配置错误的证据。"
}
```

每个 case 的 `result.json` 记录调用次数、停止原因、最终意见池和排序。各阶段的 `prompt.txt` 与 `raw_responses.txt` 可用于核对专家实际收到的内容。
