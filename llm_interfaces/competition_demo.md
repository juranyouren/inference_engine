# 竞争型 Agent 如何协作

`case + SOP 模板 → 各候选根因独立诊断 → Verifier 复核排序`

下面使用仓库内的[网络设备掉线模板](sop_templates/network_device_offline.templates.json)和简化的示意数据。示例输出用于说明数据如何流动。

## 1. 按候选根因分发诊断任务

输入 case 来自 label 文件。运行时从 `semantic_labels.sop` 读取 SOP，按每个候选根因的可达路径创建一个 Agent。网络设备掉线有五个候选，因此生成五个独立请求。每个请求使用相同的原始证据，只附上自己的决策节点、路径和结论模板。

以下是其中一个请求的核心字段。该根因需要 `step_01` 和 `step_02` 均为真。

```json
{
  "root_cause": "根因1-本端-服务器未运行",
  "case": {
    "alarm_type": "网络设备掉线",
    "alarm_time": 1700000000000,
    "semantic_labels": {
      "other_info": {
        "device_info": {
          "local_device": {"device_level": "L2"},
          "local_server_status": {}
        },
        "command_echo_info": {
          "sys_running_time": {},
          "other_command_echo": [{"cmd":"reboot","echo":{"device":"本端","time":1699999990000}}]
        }
      }
    }
  },
  "paths": [[
    {"decision_id":"step_01","required_condition_met":true},
    {"decision_id":"step_02","required_condition_met":true}
  ]]
}
```

## 2. 各 Agent 填写决策与结论

该 Agent 根据 `device_level` 判断 `step_01` 不成立。`step_02` 缺少服务器状态，填 `null`。由于唯一候选路径已有一个明确不匹配的条件，`supported` 填 `false`。

```json
{
  "decisions": {
    "step_01": {
      "diagnostic_clues": [{"path":"$.semantic_labels.other_info.device_info.local_device.device_level","value":"L2"}],
      "condition_met": false,
      "confidence": 0.95,
      "reason": "示意数据明确标记设备分区为 L2。"
    },
    "step_02": {
      "diagnostic_clues": [],
      "condition_met": null,
      "confidence": null,
      "reason": "服务器状态为空。"
    }
  },
  "conclusion": {
    "supported": false,
    "confidence": 0.95,
    "reason": "唯一候选路径要求设备属于 L1，实际为 L2。"
  }
}
```

## 3. Verifier 复核全部路径

Verifier 收到完整原始语义标签、决策条件、五份已填诊断以及各自的路径。它用原始证据独立复核判断。在这个示例中，评估根因 3 的 Agent 引用告警前 10 秒的本端 `reboot` 记录，Verifier 据此将变更重启排在首位。

```json
[
  "根因3-本端-设备重启（变更引起）",
  "根因4-本端-设备丢包率=100%",
  "根因5-本端-设备丢包率<100%",
  "根因2-本端-设备未掉线",
  "根因1-本端-服务器未运行"
]
```

每个 case 的 `result.json` 保存五份诊断和最终排序。`reasoners/` 与 `verifier/` 保存各自的 Prompt 和原始模型输出。实际 SOP 的分支及待审阅项见[预处理说明](README.md#竞争型预处理审阅稿)。
