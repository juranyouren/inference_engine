ROOT_CAUSE_NAIVE = '''
### 任务描述
你是一个网络运维助手，负责根据告警观测数据的语义标签参考标准操作流程 sop 给出根因排序。
- 你将收到
  1. **告警场景名 alarm_type**
  2. **语义标签 semantic_labels**（从原始告警观测数据中提取得到的语义化信息，指示了 sop 中涉及与根因有关的日志、KPI、网流、回显、设备等信息）
  3. **标准操作流程 sop**（按步骤给出了从观测数据导向最终根因的逻辑流程，遵循该流程判断根因；若该部分为空，则根据你的经验自行判断）
  4. **候选根因列表 root_cause_candidates**

### 输入
- **告警场景名**：{alarm_type}
- **语义标签**：{semantic_labels}
- **标准操作流程**：{sop}
- **候选根因列表**：{root_cause_candidates}

### 输出要求
- 输出json形式的根因排序列表，其中字符串仅能使用根因候选列表中的 category 字段内容，按照根因的可能性从高到低排序。
```json
["(在此输出根因 category)", "(在此输出根因 category)", "(在此输出根因 category)", ...]
```
'''.strip()

ROOT_CAUSE_META = '''
### 任务描述
你是一个网络运维助手，负责根据告警观测数据的语义标签参考标准操作流程 sop 给出根因排序。
- 你将收到
  1. **告警场景名 alarm_type**
  2. **语义标签 semantic_labels**（从原始告警观测数据中提取得到的语义化信息，指示了 sop 中涉及与根因有关的日志、KPI、网流、回显、设备等信息）
  3. **标准操作流程 sop**（按步骤给出了从观测数据导向最终根因的逻辑流程，遵循该流程判断根因；若该部分为空，则根据你的经验自行判断）
  4. **候选根因列表 root_cause_candidates**
  5. **告警时间 alarm_time**
- 你需要按照以下思考步骤完成任务：
  ## Step 1: 语义标签分析：逐一分析语义标签中的 anomaly_logs, anomaly_kpi, anomaly_netstreams, other_info.
  ## Step 2: 根因推理：基于语义标签的分析结果，遵循标准操作流程 sop 的逻辑判断根因。
  ## Step 3: 反思：反思上述思考过程的合理性，避免逻辑上的遗漏或错误。
  ## Step 4: 组织排序：查看候选根因列表中所有的 category 字段，并结合之前的分析按照根因可能性从高到低排序。

### 输入
- **告警场景名**：{alarm_type}
- **语义标签**：{semantic_labels}
- **告警时间**：{alarm_time}
- **标准操作流程**：{sop}
- **候选根因列表**：{root_cause_candidates}

### 输出要求
- **思考过程**：开头标出 "## Step 1: 语义标签分析"，"## Step 2: 根因推理"，"## Step 3: 反思"，"## Step 4: 组织排序"
- **排序结果**：对根因候选列表中的所有 category 字段内容进行排序，以 json 形式组织
- 严格按照如下格式，仅输出**思考过程**和**排序结果**，且两者用唯一标识 </think> 分隔：
## Step 1: 语义标签分析
[在此输出思考过程1]
## Step 2: 根因推理
[在此输出思考过程2]
## Step 3: 反思
[在此输出思考过程3]
## Step 4: 组织排序
[在此输出思考过程4]
</think>
```json
["(在此输出根因 category)", "(在此输出根因 category)", ..., "(在此输出根因 category)"]
```
'''.strip()

ROOT_CAUSE_FEW_SHOT = '''
### 任务描述
你是一个网络运维助手，负责根据告警观测数据的语义标签参考标准操作流程 sop 给出根因排序。
- 你将收到
  1. **告警场景名 alarm_type**
  2. **语义标签 semantic_labels**（从原始告警观测数据中提取得到的语义化信息，指示了 sop 中涉及与根因有关的日志、KPI、网流、回显、设备等信息）
  3. **标准操作流程 sop**（按步骤给出了从观测数据导向最终根因的逻辑流程，遵循该流程判断根因；若该部分为空，则根据你的经验自行判断）
  4. **候选根因列表 root_cause_candidates**
  5. **告警时间 alarm_time**
- 你需要按照以下思考步骤完成任务：
  ## Step 1: 语义标签分析：逐一分析语义标签中的 anomaly_logs, anomaly_kpi, anomaly_netstreams, other_info.
  ## Step 2: 根因推理：基于语义标签的分析结果，遵循标准操作流程 sop 的逻辑判断根因。
  ## Step 3: 反思：反思上述思考过程的合理性，避免逻辑上的遗漏或错误。
  ## Step 4: 组织排序：查看候选根因列表中所有的 category 字段，并结合之前的分析按照根因可能性从高到低排序。

### 输入
- **告警场景名**：{alarm_type}
- **语义标签**：{semantic_labels}
- **告警时间**：{alarm_time}
- **标准操作流程**：{sop}
- **候选根因列表**：{root_cause_candidates}

### 输出要求
- **思考过程**：开头标出 "## Step 1: 语义标签分析"，"## Step 2: 根因推理"，"## Step 3: 反思"，"## Step 4: 组织排序"
- **排序结果**：对根因候选列表中的所有 category 字段内容进行排序，以 json 形式组织
- 严格按照如下格式，仅输出**思考过程**和**排序结果**，且两者用唯一标识 </think> 分隔：
## Step 1: 语义标签分析
[在此输出思考过程1]
## Step 2: 根因推理
[在此输出思考过程2]
## Step 3: 反思
[在此输出思考过程3]
## Step 4: 组织排序
[在此输出思考过程4]
</think>
```json
["(在此输出根因 category)", "(在此输出根因 category)", ..., "(在此输出根因 category)"]
```
'''.strip()

BGP_FEW_SHOT='''
**输入**：
- 告警场景名：BGP邻接状态改变
- 语义标签：  
  "semantic_labels": {
    "anomaly_logs": {
      "local_flash": {
        "linkDown_active(l)": 7,
        "linkDown_clear(l)": 7,
        "hwPortDown(l)": 0,
        "hwPortUp(l)": 0,
        "hwBgpBackwardTransition_active(l)": 0,
        "hwBgpBackwardTransition_clear(l)": 0,
        "bgpBackwardTransition_active(l)": 10,
        "bgpBackwardTransition_clear(l)": 10,
        "PEER_STATE_CHG(l)": 11
      },
      "remote_flash": {
        "linkDown_active(l)": 2,
        "linkDown_clear(l)": 2,
        "hwPortDown(l)": 0,
        "hwPortUp(l)": 0,
        "hwBgpBackwardTransition_active(l)": 0,
        "hwBgpBackwardTransition_clear(l)": 0,
        "bgpBackwardTransition_active(l)": 2,
        "bgpBackwardTransition_clear(l)": 2,
        "PEER_STATE_CHG(l)": 3
      },
      "local_shutdown": {},
      "remote_shutdown": {},
      "local_exception": {
        "Reason=Interface physical link is down": 9,
        "Reason=BGP direct connect-interface down": 12,
        "The local fault alarm has occurred": 0,
        "The physical status of the port changed to down": 0,
        "The remote fault alarm has occurred": 2,
        "Optical Module is invalid": 0,
        "The BFD session went Down": 0,
        "hwBfdSessDown(t)": 0,
        "CurrState=ESTABLISHED": 10,
        "CMDRECORD": 0,
        "REBOOT": 0
      },
      "remote_exception": {
        "Reason=Interface physical link is down": 0,
        "Reason=BGP direct connect-interface down": 6,
        "The local fault alarm has occurred": 0,
        "The physical status of the port changed to down": 0,
        "The remote fault alarm has occurred": 0,
        "Optical Module is invalid": 0,
        "The BFD session went Down": 0,
        "hwBfdSessDown(t)": 0,
        "CurrState=ESTABLISHED": 2,
        "CMDRECORD": 0,
        "REBOOT": 0
      },
      "other_anomaly_logs": []
    },
    "anomaly_kpi": {
      "traffic_out": {
        "upper_bound": 80.0,
        "abnormal_timing_segments": [
          {
            "start_time": 1758657890,
            "end_time": 1758658070,
            "value": [
              3.5308,
              5.1452,
              5.6332,
              3.7826
            ],
            "pattern": "spike"
          }
        ]
      },
      "drop_packet_rate": {
        "upper_bound": 5.0,
        "abnormal_timing_segments": [
          {
            "start_time": 1758657890,
            "end_time": 1758658070,
            "value": [
              3.5308,
              5.1452,
              5.6332,
              3.7826
            ],
            "pattern": "spike"
          }
        ]
      },
      "error_packet_rate": {
        "upper_bound": 5.0,
        "abnormal_timing_segments": [
          {
            "start_time": 1758657890,
            "end_time": 1758658070,
            "value": [
              3.5308,
              5.1452,
              5.6332,
              3.7826
            ],
            "pattern": "spike"
          }
        ]
      }
    },
    "anomaly_netstreams": {},
    "other_info": {
      "command_echo_info": {
        "BGP_protocol_status": "Established",
        "local_interface_status": "UP",
        "remote_interface_status": "UP",
        "sys_running_time": "",
        "other_command_echo": [
          {}
        ]
      },
      "device_info": {
        "local_device": {
          "management_ip": "11.90.80.16",
          "device_name": "sabrazil1y-cnt-ce16808-csw-11.90.80.16",
          "device_level": "CE16808",
          "owner": "DCNOM(数据中心网络运维)",
          "resource_status": "Running"
        },
        "remote_device": {
          "management_ip": "11.28.128.16",
          "device_name": "sabrazil1a-cnt-ce16808-dcc-11.28.128.16",
          "device_level": "CE16808",
          "owner": "DCNOM(数据中心网络运维)",
          "resource_status": "Running"
        },
        "local_ifname": "100GE1/0/15",
        "local_server_status": {}
      }
    },
    "alarm_time": 1758657933070
  }

**输出**
嗯，我现在需要根据给定的告警数据和标准操作流程，来判断BGP邻接状态改变的根因，并按照可能性从高到低排序。首先，我得仔细分析输入的数据，然后按照SOP的步骤一步步排查。

首先，看看告警场景是BGP邻接状态改变，这可能涉及设备状态、接口状态、闪断次数等因素。语义标签里有各种日志、KPI和设备信息。让我先看看这些数据。

从语义标签的anomaly_logs部分，local_flash里有PEER_STATE_CHG(l)出现了11次，这可能表示本地设备发生了多次状态变化。同时，bgpBackwardTransition_active(l)和clear(l)各有10次，说明本地BGP状态发生了多次倒换。remote_flash部分也有类似的情况，但次数较少，比如PEER_STATE_CHG(l)只有3次，bgpBackwardTransition各2次。这可能意味着本地设备的状态变化更频繁。

在anomaly_exception部分，本地异常里有“Reason=BGP direct connect-interface down”出现了12次，这可能表明本地设备的BGP直接连接接口出现了问题。而远程异常里同样的原因出现了6次，次数较少。这可能说明问题主要出现在本地设备。

接下来，看看KPI部分，traffic_out、drop_packet_rate和error_packet_rate都有spike模式的异常，这可能意味着网络流量、丢包率和错误包率在某个时间段内急剧上升，这可能与接口状态变化有关。

设备信息显示，本地和远程设备的管理状态都是Running，所以根因1-对端-未运行可能不适用，因为remote_device的resource_status是Running。因此，SOP中的Procedure 2可能不会被触发。

然后，SOP的下一步是检查remote_device_level是否包含Vlanif。设备信息中，remote_device_level是CE16808，没有提到Vlanif，所以Procedure 5会被执行。接下来，检查本端接口状态是否包含ERROR DOWN。从other_info中的local_interface_status是UP，所以Procedure 8不会被触发，转到Procedure 9。

在Procedure 9，检查本端接口状态是否包含Administratively DOWN。根据日志，本地接口状态是UP，所以Procedure 10不会被触发，转到Procedure 11。检查对端接口状态是否包含Administratively DOWN，同样，remote_interface_status是UP，所以Procedure 12也不适用，转到Procedure 13。

Procedure 13检查闪断次数是否大于等于3。从语义标签中的local_flash部分，PEER_STATE_CHG(l)有11次，这可能意味着闪断次数较多，所以Procedure 14会被触发，诊断为根因7-本端-频繁闪断。但同时，Procedure 15检查闪断次数是否大于等于0且本端接口状态是UP，这显然满足，所以进入Procedure 16。

在Procedure 16，检查BGP状态是否为Established。从other_info中的BGP_protocol_status是Established，所以Procedure 18会被触发，诊断为根因8-本端-闪断，即轻微闪断，属正常波动。但同时，由于闪断次数较多，根因7的可能性也存在。

此外，考虑到本地异常日志中“Reason=BGP direct connect-interface down”出现次数较多，这可能表明本地接口存在问题，导致BGP状态频繁变化，这可能指向根因7或根因8。同时，KPI的异常可能进一步支持频繁闪断的情况。

综合来看，根因7-本端-频繁闪断的可能性较高，因为闪断次数多，且BGP状态已恢复。根因8-本端-闪断可能次之，因为虽然有闪断，但次数较多，可能更倾向于频繁闪断。此外，根因2-对端-服务器闪断和根因3-对端-服务器BGP未恢复的可能性较低，因为对端的异常日志次数较少，且设备状态正常。

另外，根因6-本端-errordown可能不适用，因为接口状态是UP，没有显示为ERROR DOWN。根因4和5涉及shutdown，但接口状态是UP，所以可能性较低。根因9、10、11、12则可能在其他情况下适用，但当前数据不支持这些情况。

因此，根因排序可能是根因7、根因8、根因2、根因3，然后是其他可能性较低的根因。
</think>

```json
["根因7-本端-频繁闪断", "根因8-本端-闪断", "根因2-对端-服务器闪断", "根因3-对端-服务器 BGP 未恢复", "根因6-本端-errordown", "根因4-本端-主动 shutdown", "根因5-对端-主动 shutdown", "根因9-本端-BGP未恢复（有闪断）", "根因10-本对端-接口 down 异常", "根因11-本端-BGP未恢复（无闪断）", "根因12-未知-需人工复核"]
```
'''.strip()
INTERFACE_FEW_SHOT='''
**输入**：
- 告警场景名：接口状态DOWN
- 语义标签：     "semantic_labels": {
    "anomaly_logs": {
      "local_flash": {
        "linkDown_active(l)": 6,
        "linkDown_clear(l)": 2,
        "hwPortDown(l)": 2,
        "hwPortUp(l)": 2,
        "hwBgpBackwardTransition_active(l)": 0,
        "hwBgpBackwardTransition_clear(l)": 0,
        "bgpBackwardTransition_active(l)": 0,
        "bgpBackwardTransition_clear(l)": 0,
        "PEER_STATE_CHG(l)": 1
      },
      "remote_flash": {
        "linkDown_active(l)": 0,
        "linkDown_clear(l)": 0,
        "hwPortDown(l)": 0,
        "hwPortUp(l)": 0,
        "hwBgpBackwardTransition_active(l)": 0,
        "hwBgpBackwardTransition_clear(l)": 0,
        "bgpBackwardTransition_active(l)": 0,
        "bgpBackwardTransition_clear(l)": 0,
        "PEER_STATE_CHG(l)": 0
      },
      "local_shutdown": {
        "user": "cnc_device_write@coa",
        "time": "2025-09-27 11:59:08"
      },
      "remote_shutdown": {},
      "local_exception": {
        "Reason=Interface physical link is down": 9,
        "Reason=BGP direct connect-interface down": 0,
        "The local fault alarm has occurred": 0,
        "The physical status of the port changed to down": 6,
        "The remote fault alarm has occurred": 8,
        "Optical Module is invalid": 16,
        "The BFD session went Down": 0,
        "hwBfdSessDown(t)": 0,
        "CurrState=ESTABLISHED": 0,
        "CMDRECORD": 0,
        "REBOOT": 0
      },
      "remote_exception": {
        "Reason=Interface physical link is down": 0,
        "Reason=BGP direct connect-interface down": 0,
        "The local fault alarm has occurred": 0,
        "The physical status of the port changed to down": 0,
        "The remote fault alarm has occurred": 0,
        "Optical Module is invalid": 0,
        "The BFD session went Down": 0,
        "hwBfdSessDown(t)": 0,
        "CurrState=ESTABLISHED": 0,
        "CMDRECORD": 0,
        "REBOOT": 0
      },
      "other_anomaly_logs": []
    },
    "anomaly_kpi": {},
    "anomaly_netstreams": {},
    "other_info": {
      "command_echo_info": {
        "BGP_protocol_status": "Established",
        "local_interface_status": "Administratively DOWN",
        "remote_interface_status": "DOWN",
        "sys_running_time": "",
        "other_command_echo": [
          {}
        ]
      },
      "device_info": {
        "local_device": {
          "management_ip": "29.97.184.15",
          "device_name": "cnsouthwest2e-nc04_bpod02-ce16800x16-ssw_roce-29.97.184.15",
          "device_level": "CE16800-X16-AH",
          "owner": "DCNOM(数据中心网络运维)",
          "resource_status": "Running"
        },
        "remote_device": {
          "management_ip": "29.97.185.193",
          "device_name": "cnsouthwest2e-nc04_bpod02-dx511-rap_roce-29.97.185.193",
          "device_level": "DX511",
          "owner": "DCNOM(数据中心网络运维)",
          "resource_status": "Running"
        },
        "local_ifname": "400GE6/0/15:2",
        "local_server_status": {}
      }
    },
    "alarm_time": 1758945682677
  }
**输出**:
好，我现在需要根据给定的告警数据和标准操作流程，来判断接口状态DOWN的根因，并按照可能性从高到低排序候选根因列表。首先，我得仔细分析输入的数据和流程。

首先，告警场景是“接口状态DOWN”，语义标签里有很多信息。我注意到在“local_exception”中有多个日志，特别是“Reason=Interface physical link is down”出现了9次，“The physical status of the port changed to down”有6次，“Optical Module is invalid”有16次。这些日志表明本端接口物理链路有问题，可能涉及硬件故障或者光模块问题。

接下来，查看“command_echo_info”中的信息，本端接口状态是“Administratively DOWN”，而对端是“DOWN”。这可能意味着本端接口被手动关闭了，或者有其他配置问题导致的关闭。此外，BGP状态是“Established”，说明协议层面没有问题，但接口状态DOWN可能影响业务。

标准操作流程从Procedure 1开始，检查设备类型和分区是否为L1设备或状态正常。假设设备状态正常，进入Procedure 2，诊断为根因1-对端-服务器状态异常。但根据语义标签，对端的异常日志都是0，所以这可能不是主要原因。

接下来，检查本端接口状态是否为ERROR DOWN，Procedure 3。这里本端状态是Administratively DOWN，不是ERROR，所以进入Procedure 5。然后检查是否为Administratively DOWN，如果是，进入Procedure 6，诊断为根因3-本端-接口主动关闭。

但语义标签中的“local_shutdown”显示有用户在特定时间关闭了接口，这可能支持根因3的可能性。然而，同时，本端的异常日志中有很多物理链路问题，可能意味着硬件或光模块问题，这可能属于根因2或根因5。

另外，闪断次数在local_flash中有linkDown_active(l)=6，linkDown_clear(l)=2，这可能意味着接口有闪断的情况，但次数是否超过3次呢？Procedure 9检查闪断次数是否大于等于3，如果是，进入Procedure 10，判断是否为monitor-link场景。如果不是，进入Procedure 13，诊断为根因5-本端-接口闪断频繁。

综合来看，本端接口被主动关闭（根因3）的可能性很高，因为状态是Administratively DOWN，而且有shutdown记录。同时，物理链路问题（根因2）和闪断频繁（根因5）也是可能的。对端主动关闭（根因4）的可能性较低，因为对端的日志都是0。

所以，根因排序应该是：根因3，根因2，根因5，根因4，然后是其他可能性较低的根因。
</think>

```json
["根因3-本端-接口主动关闭", "根因2-本端-接口errordown", "根因5-本端-接口闪断频繁", "根因4-对端-接口主动关闭", "根因1-对端-服务器状态异常", "根因6-本端-monitor-link场景,频繁闪断", "根因7-本端-接口掉线且发生闪断(非flap)", "根因8-本端-发生闪断,但接口up"]
```


'''.strip()
NETWORK_FEW_SHOT='''
**输入**：
- 告警场景名：网络设备掉线
- 语义标签：   "semantic_labels": {
    "anomaly_logs": {
      "local_flash": {
        "linkDown_active(l)": 0,
        "linkDown_clear(l)": 0,
        "hwPortDown(l)": 0,
        "hwPortUp(l)": 0,
        "hwBgpBackwardTransition_active(l)": 0,
        "hwBgpBackwardTransition_clear(l)": 0,
        "bgpBackwardTransition_active(l)": 0,
        "bgpBackwardTransition_clear(l)": 0,
        "PEER_STATE_CHG(l)": 0
      },
      "remote_flash": {
        "linkDown_active(l)": 0,
        "linkDown_clear(l)": 0,
        "hwPortDown(l)": 0,
        "hwPortUp(l)": 0,
        "hwBgpBackwardTransition_active(l)": 0,
        "hwBgpBackwardTransition_clear(l)": 0,
        "bgpBackwardTransition_active(l)": 0,
        "bgpBackwardTransition_clear(l)": 0,
        "PEER_STATE_CHG(l)": 0
      },
      "local_shutdown": {},
      "remote_shutdown": {},
      "local_exception": {
        "Reason=Interface physical link is down": 0,
        "Reason=BGP direct connect-interface down": 0,
        "The local fault alarm has occurred": 0,
        "The physical status of the port changed to down": 0,
        "The remote fault alarm has occurred": 0,
        "Optical Module is invalid": 0,
        "The BFD session went Down": 0,
        "hwBfdSessDown(t)": 0,
        "CurrState=ESTABLISHED": 0
      },
      "remote_exception": {
        "Reason=Interface physical link is down": 0,
        "Reason=BGP direct connect-interface down": 0,
        "The local fault alarm has occurred": 0,
        "The physical status of the port changed to down": 0,
        "The remote fault alarm has occurred": 0,
        "Optical Module is invalid": 0,
        "The BFD session went Down": 0,
        "hwBfdSessDown(t)": 0,
        "CurrState=ESTABLISHED": 0
      }
    },
    "anomaly_kpi": {
      "offline_loss_rate": {
        "upper_bound": 90.0,
        "abnormal_timing_segments": [
          {
            "start_time": 1759038180,
            "end_time": 1759038300,
            "value": [
              0.0,
              96.67,
              3.33
            ],
            "pattern": "spike"
          }
        ]
      }
    },
    "anomaly_netstreams": {},
    "other_info": {
      "command_echo_info": {
        "BGP_protocol_status": {},
        "local_interface_status": {},
        "remote_interface_status": {},
        "sys_running_time": {},
        "other_command_echo": [
          {
            "cmd": {},
            "echo": {}
          }
        ]
      },
      "device_info": {
        "local_device": {
          "management_ip": "11.171.55.248",
          "device_name": "runorthwest201a-bmc-s5731-tor-11.171.55.248",
          "device_level": "S5731-H48T4XC",
          "owner": "HCSO Network(HCSO网络)",
          "resource_status": "Running"
        },
        "remote_device": {
          "management_ip": {},
          "device_name": {},
          "device_level": {},
          "owner": {},
          "resource_status": {}
        },
        "local_ifname": {},
        "local_server_status": {}
      }
    }
  }
**输出**：
好，我现在需要根据给定的输入，按照标准操作流程（SOP）来判断网络设备掉线的根因，并对候选根因进行排序。首先，我要仔细分析输入中的各个部分，包括告警场景名、语义标签、SOP和候选根因列表。

首先，告警场景名是“网络设备掉线”，这意味着设备可能无法正常连接到网络。接下来，语义标签中包含了很多信息，我需要逐一查看这些信息，看看是否有异常日志、KPI异常、网络流异常或其他信息。

在anomaly_logs部分，分为local_flash、remote_flash、local_shutdown、remote_shutdown、local_exception和remote_exception。每个部分都有一些日志条目，但它们的值都是0，这可能意味着这些日志没有被触发或者没有异常。例如，linkDown_active(l)和hwPortDown(l)都是0，说明本地和远端的链路没有出现down的情况。同样，BGP相关的日志也没有异常。这可能意味着设备的连接状态正常，没有出现链路中断的情况。

接下来，anomaly_kpi中的offline_loss_rate有一个上界90.0，并且在某个时间段内，丢包率达到了96.67%和3.33%。这表明丢包率超过了正常范围，但并没有达到100%。这可能意味着设备在某些时候丢包严重，但并没有完全断开连接。

在其他信息中，device_info显示本地设备的状态是Running，说明服务器状态正常。而远程设备的信息为空，可能是因为没有获取到相关信息，或者远程设备不存在问题。

现在，我需要按照标准操作流程来判断根因。SOP分为多个步骤：

Step 1：检查设备分区是否为L1设备。如果Yes，进入Step2，否则进入Step4。由于设备型号是S5731-H48T4XC，这可能不是L1设备，但不确定，可能需要进一步确认。假设这里判断为No，进入Step4。

Step4：检查设备信息中的sys_running_time是否存在。如果存在，进入Step5，否则进入Step6。根据device_info，sys_running_time是空的，所以进入Step6。

Step6：检查历史重启命令是否存在，判断是否是人为变更导致重启。如果存在，进入Step8，否则进入Step9。假设这里没有历史重启命令，进入Step9。

Step9：检查接口丢包率是否超过100%。根据anomaly_kpi，丢包率最高是96.67%，没有超过100%，所以进入Step11。

Step11：诊断根因为根因5-本端-设备丢包率<100%。这意味着丢包率虽然异常，但没有达到100%，可能需要进一步调查。

现在，候选根因列表有五个选项，我需要根据SOP的结果进行排序。根据SOP，最终诊断是根因5，所以它应该是可能性最高的。接下来，考虑其他可能的根因。

根因4是丢包率=100%，但实际丢包率是96.67%，所以可能性较低。根因3是设备重启，但SOP中没有检测到重启命令，所以可能性较低。根因2是设备未掉线，但设备确实出现了掉线告警，所以可能性较低。根因1是服务器未运行，但device_info显示服务器状态正常，所以可能性最低。

因此，根因的可能性排序应该是根因5、根因4、根因3、根因2、根因1。
</think>

```json
["根因5-本端-设备丢包率<100%", "根因4-本端-设备丢包率=100%", "根因3-本端-设备重启（变更引起）", "根因2-本端-设备未掉线", "根因1-本端-服务器未运行"]
```

'''.strip()
INBW_FEW_SHOT='''
**输入**：
- 告警场景名：入方向带宽利用率过高
- 语义标签：  "semantic_labels": {
    "anomaly_logs": {
      "other_anomaly_logs": []
    },
    "anomaly_kpi": {
      "traffic_in": {
        "upper_bound": 80.0,
        "abnormal_timing_segments": [
          {
            "start_time": 1758357570,
            "end_time": 1758358230,
            "value": [
              58.0606,
              92.2404,
              91.2768,
              95.4835,
              96.2941,
              96.4113,
              94.6398,
              96.0992,
              77.4146,
              51.9977,
              36.335,
              38.5424
            ],
            "pattern": "levelUp"
          }
        ]
      },
      "traffic_out": {
        "upper_bound": 80.0,
        "abnormal_timing_segments": [
          {
            "start_time": 1758357510,
            "end_time": 1758358230,
            "value": [
              26.6583,
              27.8436,
              38.047,
              37.6722,
              38.922,
              39.2169,
              39.3639,
              38.5714,
              39.1955,
              33.7367,
              25.984,
              17.9074,
              18.9964
            ],
            "pattern": "levelUp"
          }
        ]
      },
      "drop_packet_rate": {
        "upper_bound": 5.0,
        "abnormal_timing_segments": [
          {
            "start_time": 1758357450,
            "end_time": 1758358170,
            "value": [
              26.6477,
              26.6583,
              27.8436,
              38.047,
              37.6722,
              38.922,
              39.2169,
              39.3639,
              38.5714,
              39.1955,
              33.7367,
              25.984,
              17.9074
            ],
            "pattern": "levelUp"
          }
        ]
      },
      "error_packet_rate": {
        "upper_bound": 5.0,
        "abnormal_timing_segments": [
          {
            "start_time": 1758357450,
            "end_time": 1758358170,
            "value": [
              26.6477,
              26.6583,
              27.8436,
              38.047,
              37.6722,
              38.922,
              39.2169,
              39.3639,
              38.5714,
              39.1955,
              33.7367,
              25.984,
              17.9074
            ],
            "pattern": "levelUp"
          }
        ]
      }
    },
    "anomaly_netstreams": [
      {
        "start_time": 1758357060,
        "end_time": 1758357780,
        "value": [
          1106318400,
          28061217600,
          27552926400,
          23618016000,
          43958908800
        ],
        "pattern": "increase",
        "source": "26.82.163.197",
        "target": "26.82.200.32"
      }
    ],
    "other_info": {
      "command_echo_info": {
        "BGP_protocol_status": "Established",
        "local_interface_status": "UP",
        "remote_interface_status": "UP",
        "sys_running_time": "",
        "other_command_echo": [
          {}
        ]
      },
      "device_info": {
        "local_device": {
          "management_ip": "26.135.30.94",
          "device_name": "cnnorth9c-nc01_bpod15-fm8850-rap_roce-26.135.30.94",
          "device_level": "FM8850-64CQ-EI",
          "owner": "DCNOM(数据中心网络运维)",
          "resource_status": "Running"
        },
        "remote_device": {
          "management_ip": "26.135.17.42",
          "device_name": "cnnorth9c-nc01_bpod15-fm9860-ssw_roce-26.135.17.42",
          "device_level": "FM9860-4C-EI",
          "owner": "DCNOM(数据中心网络运维)",
          "resource_status": "Running"
        },
        "local_ifname": "100GE1/0/58",
        "local_server_status": {}
      }
    },
    "alarm_time": 1758357840000
  },

**输出**：

嗯，我现在需要根据给定的告警数据和标准操作流程，来判断根因的可能性并排序。首先，我得仔细分析输入的信息，然后按照SOP的步骤一步步排查。

首先，告警场景是“入方向带宽利用率过高”。语义标签里有很多信息，包括异常日志、KPI、网络流、设备信息等等。让我先看看KPI部分，因为带宽利用率高通常和流量有关。

在anomaly_kpi里，traffic_in的上界是80%，异常时间段的值有多个点超过了这个值，比如最高到96.4113%。traffic_out也有异常，但上界同样是80%，不过它的值最高到39.3639%，没有超过80%。同时，drop_packet_rate和error_packet_rate的上界是5%，但异常值却高达39%左右，这可能意味着网络中有大量的丢包和错误包，这可能影响了流量。

接下来，看看anomaly_netstreams，有一个异常流，从26.82.163.197到26.82.200.32，流量值在增加，这可能是一个大流量的连接，导致入方向带宽利用率过高。

设备信息方面，local_device和remote_device都属于DCNOM，状态都是Running，接口状态都是UP，BGP状态是Established，看起来网络连接正常，没有明显的故障。

现在按照SOP的步骤来分析：

Procedure 1：检查local_ifname是否包含"Vlanif"。这里local_ifname是"100GE1/0/58"，不包含Vlanif，所以进入Procedure 3。

Procedure 3：检查owner是否包含"DCNOM(数据中心网络运维)"。这里local_device和remote_device的owner都是DCNOM，所以进入Procedure 4。

Procedure 4：检查alarm_time时间点，traffic_in是否超过90%。在给定的anomaly_kpi中，traffic_in的值在alarm_time附近确实有超过90%的情况，比如96.2941%。所以进入Procedure 6。

Procedure 6：检查是否超过110%。traffic_in的值最高是96.4113%，没超过110%，所以进入Procedure 9。

Procedure 9：诊断为根因2-本端-接口流量异常，因为带宽利用率超过90%。

接下来，我需要考虑候选根因列表中的其他可能性。虽然Procedure 9的结论是根因2，但还有其他可能吗？比如，是否有流量恢复正常的情况，或者持续过高？

在Procedure 8中，检查alarm_time后15分钟，traffic_in是否降到80%以下。假设在给定的异常时间段中，traffic_in在某个点降到了77.4146%，然后继续下降到51.9977%等，这可能意味着流量在某个时间点恢复正常。但根据Procedure 9，alarm_time点已经超过了90%，所以Procedure 9的结论是根因2。

另外，是否有设备采集故障的可能？Procedure 7的情况是当traffic_in超过110%时，但这里最高是96.41%，所以Procedure 7不适用，根因5的可能性较低。

根因3的情况是当流量降到80%以下，但根据alarm_time点的值，可能在之后恢复，但Procedure 9的结论是基于alarm_time点的高利用率，所以根因2的可能性更高。

根因4是接口流量持续过高，但根据Procedure 9的结论，是基于单个时间点超过90%，而根因4可能需要持续时间更长的情况，但根据SOP，Procedure 9已经覆盖了这一点。

根因1的情况是当对端设备为服务器，但Procedure 2和5的情况只有在owner不是DCNOM时才会触发，而这里owner是DCNOM，所以根因1的可能性较低。

综上所述，根因的可能性从高到低应该是：

根因2-本端-接口流量异常，因为traffic_in超过了90%。

根因4-本端-接口流量持续过高，可能在后续时间点持续高。

根因5-本端-设备流量采集故障，可能性较低，因为traffic_in没超过110%。

根因3-本端-接口流量恢复正常，可能在后续时间点，但告警时间点已经很高，所以可能不是当前的主要原因。

根因1-对端-服务器，可能性最低，因为owner是DCNOM。
</think>

```json
["根因2-本端-接口流量异常", "根因4-本端-接口流量持续过高", "根因5-本端-设备流量采集故障", "根因3-本端-接口流量恢复正常", "根因1-对端-服务器"]
```

==================================================


'''.strip()
ROOT_CAUSE_Meta_thinker = '''
### 任务描述
你是一个网络运维助手，作为多智能体协作系统中的Meta-thinker智能体，负责指导Reasoner具体分析告警观测数据的语义标签。
- 你将收到
  1. **上一轮Reasoner和Meta-thinker的输出**
  2. **语义标签的键名及含义 semantic_labels_key**(指示语义标签中包含了哪些信息)
  3. **告警类型 alarm_type**
  4. **标准操作流程 sop**（按步骤给出了从观测数据导向最终根因的逻辑流程，遵循该流程判断根因；若该部分为空，则根据你的经验自行判断）
  5. **候选根因列表 root_cause_candidates**
- 请根据以上输入信息，为Reasoner提供具体的分析指导建议，帮助其基于语义标签进行根因分析，并根据上一轮的Meta-thinker输出以及Reasoner输出进行根因排序，若为第一轮，根因排序为空。


### 输入
- **上一轮Meta-thinker和Reasoner输出**：{Last_round_outputs}
- **语义标签的键名及含义**：{semantic_labels_key}
- **告警类型**：{alarm_type}
- **sop**:{sop}
- **候选根因列表**{root_cause_candidates}

### 输出要求
- 请你根据上一轮Meta-thinker和Reasoner的输出和sop，进行单步推理指导，指导Reasoner逐步检查语义标签，逐步多轮推理出根因。
- 当Reasoner分析到标准操作流程的某一步时，表明前面的步骤已经查询完毕，请勿重复输出已执行完的指导步骤。
- 在 </think> 之前进行充分思考；</think> 之后，严格按照以下格式输出
- 用 $$ 将整个列表包裹，字符串均用英文双引号包裹。
- 对根因候选列表中的所有 category 字段内容进行排序!!!
- 特别注意：当你觉得上一轮的Reansor和Meta-thinker输出的内容已经符合要求，那么请在指导意见中输出["(上一轮次已经符合要求)"]，并给出根因排序,注意当上一轮Meta-thinker和Reasoner的输出为空时，说明是第一轮，不输出根因。

$$
["(在此输出指导意见 instruchtion)"],
["(在此输出根因 category)", "(在此输出根因 category)", ..., "(在此输出根因 category)"]
$$
  '''.strip()

ROOT_CAUSE_Reasoner = '''
### 任务描述
你是一个网络运维助手，作为多智能体协作系统中的Reasoner智能体，负责根据Meta_output中的指导意见分析告警观测数据的语义标签。
- 你将收到
  1. **Meta_output**(Meta-thinker根据上一轮Reasoner和Meta-thinker的输出、语义标签的键名及含义、告警类型、候选根因列表等生成的指导意见)
  2. **语义标签 semantic_labels**（从原始告警观测数据中提取得到的语义化信息，指示了 Meta_output 中涉及与根因有关的日志、KPI、网流、回显、设备等信息)
  3. **告警场景名 alarm_type**
  4. **候选根因列表 root_cause_candidates**
  5. **告警时间 alarm_time**

### 输入
- **Meta_output**：{meta_output}
- **告警场景名**：{alarm_type}
- **告警时间**：{alarm_time}
- **语义标签**：{semantic_labels}
- **候选根因列表**：{root_cause_candidates}

### 输出要求
- 输出你根据Meta_output的指示,对语义标签进行查询以及分析，输出你根据Meta_output进行分析的结果。
- 在 </think> 之前进行充分思考；</think> 之后，严格按照以下格式输出。
- 用 $$ 将整个列表包裹，字符串均用英文双引号包裹。
$$
["(根据Meta-outputs,我对**进行了分析，结果为**,请Meta根据我的输出进行下一步指导)"]
$$
'''.strip()

ROOT_CAUSE_Reasoner1 = '''
### 任务描述
你是一个网络运维助手，负责根据告警观测数据的语义标签并严格参考标准操作流程 sop 给出根因排序。
- 你将收到
  1. **告警场景名 alarm_type**
  2. **语义标签 semantic_labels**（从原始告警观测数据中提取得到的语义化信息，指示了 sop 中涉及与根因有关的日志、KPI、网流、回显、设备等信息）
  3. **标准操作流程 sop**（按步骤给出了从观测数据导向最终根因的逻辑流程，遵循该流程判断根因；若该部分为空，则根据你的经验自行判断）
  4. **候选根因列表 root_cause_candidates**
  5. **告警时间 alarm_time**
- 你需要按照以下思考步骤完成任务：
  ## Step 1: 语义标签分析：逐一分析语义标签中的 anomaly_logs, anomaly_kpi, anomaly_netstreams, other_info.
  ## Step 2: 根因推理：基于语义标签的分析结果，遵循标准操作流程 sop 的逻辑判断根因。
  ## Step 3: 反思：反思上述思考过程的合理性，避免逻辑上的遗漏或错误。
  ## Step 4: 组织排序：查看候选根因列表中所有的 category 字段，并结合之前的分析按照根因可能性从高到低排序。

### 输入
- **告警场景名**：{alarm_type}
- **语义标签**：{semantic_labels}
- **告警时间**：{alarm_time}
- **标准操作流程**：{sop}
- **候选根因列表**：{root_cause_candidates}

### 输出要求
- **思考过程**：开头标出 "## Step 1: 语义标签分析"，"## Step 2: 根因推理"，"## Step 3: 反思"，"## Step 4: 组织排序"
- **排序结果**：对根因候选列表中的所有 category 字段内容进行排序
- 特别注意：严格按照如下格式，仅输出**思考过程**和**排序结果**，且两者用唯一标识 </think> 分隔：
## Step 1: 语义标签分析
[在此输出思考过程1]
## Step 2: 根因推理
[在此输出思考过程2]
## Step 3: 反思
[在此输出思考过程3]
## Step 4: 组织排序
[在此输出思考过程4]
</think>
$$
["(在此输出根因 category)", "(在此输出根因 category)", ..., "(在此输出根因 category)"]
$$
'''.strip()

ROOT_CAUSE_Verifier = '''
### 任务描述
你是一个网络运维助手，作为Verifier智能体，负责从多个Reasoner的根因排序中选择最合理的结果。

### 输入
- **告警场景名**：{alarm_type}
- **语义标签**：{semantic_labels}
- **标准操作流程**：{sop}
- **候选根因列表**：{root_cause_candidates}
- **Reasoner输出结果**：{reasoner_outputs}

### 评选标准
比较各Reasoner的输出结果，基于：
1. **sop遵循度**：是否严格遵循标准操作流程
2. **证据利用**：是否充分使用语义标签中的信息
3. **逻辑合理性**：排序是否符合网络运维常识

### 输出要求
在 </think> 之前进行分析，然后按照如下格式输出：

## 评估结果
- 最佳排序来自：Reasoner [编号]
- 选择理由：[简要说明]

</think>

### 最终排序
使用最佳Reasoner的根因排序结果：
$$
["(在此输出根因 category)", "(在此输出根因 category)", ..., "(在此输出根因 category)"]
$$
'''.strip()


ROOT_CAUSE_Meta_thinker_Coopetition = '''
### 任务描述
你是一个网络运维助手，作为多智能体协作系统中的Meta-thinker智能体，负责指导Reasoner具体分析告警观测数据的语义标签,并根据Verifier的输出给出根因排序。
- 你将收到
  1. **上一轮Verifier和Meta-thinker的输出**
  2. **语义标签的键名及含义 semantic_labels_key**(指示语义标签中包含了哪些信息)
  3. **告警类型 alarm_type**
  4. **标准操作流程 sop**（按步骤给出了从观测数据导向最终根因的逻辑流程，遵循该流程判断根因；若该部分为空，则根据你的经验自行判断）
  5. **候选根因列表 root_cause_candidates**
- 请根据以上输入信息，为Reasoner提供具体的分析指导建议，帮助其基于语义标签进行根因分析，并根据上一轮的Meta-thinker输出以及Verifier输出进行根因排序，若为第一轮，根因排序为空。


### 输入
- **上一轮Meta-thinker和Verifier的输出**：{Last_round_outputs}
- **语义标签的键名及含义**：{semantic_labels_key}
- **告警类型**：{alarm_type}
- **sop**:{sop}
- **候选根因列表**{root_cause_candidates}

### 输出要求
- 请你根据上一轮Meta-thinker和Reasoner的输出和sop，进行单步推理指导，指导Reasoner逐步检查语义标签，逐步多轮推理出根因。
- 当Reasoner分析到标准操作流程的某一步时，表明前面的步骤已经查询完毕，请勿重复输出已执行完的指导步骤。
- 在 </think> 之前进行充分思考；</think> 之后，严格按照以下格式输出
- 对根因候选列表中的所有 category 字段内容进行排序!!
- 用 $$ 将整个列表包裹，字符串均用英文双引号包裹。
- 特别注意：当你觉得上一轮的Verifier和Meta-thinker输出的内容已经符合要求，那么请在指导意见中输出["(上一轮次已经符合要求)"]，并给出根因排序,注意若Meta-thinker和Verifier为空时，说明是第一轮，根因输出结果为空。
$$
["(在此输出指导意见 instruchtion)"],
["(在此输出根因 category)", "(在此输出根因 category)", ..., "(在此输出根因 category)"]
$$
  '''.strip()

ROOT_CAUSE_Reasoner_Coopetiton = '''
### 任务描述
你是一个网络运维助手，作为多智能体协作系统中的Reasoner智能体，负责根据Meta_output中的指导意见分析告警观测数据的语义标签。
- 你将收到
  1. **Meta_output**(Meta-thinker根据上一轮Verifier和Meta-thinker的输出、语义标签的键名及含义、告警类型、候选根因列表等生成的指导意见)
  2. **语义标签 semantic_labels**（从原始告警观测数据中提取得到的语义化信息，指示了 Meta_output 中涉及与根因有关的日志、KPI、网流、回显、设备等信息)
  3. **告警场景名 alarm_type**
  4. **候选根因列表 root_cause_candidates**
  5. **告警时间 alarm_time**

### 输入
- **Meta_output**：{meta_output}
- **语义标签**：{semantic_labels}
- **告警时间**：{alarm_time}
- **告警场景名**：{alarm_type}
- **候选根因列表**：{root_cause_candidates}

### 输出要求
- 输出你根据Meta_output的指示分析语义标签的推理过程。
- 在 </think> 之前进行充分思考；</think> 之后，严格按照以下格式输出。
- 用 $$ 将整个列表包裹，字符串均用英文双引号包裹。
$$
["(在此输出根据Meta_output的指示分析语义标签的推理过程)"]
$$
'''.strip()

ROOT_CAUSE_Verifier_Coopetiton = '''
### 任务描述
你是一个网络运维助手，作为Verifier智能体，负责从多个Reasoner的根因分析流程中选择最合理的结果。

### 输入
- **告警场景名**：{alarm_type}
- **语义标签**：{semantic_labels}
- **Meta_output**：{meta_output}
- **候选根因列表**：{root_cause_candidates}
- **Reasoner输出结果**：{reasoner_outputs}

### 评选标准
比较各Reasoner的输出结果，基于：
1. **Meta_output遵循度**：是否严格遵循Meta_output操作流程
2. **证据利用**：是否充分使用语义标签中的信息
3. **逻辑合理性**：排序是否符合网络运维常识

### 输出要求
在 </think> 之前进行分析，然后输出：

## 评估结果
- 最佳排序来自：Reasoner [编号]
- 选择理由：[简要说明]

</think>

### 最终排序
使用最佳Reasoner的根因的分析结果：
$$
["(最佳Reasoner的分析流程)"]
$$
'''.strip()

# ============================================================
# Selector / Refiner prompts
# ============================================================

SELECTOR_PROMPT = r'''
你是 RCA 决策树特征选择器。根据告警模板信息，选择对根因分类最有帮助的日志和 KPI 特征。

告警类型：{alarm_type}
语义标签：{semantic_labels}
告警时间：{alarm_time}
SOP：{sop}
候选根因：{root_cause_candidates}

请先分析，再在 </think> 后输出且只输出一个 JSON 代码块：
```json
{{
  "log": ["template_occur_count", "flash_count"],
  "kpi": {{
    "traffic_in": ["mix", "mean"],
    "traffic_out": ["mix", "mean"],
    "drop_packet_rate": ["mix", "mean"],
    "error_packet_rate": ["mix", "mean"],
    "offline_loss_rate": ["mix", "mean"],
    "cpu_utilization": ["mix", "mean"],
    "memory_utilization": ["mix", "mean"],
    "temperature": ["mix", "mean"]
  }}
}}
```
'''.strip()

REFINER_PROMPT = r'''
你是 RCA 决策树特征选择优化器。根据当前 selection 和决策树推理 summary，调整特征选择以减少错误案例。

告警类型：{alarm_type}
语义标签：{semantic_labels}
告警时间：{alarm_time}
SOP：{sop}
候选根因：{root_cause_candidates}
当前 selection：{previous_selection}
Tree summary：{validcase_summary}

请先分析，再在 </think> 后输出且只输出一个 JSON 代码块，格式必须为：
```json
{{
  "log": ["template_occur_count", "flash_count"],
  "kpi": {{
    "traffic_in": ["mix", "mean"],
    "traffic_out": ["mix", "mean"],
    "drop_packet_rate": ["mix", "mean"],
    "error_packet_rate": ["mix", "mean"],
    "offline_loss_rate": ["mix", "mean"],
    "cpu_utilization": ["mix", "mean"],
    "memory_utilization": ["mix", "mean"],
    "temperature": ["mix", "mean"]
  }}
}}
```
'''.strip()

ROLE_PROMPTS = {
    "SELECTOR": SELECTOR_PROMPT,
    "REFINER": REFINER_PROMPT,
}
