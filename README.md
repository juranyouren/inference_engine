# RCA 推理评测

以下命令在仓库根目录运行。数据集根目录的直接子目录按告警类别发现，样本按文件名中的数字 index 排序，例如 `网络设备掉线_50_label_10475196.json`。

## 只运行 Competition 和 Cooperation

```bash
python run_agent_evaluation.py \
  --label-root /path/to/dataset \
  --import-top-level-sop \
  --output-dir res/agent_evaluation
```

```bash
bash run_agent_evaluation_parallel.sh \
  --label-root /path/to/dataset \
  --output-dir res/agent_evaluation_parallel
```

并行入口默认使用 8 张卡，每两张卡运行一个分片。两个入口均支持 `--limit-per-mode N`。这套评测的 `both` 模式对同一批带 SOP 的样本运行两个 Agent，Cooperation 的输入会移除 SOP。详细参数见 [LLM 接口说明](llm_interfaces/README.md)。

## 每类前 50 条训练，接下来的 150 条测试

```bash
bash run_all_inference_modes.sh \
  /path/to/dataset \
  res/fixed_split \
  /path/to/selections \
  --ascend-devices 6,7 --import-top-level-sop
```

Tree 用每类前 50 条训练一次。Tree 和 Cooperation 使用接下来的 150 条测试，Competition 使用其中有 SOP 的样本，其余计入 `skipped_count`。两个 Agent 共用同一个 LLM 实例。SOP 已在 `semantic_labels.sop` 时可省略 `--import-top-level-sop`。

Tree 从 selection 目录下的 `<类别>/selection.json` 读取特征配置，也可直接指定一个 selection 文件。省略 selection 路径时从数据集目录查找。各模式的预测、明细和指标写入 `res/fixed_split/<mode>/`，整体指标写入输出根目录。

使用 Python 入口可以单独运行某个模式，例如 `--mode tree`：

```bash
python run_tree_agent_evaluation.py \
  --dataset-root /path/to/dataset \
  --mode tree \
  --selection-path /path/to/selections \
  --output-dir res/fixed_split
```

两套评测均独立于生产增量入口 `infer_by_index.py`，评测过程不读写增量状态。
