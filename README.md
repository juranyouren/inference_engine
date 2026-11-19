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

两套评测的 Agent 都在同一类别、同一分片内跨 case 组批。`--batch-size N` 控制每次送入模型的 prompt 数量上限，默认 8。竞争型先批量诊断候选路径，再批量运行 Verifier。合作型依次批量运行 Orchestrator、首轮专家和各轮复议，最后批量运行 Aggregator。意见池、早停和调用上限按 case 独立维护。日志中 `[llm.chat] stage=orchestrator prompts=8` 表示该次调用实际提交了 8 条 prompt。

## 每类前 50 条训练，接下来的 150 条测试

```bash
bash run_all_inference_modes.sh \
  /path/to/dataset \
  res/fixed_split \
  --ascend-devices 6,7 --import-top-level-sop
```

Tree 用每类前 50 条训练一次。Tree 和 Cooperation 使用接下来的 150 条测试，Competition 使用其中有 SOP 的样本，其余计入 `skipped_count`。两个 Agent 共用同一个 LLM 实例。SOP 已在 `semantic_labels.sop` 时可省略 `--import-top-level-sop`。

Tree 汇总每类整个训练集的字段、SOP 和显式候选根因，调用一次 Selector，按训练集中出现的事件和 KPI 固定特征列。测试样本缺少的计数记为 0。特征配置保存在 `res/fixed_split/tree/<类别>/selection.json`，该目录的 `selector/source.json` 记录训练索引和字段全集。各类别共享一次 Selector 模型初始化，`--model-path` 和 `--ascend-devices` 同时适用于 Selector。真实根因只用于 Tree 监督训练和评分。

只重跑 Tree，重新生成 selection 并评测决策树的原始排名：

```bash
bash run_tree_only.sh /home/sbp/deploy/case_pool/anomalydetect_label_0 \
  res/tree_rerun --ascend-devices 6,7
```

各模式的预测、明细和指标写入 `res/fixed_split/<mode>/`，整体指标写入输出根目录。需要读取已有配置时可显式传入 `--selection-source file --selection-path /path/to/selections`。

使用 Python 入口可以单独运行某个模式，例如 `--mode tree`：

```bash
python run_tree_agent_evaluation.py \
  --dataset-root /path/to/dataset \
  --mode tree \
  --ascend-devices 6,7 --import-top-level-sop \
  --output-dir res/fixed_split
```

两套评测均独立于生产增量入口 `infer_by_index.py`，评测过程不读写增量状态。
