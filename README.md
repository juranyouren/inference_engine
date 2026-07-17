# inference_engine

当前工程用于增量处理两个 label 根目录：

- `AgentDigest_label/<告警类别>/`：LLM 推理
- `anomalydetect_label/<告警类别>/`：Tree 推理

## 默认行为

```bash
cd /home/sbp/deployment/inference_engine
python infer_by_index.py
```

默认 `--infer-type all`，Tree 和 LLM 使用两个独立 `spawn` 进程顺序执行。

## LLM 自动分流

- `semantic_labels["sop"]` 非空 -> Competition：多采样推理 + Verifier
- `semantic_labels["sop"]` 为空 -> Cooperation：多轮 Meta/Reasoner，直到早停或 `--max-rounds`

## 增量状态

状态文件默认：

`/home/sbp/deployment/case_pool/predict_result/state/infer_state.json`

每个类别目录分别维护 `tree_infer.last_index` 与 `llm_infer.last_index`。只有连续成功的新增 index 才推进 checkpoint。

## 常用命令

```bash
# 默认：Tree + LLM
python infer_by_index.py

# 只跑 LLM
python infer_by_index.py --infer-type llm_infer

# 只跑 Tree
python infer_by_index.py --infer-type tree_infer

# 只处理一个类别
python infer_by_index.py --scenario 网络设备掉线

# 小批量新增测试
python infer_by_index.py --max-new-cases 2

# 手动窗口调试（不更新 state）
python infer_by_index.py --start 60 --end 62
```

## Ascend / vLLM

工程入口会在任何 vLLM/torch_npu 初始化前设置：

`VLLM_WORKER_MULTIPROC_METHOD=spawn`

请保留 `if __name__ == "__main__": main()`。

## 说明

`infer_by_index.py` 当前只保留 CLI、运行时初始化和进程调度。公共增量逻辑在 `inference/common.py`，Selection/Selector/Refiner 在 `inference/selection.py`，Tree 主流程在 `inference/tree.py`，LLM 主流程在 `inference/llm.py`。

`rule_inferencer/txt2sop.py` 当前是兼容接口版本；Tree 主流程不依赖它生成最终预测。

## 三种方法对比实验

独立实验入口不会修改生产增量状态。它会动态扫描类别目录，分别运行 Tree、
Competition、Cooperation，并把 Top-1/3/5、MRR、平均耗时写入
`evaluation/metrics.json` 和 `evaluation/metrics.csv`。

```bash
cd /home/sbp/deployment/inference_engine
python run_three_method_experiment.py \
  --output-dir /home/sbp/deployment/case_pool/predict_result/experiments/three_methods
```

三种方法统一读取 `anomalydetect_label`，默认校验 4 个类别，并严格使用每类
索引 51–250（含首尾）的 200 条数据。Tree 使用 51–100 初始化，依次测试
101–150、151–200、201–250，并在每轮加入已测试数据重新训练。每个 Tree 阶段均开启完整的
Selector -> Refiner -> Tree 流程，产物分别保存在 `tree/selector/`、
`tree/refiner/`、`tree/tree/`；两种 LLM 方法分别运行全部 200 条。
三个方法使用独立 spawn 进程，避免 NPU 模型运行时相互污染。

Tree 的每条预测包含 `cot`，记录样本从根节点到叶节点实际经过的判断条件及
最终叶子类别编号，例如：

```json
{
  "cot": {
    "feature_a <= 1.00 -> feature_b > 0.50": 5
  }
}
```

已有推理结果可只重新评测：

```bash
python run_three_method_experiment.py --evaluate-only --output-dir /path/to/experiment
```

## 按 CSN 补充 root_cause

`fill_root_cause_by_csn.py` 从另一个相同类别目录结构的 label 根目录读取
`root_cause`，按 `XXX_<idx>_label_<csn>.json` 文件名中的 CSN 匹配并补充到
`anomalydetect_label`。默认只生成预演报告，不修改 label：

```bash
python fill_root_cause_by_csn.py /path/to/source_label_root
```

确认 `anomalydetect_label/root_cause_fill_report.json` 后实际写入并备份：

```bash
python fill_root_cause_by_csn.py /path/to/source_label_root \
  --apply \
  --backup-dir /path/to/root_cause_backup
```

来源中同一 CSN 对应不同根因、目标缺少 CSN、没有匹配项时不会写入对应文件，
并默认返回非零退出码。

## 重新解析并评分 LLM 结果

模型已经跑完但部分结果未正确解析时，不需要重新推理：

```bash
python reparse_and_score_llm_results.py \
  --experiment-dir /home/sbp/deployment/case_pool/predict_result/experiments/three_methods
```

Competition 优先解析最终 Verifier，失败后汇总三个 Reasoner；Cooperation 从最后
一轮开始依次尝试 Reasoner、Meta，并可向前轮回溯。新结果写入实验目录下的
`reparsed/`，不会覆盖原始结果，其中 `parse_attempts` 记录每个 case 尝试过的
回答文件。重新计算的 Top-1/3/5、MRR 和平均时间位于
`reparsed/evaluation/metrics.json` 与 `metrics.csv`。
