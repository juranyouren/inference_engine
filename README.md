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

`rule_inferencer/txt2sop.py` 当前是兼容接口版本；Tree 主流程直接在 `infer_by_index.py` 中完成，不依赖它生成最终预测。
