# AGENTS.md

## Project intent

This repository is an RCA inference engine with two incremental pipelines.

1. Tree pipeline reads category subdirectories under `config.ANOMALYDETECT_LABEL_ROOT`.
2. LLM pipeline reads category subdirectories under `config.AGENTDIGEST_LABEL_ROOT`.

Do not reintroduce a hard-coded four-scenario list. Category count is determined dynamically by the subdirectories present under each root.

## Incremental processing

- State file: `config.INFER_STATE_FILE`.
- State key is the absolute category folder path plus inference mode (`tree_infer` / `llm_infer`).
- Default mode processes only indices greater than the saved checkpoint.
- Explicit `--start/--end` is a debug window and must not update state.
- Advance checkpoints only through the contiguous prefix of successful selected indices. Never skip over a failed index.

## LLM routing

Read SOP only from `label["semantic_labels"]["sop"]`.

- non-empty SOP -> Competition
  - `generate_rca_analysis_competition_batch`
  - `generate_rca_analysis_verifier_batch`
- empty SOP -> Cooperation
  - repeated `generate_rca_analysis_meta`
  - `generate_rca_analysis_reasoner`
  - early stop marker: `["(上一轮次已经符合要求)"]`

## Tree selection sources

- `file`
- `selector`
- `selector_refiner`

`selector_refiner` means initial LLM Selector -> Tree run/summary -> LLM Refiner -> final Tree run.

## Import rules

Use only `utils.public_functions`. Do not create or import `public_fuctions`.
Project root is `/home/sbp/deployment/inference_engine` on the deployment machine.

## Ascend NPU multiprocessing

Keep `VLLM_WORKER_MULTIPROC_METHOD=spawn` before importing modules that may initialize `torch_npu` or vLLM.
Default `all` mode runs Tree and LLM in separate spawn processes to avoid NPU runtime contamination across model instances.

## Main files

- `infer_by_index.py`: CLI entrypoint, runtime setup, and process orchestration.
- `inference/common.py`: shared output, dynamic scanning, incremental state, and index-based case loading.
- `inference/selection.py`: selection normalization plus Selector/Refiner integration.
- `inference/tree.py`: Tree pipeline orchestration, feature extraction, training, and prediction.
- `inference/llm.py`: LLM SOP routing plus Competition/Cooperation orchestration.
- `config.py`: paths and runtime defaults.
- `utils/public_functions.py`: shared I/O and LLM data loaders.
- `rule_inferencer/data_process_v3.py`: feature extraction.
- `llm_inference/generator.py`: vLLM generation methods.
- `llm_inference/prompts.py`: prompts for RCA and selector/refiner.
- `llm_inference/selector_refiner.py`: Selector/Refiner implementation.

## Before changing behavior

Run:

```bash
python -m py_compile infer_by_index.py config.py \
  inference/common.py inference/selection.py inference/tree.py inference/llm.py \
  utils/public_functions.py \
  rule_inferencer/data_process_v3.py \
  llm_inference/generator.py \
  llm_inference/prompts.py \
  llm_inference/selector_refiner.py
```
