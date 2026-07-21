# utils/public_functions.py
# -*- coding: utf-8 -*-

import os
import json
import glob
import pickle
import logging
import re
from copy import deepcopy
from typing import Any, Dict, List, Optional

import config

try:
    from tqdm import tqdm
except Exception:  # pragma: no cover
    def tqdm(x, *args, **kwargs):
        return x


# ============================================================
# 基础 IO
# ============================================================

def ensure_dir(path: str):
    if path:
        os.makedirs(path, exist_ok=True)


def save_txt(text: str, path: str):
    ensure_dir(os.path.dirname(path))
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)


def load_txt(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def _truncate_raw_response(text: str, max_chars: Optional[int] = None) -> str:
    """截断过长的 LLM 原始输出，防止下一轮 prompt 撑爆 token 上限。

    只截日志，不截其他 prompt 内容。
    max_chars 默认值根据 max_model_len=16384，输出预留 4096 token，
    safety=512 token。保守给其他 prompt 内容预留 ~5000 token，
    剩余 ~6000 token 给历史日志，按中文约 1.5 char/token 折算为 9000 字符。
    """
    if max_chars is None:
        max_chars = getattr(config, "RAW_RESPONSE_MAX_CHARS", 9000)
    if len(text) <= max_chars:
        return text
    half = max_chars // 2
    return (
        text[:half]
        + "\n\n[...日志过长，中间部分已截断...]\n\n"
        + text[-half:]
    )


def _load_txt_truncated(path: str, max_chars: Optional[int] = None) -> str:
    """读取文本并在过长时截断，用于加载上一轮 LLM 原始输出。"""
    text = load_txt(path)
    if not text:
        return ""
    return _truncate_raw_response(text, max_chars)


def save_pkl(data: Any, path: str):
    ensure_dir(os.path.dirname(path))
    with open(path, "wb") as f:
        pickle.dump(data, f)


def load_pkl(path: str) -> Any:
    with open(path, "rb") as f:
        return pickle.load(f)


def save_json(data: Any, path: str):
    ensure_dir(os.path.dirname(path))
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_jsonl(data: List[Any], path: str, append: bool = False):
    ensure_dir(os.path.dirname(path))
    mode = "a" if append else "w"
    with open(path, mode, encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def load_jsonl(path: str) -> List[Any]:
    lines = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                lines.append(json.loads(line))
    return lines


def get_logger(log_file_path: str):
    logger = logging.getLogger(log_file_path)
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()
    ensure_dir(os.path.dirname(log_file_path))
    file_handler = logging.FileHandler(log_file_path, encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    logger.addHandler(file_handler)
    return logger


def str_idx_in_list(s: str, lst: List[str]):
    try:
        return lst.index(s)
    except Exception:
        return None


def mean(values: List[float]):
    return round(sum(values) / len(values), 4) if values else 0.0


# ============================================================
# vLLM 调用
# ============================================================

def _get_vllm_max_model_len(llm) -> int:
    engine = getattr(llm, "llm_engine", None)
    for owner in (
        getattr(engine, "model_config", None),
        getattr(engine, "scheduler_config", None),
        engine,
        llm,
    ):
        value = getattr(owner, "max_model_len", None) if owner is not None else None
        if value:
            return int(value)
    return 16384


def _truncate_anomaly_logs_to_budget(
    tokenizer,
    prompt: str,
    max_input_tokens: int,
):
    """Replace an oversized anomaly_logs JSON value with a valid head/tail summary."""
    candidates = []
    for key_match in re.finditer(r'"anomaly_logs"\s*:\s*', prompt):
        value_start = key_match.end()
        try:
            value, value_length = json.JSONDecoder().raw_decode(
                prompt[value_start:]
            )
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict) and value.get("__truncated__") is True:
            continue
        value_end = value_start + value_length
        raw_logs = prompt[value_start:value_end]
        log_token_ids = tokenizer.encode(raw_logs, add_special_tokens=False)
        if log_token_ids:
            candidates.append((
                len(log_token_ids),
                value_start,
                value_end,
                log_token_ids,
            ))

    if not candidates:
        return prompt, None
    _size, value_start, value_end, log_token_ids = max(candidates)

    def build_candidate(keep_tokens: int) -> str:
        head_count = keep_tokens // 2
        tail_count = keep_tokens - head_count
        head = tokenizer.decode(
            log_token_ids[:head_count],
            skip_special_tokens=True,
        ) if head_count else ""
        tail = tokenizer.decode(
            log_token_ids[-tail_count:],
            skip_special_tokens=True,
        ) if tail_count else ""
        replacement = json.dumps({
            "__truncated__": True,
            "__note__": "anomaly_logs 超出 Prompt token 预算，仅保留首尾内容",
            "__original_token_count__": len(log_token_ids),
            "__head__": head,
            "__tail__": tail,
        }, ensure_ascii=False, indent=2)
        return prompt[:value_start] + replacement + prompt[value_end:]

    # Find the largest retained log slice that keeps the complete prompt in budget.
    best_prompt = build_candidate(0)
    best_tokens = len(tokenizer.encode(best_prompt, add_special_tokens=False))
    if best_tokens > max_input_tokens:
        return best_prompt, {
            "original_log_tokens": len(log_token_ids),
            "retained_log_tokens": 0,
        }

    low = 1
    high = len(log_token_ids) - 1
    retained = 0
    while low <= high:
        middle = (low + high) // 2
        candidate = build_candidate(middle)
        candidate_tokens = len(tokenizer.encode(candidate, add_special_tokens=False))
        if candidate_tokens <= max_input_tokens:
            best_prompt = candidate
            retained = middle
            low = middle + 1
        else:
            high = middle - 1

    return best_prompt, {
        "original_log_tokens": len(log_token_ids),
        "retained_log_tokens": retained,
    }


def fit_prompt_to_token_budget(
    tokenizer,
    prompt: str,
    max_input_tokens: int,
):
    """Fit one prompt, truncating anomaly_logs before whole-prompt fallback."""
    token_ids = tokenizer.encode(prompt, add_special_tokens=False)
    original_tokens = len(token_ids)
    strategy = "none"
    log_stats = None

    if original_tokens > max_input_tokens:
        truncated_sections = []
        while len(token_ids) > max_input_tokens:
            prompt, section_stats = _truncate_anomaly_logs_to_budget(
                tokenizer,
                prompt,
                max_input_tokens,
            )
            if section_stats is None:
                break
            truncated_sections.append(section_stats)
            token_ids = tokenizer.encode(prompt, add_special_tokens=False)
        if truncated_sections:
            strategy = "anomaly_logs"
            log_stats = {
                "truncated_log_sections": len(truncated_sections),
                "original_log_tokens": sum(
                    item["original_log_tokens"] for item in truncated_sections
                ),
                "retained_log_tokens": sum(
                    item["retained_log_tokens"] for item in truncated_sections
                ),
            }

    if len(token_ids) > max_input_tokens:
        marker_ids = tokenizer.encode(
            "\n\n[中间过长内容已截断]\n\n",
            add_special_tokens=False,
        )
        content_budget = max_input_tokens - len(marker_ids)
        if content_budget <= 1:
            raise ValueError("vLLM 输入预算不足以容纳截断标记")
        head_count = content_budget // 2
        tail_count = content_budget - head_count
        token_ids = token_ids[:head_count] + marker_ids + token_ids[-tail_count:]
        prompt = tokenizer.decode(token_ids, skip_special_tokens=True)
        strategy = "whole_prompt_fallback"

    stats = {
        "original_tokens": original_tokens,
        "final_tokens": len(token_ids),
        "max_input_tokens": max_input_tokens,
        "truncated": original_tokens > max_input_tokens,
        "truncation_strategy": strategy,
    }
    if log_stats is not None:
        stats.update(log_stats)
    return prompt, stats


def fit_vllm_inputs(llm, inputs: List[str], sampling_params):
    """Limit every prompt to model_len - requested output - chat safety."""
    tokenizer = llm.get_tokenizer()
    max_model_len = _get_vllm_max_model_len(llm)
    max_output_tokens = int(getattr(sampling_params, "max_tokens", 4096) or 4096)
    safety_tokens = int(getattr(config, "LLM_PROMPT_SAFETY_TOKENS", 512))
    max_input_tokens = max_model_len - max_output_tokens - safety_tokens
    if max_input_tokens <= 0:
        raise ValueError(
            "vLLM token 预算无效: "
            f"max_model_len={max_model_len}, output={max_output_tokens}, "
            f"safety={safety_tokens}"
        )

    fitted_inputs = []
    stats = []
    for index, prompt in enumerate(inputs):
        prompt, prompt_stats = fit_prompt_to_token_budget(
            tokenizer,
            prompt,
            max_input_tokens,
        )
        prompt_stats["input_index"] = index
        if prompt_stats["truncated"]:
            print(
                f"[vllm_invoke] prompt[{index}] 截断策略="
                f"{prompt_stats['truncation_strategy']}: "
                f"{prompt_stats['original_tokens']} -> "
                f"{prompt_stats['final_tokens']} tokens；"
                f"模型上限={max_model_len}，输出预留={max_output_tokens}"
            )
        fitted_inputs.append(prompt)
        stats.append(prompt_stats)
    return fitted_inputs, stats

def vllm_invoke(
    llm,
    inputs: List[str],
    sampling_params,
    lora_path=None,
    batch_size: int = 1,
    prompt_output_paths: Optional[List[str]] = None,
):
    """
    统一 vLLM chat 调用。

    sampling_params.n > 1 时，返回 list[list[str]]；否则返回 list[str]。
    prompt_output_paths 用于保存经过 token 预算处理、实际送入模型的 prompt。
    """
    all_responses = []
    n = getattr(sampling_params, "n", 1)
    inputs, _prompt_stats = fit_vllm_inputs(llm, inputs, sampling_params)
    if prompt_output_paths is not None:
        if len(prompt_output_paths) != len(inputs):
            raise ValueError(
                "prompt 输出路径数量必须与输入数量一致: "
                f"paths={len(prompt_output_paths)}, inputs={len(inputs)}"
            )
        for prompt, path in zip(inputs, prompt_output_paths):
            save_txt(prompt, path)

    if lora_path:
        print("insert lora adapter", lora_path)

    for i in tqdm(range(0, len(inputs), batch_size)):
        batch_inputs = inputs[i:i + batch_size]
        applied_prompts = [[{"role": "user", "content": prompt}] for prompt in batch_inputs]

        outputs_w_prompts = llm.chat(
            applied_prompts,
            sampling_params,
            use_tqdm=False,
        )

        if n > 1:
            for item in outputs_w_prompts:
                all_responses.append([out.text for out in item.outputs])
        else:
            all_responses.extend([item.outputs[0].text for item in outputs_w_prompts])

    return all_responses


# ============================================================
# label/case 数据加载
# ============================================================

def _get_root_cause(label_data: Dict[str, Any]):
    rc = label_data.get("root_cause", "")
    if isinstance(rc, dict):
        return rc.get("category", "") or rc.get("name", "") or rc.get("root_cause", "")
    return rc


def _get_root_cause_candidates(label_data: Dict[str, Any]):
    candidates = label_data.get("root_cause_candidates", [])
    if candidates:
        return candidates

    rc = _get_root_cause(label_data)
    if rc:
        return [rc]
    return []


def _get_semantic_labels(label_data: Dict[str, Any]) -> Dict[str, Any]:
    semantic_labels = label_data.get("semantic_labels", {})
    return semantic_labels if isinstance(semantic_labels, dict) else {}


def _get_sop(label_data: Dict[str, Any]) -> str:
    semantic_labels = _get_semantic_labels(label_data)
    sop = semantic_labels.get("sop", label_data.get("sop", ""))

    if sop is None:
        return ""
    if isinstance(sop, str):
        return sop.strip()
    if isinstance(sop, (dict, list)):
        return json.dumps(sop, ensure_ascii=False) if sop else ""
    return str(sop).strip()


def load_alarm_data(case_file_path: str, label_file_path: str, alarm_type: str) -> Dict[str, Any]:
    """
    加载 rule/tree 使用的数据。

    兼容只有 label 文件的小测试：case_file_path 可以等于 label_file_path。
    """
    # case_data 目前不强依赖，但保留读取动作以便发现坏文件。
    _case_data = load_json(case_file_path)
    label_data = load_json(label_file_path)

    alarm_data = {
        "alarm_type": alarm_type,
        "semantic_labels": _get_semantic_labels(label_data),
        "alarm_time": label_data.get("alarm_time", ""),
        "sop": _get_sop(label_data),
        "root_cause_candidates": _get_root_cause_candidates(label_data),
        "root_cause": _get_root_cause(label_data),
    }

    return alarm_data


def load_alarm_data_reasoner_competition(case_file_path: str, label_file_path: str, alarm_type: str) -> Dict[str, Any]:
    """
    Competition 第一阶段使用。
    generator.generate_rca_analysis_competition_batch 需要：
        alarm_type, semantic_labels, alarm_time, sop, root_cause_candidates
    """
    data = load_alarm_data(case_file_path, label_file_path, alarm_type)
    return {
        "alarm_type": data["alarm_type"],
        "semantic_labels": data["semantic_labels"],
        "alarm_time": data.get("alarm_time", ""),
        "sop": data.get("sop", ""),
        "root_cause_candidates": data.get("root_cause_candidates", []),
    }


def load_alarm_data_verifier(label_file_path: str, reasoner_output_file: str, alarm_type: str) -> Dict[str, Any]:
    """
    Competition 第二阶段 Verifier 使用。
    generator.generate_rca_analysis_verifier_batch 需要：
        alarm_type, semantic_labels, sop, root_cause_candidates, reasoner_outputs
    """
    label_data = load_json(label_file_path)
    reasoner_outputs = _load_txt_truncated(reasoner_output_file)

    return {
        "alarm_type": alarm_type,
        "semantic_labels": _get_semantic_labels(label_data),
        "alarm_time": label_data.get("alarm_time", ""),
        "sop": _get_sop(label_data),
        "root_cause_candidates": _get_root_cause_candidates(label_data),
        "reasoner_outputs": reasoner_outputs,
    }


def load_alarm_data_meta(
    label_file_path: str,
    alarm_type: str,
    last_meta_output: Optional[str] = None,
    last_reasoner_output: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Cooperation 的 Meta 阶段使用。
    generator.generate_rca_analysis_meta 需要：
        Last_round_outputs, alarm_type, semantic_labels_key, sop, root_cause_candidates
    """
    label_data = load_json(label_file_path)
    semantic_labels = _get_semantic_labels(label_data)

    last_outputs = []
    if last_meta_output and os.path.exists(last_meta_output):
        last_outputs.append("[Last Meta]\n" + _load_txt_truncated(last_meta_output))
    if last_reasoner_output and os.path.exists(last_reasoner_output):
        last_outputs.append("[Last Reasoner]\n" + _load_txt_truncated(last_reasoner_output))

    return {
        "Last_round_outputs": "\n\n".join(last_outputs) if last_outputs else "无上一轮输出。",
        "alarm_type": alarm_type,
        # 原代码字段名叫 semantic_labels_key，这里直接传完整 semantic_labels，更稳。
        "semantic_labels_key": semantic_labels,
        "sop": _get_sop(label_data),
        "root_cause_candidates": _get_root_cause_candidates(label_data),
    }


def load_alarm_data_reasoner_cooperation(label_file_path: str, meta_file: str, alarm_type: str) -> Dict[str, Any]:
    """
    Cooperation 的 Reasoner 阶段使用。
    generator.generate_rca_analysis_reasoner 需要：
        meta_output, alarm_type, alarm_time, semantic_labels, root_cause_candidates
    """
    label_data = load_json(label_file_path)
    return {
        "meta_output": load_txt(meta_file),
        "alarm_type": alarm_type,
        "alarm_time": label_data.get("alarm_time", ""),
        "semantic_labels": _get_semantic_labels(label_data),
        "root_cause_candidates": _get_root_cause_candidates(label_data),
    }


def load_alarm_template(alarm_type: str, template_dir: Optional[str] = None, data_dir: Optional[str] = None) -> Dict[str, Any]:
    """
    Selector/Refiner 使用的模板数据。

    优先级：
        1. template_dir/{alarm_type}_*_label_*.json
        2. data_dir/{alarm_type}_*_label_*.json
        3. template_dir 下任意包含 alarm_type 的 json
    """
    candidates = []

    if template_dir:
        candidates.extend(sorted(glob.glob(os.path.join(template_dir, f"{alarm_type}_*_label_*.json"))))
        candidates.extend(sorted(glob.glob(os.path.join(template_dir, f"*{alarm_type}*.json"))))

    if data_dir:
        candidates.extend(sorted(glob.glob(os.path.join(data_dir, f"{alarm_type}_*_label_*.json"))))
        candidates.extend(sorted(glob.glob(os.path.join(data_dir, "*.json"))))

    # 去重保持顺序
    seen = set()
    unique_candidates = []
    for p in candidates:
        if p not in seen:
            seen.add(p)
            unique_candidates.append(p)

    if not unique_candidates:
        raise FileNotFoundError(
            f"未找到 alarm_type={alarm_type} 的模板文件。template_dir={template_dir}, data_dir={data_dir}"
        )

    label_file_path = unique_candidates[0]
    label_data = load_json(label_file_path)

    semantic_labels = deepcopy(_get_semantic_labels(label_data))
    semantic_labels.setdefault("anomaly_logs", {})
    semantic_labels.setdefault("anomaly_kpi", {})

    return {
        "alarm_type": alarm_type,
        "semantic_labels": semantic_labels,
        "alarm_time": label_data.get("alarm_time", ""),
        "sop": _get_sop(label_data),
        "root_cause_candidates": _get_root_cause_candidates(label_data),
        "root_cause": _get_root_cause(label_data),
        "_template_path": label_file_path,
    }


class CaseDataLoader:
    """
    兼容旧代码的 CaseDataLoader。
    新流程不建议用 load_all_cases 做全量加载；infer_by_index.py 已实现按 index 读取。
    """
    def __init__(self, data_dir: str, alarm_type: str):
        self.data_dir = data_dir
        self.alarm_type = alarm_type
        self.all_cases = []
        self.all_indices = []
        self.blocks = []
        self.block_indices = []

    def load_all_cases(self, total_cases: int):
        for idx in range(total_cases):
            case_pattern = f"{self.data_dir}/{self.alarm_type}_{idx}_case_*.json"
            label_pattern = f"{self.data_dir}/{self.alarm_type}_{idx}_label_*.json"
            case_files = sorted(glob.glob(case_pattern))
            label_files = sorted(glob.glob(label_pattern))

            if not label_files:
                continue
            case_file = case_files[0] if case_files else label_files[0]
            label_file = label_files[0]

            try:
                case_data = load_alarm_data(case_file, label_file, self.alarm_type)
            except Exception:
                continue

            self.all_cases.append(case_data)
            self.all_indices.append(idx)

    def build_extractor_blocks(self, extractor_limit: int, cases_size: int):
        cur_cases, cur_indices = [], []
        for idx, case_data in zip(self.all_indices, self.all_cases):
            if idx >= extractor_limit:
                break
            cur_cases.append(case_data)
            cur_indices.append(idx)
            if len(cur_cases) == cases_size:
                self.blocks.append(cur_cases.copy())
                self.block_indices.append(cur_indices.copy())
                cur_cases, cur_indices = [], []
        if cur_cases:
            self.blocks.append(cur_cases)
            self.block_indices.append(cur_indices)

    def get_block(self, block_id: int):
        return self.blocks[block_id], self.block_indices[block_id]

    def get_last_cases(self, m: int):
        return self.all_cases[-m:], self.all_indices[-m:]

    def get_cases_by_id_range(self, start_id: int, end_id: int):
        selected_cases = []
        selected_indices = []
        for idx, case_data in zip(self.all_indices, self.all_cases):
            if start_id <= idx < end_id:
                selected_cases.append(case_data)
                selected_indices.append(idx)
        return selected_cases, selected_indices
