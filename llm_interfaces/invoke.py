"""接口包内部的 vLLM 调用和 prompt 长度控制。"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, List, Optional

from .io_utils import save_text

try:
    from tqdm import tqdm
except Exception:  # pragma: no cover
    def tqdm(items, *args, **kwargs):
        return items


def _get_vllm_max_model_len(llm: Any) -> int:
    engine = getattr(llm, "llm_engine", None)
    for owner in (
        getattr(engine, "model_config", None),
        getattr(engine, "scheduler_config", None),
        engine,
        llm,
    ):
        value = getattr(owner, "max_model_len", None) if owner else None
        if value:
            return int(value)
    return 16384


def _truncate_anomaly_logs_to_budget(
    tokenizer: Any,
    prompt: str,
    max_input_tokens: int,
):
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
                len(log_token_ids), value_start, value_end, log_token_ids,
            ))

    if not candidates:
        return prompt, None
    _size, value_start, value_end, log_token_ids = max(candidates)

    def build_candidate(keep_tokens: int) -> str:
        head_count = keep_tokens // 2
        tail_count = keep_tokens - head_count
        head = tokenizer.decode(
            log_token_ids[:head_count], skip_special_tokens=True,
        ) if head_count else ""
        tail = tokenizer.decode(
            log_token_ids[-tail_count:], skip_special_tokens=True,
        ) if tail_count else ""
        replacement = json.dumps({
            "__truncated__": True,
            "__note__": "anomaly_logs 超出 Prompt token 预算，仅保留首尾内容",
            "__original_token_count__": len(log_token_ids),
            "__head__": head,
            "__tail__": tail,
        }, ensure_ascii=False, indent=2)
        return prompt[:value_start] + replacement + prompt[value_end:]

    best_prompt = build_candidate(0)
    if len(tokenizer.encode(best_prompt, add_special_tokens=False)) > max_input_tokens:
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
        candidate_tokens = len(
            tokenizer.encode(candidate, add_special_tokens=False)
        )
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
    tokenizer: Any,
    prompt: str,
    max_input_tokens: int,
):
    token_ids = tokenizer.encode(prompt, add_special_tokens=False)
    original_tokens = len(token_ids)
    strategy = "none"

    while len(token_ids) > max_input_tokens:
        prompt, section_stats = _truncate_anomaly_logs_to_budget(
            tokenizer, prompt, max_input_tokens,
        )
        if section_stats is None:
            break
        strategy = "anomaly_logs"
        token_ids = tokenizer.encode(prompt, add_special_tokens=False)

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

    return prompt, {
        "original_tokens": original_tokens,
        "final_tokens": len(token_ids),
        "max_input_tokens": max_input_tokens,
        "truncated": original_tokens > max_input_tokens,
        "truncation_strategy": strategy,
    }


def fit_vllm_inputs(
    llm: Any,
    inputs: List[str],
    sampling_params: Any,
    prompt_safety_tokens: int,
):
    tokenizer = llm.get_tokenizer()
    max_model_len = _get_vllm_max_model_len(llm)
    max_output_tokens = int(
        getattr(sampling_params, "max_tokens", 4096) or 4096
    )
    max_input_tokens = (
        max_model_len - max_output_tokens - int(prompt_safety_tokens)
    )
    if max_input_tokens <= 0:
        raise ValueError(
            "vLLM token 预算无效: "
            f"max_model_len={max_model_len}, output={max_output_tokens}, "
            f"safety={prompt_safety_tokens}"
        )

    fitted_inputs = []
    stats = []
    for index, prompt in enumerate(inputs):
        fitted_prompt, prompt_stats = fit_prompt_to_token_budget(
            tokenizer, prompt, max_input_tokens,
        )
        prompt_stats["input_index"] = index
        if prompt_stats["truncated"]:
            print(
                f"[vllm_invoke] prompt[{index}] 截断策略="
                f"{prompt_stats['truncation_strategy']}: "
                f"{prompt_stats['original_tokens']} -> "
                f"{prompt_stats['final_tokens']} tokens"
            )
        fitted_inputs.append(fitted_prompt)
        stats.append(prompt_stats)
    return fitted_inputs, stats


def vllm_invoke(
    llm: Any,
    *,
    inputs: List[str],
    sampling_params: Any,
    batch_size: int,
    prompt_safety_tokens: int,
    prompt_output_paths: Optional[List[str | Path]] = None,
) -> List[str]:
    if not inputs:
        return []
    if batch_size <= 0:
        raise ValueError("batch_size 必须大于 0")

    fitted_inputs, _stats = fit_vllm_inputs(
        llm, inputs, sampling_params, prompt_safety_tokens,
    )
    if prompt_output_paths is not None:
        if len(prompt_output_paths) != len(fitted_inputs):
            raise ValueError("prompt 输出路径数量必须与输入数量一致")
        for prompt, path in zip(fitted_inputs, prompt_output_paths):
            save_text(prompt, path)

    responses: List[str] = []
    for start in tqdm(range(0, len(fitted_inputs), batch_size)):
        batch_inputs = fitted_inputs[start:start + batch_size]
        chat_prompts = [
            [{"role": "user", "content": prompt}]
            for prompt in batch_inputs
        ]
        outputs = llm.chat(
            chat_prompts,
            sampling_params,
            use_tqdm=False,
        )
        for output in outputs:
            if not getattr(output, "outputs", None):
                raise RuntimeError("vLLM 返回了空响应")
            responses.append(output.outputs[0].text)

    if len(responses) != len(inputs):
        raise RuntimeError(
            f"vLLM 响应数量不匹配: inputs={len(inputs)}, "
            f"responses={len(responses)}"
        )
    return responses
