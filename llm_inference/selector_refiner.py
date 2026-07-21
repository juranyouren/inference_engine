# llm_inference/selector_refiner.py
# -*- coding: utf-8 -*-

import os
import sys
import re
import json
from pathlib import Path
from typing import Any, Dict, List

os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.public_functions import vllm_invoke, save_json, save_jsonl
from llm_inference.prompts import ROLE_PROMPTS


def _visible_device_ids() -> List[str]:
    visible_devices = os.getenv("ASCEND_RT_VISIBLE_DEVICES", "")
    device_ids = [item.strip() for item in visible_devices.split(",") if item.strip()]
    if not device_ids:
        raise RuntimeError(
            "缺少环境变量 ASCEND_RT_VISIBLE_DEVICES。"
            "请在入口脚本中设置，或运行前 export ASCEND_RT_VISIBLE_DEVICES=0"
        )
    return device_ids


class LLMEngine:
    _instance = None

    def __init__(self, model_path: str, gpu_memory_utilization: float = 0.9, max_model_len: int = 16384):
        from vllm import LLM

        self.model_path = model_path
        self.max_model_len = max_model_len
        self.llm = LLM(
            model=model_path,
            tensor_parallel_size=len(_visible_device_ids()),
            trust_remote_code=True,
            gpu_memory_utilization=gpu_memory_utilization,
            max_model_len=max_model_len,
        )
        print(f"[LLMEngine] initialized: {model_path}")

    def fit_prompt(
        self,
        prompt: str,
        max_output_tokens: int,
        safety_tokens: int,
    ):
        tokenizer = self.llm.get_tokenizer()
        token_ids = tokenizer.encode(prompt, add_special_tokens=False)
        max_input_tokens = self.max_model_len - max_output_tokens - safety_tokens
        if max_input_tokens <= 0:
            raise ValueError(
                "Selector/Refiner token 预算无效: "
                f"max_model_len={self.max_model_len}, output={max_output_tokens}, "
                f"safety={safety_tokens}"
            )
        original_tokens = len(token_ids)
        truncated = original_tokens > max_input_tokens
        if truncated:
            marker = "\n\n[中间过长内容已截断]\n\n"
            marker_ids = tokenizer.encode(marker, add_special_tokens=False)
            content_budget = max_input_tokens - len(marker_ids)
            head_count = max(1, content_budget // 2)
            tail_count = max(1, content_budget - head_count)
            token_ids = token_ids[:head_count] + marker_ids + token_ids[-tail_count:]
            prompt = tokenizer.decode(token_ids, skip_special_tokens=True)
        return prompt, {
            "original_tokens": original_tokens,
            "final_tokens": len(token_ids),
            "max_input_tokens": max_input_tokens,
            "truncated": truncated,
        }

    @classmethod
    def get_instance(cls, model_path: str, gpu_memory_utilization: float = 0.9, max_model_len: int = 16384):
        if cls._instance is None or cls._instance.model_path != model_path:
            cls._instance = cls(model_path, gpu_memory_utilization, max_model_len)
        return cls._instance


class RCARole:
    def __init__(self, llm_engine: LLMEngine, role: str):
        self.engine = llm_engine
        self.llm = llm_engine.llm
        self.role = role
        if role not in ROLE_PROMPTS:
            raise KeyError(f"ROLE_PROMPTS 中找不到 role={role}")
        self.prompt = ROLE_PROMPTS[role]

    def _fit_inputs(self, inputs: List[str], output_dir: str, max_tokens: int):
        import config

        safety_tokens = getattr(
            config,
            "SELECTOR_REFINER_PROMPT_SAFETY_TOKENS",
            512,
        )
        fitted_inputs = []
        stats = []
        for idx, prompt in enumerate(inputs):
            fitted, prompt_stats = self.engine.fit_prompt(
                prompt,
                max_output_tokens=max_tokens,
                safety_tokens=safety_tokens,
            )
            prompt_stats["input_index"] = idx
            fitted_inputs.append(fitted)
            stats.append(prompt_stats)
            if prompt_stats["truncated"]:
                print(
                    f"[{self.role}] prompt 截断: "
                    f"{prompt_stats['original_tokens']} -> {prompt_stats['final_tokens']} tokens"
                )
        os.makedirs(output_dir, exist_ok=True)
        save_json(stats, os.path.join(output_dir, "prompt_stats.json"))
        return fitted_inputs

    @staticmethod
    def _parse_response(text: str) -> Dict[str, Any]:
        cot = text
        if "</think>" in text:
            cot = text.split("</think>", 1)[0]

        json_blocks = re.findall(r"```json\s*(.*?)\s*```", text, flags=re.S | re.I)
        if json_blocks:
            raw_json = json_blocks[-1]
        else:
            # 兜底：从第一个 { 到最后一个 } 提取
            start = text.find("{")
            end = text.rfind("}")
            raw_json = text[start:end + 1] if start >= 0 and end > start else ""

        parsed = None
        if raw_json:
            try:
                parsed = json.loads(raw_json)
            except Exception:
                parsed = None

        return {
            "cot": cot,
            "json": parsed,
            "raw": text,
        }

    @staticmethod
    def _save(records: List[Any], output_dir: str):
        os.makedirs(output_dir, exist_ok=True)
        save_json(records, os.path.join(output_dir, "records.json"))
        save_jsonl(records, os.path.join(output_dir, "res.jsonl"))


class Selector(RCARole):
    def select(self, case_list: List[Dict[str, Any]], output_dir: str, batch: int = 1):
        from vllm import SamplingParams
        import config

        max_tokens = getattr(config, "SELECTOR_REFINER_MAX_OUTPUT_TOKENS", 4096)

        inputs = []
        for case in case_list:
            inputs.append(self.prompt.format(
                alarm_type=case.get("alarm_type", ""),
                semantic_labels=json.dumps(case.get("semantic_labels", {}), ensure_ascii=False, indent=2),
                alarm_time=case.get("alarm_time", ""),
                sop=case.get("sop", ""),
                root_cause_candidates=json.dumps(case.get("root_cause_candidates", []), ensure_ascii=False, indent=2),
            ))

        inputs = self._fit_inputs(inputs, output_dir, max_tokens)
        responses = vllm_invoke(
            self.llm,
            inputs=inputs,
            sampling_params=SamplingParams(temperature=0.8, top_p=0.9, max_tokens=max_tokens, n=2),
            batch_size=batch,
            prompt_output_paths=[
                os.path.join(output_dir, f"prompt_{idx}.txt")
                for idx in range(len(inputs))
            ],
        )

        records = []
        for resp in responses:
            if isinstance(resp, list):
                records.append([self._parse_response(x) for x in resp])
            else:
                records.append(self._parse_response(resp))

        self._save(records, output_dir)
        return records


class Refiner(RCARole):
    def refine(
        self,
        case_list: List[Dict[str, Any]],
        output_dir: str,
        batch: int = 1,
        selection: Dict[str, Any] = None,
        summary: Dict[str, Any] = None,
    ):
        from vllm import SamplingParams
        import config

        max_tokens = getattr(config, "SELECTOR_REFINER_MAX_OUTPUT_TOKENS", 4096)

        inputs = []
        for case in case_list:
            inputs.append(self.prompt.format(
                alarm_type=case.get("alarm_type", ""),
                semantic_labels=json.dumps(case.get("semantic_labels", {}), ensure_ascii=False, indent=2),
                alarm_time=case.get("alarm_time", ""),
                sop=case.get("sop", ""),
                root_cause_candidates=json.dumps(case.get("root_cause_candidates", []), ensure_ascii=False, indent=2),
                previous_selection=json.dumps(selection or {}, ensure_ascii=False, indent=2),
                validcase_summary=json.dumps(summary or {}, ensure_ascii=False, indent=2),
            ))

        inputs = self._fit_inputs(inputs, output_dir, max_tokens)
        responses = vllm_invoke(
            self.llm,
            inputs=inputs,
            sampling_params=SamplingParams(temperature=0.8, top_p=0.9, max_tokens=max_tokens, n=3),
            batch_size=batch,
            prompt_output_paths=[
                os.path.join(output_dir, f"prompt_{idx}.txt")
                for idx in range(len(inputs))
            ],
        )

        records = []
        for resp in responses:
            if isinstance(resp, list):
                records.append([self._parse_response(x) for x in resp])
            else:
                records.append(self._parse_response(resp))

        self._save(records, output_dir)
        return records
