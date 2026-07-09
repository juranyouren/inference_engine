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


class LLMEngine:
    _instance = None

    def __init__(self, model_path: str, gpu_memory_utilization: float = 0.9, max_model_len: int = 16384):
        from vllm import LLM

        self.model_path = model_path
        self.llm = LLM(
            model=model_path,
            tensor_parallel_size=len(os.environ["ASCEND_RT_VISIBLE_DEVICES"].split(",")),
            trust_remote_code=True,
            gpu_memory_utilization=gpu_memory_utilization,
            max_model_len=max_model_len,
        )
        print(f"[LLMEngine] initialized: {model_path}")

    @classmethod
    def get_instance(cls, model_path: str, gpu_memory_utilization: float = 0.9, max_model_len: int = 16384):
        if cls._instance is None or cls._instance.model_path != model_path:
            cls._instance = cls(model_path, gpu_memory_utilization, max_model_len)
        return cls._instance


class RCARole:
    def __init__(self, llm_engine: LLMEngine, role: str):
        self.llm = llm_engine.llm
        self.role = role
        if role not in ROLE_PROMPTS:
            raise KeyError(f"ROLE_PROMPTS 中找不到 role={role}")
        self.prompt = ROLE_PROMPTS[role]

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

        inputs = []
        for case in case_list:
            inputs.append(self.prompt.format(
                alarm_type=case.get("alarm_type", ""),
                semantic_labels=json.dumps(case.get("semantic_labels", {}), ensure_ascii=False, indent=2),
                alarm_time=case.get("alarm_time", ""),
                sop=case.get("sop", ""),
                root_cause_candidates=json.dumps(case.get("root_cause_candidates", []), ensure_ascii=False, indent=2),
            ))

        responses = vllm_invoke(
            self.llm,
            inputs=inputs,
            sampling_params=SamplingParams(temperature=0.8, top_p=0.9, max_tokens=4096, n=2),
            batch_size=batch,
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

        responses = vllm_invoke(
            self.llm,
            inputs=inputs,
            sampling_params=SamplingParams(temperature=0.8, top_p=0.9, max_tokens=4096, n=3),
            batch_size=batch,
        )

        records = []
        for resp in responses:
            if isinstance(resp, list):
                records.append([self._parse_response(x) for x in resp])
            else:
                records.append(self._parse_response(resp))

        self._save(records, output_dir)
        return records
