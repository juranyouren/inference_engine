"""只包含两个接口所需的模型加载和生成方法。"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

# 必须在导入 vLLM / torch_npu 之前设置。
os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")

from .invoke import vllm_invoke
from .prompts import (
    ROOT_CAUSE_COMPETITION_REASONER,
    ROOT_CAUSE_META_THINKER,
    ROOT_CAUSE_REASONER,
    ROOT_CAUSE_VERIFIER,
)


def _visible_device_count() -> int:
    devices = os.environ.get("ASCEND_RT_VISIBLE_DEVICES", "")
    visible = [item.strip() for item in devices.split(",") if item.strip()]
    return len(visible) or 1


class RCAGenerator:
    """构造时加载一次模型，生成方法只复用已有模型。"""

    def __init__(
        self,
        model_path: str,
        *,
        batch_size: int = 8,
        tensor_parallel_size: Optional[int] = None,
        gpu_memory_utilization: float = 0.9,
        max_model_len: int = 16384,
        max_tokens: int = 4096,
        prompt_safety_tokens: int = 512,
        model_options: Optional[Dict[str, Any]] = None,
        llm_factory: Optional[Callable[..., Any]] = None,
        sampling_params_factory: Optional[Callable[..., Any]] = None,
    ) -> None:
        if not model_path:
            raise ValueError("model_path 不能为空")
        if batch_size <= 0:
            raise ValueError("batch_size 必须大于 0")

        if llm_factory is None or sampling_params_factory is None:
            from vllm import LLM, SamplingParams

            llm_factory = llm_factory or LLM
            sampling_params_factory = sampling_params_factory or SamplingParams

        options = dict(model_options or {})
        if "model" in options:
            raise ValueError("model_options 中不能覆盖 model")
        options.update({
            "model": model_path,
            "tensor_parallel_size": (
                tensor_parallel_size
                if tensor_parallel_size is not None
                else _visible_device_count()
            ),
            "trust_remote_code": True,
            "gpu_memory_utilization": gpu_memory_utilization,
            "max_model_len": max_model_len,
        })

        self.model_path = model_path
        self.batch_size = batch_size
        self.max_tokens = max_tokens
        self.prompt_safety_tokens = prompt_safety_tokens
        self._sampling_params_factory = sampling_params_factory

        # 唯一的模型加载点。infer/generate 方法不会再次执行该工厂。
        self.llm = llm_factory(**options)
        print(f"LLM模型已初始化: {model_path}")

    @staticmethod
    def _filtered_semantic_labels(
        alarm_type: str,
        semantic_labels: Dict[str, Any],
    ) -> Dict[str, Any]:
        if alarm_type == "入方向带宽利用率过高":
            filtered = dict(semantic_labels)
            if "anomaly_logs" in filtered:
                filtered["anomaly_logs"] = {}
            return filtered
        return semantic_labels

    @staticmethod
    def _json(value: Any) -> str:
        return json.dumps(value, ensure_ascii=False, indent=2)

    def _params(self, *, temperature: float, top_p: float):
        return self._sampling_params_factory(
            temperature=temperature,
            top_p=top_p,
            max_tokens=self.max_tokens,
        )

    def _invoke(
        self,
        prompts: List[str],
        *,
        temperature: float,
        top_p: float,
        prompt_paths: Optional[List[Path]] = None,
    ) -> List[str]:
        return vllm_invoke(
            self.llm,
            inputs=prompts,
            sampling_params=self._params(
                temperature=temperature,
                top_p=top_p,
            ),
            batch_size=self.batch_size,
            prompt_safety_tokens=self.prompt_safety_tokens,
            prompt_output_paths=prompt_paths,
        )

    def generate_competition_reasoners(
        self,
        alarm_data_list: List[Dict[str, Any]],
        output_dirs: List[Path],
    ) -> List[List[str]]:
        configs = (
            ("Reasoner1", 0.1, 0.9),
            ("Reasoner2", 0.5, 0.95),
            ("Reasoner3", 0.8, 1.0),
        )
        responses_by_case: List[List[str]] = [
            [] for _item in alarm_data_list
        ]

        for name, temperature, top_p in configs:
            prompts = [
                ROOT_CAUSE_COMPETITION_REASONER.format(
                    alarm_type=item["alarm_type"],
                    semantic_labels=self._json(
                        self._filtered_semantic_labels(
                            item["alarm_type"], item["semantic_labels"],
                        )
                    ),
                    alarm_time=item["alarm_time"],
                    sop=item["sop"],
                    root_cause_candidates=self._json(
                        item["root_cause_candidates"]
                    ),
                )
                for item in alarm_data_list
            ]
            responses = self._invoke(
                prompts,
                temperature=temperature,
                top_p=top_p,
                prompt_paths=[
                    output_dir / f"prompt_{name}.txt"
                    for output_dir in output_dirs
                ],
            )
            for index, response in enumerate(responses):
                responses_by_case[index].append(response)

        return responses_by_case

    def generate_competition_verifier(
        self,
        alarm_data_list: List[Dict[str, Any]],
        output_dirs: List[Path],
    ) -> List[str]:
        prompts = [
            ROOT_CAUSE_VERIFIER.format(
                alarm_type=item["alarm_type"],
                semantic_labels=self._json(
                    self._filtered_semantic_labels(
                        item["alarm_type"], item["semantic_labels"],
                    )
                ),
                sop=item["sop"],
                root_cause_candidates=self._json(
                    item["root_cause_candidates"]
                ),
                reasoner_outputs=item["reasoner_outputs"],
            )
            for item in alarm_data_list
        ]
        return self._invoke(
            prompts,
            temperature=0.5,
            top_p=0.9,
            prompt_paths=[output_dir / "prompt.txt" for output_dir in output_dirs],
        )

    def generate_cooperation_meta(
        self,
        alarm_data_list: List[Dict[str, Any]],
        output_dirs: List[Path],
    ) -> List[str]:
        prompts = [
            ROOT_CAUSE_META_THINKER.format(
                Last_round_outputs=item["Last_round_outputs"],
                alarm_type=item["alarm_type"],
                semantic_labels_key=self._json(item["semantic_labels_key"]),
                sop=item["sop"],
                root_cause_candidates=self._json(
                    item["root_cause_candidates"]
                ),
            )
            for item in alarm_data_list
        ]
        return self._invoke(
            prompts,
            temperature=0.5,
            top_p=0.9,
            prompt_paths=[output_dir / "prompt.txt" for output_dir in output_dirs],
        )

    def generate_cooperation_reasoner(
        self,
        alarm_data_list: List[Dict[str, Any]],
        output_dirs: List[Path],
    ) -> List[str]:
        prompts = [
            ROOT_CAUSE_REASONER.format(
                meta_output=item["meta_output"],
                alarm_type=item["alarm_type"],
                alarm_time=item["alarm_time"],
                semantic_labels=self._json(
                    self._filtered_semantic_labels(
                        item["alarm_type"], item["semantic_labels"],
                    )
                ),
                root_cause_candidates=self._json(
                    item["root_cause_candidates"]
                ),
            )
            for item in alarm_data_list
        ]
        return self._invoke(
            prompts,
            temperature=0.5,
            top_p=0.9,
            prompt_paths=[output_dir / "prompt.txt" for output_dir in output_dirs],
        )
