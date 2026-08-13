"""只包含两个接口所需的生成方法。"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from .invoke import vllm_invoke
from .prompts import (
    ROOT_CAUSE_COMPETITION_REASONER,
    ROOT_CAUSE_META_THINKER,
    ROOT_CAUSE_REASONER,
    ROOT_CAUSE_VERIFIER,
)


class RCAGenerator:
    """使用已经加载完成的 LLMEngine 执行生成。"""

    def __init__(
        self,
        engine: Any,
        *,
        batch_size: int = 8,
        max_tokens: int = 4096,
        prompt_safety_tokens: int = 512,
        sampling_params_factory: Optional[Callable[..., Any]] = None,
    ) -> None:
        if engine is None or not hasattr(engine, "llm"):
            raise TypeError("engine 必须是包含 llm 属性的 LLMEngine 实例")
        if batch_size <= 0:
            raise ValueError("batch_size 必须大于 0")

        if sampling_params_factory is None:
            from vllm import SamplingParams

            sampling_params_factory = SamplingParams

        self.engine = engine
        self.llm = engine.llm
        self.batch_size = batch_size
        self.max_tokens = max_tokens
        self.prompt_safety_tokens = prompt_safety_tokens
        self._sampling_params_factory = sampling_params_factory

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
