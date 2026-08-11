"""Competition 模式的可复用 Python 接口。"""

from __future__ import annotations

import threading
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from .data_adapter import InferenceCase, normalize_case
from .generator import RCAGenerator
from .io_utils import (
    case_output_dir,
    category_output_dir,
    ensure_dir,
    iter_batches,
    save_batch_results,
    save_category_csvs,
    save_case_result,
    save_text,
)
from .result_parser import build_prediction_fields


class CompetitionInference:
    """Competition（3 个 Reasoner + Verifier）推理接口。

    模型在本类初始化时加载一次。后续可以多次调用 infer，期间不会
    重新加载模型。调用方只应实例化当前需要的推理模式。
    """

    def __init__(
        self,
        model_path: str,
        *,
        alarm_type: Optional[str] = None,
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
        self.model_path = model_path
        self.alarm_type = alarm_type
        self.batch_size = batch_size
        self._infer_lock = threading.Lock()

        # 构造 CompetitionInference 时加载；infer 中不再加载。
        self.generator = RCAGenerator(
            model_path,
            batch_size=batch_size,
            tensor_parallel_size=tensor_parallel_size,
            gpu_memory_utilization=gpu_memory_utilization,
            max_model_len=max_model_len,
            max_tokens=max_tokens,
            prompt_safety_tokens=prompt_safety_tokens,
            model_options=model_options,
            llm_factory=llm_factory,
            sampling_params_factory=sampling_params_factory,
        )
        self.llm = self.generator.llm

    @staticmethod
    def _alarm_data(case: InferenceCase) -> Dict[str, Any]:
        return {
            "alarm_type": case.alarm_type,
            "alarm_time": case.alarm_time,
            "semantic_labels": case.semantic_labels,
            "sop": case.sop,
            "root_cause_candidates": case.root_cause_candidates,
        }

    @staticmethod
    def _merge_reasoner_outputs(responses: List[str]) -> str:
        sections = []
        for index, response in enumerate(responses, start=1):
            sections.append(
                f"=== Reasoner{index} ===\n{response}\n\n{'=' * 50}"
            )
        return "\n\n".join(sections)

    @staticmethod
    def _new_result(
        *,
        input_index: int,
        case_id: str,
        output_dir: Path,
    ) -> Dict[str, Any]:
        return {
            "input_index": input_index,
            "case_idx": input_index,
            "case_id": case_id,
            "mode": "competition",
            "status": "pending",
            "output_dir": str(output_dir.resolve()),
            "timestamp": None,
            "incident_type": None,
            "groundtruth": "",
            "rank": None,
            "ttm": None,
            "predicted_list": [],
            "predicted_list_v1": [],
            "rank_v1": None,
            "parse_strategy": "unparsed",
            "reasoner_outputs": [],
            "final_result": None,
            "error": None,
        }

    @staticmethod
    def _fail(result: Dict[str, Any], error: Exception | str) -> None:
        result["status"] = "failed"
        result["error"] = str(error)
        save_case_result(result)

    def infer(
        self,
        input_list: List[Dict[str, Any]],
        output_dir: str | Path,
    ) -> List[Dict[str, Any]]:
        """批量推理并落盘，返回与 input_list 顺序一致的结果列表。"""
        if not isinstance(input_list, list):
            raise TypeError("input_list 必须是列表")

        with self._infer_lock:
            return self._infer(input_list, output_dir)

    def _infer(
        self,
        input_list: List[Dict[str, Any]],
        output_dir: str | Path,
    ) -> List[Dict[str, Any]]:
        output_root = ensure_dir(Path(output_dir) / "competition")
        results: List[Dict[str, Any]] = []
        valid_items: List[tuple[InferenceCase, Dict[str, Any]]] = []

        for input_index, label in enumerate(input_list):
            raw_category = (
                label.get("alarm_type") or self.alarm_type
                if isinstance(label, dict)
                else self.alarm_type
            )
            current_output_dir = case_output_dir(
                category_output_dir(output_root, raw_category),
                input_index,
            )
            raw_case_id = (
                str(label.get("case_id", label.get("id", input_index)))
                if isinstance(label, dict)
                else str(input_index)
            )
            result = self._new_result(
                input_index=input_index,
                case_id=raw_case_id,
                output_dir=current_output_dir,
            )
            result["incident_type"] = raw_category
            if isinstance(label, dict):
                result["timestamp"] = label.get("alarm_time")
            results.append(result)
            try:
                case = normalize_case(
                    label,
                    input_index=input_index,
                    mode="competition",
                    default_alarm_type=self.alarm_type,
                )
                result["case_id"] = case.case_id
                result["timestamp"] = case.alarm_time
                result["incident_type"] = case.alarm_type
                result["groundtruth"] = case.groundtruth
                result["ttm"] = case.ttm
                valid_items.append((case, result))
            except (TypeError, ValueError) as exc:
                self._fail(result, exc)

        for batch in iter_batches(valid_items, self.batch_size):
            cases = [item[0] for item in batch]
            batch_results = [item[1] for item in batch]
            reasoner_dirs = [
                ensure_dir(Path(result["output_dir"]) / "reasoners")
                for result in batch_results
            ]

            try:
                reasoner_responses = (
                    self.generator.generate_competition_reasoners(
                        [self._alarm_data(case) for case in cases],
                        reasoner_dirs,
                    )
                )
            except Exception as exc:
                for result in batch_results:
                    self._fail(result, f"Competition Reasoner 推理失败: {exc}")
                continue

            verifier_data = []
            verifier_dirs = []
            for case, result, output_path, responses in zip(
                cases, batch_results, reasoner_dirs, reasoner_responses,
            ):
                merged = self._merge_reasoner_outputs(responses)
                result["reasoner_outputs"] = responses
                for response_index, response in enumerate(responses, start=1):
                    save_text(
                        response,
                        output_path / f"Reasoner{response_index}.txt",
                    )
                save_text(merged, output_path / "raw_responses.txt")

                data = self._alarm_data(case)
                data["reasoner_outputs"] = merged
                verifier_data.append(data)
                verifier_dirs.append(
                    ensure_dir(Path(result["output_dir"]) / "verifier")
                )

            try:
                verifier_responses = (
                    self.generator.generate_competition_verifier(
                        verifier_data,
                        verifier_dirs,
                    )
                )
            except Exception as exc:
                for result in batch_results:
                    self._fail(result, f"Competition Verifier 推理失败: {exc}")
                continue

            for case, result, verifier_dir, response in zip(
                cases, batch_results, verifier_dirs, verifier_responses,
            ):
                save_text(response, verifier_dir / "raw_responses.txt")
                result["status"] = "success"
                result["final_result"] = response
                result.update(build_prediction_fields(
                    response,
                    case.root_cause_candidates,
                    case.groundtruth,
                ))
                result["error"] = None
                save_case_result(result)

        save_batch_results(
            mode="competition",
            model_path=self.model_path,
            output_dir=output_root,
            results=results,
        )
        save_category_csvs(output_root, results)
        return results
