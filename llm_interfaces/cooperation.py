"""Cooperation 模式的可复用 Python 接口。"""

from __future__ import annotations

import threading
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from .data_adapter import InferenceCase, normalize_case, truncate_history
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


EARLY_STOP_MARKER = '["(上一轮次已经符合要求)"]'


class CooperationInference:
    """Cooperation（多轮 Meta/Reasoner）推理接口。

    模型在本类初始化时加载一次。后续可以多次调用 infer，期间不会
    重新加载模型。调用方只应实例化当前需要的推理模式。
    """

    def __init__(
        self,
        model_path: str,
        *,
        alarm_type: Optional[str] = None,
        batch_size: int = 8,
        max_rounds: int = 5,
        history_max_chars: int = 9000,
        tensor_parallel_size: Optional[int] = None,
        gpu_memory_utilization: float = 0.9,
        max_model_len: int = 16384,
        max_tokens: int = 4096,
        prompt_safety_tokens: int = 512,
        model_options: Optional[Dict[str, Any]] = None,
        llm_factory: Optional[Callable[..., Any]] = None,
        sampling_params_factory: Optional[Callable[..., Any]] = None,
    ) -> None:
        if max_rounds <= 0:
            raise ValueError("max_rounds 必须大于 0")
        if history_max_chars <= 0:
            raise ValueError("history_max_chars 必须大于 0")

        self.model_path = model_path
        self.alarm_type = alarm_type
        self.batch_size = batch_size
        self.max_rounds = max_rounds
        self.history_max_chars = history_max_chars
        self._infer_lock = threading.Lock()

        # 构造 CooperationInference 时加载；infer 中不再加载。
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
            "mode": "cooperation",
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
            "round_count": 0,
            "early_stopped": False,
            "rounds": [],
            "final_result": None,
            "last_reasoner_output": None,
            "error": None,
        }

    @staticmethod
    def _fail(result: Dict[str, Any], error: Exception | str) -> None:
        result["status"] = "failed"
        result["error"] = str(error)
        save_case_result(result)

    @staticmethod
    def _succeed(
        result: Dict[str, Any],
        case: InferenceCase,
        response: str,
    ) -> None:
        result["status"] = "success"
        result["final_result"] = response
        result.update(build_prediction_fields(
            response,
            case.root_cause_candidates,
            case.groundtruth,
        ))
        result["error"] = None
        save_case_result(result)

    def _last_round_outputs(self, state: Dict[str, Any]) -> str:
        last_meta = state.get("last_meta")
        last_reasoner = state.get("last_reasoner")
        if last_meta is None and last_reasoner is None:
            return "无上一轮输出。"

        sections = []
        if last_meta is not None:
            sections.append(
                "[Last Meta]\n"
                + truncate_history(last_meta, self.history_max_chars)
            )
        if last_reasoner is not None:
            sections.append(
                "[Last Reasoner]\n"
                + truncate_history(last_reasoner, self.history_max_chars)
            )
        return "\n\n".join(sections)

    @staticmethod
    def _meta_data(
        case: InferenceCase,
        last_round_outputs: str,
    ) -> Dict[str, Any]:
        return {
            "Last_round_outputs": last_round_outputs,
            "alarm_type": case.alarm_type,
            "semantic_labels_key": case.semantic_labels,
            "sop": "",
            "root_cause_candidates": case.root_cause_candidates,
        }

    @staticmethod
    def _reasoner_data(
        case: InferenceCase,
        meta_output: str,
    ) -> Dict[str, Any]:
        return {
            "meta_output": meta_output,
            "alarm_type": case.alarm_type,
            "alarm_time": case.alarm_time,
            "semantic_labels": case.semantic_labels,
            "root_cause_candidates": case.root_cause_candidates,
        }

    def infer(
        self,
        input_list: List[Dict[str, Any]],
        output_dir: str | Path,
    ) -> List[Dict[str, Any]]:
        """批量执行多轮推理并落盘，返回与输入顺序一致的结果。"""
        if not isinstance(input_list, list):
            raise TypeError("input_list 必须是列表")

        with self._infer_lock:
            return self._infer(input_list, output_dir)

    def _infer(
        self,
        input_list: List[Dict[str, Any]],
        output_dir: str | Path,
    ) -> List[Dict[str, Any]]:
        output_root = ensure_dir(Path(output_dir) / "cooperation")
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
                    mode="cooperation",
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
            states = [
                {
                    "case": case,
                    "result": result,
                    "last_meta": None,
                    "last_reasoner": None,
                    "done": False,
                }
                for case, result in batch
            ]

            for round_index in range(self.max_rounds):
                active = [state for state in states if not state["done"]]
                if not active:
                    break

                meta_dirs = [
                    ensure_dir(
                        Path(state["result"]["output_dir"])
                        / f"round{round_index}"
                        / "meta"
                    )
                    for state in active
                ]
                meta_data = [
                    self._meta_data(
                        state["case"], self._last_round_outputs(state),
                    )
                    for state in active
                ]

                try:
                    meta_responses = self.generator.generate_cooperation_meta(
                        meta_data, meta_dirs,
                    )
                except Exception as exc:
                    for state in active:
                        self._fail(
                            state["result"],
                            f"Cooperation Meta 推理失败: {exc}",
                        )
                        state["done"] = True
                    break

                reasoner_states = []
                reasoner_data = []
                reasoner_dirs = []
                for state, meta_dir, meta_response in zip(
                    active, meta_dirs, meta_responses,
                ):
                    save_text(meta_response, meta_dir / "raw_responses.txt")
                    state["last_meta"] = meta_response
                    state["result"]["rounds"].append({
                        "round": round_index,
                        "meta_output": meta_response,
                        "reasoner_output": None,
                    })
                    state["result"]["round_count"] = round_index + 1

                    if EARLY_STOP_MARKER in meta_response:
                        state["result"]["early_stopped"] = True
                        state["done"] = True
                        self._succeed(
                            state["result"], state["case"], meta_response,
                        )
                        continue

                    reasoner_states.append(state)
                    reasoner_data.append(
                        self._reasoner_data(state["case"], meta_response)
                    )
                    reasoner_dirs.append(
                        ensure_dir(
                            Path(state["result"]["output_dir"])
                            / f"round{round_index}"
                            / "reasoner"
                        )
                    )

                if not reasoner_states:
                    continue

                try:
                    reasoner_responses = (
                        self.generator.generate_cooperation_reasoner(
                            reasoner_data, reasoner_dirs,
                        )
                    )
                except Exception as exc:
                    for state in reasoner_states:
                        self._fail(
                            state["result"],
                            f"Cooperation Reasoner 推理失败: {exc}",
                        )
                        state["done"] = True
                    continue

                for state, reasoner_dir, reasoner_response in zip(
                    reasoner_states, reasoner_dirs, reasoner_responses,
                ):
                    save_text(
                        reasoner_response,
                        reasoner_dir / "raw_responses.txt",
                    )
                    state["last_reasoner"] = reasoner_response
                    state["result"]["last_reasoner_output"] = reasoner_response
                    state["result"]["rounds"][-1][
                        "reasoner_output"
                    ] = reasoner_response

            for state in states:
                result = state["result"]
                if result["status"] == "pending":
                    # Meta 输出包含根因排序；Reasoner 输出是供下一轮 Meta 使用的分析。
                    self._succeed(
                        result, state["case"], state["last_meta"] or "",
                    )

        save_batch_results(
            mode="cooperation",
            model_path=self.model_path,
            output_dir=output_root,
            results=results,
        )
        save_category_csvs(output_root, results)
        return results
