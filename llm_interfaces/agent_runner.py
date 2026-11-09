"""共享的单 case 执行、结果校验和落盘。"""

from __future__ import annotations

import json
import re
import threading
from pathlib import Path

from .data_adapter import groundtruth_category, normalize_case
from .generator import RCAGenerator
from .io_utils import (
    case_output_dir, category_output_dir, csv_result_dict, save_batch_results,
    save_case_result, save_category_csvs,
)
from .result_parser import prediction_fields


def final_text(response):
    text = response.rsplit("</think>", 1)[-1].strip()
    fenced = re.fullmatch(r"```(?:json|text)?\s*\n(.*?)\n```", text, re.S | re.I)
    return fenced.group(1).strip() if fenced else text


def response_json(response):
    try:
        return json.loads(final_text(response))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Agent 返回的 JSON 无法解析: {exc.msg}") from exc


def validate_ranking(ranking, candidates):
    if not isinstance(ranking, list) or not ranking or any(
        not isinstance(name, str) or not name.strip() for name in ranking
    ):
        raise ValueError("根因排序必须为非空字符串列表")
    if len(set(ranking)) != len(ranking):
        raise ValueError("根因排序包含重复项")
    if candidates and set(ranking) != set(candidates):
        raise ValueError("根因排序必须恰好包含所有候选根因")
    return ranking


def case_evidence(case):
    return {
        "alarm_type": case.alarm_type,
        "alarm_time": case.alarm_time,
        "semantic_labels": case.semantic_labels,
    }


class AgentInference:
    def __init__(
        self, engine, *, batch_size=8, max_tokens=4096, prompt_safety_tokens=512,
        sampling_params_factory=None,
    ):
        self.engine = engine
        self.generator = RCAGenerator(
            engine, batch_size=batch_size, max_tokens=max_tokens,
            prompt_safety_tokens=prompt_safety_tokens,
            sampling_params_factory=sampling_params_factory,
        )
        self.llm = self.generator.llm
        self.batch_size = batch_size
        self._infer_lock = threading.Lock()

    def infer(self, input_list, alarm_type, output_dir="res"):
        return [csv_result_dict(row) for row in self.infer_results(input_list, alarm_type, output_dir)]

    def infer_results(self, input_list, alarm_type, output_dir="res", *, case_indices=None):
        if not isinstance(input_list, list):
            raise TypeError("input_list 必须是列表")
        if not isinstance(alarm_type, str) or not alarm_type.strip():
            raise ValueError("alarm_type 不能为空")
        indices = list(range(len(input_list))) if case_indices is None else list(case_indices)
        if len(indices) != len(input_list) or len(set(indices)) != len(indices):
            raise ValueError("case_indices 必须为每个 case 提供一个独立索引")
        output_root = Path(output_dir) / self.mode
        category_dir = category_output_dir(output_root, alarm_type)
        with self._infer_lock:
            results = [
                self._run(label, index, alarm_type, case_output_dir(category_dir, index))
                for index, label in zip(indices, input_list)
            ]
            save_batch_results(mode=self.mode, output_dir=output_root, results=results)
            save_category_csvs(output_root, results)
        return results

    def _run(self, label, index, alarm_type, output_dir):
        groundtruth = groundtruth_category(label) if isinstance(label, dict) else ""
        result = {
            "input_index": index, "case_idx": index, "case_id": str(index),
            "mode": self.mode, "output_dir": str(output_dir.resolve()),
            "incident_type": alarm_type, "timestamp": None, "groundtruth": groundtruth,
            "ttm": None, "final_result": None, "error": None,
            **prediction_fields([], "", "unparsed"),
        }
        try:
            case = normalize_case(label, input_index=index, mode=self.mode, alarm_type=alarm_type)
            result.update(case_id=case.case_id, timestamp=case.alarm_time,
                          groundtruth=case.groundtruth, ttm=case.ttm)
            ranking = self._infer_case(case, output_dir, result)
            result.update(prediction_fields(ranking, case.groundtruth, "structured_json"))
            result["status"] = "success"
        except Exception as exc:
            result.update(status="failed", error=str(exc))
        save_case_result(result)
        return result
