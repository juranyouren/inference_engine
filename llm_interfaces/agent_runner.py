"""同类 case 分阶段批量执行，独立维护每个 case 的状态与结果。"""

from __future__ import annotations

import json
import re
import threading
from itertools import islice
from pathlib import Path

from .data_adapter import groundtruth_category, normalize_case
from .generator import AgentStage, RCAGenerator, RequestFailure
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


def complete_ranking(ranking, candidates, result):
    if not isinstance(ranking, list):
        raise ValueError("根因排序必须为 JSON 数组")
    ordered = {}
    allowed = set(candidates)
    for item in ranking:
        name = item.get("category") if isinstance(item, dict) else item
        if isinstance(name, str):
            name = name.strip()
            if name and (not allowed or name in allowed):
                ordered.setdefault(name, None)
    appended = [name for name in candidates if name not in ordered]
    result["ranking_appended"] = appended
    completed = list(ordered) + appended
    if not completed:
        raise ValueError("根因排序为空，且没有候选根因可补齐")
    return completed


def case_evidence(case):
    return {
        "alarm_type": case.alarm_type,
        "alarm_time": case.alarm_time,
        "semantic_labels": case.semantic_labels,
    }


def request_stage(result, name, requests, *, final=False):
    """暂停当前 case，待同阶段请求批量完成后恢复。"""
    responses = yield AgentStage(name, requests, final)
    answers = []
    for (path, _, _), response in zip(requests, responses):
        if isinstance(response, RequestFailure):
            result.setdefault("request_issues", []).append({"stage": path, "error": response.message})
            if final:
                raise RuntimeError(f"{path}: {response.message}")
            response = ""
        if final and not response:
            raise RuntimeError(f"{path}: vLLM 返回了空响应")
        answers.append(response)
    return answers


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
            folders = [case_output_dir(category_dir, index) for index in indices]
            workflows = [
                self._run(label, index, alarm_type, folder)
                for index, label, folder in zip(indices, input_list, folders)
            ]
            results = self._run_stages(workflows, folders)
            save_batch_results(mode=self.mode, output_dir=output_root, results=results)
            save_category_csvs(output_root, results)
        return results

    def _run_stages(self, workflows, folders):
        results = [None] * len(workflows)
        pending = []
        finals = []

        def advance(pos, answers):
            try:
                stage = workflows[pos].send(answers)
            except StopIteration as done:
                results[pos] = done.value
            else:
                queue = finals if stage.final else pending
                queue.append((pos, stage))

        for pos in range(len(workflows)):
            advance(pos, None)
        while pending or finals:
            # 先处理同一轮的活跃 case，已停止复议的 case 一起做最终排序。
            if pending:
                wave, pending = pending, []
            else:
                wave, finals = finals, []
            requests = (
                (folders[pos], request)
                for pos, stage in wave for request in stage.requests
            )
            responses = self.generator.generate_stage(wave[0][1].name, requests)
            for pos, stage in wave:
                advance(pos, list(islice(responses, len(stage.requests))))
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
        if isinstance(label, dict) and "_label_file_path" in label:
            result["label_file_path"] = label["_label_file_path"]
        try:
            case = normalize_case(label, input_index=index, mode=self.mode, alarm_type=alarm_type)
            result.update(case_id=case.case_id, timestamp=case.alarm_time,
                          groundtruth=case.groundtruth, ttm=case.ttm)
            ranking = yield from self._case_stages(case, result)
            strategy = "structured_json_completed" if result.get("ranking_appended") else "structured_json"
            result.update(prediction_fields(ranking, case.groundtruth, strategy))
            result["status"] = "success"
        except Exception as exc:
            result.update(status="failed", error=str(exc))
        save_case_result(result)
        return result
