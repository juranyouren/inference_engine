# -*- coding: utf-8 -*-

import csv
import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from bian_reproduction.parsing import (
    normalize_candidates,
    normalize_joint_scoring,
    parse_document,
)
from bian_reproduction.prompts import (
    ALERT_SUMMARY_PROMPT,
    ANOMALY_ANALYSIS_PROMPT,
    JOINT_SCORING_PROMPT,
)


def _json_text(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, indent=2, default=str)


def _root_cause(value: Any) -> str:
    if isinstance(value, dict):
        value = (
            value.get("category")
            or value.get("name")
            or value.get("root_cause")
        )
    return str(value).strip() if value is not None else ""


def _sop(label: Dict[str, Any], semantic_labels: Dict[str, Any]) -> Any:
    value = semantic_labels.get("sop")
    if value is not None and value != "" and value != []:
        return value
    # Compatibility with the sample format supplied for this standalone
    # reproduction.  The project repository's normal input contract is not
    # changed by this fallback.
    return label.get("sop", "")


def _parse_index(path: Path, alarm_type: str) -> Optional[int]:
    pattern = rf"^{re.escape(alarm_type)}_(\d+)_label_.*\.json$"
    match = re.match(pattern, path.name)
    if match:
        return int(match.group(1))
    match = re.search(r"_(\d+)_label_", path.name)
    if match:
        return int(match.group(1))
    match = re.fullmatch(r"label_(\d+)\.json", path.name)
    return int(match.group(1)) if match else None


def discover_label_files(data_dir: Path, alarm_type: str) -> Dict[int, Path]:
    files = sorted(data_dir.glob("*_label_*.json"))
    files.extend(sorted(data_dir.glob("label_*.json")))
    result: Dict[int, Path] = {}
    for path in files:
        idx = _parse_index(path, alarm_type)
        if idx is not None:
            result.setdefault(idx, path)
    return dict(sorted(result.items()))


def load_record(label_path: Path, alarm_type: str) -> Dict[str, Any]:
    label = json.loads(label_path.read_text(encoding="utf-8"))
    semantic_labels = label.get("semantic_labels", {})
    if not isinstance(semantic_labels, dict):
        semantic_labels = {}

    candidates = label.get("root_cause_candidates", [])
    if not candidates and label.get("root_cause"):
        candidates = [label["root_cause"]]

    return {
        "alarm_type": alarm_type,
        "alarm_time": label.get("alarm_time", semantic_labels.get("alarm_time", "")),
        "semantic_labels": semantic_labels,
        "sop": _sop(label, semantic_labels),
        "root_cause_candidates": candidates,
        "root_cause": _root_cause(label.get("root_cause", "")),
        "label_file_path": str(label_path.resolve()),
    }


def _result(
    record: Dict[str, Any],
    idx: int,
    summary_text: str,
    anomaly_text: str,
    joint_text: str,
) -> Dict[str, Any]:
    candidates = normalize_candidates(record["root_cause_candidates"])
    summary = parse_document(summary_text)
    anomaly = parse_document(anomaly_text)
    joint = parse_document(joint_text)
    ranked, strategy = normalize_joint_scoring(joint, candidates)
    pred_rc = [item["category"] for item in ranked]
    pred_top1 = pred_rc[0] if pred_rc else None
    groundtruth = record.get("root_cause", "")
    rank = pred_rc.index(groundtruth) + 1 if groundtruth in pred_rc else None

    return {
        "case_idx": idx,
        "alarm_type": record["alarm_type"],
        "alarm_time": record.get("alarm_time"),
        "label_file_path": record.get("label_file_path"),
        "groundtruth": groundtruth,
        "pred_top1_rc": pred_top1,
        "pred_rc": pred_rc,
        "pred_scores": ranked,
        "rank": rank,
        "is_correct": pred_top1 == groundtruth if pred_top1 else None,
        "parse_strategy": strategy,
        "bian_stage1": {
            "alert_summary": summary,
            "anomaly_analysis": anomaly,
            "joint_scoring": joint,
            "candidate_semantics": "root_cause_candidates",
        },
    }


class BianStage1Runner:
    def __init__(
        self,
        model_path: str,
        visible_devices: str = "6,7",
        batch_size: int = 16,
        max_new_tokens: int = 2048,
    ) -> None:
        self.model_path = model_path
        self.visible_devices = visible_devices
        self.batch_size = batch_size
        self.max_new_tokens = max_new_tokens
        self.llm = None
        self.sampling_params = None

    def _init_model(self) -> None:
        if self.llm is not None:
            return
        os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
        if self.visible_devices:
            os.environ["ASCEND_RT_VISIBLE_DEVICES"] = self.visible_devices
        from vllm import LLM, SamplingParams

        tensor_parallel_size = len(
            [item for item in self.visible_devices.split(",") if item.strip()]
        ) or 1
        self.llm = LLM(
            model=self.model_path,
            tensor_parallel_size=tensor_parallel_size,
            trust_remote_code=True,
            gpu_memory_utilization=0.9,
            max_model_len=16384,
        )
        self.sampling_params = SamplingParams(
            temperature=0.2,
            top_p=0.9,
            max_tokens=self.max_new_tokens,
        )

    def _invoke(self, prompts: Sequence[str]) -> List[str]:
        self._init_model()
        responses: List[str] = []
        for start in range(0, len(prompts), self.batch_size):
            batch = prompts[start:start + self.batch_size]
            chat_inputs = [
                [{"role": "user", "content": prompt}]
                for prompt in batch
            ]
            outputs = self.llm.chat(
                chat_inputs,
                self.sampling_params,
                use_tqdm=False,
            )
            responses.extend(item.outputs[0].text for item in outputs)
        return responses

    @staticmethod
    def _save_stage(
        stage_dir: Path,
        prompt: str,
        response: str,
        parsed: Dict[str, Any],
    ) -> None:
        stage_dir.mkdir(parents=True, exist_ok=True)
        (stage_dir / "prompt.txt").write_text(prompt, encoding="utf-8")
        (stage_dir / "raw_responses.txt").write_text(response, encoding="utf-8")
        (stage_dir / "parsed.json").write_text(
            json.dumps(parsed, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def run(
        self,
        data_dir: str,
        alarm_type: str,
        indices: Sequence[int],
        output_dir: str,
    ) -> Dict[str, Any]:
        data_path = Path(data_dir).resolve()
        selected = sorted(set(int(idx) for idx in indices))
        if not selected:
            raise ValueError("没有可处理的 index")

        run_name = f"idx_{min(selected)}_{max(selected)}"
        final_dir = (
            Path(output_dir).resolve()
            / "llm_infer_bian_stage1"
            / alarm_type
            / run_name
        )
        final_dir.mkdir(parents=True, exist_ok=True)

        processed: List[int] = []
        skipped: List[int] = []
        results: List[Dict[str, Any]] = []
        label_map = discover_label_files(data_path, alarm_type)

        for start in range(0, len(selected), self.batch_size):
            batch = selected[start:start + self.batch_size]
            records: List[Dict[str, Any]] = []
            valid_indices: List[int] = []
            case_dirs: List[Path] = []
            for idx in batch:
                label_path = label_map.get(idx)
                if label_path is None:
                    skipped.append(idx)
                    continue
                try:
                    record = load_record(label_path, alarm_type)
                    if not record["root_cause_candidates"]:
                        raise ValueError("root_cause_candidates 为空")
                    case_dir = final_dir / str(idx)
                    case_dir.mkdir(parents=True, exist_ok=True)
                    records.append(record)
                    valid_indices.append(idx)
                    case_dirs.append(case_dir)
                except Exception as exc:
                    print(f"idx={idx}: 数据加载失败 - {exc}")
                    skipped.append(idx)

            if not records:
                continue

            try:
                summary_prompts = [
                    ALERT_SUMMARY_PROMPT.format(
                        alarm_type=item["alarm_type"],
                        alarm_time=item["alarm_time"],
                        semantic_labels=_json_text(item["semantic_labels"]),
                        root_cause_candidates=_json_text(item["root_cause_candidates"]),
                    )
                    for item in records
                ]
                summary_responses = self._invoke(summary_prompts)
                if len(summary_responses) != len(records):
                    raise RuntimeError("告警摘要响应数量不匹配")

                anomaly_prompts = [
                    ANOMALY_ANALYSIS_PROMPT.format(
                        alarm_type=item["alarm_type"],
                        alarm_time=item["alarm_time"],
                        alert_summary=summary,
                        semantic_labels=_json_text(item["semantic_labels"]),
                        sop=_json_text(item["sop"]),
                        root_cause_candidates=_json_text(item["root_cause_candidates"]),
                    )
                    for item, summary in zip(records, summary_responses)
                ]
                anomaly_responses = self._invoke(anomaly_prompts)
                if len(anomaly_responses) != len(records):
                    raise RuntimeError("异常分析响应数量不匹配")

                joint_prompts = [
                    JOINT_SCORING_PROMPT.format(
                        alarm_type=item["alarm_type"],
                        alarm_time=item["alarm_time"],
                        root_cause_candidates=_json_text(item["root_cause_candidates"]),
                        alert_summary=summary,
                        anomaly_analysis=anomaly,
                    )
                    for item, summary, anomaly in zip(
                        records,
                        summary_responses,
                        anomaly_responses,
                    )
                ]
                joint_responses = self._invoke(joint_prompts)
                if len(joint_responses) != len(records):
                    raise RuntimeError("联合评分响应数量不匹配")

                for idx, record, case_dir, summary_prompt, summary, anomaly_prompt, anomaly, joint_prompt, joint in zip(
                    valid_indices,
                    records,
                    case_dirs,
                    summary_prompts,
                    summary_responses,
                    anomaly_prompts,
                    anomaly_responses,
                    joint_prompts,
                    joint_responses,
                ):
                    self._save_stage(
                        case_dir / "alert_summary",
                        summary_prompt,
                        summary,
                        parse_document(summary),
                    )
                    self._save_stage(
                        case_dir / "anomaly_analysis",
                        anomaly_prompt,
                        anomaly,
                        parse_document(anomaly),
                    )
                    self._save_stage(
                        case_dir / "joint_scoring",
                        joint_prompt,
                        joint,
                        parse_document(joint),
                    )
                    result = _result(record, idx, summary, anomaly, joint)
                    (case_dir / "prediction.json").write_text(
                        json.dumps(result, ensure_ascii=False, indent=2),
                        encoding="utf-8",
                    )
                    results.append(result)
                    processed.append(idx)
            except Exception as exc:
                print(f"BiAn Stage 1 批处理失败 - {exc}")
                skipped.extend(valid_indices)

        processed = sorted(set(processed))
        skipped = sorted(set(skipped) - set(processed))
        results.sort(key=lambda item: item["case_idx"])
        payload = {
            "meta": {
                "pipeline": "bian_stage1",
                "candidate_semantics": "root_cause_candidates",
                "alarm_type": alarm_type,
                "data_dir": str(data_path),
                "indices": selected,
                "model_path": self.model_path,
                "output_dir": str(final_dir),
            },
            "summary": {
                "processed_count": len(processed),
                "skipped_count": len(skipped),
                "processed_indices": processed,
                "skipped_indices": skipped,
            },
            "results": results,
        }
        (final_dir / "predictions.json").write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        with (final_dir / "predictions.jsonl").open("w", encoding="utf-8") as stream:
            for item in results:
                stream.write(json.dumps(item, ensure_ascii=False) + "\n")
        with (final_dir / "predictions.csv").open("w", encoding="utf-8-sig", newline="") as stream:
            fields = [
                "case_idx", "alarm_type", "groundtruth", "pred_top1_rc",
                "pred_rc", "rank", "is_correct", "parse_strategy",
            ]
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            for item in results:
                writer.writerow({
                    "case_idx": item.get("case_idx"),
                    "alarm_type": item.get("alarm_type"),
                    "groundtruth": item.get("groundtruth"),
                    "pred_top1_rc": item.get("pred_top1_rc"),
                    "pred_rc": json.dumps(item.get("pred_rc", []), ensure_ascii=False),
                    "rank": item.get("rank"),
                    "is_correct": item.get("is_correct"),
                    "parse_strategy": item.get("parse_strategy"),
                })
        (final_dir / "meta.json").write_text(
            json.dumps(payload["meta"], ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return payload

