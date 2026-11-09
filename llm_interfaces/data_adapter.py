"""把已经 json.load 的 label 字典转换为模型所需的统一结构。"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Dict, Literal


InferenceMode = Literal["competition", "cooperation"]


@dataclass(frozen=True)
class InferenceCase:
    input_index: int
    case_id: str
    alarm_type: str
    alarm_time: Any
    semantic_labels: Dict[str, Any]
    sop: str
    root_cause_candidates: Any
    groundtruth: str
    ttm: Any


def _sop_to_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, list) and all(isinstance(item, str) for item in value):
        return "\n".join(value).strip()
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False) if value else ""
    return str(value).strip()


def read_sop(label: Dict[str, Any]) -> str:
    labels = label.get("semantic_labels")
    return _sop_to_text(labels.get("sop")) if isinstance(labels, dict) else ""


def without_sop(label: Dict[str, Any]) -> Dict[str, Any]:
    labels = label.get("semantic_labels")
    if not isinstance(labels, dict):
        return label
    return {**label, "semantic_labels": {key: value for key, value in labels.items() if key != "sop"}}


def _root_cause_name(root_cause: Any) -> str:
    if isinstance(root_cause, dict):
        root_cause = (
            root_cause.get("category", "")
            or root_cause.get("name", "")
            or root_cause.get("root_cause", "")
        )
    return str(root_cause).strip() if root_cause is not None else ""


def groundtruth_category(label: Dict[str, Any]) -> str:
    labels = label.get("semantic_labels")
    if isinstance(labels, dict):
        category = _root_cause_name(labels.get("root_cause"))
        if category:
            return category
    return _root_cause_name(label.get("root_cause"))


def _case_id(label: Dict[str, Any], input_index: int) -> str:
    for key in ("case_id", "id", "index", "idx"):
        value = label.get(key)
        if value is not None and str(value).strip():
            return str(value)
    return str(input_index)


def normalize_case(
    label: Dict[str, Any],
    *,
    input_index: int,
    mode: InferenceMode,
    alarm_type: str,
) -> InferenceCase:
    """校验单条输入。

    SOP 严格且只从 label["semantic_labels"]["sop"] 读取。
    """
    if not isinstance(label, dict):
        raise TypeError(
            f"input_list[{input_index}] 必须是 json.load 后的 dict，"
            f"实际为 {type(label).__name__}"
        )

    semantic_labels = label.get("semantic_labels")
    if not isinstance(semantic_labels, dict):
        raise ValueError(
            f"input_list[{input_index}].semantic_labels 必须是字典"
        )

    if not alarm_type or not str(alarm_type).strip():
        raise ValueError("alarm_type 不能为空")

    sop = read_sop(label)
    if mode == "competition" and not sop:
        raise ValueError(
            f"input_list[{input_index}] 的 semantic_labels.sop 为空，"
            "不能使用 Competition 模式"
        )
    if mode == "cooperation" and sop:
        raise ValueError(
            f"input_list[{input_index}] 的 semantic_labels.sop 非空，"
            "不能使用 Cooperation 模式"
        )

    return InferenceCase(
        input_index=input_index,
        case_id=_case_id(label, input_index),
        alarm_type=str(alarm_type),
        alarm_time=label.get(
            "alarm_time",
            semantic_labels.get("alarm_time", ""),
        ),
        semantic_labels={
            key: value for key, value in semantic_labels.items()
            if key not in {"sop", "root_cause"}
        },
        sop=sop,
        root_cause_candidates=label.get("root_cause_candidates") or [],
        groundtruth=groundtruth_category(label),
        ttm=label.get("ttm", semantic_labels.get("ttm")),
    )
