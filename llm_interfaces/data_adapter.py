"""把已经 json.load 的 label 字典转换为模型所需的统一结构。"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Dict, Literal, Optional


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


def _sop_to_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False) if value else ""
    return str(value).strip()


def _root_cause_candidates(label: Dict[str, Any]) -> Any:
    candidates = label.get("root_cause_candidates", [])
    if candidates:
        return candidates

    root_cause = label.get("root_cause", "")
    if isinstance(root_cause, dict):
        root_cause = (
            root_cause.get("category", "")
            or root_cause.get("name", "")
            or root_cause.get("root_cause", "")
        )
    return [root_cause] if root_cause else []


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
    default_alarm_type: Optional[str] = None,
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

    alarm_type = label.get("alarm_type") or default_alarm_type
    if not alarm_type or not str(alarm_type).strip():
        raise ValueError(
            f"input_list[{input_index}] 缺少 alarm_type；"
            "请在 JSON 中提供，或在类初始化时传入 alarm_type"
        )

    sop = _sop_to_text(semantic_labels.get("sop"))
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
        semantic_labels=semantic_labels,
        sop=sop,
        root_cause_candidates=_root_cause_candidates(label),
    )


def truncate_history(text: str, max_chars: int) -> str:
    if max_chars <= 0 or len(text) <= max_chars:
        return text
    half = max_chars // 2
    return (
        text[:half]
        + "\n\n[...输出过长，中间部分已截断...]\n\n"
        + text[-half:]
    )
