# inference/selection.py
# -*- coding: utf-8 -*-

import json
import os
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import config
from inference.common import ensure_dir, load_json, save_json
from utils.public_functions import load_alarm_template


# ============================================================
# 2. selection 规范化
# ============================================================

KPI_KEYS = [
    "traffic_in",
    "traffic_out",
    "drop_packet_rate",
    "error_packet_rate",
    "offline_loss_rate",
    "cpu_utilization",
    "memory_utilization",
    "temperature",
]

SEGMENT_CHOICES = {"mix", "alarming"}
STAT_CHOICES = {"max", "min", "mean", "duration"}
KPI_CHOICES = SEGMENT_CHOICES | STAT_CHOICES


def build_selection_from_selector(selector_result: Any) -> Dict[str, Any]:
    """把 Selector / Refiner 输出规范化为标准 selection。"""

    def dedup_keep_order(seq: Iterable[Any]) -> List[Any]:
        seen = set()
        out = []
        for item in seq:
            if item not in seen:
                seen.add(item)
                out.append(item)
        return out

    def find_first_json(obj: Any) -> Optional[Dict[str, Any]]:
        if isinstance(obj, str):
            try:
                obj = json.loads(obj)
            except Exception:
                return None

        if isinstance(obj, dict):
            if "log" in obj and "kpi" in obj:
                return obj

            inner = obj.get("json")
            if isinstance(inner, dict) and "log" in inner and "kpi" in inner:
                return inner

            return None

        if isinstance(obj, list):
            for item in obj:
                found = find_first_json(item)
                if found is not None:
                    return found

        return None

    raw = find_first_json(selector_result)
    if raw is None:
        raise ValueError("未能从 selector/refiner 输出中提取有效 selection JSON")

    raw_log = raw.get("log", [])
    if isinstance(raw_log, str):
        raw_log = [raw_log]
    elif not isinstance(raw_log, list):
        raw_log = []

    # 保留具体日志模板名；为空时使用默认聚合特征。
    log = dedup_keep_order(raw_log) if raw_log else ["template_occur_count"]

    raw_kpi = raw.get("kpi", {})
    if not isinstance(raw_kpi, dict):
        raw_kpi = {}

    normalized_kpi = {}
    for kpi in KPI_KEYS:
        raw_values = raw_kpi.get(kpi, [])
        if not isinstance(raw_values, list):
            raw_values = [raw_values] if raw_values else []

        valid_values = dedup_keep_order(
            [value for value in raw_values if value in KPI_CHOICES]
        )

        final_values = list(valid_values)

        if not any(value in SEGMENT_CHOICES for value in final_values):
            final_values.append("mix")

        if not any(value in STAT_CHOICES for value in final_values):
            final_values.append("mean")

        normalized_kpi[kpi] = dedup_keep_order(final_values)

    return {
        "log": log,
        "kpi": normalized_kpi,
    }

# ============================================================
# 7. selection 文件 / Selector / Refiner
# ============================================================


def load_selection_from_jsonl(path: str) -> Dict[str, Any]:
    records = []

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    return build_selection_from_selector(records)


def resolve_selection_path(
    scenario: Dict[str, Any],
    selection_path: Optional[str],
) -> str:
    scenario_name = scenario["name"]
    candidates: List[Path] = []

    if selection_path:
        path = Path(selection_path)

        if path.is_file():
            return str(path)

        if path.is_dir():
            candidates.extend([
                path / scenario_name / getattr(
                    config,
                    "SELECTION_FILE_NAME",
                    "selection.json",
                ),
                path / scenario_name / "selector" / "res.jsonl",
                path / getattr(
                    config,
                    "SELECTION_FILE_NAME",
                    "selection.json",
                ),
                path / "selector" / "res.jsonl",
            ])
    else:
        predict_res_dir = getattr(config, "PREDICT_RES_DIR", None)
        base_res_dir = getattr(config, "BASE_RES_DIR", None)

        if predict_res_dir:
            base = Path(predict_res_dir)
            candidates.extend([
                base / scenario_name / getattr(
                    config,
                    "SELECTION_FILE_NAME",
                    "selection.json",
                ),
                base / scenario_name / "selector" / "res.jsonl",
                base / "selection" / f"{scenario_name}.json",
            ])

        if base_res_dir:
            base = Path(base_res_dir)
            candidates.extend([
                base / "selection" / f"{scenario_name}.json",
                base / "selection" / getattr(
                    config,
                    "SELECTION_FILE_NAME",
                    "selection.json",
                ),
            ])

    for candidate in candidates:
        if candidate.exists():
            return str(candidate)

    raise FileNotFoundError(
        f"未找到场景 {scenario_name} 的 selection 文件。\n"
        "请用 --selection-path 指定 selection.json、selector/res.jsonl "
        "或包含这些文件的目录。\n"
        "已尝试：\n"
        + "\n".join(str(path) for path in candidates)
    )


def load_selection(
    scenario: Dict[str, Any],
    selection_path: Optional[str],
) -> Dict[str, Any]:
    real_path = resolve_selection_path(scenario, selection_path)

    if real_path.endswith(".jsonl"):
        selection = load_selection_from_jsonl(real_path)
    else:
        selection = build_selection_from_selector(load_json(real_path))

    print(f"[Selection][file] {scenario['name']} 使用: {real_path}")
    return selection


def _lazy_import_selector_refiner():
    try:
        from llm_inference.selector_refiner import LLMEngine, Selector, Refiner
    except ImportError as exc:
        raise ImportError(
            "selection-source=selector/selector_refiner 需要 "
            "llm_inference/selector_refiner.py"
        ) from exc

    return LLMEngine, Selector, Refiner


def load_alarm_template_compat(alarm_type: str):
    template_dir = getattr(config, "TEMPLATE_DIR", None)

    if template_dir:
        try:
            return load_alarm_template(
                alarm_type,
                template_dir=template_dir,
            )
        except TypeError:
            pass

    return load_alarm_template(alarm_type)


def generate_selection_by_selector(
    scenario: Dict[str, Any],
    output_dir: str,
) -> Dict[str, Any]:
    LLMEngine, Selector, _Refiner = _lazy_import_selector_refiner()

    model_path = getattr(config, "MODEL_PATH")
    selector_ex_num = getattr(config, "SELECTOR_EX_NUM", 1)

    engine = LLMEngine.get_instance(
        model_path=model_path,
        gpu_memory_utilization=0.9,
        max_model_len=16384,
    )
    selector = Selector(engine, "SELECTOR")

    template = load_alarm_template_compat(scenario["alarm_type"])

    selector_output_dir = os.path.join(output_dir, "selector")
    ensure_dir(selector_output_dir)

    selector_result = selector.select(
        [template],
        selector_output_dir,
        selector_ex_num,
    )

    selection = build_selection_from_selector(selector_result)
    save_json(
        selection,
        os.path.join(output_dir, "selection_selector.json"),
    )
    return selection


def refine_selection_by_tree_summary(
    scenario: Dict[str, Any],
    previous_selection: Dict[str, Any],
    tree_summary: Dict[str, Any],
    output_dir: str,
    round_id: int,
) -> Dict[str, Any]:
    LLMEngine, _Selector, Refiner = _lazy_import_selector_refiner()

    model_path = getattr(config, "MODEL_PATH")
    refiner_ex_num = getattr(config, "REFINER_EX_NUM", 1)

    engine = LLMEngine.get_instance(
        model_path=model_path,
        gpu_memory_utilization=0.9,
        max_model_len=16384,
    )
    refiner = Refiner(engine, "REFINER")

    template = load_alarm_template_compat(scenario["alarm_type"])

    refiner_output_dir = os.path.join(
        output_dir,
        f"refiner_round_{round_id}",
    )
    ensure_dir(refiner_output_dir)

    refiner_result = refiner.refine(
        [template],
        refiner_output_dir,
        refiner_ex_num,
        selection=previous_selection,
        summary=tree_summary,
    )

    refined_selection = build_selection_from_selector(refiner_result)

    save_json(
        refined_selection,
        os.path.join(
            output_dir,
            f"selection_refined_round_{round_id}.json",
        ),
    )

    return refined_selection


def build_tree_summary_for_refiner(
    tree_payload: Dict[str, Any],
) -> Dict[str, Any]:
    wrong_cases = []
    correct_cases = []

    for item in tree_payload.get("results", []):
        compact = {
            "case_idx": item.get("case_idx"),
            "alarm_type": item.get("alarm_type"),
            "groundtruth": item.get("groundtruth"),
            "pred_top1_rc": item.get("pred_top1_rc"),
            "pred_rc": item.get("pred_rc"),
            "cot": item.get("cot"),
            "rank": item.get("rank"),
            "features": item.get("features"),
        }

        if item.get("is_correct") is True:
            correct_cases.append(compact)
        elif item.get("is_correct") is False:
            wrong_cases.append(compact)

    return {
        "summary": tree_payload.get("summary", {}),
        "wrong_cases": wrong_cases[:20],
        "correct_cases": correct_cases[:10],
        "note": (
            "wrong_cases 是当前 selection 下决策树预测错误的样本；"
            "Refiner 应优先根据这些样本调整 log/kpi 特征选择。"
        ),
    }
