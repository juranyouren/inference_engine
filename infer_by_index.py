# infer_by_index.py
# -*- coding: utf-8 -*-

import os

# Ascend NPU + vLLM multiprocessing must use spawn.
# This block must stay before importing any module that may import torch_npu/vllm.
os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")

import multiprocessing as mp

try:
    mp.set_start_method("spawn", force=True)
except RuntimeError:
    pass

import argparse
import glob
import json
import re
import sys
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier, export_text


# ============================================================
# 1. 项目路径与环境
# ============================================================

CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

import config  # noqa: E402

PROJECT_ROOT = Path(
    getattr(config, "PROJECT_ROOT", str(CURRENT_DIR))
).resolve()

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

if hasattr(config, "ASCEND_RT_VISIBLE_DEVICES"):
    os.environ["ASCEND_RT_VISIBLE_DEVICES"] = str(
        config.ASCEND_RT_VISIBLE_DEVICES
    )

from utils.public_functions import (  # noqa: E402
    load_alarm_data,
    load_alarm_template,
    load_json as util_load_json,
)
from rule_inferencer.data_process_v3 import extract_all_features  # noqa: E402


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
# 3. 通用工具
# ============================================================


def ensure_dir(path: str) -> None:
    if path:
        os.makedirs(path, exist_ok=True)


def make_json_safe(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): make_json_safe(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [make_json_safe(x) for x in obj]
    if isinstance(obj, tuple):
        return [make_json_safe(x) for x in obj]
    if isinstance(obj, set):
        return sorted(make_json_safe(x) for x in obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, Path):
        return str(obj)
    return obj


def load_json(path: str) -> Any:
    return util_load_json(path)


def save_json(obj: Any, path: str) -> None:
    ensure_dir(os.path.dirname(path))
    with open(path, "w", encoding="utf-8") as f:
        json.dump(make_json_safe(obj), f, ensure_ascii=False, indent=2)


def save_jsonl(items: List[Dict[str, Any]], path: str) -> None:
    ensure_dir(os.path.dirname(path))
    with open(path, "w", encoding="utf-8") as f:
        for item in items:
            f.write(json.dumps(make_json_safe(item), ensure_ascii=False) + "\n")


def save_csv(items: List[Dict[str, Any]], path: str) -> None:
    ensure_dir(os.path.dirname(path))
    pd.DataFrame(make_json_safe(items)).to_csv(
        path,
        index=False,
        encoding="utf-8-sig",
    )


def iter_batches(indices: List[int], batch_size: int):
    if batch_size <= 0:
        raise ValueError(f"batch_size 必须 > 0，当前为 {batch_size}")

    for pos in range(0, len(indices), batch_size):
        yield indices[pos:pos + batch_size]


def build_output_dir(
    mode: str,
    scenario_name: str,
    output_dir: Optional[str],
    run_name: Optional[str] = None,
) -> str:
    if output_dir:
        final_dir = os.path.join(output_dir, mode, scenario_name)
    else:
        predict_res_dir = getattr(
            config,
            "PREDICT_RES_DIR",
            os.path.join(getattr(config, "BASE_RES_DIR", "."), "predict_result"),
        )
        final_dir = os.path.join(predict_res_dir, mode, scenario_name)

    if run_name:
        final_dir = os.path.join(final_dir, run_name)

    ensure_dir(final_dir)
    return final_dir


def save_infer_outputs(
    payload: Dict[str, Any],
    output_dir: str,
    output_format: str,
) -> None:
    results = payload.get("results", [])

    if output_format in ("json", "all"):
        save_json(payload, os.path.join(output_dir, "predictions.json"))

    if output_format in ("jsonl", "all"):
        save_jsonl(results, os.path.join(output_dir, "predictions.jsonl"))

    if output_format in ("csv", "all"):
        rows = []
        for item in results:
            rows.append({
                "case_idx": item.get("case_idx"),
                "alarm_type": item.get("alarm_type"),
                "alarm_time": item.get("alarm_time"),
                "case_file_path": item.get("case_file_path"),
                "label_file_path": item.get("label_file_path"),
                "groundtruth": item.get("groundtruth"),
                "pred_top1_rc": item.get("pred_top1_rc"),
                "pred_rc": json.dumps(
                    item.get("pred_rc", []),
                    ensure_ascii=False,
                ),
                "rank": item.get("rank"),
                "is_correct": item.get("is_correct"),
            })
        save_csv(rows, os.path.join(output_dir, "predictions.csv"))

    save_json(payload.get("meta", {}), os.path.join(output_dir, "meta.json"))


# ============================================================
# 4. 动态扫描类别目录 / 发现 index
# ============================================================


def scan_label_root(
    label_root: str,
    scenario_arg: str = "all",
) -> List[Dict[str, Any]]:
    """
    扫描根目录下的直接子目录。

    每个子目录就是一个告警类别，不再写死四种类别。
    """
    root = Path(label_root).resolve()

    if not root.exists():
        raise FileNotFoundError(f"label 根目录不存在: {root}")
    if not root.is_dir():
        raise NotADirectoryError(f"label_root 不是目录: {root}")

    scenarios = []

    for child in sorted(root.iterdir(), key=lambda p: p.name):
        if not child.is_dir():
            continue

        alarm_type = child.name

        if scenario_arg != "all" and scenario_arg != alarm_type:
            continue

        scenarios.append({
            "name": alarm_type,
            "alarm_type": alarm_type,
            "data_dir": str(child.resolve()),
        })

    if scenario_arg != "all" and not scenarios:
        raise ValueError(
            f"在 {root} 下未找到类别子目录: {scenario_arg}"
        )

    return scenarios


def parse_index_from_label_filename(
    file_name: str,
    alarm_type: Optional[str] = None,
) -> Optional[int]:
    """
    支持：
        网络设备掉线_60_label_1674208771.json
        任意前缀_60_label_xxx.json
    """
    base = os.path.basename(file_name)

    if alarm_type:
        pattern = rf"^{re.escape(alarm_type)}_(\d+)_label_.*\.json$"
        match = re.match(pattern, base)
        if match:
            return int(match.group(1))

    match = re.search(r"_(\d+)_label_", base)
    if match:
        return int(match.group(1))

    return None


def discover_label_index_map(
    data_dir: str,
    alarm_type: str,
) -> Dict[int, str]:
    """返回 {index: label_file_path}。同一 index 多文件时取排序后的第一个。"""
    pattern = os.path.join(data_dir, "*_label_*.json")
    files = sorted(glob.glob(pattern))

    index_map: Dict[int, str] = {}

    for file_path in files:
        idx = parse_index_from_label_filename(file_path, alarm_type)
        if idx is None:
            continue

        index_map.setdefault(idx, file_path)

    return dict(sorted(index_map.items()))


def discover_label_indices(data_dir: str, alarm_type: str) -> List[int]:
    return list(discover_label_index_map(data_dir, alarm_type).keys())


# ============================================================
# 5. 增量状态管理
# ============================================================


def load_infer_state(state_file: str) -> Dict[str, Any]:
    if not os.path.exists(state_file):
        return {
            "version": 2,
            "folders": {},
        }

    data = load_json(state_file)
    if not isinstance(data, dict):
        raise ValueError(f"状态文件不是 JSON object: {state_file}")

    data.setdefault("version", 2)
    data.setdefault("folders", {})
    return data


def save_infer_state(state: Dict[str, Any], state_file: str) -> None:
    save_json(state, state_file)


def get_folder_mode_state(
    state: Dict[str, Any],
    mode: str,
    data_dir: str,
) -> Dict[str, Any]:
    folders = state.setdefault("folders", {})
    folder_state = folders.setdefault(str(Path(data_dir).resolve()), {})
    return folder_state.setdefault(mode, {})


def get_last_processed_index(
    state: Dict[str, Any],
    mode: str,
    data_dir: str,
    default_last_index: int,
) -> int:
    mode_state = get_folder_mode_state(state, mode, data_dir)
    return int(mode_state.get("last_index", default_last_index))


def update_last_processed_index(
    state: Dict[str, Any],
    mode: str,
    scenario: Dict[str, Any],
    last_index: int,
) -> None:
    mode_state = get_folder_mode_state(
        state,
        mode,
        scenario["data_dir"],
    )

    mode_state.update({
        "alarm_type": scenario["alarm_type"],
        "last_index": int(last_index),
        "updated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    })


def get_incremental_indices(
    scenario: Dict[str, Any],
    mode: str,
    args: argparse.Namespace,
    state: Dict[str, Any],
) -> Tuple[List[int], bool, int]:
    """
    返回：
        selected_indices
        should_update_state
        previous_last_index

    默认不传 start/end：只处理 idx > last_index 的新增数据，并更新状态。
    手动传 start/end：仅调试该窗口，不更新状态。
    """
    all_indices = discover_label_indices(
        scenario["data_dir"],
        scenario["alarm_type"],
    )

    if not all_indices:
        return [], False, getattr(config, "STATE_INITIAL_LAST_INDEX", -1)

    # 手动窗口：调试模式，不推进增量 checkpoint。
    if args.start is not None or args.end is not None:
        start = args.start if args.start is not None else min(all_indices)
        end = args.end if args.end is not None else max(all_indices) + 1

        selected = [idx for idx in all_indices if start <= idx < end]
        return selected, False, start - 1

    default_last_index = getattr(config, "STATE_INITIAL_LAST_INDEX", -1)

    if args.reset_state:
        previous_last_index = default_last_index
    else:
        previous_last_index = get_last_processed_index(
            state=state,
            mode=mode,
            data_dir=scenario["data_dir"],
            default_last_index=default_last_index,
        )

    selected = [idx for idx in all_indices if idx > previous_last_index]

    max_new_cases = args.max_new_cases
    if max_new_cases is None:
        max_new_cases = getattr(config, "MAX_NEW_CASES", None)

    if max_new_cases is not None:
        if max_new_cases <= 0:
            raise ValueError("max_new_cases 必须 > 0 或为 None")
        selected = selected[:max_new_cases]

    return selected, True, previous_last_index


def advance_checkpoint(
    selected_indices: List[int],
    success_indices: Set[int],
    previous_last_index: int,
) -> int:
    """
    只把 checkpoint 推进到“按本次 selected 顺序连续成功”的最后一条。

    例如：selected=[60, 61, 62]，60/62 成功、61 失败，
    checkpoint 只推进到 60，避免下次漏掉 61。
    """
    checkpoint = previous_last_index

    for idx in sorted(selected_indices):
        if idx not in success_indices:
            break
        checkpoint = idx

    return checkpoint


# ============================================================
# 6. 按 index 读取 case / label
# ============================================================


class IndexedCaseReader:
    def __init__(self, data_dir: str, alarm_type: str):
        self.data_dir = Path(data_dir)
        self.alarm_type = alarm_type

        if not self.data_dir.exists():
            raise FileNotFoundError(f"数据目录不存在: {self.data_dir}")

        self.case_file_patterns = getattr(config, "CASE_FILE_PATTERNS", [
            "{alarm_type}_{idx}_case_*.json",
            "{idx}_case_*.json",
            "case_{idx}.json",
            "{idx}.json",
        ])
        self.label_file_patterns = getattr(config, "LABEL_FILE_PATTERNS", [
            "{alarm_type}_{idx}_label_*.json",
            "{idx}_label_*.json",
            "label_{idx}.json",
            "{idx}.json",
        ])

    def _expand_patterns(
        self,
        patterns: List[str],
        idx: int,
    ) -> List[str]:
        return [
            str(
                self.data_dir
                / pattern.format(idx=idx, alarm_type=self.alarm_type)
            )
            for pattern in patterns
        ]

    @staticmethod
    def _find_first_match(patterns: List[str]) -> Optional[str]:
        for pattern in patterns:
            matched = sorted(glob.glob(pattern))
            if matched:
                return matched[0]
        return None

    def find_case_and_label_path(self, idx: int) -> Tuple[str, str]:
        case_patterns = self._expand_patterns(self.case_file_patterns, idx)
        label_patterns = self._expand_patterns(self.label_file_patterns, idx)

        case_file_path = self._find_first_match(case_patterns)
        label_file_path = self._find_first_match(label_patterns)

        if label_file_path is None:
            raise FileNotFoundError(
                f"找不到 index={idx} 对应的 label 文件。\n"
                + "\n".join(label_patterns)
            )

        # 你的 label 目录可能只有 label 文件。
        # 当前 load_alarm_data() 的核心字段来自 label，因此允许 label 兜底。
        if case_file_path is None:
            case_file_path = label_file_path

        return case_file_path, label_file_path

    def load_case(self, idx: int) -> Dict[str, Any]:
        case_file_path, label_file_path = self.find_case_and_label_path(idx)

        case = load_alarm_data(
            case_file_path,
            label_file_path,
            self.alarm_type,
        )

        case.setdefault("case_idx", idx)
        case.setdefault("alarm_type", self.alarm_type)
        case["_case_file_path"] = case_file_path
        case["_label_file_path"] = label_file_path
        return case

    def load_cases(self, indices: List[int]) -> List[Dict[str, Any]]:
        return [self.load_case(idx) for idx in indices]


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


# ============================================================
# 8. Tree 推理
# ============================================================


def extract_features_for_cases(
    cases: List[Dict[str, Any]],
    selection: Dict[str, Any],
):
    X = []
    feature_names = None
    features_list = []

    for case_data in cases:
        if "semantic_labels" not in case_data:
            raise KeyError(
                "case 缺少 semantic_labels，"
                f"case_idx={case_data.get('case_idx')}"
            )

        names, values, features_dict = extract_all_features(
            case_data["semantic_labels"],
            selection,
        )

        if feature_names is None:
            feature_names = names
        elif names != feature_names:
            raise ValueError(
                "不同 case 抽取出的 feature_names 不一致，"
                f"case_idx={case_data.get('case_idx')}"
            )

        X.append(values)
        features_list.append(features_dict)

    return feature_names, X, features_list


def extract_labels(
    train_cases: List[Dict[str, Any]],
) -> Tuple[np.ndarray, LabelEncoder]:
    raw_labels = []

    for case in train_cases:
        root_cause = case.get("root_cause")

        if root_cause is None or root_cause == "":
            raise ValueError(
                "训练数据 root_cause 为空，"
                f"case_idx={case.get('case_idx')}"
            )

        raw_labels.append(root_cause)

    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(raw_labels)
    return np.array(y), label_encoder


def train_tree_model(
    train_cases: List[Dict[str, Any]],
    selection: Dict[str, Any],
):
    feature_names, X_train, _ = extract_features_for_cases(
        train_cases,
        selection,
    )
    y_train, label_encoder = extract_labels(train_cases)

    clf = DecisionTreeClassifier(
        max_depth=getattr(config, "MAX_DEPTH", 3),
        min_samples_leaf=getattr(config, "MIN_SAMPLES_LEAF", 10),
        random_state=getattr(config, "RANDOM_STATE", 42),
    )

    clf.fit(np.array(X_train), y_train)

    return clf, feature_names, label_encoder


def save_tree_rules(
    clf,
    feature_names,
    label_encoder,
    output_dir: str,
    scenario_name: str,
    tag: str,
) -> str:
    ensure_dir(output_dir)

    output_path = os.path.join(
        output_dir,
        f"{scenario_name}_{tag}.txt",
    )

    tree_rules = export_text(
        clf,
        feature_names=list(feature_names),
    )

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(tree_rules)
        f.write("\n\nClass mapping:\n")

        for idx, cls in enumerate(label_encoder.classes_):
            f.write(f"class {idx} -> {cls}\n")

    return output_path


def predict_by_tree(
    infer_cases,
    infer_indices,
    clf,
    feature_names,
    label_encoder,
    selection,
):
    results = []

    for case, idx in zip(infer_cases, infer_indices):
        names, X_values, features_list = extract_features_for_cases(
            [case],
            selection,
        )

        if names != feature_names:
            raise ValueError(
                f"case_idx={idx} 的特征名和训练集不一致"
            )

        X = np.array(X_values)
        pred_id = clf.predict(X)[0]
        pred_top1 = label_encoder.inverse_transform([pred_id])[0]

        pred_rc = [pred_top1]
        pred_scores = []

        if hasattr(clf, "predict_proba"):
            proba = clf.predict_proba(X)[0]
            pairs = sorted(
                zip(clf.classes_, proba),
                key=lambda item: item[1],
                reverse=True,
            )

            class_ids = [int(item[0]) for item in pairs]
            scores = [float(item[1]) for item in pairs]
            pred_rc = label_encoder.inverse_transform(class_ids).tolist()
            pred_scores = [
                {"root_cause": rc, "score": score}
                for rc, score in zip(pred_rc, scores)
            ]

        groundtruth = case.get("root_cause")
        rank = None
        is_correct = None

        if groundtruth is not None and groundtruth != "":
            rank = (
                pred_rc.index(groundtruth) + 1
                if groundtruth in pred_rc
                else None
            )
            is_correct = pred_top1 == groundtruth

        results.append({
            "case_idx": idx,
            "alarm_type": case.get("alarm_type"),
            "alarm_time": case.get("alarm_time"),
            "case_file_path": case.get("_case_file_path"),
            "label_file_path": case.get("_label_file_path"),
            "groundtruth": groundtruth,
            "pred_top1_rc": pred_top1,
            "pred_rc": pred_rc,
            "pred_scores": pred_scores,
            "rank": rank,
            "is_correct": is_correct,
            "features": features_list[0] if features_list else None,
        })

    return results


def run_tree_once(
    scenario: Dict[str, Any],
    train_cases: List[Dict[str, Any]],
    infer_cases: List[Dict[str, Any]],
    train_indices: List[int],
    infer_indices: List[int],
    selection: Dict[str, Any],
    output_dir: str,
    output_format: str,
    tag: str,
) -> Dict[str, Any]:
    scenario_name = scenario["name"]
    alarm_type = scenario["alarm_type"]

    clf, feature_names, label_encoder = train_tree_model(
        train_cases,
        selection,
    )

    tree_rule_path = save_tree_rules(
        clf,
        feature_names,
        label_encoder,
        os.path.join(output_dir, "tree_rules"),
        scenario_name,
        tag,
    )

    results = predict_by_tree(
        infer_cases,
        infer_indices,
        clf,
        feature_names,
        label_encoder,
        selection,
    )

    labeled = [
        item for item in results
        if item.get("is_correct") is not None
    ]
    correct = sum(
        1 for item in labeled
        if item.get("is_correct") is True
    )
    total = len(labeled)
    accuracy = correct / total if total > 0 else None

    payload = {
        "meta": {
            "mode": "tree_infer",
            "tag": tag,
            "scenario_name": scenario_name,
            "alarm_type": alarm_type,
            "data_dir": scenario.get("data_dir"),
            "train_indices": train_indices,
            "infer_indices": infer_indices,
            "tree_rule_path": tree_rule_path,
            "max_depth": getattr(config, "MAX_DEPTH", 3),
            "min_samples_leaf": getattr(
                config,
                "MIN_SAMPLES_LEAF",
                10,
            ),
            "random_state": getattr(config, "RANDOM_STATE", 42),
            "output_dir": output_dir,
        },
        "selection": selection,
        "summary": {
            "labeled_total": total,
            "correct": correct,
            "accuracy": accuracy,
            "processed_count": len(results),
            "processed_indices": [
                item["case_idx"] for item in results
            ],
        },
        "results": results,
    }

    save_infer_outputs(payload, output_dir, output_format)
    return payload


def tree_infer_incremental_one_scenario(
    scenario: Dict[str, Any],
    infer_indices: List[int],
    train_n: int,
    selection_path: Optional[str],
    selection_source: str,
    refiner_rounds: int,
    output_dir: Optional[str],
    output_format: str,
) -> Dict[str, Any]:
    scenario_name = scenario["name"]
    alarm_type = scenario["alarm_type"]
    data_dir = scenario["data_dir"]

    if not infer_indices:
        raise ValueError(f"[{scenario_name}] infer_indices 为空")

    all_indices = discover_label_indices(data_dir, alarm_type)
    first_infer_idx = min(infer_indices)

    train_candidates = [
        idx for idx in all_indices
        if idx < first_infer_idx
    ]
    train_indices = train_candidates[-train_n:]

    if len(train_indices) < train_n:
        raise ValueError(
            f"[{scenario_name}] 训练数据不足："
            f"需要 {train_n} 条，当前只有 {len(train_indices)} 条；"
            f"first_infer_idx={first_infer_idx}"
        )

    run_name = (
        f"train_{min(train_indices)}_{max(train_indices)}_"
        f"infer_{min(infer_indices)}_{max(infer_indices)}"
    )

    print("=" * 100)
    print(f"[tree_infer][incremental] 场景: {scenario_name}")
    print(f"data_dir         : {data_dir}")
    print(f"train_indices    : {train_indices}")
    print(f"infer_indices    : {infer_indices}")
    print(f"selection_source : {selection_source}")
    print("=" * 100)

    reader = IndexedCaseReader(
        data_dir=data_dir,
        alarm_type=alarm_type,
    )

    train_cases = reader.load_cases(train_indices)
    infer_cases = reader.load_cases(infer_indices)

    selection_work_dir = build_output_dir(
        "selection_pipeline",
        scenario_name,
        output_dir,
        run_name,
    )

    final_output_dir = build_output_dir(
        "tree_infer",
        scenario_name,
        output_dir,
        run_name,
    )

    if selection_source == "file":
        selection = load_selection(scenario, selection_path)

    elif selection_source == "selector":
        selection = generate_selection_by_selector(
            scenario,
            selection_work_dir,
        )

    elif selection_source == "selector_refiner":
        selection = generate_selection_by_selector(
            scenario,
            selection_work_dir,
        )

        for round_id in range(refiner_rounds):
            round_dir = os.path.join(
                selection_work_dir,
                f"refiner_round_{round_id}",
                "tree",
            )
            ensure_dir(round_dir)

            tmp_payload = run_tree_once(
                scenario=scenario,
                train_cases=train_cases,
                infer_cases=infer_cases,
                train_indices=train_indices,
                infer_indices=infer_indices,
                selection=selection,
                output_dir=round_dir,
                output_format="json",
                tag=f"refiner_round_{round_id}",
            )

            tree_summary = build_tree_summary_for_refiner(tmp_payload)

            save_json(
                tree_summary,
                os.path.join(
                    selection_work_dir,
                    f"tree_summary_round_{round_id}.json",
                ),
            )

            selection = refine_selection_by_tree_summary(
                scenario=scenario,
                previous_selection=selection,
                tree_summary=tree_summary,
                output_dir=selection_work_dir,
                round_id=round_id,
            )

    else:
        raise ValueError(
            f"未知 selection_source: {selection_source}"
        )

    save_json(
        selection,
        os.path.join(final_output_dir, "selection_final.json"),
    )

    payload = run_tree_once(
        scenario=scenario,
        train_cases=train_cases,
        infer_cases=infer_cases,
        train_indices=train_indices,
        infer_indices=infer_indices,
        selection=selection,
        output_dir=final_output_dir,
        output_format=output_format,
        tag="final",
    )

    accuracy = payload.get("summary", {}).get("accuracy")
    print(f"[Done][tree_infer] {scenario_name}")
    print(f"output_dir: {final_output_dir}")

    if accuracy is not None:
        print(f"accuracy  : {accuracy:.4f}")

    return payload


def tree_infer(args: argparse.Namespace) -> List[Dict[str, Any]]:
    scenarios = scan_label_root(
        args.anomalydetect_label_root,
        args.scenario,
    )

    train_n = (
        args.train_n
        if args.train_n is not None
        else getattr(config, "TRAIN_N", 50)
    )

    state = load_infer_state(args.state_file)
    payloads = []

    for scenario in scenarios:
        indices, should_update_state, previous_last_index = (
            get_incremental_indices(
                scenario=scenario,
                mode="tree_infer",
                args=args,
                state=state,
            )
        )

        if not indices:
            print(
                f"[tree_infer] {scenario['name']} 没有新增数据，跳过"
            )
            continue

        try:
            payload = tree_infer_incremental_one_scenario(
                scenario=scenario,
                infer_indices=indices,
                train_n=train_n,
                selection_path=args.selection_path,
                selection_source=args.selection_source,
                refiner_rounds=args.refiner_rounds,
                output_dir=args.output_dir,
                output_format=args.output_format,
            )
            payloads.append(payload)

            success_indices = set(
                payload.get("summary", {}).get(
                    "processed_indices",
                    [],
                )
            )

            if should_update_state:
                new_checkpoint = advance_checkpoint(
                    selected_indices=indices,
                    success_indices=success_indices,
                    previous_last_index=previous_last_index,
                )

                if new_checkpoint > previous_last_index:
                    update_last_processed_index(
                        state=state,
                        mode="tree_infer",
                        scenario=scenario,
                        last_index=new_checkpoint,
                    )
                    save_infer_state(state, args.state_file)

        except Exception as exc:
            print(
                f"[Error][tree_infer] {scenario['name']} 处理失败: {exc}"
            )
            traceback.print_exc()

            if args.strict:
                raise

    return payloads


# ============================================================
# 9. LLM 推理：完整 Competition / Cooperation
# ============================================================


def _lazy_import_llm_modules():
    from llm_inference.generator import RCAGenerator
    from utils.public_functions import (
        load_alarm_data_meta,
        load_alarm_data_reasoner_cooperation,
        load_alarm_data_verifier,
    )

    return (
        RCAGenerator,
        load_alarm_data_meta,
        load_alarm_data_reasoner_cooperation,
        load_alarm_data_verifier,
    )


def llm_find_case_and_label(
    data_dir: str,
    alarm_type: str,
    idx: int,
) -> Optional[Tuple[str, str]]:
    reader = IndexedCaseReader(data_dir, alarm_type)

    try:
        return reader.find_case_and_label_path(idx)
    except FileNotFoundError:
        return None


def get_sop_value_from_label(label_file_path: str) -> Any:
    """SOP 必须从 label["semantic_labels"]["sop"] 读取。"""
    label_data = load_json(label_file_path)
    semantic_labels = label_data.get("semantic_labels", {})

    if not isinstance(semantic_labels, dict):
        return ""

    return semantic_labels.get("sop", "")


def is_non_empty_sop(sop: Any) -> bool:
    if sop is None:
        return False
    if isinstance(sop, str):
        return bool(sop.strip())
    if isinstance(sop, (list, dict, tuple, set)):
        return len(sop) > 0
    return bool(str(sop).strip())


def split_indices_by_sop(
    data_dir: str,
    alarm_type: str,
    indices: List[int],
) -> Tuple[List[int], List[int]]:
    """
    分流规则：
        semantic_labels["sop"] 非空 -> competition
        semantic_labels["sop"] 为空 -> cooperation
    """
    cooperation_indices: List[int] = []
    competition_indices: List[int] = []

    for idx in indices:
        found = llm_find_case_and_label(data_dir, alarm_type, idx)
        if found is None:
            continue

        _case_file, label_file = found
        sop = get_sop_value_from_label(label_file)

        if is_non_empty_sop(sop):
            competition_indices.append(idx)
        else:
            cooperation_indices.append(idx)

    return cooperation_indices, competition_indices


def build_competition_alarm_data(
    label_file: str,
    alarm_type: str,
) -> Dict[str, Any]:
    """直接从 label 构建 Competition 输入，确保 SOP 来源正确。"""
    label_data = load_json(label_file)
    semantic_labels = label_data.get("semantic_labels", {})
    if not isinstance(semantic_labels, dict):
        semantic_labels = {}

    return {
        "alarm_type": alarm_type,
        "semantic_labels": semantic_labels,
        "alarm_time": label_data.get(
            "alarm_time",
            semantic_labels.get("alarm_time"),
        ),
        "sop": semantic_labels.get("sop", ""),
        "root_cause_candidates": label_data.get("root_cause_candidates", []),
    }


def llm_competition_one_scenario(
    scenario: Dict[str, Any],
    indices: List[int],
    rca_generator,
    load_alarm_data_verifier,
    output_dir: Optional[str],
    output_format: str,
    batch_size: int,
) -> Dict[str, Any]:
    """
    Competition 完整流程：
        多采样 Competition -> Verifier
    """
    scenario_name = scenario["name"]
    alarm_type = scenario["alarm_type"]
    data_dir = scenario["data_dir"]

    run_name = f"idx_{min(indices)}_{max(indices)}" if indices else "empty"
    final_output_dir = build_output_dir(
        "llm_infer_competition",
        scenario_name,
        output_dir,
        run_name,
    )

    processed: List[int] = []
    skipped: List[int] = []

    for batch_indices in iter_batches(indices, batch_size):
        alarm_data_list = []
        label_files_list = []
        valid_indices = []
        meta_output_dirs = []
        verifier_output_dirs = []

        for idx in batch_indices:
            found = llm_find_case_and_label(data_dir, alarm_type, idx)
            if found is None:
                skipped.append(idx)
                continue

            _case_file, label_file = found
            try:
                alarm_data = build_competition_alarm_data(label_file, alarm_type)

                meta_output_dir = os.path.join(
                    final_output_dir,
                    f"{alarm_type}_analysis_Meta_{idx}",
                )
                verifier_output_dir = os.path.join(
                    final_output_dir,
                    f"{alarm_type}_analysis_Reasoner_{idx}",
                )
                ensure_dir(meta_output_dir)
                ensure_dir(verifier_output_dir)

                alarm_data_list.append(alarm_data)
                label_files_list.append(label_file)
                valid_indices.append(idx)
                meta_output_dirs.append(meta_output_dir)
                verifier_output_dirs.append(verifier_output_dir)
            except Exception as exc:
                print(f"idx={idx}: Competition 数据准备失败 - {exc}")
                skipped.append(idx)

        if not alarm_data_list:
            continue

        try:
            meta_responses = rca_generator.generate_rca_analysis_competition_batch(
                alarm_data_list,
                meta_output_dirs,
            )
        except Exception as exc:
            print(f"Competition 分析失败: {exc}")
            skipped.extend(valid_indices)
            continue

        meta_success_count = min(len(meta_responses), len(valid_indices))
        skipped.extend(valid_indices[meta_success_count:])

        verifier_data_list = []
        valid_verifier_dirs = []
        valid_verifier_indices = []

        for pos in range(meta_success_count):
            idx = valid_indices[pos]
            label_file = label_files_list[pos]
            current_reasoner_file = os.path.join(
                meta_output_dirs[pos],
                "raw_responses.txt",
            )

            if not os.path.exists(current_reasoner_file):
                print(f"idx={idx}: 未找到 Competition 输出文件: {current_reasoner_file}")
                skipped.append(idx)
                continue

            try:
                verifier_data = load_alarm_data_verifier(
                    label_file,
                    current_reasoner_file,
                    alarm_type,
                )
                verifier_data["sop"] = get_sop_value_from_label(label_file)
                verifier_data_list.append(verifier_data)
                valid_verifier_dirs.append(verifier_output_dirs[pos])
                valid_verifier_indices.append(idx)
            except Exception as exc:
                print(f"idx={idx}: Verifier 数据准备失败 - {exc}")
                skipped.append(idx)

        if not verifier_data_list:
            continue

        try:
            verifier_responses = rca_generator.generate_rca_analysis_verifier_batch(
                verifier_data_list,
                valid_verifier_dirs,
            )
            success_count = min(len(verifier_responses), len(valid_verifier_indices))
            processed.extend(valid_verifier_indices[:success_count])
            skipped.extend(valid_verifier_indices[success_count:])
        except Exception as exc:
            print(f"Verifier 分析失败: {exc}")
            skipped.extend(valid_verifier_indices)

    processed = sorted(set(processed))
    skipped = sorted(set(skipped) - set(processed))

    payload = {
        "meta": {
            "mode": "llm_infer",
            "llm_mode": "competition",
            "meta_only": False,
            "pipeline": "competition_then_verifier",
            "scenario_name": scenario_name,
            "alarm_type": alarm_type,
            "data_dir": data_dir,
            "indices": indices,
            "batch_size": batch_size,
            "model_path": getattr(config, "MODEL_PATH"),
            "output_dir": final_output_dir,
        },
        "summary": {
            "processed_count": len(processed),
            "skipped_count": len(skipped),
            "processed_indices": processed,
            "skipped_indices": skipped,
        },
        "results": [],
    }
    save_infer_outputs(payload, final_output_dir, output_format)
    return payload


def llm_cooperation_one_scenario(
    scenario: Dict[str, Any],
    indices: List[int],
    rca_generator,
    load_alarm_data_meta,
    load_alarm_data_reasoner_cooperation,
    output_dir: Optional[str],
    output_format: str,
    batch_size: int,
    max_rounds: int,
) -> Dict[str, Any]:
    """
    Cooperation 完整流程：
        Meta -> Reasoner -> 下一轮 Meta/Reasoner，直到早停或 max_rounds。
    """
    scenario_name = scenario["name"]
    alarm_type = scenario["alarm_type"]
    data_dir = scenario["data_dir"]

    run_name = f"idx_{min(indices)}_{max(indices)}" if indices else "empty"
    final_output_dir = build_output_dir(
        "llm_infer_cooperation",
        scenario_name,
        output_dir,
        run_name,
    )

    processed: List[int] = []
    skipped: List[int] = []
    early_stop_marker = '["(上一轮次已经符合要求)"]'

    for batch_indices in iter_batches(indices, batch_size):
        label_file_by_idx: Dict[int, str] = {}
        valid_indices: List[int] = []

        for idx in batch_indices:
            found = llm_find_case_and_label(data_dir, alarm_type, idx)
            if found is None:
                skipped.append(idx)
                continue
            _case_file, label_file = found
            label_file_by_idx[idx] = label_file
            valid_indices.append(idx)

        if not valid_indices:
            continue

        early_stop_flags = {idx: False for idx in valid_indices}
        current_rounds = {idx: 0 for idx in valid_indices}
        failed_indices: Set[int] = set()

        for round_num in range(max_rounds):
            print(f"[Cooperation] round={round_num}, batch={batch_indices}")
            active_indices = [
                idx for idx in valid_indices
                if (
                    not early_stop_flags[idx]
                    and idx not in failed_indices
                    and current_rounds[idx] == round_num
                )
            ]
            if not active_indices:
                break

            meta_data_list = []
            meta_output_dirs = []
            meta_indices = []

            for idx in active_indices:
                label_file = label_file_by_idx[idx]
                current_round = current_rounds[idx]
                last_meta_output = None
                last_reasoner_output = None

                if current_round > 0:
                    prev_round = current_round - 1
                    last_meta_output = os.path.join(
                        final_output_dir,
                        str(idx),
                        f"round{prev_round}",
                        "meta",
                        "raw_responses.txt",
                    )
                    last_reasoner_output = os.path.join(
                        final_output_dir,
                        str(idx),
                        f"round{prev_round}",
                        "reasoner",
                        "raw_responses.txt",
                    )

                    if os.path.exists(last_meta_output):
                        try:
                            with open(last_meta_output, "r", encoding="utf-8") as f:
                                content = f.read()
                            if early_stop_marker in content:
                                print(f"idx={idx}: 上一轮 Meta 已符合要求，早停")
                                early_stop_flags[idx] = True
                                continue
                        except Exception as exc:
                            print(f"idx={idx}: 读取上一轮 Meta 失败 - {exc}")
                            failed_indices.add(idx)
                            continue

                output_dir_meta = os.path.join(
                    final_output_dir,
                    str(idx),
                    f"round{current_round}",
                    "meta",
                )
                ensure_dir(output_dir_meta)

                try:
                    meta_data = load_alarm_data_meta(
                        label_file,
                        alarm_type,
                        last_meta_output,
                        last_reasoner_output,
                    )
                    meta_data["sop"] = get_sop_value_from_label(label_file)
                    meta_data_list.append(meta_data)
                    meta_output_dirs.append(output_dir_meta)
                    meta_indices.append(idx)
                except Exception as exc:
                    print(f"idx={idx}: Cooperation Meta 数据加载失败 - {exc}")
                    failed_indices.add(idx)

            if not meta_data_list:
                continue

            try:
                meta_responses = rca_generator.generate_rca_analysis_meta(
                    meta_data_list,
                    meta_output_dirs,
                    meta_indices,
                )
            except Exception as exc:
                print(f"Cooperation Meta 批处理失败: {exc}")
                failed_indices.update(meta_indices)
                continue

            meta_success_count = min(len(meta_responses), len(meta_indices))
            meta_success_indices = meta_indices[:meta_success_count]
            failed_indices.update(meta_indices[meta_success_count:])

            for idx in meta_success_indices:
                meta_file = os.path.join(
                    final_output_dir,
                    str(idx),
                    f"round{current_rounds[idx]}",
                    "meta",
                    "raw_responses.txt",
                )
                if not os.path.exists(meta_file):
                    print(f"idx={idx}: Meta 输出不存在: {meta_file}")
                    failed_indices.add(idx)
                    continue
                try:
                    with open(meta_file, "r", encoding="utf-8") as f:
                        content = f.read()
                    if early_stop_marker in content:
                        print(f"idx={idx}: 本轮 Meta 已符合要求，早停")
                        early_stop_flags[idx] = True
                except Exception as exc:
                    print(f"idx={idx}: 读取 Meta 输出失败 - {exc}")
                    failed_indices.add(idx)

            reasoner_data_list = []
            reasoner_output_dirs = []
            reasoner_indices = []

            for idx in meta_success_indices:
                if early_stop_flags[idx] or idx in failed_indices:
                    continue

                current_round = current_rounds[idx]
                meta_file = os.path.join(
                    final_output_dir,
                    str(idx),
                    f"round{current_round}",
                    "meta",
                    "raw_responses.txt",
                )
                try:
                    reasoner_data = load_alarm_data_reasoner_cooperation(
                        label_file_by_idx[idx],
                        meta_file,
                        alarm_type,
                    )
                    output_dir_reasoner = os.path.join(
                        final_output_dir,
                        str(idx),
                        f"round{current_round}",
                        "reasoner",
                    )
                    ensure_dir(output_dir_reasoner)
                    reasoner_data_list.append(reasoner_data)
                    reasoner_output_dirs.append(output_dir_reasoner)
                    reasoner_indices.append(idx)
                except Exception as exc:
                    print(f"idx={idx}: Reasoner 数据加载失败 - {exc}")
                    failed_indices.add(idx)

            if not reasoner_data_list:
                continue

            try:
                reasoner_responses = rca_generator.generate_rca_analysis_reasoner(
                    reasoner_data_list,
                    reasoner_output_dirs,
                    reasoner_indices,
                )
            except Exception as exc:
                print(f"Reasoner 批处理失败: {exc}")
                failed_indices.update(reasoner_indices)
                continue

            reasoner_success_count = min(len(reasoner_responses), len(reasoner_indices))
            reasoner_success_indices = reasoner_indices[:reasoner_success_count]
            failed_indices.update(reasoner_indices[reasoner_success_count:])
            for idx in reasoner_success_indices:
                current_rounds[idx] += 1

        batch_processed = [idx for idx in valid_indices if idx not in failed_indices]
        processed.extend(batch_processed)
        skipped.extend(sorted(failed_indices))

    processed = sorted(set(processed))
    skipped = sorted(set(skipped) - set(processed))

    payload = {
        "meta": {
            "mode": "llm_infer",
            "llm_mode": "cooperation",
            "meta_only": False,
            "pipeline": "multi_round_meta_reasoner",
            "scenario_name": scenario_name,
            "alarm_type": alarm_type,
            "data_dir": data_dir,
            "indices": indices,
            "batch_size": batch_size,
            "max_rounds": max_rounds,
            "model_path": getattr(config, "MODEL_PATH"),
            "output_dir": final_output_dir,
        },
        "summary": {
            "processed_count": len(processed),
            "skipped_count": len(skipped),
            "processed_indices": processed,
            "skipped_indices": skipped,
        },
        "results": [],
    }
    save_infer_outputs(payload, final_output_dir, output_format)
    return payload


def llm_infer(args: argparse.Namespace) -> List[Dict[str, Any]]:
    scenarios = scan_label_root(
        args.agentdigest_label_root,
        args.scenario,
    )

    batch_size = (
        args.batch_size
        if args.batch_size is not None
        else getattr(config, "BATCH_SIZE", 8)
    )

    state = load_infer_state(args.state_file)
    plans = []

    for scenario in scenarios:
        indices, should_update_state, previous_last_index = get_incremental_indices(
            scenario=scenario,
            mode="llm_infer",
            args=args,
            state=state,
        )
        if not indices:
            print(f"[llm_infer] {scenario['name']} 没有新增数据，跳过")
            continue
        plans.append((scenario, indices, should_update_state, previous_last_index))

    if not plans:
        return []

    (
        RCAGenerator,
        load_alarm_data_meta,
        load_alarm_data_reasoner_cooperation,
        load_alarm_data_verifier,
    ) = _lazy_import_llm_modules()
    rca_generator = RCAGenerator(getattr(config, "MODEL_PATH"))

    payloads = []

    for scenario, indices, should_update_state, previous_last_index in plans:
        if args.llm_mode == "competition":
            competition_indices = list(indices)
            cooperation_indices = []
        elif args.llm_mode == "cooperation":
            cooperation_indices = list(indices)
            competition_indices = []
        else:
            cooperation_indices, competition_indices = split_indices_by_sop(
                scenario["data_dir"],
                scenario["alarm_type"],
                indices,
            )

        print("=" * 100)
        print(f"[llm_infer][incremental] 场景: {scenario['name']}")
        print(f"data_dir          : {scenario['data_dir']}")
        print(f"indices           : {indices}")
        print(f"llm_mode          : {args.llm_mode}")
        print("full_pipeline     : True")
        print(
            "cooperation count : "
            f"{len(cooperation_indices)}，条件：semantic_labels['sop'] 为空"
        )
        print(
            "competition count : "
            f"{len(competition_indices)}，条件：semantic_labels['sop'] 非空"
        )
        print("=" * 100)

        scenario_payloads = []
        success_indices: Set[int] = set()

        try:
            if cooperation_indices:
                payload = llm_cooperation_one_scenario(
                    scenario=scenario,
                    indices=cooperation_indices,
                    rca_generator=rca_generator,
                    load_alarm_data_meta=load_alarm_data_meta,
                    load_alarm_data_reasoner_cooperation=load_alarm_data_reasoner_cooperation,
                    output_dir=args.output_dir,
                    output_format=args.output_format,
                    batch_size=batch_size,
                    max_rounds=args.max_rounds,
                )
                scenario_payloads.append(payload)
                success_indices.update(
                    payload.get("summary", {}).get("processed_indices", [])
                )

            if competition_indices:
                payload = llm_competition_one_scenario(
                    scenario=scenario,
                    indices=competition_indices,
                    rca_generator=rca_generator,
                    load_alarm_data_verifier=load_alarm_data_verifier,
                    output_dir=args.output_dir,
                    output_format=args.output_format,
                    batch_size=batch_size,
                )
                scenario_payloads.append(payload)
                success_indices.update(
                    payload.get("summary", {}).get("processed_indices", [])
                )

            payloads.extend(scenario_payloads)

            if should_update_state:
                new_checkpoint = advance_checkpoint(
                    selected_indices=indices,
                    success_indices=success_indices,
                    previous_last_index=previous_last_index,
                )
                if new_checkpoint > previous_last_index:
                    update_last_processed_index(
                        state=state,
                        mode="llm_infer",
                        scenario=scenario,
                        last_index=new_checkpoint,
                    )
                    save_infer_state(state, args.state_file)

        except Exception as exc:
            print(f"[Error][llm_infer] {scenario['name']} 处理失败: {exc}")
            traceback.print_exc()
            if args.strict:
                raise

    return payloads


# ============================================================
# 10. main
# ============================================================


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "扫描两个 label 根目录的类别子目录，"
            "按状态文件只处理新增数据。默认同时执行 tree_infer 和 llm_infer。"
        )
    )

    parser.add_argument(
        "--infer-type",
        choices=["all", "tree_infer", "llm_infer"],
        default=getattr(config, "DEFAULT_INFER_TYPE", "all"),
        help="默认 all：同时执行 tree_infer 和 llm_infer。",
    )

    parser.add_argument(
        "--scenario",
        default="all",
        help="默认 all：处理根目录下所有类别子目录；也可指定单个子目录名。",
    )

    # 两个输入根目录
    parser.add_argument(
        "--agentdigest-label-root",
        default=getattr(
            config,
            "AGENTDIGEST_LABEL_ROOT",
            "/home/sbp/deployment/case_pool/AgentDigest_label",
        ),
        help="LLM 推理数据根目录。",
    )

    parser.add_argument(
        "--anomalydetect-label-root",
        default=getattr(
            config,
            "ANOMALYDETECT_LABEL_ROOT",
            "/home/sbp/deployment/case_pool/anomalydetect_label",
        ),
        help="Tree 推理数据根目录。",
    )

    # 增量状态
    parser.add_argument(
        "--state-file",
        default=getattr(
            config,
            "INFER_STATE_FILE",
            "/home/sbp/deployment/case_pool/predict_result/state/infer_state.json",
        ),
        help="记录每个文件夹、每种推理上次处理到的 index。",
    )

    parser.add_argument(
        "--reset-state",
        action="store_true",
        help="本次忽略已有 checkpoint，从 STATE_INITIAL_LAST_INDEX 后重新扫描。",
    )

    parser.add_argument(
        "--max-new-cases",
        type=int,
        default=getattr(config, "MAX_NEW_CASES", None),
        help="每个类别本次最多处理多少条新增数据；默认不限制。",
    )

    # 手动调试窗口；使用后不更新 state。
    parser.add_argument(
        "--start",
        type=int,
        default=None,
        help="调试窗口起点；传入 start/end 后不更新增量状态。",
    )

    parser.add_argument(
        "--end",
        type=int,
        default=None,
        help="调试窗口终点（不含）；传入 start/end 后不更新增量状态。",
    )

    # Tree
    parser.add_argument(
        "--train-n",
        type=int,
        default=None,
        help="Tree 使用当前新增数据之前最近多少条样本训练。",
    )

    parser.add_argument(
        "--selection-path",
        default=None,
        help="selection.json、selector/res.jsonl 或包含它们的目录。",
    )

    parser.add_argument(
        "--selection-source",
        choices=["file", "selector", "selector_refiner"],
        default=getattr(config, "SELECTION_SOURCE", "file"),
        help=(
            "file=读取已有 selection；"
            "selector=现场生成；"
            "selector_refiner=Selector -> Tree summary -> Refiner。"
        ),
    )

    parser.add_argument(
        "--refiner-rounds",
        type=int,
        default=getattr(config, "REFINER_ROUNDS", 1),
        help="selector_refiner 的 Refiner 轮数。",
    )

    # LLM
    parser.add_argument(
        "--llm-mode",
        choices=["auto", "competition", "cooperation"],
        default=getattr(config, "DEFAULT_LLM_MODE", "auto"),
        help=(
            "默认 auto：semantic_labels['sop'] 非空走 competition，"
            "为空走 cooperation。Competition=多采样+Verifier；"
            "Cooperation=多轮 Meta/Reasoner。"
        ),
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="LLM batch size；默认使用 config.BATCH_SIZE。",
    )

    parser.add_argument(
        "--max-rounds",
        type=int,
        default=getattr(config, "LLM_COOPERATION_MAX_ROUNDS", 5),
        help="Cooperation 最大 Meta/Reasoner 迭代轮数。",
    )

    # 输出
    parser.add_argument(
        "--output-dir",
        default=None,
        help="自定义输出根目录；默认使用 config.PREDICT_RES_DIR。",
    )

    parser.add_argument(
        "--output-format",
        choices=["json", "jsonl", "csv", "all"],
        default="all",
    )

    parser.add_argument(
        "--strict",
        action="store_true",
        help="任一类别失败时立即抛出异常；默认记录错误后继续其他类别。",
    )

    return parser


def _run_tree_process(args: argparse.Namespace) -> None:
    tree_infer(args)


def _run_llm_process(args: argparse.Namespace) -> None:
    llm_infer(args)


def run_all_in_separate_processes(args: argparse.Namespace) -> None:
    """
    Tree 与 LLM 使用两个独立 spawn 进程，避免 Selector/Refiner 与 RCAGenerator
    在同一个已初始化 NPU 的 Python 进程中互相影响。
    """
    ctx = mp.get_context("spawn")

    print("=" * 100)
    print("[all] 启动 tree_infer 独立进程")
    print("=" * 100)
    tree_process = ctx.Process(
        target=_run_tree_process,
        args=(args,),
        name="tree_infer_process",
    )
    tree_process.start()
    tree_process.join()
    if tree_process.exitcode != 0:
        raise RuntimeError(f"tree_infer 进程失败，exitcode={tree_process.exitcode}")

    print("=" * 100)
    print("[all] 启动 llm_infer 独立进程")
    print("=" * 100)
    llm_process = ctx.Process(
        target=_run_llm_process,
        args=(args,),
        name="llm_infer_process",
    )
    llm_process.start()
    llm_process.join()
    if llm_process.exitcode != 0:
        raise RuntimeError(f"llm_infer 进程失败，exitcode={llm_process.exitcode}")


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    print("=" * 100)
    print("inference_engine 增量推理启动")
    print(f"infer_type               : {args.infer_type}")
    print(f"scenario                 : {args.scenario}")
    print(f"AgentDigest_label root   : {args.agentdigest_label_root}")
    print(f"anomalydetect_label root : {args.anomalydetect_label_root}")
    print(f"state_file               : {args.state_file}")
    print("=" * 100)

    if args.infer_type == "tree_infer":
        tree_infer(args)
    elif args.infer_type == "llm_infer":
        llm_infer(args)
    elif args.infer_type == "all":
        run_all_in_separate_processes(args)
    else:
        raise ValueError(f"未知 infer_type: {args.infer_type}")

    print("所有增量推理任务完成。")


if __name__ == "__main__":
    main()
