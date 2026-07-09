# inference/common.py
# -*- coding: utf-8 -*-

import argparse
import glob
import json
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

try:
    import numpy as np
except ImportError:  # pragma: no cover - optional at import time
    np = None

import config
from utils.public_functions import (
    load_alarm_data,
    load_json as util_load_json,
)


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
    if np is not None and isinstance(obj, np.integer):
        return int(obj)
    if np is not None and isinstance(obj, np.floating):
        return float(obj)
    if np is not None and isinstance(obj, np.ndarray):
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
    import pandas as pd

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
