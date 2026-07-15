# -*- coding: utf-8 -*-
"""Fill target label root_cause values by matching CSN against another root."""

import argparse
import json
import os
import re
import shutil
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import config
from utils.public_functions import load_json, save_json


def discover_label_files(root: str) -> List[Path]:
    root_path = Path(root).resolve()
    if not root_path.is_dir():
        raise NotADirectoryError(f"目录不存在或不是目录: {root_path}")
    return sorted(
        path for path in root_path.rglob("*.json")
        if "label" in path.name.lower()
    )


def _walk_csn_values(value: Any) -> Iterable[Any]:
    if isinstance(value, dict):
        for key, child in value.items():
            if str(key).lower() == "csn":
                yield child
            yield from _walk_csn_values(child)
    elif isinstance(value, list):
        for child in value:
            yield from _walk_csn_values(child)


def extract_csn(label_data: Dict[str, Any]) -> Optional[str]:
    values: List[str] = []
    for raw_value in _walk_csn_values(label_data):
        if isinstance(raw_value, (dict, list)) or raw_value is None:
            continue
        normalized = str(raw_value).strip()
        if normalized and normalized not in values:
            values.append(normalized)
    if not values:
        return None
    if len(values) > 1:
        raise ValueError(f"同一 label 中发现多个不同 CSN: {values}")
    return values[0]


def extract_csn_from_filename(file_path: Path) -> Optional[str]:
    """Parse CSN from XXX_<idx>_label_<csn>.json; CSN may contain underscores."""
    match = re.search(r"_(\d+)_label_(.+)\.json$", file_path.name, re.IGNORECASE)
    if not match:
        return None
    csn = match.group(2).strip()
    return csn or None


def extract_matching_csn(label_data: Dict[str, Any], file_path: Path) -> Optional[str]:
    filename_csn = extract_csn_from_filename(file_path)
    content_csn = extract_csn(label_data)
    if filename_csn and content_csn and filename_csn != content_csn:
        raise ValueError(
            "文件名 CSN 与 JSON 内 CSN 不一致: "
            f"filename={filename_csn}, content={content_csn}"
        )
    return filename_csn or content_csn


def is_valid_root_cause(root_cause: Any) -> bool:
    if not isinstance(root_cause, dict):
        return False
    category = root_cause.get("category")
    return category is not None and bool(str(category).strip())


def _scenario_name(file_path: Path, root: Path) -> str:
    relative = file_path.resolve().relative_to(root.resolve())
    return relative.parts[0] if len(relative.parts) > 1 else ""


def _root_cause_key(root_cause: Dict[str, Any]) -> str:
    return json.dumps(root_cause, ensure_ascii=False, sort_keys=True)


def build_source_index(source_root: str) -> Tuple[Dict[str, List[Dict[str, Any]]], List[Dict[str, Any]]]:
    root = Path(source_root).resolve()
    index: Dict[str, List[Dict[str, Any]]] = {}
    issues: List[Dict[str, Any]] = []

    for file_path in discover_label_files(str(root)):
        try:
            data = load_json(str(file_path))
            csn = extract_matching_csn(data, file_path)
        except Exception as exc:
            issues.append({
                "type": "invalid_source_label",
                "file": str(file_path),
                "error": str(exc),
            })
            continue
        if not csn:
            issues.append({"type": "source_missing_csn", "file": str(file_path)})
            continue
        root_cause = data.get("root_cause")
        if not is_valid_root_cause(root_cause):
            issues.append({
                "type": "source_missing_root_cause_category",
                "file": str(file_path),
                "csn": csn,
            })
            continue
        index.setdefault(csn, []).append({
            "scenario": _scenario_name(file_path, root),
            "file": str(file_path),
            "root_cause": deepcopy(root_cause),
        })
    return index, issues


def resolve_source_entry(
    entries: List[Dict[str, Any]],
    target_scenario: str,
) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    same_scenario = [
        entry for entry in entries
        if entry["scenario"] == target_scenario
    ]
    candidates = same_scenario or entries
    unique_root_causes = {
        _root_cause_key(entry["root_cause"])
        for entry in candidates
    }
    if len(unique_root_causes) > 1:
        return None, "同一 CSN 对应多个不同 root_cause"
    return candidates[0], None


def _write_json_atomic(data: Dict[str, Any], path: Path) -> None:
    temporary = path.with_name(path.name + ".tmp")
    with open(temporary, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    os.replace(temporary, path)


def fill_root_causes(
    target_root: str,
    source_root: str,
    apply_changes: bool = False,
    overwrite: bool = False,
    backup_dir: Optional[str] = None,
) -> Dict[str, Any]:
    target_root_path = Path(target_root).resolve()
    source_index, source_issues = build_source_index(source_root)
    report: Dict[str, Any] = {
        "target_root": str(target_root_path),
        "source_root": str(Path(source_root).resolve()),
        "apply": apply_changes,
        "overwrite": overwrite,
        "summary": {
            "target_label_count": 0,
            "matched_count": 0,
            "updated_count": 0,
            "already_present_count": 0,
            "unmatched_count": 0,
            "conflict_count": 0,
            "invalid_target_count": 0,
        },
        "source_issues": source_issues,
        "matched": [],
        "unmatched": [],
        "conflicts": [],
        "invalid_targets": [],
    }

    target_files = discover_label_files(str(target_root_path))
    report["summary"]["target_label_count"] = len(target_files)
    backup_root = Path(backup_dir).resolve() if backup_dir else None

    for target_path in target_files:
        scenario = _scenario_name(target_path, target_root_path)
        try:
            target_data = load_json(str(target_path))
            csn = extract_matching_csn(target_data, target_path)
        except Exception as exc:
            report["summary"]["invalid_target_count"] += 1
            report["invalid_targets"].append({
                "file": str(target_path),
                "error": str(exc),
            })
            continue
        if not csn:
            report["summary"]["invalid_target_count"] += 1
            report["invalid_targets"].append({
                "file": str(target_path),
                "error": "缺少 CSN",
            })
            continue
        if is_valid_root_cause(target_data.get("root_cause")) and not overwrite:
            report["summary"]["already_present_count"] += 1
            continue

        entries = source_index.get(csn, [])
        if not entries:
            report["summary"]["unmatched_count"] += 1
            report["unmatched"].append({
                "file": str(target_path),
                "scenario": scenario,
                "csn": csn,
            })
            continue
        source_entry, conflict = resolve_source_entry(entries, scenario)
        if conflict:
            report["summary"]["conflict_count"] += 1
            report["conflicts"].append({
                "file": str(target_path),
                "scenario": scenario,
                "csn": csn,
                "error": conflict,
                "source_files": [entry["file"] for entry in entries],
            })
            continue

        report["summary"]["matched_count"] += 1
        report["matched"].append({
            "target_file": str(target_path),
            "source_file": source_entry["file"],
            "scenario": scenario,
            "csn": csn,
            "root_cause": source_entry["root_cause"],
        })
        if not apply_changes:
            continue
        if backup_root:
            relative = target_path.relative_to(target_root_path)
            backup_path = backup_root / relative
            backup_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(target_path, backup_path)
        target_data["root_cause"] = deepcopy(source_entry["root_cause"])
        _write_json_atomic(target_data, target_path)
        report["summary"]["updated_count"] += 1

    return report


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="根据 CSN 从来源 label 复制 root_cause 到 anomalydetect_label。",
    )
    parser.add_argument("source_root", help="含有正确 root_cause 的 label 根目录")
    parser.add_argument(
        "--target-root",
        default=getattr(config, "ANOMALYDETECT_LABEL_ROOT"),
        help="待补充 root_cause 的 anomalydetect_label 根目录",
    )
    parser.add_argument("--apply", action="store_true", help="实际写入；默认仅 dry-run")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="覆盖目标中已有且有效的 root_cause；默认跳过",
    )
    parser.add_argument("--backup-dir", default=None, help="写入前备份目标 label 的目录")
    parser.add_argument("--report-path", default=None, help="匹配报告 JSON 路径")
    parser.add_argument(
        "--allow-unmatched",
        action="store_true",
        help="存在未匹配、冲突或坏文件时仍返回退出码 0",
    )
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()
    report = fill_root_causes(
        target_root=args.target_root,
        source_root=args.source_root,
        apply_changes=args.apply,
        overwrite=args.overwrite,
        backup_dir=args.backup_dir,
    )
    report_path = args.report_path or os.path.join(
        args.target_root,
        "root_cause_fill_report.json",
    )
    save_json(report, report_path)
    print(json.dumps(report["summary"], ensure_ascii=False, indent=2))
    print(f"报告: {report_path}")
    if not args.apply:
        print("当前为 dry-run；确认报告后添加 --apply 才会写入目标 label。")

    summary = report["summary"]
    unresolved = (
        summary["unmatched_count"]
        + summary["conflict_count"]
        + summary["invalid_target_count"]
        + len(report["source_issues"])
    )
    return 0 if args.allow_unmatched or unresolved == 0 else 2


if __name__ == "__main__":
    sys.exit(main())
