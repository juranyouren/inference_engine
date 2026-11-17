"""接口包内部使用的文件读写工具。"""

from __future__ import annotations

import csv
import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, TypeVar


T = TypeVar("T")


def ensure_dir(path: str | Path) -> Path:
    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def save_text(text: str, path: str | Path) -> None:
    file_path = Path(path)
    ensure_dir(file_path.parent)
    file_path.write_text(text, encoding="utf-8")


def save_json(data: Any, path: str | Path) -> None:
    file_path = Path(path)
    ensure_dir(file_path.parent)
    with file_path.open("w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=2)


def case_output_dir(output_dir: str | Path, input_index: int) -> Path:
    return ensure_dir(Path(output_dir) / f"case_{input_index:06d}")


def safe_category_name(category: Any) -> str:
    value = str(category).strip() if category is not None else ""
    if not value:
        return "_unknown"
    value = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "_", value)
    return value if value not in (".", "..") else "_unknown"


def category_output_dir(
    mode_output_dir: str | Path,
    category: Any,
) -> Path:
    return ensure_dir(Path(mode_output_dir) / safe_category_name(category))


def iter_batches(items: List[T], batch_size: int) -> Iterator[List[T]]:
    if batch_size <= 0:
        raise ValueError("batch_size 必须大于 0")
    for start in range(0, len(items), batch_size):
        yield items[start:start + batch_size]


def save_case_result(result: dict[str, Any]) -> None:
    output_dir = result.get("output_dir")
    if output_dir:
        save_json(result, Path(output_dir) / "result.json")


def save_batch_results(
    *,
    mode: str,
    output_dir: str | Path,
    results: Iterable[dict[str, Any]],
) -> None:
    result_list = list(results)
    succeeded = sum(item.get("status") == "success" for item in result_list)
    payload = {
        "meta": {
            "mode": mode,
            "output_dir": str(Path(output_dir).resolve()),
        },
        "summary": {
            "input_count": len(result_list),
            "success_count": succeeded,
            "failed_count": len(result_list) - succeeded,
        },
        "results": result_list,
    }
    save_json(payload, Path(output_dir) / "results.json")


CSV_FIELDS = (
    "case_idx",
    "timestamp",
    "incident_type",
    "predicted_list",
    "groundtruth",
    "rank",
    "ttm",
    "predicted_list_v1",
    "rank_v1",
)


def csv_result_dict(result: Dict[str, Any]) -> Dict[str, Any]:
    """返回与 CSV 列一一对应、保留列表类型的结果字典。"""
    return {field: result.get(field) for field in CSV_FIELDS}


def save_category_csvs(
    mode_output_dir: str | Path,
    results: Iterable[Dict[str, Any]],
) -> None:
    grouped: Dict[str, List[Dict[str, Any]]] = {}
    for result in results:
        category = str(result.get("incident_type") or "_unknown")
        grouped.setdefault(category, []).append(result)

    for category, category_results in grouped.items():
        csv_path = category_output_dir(mode_output_dir, category) / "results.csv"
        with csv_path.open("w", encoding="utf-8-sig", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=CSV_FIELDS)
            writer.writeheader()
            for result in sorted(
                category_results,
                key=lambda item: int(item.get("case_idx", 0)),
            ):
                row = csv_result_dict(result)
                writer.writerow({
                    "case_idx": row["case_idx"],
                    "timestamp": row["timestamp"],
                    "incident_type": row["incident_type"],
                    "predicted_list": json.dumps(
                        row["predicted_list"] or [],
                        ensure_ascii=False,
                    ),
                    "groundtruth": row["groundtruth"],
                    "rank": row["rank"],
                    "ttm": row["ttm"],
                    "predicted_list_v1": json.dumps(
                        row["predicted_list_v1"] or [],
                        ensure_ascii=False,
                    ),
                    "rank_v1": row["rank_v1"],
                })
