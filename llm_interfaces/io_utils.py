"""接口包内部使用的文件读写工具。"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, Iterator, List, TypeVar


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
    model_path: str,
    output_dir: str | Path,
    results: Iterable[dict[str, Any]],
) -> None:
    result_list = list(results)
    succeeded = sum(item.get("status") == "success" for item in result_list)
    payload = {
        "meta": {
            "mode": mode,
            "model_path": model_path,
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
