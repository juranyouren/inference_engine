#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DTW-based abnormal-pattern classifier for ARIMA-detected KPI segments.

The classifier is trained from historical ``ts_info`` JSON files.  It accepts
the historical structures used by this project, including either
``abnormal_timing_segments`` or ``segments`` lists at arbitrary nesting levels.

Default training data:
    /home/sbp/huangzeshun/评测角度一/data/<scenario>/ts_info/*.json

Default train/test split source:
    /home/sbp/huangzeshun/RCA_Units/<scenario>/*_<idx>_case_<csn>.json
    idx < 60 is used as DTW training data.

The primary backend is aeon's KNeighborsTimeSeriesClassifier(distance="dtw").
If aeon is unavailable, a small NumPy 1-NN DTW implementation is used so that
the integrated pipeline can still produce a pattern for every segment.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import pickle
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Sequence

import numpy as np


CURRENT_DIR = Path(__file__).resolve().parent

DEFAULT_SCENARIOS = (
    "BGP邻接状态改变",
    "入方向带宽利用率过高",
    "网络设备掉线",
    "接口状态DOWN",
)
DEFAULT_DTW_DATA_ROOT = Path("/home/sbp/huangzeshun/评测角度一/data")
DEFAULT_RCA_UNITS_ROOT = Path("/home/sbp/huangzeshun/RCA_Units")
DEFAULT_DTW_MODEL_PATH = (
    CURRENT_DIR / "models" / "dtw_pattern_classifier.pkl"
)
DEFAULT_SEQUENCE_LENGTH = 60
DEFAULT_TRAIN_SPLIT_INDEX = 60


def normalize_01(values: Sequence[float] | np.ndarray) -> np.ndarray:
    """Normalize one sequence to [0, 1] while handling constants and NaNs."""
    array = np.asarray(values, dtype=np.float64).reshape(-1)
    array = array[np.isfinite(array)]
    if len(array) == 0:
        return np.zeros(1, dtype=np.float64)

    minimum = float(np.min(array))
    maximum = float(np.max(array))
    difference = maximum - minimum
    if not math.isfinite(difference) or difference < 1e-12:
        return np.zeros_like(array, dtype=np.float64)
    return (array - minimum) / difference


def resample_sequence(
    values: Sequence[float] | np.ndarray,
    target_length: int = DEFAULT_SEQUENCE_LENGTH,
) -> np.ndarray:
    """
    Normalize and linearly resample a segment to a fixed length.

    Resampling avoids the variable-length stacking failure in the original DTW
    script and avoids creating a long artificial plateau when a short segment
    is padded with its last value.
    """
    if target_length < 2:
        raise ValueError("target_length must be at least 2")

    array = normalize_01(values)
    if len(array) == 1:
        return np.full(target_length, float(array[0]), dtype=np.float64)
    if len(array) == target_length:
        return array.astype(np.float64, copy=True)

    old_axis = np.linspace(0.0, 1.0, num=len(array), dtype=np.float64)
    new_axis = np.linspace(0.0, 1.0, num=target_length, dtype=np.float64)
    return np.interp(new_axis, old_axis, array).astype(np.float64)


def canonicalize_pattern(value: Any) -> str | None:
    """Normalize common historical label spellings without inventing labels."""
    if value is None:
        return None

    text = str(value).strip()
    if not text:
        return None

    compact = re.sub(r"[\s_\-]+", "", text).lower()

    if "mul" in compact or "fluc" in compact:
        return "multiSpikes"
    if compact in {"increase", "shiftup"}:
        return "shiftUp"
    if compact in {"decrease", "shiftdown"}:
        return "shiftDown"

    fixed_names = {
        "spike": "spike",
        "dip": "dip",
        "levelup": "levelUp",
        "leveldown": "levelDown",
        "steadyup": "steadyUp",
        "steadydown": "steadyDown",
        "multispikes": "multiSpikes",
    }
    return fixed_names.get(compact, text)


def _iter_segment_records(value: Any) -> Iterator[tuple[list[float], str]]:
    """Recursively yield (values, pattern) from supported historical JSONs."""
    if isinstance(value, Mapping):
        for segment_key in ("abnormal_timing_segments", "segments"):
            segment_list = value.get(segment_key)
            if isinstance(segment_list, list):
                for segment in segment_list:
                    if not isinstance(segment, Mapping):
                        continue
                    raw_values = segment.get("value")
                    pattern = canonicalize_pattern(segment.get("pattern"))
                    if pattern is None or not isinstance(raw_values, (list, tuple)):
                        continue
                    clean_values: list[float] = []
                    for item in raw_values:
                        try:
                            number = float(item)
                        except (TypeError, ValueError):
                            continue
                        if math.isfinite(number):
                            clean_values.append(number)
                    if clean_values:
                        yield clean_values, pattern

        for child in value.values():
            yield from _iter_segment_records(child)

    elif isinstance(value, list):
        for child in value:
            yield from _iter_segment_records(child)


_CASE_FILE_RE = re.compile(r".*_(\d+)_case_(\d+)\.json$")


def training_csns_for_scenario(
    rca_units_root: Path,
    scenario: str,
    split_index: int = DEFAULT_TRAIN_SPLIT_INDEX,
) -> set[str]:
    """Return CSNs whose case index is before the requested training split."""
    scenario_dir = rca_units_root / scenario
    if not scenario_dir.is_dir():
        return set()

    result: set[str] = set()
    for path in scenario_dir.iterdir():
        if not path.is_file():
            continue
        match = _CASE_FILE_RE.match(path.name)
        if match and int(match.group(1)) < split_index:
            result.add(match.group(2))
    return result


def _numeric_file_csn(path: Path) -> str | None:
    match = re.fullmatch(r"(\d+)", path.stem)
    return match.group(1) if match else None


@dataclass
class TrainingSummary:
    sample_count: int
    label_counts: dict[str, int]
    scenario_file_counts: dict[str, int]
    skipped_files: int


def load_historical_training_data(
    *,
    data_root: Path = DEFAULT_DTW_DATA_ROOT,
    rca_units_root: Path = DEFAULT_RCA_UNITS_ROOT,
    scenarios: Sequence[str] = DEFAULT_SCENARIOS,
    sequence_length: int = DEFAULT_SEQUENCE_LENGTH,
    split_index: int = DEFAULT_TRAIN_SPLIT_INDEX,
) -> tuple[np.ndarray, np.ndarray, TrainingSummary]:
    """Load and preprocess historical abnormal segments for DTW training."""
    samples: list[np.ndarray] = []
    labels: list[str] = []
    scenario_file_counts: dict[str, int] = {}
    skipped_files = 0

    for scenario in scenarios:
        ts_info_dir = data_root / scenario / "ts_info"
        if not ts_info_dir.is_dir():
            scenario_file_counts[scenario] = 0
            continue

        allowed_csns = training_csns_for_scenario(
            rca_units_root=rca_units_root,
            scenario=scenario,
            split_index=split_index,
        )
        use_split_filter = bool(allowed_csns)
        used_files = 0

        for path in sorted(ts_info_dir.glob("*.json")):
            file_csn = _numeric_file_csn(path)
            if use_split_filter and (file_csn is None or file_csn not in allowed_csns):
                continue

            try:
                with path.open("r", encoding="utf-8-sig") as file:
                    document = json.load(file)
            except Exception:
                skipped_files += 1
                continue

            found_in_file = False
            for values, pattern in _iter_segment_records(document):
                samples.append(resample_sequence(values, sequence_length))
                labels.append(pattern)
                found_in_file = True

            if found_in_file:
                used_files += 1

        scenario_file_counts[scenario] = used_files

    if not samples:
        raise ValueError(
            "No DTW training segments were found. Check --dtw-data-root, "
            "--dtw-rca-units-root, scenarios, and historical JSON structure."
        )

    x = np.stack(samples).astype(np.float64)[:, None, :]
    y = np.asarray(labels, dtype=str)

    unique, counts = np.unique(y, return_counts=True)
    summary = TrainingSummary(
        sample_count=len(y),
        label_counts={
            str(label): int(count)
            for label, count in zip(unique.tolist(), counts.tolist())
        },
        scenario_file_counts=scenario_file_counts,
        skipped_files=skipped_files,
    )
    return x, y, summary


def _dtw_distance(left: np.ndarray, right: np.ndarray) -> float:
    """Memory-efficient classic DTW distance used by the fallback backend."""
    left = np.asarray(left, dtype=np.float64).reshape(-1)
    right = np.asarray(right, dtype=np.float64).reshape(-1)

    previous = np.full(len(right) + 1, np.inf, dtype=np.float64)
    previous[0] = 0.0

    for left_value in left:
        current = np.full(len(right) + 1, np.inf, dtype=np.float64)
        for j, right_value in enumerate(right, start=1):
            cost = (left_value - right_value) ** 2
            current[j] = cost + min(
                current[j - 1],
                previous[j],
                previous[j - 1],
            )
        previous = current

    return float(math.sqrt(previous[-1]))


class DtwPatternClassifier:
    """Serializable DTW k-NN wrapper with aeon and NumPy backends."""

    def __init__(
        self,
        sequence_length: int = DEFAULT_SEQUENCE_LENGTH,
        n_neighbors: int = 1,
        backend: str = "auto",
    ) -> None:
        if sequence_length < 2:
            raise ValueError("sequence_length must be at least 2")
        if n_neighbors < 1:
            raise ValueError("n_neighbors must be at least 1")
        if backend not in {"auto", "aeon", "numpy"}:
            raise ValueError("backend must be auto, aeon, or numpy")

        self.sequence_length = int(sequence_length)
        self.n_neighbors = int(n_neighbors)
        self.backend_requested = backend
        self.backend_used = ""
        self.model: Any = None
        self.training_x: np.ndarray | None = None
        self.training_y: np.ndarray | None = None
        self.training_summary: dict[str, Any] = {}

    def fit(
        self,
        x: np.ndarray,
        y: np.ndarray,
        summary: TrainingSummary | None = None,
    ) -> "DtwPatternClassifier":
        x = np.asarray(x, dtype=np.float64)
        y = np.asarray(y, dtype=str)

        if x.ndim != 3 or x.shape[1] != 1:
            raise ValueError("x must have shape (samples, 1, sequence_length)")
        if len(x) != len(y) or len(y) == 0:
            raise ValueError("x and y must contain the same non-zero sample count")

        use_aeon = self.backend_requested in {"auto", "aeon"}
        if use_aeon:
            try:
                from aeon.classification.distance_based import (
                    KNeighborsTimeSeriesClassifier,
                )

                model = KNeighborsTimeSeriesClassifier(
                    distance="dtw",
                    n_neighbors=min(self.n_neighbors, len(y)),
                )
                model.fit(x, y)
                self.model = model
                self.backend_used = "aeon"
            except Exception:
                if self.backend_requested == "aeon":
                    raise
                self.backend_used = "numpy"

        if self.backend_used != "aeon":
            self.training_x = x[:, 0, :].copy()
            self.training_y = y.copy()
            self.model = None
            self.backend_used = "numpy"

        if summary is not None:
            self.training_summary = {
                "sample_count": summary.sample_count,
                "label_counts": summary.label_counts,
                "scenario_file_counts": summary.scenario_file_counts,
                "skipped_files": summary.skipped_files,
            }
        else:
            unique, counts = np.unique(y, return_counts=True)
            self.training_summary = {
                "sample_count": int(len(y)),
                "label_counts": {
                    str(label): int(count)
                    for label, count in zip(unique.tolist(), counts.tolist())
                },
            }

        return self

    def _predict_numpy(self, sequence: np.ndarray) -> str:
        if self.training_x is None or self.training_y is None:
            raise RuntimeError("NumPy DTW classifier has not been fitted")

        distances = np.asarray(
            [_dtw_distance(sequence, item) for item in self.training_x],
            dtype=np.float64,
        )
        neighbor_count = min(self.n_neighbors, len(distances))
        indices = np.argsort(distances, kind="stable")[:neighbor_count]

        votes: dict[str, tuple[int, float]] = {}
        for index in indices:
            label = str(self.training_y[index])
            count, distance_sum = votes.get(label, (0, 0.0))
            votes[label] = (count + 1, distance_sum + float(distances[index]))

        # Prefer more votes, then the smaller summed distance, then label text.
        return min(
            votes,
            key=lambda label: (
                -votes[label][0],
                votes[label][1],
                label,
            ),
        )

    def predict_pattern(self, values: Sequence[float] | np.ndarray) -> str:
        sequence = resample_sequence(values, self.sequence_length)

        if self.backend_used == "aeon":
            prediction = self.model.predict(sequence[None, None, :])
            return str(prediction[0])

        if self.backend_used == "numpy":
            return self._predict_numpy(sequence)

        raise RuntimeError("DTW classifier has not been fitted")

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=path.name + ".",
            suffix=".tmp",
            dir=str(path.parent),
        )
        try:
            with os.fdopen(descriptor, "wb") as file:
                pickle.dump(self, file, protocol=pickle.HIGHEST_PROTOCOL)
                file.flush()
                os.fsync(file.fileno())
            os.replace(temporary_name, path)
        finally:
            if os.path.exists(temporary_name):
                os.unlink(temporary_name)

    @classmethod
    def load(cls, path: Path) -> "DtwPatternClassifier":
        with path.open("rb") as file:
            model = pickle.load(file)
        if not isinstance(model, cls):
            raise TypeError(f"Unexpected object in DTW model file: {type(model)!r}")
        return model


def load_or_train_dtw_classifier(
    *,
    model_path: Path = DEFAULT_DTW_MODEL_PATH,
    data_root: Path = DEFAULT_DTW_DATA_ROOT,
    rca_units_root: Path = DEFAULT_RCA_UNITS_ROOT,
    scenarios: Sequence[str] = DEFAULT_SCENARIOS,
    sequence_length: int = DEFAULT_SEQUENCE_LENGTH,
    split_index: int = DEFAULT_TRAIN_SPLIT_INDEX,
    n_neighbors: int = 1,
    backend: str = "auto",
    force_retrain: bool = False,
) -> DtwPatternClassifier:
    """Load a cached classifier, or train and atomically cache a new one."""
    if model_path.is_file() and not force_retrain:
        model = DtwPatternClassifier.load(model_path)
        return model

    x, y, summary = load_historical_training_data(
        data_root=data_root,
        rca_units_root=rca_units_root,
        scenarios=scenarios,
        sequence_length=sequence_length,
        split_index=split_index,
    )
    classifier = DtwPatternClassifier(
        sequence_length=sequence_length,
        n_neighbors=n_neighbors,
        backend=backend,
    ).fit(x, y, summary)
    classifier.save(model_path)
    return classifier


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train/cache a DTW abnormal-pattern classifier."
    )
    parser.add_argument("--model-path", type=Path, default=DEFAULT_DTW_MODEL_PATH)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DTW_DATA_ROOT)
    parser.add_argument("--rca-units-root", type=Path, default=DEFAULT_RCA_UNITS_ROOT)
    parser.add_argument("--scenario", action="append", dest="scenarios")
    parser.add_argument("--sequence-length", type=int, default=DEFAULT_SEQUENCE_LENGTH)
    parser.add_argument("--split-index", type=int, default=DEFAULT_TRAIN_SPLIT_INDEX)
    parser.add_argument("--neighbors", type=int, default=1)
    parser.add_argument("--backend", choices=("auto", "aeon", "numpy"), default="auto")
    parser.add_argument("--force-retrain", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    classifier = load_or_train_dtw_classifier(
        model_path=args.model_path,
        data_root=args.data_root,
        rca_units_root=args.rca_units_root,
        scenarios=tuple(args.scenarios or DEFAULT_SCENARIOS),
        sequence_length=args.sequence_length,
        split_index=args.split_index,
        n_neighbors=args.neighbors,
        backend=args.backend,
        force_retrain=args.force_retrain,
    )

    print(f"model_path={args.model_path}")
    print(f"backend={classifier.backend_used}")
    print(f"sequence_length={classifier.sequence_length}")
    print(f"n_neighbors={classifier.n_neighbors}")
    print(
        json.dumps(
            classifier.training_summary,
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
