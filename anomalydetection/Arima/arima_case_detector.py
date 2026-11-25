#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Run ARIMA KPI anomaly detection plus DTW pattern classification on one raw RCA case JSON.

The detector recursively walks raw_data.kpi (or a top-level kpi object), finds
all nested ``v`` time-series nodes, and normalizes them into eight logical KPI
names:

    traffic_in
    traffic_out
    drop_packet_rate
    error_packet_rate
    offline_loss_rate
    cpu_usage
    memory_usage
    device_temperature

The ``带宽`` branch is treated as interface-capacity metadata rather than a
ninth anomaly KPI. It is used, when possible, to convert interface traffic from
bit/s to percentage utilization.

Output format:

{
  "anomaly_kpi": {
    "traffic_in": {
      "upper_bound": 80.0,
      "abnormal_timing_segments": [
        {
          "start_time": 1761018470,
          "end_time": 1761020210,
          "value": [0.4458, 0.8136],
          "pattern": "shiftUp"
        }
      ]
    }
  }
}

ARIMA first locates sensitive candidate anomalies. For each candidate segment,\nthe detector keeps only the peak-centered core. Across all physical sources of\nthe same logical KPI, only the strongest segment is emitted. DTW assigns the\n``pattern`` of that final segment. A deterministic fallback guarantees a pattern\nwhen DTW is unavailable.\n\nThe module can be imported by the integrated LogBERT runner or executed alone.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import tempfile
import warnings
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Sequence

import numpy as np

CURRENT_DIR = Path(__file__).resolve().parent
ANOMALY_DETECTION_ROOT = CURRENT_DIR.parent
DTW_DIR = ANOMALY_DETECTION_ROOT / "DTW"

if str(DTW_DIR) not in sys.path:
    sys.path.insert(0, str(DTW_DIR))

from dtw_pattern_classifier import (
    DEFAULT_DTW_DATA_ROOT,
    DEFAULT_DTW_MODEL_PATH,
    DEFAULT_RCA_UNITS_ROOT,
    DEFAULT_SCENARIOS,
    DEFAULT_SEQUENCE_LENGTH,
    DEFAULT_TRAIN_SPLIT_INDEX,
    DtwPatternClassifier,
    load_or_train_dtw_classifier,
)

try:
    from statsmodels.tsa.arima.model import ARIMA
except Exception as exc:  # pragma: no cover - checked again at runtime
    ARIMA = None  # type: ignore[assignment]
    _STATSMODELS_IMPORT_ERROR = exc
else:
    _STATSMODELS_IMPORT_ERROR = None

warnings.filterwarnings("ignore")


DEFAULT_PARAMETER_DIR = CURRENT_DIR / "pdqtout"

# After the alarm first occurs, collect one minimum point per logical KPI
# from this inclusive time window.
POST_ALARM_MINIMUM_WINDOW_SECONDS = 15 * 60

# traffic_in keeps the complete merged interval. Every other logical KPI keeps
# at most this many consecutive samples around the post-alarm minimum point.
NON_TRAFFIC_IN_MAX_POINTS = 5
PACKET_LOSS_METRICS = {"offline_loss_rate", "drop_packet_rate"}


@dataclass(frozen=True)
class MetricConfig:
    logical_name: str
    upper_bound: float
    default_order: tuple[int, int, int] = (1, 1, 1)
    default_anomaly_ratio: float = 10.0
    aliases: tuple[str, ...] = ()


METRIC_CONFIGS: dict[str, MetricConfig] = {
    "traffic_in": MetricConfig(
        logical_name="traffic_in",
        upper_bound=80.0,
        default_order=(1, 1, 1),
        default_anomaly_ratio=10.0,
        aliases=("-in", "traffic_in", "in", "inbound"),
    ),
    "traffic_out": MetricConfig(
        logical_name="traffic_out",
        upper_bound=80.0,
        default_order=(1, 1, 1),
        default_anomaly_ratio=10.0,
        aliases=("-out", "traffic_out", "out", "outbound"),
    ),
    "drop_packet_rate": MetricConfig(
        logical_name="drop_packet_rate",
        upper_bound=5.0,
        default_order=(1, 1, 1),
        default_anomaly_ratio=10.0,
        aliases=("DropPktRate", "drop_packet_rate", "drop rate"),
    ),
    "error_packet_rate": MetricConfig(
        logical_name="error_packet_rate",
        upper_bound=5.0,
        default_order=(1, 1, 1),
        default_anomaly_ratio=10.0,
        aliases=("ErrorPktRate", "error_packet_rate", "error rate"),
    ),
    "offline_loss_rate": MetricConfig(
        logical_name="offline_loss_rate",
        upper_bound=5.0,
        default_order=(1, 1, 1),
        default_anomaly_ratio=10.0,
        aliases=("OfflineLossRate", "offline_loss_rate"),
    ),
    "cpu_usage": MetricConfig(
        logical_name="cpu_usage",
        upper_bound=80.0,
        default_order=(1, 1, 1),
        default_anomaly_ratio=10.0,
        aliases=("CPUUsage", "CpuUsage", "cpu_usage", "CPU占用率"),
    ),
    "memory_usage": MetricConfig(
        logical_name="memory_usage",
        upper_bound=80.0,
        default_order=(1, 1, 1),
        default_anomaly_ratio=10.0,
        aliases=("MemoryUsage", "MemUsage", "memory_usage", "内存占用率"),
    ),
    "device_temperature": MetricConfig(
        logical_name="device_temperature",
        upper_bound=80.0,
        default_order=(1, 1, 1),
        default_anomaly_ratio=10.0,
        aliases=("Temperature", "DeviceTemperature", "device_temperature", "设备温度"),
    ),
}


@dataclass
class SeriesCandidate:
    metric_name: str
    path: tuple[str, ...]
    timestamps: np.ndarray
    values: np.ndarray
    unit: str = ""
    source_name: str = ""
    category: str = ""
    declared_upper_bound: float | None = None
    raw_node: dict[str, Any] = field(default_factory=dict)

    @property
    def source_id(self) -> str:
        if self.source_name:
            return self.source_name
        return "/".join(self.path)


@dataclass
class DetectionParameters:
    order: tuple[int, int, int]
    anomaly_ratio: float
    source: str


@dataclass
class CandidateDetection:
    metric_name: str
    source_id: str
    upper_bound: float
    segments: list[dict[str, Any]]
    order: tuple[int, int, int]
    anomaly_ratio: float
    parameter_source: str
    point_count: int
    train_point_count: int
    test_point_count: int
    status: str = "success"
    message: str = ""


# ---------------------------------------------------------------------------
# Generic JSON/KPI traversal
# ---------------------------------------------------------------------------


def load_json_file(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Case file not found: {path}")

    with path.open("r", encoding="utf-8-sig") as file:
        document = json.load(file)

    if not isinstance(document, dict):
        raise TypeError("The case JSON root must be an object.")

    return document


def infer_case_id(document: Mapping[str, Any], case_path: Path | None = None) -> str:
    raw_data = document.get("raw_data")
    if isinstance(raw_data, Mapping):
        value = raw_data.get("csn")
        if value not in (None, ""):
            return str(value)

    value = document.get("case_id")
    if value not in (None, ""):
        return str(value)

    if case_path is not None:
        match = re.search(r"(?:case_)?(\d+)", case_path.stem)
        if match:
            return match.group(1)

    return "unknown"


def infer_scenario(case_path: Path | None) -> str:
    if case_path is None:
        return ""

    stem = case_path.stem
    match = re.match(r"(.+?)_\d+_case_\d+$", stem)
    if match:
        return match.group(1)

    parent_name = case_path.parent.name
    return parent_name if parent_name else stem


def _iter_keyed_objects(
    value: Any,
    path: tuple[str, ...] = (),
) -> Iterator[tuple[tuple[str, ...], Any]]:
    yield path, value

    if isinstance(value, Mapping):
        for key, child in value.items():
            yield from _iter_keyed_objects(child, path + (str(key),))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            yield from _iter_keyed_objects(child, path + (str(index),))


def find_kpi_root(document: Mapping[str, Any]) -> Mapping[str, Any]:
    raw_data = document.get("raw_data")
    if isinstance(raw_data, Mapping):
        kpi = raw_data.get("kpi")
        if isinstance(kpi, Mapping):
            return kpi

    direct = document.get("kpi")
    if isinstance(direct, Mapping):
        return direct

    candidates: list[Mapping[str, Any]] = []
    for path, value in _iter_keyed_objects(document):
        if not path:
            continue
        if path[-1].lower() == "kpi" and isinstance(value, Mapping):
            candidates.append(value)

    if not candidates:
        raise KeyError("Cannot find a KPI object in raw_data.kpi or elsewhere in the case JSON.")

    # Prefer the largest candidate when duplicated wrappers exist.
    return max(candidates, key=lambda item: len(item))


def _is_numeric(value: Any) -> bool:
    if isinstance(value, bool) or value is None:
        return False
    try:
        number = float(value)
    except (TypeError, ValueError):
        return False
    return math.isfinite(number)


def _to_float(value: Any) -> float | None:
    if not _is_numeric(value):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def _timestamp_to_epoch_seconds(value: Any) -> int | None:
    if value is None:
        return None

    if _is_numeric(value):
        number = float(value)
        absolute = abs(number)
        if absolute >= 1e18:
            number /= 1e9
        elif absolute >= 1e15:
            number /= 1e6
        elif absolute >= 1e12:
            number /= 1e3
        return int(round(number))

    text = str(value).strip()
    if not text:
        return None

    try:
        number = float(text)
    except ValueError:
        number = None

    if number is not None:
        return _timestamp_to_epoch_seconds(number)

    normalized = text.replace("Z", "+00:00")
    try:
        dt = datetime.fromisoformat(normalized)
    except ValueError:
        for fmt in (
            "%Y-%m-%d %H:%M:%S",
            "%Y/%m/%d %H:%M:%S",
            "%Y-%m-%d-%H.%M.%S.%f",
        ):
            try:
                dt = datetime.strptime(text, fmt)
                break
            except ValueError:
                continue
        else:
            return None

    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return int(dt.timestamp())


def _extract_point_from_mapping(item: Mapping[str, Any]) -> tuple[int, float] | None:
    timestamp_keys = (
        "timestamp",
        "time",
        "ts",
        "collect_time",
        "collectTime",
        "occur_time",
        "occurTime",
        "x",
    )
    value_keys = (
        "value",
        "val",
        "avg",
        "mean",
        "y",
        "metric_value",
        "metricValue",
    )

    timestamp: int | None = None
    for key in timestamp_keys:
        if key in item:
            timestamp = _timestamp_to_epoch_seconds(item.get(key))
            if timestamp is not None:
                break

    number: float | None = None
    for key in value_keys:
        if key in item:
            number = _to_float(item.get(key))
            if number is not None:
                break

    # Some exports use {"v": 1.2, "t": 123} inside a list.
    if number is None and "v" in item and not isinstance(item.get("v"), (dict, list)):
        number = _to_float(item.get("v"))

    if timestamp is None or number is None:
        return None

    return timestamp, number


def extract_points(v_object: Any) -> tuple[np.ndarray, np.ndarray]:
    """Extract and sort numeric time-series points from several common shapes."""
    points: list[tuple[int, float]] = []

    if isinstance(v_object, Mapping):
        for key, value in v_object.items():
            timestamp = _timestamp_to_epoch_seconds(key)
            number = _to_float(value)
            if timestamp is not None and number is not None:
                points.append((timestamp, number))
                continue

            if isinstance(value, Mapping):
                parsed = _extract_point_from_mapping(value)
                if parsed is None and timestamp is not None:
                    for candidate_key in ("value", "val", "avg", "mean", "v"):
                        if candidate_key in value:
                            candidate_number = _to_float(value.get(candidate_key))
                            if candidate_number is not None:
                                parsed = (timestamp, candidate_number)
                                break
                if parsed is not None:
                    points.append(parsed)

    elif isinstance(v_object, list):
        for item in v_object:
            if isinstance(item, Mapping):
                parsed = _extract_point_from_mapping(item)
                if parsed is not None:
                    points.append(parsed)
            elif isinstance(item, (list, tuple)) and len(item) >= 2:
                timestamp = _timestamp_to_epoch_seconds(item[0])
                number = _to_float(item[1])
                if timestamp is not None and number is not None:
                    points.append((timestamp, number))

    if not points:
        return np.array([], dtype=np.int64), np.array([], dtype=np.float64)

    # If a timestamp appears more than once, average its finite observations.
    grouped: dict[int, list[float]] = {}
    for timestamp, number in points:
        grouped.setdefault(timestamp, []).append(number)

    ordered = sorted((timestamp, float(np.mean(values))) for timestamp, values in grouped.items())
    timestamps = np.asarray([item[0] for item in ordered], dtype=np.int64)
    values = np.asarray([item[1] for item in ordered], dtype=np.float64)
    return timestamps, values


def iter_v_nodes(kpi_root: Mapping[str, Any]) -> Iterator[tuple[tuple[str, ...], dict[str, Any]]]:
    """Yield every dictionary that owns a usable ``v`` child."""

    def walk(value: Any, path: tuple[str, ...]) -> Iterator[tuple[tuple[str, ...], dict[str, Any]]]:
        if isinstance(value, Mapping):
            value_dict = dict(value)
            if "v" in value_dict:
                timestamps, values = extract_points(value_dict.get("v"))
                if len(timestamps) > 0 and len(values) > 0:
                    yield path, value_dict

            for key, child in value.items():
                # Do not recurse into the time-value map after yielding its owner.
                if key == "v":
                    continue
                yield from walk(child, path + (str(key),))

        elif isinstance(value, list):
            for index, child in enumerate(value):
                yield from walk(child, path + (str(index),))

    yield from walk(kpi_root, ())


# ---------------------------------------------------------------------------
# KPI classification and normalization
# ---------------------------------------------------------------------------


def _normalized_text(path: Sequence[str], node: Mapping[str, Any]) -> str:
    metadata_values: list[str] = []
    for key in (
        "name",
        "metric",
        "metric_name",
        "metricName",
        "description",
        "desc",
        "unit",
        "direction",
        "type",
        "m",
        "mk",
    ):
        value = node.get(key)
        if value not in (None, "") and not isinstance(value, (dict, list)):
            metadata_values.append(str(value))

    return " / ".join([*map(str, path), *metadata_values]).lower()


def _contains_any(text: str, keywords: Iterable[str]) -> bool:
    return any(keyword.lower() in text for keyword in keywords)


def _has_direction(path: Sequence[str], text: str, direction: str) -> bool:
    path_tokens = {str(token).strip().lower() for token in path}

    if direction == "in":
        exact = {"in", "input", "inbound", "rx", "receive", "ingress", "入", "入方向"}
        phrases = ("-in", "_in", "traffic_in", "inbound", "ingress", "入方向", "入流量")
    else:
        exact = {"out", "output", "outbound", "tx", "send", "egress", "出", "出方向"}
        phrases = ("-out", "_out", "traffic_out", "outbound", "egress", "出方向", "出流量")

    if path_tokens.intersection(exact):
        return True
    return any(phrase in text for phrase in phrases)


def classify_metric(path: Sequence[str], node: Mapping[str, Any]) -> str | None:
    text = _normalized_text(path, node)

    # Capacity is metadata; do not turn it into a ninth anomaly KPI.
    if _contains_any(text, ("带宽", "bandwidth", "capacity")) and not _contains_any(
        text,
        ("带宽利用率", "bandwidth utilization", "utilization"),
    ):
        return None

    if _contains_any(text, ("丢弃包速率", "丢包速率", "droppktrate", "drop_packet_rate", "drop packet", "packet drop")):
        return "drop_packet_rate"

    if _contains_any(text, ("错误包速率", "错包速率", "errorpktrate", "error packet", "packet error")):
        return "error_packet_rate"

    if _contains_any(text, ("掉线监控丢包率", "掉线丢包率", "offlinelossrate", "offline_loss_rate", "offline loss")):
        return "offline_loss_rate"

    if _contains_any(text, ("cpu占用率", "cpu利用率", "cpuusage", "cpu usage", "cpu utilization")):
        return "cpu_usage"

    if _contains_any(text, ("内存占用率", "内存利用率", "memoryusage", "memusage", "memory usage", "memory utilization")):
        return "memory_usage"

    if _contains_any(text, ("设备温度", "单板温度", "temperature", "temp")):
        return "device_temperature"

    traffic_keywords = (
        "接口速率",
        "端口速率",
        "interface rate",
        "traffic rate",
        "带宽利用率",
        "traffic_in",
        "traffic_out",
    )
    if _contains_any(text, traffic_keywords):
        if _has_direction(path, text, "in"):
            return "traffic_in"
        if _has_direction(path, text, "out"):
            return "traffic_out"

    # Last-resort aliases for old CSV-style field names.
    if "-in" in text:
        return "traffic_in"
    if "-out" in text:
        return "traffic_out"

    return None


def infer_category(path: Sequence[str]) -> str:
    text = "/".join(map(str, path)).lower()
    if "接口" in text or "interface" in text:
        return "interface"
    if "设备" in text or "device" in text:
        return "device"
    if "单板" in text or "board" in text:
        return "board"
    return "unknown"


def infer_source_name(path: Sequence[str], node: Mapping[str, Any]) -> str:
    for key in ("name", "interface", "if_name", "ifName", "board_name", "boardName", "device_name"):
        value = node.get(key)
        if value not in (None, "") and not isinstance(value, (dict, list)):
            return str(value)

    # Prefer path components that resemble interface/board identifiers.
    identifier_pattern = re.compile(
        r"(?:^|[^a-z0-9])(?:\d+(?:\.\d+)?(?:t|g|m)?e|eth|ge|xge|port|slot|board|\d+/\d+)",
        re.IGNORECASE,
    )
    for token in reversed(path):
        if identifier_pattern.search(str(token)):
            return str(token)

    return str(path[-1]) if path else ""


def _extract_declared_upper_bound(node: Mapping[str, Any]) -> float | None:
    for key in (
        "upper_bound",
        "upperBound",
        "threshold",
        "alarm_threshold",
        "alarmThreshold",
        "max",
    ):
        number = _to_float(node.get(key))
        if number is not None:
            return number
    return None


def extract_series_candidates(kpi_root: Mapping[str, Any]) -> list[SeriesCandidate]:
    candidates: list[SeriesCandidate] = []

    for path, node in iter_v_nodes(kpi_root):
        metric_name = classify_metric(path, node)
        if metric_name is None:
            continue

        timestamps, values = extract_points(node.get("v"))
        if len(timestamps) == 0:
            continue

        candidates.append(
            SeriesCandidate(
                metric_name=metric_name,
                path=tuple(path),
                timestamps=timestamps,
                values=values,
                unit=str(node.get("unit") or ""),
                source_name=infer_source_name(path, node),
                category=infer_category(path),
                declared_upper_bound=_extract_declared_upper_bound(node),
                raw_node=dict(node),
            )
        )

    return candidates


def _is_bandwidth_path(path: Sequence[str], node: Mapping[str, Any]) -> bool:
    text = _normalized_text(path, node)
    return _contains_any(text, ("带宽", "bandwidth", "capacity")) and not _contains_any(
        text,
        ("利用率", "utilization"),
    )


def _unit_to_bps_factor(unit: str) -> float | None:
    text = unit.strip().lower().replace(" ", "")
    if not text:
        return None

    if text in {"bit/s", "bps", "b/s"}:
        return 1.0
    if text in {"kbit/s", "kbps", "kb/s"}:
        return 1e3
    if text in {"mbit/s", "mbps", "mb/s"}:
        return 1e6
    if text in {"gbit/s", "gbps", "gb/s"}:
        return 1e9
    if text in {"tbit/s", "tbps", "tb/s"}:
        return 1e12
    if text in {"byte/s", "bytes/s", "b/s(byte)"}:
        return 8.0
    if text in {"kbyte/s", "kbytes/s", "kb/s(byte)"}:
        return 8e3
    if text in {"mbyte/s", "mbytes/s", "mb/s(byte)"}:
        return 8e6
    if text in {"gbyte/s", "gbytes/s", "gb/s(byte)"}:
        return 8e9
    return None


def _parse_capacity_from_text(text: str) -> float | None:
    """Parse identifiers such as 100GE1/3/25 or 40Gbps into bit/s."""
    patterns = (
        r"(?<!\d)(\d+(?:\.\d+)?)\s*te(?:\d|/|$)",
        r"(?<!\d)(\d+(?:\.\d+)?)\s*ge(?:\d|/|$)",
        r"(?<!\d)(\d+(?:\.\d+)?)\s*me(?:\d|/|$)",
        r"(?<!\d)(\d+(?:\.\d+)?)\s*(tbps|gbps|mbps|kbps|bps)\b",
    )

    lower = text.lower()
    for index, pattern in enumerate(patterns):
        match = re.search(pattern, lower, flags=re.IGNORECASE)
        if not match:
            continue
        number = float(match.group(1))
        if index == 0:
            return number * 1e12
        if index == 1:
            return number * 1e9
        if index == 2:
            return number * 1e6

        suffix = match.group(2).lower()
        factors = {
            "tbps": 1e12,
            "gbps": 1e9,
            "mbps": 1e6,
            "kbps": 1e3,
            "bps": 1.0,
        }
        return number * factors[suffix]

    return None


def collect_bandwidth_capacities(kpi_root: Mapping[str, Any]) -> dict[str, float]:
    capacities: dict[str, float] = {}

    for path, node in iter_v_nodes(kpi_root):
        if not _is_bandwidth_path(path, node):
            continue
        timestamps, values = extract_points(node.get("v"))
        if len(values) == 0:
            continue

        source_name = infer_source_name(path, node)
        unit = str(node.get("unit") or "")
        factor = _unit_to_bps_factor(unit)
        median_value = float(np.nanmedian(values))
        capacity = median_value * factor if factor is not None else median_value
        if capacity > 0:
            capacities[source_name] = capacity

    # Also inspect scalar values under a bandwidth branch.
    for path, value in _iter_keyed_objects(kpi_root):
        if not path or not _contains_any("/".join(path).lower(), ("带宽", "bandwidth", "capacity")):
            continue
        number = _to_float(value)
        if number is None or number <= 0:
            continue

        source_name = ""
        for token in reversed(path[:-1]):
            parsed = _parse_capacity_from_text(str(token))
            if parsed is not None:
                source_name = str(token)
                break
        if not source_name:
            continue

        # If the scalar is small, infer the unit from the interface identifier.
        inferred = _parse_capacity_from_text(source_name)
        if inferred is not None and number < 1e6:
            capacities[source_name] = inferred
        else:
            capacities[source_name] = number

    return capacities


def _find_capacity_for_candidate(
    candidate: SeriesCandidate,
    capacities: Mapping[str, float],
) -> float | None:
    if candidate.source_name in capacities:
        return capacities[candidate.source_name]

    # Fuzzy source-name match for wrappers that add prefixes/suffixes.
    for source_name, capacity in capacities.items():
        if source_name and candidate.source_name and (
            source_name in candidate.source_name or candidate.source_name in source_name
        ):
            return capacity

    for token in (candidate.source_name, *reversed(candidate.path)):
        capacity = _parse_capacity_from_text(str(token))
        if capacity is not None:
            return capacity

    return None


def normalize_candidate_values(
    candidate: SeriesCandidate,
    capacities: Mapping[str, float],
) -> SeriesCandidate:
    values = candidate.values.astype(np.float64, copy=True)
    unit_text = candidate.unit.strip().lower()

    if candidate.metric_name in {"traffic_in", "traffic_out"}:
        already_percentage = "%" in unit_text or "percent" in unit_text or "利用率" in "/".join(candidate.path)
        if not already_percentage:
            factor = _unit_to_bps_factor(candidate.unit)
            capacity = _find_capacity_for_candidate(candidate, capacities)
            if factor is not None and capacity is not None and capacity > 0:
                values = values * factor / capacity * 100.0

    # ARIMA/DTW use historical percentage points internally. Filter impossible
    # measurements instead of turning an invalid counter into a 100% loss event.
    timestamps = candidate.timestamps.copy()
    upper = candidate.declared_upper_bound
    if candidate.metric_name in PACKET_LOSS_METRICS:
        fractional = unit_text in {"ratio", "fraction"}
        scale = 1.0 if fractional else 100.0
        valid = (values >= 0) & (values <= scale)
        if unit_text in {"pps", "pkt/s", "pkts/s", "packets/s"}:
            valid[:] = False  # No packet total is available for conversion.
        timestamps = timestamps[valid]
        values = values[valid]
        if upper is not None and not 0 <= upper <= scale:
            upper = None
        if fractional:
            values *= 100.0
            if upper is not None:
                upper *= 100.0

    # Convert fractional percentage-style rates to 0-100 when explicitly marked.
    if candidate.metric_name in {
        "error_packet_rate",
        "cpu_usage",
        "memory_usage",
    }:
        if unit_text in {"ratio", "fraction"} and len(values) and np.nanmax(np.abs(values)) <= 1.0:
            values = values * 100.0

    return SeriesCandidate(
        metric_name=candidate.metric_name,
        path=candidate.path,
        timestamps=timestamps,
        values=values,
        unit="%" if candidate.metric_name in {"traffic_in", "traffic_out"} else candidate.unit,
        source_name=candidate.source_name,
        category=candidate.category,
        declared_upper_bound=upper,
        raw_node=candidate.raw_node,
    )


def extract_firstoccurutc(case_document: Mapping[str, Any]) -> int | None:
    """Read raw_data.alarm_content.firstoccurutc and normalize it to seconds."""
    raw_data = case_document.get("raw_data")
    if isinstance(raw_data, Mapping):
        alarm_content = raw_data.get("alarm_content")
        if isinstance(alarm_content, Mapping):
            timestamp = _timestamp_to_epoch_seconds(
                alarm_content.get("firstoccurutc")
            )
            if timestamp is not None:
                return timestamp

    # Defensive fallback for exports that put alarm_content at the root.
    alarm_content = case_document.get("alarm_content")
    if isinstance(alarm_content, Mapping):
        return _timestamp_to_epoch_seconds(
            alarm_content.get("firstoccurutc")
        )

    return None


def _round_public_value(raw_value: float) -> float:
    """Use the same public-value rounding policy as ARIMA anomaly segments."""
    number = float(raw_value)
    rounded = round(number, 4)
    if rounded == 0.0 and number != 0.0:
        rounded = round(number, 8)
    return rounded


def collect_post_alarm_minimum_segments(
    candidates: Sequence[SeriesCandidate],
    *,
    firstoccurutc: int | None,
    window_seconds: int = POST_ALARM_MINIMUM_WINDOW_SECONDS,
    pattern_classifier: DtwPatternClassifier | None = None,
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    """
    Find the global minimum point for each logical KPI in the inclusive window:

        [firstoccurutc, firstoccurutc + window_seconds]

    All physical sources belonging to the same logical KPI are compared after
    the existing normalization step. The returned item uses the normal segment
    schema and deliberately does not add a min_value/is_minimum field.
    """
    diagnostics: dict[str, Any] = {
        "firstoccurutc": firstoccurutc,
        "window_seconds": int(window_seconds),
        "window_end": (
            int(firstoccurutc + window_seconds)
            if firstoccurutc is not None
            else None
        ),
        "metrics": {},
    }

    if firstoccurutc is None or window_seconds < 0:
        return {}, diagnostics

    window_end = int(firstoccurutc + window_seconds)
    best_by_metric: dict[str, dict[str, Any]] = {}

    for candidate in candidates:
        timestamps, values = _clean_series(
            candidate.timestamps,
            candidate.values,
        )
        if len(values) == 0:
            continue

        window_mask = (
            (timestamps >= int(firstoccurutc))
            & (timestamps <= window_end)
        )
        window_indices = np.flatnonzero(window_mask)
        if len(window_indices) == 0:
            continue

        local_values = values[window_indices]
        finite_mask = np.isfinite(local_values)
        if not finite_mask.any():
            continue

        valid_indices = window_indices[finite_mask]
        valid_values = values[valid_indices]

        # np.argmin returns the earliest occurrence when the minimum repeats.
        relative_index = int(np.argmin(valid_values))
        point_index = int(valid_indices[relative_index])
        point_value = float(values[point_index])
        point_timestamp = int(timestamps[point_index])

        previous = best_by_metric.get(candidate.metric_name)
        if previous is not None:
            previous_value = float(previous["raw_value"])
            previous_timestamp = int(previous["timestamp"])
            if point_value > previous_value:
                continue
            if (
                point_value == previous_value
                and point_timestamp >= previous_timestamp
            ):
                continue

        baseline_values = values[timestamps < int(firstoccurutc)]
        if len(baseline_values) == 0:
            baseline_values = values

        pattern = predict_segment_pattern(
            segment_values=np.asarray([point_value], dtype=np.float64),
            baseline_values=np.asarray(baseline_values, dtype=np.float64),
            pattern_classifier=pattern_classifier,
        )

        upper_bound = (
            float(candidate.declared_upper_bound)
            if candidate.declared_upper_bound is not None
            else float(METRIC_CONFIGS[candidate.metric_name].upper_bound)
        )

        best_by_metric[candidate.metric_name] = {
            "timestamp": point_timestamp,
            "raw_value": point_value,
            "upper_bound": upper_bound,
            "source_id": candidate.source_id,
            "segment": {
                "start_time": point_timestamp,
                "end_time": point_timestamp,
                "value": [_round_public_value(point_value)],
                "pattern": pattern,
            },
        }

    segments: dict[str, dict[str, Any]] = {}
    for metric_name, item in best_by_metric.items():
        segments[metric_name] = {
            "upper_bound": round(float(item["upper_bound"]), 4),
            "source_id": str(item["source_id"]),
            "timestamp": int(item["timestamp"]),
            "raw_value": float(item["raw_value"]),
            "segment": dict(item["segment"]),
        }
        diagnostics["metrics"][metric_name] = {
            "source_id": str(item["source_id"]),
            "timestamp": int(item["timestamp"]),
            "value": _round_public_value(float(item["raw_value"])),
        }

    return segments, diagnostics


def _candidate_contains_point(
    candidate: SeriesCandidate,
    timestamp: int,
    value: float,
    *,
    atol: float = 1e-6,
) -> bool:
    timestamps, values = _clean_series(
        candidate.timestamps,
        candidate.values,
    )
    indices = np.flatnonzero(timestamps == int(timestamp))
    return any(
        math.isclose(
            float(values[index]),
            float(value),
            rel_tol=0.0,
            abs_tol=atol,
        )
        for index in indices.tolist()
    )


def _candidate_matches_public_segment(
    candidate: SeriesCandidate,
    segment: Mapping[str, Any],
    *,
    atol: float = 1e-4,
) -> bool:
    """Check whether a public segment was produced from this candidate."""
    raw_values = segment.get("value")
    if not isinstance(raw_values, list) or not raw_values:
        return False

    try:
        start_time = int(segment["start_time"])
        end_time = int(segment["end_time"])
        expected = np.asarray(
            [float(item) for item in raw_values],
            dtype=np.float64,
        )
    except (KeyError, TypeError, ValueError):
        return False

    timestamps, values = _clean_series(
        candidate.timestamps,
        candidate.values,
    )
    mask = (
        (timestamps >= start_time)
        & (timestamps <= end_time)
    )
    actual = values[mask]

    if len(actual) != len(expected):
        return False

    rounded_actual = np.asarray(
        [_round_public_value(value) for value in actual],
        dtype=np.float64,
    )
    return bool(
        np.allclose(
            rounded_actual,
            expected,
            rtol=0.0,
            atol=atol,
            equal_nan=False,
        )
    )


def _choose_merge_candidate(
    metric_name: str,
    candidates: Sequence[SeriesCandidate],
    existing_segments: Sequence[Mapping[str, Any]],
    minimum_payload: Mapping[str, Any],
    merged_start: int,
    merged_end: int,
) -> SeriesCandidate | None:
    """
    Select one physical source from which a continuous merged interval can be
    reconstructed. Prefer a source matching the ARIMA segment and the selected
    minimum point.
    """
    minimum_source = str(minimum_payload.get("source_id") or "")
    minimum_timestamp = int(minimum_payload["timestamp"])
    minimum_value = float(minimum_payload["raw_value"])

    best_candidate: SeriesCandidate | None = None
    best_score = -math.inf

    for candidate in candidates:
        if candidate.metric_name != metric_name:
            continue

        timestamps, values = _clean_series(
            candidate.timestamps,
            candidate.values,
        )
        if len(values) == 0:
            continue

        interval_mask = (
            (timestamps >= merged_start)
            & (timestamps <= merged_end)
        )
        interval_count = int(np.count_nonzero(interval_mask))
        if interval_count == 0:
            continue

        score = float(interval_count)

        if candidate.source_id == minimum_source:
            score += 100.0

        if _candidate_contains_point(
            candidate,
            minimum_timestamp,
            minimum_value,
        ):
            score += 200.0

        matching_segment_count = sum(
            1
            for segment in existing_segments
            if _candidate_matches_public_segment(
                candidate,
                segment,
            )
        )
        score += matching_segment_count * 1000.0

        if (
            np.any(timestamps == merged_start)
            and np.any(timestamps == merged_end)
        ):
            score += 50.0

        if score > best_score:
            best_score = score
            best_candidate = candidate

    return best_candidate


def _limit_non_traffic_interval_around_minimum(
    metric_name: str,
    timestamps: np.ndarray,
    values: np.ndarray,
    *,
    minimum_timestamp: int,
    max_points: int = NON_TRAFFIC_IN_MAX_POINTS,
) -> tuple[np.ndarray, np.ndarray, bool]:
    """
    Keep traffic_in unchanged. For every other KPI, retain at most max_points
    consecutive samples centred around the post-alarm minimum point.

    The returned start_time/end_time therefore continue to match the first and
    last values actually emitted in the public JSON.
    """
    if (
        metric_name == "traffic_in"
        or max_points <= 0
        or len(values) <= max_points
    ):
        return timestamps, values, False

    # The minimum point should normally be an exact timestamp in this interval.
    exact_indices = np.flatnonzero(
        timestamps == int(minimum_timestamp)
    )
    if len(exact_indices):
        center_index = int(exact_indices[0])
    else:
        center_index = int(
            np.argmin(
                np.abs(
                    timestamps.astype(np.int64)
                    - int(minimum_timestamp)
                )
            )
        )

    left = center_index - max_points // 2
    left = max(0, left)
    right = min(len(values), left + max_points)
    left = max(0, right - max_points)

    return (
        timestamps[left:right],
        values[left:right],
        True,
    )


def merge_post_alarm_minimum_segments(
    anomaly_kpi: dict[str, Any],
    minimum_segments: Mapping[str, Mapping[str, Any]],
    candidates: Sequence[SeriesCandidate],
    *,
    pattern_classifier: DtwPatternClassifier | None = None,
) -> dict[str, Any]:
    """
    Merge every KPI's existing ARIMA segment(s) and its post-alarm minimum into
    one continuous interval.

    The output values are re-read from one normalized raw KPI source over the
    entire [merged_start, merged_end] range. This includes normal points between
    the ARIMA anomaly and the later minimum point. No special minimum-value key
    is added to the public JSON.
    """
    merge_diagnostics: dict[str, Any] = {}

    for metric_name, payload in minimum_segments.items():
        minimum_segment = payload.get("segment")
        if not isinstance(minimum_segment, Mapping):
            continue

        metric_payload = anomaly_kpi.setdefault(
            metric_name,
            {
                "upper_bound": round(
                    float(
                        payload.get(
                            "upper_bound",
                            METRIC_CONFIGS[metric_name].upper_bound,
                        )
                    ),
                    4,
                ),
                "abnormal_timing_segments": [],
            },
        )
        if not isinstance(metric_payload, dict):
            continue

        raw_segment_list = metric_payload.get(
            "abnormal_timing_segments",
            [],
        )
        existing_segments = [
            segment
            for segment in raw_segment_list
            if isinstance(segment, Mapping)
        ]

        minimum_timestamp = int(minimum_segment["start_time"])

        if existing_segments:
            merged_start = min(
                minimum_timestamp,
                *[
                    int(segment["start_time"])
                    for segment in existing_segments
                ],
            )
            merged_end = max(
                minimum_timestamp,
                *[
                    int(segment["end_time"])
                    for segment in existing_segments
                ],
            )
        else:
            merged_start = minimum_timestamp
            merged_end = minimum_timestamp

        candidate = _choose_merge_candidate(
            metric_name=metric_name,
            candidates=candidates,
            existing_segments=existing_segments,
            minimum_payload=payload,
            merged_start=merged_start,
            merged_end=merged_end,
        )

        if candidate is None:
            # Defensive fallback: preserve a valid single-point minimum rather
            # than silently dropping the KPI.
            metric_payload["abnormal_timing_segments"] = [
                {
                    "start_time": minimum_timestamp,
                    "end_time": minimum_timestamp,
                    "value": list(
                        minimum_segment.get("value") or []
                    ),
                    "pattern": str(
                        minimum_segment.get("pattern") or "dip"
                    ),
                }
            ]
            merge_diagnostics[metric_name] = {
                "status": "fallback_single_minimum",
                "start_time": minimum_timestamp,
                "end_time": minimum_timestamp,
            }
            continue

        timestamps, values = _clean_series(
            candidate.timestamps,
            candidate.values,
        )
        interval_mask = (
            (timestamps >= merged_start)
            & (timestamps <= merged_end)
        )
        merged_timestamps = timestamps[interval_mask]
        merged_values = values[interval_mask]

        if len(merged_values) == 0:
            continue

        original_merged_point_count = int(len(merged_values))
        (
            merged_timestamps,
            merged_values,
            point_limit_applied,
        ) = _limit_non_traffic_interval_around_minimum(
            metric_name,
            merged_timestamps,
            merged_values,
            minimum_timestamp=minimum_timestamp,
            max_points=NON_TRAFFIC_IN_MAX_POINTS,
        )

        # start_time/end_time always match the actual retained value sequence.
        actual_start = int(merged_timestamps[0])
        actual_end = int(merged_timestamps[-1])

        baseline_values = values[timestamps < actual_start]
        if len(baseline_values) == 0:
            baseline_values = values

        merged_pattern = predict_segment_pattern(
            segment_values=merged_values,
            baseline_values=baseline_values,
            pattern_classifier=pattern_classifier,
        )

        metric_payload["abnormal_timing_segments"] = [
            {
                "start_time": actual_start,
                "end_time": actual_end,
                "value": [
                    _round_public_value(value)
                    for value in merged_values
                ],
                "pattern": merged_pattern,
            }
        ]

        merge_diagnostics[metric_name] = {
            "status": "merged",
            "source_id": candidate.source_id,
            "start_time": actual_start,
            "end_time": actual_end,
            "point_count": int(len(merged_values)),
            "original_merged_point_count": original_merged_point_count,
            "point_limit_applied": bool(point_limit_applied),
            "point_limit": (
                None
                if metric_name == "traffic_in"
                else NON_TRAFFIC_IN_MAX_POINTS
            ),
            "original_segment_count": int(
                len(existing_segments)
            ),
            "minimum_timestamp": minimum_timestamp,
        }

    return merge_diagnostics


# ---------------------------------------------------------------------------
# Parameter loading
# ---------------------------------------------------------------------------


_PARAMETER_FILE_RE = re.compile(r"testall_anoratio(\d+)_p(\d+)_d(\d+)_q(\d+)\.json$")


def _metric_aliases(metric_name: str) -> tuple[str, ...]:
    config = METRIC_CONFIGS[metric_name]
    return (metric_name, *config.aliases)


def _extract_metric_result(mapping: Mapping[str, Any], aliases: Sequence[str]) -> Mapping[str, Any] | None:
    for alias in aliases:
        value = mapping.get(alias)
        if isinstance(value, Mapping):
            return value

    lowered = {str(key).lower(): key for key in mapping.keys()}
    for alias in aliases:
        key = lowered.get(alias.lower())
        if key is not None:
            value = mapping.get(key)
            if isinstance(value, Mapping):
                return value

    return None


def resolve_detection_parameters(
    metric_name: str,
    scenario: str,
    parameter_dir: Path | None,
    forced_order: tuple[int, int, int] | None = None,
    forced_anomaly_ratio: float | None = None,
) -> DetectionParameters:
    config = METRIC_CONFIGS[metric_name]

    if forced_order is not None or forced_anomaly_ratio is not None:
        return DetectionParameters(
            order=forced_order or config.default_order,
            anomaly_ratio=(
                float(forced_anomaly_ratio)
                if forced_anomaly_ratio is not None
                else config.default_anomaly_ratio
            ),
            source="command_line",
        )

    if parameter_dir is None or not parameter_dir.is_dir() or not scenario:
        return DetectionParameters(
            order=config.default_order,
            anomaly_ratio=config.default_anomaly_ratio,
            source="default",
        )

    best_score = -math.inf
    best: DetectionParameters | None = None
    aliases = _metric_aliases(metric_name)

    for path in parameter_dir.iterdir():
        match = _PARAMETER_FILE_RE.match(path.name)
        if not match:
            continue

        anomaly_ratio, p_value, d_value, q_value = map(int, match.groups())
        try:
            with path.open("r", encoding="utf-8") as file:
                result_document = json.load(file)
        except Exception:
            continue

        scenario_mapping = result_document.get(scenario)
        if not isinstance(scenario_mapping, Mapping):
            continue

        fscores: list[float] = []
        for case_result in scenario_mapping.values():
            if not isinstance(case_result, Mapping):
                continue
            metric_result = _extract_metric_result(case_result, aliases)
            if metric_result is None:
                continue
            fscore = _to_float(metric_result.get("fscore"))
            if fscore is not None:
                fscores.append(fscore)

        if not fscores:
            continue

        mean_fscore = float(np.mean(fscores))
        if mean_fscore > best_score:
            best_score = mean_fscore
            best = DetectionParameters(
                order=(p_value, d_value, q_value),
                anomaly_ratio=float(anomaly_ratio),
                source=str(path),
            )

    if best is not None:
        return best

    return DetectionParameters(
        order=config.default_order,
        anomaly_ratio=config.default_anomaly_ratio,
        source="default",
    )


# ---------------------------------------------------------------------------
# ARIMA detection and segment extraction
# ---------------------------------------------------------------------------


def _clean_series(timestamps: np.ndarray, values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mask = np.isfinite(values)
    timestamps = timestamps[mask]
    values = values[mask]

    if len(values) == 0:
        return timestamps, values

    order = np.argsort(timestamps, kind="stable")
    timestamps = timestamps[order]
    values = values[order]

    # Replace repeated timestamps with their mean.
    grouped: dict[int, list[float]] = {}
    for timestamp, value in zip(timestamps.tolist(), values.tolist()):
        grouped.setdefault(int(timestamp), []).append(float(value))

    ordered = sorted((timestamp, float(np.mean(items))) for timestamp, items in grouped.items())
    return (
        np.asarray([item[0] for item in ordered], dtype=np.int64),
        np.asarray([item[1] for item in ordered], dtype=np.float64),
    )


def _forecast_with_arima(
    train_values: np.ndarray,
    test_length: int,
    order: tuple[int, int, int],
) -> np.ndarray:
    if ARIMA is None:
        raise RuntimeError(f"statsmodels ARIMA is unavailable: {_STATSMODELS_IMPORT_ERROR}")

    model = ARIMA(
        train_values,
        order=order,
        enforce_stationarity=False,
        enforce_invertibility=False,
    )
    fit_model = model.fit()
    forecast = fit_model.forecast(steps=test_length)
    return np.asarray(forecast, dtype=np.float64)


def _fallback_forecast(train_values: np.ndarray, test_length: int) -> np.ndarray:
    """Stable fallback when a particular ARIMA order cannot be fit."""
    if len(train_values) == 0:
        return np.zeros(test_length, dtype=np.float64)

    window = min(10, len(train_values))
    baseline = float(np.nanmedian(train_values[-window:]))
    return np.full(test_length, baseline, dtype=np.float64)


def _fill_small_prediction_gaps(prediction: np.ndarray, bridge_gap: int) -> np.ndarray:
    if bridge_gap <= 0 or len(prediction) == 0:
        return prediction.astype(np.int8, copy=True)

    result = prediction.astype(np.int8, copy=True)
    anomaly_indices = np.flatnonzero(result == 1)
    if len(anomaly_indices) < 2:
        return result

    for left, right in zip(anomaly_indices[:-1], anomaly_indices[1:]):
        gap = int(right - left - 1)
        if 0 < gap <= bridge_gap:
            result[left + 1 : right] = 1

    return result


def _robust_scale(values: np.ndarray) -> float:
    if len(values) == 0:
        return 0.0
    median = float(np.nanmedian(values))
    mad = float(np.nanmedian(np.abs(values - median)))
    scale = 1.4826 * mad
    if scale > 0:
        return scale
    std = float(np.nanstd(values))
    return std if math.isfinite(std) else 0.0


def classify_pattern(segment_values: np.ndarray, baseline_values: np.ndarray) -> str:
    """
    Deterministic fallback pattern classifier.

    DTW is the primary classifier. This fallback is used only when no DTW
    model is supplied or when a specific DTW prediction fails.
    """
    if len(segment_values) == 0:
        return "shiftUp"

    baseline_median = (
        float(np.nanmedian(baseline_values))
        if len(baseline_values)
        else 0.0
    )
    segment_mean = float(np.nanmean(segment_values))

    if segment_mean > baseline_median:
        return "shiftUp"
    if segment_mean < baseline_median:
        return "dip"

    deviations = segment_values - baseline_median
    max_index = int(np.nanargmax(np.abs(deviations)))
    return "shiftUp" if float(deviations[max_index]) >= 0 else "dip"


def predict_segment_pattern(
    segment_values: np.ndarray,
    baseline_values: np.ndarray,
    pattern_classifier: DtwPatternClassifier | None,
) -> str:
    """Return a DTW label, with a guaranteed deterministic fallback."""
    if pattern_classifier is not None:
        try:
            pattern = str(
                pattern_classifier.predict_pattern(segment_values)
            ).strip()
            if pattern:
                return pattern
        except Exception:
            # A malformed/very short segment must not remove the whole KPI
            # result. The fallback still adds a valid pattern field.
            pass

    return classify_pattern(segment_values, baseline_values)


def _select_peak_core_bounds(
    scores: np.ndarray,
    *,
    core_score_ratio: float,
    max_segment_points: int,
    min_segment_points: int,
) -> tuple[int, int, float]:
    """
    Select the strongest contiguous core inside one candidate segment.

    The core contains the maximum-severity point and expands while adjacent
    points retain at least ``core_score_ratio`` of the peak severity. It is
    capped at ``max_segment_points`` points.
    """
    if len(scores) == 0:
        return 0, -1, 0.0

    safe_scores = np.asarray(scores, dtype=np.float64)
    safe_scores = np.where(np.isfinite(safe_scores), safe_scores, 0.0)

    peak_index = int(np.argmax(safe_scores))
    peak_score = float(safe_scores[peak_index])
    core_threshold = peak_score * min(max(float(core_score_ratio), 0.0), 1.0)

    left = peak_index
    right = peak_index

    while left > 0 and float(safe_scores[left - 1]) >= core_threshold:
        left -= 1
    while right + 1 < len(safe_scores) and float(safe_scores[right + 1]) >= core_threshold:
        right += 1

    required_points = min(max(int(min_segment_points), 1), len(safe_scores))
    while right - left + 1 < required_points:
        left_score = float(safe_scores[left - 1]) if left > 0 else -math.inf
        right_score = (
            float(safe_scores[right + 1])
            if right + 1 < len(safe_scores)
            else -math.inf
        )
        if left_score == -math.inf and right_score == -math.inf:
            break
        if right_score > left_score:
            right += 1
        else:
            left -= 1

    if max_segment_points > 0 and right - left + 1 > max_segment_points:
        window_length = min(int(max_segment_points), len(safe_scores))
        minimum_start = max(left, peak_index - window_length + 1)
        maximum_start = min(peak_index, right - window_length + 1)

        best_start = minimum_start
        best_sum = -math.inf
        for window_start in range(minimum_start, maximum_start + 1):
            window_end = window_start + window_length
            window_sum = float(np.sum(safe_scores[window_start:window_end]))
            if window_sum > best_sum:
                best_sum = window_sum
                best_start = window_start

        left = best_start
        right = best_start + window_length - 1

    return left, right, peak_score


def extract_abnormal_segments(
    prediction: np.ndarray,
    values: np.ndarray,
    timestamps: np.ndarray,
    baseline_values: np.ndarray,
    min_segment_points: int,
    point_severity: np.ndarray,
    max_segment_points: int = 15,
    core_score_ratio: float = 0.6,
    pattern_classifier: DtwPatternClassifier | None = None,
) -> list[dict[str, Any]]:
    """
    Convert point predictions to compact peak-centered segments.

    Internal anomaly scores are used only for ranking and are removed before
    the public JSON is written.
    """
    if not (
        len(prediction)
        == len(values)
        == len(timestamps)
        == len(point_severity)
    ):
        raise ValueError(
            "prediction, values, timestamps, and point_severity "
            "must have equal length"
        )

    segments: list[dict[str, Any]] = []
    start_index: int | None = None

    def append_segment(candidate_start: int, candidate_end: int) -> None:
        if candidate_end - candidate_start + 1 < min_segment_points:
            return

        relative_start, relative_end, peak_score = _select_peak_core_bounds(
            point_severity[candidate_start : candidate_end + 1],
            core_score_ratio=core_score_ratio,
            max_segment_points=max_segment_points,
            min_segment_points=min_segment_points,
        )
        if relative_end < relative_start:
            return

        start = candidate_start + relative_start
        end = candidate_start + relative_end
        segment_values = values[start : end + 1]
        segment_scores = point_severity[start : end + 1]

        rounded_values: list[float] = []
        for raw_value in segment_values:
            number = float(raw_value)
            rounded = round(number, 4)
            if rounded == 0.0 and number != 0.0:
                rounded = round(number, 8)
            rounded_values.append(rounded)

        pattern = predict_segment_pattern(
            segment_values=segment_values,
            baseline_values=baseline_values,
            pattern_classifier=pattern_classifier,
        )
        finite_scores = segment_scores[np.isfinite(segment_scores)]
        mean_score = float(np.mean(finite_scores)) if len(finite_scores) else 0.0

        segments.append(
            {
                "start_time": int(timestamps[start]),
                "end_time": int(timestamps[end]),
                "value": rounded_values,
                "pattern": pattern,
                "_anomaly_score": round(float(peak_score), 8),
                "_mean_anomaly_score": round(mean_score, 8),
            }
        )

    for index, flag in enumerate(prediction):
        if int(flag) == 1 and start_index is None:
            start_index = index
        elif int(flag) == 0 and start_index is not None:
            append_segment(start_index, index - 1)
            start_index = None

    if start_index is not None:
        append_segment(start_index, len(prediction) - 1)

    return segments


def detect_candidate(
    candidate: SeriesCandidate,
    parameters: DetectionParameters,
    train_ratio: float,
    min_points: int,
    bridge_gap: int,
    min_segment_points: int,
    max_segment_points: int,
    core_score_ratio: float,
    pattern_classifier: DtwPatternClassifier | None = None,
) -> CandidateDetection:
    timestamps, values = _clean_series(candidate.timestamps, candidate.values)
    config = METRIC_CONFIGS[candidate.metric_name]
    upper_bound = (
        float(candidate.declared_upper_bound)
        if candidate.declared_upper_bound is not None
        else config.upper_bound
    )

    if len(values) < min_points:
        return CandidateDetection(
            metric_name=candidate.metric_name,
            source_id=candidate.source_id,
            upper_bound=upper_bound,
            segments=[],
            order=parameters.order,
            anomaly_ratio=parameters.anomaly_ratio,
            parameter_source=parameters.source,
            point_count=len(values),
            train_point_count=0,
            test_point_count=0,
            status="skipped",
            message=f"not enough points: {len(values)} < {min_points}",
        )

    train_size = int(math.floor(len(values) * train_ratio))
    train_size = max(min_points, train_size)
    train_size = min(train_size, len(values) - 1)

    if train_size <= 0 or train_size >= len(values):
        return CandidateDetection(
            metric_name=candidate.metric_name,
            source_id=candidate.source_id,
            upper_bound=upper_bound,
            segments=[],
            order=parameters.order,
            anomaly_ratio=parameters.anomaly_ratio,
            parameter_source=parameters.source,
            point_count=len(values),
            train_point_count=max(train_size, 0),
            test_point_count=max(len(values) - train_size, 0),
            status="skipped",
            message="cannot create a non-empty train/test split",
        )

    train_values = values[:train_size]
    test_values = values[train_size:]
    test_timestamps = timestamps[train_size:]

    try:
        forecast = _forecast_with_arima(train_values, len(test_values), parameters.order)
        forecast_source = "arima"
    except Exception as exc:
        forecast = _fallback_forecast(train_values, len(test_values))
        forecast_source = f"fallback: {type(exc).__name__}: {exc}"

    if len(forecast) != len(test_values):
        forecast = np.resize(forecast, len(test_values))

    residual_errors = np.square(test_values - forecast)
    finite_errors = residual_errors[np.isfinite(residual_errors)]

    if len(finite_errors) == 0:
        residual_prediction = np.zeros(len(test_values), dtype=np.int8)
    else:
        ratio = min(max(float(parameters.anomaly_ratio), 0.0), 100.0)
        percentile = 100.0 - ratio
        threshold = float(np.percentile(finite_errors, percentile))
        residual_prediction = (residual_errors > threshold).astype(np.int8)

        # With very short tests, strict > can yield no point at the maximum.
        if ratio > 0 and residual_prediction.sum() == 0 and len(test_values) > 0:
            residual_prediction[int(np.nanargmax(residual_errors))] = 1

    # Domain upper bounds are combined with ARIMA residual anomalies.
    upper_prediction = (test_values > upper_bound).astype(np.int8)
    prediction = np.maximum(residual_prediction, upper_prediction)
    prediction = _fill_small_prediction_gaps(prediction, bridge_gap)

    # The loose threshold discovers candidates. Severity then selects only
    # the strongest compact core without reducing recall at detection time.
    if len(finite_errors):
        positive_threshold = max(
            float(threshold),
            float(np.nanmedian(finite_errors)),
            np.finfo(np.float64).eps,
        )
        residual_severity = residual_errors / positive_threshold
    else:
        residual_severity = np.zeros(len(test_values), dtype=np.float64)

    value_scale = max(
        _robust_scale(train_values),
        abs(float(upper_bound)) * 0.05,
        np.finfo(np.float64).eps,
    )
    upper_severity = np.maximum(
        (test_values - float(upper_bound)) / value_scale,
        0.0,
    )
    point_severity = np.maximum(residual_severity, upper_severity)
    point_severity = np.where(np.isfinite(point_severity), point_severity, 0.0)

    segments = extract_abnormal_segments(
        prediction=prediction,
        values=test_values,
        timestamps=test_timestamps,
        baseline_values=train_values,
        min_segment_points=min_segment_points,
        point_severity=point_severity,
        max_segment_points=max_segment_points,
        core_score_ratio=core_score_ratio,
        pattern_classifier=pattern_classifier,
    )

    return CandidateDetection(
        metric_name=candidate.metric_name,
        source_id=candidate.source_id,
        upper_bound=upper_bound,
        segments=segments,
        order=parameters.order,
        anomaly_ratio=parameters.anomaly_ratio,
        parameter_source=parameters.source,
        point_count=len(values),
        train_point_count=len(train_values),
        test_point_count=len(test_values),
        status="success",
        message=forecast_source,
    )


def _segment_rank_key(segment: Mapping[str, Any]) -> tuple[float, float, int]:
    """Peak severity, mean severity, then concentration."""
    return (
        float(segment.get("_anomaly_score") or 0.0),
        float(segment.get("_mean_anomaly_score") or 0.0),
        -len(segment.get("value") or []),
    )


def _clean_output_segment(segment: Mapping[str, Any]) -> dict[str, Any]:
    """Remove private ranking fields from final output."""
    return {
        "start_time": int(segment["start_time"]),
        "end_time": int(segment["end_time"]),
        "value": list(segment.get("value") or []),
        "pattern": str(segment.get("pattern") or "shiftUp"),
    }


def aggregate_detections(
    detections: Sequence[CandidateDetection],
    include_empty_metrics: bool,
    max_segments_per_metric: int = 1,
) -> dict[str, Any]:
    """Keep only the globally strongest segment(s) for each logical KPI."""
    if max_segments_per_metric < 1:
        raise ValueError("max_segments_per_metric must be at least 1")

    output: dict[str, Any] = {}
    by_metric: dict[str, list[CandidateDetection]] = {}

    for detection in detections:
        by_metric.setdefault(detection.metric_name, []).append(detection)

    for metric_name in METRIC_CONFIGS:
        metric_detections = by_metric.get(metric_name, [])
        if not metric_detections:
            continue

        upper_bound = min(item.upper_bound for item in metric_detections)
        ranked_segments = [
            dict(segment)
            for detection in metric_detections
            for segment in detection.segments
        ]
        ranked_segments.sort(key=_segment_rank_key, reverse=True)

        selected_segments = [
            _clean_output_segment(segment)
            for segment in ranked_segments[:max_segments_per_metric]
        ]

        if not selected_segments and not include_empty_metrics:
            continue

        output[metric_name] = {
            "upper_bound": round(float(upper_bound), 4),
            "abnormal_timing_segments": selected_segments,
        }

    return output


def detect_case_kpis(
    case_document: Mapping[str, Any],
    *,
    case_path: Path | None = None,
    parameter_dir: Path | None = DEFAULT_PARAMETER_DIR,
    train_ratio: float = 0.9,
    min_points: int = 20,
    bridge_gap: int = 2,
    min_segment_points: int = 1,
    max_segment_points: int = 15,
    core_score_ratio: float = 0.6,
    max_segments_per_metric: int = 1,
    forced_order: tuple[int, int, int] | None = None,
    forced_anomaly_ratio: float | None = None,
    minimum_anomaly_ratio: float = 10.0,
    pattern_classifier: DtwPatternClassifier | None = None,
    include_empty_metrics: bool = False,
) -> tuple[dict[str, Any], dict[str, Any]]:
    if not 0.0 < train_ratio < 1.0:
        raise ValueError("train_ratio must be between 0 and 1")
    if min_points < 3:
        raise ValueError("min_points must be at least 3")
    if bridge_gap < 0:
        raise ValueError("bridge_gap cannot be negative")
    if min_segment_points < 1:
        raise ValueError("min_segment_points must be at least 1")
    if max_segment_points < 1:
        raise ValueError("max_segment_points must be at least 1")
    if not 0.0 <= core_score_ratio <= 1.0:
        raise ValueError("core_score_ratio must be between 0 and 1")
    if max_segments_per_metric < 1:
        raise ValueError("max_segments_per_metric must be at least 1")
    if not 0.0 <= minimum_anomaly_ratio <= 100.0:
        raise ValueError("minimum_anomaly_ratio must be between 0 and 100")

    kpi_root = find_kpi_root(case_document)
    raw_candidates = extract_series_candidates(kpi_root)
    capacities = collect_bandwidth_capacities(kpi_root)
    candidates = [normalize_candidate_values(candidate, capacities) for candidate in raw_candidates]

    candidates = [candidate for candidate in candidates if candidate.values.size]

    scenario = infer_scenario(case_path)
    detections: list[CandidateDetection] = []

    parameter_cache: dict[str, DetectionParameters] = {}
    for candidate in candidates:
        parameters = parameter_cache.get(candidate.metric_name)
        if parameters is None:
            parameters = resolve_detection_parameters(
                metric_name=candidate.metric_name,
                scenario=scenario,
                parameter_dir=parameter_dir,
                forced_order=forced_order,
                forced_anomaly_ratio=forced_anomaly_ratio,
            )

            # anomaly_ratio means "top X percent residual errors are abnormal".
            # A larger ratio lowers the percentile threshold and is therefore
            # looser/more sensitive. Explicit command-line ratios are respected.
            if (
                forced_anomaly_ratio is None
                and parameters.anomaly_ratio < minimum_anomaly_ratio
            ):
                parameters = DetectionParameters(
                    order=parameters.order,
                    anomaly_ratio=float(minimum_anomaly_ratio),
                    source=(
                        f"{parameters.source}; "
                        f"relaxed_min_ratio={minimum_anomaly_ratio}"
                    ),
                )

            parameter_cache[candidate.metric_name] = parameters

        detections.append(
            detect_candidate(
                candidate=candidate,
                parameters=parameters,
                train_ratio=train_ratio,
                min_points=min_points,
                bridge_gap=bridge_gap,
                min_segment_points=min_segment_points,
                max_segment_points=max_segment_points,
                core_score_ratio=core_score_ratio,
                pattern_classifier=pattern_classifier,
            )
        )

    anomaly_kpi = aggregate_detections(
        detections,
        include_empty_metrics,
        max_segments_per_metric=max_segments_per_metric,
    )

    firstoccurutc = extract_firstoccurutc(case_document)
    minimum_segments, minimum_diagnostics = (
        collect_post_alarm_minimum_segments(
            candidates,
            firstoccurutc=firstoccurutc,
            window_seconds=POST_ALARM_MINIMUM_WINDOW_SECONDS,
            pattern_classifier=pattern_classifier,
        )
    )
    minimum_merge_diagnostics = (
        merge_post_alarm_minimum_segments(
            anomaly_kpi,
            minimum_segments,
            candidates,
            pattern_classifier=pattern_classifier,
        )
    )

    # Collect peaks before limiting the public output to short anomaly segments.
    # Ties across sources use the earliest observed timestamp.
    for metric in sorted(PACKET_LOSS_METRICS):
        sources = [c for c in candidates if c.metric_name == metric]
        if not sources:
            continue
        peaks = []
        for source in sources:
            idx = int(np.argmax(source.values))
            peaks.append((float(source.values[idx]), int(source.timestamps[idx])))
        peak, timestamp = min(peaks, key=lambda item: (-item[0], item[1]))
        if metric not in anomaly_kpi:
            thresholds = [d.upper_bound for d in detections if d.metric_name == metric]
            anomaly_kpi[metric] = {"upper_bound": min(thresholds), "abnormal_timing_segments": []}
        payload = anomaly_kpi[metric]
        payload["unit"] = "ratio"
        payload["upper_bound"] /= 100.0
        for segment in payload["abnormal_timing_segments"]:
            segment["value"] = [v / 100.0 for v in segment["value"]]
        payload["max_loss_rate"] = peak / 100.0
        payload["max_loss_rate_time"] = timestamp

    diagnostics = {
        "case_id": infer_case_id(case_document, case_path),
        "scenario": scenario,
        "recognized_series_count": len(candidates),
        "recognized_metrics": sorted({candidate.metric_name for candidate in candidates}),
        "bandwidth_capacity_count": len(capacities),
        "minimum_anomaly_ratio": minimum_anomaly_ratio,
        "max_segment_points": max_segment_points,
        "core_score_ratio": core_score_ratio,
        "max_segments_per_metric": max_segments_per_metric,
        "post_alarm_minimum_window": minimum_diagnostics,
        "post_alarm_minimum_merge": minimum_merge_diagnostics,
        "dtw_pattern_classifier": (
            {
                "enabled": True,
                "backend": pattern_classifier.backend_used,
                "sequence_length": pattern_classifier.sequence_length,
                "n_neighbors": pattern_classifier.n_neighbors,
                "training_summary": pattern_classifier.training_summary,
            }
            if pattern_classifier is not None
            else {"enabled": False, "backend": "fallback"}
        ),
        "detections": [
            {
                "metric_name": item.metric_name,
                "source_id": item.source_id,
                "status": item.status,
                "message": item.message,
                "point_count": item.point_count,
                "train_point_count": item.train_point_count,
                "test_point_count": item.test_point_count,
                "order": list(item.order),
                "anomaly_ratio": item.anomaly_ratio,
                "parameter_source": item.parameter_source,
                "segment_count": len(item.segments),
            }
            for item in detections
        ],
    }
    return anomaly_kpi, diagnostics


# ---------------------------------------------------------------------------
# Standalone CLI
# ---------------------------------------------------------------------------


def atomic_write_json(output_file: Path, document: Mapping[str, Any]) -> None:
    output_file.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temp_name = tempfile.mkstemp(
        prefix=output_file.name + ".",
        suffix=".tmp",
        dir=str(output_file.parent),
    )

    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as file:
            json.dump(document, file, ensure_ascii=False, indent=2)
            file.write("\n")
            file.flush()
            os.fsync(file.fileno())
        os.replace(temp_name, output_file)
    finally:
        if os.path.exists(temp_name):
            os.unlink(temp_name)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Recursively extract KPI v nodes, detect ARIMA abnormal segments, "
            "and classify every segment pattern with DTW."
        )
    )
    parser.add_argument("--case", required=True, type=Path)
    parser.add_argument("--output-file", type=Path)
    parser.add_argument("--parameter-dir", type=Path, default=DEFAULT_PARAMETER_DIR)
    parser.add_argument("--train-ratio", type=float, default=0.9)
    parser.add_argument("--min-points", type=int, default=20)
    parser.add_argument(
        "--bridge-gap",
        type=int,
        default=2,
        help="Join abnormal points separated by at most this many normal points.",
    )
    parser.add_argument("--min-segment-points", type=int, default=1)
    parser.add_argument(
        "--max-segment-points",
        type=int,
        default=15,
        help="Maximum points retained around the strongest anomaly peak.",
    )
    parser.add_argument(
        "--core-score-ratio",
        type=float,
        default=0.6,
        help="Retain adjacent points with at least this fraction of peak severity.",
    )
    parser.add_argument(
        "--max-segments-per-metric",
        type=int,
        default=1,
        help="Keep only the strongest N segments for each logical KPI.",
    )
    parser.add_argument("--p", type=int)
    parser.add_argument("--d", type=int)
    parser.add_argument("--q", type=int)
    parser.add_argument(
        "--anomaly-ratio",
        type=float,
        help=(
            "Force the top X percent ARIMA residuals to be abnormal. "
            "A larger value is looser/more sensitive."
        ),
    )
    parser.add_argument(
        "--minimum-anomaly-ratio",
        type=float,
        default=10.0,
        help=(
            "Relax tuned/default ARIMA ratios to at least this value. "
            "Ignored when --anomaly-ratio is explicitly provided."
        ),
    )
    parser.add_argument("--include-empty-metrics", action="store_true")
    parser.add_argument("--diagnostics-file", type=Path)

    parser.add_argument("--disable-dtw", action="store_true")
    parser.add_argument("--dtw-model", type=Path, default=DEFAULT_DTW_MODEL_PATH)
    parser.add_argument("--dtw-data-root", type=Path, default=DEFAULT_DTW_DATA_ROOT)
    parser.add_argument(
        "--dtw-rca-units-root",
        type=Path,
        default=DEFAULT_RCA_UNITS_ROOT,
    )
    parser.add_argument("--dtw-scenario", action="append", dest="dtw_scenarios")
    parser.add_argument(
        "--dtw-sequence-length",
        type=int,
        default=DEFAULT_SEQUENCE_LENGTH,
    )
    parser.add_argument(
        "--dtw-split-index",
        type=int,
        default=DEFAULT_TRAIN_SPLIT_INDEX,
    )
    parser.add_argument("--dtw-neighbors", type=int, default=1)
    parser.add_argument(
        "--dtw-backend",
        choices=("auto", "aeon", "numpy"),
        default="auto",
    )
    parser.add_argument("--dtw-retrain", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    case_document = load_json_file(args.case)

    forced_order: tuple[int, int, int] | None = None
    order_values = (args.p, args.d, args.q)
    if any(value is not None for value in order_values):
        if not all(value is not None for value in order_values):
            raise SystemExit("--p, --d, and --q must be provided together")
        forced_order = (int(args.p), int(args.d), int(args.q))

    pattern_classifier: DtwPatternClassifier | None = None
    if not args.disable_dtw:
        try:
            pattern_classifier = load_or_train_dtw_classifier(
                model_path=args.dtw_model,
                data_root=args.dtw_data_root,
                rca_units_root=args.dtw_rca_units_root,
                scenarios=tuple(args.dtw_scenarios or DEFAULT_SCENARIOS),
                sequence_length=args.dtw_sequence_length,
                split_index=args.dtw_split_index,
                n_neighbors=args.dtw_neighbors,
                backend=args.dtw_backend,
                force_retrain=args.dtw_retrain,
            )
        except Exception as exc:
            print(
                "[WARN] DTW model unavailable; pattern fallback will be used: "
                f"{type(exc).__name__}: {exc}"
            )

    anomaly_kpi, diagnostics = detect_case_kpis(
        case_document,
        case_path=args.case,
        parameter_dir=args.parameter_dir,
        train_ratio=args.train_ratio,
        min_points=args.min_points,
        bridge_gap=args.bridge_gap,
        min_segment_points=args.min_segment_points,
        max_segment_points=args.max_segment_points,
        core_score_ratio=args.core_score_ratio,
        max_segments_per_metric=args.max_segments_per_metric,
        forced_order=forced_order,
        forced_anomaly_ratio=args.anomaly_ratio,
        minimum_anomaly_ratio=args.minimum_anomaly_ratio,
        pattern_classifier=pattern_classifier,
        include_empty_metrics=args.include_empty_metrics,
    )

    output_document = {"anomaly_kpi": anomaly_kpi}
    output_file = args.output_file or args.case.with_name(
        f"{infer_case_id(case_document, args.case)}_arima_dtw_anomaly_kpi.json"
    )
    atomic_write_json(output_file, output_document)

    if args.diagnostics_file is not None:
        atomic_write_json(args.diagnostics_file, diagnostics)

    print(f"case_id={diagnostics['case_id']}")
    print(f"scenario={diagnostics['scenario']}")
    print(f"recognized_series_count={diagnostics['recognized_series_count']}")
    print(f"recognized_metrics={diagnostics['recognized_metrics']}")
    print(f"minimum_anomaly_ratio={diagnostics['minimum_anomaly_ratio']}")
    print(
        "dtw_pattern_classifier="
        f"{diagnostics['dtw_pattern_classifier']}"
    )
    print(f"anomaly_metric_count={len(anomaly_kpi)}")
    print(f"output_file={output_file}")
    print(json.dumps(output_document, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
