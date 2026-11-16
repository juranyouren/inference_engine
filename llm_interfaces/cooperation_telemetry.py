"""合作型的遥测目录及按 persona 选取原始证据。"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path


@lru_cache(maxsize=1)
def _base_template():
    path = Path(__file__).with_name("semantic_labels_template.json")
    return json.loads(path.read_text(encoding="utf-8"))


def _child_key(parent, field):
    return parent + "/" + field.replace("~", "~0").replace("/", "~1")


def _fields(node, parent=""):
    if isinstance(node, dict):
        children = ((_child_key(parent, field), value) for field, value in node.items())
    elif isinstance(node, list):
        children = ((parent + "/*", item) for item in node)
    else:
        return
    for key, value in children:
        yield key, value
        yield from _fields(value, key)


def has_telemetry(value):
    if isinstance(value, dict):
        return any(has_telemetry(item) for item in value.values())
    if isinstance(value, list):
        return any(has_telemetry(item) for item in value)
    if isinstance(value, str):
        return bool(value.strip())
    return value is not None


def normalize_telemetry(labels):
    """兼容以 JSON 字符串存储的遥测容器。"""
    normalized = dict(labels)
    for field in ("anomaly_logs", "anomaly_kpi", "anomaly_netstreams", "other_info"):
        raw = normalized.get(field)
        if not isinstance(raw, str) or not raw.lstrip().startswith(("{", "[")):
            continue
        try:
            decoded = json.loads(raw)
        except json.JSONDecodeError:
            continue
        normalized[field] = decoded
    return normalized


def semantic_labels_template(labels):
    base = _base_template()
    root_roles = {
        "/" + key.split("/", 2)[1]: role
        for role, fields in base.items()
        for key in fields
    }
    catalog = {
        role: {
            key: {"key": key, "desc": desc, "available": False}
            for key, desc in fields.items()
        }
        for role, fields in base.items()
    }
    for field, value in labels.items():
        fields = catalog[root_roles.get(_child_key("", field), "metadata")]
        for key, raw_value in _fields({field: value}):
            entry = fields.setdefault(key, {
                "key": key, "desc": "样本提供的扩展字段，结合字段名及上级遥测解释。", "available": False,
            })
            entry["available"] |= has_telemetry(raw_value)
    return {role: list(fields.values()) for role, fields in catalog.items()}


def select_telemetry(labels, focus):
    keys = {detail["key"] for detail in focus}
    missing = object()

    def project(value, key):
        if key in keys:
            return value
        if not any(path.startswith(key + "/") for path in keys):
            return missing
        if isinstance(value, dict):
            children = {}
            for field, item in value.items():
                selected = project(item, _child_key(key, field))
                if selected is not missing:
                    children[field] = selected
            return children if children else missing
        if isinstance(value, list):
            items = [project(item, key + "/*") for item in value]
            if any(item is not missing for item in items):
                return [None if item is missing else item for item in items]
        return missing

    selected = project(labels, "")
    return {} if selected is missing else selected
