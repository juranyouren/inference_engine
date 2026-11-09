"""将当前 case 的 SOP 解析为竞争型可执行模板。"""

from __future__ import annotations

import json
import re
from functools import lru_cache
from pathlib import Path

from .sop_preprocess import compile_sop


def _procedure_graph(alarm_type, sop):
    records = re.split(r"(?m)(?=^\s*Procedure\s+\d+:::)", sop)
    decisions = {}
    terminal_names = {}
    for record in records:
        if not record.strip():
            continue
        fields = record.strip().split(":::")
        head = re.fullmatch(r"Procedure\s+(\d+)", fields[0])
        if not head or len(fields) != 4:
            raise ValueError("SOP 必须为 Procedure Decision/Terminal 格式或结构化 SOP JSON")
        step = int(head[1])
        kind, body, branches = fields[1:]
        if step in decisions or step in terminal_names:
            raise ValueError(f"SOP 包含重复的 Procedure {step}")
        if kind == "Decision":
            branch = re.fullmatch(r"Yes::Procedure\s+(\d+)::No::Procedure\s+(\d+)", branches)
            if not branch:
                raise ValueError(f"Procedure {step} 缺少 Yes 或 No 分支")
            decisions[step] = (body.strip(), int(branch[1]), int(branch[2]))
        elif kind == "Terminal":
            cause = re.search(r'诊断根因为\s*["“]([^"”]+)["”]', body)
            if not cause:
                raise ValueError(f"Procedure {step} 缺少带引号的根因名称")
            terminal_names[step] = cause[1]
        else:
            raise ValueError(f"Procedure {step} 的类型不受支持: {kind}")

    if not decisions or not terminal_names:
        raise ValueError("SOP 必须包含决策和结论")
    cause_ids = {}
    for step in sorted(terminal_names):
        name = terminal_names[step]
        if name not in cause_ids:
            cause_ids[name] = f"rc_{len(cause_ids) + 1}"

    def node_id(step):
        if step in terminal_names:
            return cause_ids[terminal_names[step]]
        return f"step_{step:02d}"

    return {
        "alarm_type": alarm_type,
        "entry": node_id(min(decisions.keys() | terminal_names.keys())),
        "decisions": {
            node_id(step): {
                "condition": condition,
                "diagnostic_clue_paths": ["$.semantic_labels", "$.alarm_time"],
                "guidance": "绑定与本节点有关的具体字段。缺失证据时保留未知。",
                "branches": {"true": node_id(yes), "false": node_id(no)},
            }
            for step, (condition, yes, no) in decisions.items()
        },
        "root_causes": {cause_id: name for name, cause_id in cause_ids.items()},
    }


def _signature(template):
    return (
        {key: node["condition"] for key, node in template["decision_templates"].items()},
        {node["root_cause"]: node["paths"] for node in template["root_cause_prompts"].values()},
    )


@lru_cache(maxsize=128)
def load_competition_template(alarm_type, sop):
    if sop.startswith("{"):
        graph = json.loads(sop)
        if graph.get("alarm_type", alarm_type) != alarm_type:
            raise ValueError("结构化 SOP 的告警类别与 case 不一致")
        graph["alarm_type"] = alarm_type
        return compile_sop(graph)

    compiled = compile_sop(_procedure_graph(alarm_type, sop))
    template_dir = Path(__file__).with_name("sop_templates")
    for path in sorted(template_dir.glob("*.templates.json")):
        template = json.loads(path.read_text(encoding="utf-8"))
        if template["alarm_type"] == alarm_type and _signature(template) == _signature(compiled):
            return template
    return compiled
