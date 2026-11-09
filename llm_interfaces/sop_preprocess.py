"""将结构化 SOP 编译成竞争型推理的决策模板和候选根因提示词。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any


PROMPT_INSTRUCTIONS = (
    "评估模板中的 root_cause，只使用 case 的语义标签和告警时间作为证据。"
    "只输出 JSON，包含 decisions 和 conclusion 两个对象。decisions 以 paths 引用的"
    "决策 ID 为键，每个节点按 result_template 填写一次。diagnostic_clues 每项记录相关"
    "证据的具体 path 和原始 value。condition_met 是实际判断，取 true、false 或 null。"
    "required_condition_met 是路径要求。证据缺失、无法解释或冲突时填 null 并说明原因。"
    "按 conclusion_template 填写 conclusion。有一条路径的所有节点均匹配时，"
    "supported 填 true。每条路径均有明确不匹配的节点时填 false，其余填 null。"
    "已判定时 confidence 取 0 到 1，未知时填 null。reason 简述判断依据。"
)


def compile_sop(sop: dict[str, Any]) -> dict[str, Any]:
    """从入口展开所有可达路径，允许不同前缀复用同一决策节点。"""
    if not isinstance(sop, dict):
        raise ValueError("SOP 必须是 JSON 对象")
    alarm_type = sop.get("alarm_type")
    if not isinstance(alarm_type, str) or not alarm_type.strip():
        raise ValueError("alarm_type 不能为空")

    decisions = sop.get("decisions")
    root_causes = sop.get("root_causes")
    if not isinstance(decisions, dict) or not decisions:
        raise ValueError("decisions 必须是非空对象")
    if not isinstance(root_causes, dict) or not root_causes:
        raise ValueError("root_causes 必须是非空对象")
    if decisions.keys() & root_causes.keys():
        raise ValueError("决策节点 ID 和根因 ID 必须互异")
    if any(not isinstance(name, str) or not name.strip() for name in root_causes.values()):
        raise ValueError("根因名称必须为非空字符串")
    if len(set(root_causes.values())) != len(root_causes):
        raise ValueError("根因名称必须唯一")

    entry = sop.get("entry")
    if not isinstance(entry, str) or entry not in decisions:
        raise ValueError("entry 必须指向一个决策节点")

    nodes = decisions.keys() | root_causes.keys()
    decision_templates = {}
    for node_id, node in decisions.items():
        if not isinstance(node, dict):
            raise ValueError(f"{node_id} 必须是对象")
        for field in ("condition", "guidance"):
            value = node.get(field)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{node_id}.{field} 必须是非空字符串")
        clue_paths = node.get("diagnostic_clue_paths")
        if not isinstance(clue_paths, list) or not clue_paths or any(
            not isinstance(path, str) or not path.startswith("$.") for path in clue_paths
        ):
            raise ValueError(f"{node_id}.diagnostic_clue_paths 必须包含从 $. 开始的证据路径")
        branches = node.get("branches")
        if not isinstance(branches, dict) or branches.keys() != {"true", "false"}:
            raise ValueError(f"{node_id}.branches 必须定义 true 和 false")
        for target in branches.values():
            if not isinstance(target, str) or target not in nodes:
                raise ValueError(f"{node_id} 引用了不存在的节点 {target!r}")
        if branches["true"] == branches["false"]:
            raise ValueError(f"{node_id} 的两个分支必须指向不同节点")

        decision_templates[node_id] = {
            "condition": node["condition"],
            "diagnostic_clue_paths": clue_paths,
            "guidance": node["guidance"],
            "result_template": {
                "diagnostic_clues": [],
                "condition_met": None,
                "confidence": None,
                "reason": "",
            },
        }

    paths_by_cause = {cause_id: [] for cause_id in root_causes}
    reached = set()

    def visit(node_id, path, active):
        if node_id in active:
            raise ValueError(f"SOP 存在循环，重复进入 {node_id}")
        reached.add(node_id)
        if node_id in root_causes:
            paths_by_cause[node_id].append(path)
            return

        branches = decisions[node_id]["branches"]
        for outcome, target in branches.items():
            step = {"decision_id": node_id, "required_condition_met": outcome == "true"}
            visit(target, path + [step], active | {node_id})

    visit(entry, [], set())
    if nodes - reached:
        raise ValueError(f"SOP 存在不可达节点: {', '.join(sorted(nodes - reached))}")

    root_cause_prompts = {}
    for cause_id, name in root_causes.items():
        root_cause_prompts[cause_id] = {
            "root_cause": name,
            "prompt": PROMPT_INSTRUCTIONS,
            "paths": paths_by_cause[cause_id],
            "conclusion_template": {
                "supported": None,
                "confidence": None,
                "reason": "",
            },
        }

    return {
        "alarm_type": alarm_type,
        "decision_templates": decision_templates,
        "root_cause_prompts": root_cause_prompts,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sop_file", type=Path, help="结构化 SOP JSON 文件")
    parser.add_argument("output_file", type=Path, help="决策及候选根因提示词 JSON 文件")
    args = parser.parse_args()
    try:
        if args.sop_file.resolve() == args.output_file.resolve():
            raise ValueError("输入和输出必须使用不同的文件")
        sop = json.loads(args.sop_file.read_text(encoding="utf-8-sig"))
        templates = compile_sop(sop)
        args.output_file.parent.mkdir(parents=True, exist_ok=True)
        staged_file = NamedTemporaryFile(
            mode="w", encoding="utf-8", newline="\n",
            dir=args.output_file.parent, delete=False,
        )
        staged_path = Path(staged_file.name)
        try:
            with staged_file:
                json.dump(templates, staged_file, ensure_ascii=False, indent=2)
                staged_file.write("\n")
            staged_path.replace(args.output_file)
        finally:
            staged_path.unlink(missing_ok=True)
    except (OSError, ValueError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
