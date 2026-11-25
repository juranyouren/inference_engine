"""将结构化 SOP 编译成竞争型推理的决策模板和候选根因提示词。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any


PROMPT_INSTRUCTIONS = (
    "评估给定的 root_cause。只输出一个 JSON 对象，顶层恰含 decisions 和 conclusion。\n"
    "1. 收集 paths 中所有不同的 decision_id，逐一作为 decisions 的键。共享节点只填一次。"
    "复制对应的 result_template，只填写模板已有字段。\n"
    "2. 逐节点阅读 condition 和 guidance，核对设备端、时间窗、单位及比较符。"
    "可查阅 case 的全部相关字段，diagnostic_clue_paths 是查找提示。"
    "diagnostic_clues 填数组，每项恰含 path 和 value。以 case 为 $，填写实际字段路径和原始 JSON 值；"
    "计算或单位换算写在 reason。没有可引用的观测时填 []；有相互矛盾的观测时分别引用。\n"
    "3. condition_met 填条件本身的实际判断：观测足以确认条件成立填 true，足以确认不成立填 false，"
    "缺少判定所需信息或证据冲突无法消解填 null。候选名称、描述和 required_condition_met 均不能充当观测。"
    "节点 confidence 表示对该 true 或 false 判断的把握（0 到 1），判断为 null 时 confidence 也填 null。"
    "reason 写明哪个观测如何满足或违反条件，或具体缺什么。\n"
    "4. 按 conclusion_template 填 confidence 和 reason。逐条路径将每个节点的 condition_met 与该步的 "
    "required_condition_met 对照：有明确相反的节点，该路径被反证；全部相符才完整支持；"
    "其余为待定。多条路径任选一条成立即可。所有路径均被明确反证时根因 confidence 填 0；"
    "有完整支持路径时按该路径证据把握评分；只有待定路径仍可能成立时给出保守的 0 到 1 数值，"
    "说明缺口，不能把未知计为匹配。reason 点名最有依据的路径及其关键节点，或指出各路径的反证。"
    "例如某步要求 false，观测支持 true，则该节点填 true，即使节点 confidence 很高，这条路径也被反证。"
    "最终根因排序由 Verifier 完成。"
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
