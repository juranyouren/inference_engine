"""竞争型按 SOP 候选根因独立诊断，再由 Verifier 重排。"""

from __future__ import annotations

from .agent_runner import AgentInference, case_evidence, response_json, validate_ranking
from .competition_templates import load_competition_template
from .result_parser import candidate_categories


VERIFIER_PROMPT = (
    "你是 Verifier。结合 case 中的原始证据、SOP 的决策条件以及每条候选诊断的"
    "节点检查状态、证据引用和根因分支置信度，独立复核每个根因。"
    "把置信度作为参考，结合矛盾和缺失证据判断全部候选的相对可能性。"
    "候选 Agent 未报告的节点标为未知，请结合原始证据自行复核。"
    "只输出候选根因名称的 JSON 数组，按可能性从高到低排列，全部候选恰好出现一次。"
)


def _validate_judgment(judgment, template, name, *, check_node=False):
    if not isinstance(judgment, dict) or judgment.keys() != template.keys():
        raise ValueError(f"Agent 必须完整填写 {name} 对应的模板")
    outcome = judgment.get("condition_met")
    if check_node and outcome is not None and type(outcome) is not bool:
        raise ValueError(f"{name} 的 condition_met 必须为 true、false 或 null")
    confidence = judgment["confidence"]
    if check_node and outcome is None:
        valid_confidence = confidence is None
    else:
        valid_confidence = type(confidence) in (int, float) and 0 <= confidence <= 1
    if not valid_confidence:
        raise ValueError(f"{name} 的置信度无效")
    if not isinstance(judgment["reason"], str) or not judgment["reason"].strip():
        raise ValueError(f"{name} 缺少判断依据")


def _validate_diagnosis(response, prompt, templates):
    diagnosis = response_json(response)
    if not isinstance(diagnosis, dict) or not diagnosis.keys() <= {"decisions", "conclusion"}:
        raise ValueError("竞争型 Agent 必须返回 decisions 和 conclusion 对象")
    reported = diagnosis.get("decisions")
    if not isinstance(reported, dict):
        reported = {}
    decisions = {}
    for node_id, template in templates.items():
        node = reported.get(node_id)
        result_template = template["result_template"]
        if node is None or node == {}:
            decisions[node_id] = {**result_template, "reason": "候选 Agent 未报告该节点，等待 Verifier 复核。"}
            continue
        try:
            _validate_judgment(node, result_template, node_id, check_node=True)
            clues = node["diagnostic_clues"]
            if not isinstance(clues, list) or any(
                not isinstance(item, dict) or item.keys() != {"path", "value"}
                or not isinstance(item["path"], str) or not item["path"].startswith("$.")
                for item in clues
            ):
                raise ValueError(f"{node_id} 的证据必须是 path/value 列表")
        except ValueError as exc:
            decisions[node_id] = {**result_template, "reason": f"节点报告无效，等待 Verifier 复核。{exc}"}
        else:
            decisions[node_id] = node
    conclusion = diagnosis.get("conclusion")
    _validate_judgment(conclusion, prompt["conclusion_template"], "conclusion")
    return {"root_cause": prompt["root_cause"], "paths": prompt["paths"],
            "decisions": decisions, "conclusion": conclusion}


class CompetitionInference(AgentInference):
    """每个候选根因一个独立 Agent，Verifier 查看所有路径。"""

    mode = "competition"

    def _infer_case(self, case, output_dir, result):
        template = load_competition_template(case.alarm_type, case.sop)
        all_decisions = template["decision_templates"]
        by_name = {entry["root_cause"]: entry for entry in template["root_cause_prompts"].values()}
        candidates = candidate_categories(case.root_cause_candidates) or list(by_name)
        missing = [name for name in candidates if name not in by_name]
        if missing:
            raise ValueError(f"SOP 缺少候选根因路径: {', '.join(missing)}")

        evidence = case_evidence(case)
        requests = []
        jobs = []
        for index, name in enumerate(candidates, start=1):
            entry = by_name[name]
            node_ids = dict.fromkeys(
                step["decision_id"] for path in entry["paths"] for step in path
            )
            nodes = {node_id: all_decisions[node_id] for node_id in node_ids}
            payload = {
                "case": evidence, "root_cause": name, "paths": entry["paths"],
                "decision_templates": nodes, "conclusion_template": entry["conclusion_template"],
            }
            requests.append((f"reasoners/candidate_{index:02d}", entry["prompt"], payload))
            jobs.append((entry, nodes))

        answers = self.generator.generate(requests, output_dir)
        diagnoses = []
        result["diagnoses"] = diagnoses
        for answer, (entry, nodes) in zip(answers, jobs):
            diagnoses.append(_validate_diagnosis(answer, entry, nodes))
        used_nodes = dict.fromkeys(node_id for _entry, nodes in jobs for node_id in nodes)
        verifier_payload = {
            "case": evidence,
            "root_cause_candidates": candidates,
            "decision_conditions": {
                node_id: all_decisions[node_id]["condition"] for node_id in used_nodes
            },
            "diagnoses": diagnoses,
        }
        answer, = self.generator.generate([("verifier", VERIFIER_PROMPT, verifier_payload)], output_dir)
        result["final_result"] = answer
        return validate_ranking(response_json(answer), candidates)
