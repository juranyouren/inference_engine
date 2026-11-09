"""竞争型按 SOP 候选根因独立诊断，再由 Verifier 重排。"""

from __future__ import annotations

from .agent_runner import AgentInference, case_evidence, response_json, validate_ranking
from .competition_templates import load_competition_template
from .result_parser import candidate_categories


VERIFIER_PROMPT = (
    "你是 Verifier。结合 case 中的原始证据、SOP 的决策条件以及每条候选诊断的"
    "证据引用、判断和结论，独立核对路径是否成立。发现矛盾或缺少证据时降低可信度。"
    "只输出候选根因名称的 JSON 数组，按可能性从高到低排列，全部候选恰好出现一次。"
)


def _validate_judgment(judgment, template, outcome_key):
    if not isinstance(judgment, dict) or judgment.keys() != template.keys():
        raise ValueError(f"Agent 必须完整填写 {outcome_key} 对应的模板")
    outcome = judgment[outcome_key]
    if outcome is not None and type(outcome) is not bool:
        raise ValueError(f"{outcome_key} 必须为 true、false 或 null")
    confidence = judgment["confidence"]
    if outcome is None:
        valid_confidence = confidence is None
    else:
        valid_confidence = type(confidence) in (int, float) and 0 <= confidence <= 1
    if not valid_confidence:
        raise ValueError(f"{outcome_key} 的置信度无效")
    if not isinstance(judgment["reason"], str) or not judgment["reason"].strip():
        raise ValueError(f"{outcome_key} 缺少判断依据")


def _path_support(paths, decisions):
    unmatched = []
    for path in paths:
        matches = []
        for step in path:
            actual = decisions[step["decision_id"]]["condition_met"]
            matches.append(None if actual is None else actual is step["required_condition_met"])
        if all(match is True for match in matches):
            return True
        unmatched.append(False in matches)
    return False if all(unmatched) else None


def _validate_diagnosis(response, prompt, templates):
    diagnosis = response_json(response)
    if not isinstance(diagnosis, dict):
        raise ValueError("竞争型 Agent 必须返回 decisions 和 conclusion 对象")
    decisions = diagnosis.get("decisions")
    if not isinstance(decisions, dict) or decisions.keys() != templates.keys():
        raise ValueError("竞争型 Agent 必须填写路径涉及的全部决策节点")
    for node_id, node in decisions.items():
        _validate_judgment(node, templates[node_id]["result_template"], "condition_met")
        clues = node["diagnostic_clues"]
        if not isinstance(clues, list) or any(
            not isinstance(item, dict) or item.keys() != {"path", "value"}
            or not isinstance(item["path"], str) or not item["path"].startswith("$.")
            for item in clues
        ):
            raise ValueError(f"{node_id} 的证据必须是 path/value 列表")
    conclusion = diagnosis.get("conclusion")
    _validate_judgment(conclusion, prompt["conclusion_template"], "supported")
    if conclusion["supported"] is not _path_support(prompt["paths"], decisions):
        raise ValueError("根因支持结论与所填决策路径不一致")
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
