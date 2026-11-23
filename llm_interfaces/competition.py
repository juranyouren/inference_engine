"""竞争型按 SOP 候选根因独立诊断，再由 Verifier 重排。"""

from __future__ import annotations

from .agent_runner import AgentInference, case_evidence, complete_ranking, request_stage, response_json
from .competition_templates import load_competition_template
from .invoke import ContextLengthError
from .result_parser import candidate_details


VERIFIER_PROMPT = (
    "你是 Verifier。结合 case 中的原始证据、SOP 的决策条件以及每条候选诊断的"
    "节点检查状态、证据引用和根因分支置信度，独立复核每个根因。"
    "对照所有根因的详细描述和 diagnostic clues，结合原始证据比较各根因。"
    "若只提供部分分支结论，仍须依据原始证据评估全部候选。"
    "候选 Agent 未报告的节点标为未知，请结合原始证据自行复核。"
    "根因分支置信度为 null 表示候选 Agent 未给出有效评分。"
    "只输出候选根因名称的 JSON 数组，按可能性从高到低排列，全部候选恰好出现一次。"
)


def _validate_node(judgment, template, name):
    if not isinstance(judgment, dict) or judgment.keys() != template.keys():
        raise ValueError(f"Agent 必须完整填写 {name} 对应的模板")
    outcome = judgment.get("condition_met")
    if outcome is not None and type(outcome) is not bool:
        raise ValueError(f"{name} 的 condition_met 必须为 true、false 或 null")
    confidence = judgment["confidence"]
    if outcome is None:
        valid_confidence = confidence is None
    else:
        valid_confidence = type(confidence) in (int, float) and 0 <= confidence <= 1
    if not valid_confidence:
        raise ValueError(f"{name} 的置信度无效")
    if not isinstance(judgment["reason"], str) or not judgment["reason"].strip():
        raise ValueError(f"{name} 缺少判断依据")


def _normalize_conclusion(conclusion):
    if not isinstance(conclusion, dict):
        conclusion = {}
    confidence = conclusion.get("confidence")
    if isinstance(confidence, str):
        try:
            confidence = float(confidence)
        except ValueError:
            confidence = None
    reason = conclusion.get("reason")
    reason = reason.strip() if isinstance(reason, str) else ""
    if type(confidence) not in (int, float) or not 0 <= confidence <= 1:
        confidence = None
        reason = "候选 Agent 未报告有效的分支置信度，等待 Verifier 复核。" + reason
    return {"confidence": confidence, "reason": reason or "候选 Agent 未报告结论依据。"}


def _normalize_diagnosis(response, prompt, templates):
    try:
        diagnosis = response_json(response)
    except ValueError:
        diagnosis = {}
    if not isinstance(diagnosis, dict):
        diagnosis = {}
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
            _validate_node(node, result_template, node_id)
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
    return {"root_cause": prompt["root_cause"], "paths": prompt["paths"],
            "decisions": decisions, "conclusion": _normalize_conclusion(diagnosis.get("conclusion"))}


def _top_conclusions(diagnoses):
    def score(item):
        confidence = item["conclusion"]["confidence"]
        return -1 if confidence is None else confidence

    return [{"root_cause": item["root_cause"], "conclusion": item["conclusion"]}
            for item in sorted(diagnoses, key=score, reverse=True)[:5]]


class CompetitionInference(AgentInference):
    """每个候选根因一个独立 Agent，Verifier 查看所有路径。"""

    mode = "competition"

    def _case_stages(self, case, result):
        template = load_competition_template(case.alarm_type, case.sop)
        all_decisions = template["decision_templates"]
        by_name = {entry["root_cause"]: entry for entry in template["root_cause_prompts"].values()}
        details = candidate_details(case.root_cause_candidates) or candidate_details(list(by_name))
        descriptions = {item['category']: item['description'] for item in details}
        candidates = list(descriptions)
        missing = [name for name in candidates if name not in by_name]
        if missing:
            raise ValueError(f"SOP 缺少候选根因路径: {', '.join(missing)}")

        evidence = case_evidence(case)
        if case.diagnostic_clue is not None:
            evidence['diagnostic_clue'] = case.diagnostic_clue
        requests = []
        jobs = []
        for index, name in enumerate(candidates, start=1):
            entry = by_name[name]
            node_ids = dict.fromkeys(
                step["decision_id"] for path in entry["paths"] for step in path
            )
            nodes = {node_id: all_decisions[node_id] for node_id in node_ids}
            payload = {
                "case": evidence, "root_cause": name, "description": descriptions[name], "paths": entry["paths"],
                "decision_templates": nodes, "conclusion_template": entry["conclusion_template"],
            }
            requests.append((f"reasoners/candidate_{index:02d}", entry["prompt"], payload))
            jobs.append((entry, nodes))

        answers = yield from request_stage(result, "reasoners", requests)
        diagnoses = []
        result["diagnoses"] = diagnoses
        for answer, (entry, nodes) in zip(answers, jobs):
            diagnoses.append(_normalize_diagnosis(answer, entry, nodes))
        used_nodes = dict.fromkeys(node_id for _entry, nodes in jobs for node_id in nodes)
        verifier_payload = {
            "case": evidence,
            "root_cause_candidates": candidates,
            "candidate_descriptions": descriptions,
            "decision_conditions": {
                node_id: all_decisions[node_id]["condition"] for node_id in used_nodes
            },
            "diagnoses": diagnoses,
        }
        candidate_clues = {}
        for item in case.root_cause_candidates:
            if isinstance(item, dict):
                for field in ("diagnostic_clue", "diagnostic_clues"):
                    if item.get(field) is not None:
                        candidate_clues.setdefault(str(item.get("category", "")).strip(), {})[field] = item[field]
        if candidate_clues:
            verifier_payload["candidate_diagnostic_clues"] = candidate_clues
        compact = self.generator.exceeds_input_budget(VERIFIER_PROMPT, verifier_payload)
        if not compact:
            result["verifier_input_mode"] = "all_branches"
            try:
                answer, = yield from request_stage(
                    result, "verifier", [("verifier", VERIFIER_PROMPT, verifier_payload)], final=True,
                )
            except ContextLengthError:
                compact = True
        if compact:
            conclusions = _top_conclusions(diagnoses)
            result["verifier_input_mode"] = "top5_conclusions"
            result["verifier_branches"] = [item["root_cause"] for item in conclusions]
            verifier_payload["diagnoses"] = conclusions
            # 多条 SOP 路径可引用同一个节点，紧凑输入只保留一份相同的证据。
            clues_by_path = {}
            for diagnosis in diagnoses:
                for node in diagnosis["decisions"].values():
                    for clue in node["diagnostic_clues"]:
                        values = clues_by_path.setdefault(clue["path"], [])
                        if clue["value"] not in values:
                            values.append(clue["value"])
            verifier_payload["branch_diagnostic_clues"] = [
                {"path": path, "value": value}
                for path, values in clues_by_path.items() for value in values
            ]
            answer, = yield from request_stage(
                result, "verifier", [("verifier", VERIFIER_PROMPT, verifier_payload)], final=True,
            )
        result["final_result"] = answer
        ranking = response_json(answer)
        if isinstance(ranking, dict):
            ranking = ranking.get("ranking")
        return complete_ranking(ranking, candidates, result)
