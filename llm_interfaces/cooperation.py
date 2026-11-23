"""合作型 Orchestrator、三位专家、意见池与 Aggregator。"""

from __future__ import annotations

from .agent_runner import (
    AgentInference, complete_ranking, final_text, request_stage, response_json,
)
from .cooperation_telemetry import normalize_telemetry, select_telemetry, semantic_labels_template
from .result_parser import candidate_details


SPECIALISTS = {
    "log": "log_data",
    "time_series": "time_series_data",
    "metadata": "metadata",
}
INCIDENT_DESCRIPTIONS = {
    "BGP邻接状态改变": "BGP 对等会话的邻接状态发生变化，需区分链路、接口、对端设备和协议异常。",
    "接口状态DOWN": "网络设备接口进入 Down 状态，需区分主动关闭、接口故障及链路闪断。",
    "网络设备掉线": "网络设备出现不可达或探测丢包异常，需核查设备、关联服务器和重启或变更线索。",
    "入方向带宽利用率过高": "接口入方向带宽利用率触发告警，需核查流量偏高的幅度、持续时间及后续恢复情况。",
}
ORCHESTRATOR_PROMPT = (
    "你是 Orchestrator。根据 incident_type、incident_type_description 和全部候选根因的名称及简要描述，"
    "从 semantic_labels_template 的遥测全集中选择能区分这些候选的证据字段。"
    "为 log、time_series、metadata 各生成一位面向当前告警的专家。只返回 JSON 对象，"
    '三个键分别为 log、time_series、metadata，每个值形如 '
    '{"persona":"角色及诊断职责","focus":[{"key":"模板中的原始路径","desc":"具体检查什么、如何区分候选"}]}。'
    "focus 只能使用本类别的模板 key，同一 key 至多一次。优先选择有数据的最细字段。"
    "key 中的 /* 可匹配数组各项，例如 /anomaly_kpi/traffic_in/abnormal_timing_segments/*/value。"
    "需要时间戳与样本值相互印证时，同时选择相应字段。"
    "available 为 false 的关键字段可用于指出证据缺口。该类没有相关线索时 focus 可为空。"
)
SPECIALIST_PROMPT = (
    "你是 specialist.persona 指定的专家。按 specialist.focus 核对所选遥测的原始值，"
    "区分已观测证据和缺失数据。"
)
OPINION_PROMPT = (
    '能形成意见时，只输出 JSON 对象 {"reasoning_chain":["证据 → 诊断判断"],"root_cause":"根因名称"}。'
    "简述可核查的证据和诊断依据，并标注缺失证据。候选列表非空时从中选择根因。"
    "有可核查证据但无法确定候选根因时，保留依据并将 root_cause 填为 null。"
    "没有可补充证据时，只输出 NO COMMENT。"
)
REVIEW_PROMPT = (
    "审阅同一轮收到的各专家最新意见及你的原始证据。有新增证据或修正时提交一份完整的更新意见，"
    "其中保留仍然有效的依据。更新意见会替换你此前的意见。"
    "无需补充时只输出 NO COMMENT。"
)
AGGREGATOR_PROMPT = (
    '你是 Aggregator。综合各专家最新意见中的证据与分歧，输出 JSON 对象 '
    '{"ranking":["最可能根因","其他根因"],"reason":"主要依据和不确定性"}。'
    "skipped_specialists 列出缺少可用遥测或未选择相关线索的专家，将其视为证据缺口。"
    "abstained_specialists 列出未形成可用意见的专家。"
    "意见中 root_cause 为 null 表示该专家只提供证据，请结合这些证据判断候选根因。"
    "若提供候选列表，按可能性排列全部候选，每项恰好一次。"
)


class CooperationInference(AgentInference):
    """专家的初评与复评共享意见池，结束后由 Aggregator 汇总。"""

    mode = "cooperation"

    def __init__(
        self, engine, *, invoke_limit=10, max_rounds=None, telemetry_template=None, **kwargs,
    ):
        if type(invoke_limit) is not int or invoke_limit < 5:
            raise ValueError("invoke_limit 必须为至少 5 的整数")
        if max_rounds is not None:
            if type(max_rounds) is not int or max_rounds <= 0:
                raise ValueError("max_rounds 必须为正整数")
            invoke_limit = min(invoke_limit, 5 + 3 * max_rounds)
        self.invoke_limit = invoke_limit
        self.telemetry_template = telemetry_template
        super().__init__(engine, **kwargs)

    @staticmethod
    def _personas(response, template, issues=None):
        """尽量修复 Orchestrator 计划，避免单个 focus 错误中断 case。"""
        issues = issues if issues is not None else []
        try:
            personas = response_json(response)
        except (ValueError, TypeError) as exc:
            issues.append(f"Orchestrator JSON 无法解析，已回退到各类别全部可用遥测: {exc}")
            personas = {}
        if not isinstance(personas, dict):
            issues.append("Orchestrator 输出不是对象，已回退到各类别全部可用遥测")
            personas = {}

        normalized = {}
        for role in SPECIALISTS:
            specialist = personas.get(role)
            if not isinstance(specialist, dict):
                issues.append(f"{role} persona 缺失或格式错误，已使用默认角色和遥测范围")
                specialist = {}
            persona = specialist.get("persona")
            if not isinstance(persona, str) or not persona.strip():
                persona = f"{role} 遥测分析专家，检查相关证据并判断告警根因。"

            raw_focus = specialist.get("focus")
            if not isinstance(raw_focus, list):
                issues.append(f"{role} focus 格式错误，已使用全部可用遥测")
                raw_focus = []
            allowed = {item["key"]: item for item in template[role]}
            selected = set()
            focus = []
            for detail in raw_focus:
                if not isinstance(detail, dict):
                    issues.append(f"{role} 有格式错误的关注点，已忽略")
                    continue
                key = detail.get("key")
                if not isinstance(key, str) or key not in allowed:
                    issues.append(f"{role} 有模板中不存在的 key，已忽略: {key}")
                    continue
                if key in selected:
                    issues.append(f"{role} 有重复 key，已合并: {key}")
                    continue
                desc = detail.get("desc")
                if not isinstance(desc, str) or not desc.strip():
                    desc = allowed[key].get("desc") or f"检查遥测字段 {key}。"
                focus.append({"key": key, "desc": desc.strip()})
                selected.add(key)

            available = [item for item in template[role] if item["available"]]
            if not focus or not any(item["key"] in {entry["key"] for entry in available} for item in focus):
                if available:
                    issues.append(f"{role} 没有可用的有效关注点，已回退到全部可用遥测")
                    focus = [
                        {"key": item["key"], "desc": item["desc"]}
                        for item in available
                    ]
            normalized[role] = {"persona": persona, "focus": focus}
        return normalized

    @staticmethod
    def _opinion(response, candidates, role, round_index):
        text = final_text(response)
        if not text:
            raise ValueError(f"{role} 专家未返回最终回答")
        if text == "NO COMMENT":
            return None
        opinion = response_json(response)
        if not isinstance(opinion, dict):
            raise ValueError(f"{role} 专家的意见必须为 JSON 对象")
        chain = opinion.get("reasoning_chain")
        cause = opinion.get("root_cause")
        if "root_cause" in opinion and cause is None and not chain:
            return None
        if not isinstance(chain, list) or not chain or any(
            not isinstance(item, str) or not item.strip() for item in chain
        ):
            raise ValueError(f"{role} 专家的 reasoning_chain 必须为非空字符串列表")
        cause = cause.strip() if isinstance(cause, str) else None
        if not cause or (candidates and cause not in candidates):
            cause = None
        return {"agent": role, "round": round_index, "reasoning_chain": chain, "root_cause": cause}

    def _case_stages(self, case, result):
        labels = normalize_telemetry(case.semantic_labels)
        details = candidate_details(case.root_cause_candidates)
        candidates = [item["category"] for item in details]
        incident_description = case.incident_type_description or INCIDENT_DESCRIPTIONS.get(
            case.alarm_type, f"{case.alarm_type} 告警，结合候选根因及其描述确定诊断范围。",
        )
        context = {
            "alarm_type": case.alarm_type, "alarm_time": case.alarm_time,
            "incident_type_description": incident_description,
            "root_cause_candidates": candidates,
        }
        result.update(
            opinion_pool=[], invoke_limit=self.invoke_limit, agent_invocations=0,
            round_count=0, early_stopped=False, rounds=[],
            skipped_specialists={}, specialist_issues=[], orchestrator_issues=[],
        )

        def invoke(name, requests, *, final=False):
            result["agent_invocations"] += len(requests)
            return request_stage(result, name, requests, final=final)

        template = semantic_labels_template(labels, self.telemetry_template)
        plan, = yield from invoke("orchestrator", [("orchestrator", ORCHESTRATOR_PROMPT, {
            "incident_type": case.alarm_type,
            "incident_type_description": incident_description,
            "root_cause_candidates": details,
            "semantic_labels_template": template,
        })])
        personas = self._personas(plan, template, result["orchestrator_issues"])
        result["personas"] = personas
        specialist_data = {}
        for role, group in SPECIALISTS.items():
            available = {item["key"] for item in template[role] if item["available"]}
            focus = personas[role]["focus"]
            missing = [detail["key"] for detail in focus if detail["key"] not in available]
            if len(missing) == len(focus):
                result["skipped_specialists"][role] = (
                    "所选遥测没有可用数据" if focus else "未选择相关遥测"
                )
                continue
            specialist_data[role] = {
                **context, "specialist": personas[role], "missing_focus_keys": missing,
                group: select_telemetry(labels, focus),
            }

        roles = list(specialist_data)
        latest = {}

        def collect_opinions(answers, active, round_index):
            commented, no_comment = [], []
            for role, answer in zip(active, answers):
                try:
                    opinion = self._opinion(answer, candidates, role, round_index)
                except ValueError as exc:
                    result["specialist_issues"].append({"agent": role, "round": round_index, "error": str(exc)})
                    opinion = None
                if opinion is None:
                    no_comment.append(role)
                else:
                    if opinion["root_cause"] is None:
                        result["specialist_issues"].append({
                            "agent": role, "round": round_index,
                            "error": "专家未给出可匹配的根因，已保留证据链",
                        })
                    latest[role] = opinion
                    commented.append(role)
            result["opinion_pool"] = [latest[role] for role in roles if role in latest]
            result["rounds"].append({"round": round_index, "commented": commented, "no_comment": no_comment})
            result["round_count"] = round_index + 1
            return not commented

        if roles:
            initial = yield from invoke("round0", [
                (f"round0/{role}", SPECIALIST_PROMPT + OPINION_PROMPT, specialist_data[role])
                for role in roles
            ])
            if collect_opinions(initial, roles, 0):
                result["early_stopped"] = True

        if not latest:
            result["stop_reason"] = "insufficient_evidence"
        else:
            while result["agent_invocations"] < self.invoke_limit - 1:
                remaining = self.invoke_limit - result["agent_invocations"] - 1
                active = roles[:remaining]
                round_index = result["round_count"]
                snapshot = result["opinion_pool"]
                answers = yield from invoke(f"round{round_index}", [
                    (
                        f"round{round_index}/{role}",
                        SPECIALIST_PROMPT + REVIEW_PROMPT + OPINION_PROMPT,
                        {**specialist_data[role], "opinion_pool": snapshot},
                    )
                    for role in active
                ])
                if collect_opinions(answers, active, round_index) and len(active) == len(roles):
                    result["early_stopped"] = True
                    break

            result["stop_reason"] = "all_no_comment" if result["early_stopped"] else "invoke_limit"
        answer, = yield from invoke("aggregator", [
            ("aggregator", AGGREGATOR_PROMPT, {
                **context, "opinion_pool": result["opinion_pool"],
                "skipped_specialists": result["skipped_specialists"],
                "abstained_specialists": [role for role in roles if role not in latest],
            }),
        ], final=True)
        result["final_result"] = answer
        conclusion = response_json(answer)
        if isinstance(conclusion, dict):
            ranking = conclusion.get("ranking")
            reason = conclusion.get("reason")
        else:
            ranking, reason = conclusion, None
        ranking = complete_ranking(ranking, candidates, result)
        result["conclusion"] = {"ranking": ranking, "reason": reason if isinstance(reason, str) else ""}
        return ranking
