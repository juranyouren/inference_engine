"""合作型 Orchestrator、三位专家、意见池与 Aggregator。"""

from __future__ import annotations

from .agent_runner import (
    AgentInference, final_text, response_json, validate_ranking,
)
from .cooperation_telemetry import select_telemetry, semantic_labels_template
from .result_parser import candidate_categories


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
    '只输出 JSON 对象 {"reasoning_chain":["证据 → 诊断判断"],"root_cause":"根因名称"}。'
    "简述可核查的证据和诊断依据，并标注缺失证据。候选列表非空时从中选择根因。"
)
REVIEW_PROMPT = (
    "审阅同一轮收到的各专家最新意见及你的原始证据。有新增证据或修正时提交一份完整的更新意见，"
    "其中保留仍然有效的依据。更新意见会替换你此前的意见。"
    "无需补充时只输出 NO COMMENT。"
)
AGGREGATOR_PROMPT = (
    '你是 Aggregator。综合各专家最新意见中的证据与分歧，输出 JSON 对象 '
    '{"ranking":["最可能根因","其他根因"],"reason":"主要依据和不确定性"}。'
    "若提供候选列表，按可能性排列全部候选，每项恰好一次。"
)


def _candidate_details(candidates):
    descriptions = {}
    for item in candidates or []:
        if not isinstance(item, dict):
            continue
        names = candidate_categories([item])
        if not names or names[0] in descriptions:
            continue
        for field in ("description", "desc", "reason"):
            value = item.get(field)
            if isinstance(value, str) and value.strip():
                descriptions[names[0]] = value.strip()
                break
    return [
        {"category": name, "description": descriptions.get(name, "")}
        for name in candidate_categories(candidates)
    ]


class CooperationInference(AgentInference):
    """专家的初评与复评共享意见池，结束后由 Aggregator 汇总。"""

    mode = "cooperation"

    def __init__(
        self, engine, *, invoke_limit=10, max_rounds=None, **kwargs,
    ):
        if type(invoke_limit) is not int or invoke_limit < 5:
            raise ValueError("invoke_limit 必须为至少 5 的整数")
        if max_rounds is not None:
            if type(max_rounds) is not int or max_rounds <= 0:
                raise ValueError("max_rounds 必须为正整数")
            invoke_limit = min(invoke_limit, 5 + 3 * max_rounds)
        self.invoke_limit = invoke_limit
        super().__init__(engine, **kwargs)

    @staticmethod
    def _personas(response, template):
        personas = response_json(response)
        if not isinstance(personas, dict) or personas.keys() != SPECIALISTS.keys():
            raise ValueError("Orchestrator 必须为 log、time_series、metadata 各生成一个 persona")
        for role, specialist in personas.items():
            if not isinstance(specialist, dict) or specialist.keys() != {"persona", "focus"}:
                raise ValueError(f"{role} 专家必须包含 persona 和 focus")
            persona, focus = specialist["persona"], specialist["focus"]
            if not isinstance(persona, str) or not persona.strip() or not isinstance(focus, list):
                raise ValueError(f"{role} 专家需要角色说明和关注点列表")
            allowed = {item["key"] for item in template[role]}
            selected = set()
            for detail in focus:
                if not isinstance(detail, dict) or detail.keys() != {"key", "desc"}:
                    raise ValueError(f"{role} 的关注点必须包含 key 和 desc")
                key, desc = detail["key"], detail["desc"]
                if not isinstance(key, str) or key not in allowed or key in selected:
                    raise ValueError(f"{role} 的关注点必须使用该类别中互不重复的模板 key")
                if not isinstance(desc, str) or not desc.strip():
                    raise ValueError(f"{role} 的关注点必须有具体检查说明")
                selected.add(key)
        return personas

    @staticmethod
    def _opinion(response, candidates, role, round_index):
        opinion = response_json(response)
        if not isinstance(opinion, dict):
            raise ValueError(f"{role} 专家的意见必须为 JSON 对象")
        chain = opinion.get("reasoning_chain")
        cause = opinion.get("root_cause")
        if not isinstance(chain, list) or not chain or any(
            not isinstance(item, str) or not item.strip() for item in chain
        ):
            raise ValueError(f"{role} 专家的 reasoning_chain 必须为非空字符串列表")
        if not isinstance(cause, str) or not cause.strip() or (candidates and cause not in candidates):
            raise ValueError(f"{role} 专家返回了无效的根因")
        return {"agent": role, "round": round_index, "reasoning_chain": chain, "root_cause": cause}

    def _infer_case(self, case, output_dir, result):
        labels = case.semantic_labels
        candidate_details = _candidate_details(case.root_cause_candidates)
        candidates = [item["category"] for item in candidate_details]
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
        )

        def invoke(requests):
            result["agent_invocations"] += len(requests)
            return self.generator.generate(requests, output_dir)

        template = semantic_labels_template(labels)
        plan, = invoke([("orchestrator", ORCHESTRATOR_PROMPT, {
            "incident_type": case.alarm_type,
            "incident_type_description": incident_description,
            "root_cause_candidates": candidate_details,
            "semantic_labels_template": template,
        })])
        personas = self._personas(plan, template)
        result["personas"] = personas
        specialist_data = {
            role: {
                **context, "specialist": personas[role],
                group: select_telemetry(labels, personas[role]["focus"]),
            }
            for role, group in SPECIALISTS.items()
        }

        roles = list(SPECIALISTS)
        initial = invoke([
            (f"round0/{role}", SPECIALIST_PROMPT + OPINION_PROMPT, specialist_data[role])
            for role in roles
        ])
        latest = {
            role: self._opinion(answer, candidates, role, 0) for role, answer in zip(roles, initial)
        }
        result["opinion_pool"] = list(latest.values())
        result["rounds"].append({"round": 0, "commented": roles, "no_comment": []})
        result["round_count"] = 1

        while result["agent_invocations"] < self.invoke_limit - 1:
            remaining = self.invoke_limit - result["agent_invocations"] - 1
            active = roles[:remaining]
            round_index = result["round_count"]
            snapshot = result["opinion_pool"]
            answers = invoke([
                (
                    f"round{round_index}/{role}",
                    SPECIALIST_PROMPT + REVIEW_PROMPT + OPINION_PROMPT,
                    {**specialist_data[role], "opinion_pool": snapshot},
                )
                for role in active
            ])
            no_comment = []
            new_opinions = []
            for role, answer in zip(active, answers):
                if final_text(answer) == "NO COMMENT":
                    no_comment.append(role)
                else:
                    new_opinions.append(self._opinion(answer, candidates, role, round_index))
            latest.update((opinion["agent"], opinion) for opinion in new_opinions)
            result["opinion_pool"] = list(latest.values())
            result["rounds"].append({
                "round": round_index, "commented": [item["agent"] for item in new_opinions],
                "no_comment": no_comment,
            })
            result["round_count"] += 1
            if len(no_comment) == len(roles):
                result["early_stopped"] = True
                break

        result["stop_reason"] = "all_no_comment" if result["early_stopped"] else "invoke_limit"
        answer, = invoke([
            ("aggregator", AGGREGATOR_PROMPT, {**context, "opinion_pool": result["opinion_pool"]}),
        ])
        result["final_result"] = answer
        conclusion = response_json(answer)
        if not isinstance(conclusion, dict) or not isinstance(conclusion.get("reason"), str):
            raise ValueError("Aggregator 必须返回 ranking 和 reason")
        result["conclusion"] = conclusion
        return validate_ranking(conclusion.get("ranking"), candidates)
