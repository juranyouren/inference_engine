"""合作型 Orchestrator、三位专家、意见池与 Aggregator。"""

from __future__ import annotations

from .agent_runner import (
    AgentInference, case_evidence, final_text, response_json, validate_ranking,
)
from .result_parser import candidate_categories


SPECIALISTS = {
    "log": ("log_data", "日志专家。核对告警日志、事件和命令日志的时序关系。"),
    "time_series": ("time_series_data", "时序专家。核对 KPI、异常流量、时间窗口及阈值。"),
    "metadata": ("metadata", "元数据专家。核对设备、接口、拓扑和命令回显。"),
}
ORCHESTRATOR_PROMPT = (
    "你是 Orchestrator。把 case.semantic_labels 的每个顶层字段分配给一位专家。"
    "只返回 JSON 对象，键为 log_data、time_series_data、metadata，值为字段名列表。"
    "每个字段恰好分配一次，允许空列表。anomaly_logs 给日志专家，anomaly_kpi 和 "
    "anomaly_netstreams 给时序专家，other_info 给元数据专家。"
)
OPINION_PROMPT = (
    '只输出 JSON 对象 {"reasoning_chain":["证据 → 诊断判断"],"root_cause":"根因名称"}。'
    "简述可核查的证据和诊断依据，并标注缺失证据。候选列表非空时从中选择根因。"
)
REVIEW_PROMPT = (
    "审阅同一轮收到的意见池及你的原始证据。有新增证据、修正或不同结论时提交新意见。"
    "无需补充时只输出 NO COMMENT。"
)
AGGREGATOR_PROMPT = (
    '你是 Aggregator。综合意见池中的证据与分歧，输出 JSON 对象 '
    '{"ranking":["最可能根因","其他根因"],"reason":"主要依据和不确定性"}。'
    "若提供候选列表，按可能性排列全部候选，每项恰好一次。"
)


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
    def _route(response, labels):
        routes = response_json(response)
        expected = {group for group, _prompt in SPECIALISTS.values()}
        if not isinstance(routes, dict) or routes.keys() != expected:
            raise ValueError("Orchestrator 必须返回三组字段")
        fields = []
        for keys in routes.values():
            if not isinstance(keys, list) or any(not isinstance(key, str) for key in keys):
                raise ValueError("Orchestrator 分组必须是字段名列表")
            fields.extend(keys)
        if sorted(fields) != sorted(labels):
            raise ValueError("Orchestrator 必须把每个实际字段恰好分配一次")
        for field, group in (
            ("anomaly_logs", "log_data"), ("anomaly_kpi", "time_series_data"),
            ("anomaly_netstreams", "time_series_data"), ("other_info", "metadata"),
        ):
            if field in labels and field not in routes[group]:
                raise ValueError(f"Orchestrator 将 {field} 分配到了错误的专家")
        return routes

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
        evidence = case_evidence(case)
        candidates = candidate_categories(case.root_cause_candidates)
        context = {
            "alarm_type": case.alarm_type, "alarm_time": case.alarm_time,
            "root_cause_candidates": candidates,
        }
        pool = []
        result.update(
            opinion_pool=pool, invoke_limit=self.invoke_limit, agent_invocations=0,
            round_count=0, early_stopped=False, rounds=[],
        )

        def invoke(requests):
            result["agent_invocations"] += len(requests)
            return self.generator.generate(requests, output_dir)

        routed, = invoke([("orchestrator", ORCHESTRATOR_PROMPT, {"case": evidence})])
        labels = evidence["semantic_labels"]
        routes = self._route(routed, labels)
        result["routes"] = routes
        specialist_data = {
            role: {**context, group: {key: labels[key] for key in routes[group]}}
            for role, (group, _prompt) in SPECIALISTS.items()
        }

        roles = list(SPECIALISTS)
        initial = invoke([
            (f"round0/{role}", SPECIALISTS[role][1] + OPINION_PROMPT, specialist_data[role])
            for role in roles
        ])
        pool.extend(self._opinion(answer, candidates, role, 0) for role, answer in zip(roles, initial))
        result["rounds"].append({"round": 0, "commented": roles, "no_comment": []})
        result["round_count"] = 1

        while result["agent_invocations"] < self.invoke_limit - 1:
            remaining = self.invoke_limit - result["agent_invocations"] - 1
            active = roles[:remaining]
            round_index = result["round_count"]
            snapshot = list(pool)
            answers = invoke([
                (
                    f"round{round_index}/{role}",
                    SPECIALISTS[role][1] + REVIEW_PROMPT + OPINION_PROMPT,
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
            pool.extend(new_opinions)
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
            ("aggregator", AGGREGATOR_PROMPT, {**context, "opinion_pool": pool}),
        ])
        result["final_result"] = answer
        conclusion = response_json(answer)
        if not isinstance(conclusion, dict) or not isinstance(conclusion.get("reason"), str):
            raise ValueError("Aggregator 必须返回 ranking 和 reason")
        result["conclusion"] = conclusion
        return validate_ranking(conclusion.get("ranking"), candidates)
