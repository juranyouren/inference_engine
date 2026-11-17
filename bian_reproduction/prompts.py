# -*- coding: utf-8 -*-

ALERT_SUMMARY_PROMPT = '''
你是生产级网络运维中的监控告警摘要专家。

请把输入的半结构化告警数据压缩成后续异常分析可以直接使用的事实摘要：
1. 删除空字段、无关字段和重复信息，但不能丢失根因证据；
2. 提取日志模板及次数、KPI 异常时间段、设备/接口状态、BGP 状态和命令回显；
3. 区分本端与对端；
4. 不要直接决定最终根因，也不要编造输入中不存在的事件。

告警场景：{alarm_type}
告警时间：{alarm_time}
语义标签：{semantic_labels}
候选根因：{root_cause_candidates}

只输出 JSON 对象，不要输出 Markdown 代码围栏：
{{
  "incident_overview": "...",
  "device_context": {{"local": {{}}, "remote": {{}}}},
  "key_alerts": [{{"source": "...", "evidence": "...", "count": 0}}],
  "kpi_anomalies": [{{"metric": "...", "pattern": "...", "start_time": null, "end_time": null}}],
  "state_evidence": ["..."],
  "candidate_relevant_evidence": {{"候选根因 category": ["..."]}}
}}
'''.strip()


ANOMALY_ANALYSIS_PROMPT = '''
你是网络运维 SOP 执行专家，负责对当前告警案例进行异常分析。

请完成：
1. 按 SOP 顺序检查条件，并记录每一步使用的证据；
2. 判断本端、对端、接口、BGP、日志和 KPI 是否存在异常及严重程度；
3. 对每个候选根因给出支持证据、反驳证据和置信度；
4. 不要输出候选列表之外的根因。

告警场景：{alarm_type}
告警时间：{alarm_time}
告警摘要：{alert_summary}
原始语义标签：{semantic_labels}
SOP：{sop}
候选根因：{root_cause_candidates}

只输出 JSON 对象，不要输出 Markdown 代码围栏：
{{
  "sop_trace": [{{"step": "...", "decision": "...", "evidence": ["..."]}}],
  "anomaly_findings": [{{"scope": "local|remote|interface|bgp|kpi|log", "finding": "...", "severity": "low|medium|high", "evidence": ["..."]}}],
  "candidate_assessments": [{{"category": "候选根因 category", "supporting_evidence": ["..."], "contradicting_evidence": ["..."], "confidence": 0.0}}],
  "uncertainties": ["..."]
}}
'''.strip()


JOINT_SCORING_PROMPT = '''
你是网络事件联合根因分析专家。

请根据告警摘要、异常分析、SOP 轨迹和原始证据，对候选根因联合评分：
1. 只能使用候选列表中的 category；
2. 必须输出全部候选根因；
3. 每个分数在 0 到 1 之间，所有分数之和必须为 1；
4. 按分数从高到低排序，并提供简短可审计证据。

告警场景：{alarm_type}
告警时间：{alarm_time}
候选根因：{root_cause_candidates}
告警摘要：{alert_summary}
异常分析：{anomaly_analysis}

只输出 JSON 对象，不要输出 Markdown 代码围栏：
{{
  "ranked_candidates": [
    {{"category": "候选根因 category", "score": 0.0, "evidence": ["..."]}}
  ],
  "final_explanation": "..."
}}
'''.strip()

