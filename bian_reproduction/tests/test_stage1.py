# -*- coding: utf-8 -*-

import json
from pathlib import Path
from typing import List, Sequence

from bian_reproduction.parsing import normalize_joint_scoring
from bian_reproduction.stage1 import BianStage1Runner


class FakeRunner(BianStage1Runner):
    """Use deterministic responses to verify orchestration without loading vLLM."""

    def _invoke(self, prompts: Sequence[str]) -> List[str]:
        responses = []
        for prompt in prompts:
            if "监控告警摘要专家" in prompt:
                responses.append(json.dumps({
                    "incident_overview": "接口发生一次闪断，当前已恢复",
                    "key_alerts": [
                        {"source": "syslog", "evidence": "linkDown_active=1", "count": 1}
                    ],
                }, ensure_ascii=False))
            elif "SOP 执行专家" in prompt:
                responses.append(json.dumps({
                    "sop_trace": [
                        {"step": "检查接口与 BGP", "decision": "均已恢复", "evidence": ["UP", "Established"]}
                    ],
                    "candidate_assessments": [],
                }, ensure_ascii=False))
            else:
                responses.append(json.dumps({
                    "ranked_candidates": [
                        {"category": "根因B", "score": 0.8, "evidence": ["接口已恢复"]},
                        {"category": "根因A", "score": 0.2, "evidence": []},
                    ],
                    "final_explanation": "根因B证据更充分",
                }, ensure_ascii=False))
        return responses


def test_joint_scoring_normalizes_and_sorts_scores() -> None:
    ranked, strategy = normalize_joint_scoring(
        {
            "ranked_candidates": [
                {"category": "根因A", "score": 2},
                {"category": "根因B", "score": 8},
            ]
        },
        ["根因A", "根因B"],
    )

    assert strategy == "scored_ranking"
    assert [item["category"] for item in ranked] == ["根因B", "根因A"]
    assert sum(item["score"] for item in ranked) == 1.0


def test_stage1_flow_writes_ranked_prediction(tmp_path: Path) -> None:
    alarm_type = "BGP邻接状态改变"
    data_dir = tmp_path / "input" / alarm_type
    data_dir.mkdir(parents=True)
    label = {
        "semantic_labels": {
            "anomaly_logs": {"syslog_brief": {"linkDown_active(l)": 1}},
            "anomaly_kpi": {},
            "other_info": {
                "command_echo_info": {
                    "BGP_protocol_status": "Established",
                    "local_interface_status": "UP",
                }
            },
        },
        "alarm_time": 1760323581686,
        "sop": ["检查接口和 BGP 状态"],
        "root_cause_candidates": [
            {"category": "根因A", "description": "未恢复"},
            {"category": "根因B", "description": "轻微闪断"},
        ],
        "root_cause": {"category": "根因B", "description": "轻微闪断"},
    }
    (data_dir / f"{alarm_type}_0_label_sample.json").write_text(
        json.dumps(label, ensure_ascii=False),
        encoding="utf-8",
    )

    payload = FakeRunner(model_path="unused", batch_size=4).run(
        data_dir=str(data_dir),
        alarm_type=alarm_type,
        indices=[0],
        output_dir=str(tmp_path / "output"),
    )

    result = payload["results"][0]
    assert payload["summary"]["processed_indices"] == [0]
    assert result["pred_top1_rc"] == "根因B"
    assert result["rank"] == 1
    assert result["is_correct"] is True

    final_dir = Path(payload["meta"]["output_dir"])
    assert (final_dir / "predictions.json").is_file()
    assert (final_dir / "predictions.jsonl").is_file()
    assert (final_dir / "predictions.csv").is_file()
    assert (final_dir / "0" / "alert_summary" / "prompt.txt").is_file()
    assert (final_dir / "0" / "anomaly_analysis" / "parsed.json").is_file()
    assert (final_dir / "0" / "joint_scoring" / "raw_responses.txt").is_file()
