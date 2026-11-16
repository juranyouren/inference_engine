import csv
import importlib
import json
import os
import sys
import tempfile
import types
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from inference.common import get_last_processed_index, load_infer_state
from inference.llm import llm_infer
from llm_interfaces.competition import CompetitionInference
from llm_interfaces.cooperation import CooperationInference
from llm_interfaces.evaluate import run as evaluate
from llm_interfaces.merge_agent_evaluation import merge_shards


class FakeTokenizer:
    @staticmethod
    def encode(text, add_special_tokens=False):
        return [ord(char) for char in text]

    @staticmethod
    def decode(token_ids, skip_special_tokens=True):
        return "".join(chr(char) for char in token_ids)


def default_answer(instruction, payload):
    if "decision_templates" in payload:
        return {
            "decisions": {
                node_id: {**node["result_template"], "reason": "证据不足"}
                for node_id, node in payload["decision_templates"].items()
            },
            "conclusion": {**payload["conclusion_template"], "confidence": 0.2, "reason": "证据不足"},
        }
    if "diagnoses" in payload:
        return payload["root_cause_candidates"]
    if instruction.startswith("你是 Orchestrator"):
        return {
            role: {
                "persona": f"{role} 诊断专家",
                "focus": [
                    {"key": field["key"], "desc": "核对告警窗口内的证据"}
                    for field in fields if field["available"] and field["key"].count("/") == 1
                ],
            }
            for role, fields in payload["semantic_labels_template"].items()
        }
    if instruction.startswith("你是 Aggregator"):
        return {"ranking": payload["root_cause_candidates"], "reason": "综合意见池"}
    if "opinion_pool" in payload:
        return "NO COMMENT"
    return {
        "reasoning_chain": ["本阶段证据"],
        "root_cause": payload["root_cause_candidates"][0],
    }


class FakeLlm:
    def __init__(self, answer=default_answer):
        self.llm_engine = SimpleNamespace(
            model_config=SimpleNamespace(max_model_len=100000)
        )
        self.answer = answer
        self.requests = []

    @staticmethod
    def get_tokenizer():
        return FakeTokenizer()

    def chat(self, prompts, sampling_params, use_tqdm=False):
        responses = []
        for chat in prompts:
            instruction, raw_payload = chat[0]["content"].split("\n\n输入 JSON\n", 1)
            payload = json.loads(raw_payload)
            self.requests.append((instruction, payload))
            answer = self.answer(instruction, payload)
            text = answer if isinstance(answer, str) else json.dumps(answer, ensure_ascii=False)
            responses.append(SimpleNamespace(outputs=[SimpleNamespace(text=text)]))
        return responses


def fake_sampling_params(**kwargs):
    return SimpleNamespace(n=1, **kwargs)


def fake_engine(answer=default_answer):
    return SimpleNamespace(llm=FakeLlm(answer))


def competition_label(categories=("链路故障", "配置错误")):
    terminal_start = len(categories)
    rules = []
    for step in range(1, terminal_start):
        yes = terminal_start + step - 1
        no = yes + 1 if step == terminal_start - 1 else step + 1
        rules.append(
            f"Procedure {step}:::Decision:::检查证据 {step}."
            f":::Yes::Procedure {yes}::No::Procedure {no}"
        )
    for index, category in enumerate(categories, start=terminal_start):
        rules.append(f'Procedure {index}:::Terminal:::诊断根因为"{category}".:::')
    return {
        "case_id": "case-a",
        "alarm_time": 123,
        "semantic_labels": {
            "sop": "\n\n".join(rules),
            "anomaly_logs": {"message": "peer down"},
            "anomaly_kpi": {"value": 100},
            "other_info": {"local_device": "router"},
        },
        "root_cause_candidates": [{"category": category} for category in categories],
        "root_cause": {"category": categories[-1]},
        "ttm": 16.0,
    }


def cooperation_label():
    label = competition_label()
    label["semantic_labels"].pop("sop")
    return label


class LlmInterfaceTests(unittest.TestCase):
    def test_two_inference_classes_share_the_same_engine(self):
        engine = fake_engine()
        competition = CompetitionInference(engine, sampling_params_factory=fake_sampling_params)
        cooperation = CooperationInference(engine, sampling_params_factory=fake_sampling_params)
        self.assertIs(competition.engine, engine)
        self.assertIs(cooperation.engine, engine)
        self.assertIs(competition.llm, engine.llm)
        self.assertIs(cooperation.llm, engine.llm)

    def test_llm_engine_get_instance_loads_once(self):
        calls = []

        def fake_llm(**kwargs):
            calls.append(kwargs)
            return "loaded-llm"

        fake_vllm = types.ModuleType("vllm")
        fake_vllm.LLM = fake_llm
        module_name = "llm_interfaces.llm_engine"
        previous_module = sys.modules.pop(module_name, None)
        try:
            with patch.dict(sys.modules, {"vllm": fake_vllm}):
                module = importlib.import_module(module_name)
                with patch.dict(os.environ, {"ASCEND_RT_VISIBLE_DEVICES": "0,1"}):
                    first = module.LLMEngine.get_instance(model_path="fake-model")
                    second = module.LLMEngine.get_instance(model_path="unused")
            self.assertIs(first, second)
            self.assertEqual(len(calls), 1)
            self.assertEqual(calls[0]["tensor_parallel_size"], 2)
        finally:
            sys.modules.pop(module_name, None)
            if previous_module is not None:
                sys.modules[module_name] = previous_module

    def test_competition_sends_all_candidate_paths_and_original_evidence(self):
        engine = fake_engine()
        inference = CompetitionInference(engine, sampling_params_factory=fake_sampling_params)
        label = competition_label()
        with tempfile.TemporaryDirectory() as root:
            row, = inference.infer_results([label], "场景", root)
            output = Path(row["output_dir"])
            self.assertTrue((output / "reasoners/candidate_01/prompt.txt").is_file())
            self.assertTrue((output / "verifier/raw_responses.txt").is_file())

        self.assertEqual(row["status"], "success")
        self.assertEqual(len(engine.llm.requests), 3)
        first, second, verifier = [payload for _prompt, payload in engine.llm.requests]
        self.assertTrue(first["paths"][0][0]["required_condition_met"])
        self.assertFalse(second["paths"][0][0]["required_condition_met"])
        self.assertEqual(verifier["diagnoses"], row["diagnoses"])
        self.assertTrue(all(item["conclusion"]["confidence"] == 0.2 for item in verifier["diagnoses"]))
        self.assertTrue(all("supported" not in item["conclusion"] for item in verifier["diagnoses"]))
        self.assertEqual(verifier["case"]["semantic_labels"]["anomaly_logs"], {"message": "peer down"})
        self.assertNotIn("root_cause", verifier["case"])
        self.assertNotIn("sop", verifier["case"]["semantic_labels"])

    def test_cooperation_routes_evidence_and_stops_when_all_decline(self):
        engine = fake_engine()
        inference = CooperationInference(engine, sampling_params_factory=fake_sampling_params)
        label = cooperation_label()
        label["sop"] = "顶层 SOP 不参与运行时路由"
        with tempfile.TemporaryDirectory() as root:
            row, = inference.infer_results([label], "场景", root)
        self.assertEqual(row["status"], "success")
        self.assertEqual(row["agent_invocations"], 8)
        self.assertEqual(row["stop_reason"], "all_no_comment")
        self.assertEqual(len(row["opinion_pool"]), 3)
        requests = [payload for _prompt, payload in engine.llm.requests]
        self.assertEqual(requests[1]["log_data"], {"anomaly_logs": {"message": "peer down"}})
        self.assertEqual(requests[2]["time_series_data"], {"anomaly_kpi": {"value": 100}})
        self.assertEqual(requests[3]["metadata"], {"other_info": {"local_device": "router"}})
        self.assertEqual([request["opinion_pool"] for request in requests[4:7]], [row["opinion_pool"]] * 3)

    def test_cooperation_plans_personas_from_candidates_and_telemetry_template(self):
        def focus_on_log_detail(instruction, payload):
            answer = default_answer(instruction, payload)
            if instruction.startswith("你是 Orchestrator"):
                answer["log"] = {
                    "persona": "端口事件专家",
                    "focus": [{"key": "/anomaly_logs/local_flash/port_down", "desc": "检查端口掉线次数"}],
                }
            return answer

        label = cooperation_label()
        label["root_cause_candidates"] = [
            {"category": "链路故障", "description": "物理链路异常"},
            {"category": "配置错误", "reason": "配置发生变化"},
        ]
        label["semantic_labels"]["anomaly_logs"] = {
            "local_flash": {"port_down": 0, "port_up": 2},
            "remote_shutdown": {"user": "UNSELECTED_USER_VALUE"},
        }
        label["incident_type_description"] = "监控发现端口状态变化"
        engine = fake_engine(focus_on_log_detail)
        inference = CooperationInference(engine, sampling_params_factory=fake_sampling_params)
        with tempfile.TemporaryDirectory() as root:
            row, = inference.infer_results([label], "端口告警", root)

        self.assertEqual(row["status"], "success")
        planner = engine.llm.requests[0][1]
        self.assertEqual(planner["incident_type"], "端口告警")
        self.assertEqual(planner["incident_type_description"], "监控发现端口状态变化")
        self.assertEqual(planner["root_cause_candidates"], [
            {"category": "链路故障", "description": "物理链路异常"},
            {"category": "配置错误", "description": "配置发生变化"},
        ])
        template = planner["semantic_labels_template"]
        time_fields = {field["key"]: field for field in template["time_series"]}
        self.assertFalse(time_fields["/anomaly_kpi/offline_loss_rate"]["available"])
        self.assertNotIn("/anomaly_kpi/temperature", time_fields)
        log_fields = {field["key"]: field for field in template["log"]}
        self.assertTrue(log_fields["/anomaly_logs/local_flash/port_down"]["available"])
        self.assertNotIn("UNSELECTED_USER_VALUE", json.dumps(planner))
        for index in (1, 4):
            expert = engine.llm.requests[index][1]
            self.assertEqual(expert["specialist"], row["personas"]["log"])
            self.assertEqual(expert["log_data"], {"anomaly_logs": {"local_flash": {"port_down": 0}}})

    def test_cooperation_rejects_focus_from_another_telemetry_group(self):
        def invalid_persona(instruction, payload):
            answer = default_answer(instruction, payload)
            answer["log"]["focus"] = [{"key": "/anomaly_kpi", "desc": "检查 KPI"}]
            return answer

        engine = fake_engine(invalid_persona)
        inference = CooperationInference(engine, sampling_params_factory=fake_sampling_params)
        with tempfile.TemporaryDirectory() as root:
            row, = inference.infer_results([cooperation_label()], "场景", root)
        self.assertEqual(row["status"], "failed")
        self.assertIn("该类别", row["error"])
        self.assertEqual(len(engine.llm.requests), 1)

    def test_cooperation_selects_leaf_values_inside_arrays(self):
        key = "/anomaly_kpi/offline_loss_rate/abnormal_timing_segments/*/value/*"

        def select_samples(instruction, payload):
            answer = default_answer(instruction, payload)
            if instruction.startswith("你是 Orchestrator"):
                answer["time_series"]["focus"] = [{"key": key, "desc": "核查丢包采样值"}]
            return answer

        label = cooperation_label()
        label["semantic_labels"]["anomaly_kpi"] = {"offline_loss_rate": {
            "abnormal_timing_segments": [
                {"start_time": 10, "value": [0, 100]},
                {"start_time": 20},
                {"value": [None]},
            ],
        }}
        engine = fake_engine(select_samples)
        inference = CooperationInference(engine, sampling_params_factory=fake_sampling_params)
        with tempfile.TemporaryDirectory() as root:
            row, = inference.infer_results([label], "网络设备掉线", root)
        self.assertEqual(row["status"], "success")
        planner = engine.llm.requests[0][1]
        self.assertIn("探测丢包", planner["incident_type_description"])
        fields = {entry["key"]: entry for entry in planner["semantic_labels_template"]["time_series"]}
        self.assertTrue(fields[key]["available"])
        self.assertEqual(engine.llm.requests[2][1]["time_series_data"], {"anomaly_kpi": {
            "offline_loss_rate": {"abnormal_timing_segments": [{"value": [0, 100]}, None, {"value": [None]}]},
        }})

    def test_cooperation_extracts_shutdown_records_from_json_encoded_logs(self):
        key = "/anomaly_logs/local_shutdown"

        def select_shutdown(instruction, payload):
            answer = default_answer(instruction, payload)
            if instruction.startswith("你是 Orchestrator"):
                answer["log"]["focus"] = [{"key": key, "desc": "检查本端接口关闭记录"}]
            return answer

        label = cooperation_label()
        shutdown = {"user": "operator", "time": "12:30"}
        encoded_logs = json.dumps({"local_shutdown": shutdown, "local_flash": {"port_down": 0}})
        label["semantic_labels"]["anomaly_logs"] = encoded_logs
        engine = fake_engine(select_shutdown)
        inference = CooperationInference(engine, sampling_params_factory=fake_sampling_params)
        with tempfile.TemporaryDirectory() as root:
            row, = inference.infer_results([label], "接口状态DOWN", root)

        self.assertEqual(row["status"], "success")
        planner = engine.llm.requests[0][1]
        log_fields = {entry["key"]: entry for entry in planner["semantic_labels_template"]["log"]}
        self.assertTrue(log_fields[key]["available"])
        expert = engine.llm.requests[1][1]
        self.assertEqual(expert["log_data"], {"anomaly_logs": {"local_shutdown": shutdown}})
        self.assertEqual(expert["missing_focus_keys"], [])
        self.assertEqual(label["semantic_labels"]["anomaly_logs"], encoded_logs)

    def test_cooperation_keeps_only_latest_opinion_and_reserves_aggregator(self):
        def review_opinion(instruction, payload):
            if "opinion_pool" in payload and not instruction.startswith("你是 Aggregator"):
                time_series_has_revised = any(
                    item["agent"] == "time_series" and item["round"] == 1
                    for item in payload["opinion_pool"]
                )
                if "metadata" in payload or ("time_series_data" in payload and time_series_has_revised):
                    return "NO COMMENT"
                return {"reasoning_chain": ["新增证据"], "root_cause": "链路故障"}
            return default_answer(instruction, payload)

        engine = fake_engine(review_opinion)
        inference = CooperationInference(engine, sampling_params_factory=fake_sampling_params)
        with tempfile.TemporaryDirectory() as root:
            row, = inference.infer_results([cooperation_label()], "场景", root)
        self.assertEqual(row["status"], "success")
        self.assertEqual(row["agent_invocations"], 10)
        self.assertEqual(row["stop_reason"], "invoke_limit")
        pools = [payload["opinion_pool"] for _prompt, payload in engine.llm.requests[4:]]
        self.assertEqual([[item["round"] for item in pool] for pool in pools], [
            [0, 0, 0], [0, 0, 0], [0, 0, 0], [1, 1, 0], [1, 1, 0], [2, 1, 0],
        ])
        self.assertEqual(pools[-1], row["opinion_pool"])

    def test_cooperation_skips_missing_telemetry_and_accepts_initial_abstention(self):
        def missing_telemetry(instruction, payload):
            if "metadata" in payload:
                return "NO COMMENT"
            return default_answer(instruction, payload)

        label = cooperation_label()
        label["semantic_labels"]["anomaly_kpi"] = {"offline_loss_rate": {"abnormal_timing_segments": []}}
        engine = fake_engine(missing_telemetry)
        inference = CooperationInference(engine, sampling_params_factory=fake_sampling_params)
        with tempfile.TemporaryDirectory() as root:
            row, = inference.infer_results([label], "场景", root)
            self.assertTrue((Path(row["output_dir"]) / "aggregator/raw_responses.txt").is_file())
        self.assertEqual(row["status"], "success")
        self.assertEqual(row["predicted_list_v1"], ["链路故障", "配置错误"])
        self.assertEqual(row["rounds"][0]["no_comment"], ["metadata"])
        self.assertFalse(any("time_series_data" in payload for _prompt, payload in engine.llm.requests))
        aggregator = engine.llm.requests[-1][1]
        self.assertEqual(list(aggregator["skipped_specialists"]), ["time_series"])
        self.assertEqual(aggregator["abstained_specialists"], ["metadata"])
        self.assertEqual([item["agent"] for item in aggregator["opinion_pool"]], ["log"])

    def test_cooperation_keeps_other_responses_when_log_completion_is_empty(self):
        engine = fake_engine()
        original_chat = engine.llm.chat

        def chat_without_log_completion(prompts, params, use_tqdm=False):
            outputs = original_chat(prompts, params, use_tqdm=use_tqdm)
            for chat, output in zip(prompts, outputs):
                _, payload = chat[0]["content"].split("\n\n输入 JSON\n", 1)
                if "log_data" in json.loads(payload):
                    output.outputs = []
            return outputs

        engine.llm.chat = chat_without_log_completion
        inference = CooperationInference(engine, sampling_params_factory=fake_sampling_params)
        label = cooperation_label()
        label["semantic_labels"]["anomaly_kpi"] = {}
        with tempfile.TemporaryDirectory() as root:
            row, = inference.infer_results([label], "场景", root)
            case_dir = Path(row["output_dir"])
            self.assertTrue((case_dir / "round0/log/prompt.txt").is_file())
            self.assertEqual((case_dir / "round0/log/raw_responses.txt").read_text(encoding="utf-8"), "")
            self.assertTrue((case_dir / "round0/metadata/raw_responses.txt").read_text(encoding="utf-8"))

        self.assertEqual(row["status"], "success")
        self.assertEqual([entry["agent"] for entry in row["opinion_pool"]], ["metadata"])
        self.assertEqual(row["specialist_issues"][0], {
            "agent": "log", "round": 0, "error": "log 专家未返回最终回答",
        })

    def test_cooperation_records_insufficient_evidence_when_all_telemetry_is_empty(self):
        label = cooperation_label()
        label["semantic_labels"] = {"anomaly_logs": {}, "anomaly_kpi": {"value": [None, " "]}}

        def omit_unavailable_personas(instruction, payload):
            return {}

        engine = fake_engine(omit_unavailable_personas)
        inference = CooperationInference(engine, sampling_params_factory=fake_sampling_params)
        with tempfile.TemporaryDirectory() as root:
            row, = inference.infer_results([label], "场景", root)
            saved = json.loads((Path(row["output_dir"]) / "result.json").read_text(encoding="utf-8"))
        self.assertEqual(saved["stop_reason"], "insufficient_evidence")
        self.assertEqual(saved["predicted_list_v1"], [])
        self.assertEqual(set(saved["skipped_specialists"]), {"log", "metadata", "time_series"})
        self.assertEqual(len(engine.llm.requests), 1)

    def test_nested_groundtruth_is_scored_but_never_sent_to_agents(self):
        sentinel = "EVALUATION_ONLY_PRIVATE_ANNOTATION"
        for inference_class, label_factory in (
            (CompetitionInference, competition_label), (CooperationInference, cooperation_label),
        ):
            with self.subTest(mode=inference_class.mode), tempfile.TemporaryDirectory() as root:
                label = label_factory()
                label.pop("root_cause")
                label["semantic_labels"]["root_cause"] = {
                    "category": "配置错误", "description": sentinel,
                }
                engine = fake_engine()
                inference = inference_class(engine, sampling_params_factory=fake_sampling_params)
                row, = inference.infer_results([label], "场景", root)

                self.assertEqual(row["status"], "success")
                self.assertEqual(row["groundtruth"], "配置错误")
                self.assertEqual(row["rank_v1"], 2)
                self.assertEqual(label["semantic_labels"]["root_cause"]["description"], sentinel)
                prompts = list(Path(row["output_dir"]).rglob("prompt.txt"))
                self.assertEqual(len(prompts), len(engine.llm.requests))
                for prompt in prompts:
                    self.assertNotIn(sentinel, prompt.read_text(encoding="utf-8"))
                for _instruction, payload in engine.llm.requests:
                    if "case" in payload:
                        self.assertNotIn("root_cause", payload["case"]["semantic_labels"])

    def test_bad_item_does_not_block_other_items(self):
        bad = competition_label()
        bad["semantic_labels"]["sop"] = "无法编译的文本"
        inference = CompetitionInference(fake_engine(), sampling_params_factory=fake_sampling_params)
        with tempfile.TemporaryDirectory() as root:
            first, second = inference.infer_results([bad, competition_label()], "场景", root)
        self.assertEqual(first["status"], "failed")
        self.assertEqual(second["status"], "success")

    def test_competition_sends_low_confidence_candidates_without_eliminating_them(self):
        def low_confidence(instruction, payload):
            answer = default_answer(instruction, payload)
            if "decision_templates" in payload:
                for decision in answer["decisions"].values():
                    decision.update(condition_met=False, confidence=0.95, reason="观察到相反状态")
                if payload["root_cause"] == "链路故障":
                    answer["conclusion"].update(confidence=0, reason="节点证据与该分支要求相反")
            return answer

        engine = fake_engine(low_confidence)
        inference = CompetitionInference(engine, sampling_params_factory=fake_sampling_params)
        with tempfile.TemporaryDirectory() as root:
            row, = inference.infer_results([competition_label()], "场景", root)
        self.assertEqual(row["status"], "success")
        verifier = engine.llm.requests[-1][1]
        self.assertEqual(len(verifier["diagnoses"]), 2)
        first = verifier["diagnoses"][0]
        self.assertEqual(first["conclusion"]["confidence"], 0)
        self.assertIs(first["decisions"]["step_01"]["condition_met"], False)

    def test_competition_fills_missing_decisions_before_verification(self):
        def omit_decisions(instruction, payload):
            answer = default_answer(instruction, payload)
            if "decision_templates" in payload:
                if payload["root_cause"] == "根因1":
                    answer.pop("decisions")
                else:
                    answer["decisions"].pop("step_01")
                    answer["decisions"]["step_02"].update(condition_met=False, confidence=0.9)
            return answer

        engine = fake_engine(omit_decisions)
        inference = CompetitionInference(engine, sampling_params_factory=fake_sampling_params)
        with tempfile.TemporaryDirectory() as root:
            row, = inference.infer_results([competition_label(("根因1", "根因2", "根因3"))], "场景", root)
            self.assertTrue((Path(row["output_dir"]) / "verifier/raw_responses.txt").is_file())
        self.assertEqual(row["status"], "success")
        self.assertEqual(row["predicted_list_v1"], ["根因1", "根因2", "根因3"])
        diagnoses = engine.llm.requests[-1][1]["diagnoses"]
        for diagnosis in diagnoses:
            missing = diagnosis["decisions"]["step_01"]
            self.assertIsNone(missing["condition_met"])
            self.assertIsNone(missing["confidence"])
            self.assertIn("未报告", missing["reason"])
            self.assertEqual(diagnosis["conclusion"]["confidence"], 0.2)
        self.assertIs(diagnoses[1]["decisions"]["step_02"]["condition_met"], False)

    def test_competition_rejects_binary_root_cause_verdicts(self):
        def binary_verdict(instruction, payload):
            answer = default_answer(instruction, payload)
            if "decision_templates" in payload:
                answer["conclusion"]["supported"] = False
            return answer

        engine = fake_engine(binary_verdict)
        inference = CompetitionInference(engine, sampling_params_factory=fake_sampling_params)
        with tempfile.TemporaryDirectory() as root:
            row, = inference.infer_results([competition_label()], "场景", root)
        self.assertEqual(row["status"], "failed")
        self.assertIn("conclusion", row["error"])
        self.assertEqual(len(engine.llm.requests), 2)

    def test_competition_writes_category_csv_using_list_index(self):
        categories = [f"根因{i}" for i in range(1, 7)]
        inference = CompetitionInference(fake_engine(), sampling_params_factory=fake_sampling_params)
        with tempfile.TemporaryDirectory() as root:
            result, = inference.infer([competition_label(categories)], "场景", root)
            csv_path = Path(root) / "competition" / "场景" / "results.csv"
            with csv_path.open(encoding="utf-8-sig", newline="") as file:
                row, = list(csv.DictReader(file))
        self.assertEqual(result["case_idx"], 0)
        self.assertEqual(result["predicted_list"], categories[:5])
        self.assertIsNone(result["rank"])
        self.assertEqual(result["rank_v1"], 6)
        self.assertEqual(list(result), list(row))
        self.assertEqual(json.loads(row["predicted_list_v1"]), categories)

    def test_evaluation_runs_supplied_dataset_on_one_engine(self):
        engine = fake_engine()
        get_instance = Mock(return_value=engine)
        fake_llm_engine = types.ModuleType("llm_interfaces.llm_engine")
        fake_llm_engine.LLMEngine = SimpleNamespace(get_instance=get_instance)
        fake_vllm = types.ModuleType("vllm")
        fake_vllm.SamplingParams = fake_sampling_params
        with tempfile.TemporaryDirectory() as root, patch.dict(
            sys.modules, {"vllm": fake_vllm, "llm_interfaces.llm_engine": fake_llm_engine},
        ), patch.dict(os.environ):
            label_root = Path(root) / "labels"
            for category in ("场景A", "场景B"):
                folder = label_root / category
                folder.mkdir(parents=True)
                label = competition_label()
                label["sop"] = label["semantic_labels"].pop("sop")
                (folder / f"{category}_0_label_sample.json").write_text(
                    json.dumps(label), encoding="utf-8",
                )
            output_dir = Path(root) / "output"
            args = SimpleNamespace(
                label_root=label_root, output_dir=output_dir, mode="both", model_path="fake-model",
                ascend_devices="0", max_model_len=16384, batch_size=8, invoke_limit=10,
                limit_per_category=None, import_top_level_sop=True,
            )
            rows = evaluate(args)
            self.assertTrue((output_dir / "metrics.csv").is_file())
            trace = output_dir / "traces" / "场景A" / "cooperation" / "场景A" / "case_000000"
            result = json.loads((trace / "result.json").read_text(encoding="utf-8"))
            self.assertEqual(Path(result["label_file_path"]), label_root / "场景A" / "场景A_0_label_sample.json")
            self.assertNotIn("label_file_path", (trace / "orchestrator/prompt.txt").read_text(encoding="utf-8"))
        get_instance.assert_called_once()
        totals = [row for row in rows if row["alarm_type"] is None]
        self.assertEqual([row["attempted_count"] for row in totals], [2, 2])
        self.assertEqual([row["failed_count"] for row in totals], [0, 0])

    def test_sharded_evaluation_uses_disjoint_cases_and_matches_serial_metrics(self):
        get_instance = Mock(side_effect=lambda **_kwargs: fake_engine())
        fake_llm_engine = types.ModuleType("llm_interfaces.llm_engine")
        fake_llm_engine.LLMEngine = SimpleNamespace(get_instance=get_instance)
        fake_vllm = types.ModuleType("vllm")
        fake_vllm.SamplingParams = fake_sampling_params
        with tempfile.TemporaryDirectory() as root, patch.dict(
            sys.modules, {"vllm": fake_vllm, "llm_interfaces.llm_engine": fake_llm_engine},
        ), patch.dict(os.environ):
            root = Path(root)
            category = root / "labels" / "场景"
            category.mkdir(parents=True)
            first = competition_label()
            first["root_cause"] = {"category": "链路故障"}
            labels = [first, cooperation_label(), competition_label(), competition_label()]
            for index, label in enumerate(labels):
                (category / f"场景_{index + 10}_label_sample.json").write_text(
                    json.dumps(label), encoding="utf-8",
                )
            second_category = category.parent / "场景二"
            second_category.mkdir()
            for index in range(2):
                (second_category / f"场景二_{index}_label_sample.json").write_text(
                    json.dumps(competition_label()), encoding="utf-8",
                )
            args = SimpleNamespace(
                label_root=category.parent, output_dir=root / "serial", mode="both",
                model_path="fake-model", ascend_devices="0,1", max_model_len=16384,
                batch_size=8, invoke_limit=10, limit_per_category=3, import_top_level_sop=False,
                limit_per_mode=2,
            )
            serial = evaluate(args)
            parallel = root / "parallel"
            args.shard_count = 4
            for index in range(args.shard_count):
                args.shard_index = index
                args.output_dir = parallel / f"shard_{index}"
                evaluate(args)

            self.assertEqual(get_instance.call_count, 3)
            trace = parallel / "shard_2" / "traces" / "场景" / "competition" / "results.json"
            cases = json.loads(trace.read_text(encoding="utf-8"))["results"]
            self.assertEqual([case["case_idx"] for case in cases], [2])
            merged = merge_shards(parallel, args.shard_count, 5)
            report = json.loads((parallel / "metrics.json").read_text(encoding="utf-8"))
            self.assertEqual(report["excluded_without_sop"], 1)
            self.assertEqual(report["limit_per_mode"], 2)
            self.assertEqual(report["wall_elapsed_seconds"], 5)
            self.assertTrue((parallel / "metrics.csv").is_file())

        fields = ("mode", "alarm_type", "attempted_count", "top1", "top3", "top5", "mrr")
        self.assertEqual(
            [[row[field] for field in fields] for row in merged],
            [[row[field] for field in fields] for row in serial],
        )
        self.assertEqual(merged[0]["attempted_count"], 2)
        self.assertEqual(merged[0]["mrr"], 0.75)
        self.assertEqual([row["attempted_count"] for row in merged if row["alarm_type"] is None], [2, 2])

    def test_incremental_agents_route_nested_sop_and_stop_checkpoint_at_failure(self):
        engine = fake_engine()
        get_instance = Mock(return_value=engine)
        fake_llm_engine = types.ModuleType("llm_interfaces.llm_engine")
        fake_llm_engine.LLMEngine = SimpleNamespace(get_instance=get_instance)
        fake_vllm = types.ModuleType("vllm")
        fake_vllm.SamplingParams = fake_sampling_params
        with tempfile.TemporaryDirectory() as root, patch.dict(
            sys.modules, {"vllm": fake_vllm, "llm_interfaces.llm_engine": fake_llm_engine},
        ):
            label_root = Path(root) / "labels"
            category = label_root / "场景"
            category.mkdir(parents=True)
            failed = competition_label()
            failed["semantic_labels"]["sop"] = "无法编译的文本"
            labels = [
                {**cooperation_label(), "sop": "顶层 SOP 不用于分流"},
                competition_label(), failed, cooperation_label(),
            ]
            for idx, label in enumerate(labels):
                (category / f"场景_{idx}_label_sample.json").write_text(
                    json.dumps(label), encoding="utf-8",
                )
            args = SimpleNamespace(
                agentdigest_label_root=str(label_root), scenario="all", batch_size=8,
                state_file=str(Path(root) / "state.json"), start=None, end=None,
                reset_state=False, max_new_cases=None, llm_mode="auto", invoke_limit=10,
                output_dir=str(Path(root) / "output"), output_format="json", strict=False,
            )
            payloads = {item["meta"]["llm_mode"]: item for item in llm_infer(args)}
            get_instance.assert_called_once()
            self.assertEqual(payloads["competition"]["summary"]["processed_indices"], [1])
            self.assertEqual(payloads["competition"]["summary"]["skipped_indices"], [2])
            self.assertEqual(payloads["cooperation"]["summary"]["processed_indices"], [0, 3])
            state = load_infer_state(args.state_file)
            self.assertEqual(get_last_processed_index(state, "llm_infer", str(category), -1), 1)
            state_before_debug = Path(args.state_file).read_bytes()
            args.start, args.end = 3, 4
            llm_infer(args)
            self.assertEqual(Path(args.state_file).read_bytes(), state_before_debug)


if __name__ == "__main__":
    unittest.main()
