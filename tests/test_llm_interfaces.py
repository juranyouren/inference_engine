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
            "conclusion": {**payload["conclusion_template"], "reason": "证据不足"},
        }
    if "diagnoses" in payload:
        return payload["root_cause_candidates"]
    if instruction.startswith("你是 Orchestrator"):
        groups = {"log_data": [], "time_series_data": [], "metadata": []}
        for key in payload["case"]["semantic_labels"]:
            group = {
                "anomaly_logs": "log_data", "anomaly_kpi": "time_series_data",
                "anomaly_netstreams": "time_series_data",
            }.get(key, "metadata")
            groups[group].append(key)
        return groups
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

    def test_cooperation_shares_new_opinions_on_next_round_and_reserves_aggregator(self):
        def comment_on_every_review(instruction, payload):
            if "opinion_pool" in payload and not instruction.startswith("你是 Aggregator"):
                return {"reasoning_chain": ["新增证据"], "root_cause": "链路故障"}
            return default_answer(instruction, payload)

        engine = fake_engine(comment_on_every_review)
        inference = CooperationInference(engine, sampling_params_factory=fake_sampling_params)
        with tempfile.TemporaryDirectory() as root:
            row, = inference.infer_results([cooperation_label()], "场景", root)
        self.assertEqual(row["status"], "success")
        self.assertEqual(row["agent_invocations"], 10)
        self.assertEqual(row["stop_reason"], "invoke_limit")
        pools = [payload["opinion_pool"] for _prompt, payload in engine.llm.requests[4:]]
        self.assertEqual([len(pool) for pool in pools], [3, 3, 3, 6, 6, 8])
        self.assertEqual(pools[-1], row["opinion_pool"])

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

    def test_inconsistent_candidate_is_not_sent_to_verifier(self):
        def inconsistent(instruction, payload):
            answer = default_answer(instruction, payload)
            if "decision_templates" in payload:
                answer["conclusion"].update(supported=True, confidence=0.8)
            return answer

        engine = fake_engine(inconsistent)
        inference = CompetitionInference(engine, sampling_params_factory=fake_sampling_params)
        with tempfile.TemporaryDirectory() as root:
            row, = inference.infer_results([competition_label()], "场景", root)
        self.assertEqual(row["status"], "failed")
        self.assertIn("与所填决策路径不一致", row["error"])
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
        get_instance.assert_called_once()
        totals = [row for row in rows if row["alarm_type"] is None]
        self.assertEqual([row["attempted_count"] for row in totals], [2, 2])
        self.assertEqual([row["failed_count"] for row in totals], [0, 0])

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
