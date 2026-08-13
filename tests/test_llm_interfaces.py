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
from unittest.mock import patch

from llm_interfaces.competition import CompetitionInference
from llm_interfaces.cooperation import CooperationInference


class FakeTokenizer:
    @staticmethod
    def encode(text, add_special_tokens=False):
        return [ord(char) for char in text]

    @staticmethod
    def decode(token_ids, skip_special_tokens=True):
        return "".join(chr(token_id) for token_id in token_ids)


class FakeOutput:
    def __init__(self, text):
        self.outputs = [SimpleNamespace(text=text)]


class FakeLlm:
    def __init__(self, scripted_responses=None):
        self.llm_engine = SimpleNamespace(
            model_config=SimpleNamespace(max_model_len=100000)
        )
        self.scripted_responses = list(scripted_responses or [])
        self.chat_calls = 0

    @staticmethod
    def get_tokenizer():
        return FakeTokenizer()

    def chat(self, prompts, sampling_params, use_tqdm=False):
        self.chat_calls += 1
        if self.scripted_responses:
            response = self.scripted_responses.pop(0)
        else:
            response = f"response-{self.chat_calls}"
        return [FakeOutput(response) for _prompt in prompts]


def fake_sampling_params(**kwargs):
    return SimpleNamespace(n=1, **kwargs)


def fake_engine(scripted_responses=None):
    return SimpleNamespace(llm=FakeLlm(scripted_responses))


def competition_label():
    return {
        "case_id": "case-a",
        "alarm_time": 123,
        "semantic_labels": {
            "sop": "先检查邻居状态，再检查链路。",
            "anomaly_logs": {"message": "peer down"},
        },
        "root_cause_candidates": [
            {"category": "链路故障"},
            {"category": "配置错误"},
        ],
        "root_cause": {"category": "链路故障"},
        "ttm": 16.0,
    }


def cooperation_label():
    return {
        "case_id": "case-b",
        "alarm_time": 456,
        "semantic_labels": {
            "sop": "",
            "anomaly_logs": {"message": "peer down"},
        },
        "root_cause_candidates": [
            {"category": "链路故障"},
            {"category": "配置错误"},
        ],
        "root_cause": {"category": "链路故障"},
        "ttm": 50.0,
    }


class LlmInterfaceTests(unittest.TestCase):
    def test_two_inference_classes_share_the_same_engine(self):
        engine = fake_engine()
        competition = CompetitionInference(
            engine,
            sampling_params_factory=fake_sampling_params,
        )
        cooperation = CooperationInference(
            engine,
            sampling_params_factory=fake_sampling_params,
        )

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
                module.LLMEngine._instance = None
                with patch.dict(
                    os.environ,
                    {"ASCEND_RT_VISIBLE_DEVICES": "0,1"},
                ):
                    first = module.LLMEngine.get_instance(
                        model_path="fake-model",
                    )
                    second = module.LLMEngine.get_instance()
        finally:
            sys.modules.pop(module_name, None)
            if previous_module is not None:
                sys.modules[module_name] = previous_module

        self.assertIs(first, second)
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0]["model"], "fake-model")
        self.assertEqual(calls[0]["tensor_parallel_size"], 2)

    def test_competition_reuses_shared_model_across_calls(self):
        engine_instance = fake_engine()
        with tempfile.TemporaryDirectory() as root:
            engine = CompetitionInference(
                engine_instance,
                sampling_params_factory=fake_sampling_params,
            )

            first = engine.infer(
                [competition_label()],
                "BGP邻接状态改变",
                Path(root) / "first",
            )
            second = engine.infer(
                [competition_label()],
                "BGP邻接状态改变",
                Path(root) / "second",
            )

            self.assertIs(engine.llm, engine_instance.llm)
            self.assertIn("predicted_list", first[0])
            self.assertIn("rank", second[0])
            self.assertTrue(
                (Path(root) / "first" / "competition" / "results.json").exists()
            )

    def test_cooperation_stops_on_marker_and_saves_rounds(self):
        marker = '["(上一轮次已经符合要求)"]'
        engine_instance = fake_engine([
            "meta-round-0",
            "reasoner-round-0",
            f'meta-round-1 {marker}\n["链路故障", "配置错误"]',
        ])
        with tempfile.TemporaryDirectory() as root:
            engine = CooperationInference(
                engine_instance,
                max_rounds=5,
                sampling_params_factory=fake_sampling_params,
            )
            results = engine.infer(
                [cooperation_label()], "BGP邻接状态改变", root,
            )

            self.assertEqual(results[0]["predicted_list"], [
                "链路故障", "配置错误",
            ])
            self.assertEqual(results[0]["rank"], 1)
            self.assertTrue(
                (
                    Path(root)
                    / "cooperation"
                    / "BGP邻接状态改变"
                    / "case_000000"
                    / "result.json"
                ).exists()
            )
            self.assertTrue(
                (
                    Path(root)
                    / "cooperation"
                    / "BGP邻接状态改变"
                    / "results.csv"
                ).exists()
            )

    def test_sop_is_read_only_from_semantic_labels(self):
        label = cooperation_label()
        label["sop"] = "顶层 SOP 不应参与路由"
        engine_instance = fake_engine(['["(上一轮次已经符合要求)"]'])
        with tempfile.TemporaryDirectory() as root:
            engine = CooperationInference(
                engine_instance,
                max_rounds=1,
                sampling_params_factory=fake_sampling_params,
            )
            results = engine.infer([label], "场景", root)

        self.assertEqual(results[0]["incident_type"], "场景")

    def test_bad_item_does_not_block_other_items(self):
        bad = competition_label()
        bad["semantic_labels"]["sop"] = ""
        engine_instance = fake_engine()
        with tempfile.TemporaryDirectory() as root:
            engine = CompetitionInference(
                engine_instance,
                sampling_params_factory=fake_sampling_params,
            )
            results = engine.infer([bad, competition_label()], "场景", root)
            payload = json.loads(
                (Path(root) / "competition" / "results.json").read_text(
                    encoding="utf-8"
                )
            )

        self.assertEqual(results[0]["predicted_list"], [])
        self.assertIn("rank", results[1])
        self.assertEqual(payload["summary"]["failed_count"], 1)
        self.assertEqual(payload["summary"]["success_count"], 1)

    def test_competition_writes_category_csv_using_list_index(self):
        categories = [f"根因{i}" for i in range(1, 7)]
        label = competition_label()
        label["case_id"] = "不是case_idx"
        label["root_cause_candidates"] = [
            {"category": category} for category in categories
        ]
        label["root_cause"] = {"category": "根因6"}
        verifier_response = json.dumps(categories, ensure_ascii=False)
        engine_instance = fake_engine([
            "reasoner-1",
            "reasoner-2",
            "reasoner-3",
            verifier_response,
        ])

        with tempfile.TemporaryDirectory() as root:
            engine = CompetitionInference(
                engine_instance,
                sampling_params_factory=fake_sampling_params,
            )
            results = engine.infer(
                [label], "BGP邻接状态改变", root,
            )
            csv_path = (
                Path(root)
                / "competition"
                / "BGP邻接状态改变"
                / "results.csv"
            )
            with csv_path.open("r", encoding="utf-8-sig", newline="") as file:
                rows = list(csv.DictReader(file))

        self.assertEqual(results[0]["case_idx"], 0)
        self.assertEqual(results[0]["predicted_list"], categories[:5])
        self.assertEqual(results[0]["predicted_list_v1"], categories)
        self.assertIsNone(results[0]["rank"])
        self.assertEqual(results[0]["rank_v1"], 6)
        self.assertEqual(list(results[0]), [
            "case_idx",
            "timestamp",
            "incident_type",
            "predicted_list",
            "groundtruth",
            "rank",
            "ttm",
            "predicted_list_v1",
            "rank_v1",
        ])
        self.assertEqual(list(rows[0]), [
            "case_idx",
            "timestamp",
            "incident_type",
            "predicted_list",
            "groundtruth",
            "rank",
            "ttm",
            "predicted_list_v1",
            "rank_v1",
        ])
        self.assertEqual(rows[0]["case_idx"], "0")
        self.assertEqual(rows[0]["incident_type"], "BGP邻接状态改变")
        self.assertEqual(json.loads(rows[0]["predicted_list"]), categories[:5])
        self.assertEqual(
            json.loads(rows[0]["predicted_list_v1"]), categories,
        )
        self.assertEqual(rows[0]["groundtruth"], "根因6")
        self.assertEqual(rows[0]["ttm"], "16.0")
        self.assertEqual(rows[0]["rank"], "")
        self.assertEqual(rows[0]["rank_v1"], "6")


if __name__ == "__main__":
    unittest.main()
