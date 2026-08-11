import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

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


class CountingFactory:
    def __init__(self, scripted_responses=None):
        self.calls = 0
        self.scripted_responses = scripted_responses
        self.instance = None

    def __call__(self, **kwargs):
        self.calls += 1
        self.instance = FakeLlm(self.scripted_responses)
        return self.instance


def fake_sampling_params(**kwargs):
    return SimpleNamespace(n=1, **kwargs)


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
    }


class LlmInterfaceTests(unittest.TestCase):
    def test_competition_loads_model_once_and_reuses_it(self):
        factory = CountingFactory()
        with tempfile.TemporaryDirectory() as root:
            engine = CompetitionInference(
                "fake-model",
                alarm_type="BGP邻接状态改变",
                llm_factory=factory,
                sampling_params_factory=fake_sampling_params,
            )
            self.assertEqual(factory.calls, 1)

            first = engine.infer([competition_label()], Path(root) / "first")
            second = engine.infer([competition_label()], Path(root) / "second")

            self.assertEqual(factory.calls, 1)
            self.assertEqual(first[0]["status"], "success")
            self.assertEqual(second[0]["status"], "success")
            self.assertEqual(len(first[0]["reasoner_outputs"]), 3)
            self.assertTrue(
                (Path(root) / "first" / "results.json").exists()
            )

    def test_cooperation_stops_on_marker_and_saves_rounds(self):
        marker = '["(上一轮次已经符合要求)"]'
        factory = CountingFactory([
            "meta-round-0",
            "reasoner-round-0",
            f"meta-round-1 {marker}",
        ])
        with tempfile.TemporaryDirectory() as root:
            engine = CooperationInference(
                "fake-model",
                alarm_type="BGP邻接状态改变",
                max_rounds=5,
                llm_factory=factory,
                sampling_params_factory=fake_sampling_params,
            )
            results = engine.infer([cooperation_label()], root)

            self.assertEqual(factory.calls, 1)
            self.assertEqual(results[0]["status"], "success")
            self.assertTrue(results[0]["early_stopped"])
            self.assertEqual(results[0]["round_count"], 2)
            self.assertEqual(len(results[0]["rounds"]), 2)
            self.assertIsNone(results[0]["rounds"][1]["reasoner_output"])
            self.assertTrue(
                (Path(root) / "case_000000" / "result.json").exists()
            )

    def test_sop_is_read_only_from_semantic_labels(self):
        label = cooperation_label()
        label["sop"] = "顶层 SOP 不应参与路由"
        factory = CountingFactory(['["(上一轮次已经符合要求)"]'])
        with tempfile.TemporaryDirectory() as root:
            engine = CooperationInference(
                "fake-model",
                alarm_type="场景",
                max_rounds=1,
                llm_factory=factory,
                sampling_params_factory=fake_sampling_params,
            )
            results = engine.infer([label], root)

        self.assertEqual(results[0]["status"], "success")

    def test_bad_item_does_not_block_other_items(self):
        bad = competition_label()
        bad["semantic_labels"]["sop"] = ""
        factory = CountingFactory()
        with tempfile.TemporaryDirectory() as root:
            engine = CompetitionInference(
                "fake-model",
                alarm_type="场景",
                llm_factory=factory,
                sampling_params_factory=fake_sampling_params,
            )
            results = engine.infer([bad, competition_label()], root)
            payload = json.loads(
                (Path(root) / "results.json").read_text(encoding="utf-8")
            )

        self.assertEqual(results[0]["status"], "failed")
        self.assertEqual(results[1]["status"], "success")
        self.assertEqual(payload["summary"]["failed_count"], 1)
        self.assertEqual(payload["summary"]["success_count"], 1)


if __name__ == "__main__":
    unittest.main()
