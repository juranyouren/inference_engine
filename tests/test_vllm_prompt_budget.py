import os
import tempfile
import unittest
from unittest.mock import patch

from utils.public_functions import fit_vllm_inputs, vllm_invoke


class FakeTokenizer:
    @staticmethod
    def encode(text, add_special_tokens=False):
        return [ord(char) for char in text]

    @staticmethod
    def decode(token_ids, skip_special_tokens=True):
        return "".join(chr(token_id) for token_id in token_ids)


class FakeModelConfig:
    max_model_len = 80


class FakeEngine:
    model_config = FakeModelConfig()


class FakeOutputText:
    text = "ok"


class FakeRequestOutput:
    outputs = [FakeOutputText()]


class FakeLlm:
    llm_engine = FakeEngine()

    def __init__(self):
        self.received_prompts = None

    @staticmethod
    def get_tokenizer():
        return FakeTokenizer()

    def chat(self, prompts, sampling_params, use_tqdm=False):
        self.received_prompts = prompts
        return [FakeRequestOutput() for _item in prompts]


class FakeSamplingParams:
    n = 1
    max_tokens = 20


class VllmPromptBudgetTests(unittest.TestCase):
    @patch("config.LLM_PROMPT_SAFETY_TOKENS", 10)
    def test_fit_inputs_preserves_head_and_tail_within_budget(self):
        llm = FakeLlm()
        fitted, stats = fit_vllm_inputs(
            llm,
            ["A" * 50 + "B" * 50],
            FakeSamplingParams(),
        )
        self.assertTrue(stats[0]["truncated"])
        self.assertLessEqual(stats[0]["final_tokens"], 50)
        self.assertTrue(fitted[0].startswith("A"))
        self.assertTrue(fitted[0].endswith("B"))

    @patch("config.LLM_PROMPT_SAFETY_TOKENS", 10)
    def test_vllm_invoke_sends_only_fitted_prompt(self):
        llm = FakeLlm()
        responses = vllm_invoke(
            llm,
            ["X" * 100],
            FakeSamplingParams(),
            batch_size=1,
        )
        sent = llm.received_prompts[0][0]["content"]
        self.assertLessEqual(len(FakeTokenizer.encode(sent)), 50)
        self.assertEqual(responses, ["ok"])

    @patch("config.LLM_PROMPT_SAFETY_TOKENS", 10)
    def test_vllm_invoke_saves_exact_fitted_prompt(self):
        llm = FakeLlm()
        with tempfile.TemporaryDirectory() as root:
            prompt_path = os.path.join(root, "case", "prompt.txt")
            vllm_invoke(
                llm,
                ["X" * 100],
                FakeSamplingParams(),
                batch_size=1,
                prompt_output_paths=[prompt_path],
            )
            with open(prompt_path, "r", encoding="utf-8") as f:
                saved = f.read()

        sent = llm.received_prompts[0][0]["content"]
        self.assertEqual(saved, sent)
        self.assertLessEqual(len(FakeTokenizer.encode(saved)), 50)


if __name__ == "__main__":
    unittest.main()
