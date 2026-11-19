import json
import os
import sys
import tempfile
import types
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from llm_inference.selector_refiner import LLMEngine
from run_tree_agent_evaluation import build_arg_parser, run


class SelectionModel:
    llm_engine = SimpleNamespace(model_config=SimpleNamespace(max_model_len=100000))

    def __init__(self):
        self.prompts = []

    def get_tokenizer(self):
        return SimpleNamespace(
            encode=lambda text, **kwargs: list(map(ord, text)),
            decode=lambda tokens, **kwargs: "".join(map(chr, tokens)),
        )

    def chat(self, chats, params, use_tqdm=False):
        self.prompts.extend(chat[0]["content"] for chat in chats)
        text = json.dumps({"log": ["flash_count"], "kpi": {}})
        return [
            SimpleNamespace(outputs=[SimpleNamespace(text=text) for _ in range(params.n)])
            for _ in chats
        ]


class FixedSplitSelectionTests(unittest.TestCase):
    def test_regenerates_per_category_from_training_without_truth_or_external_templates(self):
        engine = LLMEngine.__new__(LLMEngine)
        engine.llm = SelectionModel()
        engine.max_model_len = 100000
        fake_vllm = types.ModuleType("vllm")
        fake_vllm.SamplingParams = lambda **kwargs: SimpleNamespace(**kwargs)
        tree_runs = []

        def tree_once(**kwargs):
            self.assertEqual(kwargs["train_indices"], list(range(50)))
            self.assertEqual(kwargs["infer_indices"], list(range(50, 200)))
            self.assertEqual(kwargs["selection"]["log"], ["flash_count"])
            tree_runs.append(kwargs)
            return {"results": []}

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for category in ("category_a", "category_b"):
                folder = root / "dataset" / category
                folder.mkdir(parents=True)
                stale = {"log": ["STALE_SELECTION"], "kpi": {}}
                (folder / "selection.json").write_text(json.dumps(stale), encoding="utf-8")
                for index in range(200):
                    label = {
                        "alarm_time": 123,
                        "sop": "CURRENT_DATASET_SOP",
                        "root_cause": {"category": "PRIVATE_TRUTH"},
                        "root_cause_candidates": [{"category": "candidate_a"}, {"category": "candidate_b"}],
                        "semantic_labels": {
                            "root_cause": {"category": "PRIVATE_TRUTH"},
                            "anomaly_logs": {
                                "message": category if index < 50 else "TEST_EVIDENCE",
                            },
                        },
                    }
                    (folder / f"{category}_{index}_label_sample.json").write_text(
                        json.dumps(label), encoding="utf-8",
                    )
                output = root / "output" / "tree" / category
                output.mkdir(parents=True)
                (output / "selection.json").write_text(json.dumps(stale), encoding="utf-8")

            args = build_arg_parser().parse_args([
                "--dataset-root", str(root / "dataset"), "--mode", "tree",
                "--output-dir", str(root / "output"), "--import-top-level-sop",
                "--model-path", "current-model", "--ascend-devices", "2,3",
                "--max-model-len", "100000",
            ])
            self.assertEqual(args.selection_source, "selector")
            with patch.dict(sys.modules, {"vllm": fake_vllm}), patch.dict(os.environ), patch(
                "llm_inference.selector_refiner.LLMEngine.get_instance", return_value=engine,
            ) as get_engine, patch(
                "inference.selection.load_alarm_template_compat", side_effect=AssertionError("external template"),
            ), patch("inference.tree.run_tree_once", side_effect=tree_once):
                run(args)
                get_engine.assert_called_once_with(
                    model_path="current-model", gpu_memory_utilization=0.9, max_model_len=100000,
                )
                self.assertEqual(os.environ["ASCEND_RT_VISIBLE_DEVICES"], "2,3")
                run(args)

            self.assertEqual(len(tree_runs), 4)
            self.assertEqual(len(engine.llm.prompts), 4)
            for prompt in engine.llm.prompts:
                self.assertIn("CURRENT_DATASET_SOP", prompt)
                self.assertIn("candidate_a", prompt)
                self.assertNotIn("PRIVATE_TRUTH", prompt)
                self.assertNotIn("TEST_EVIDENCE", prompt)
                self.assertNotIn("STALE_SELECTION", prompt)
            for category in ("category_a", "category_b"):
                output = root / "output" / "tree" / category
                selection = json.loads((output / "selection.json").read_text(encoding="utf-8"))
                self.assertEqual(selection["log"], ["flash_count"])
                self.assertTrue((output / "selector" / "prompt_0.txt").is_file())
                self.assertTrue((output / "selector" / "records.json").is_file())
                source = json.loads((output / "selector" / "source.json").read_text(encoding="utf-8"))
                self.assertEqual(source["input_index"], 0)
                self.assertEqual(source["train_indices"], list(range(50)))


if __name__ == "__main__":
    unittest.main()
