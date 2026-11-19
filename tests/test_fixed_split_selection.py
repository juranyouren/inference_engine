import json
import math
import os
import sys
import tempfile
import types
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from llm_inference.selector_refiner import LLMEngine
from inference.selection import bind_training_selection, build_selection_from_selector, training_feature_catalog
from inference.tree import extract_features_for_cases
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
        text = json.dumps({"log": ["syslog_brief", "flash_count", "unknown"], "kpi": {}})
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
            start = kwargs["infer_indices"][0]
            self.assertEqual(kwargs["train_indices"], list(range(start)))
            self.assertEqual(kwargs["infer_indices"], list(range(start, min(start + args.tree_block_size, 200))))
            self.assertEqual(set(kwargs["selection"]["log"]), {
                f"syslog_brief:block_{i // 50}" for i in range(start)
            })
            tree_runs.append(kwargs)
            return {"results": [{"case_idx": i, "groundtruth": "x", "rank": 1} for i in kwargs["infer_indices"]]}

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
                                "syslog_brief": {f"block_{index // 50}": 1},
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
            ), patch("inference.tree.run_tree_once", side_effect=tree_once), patch("inference.common.save_csv"):
                run(args)
                get_engine.assert_called_once_with(
                    model_path="current-model", gpu_memory_utilization=0.9, max_model_len=100000,
                )
                self.assertEqual(os.environ["ASCEND_RT_VISIBLE_DEVICES"], "2,3")
                args.tree_block_size = 60
                run(args)

            self.assertEqual([len(call["train_indices"]) for call in tree_runs],
                             [50, 100, 150] * 2 + [50, 110, 170] * 2)
            self.assertEqual(len(engine.llm.prompts), 12)
            for prompt, call in zip(engine.llm.prompts, tree_runs):
                self.assertIn("CURRENT_DATASET_SOP", prompt)
                self.assertIn("candidate_a", prompt)
                for block in range(4):
                    self.assertEqual(f"block_{block}" in prompt, block * 50 < len(call["train_indices"]))
                self.assertNotIn("PRIVATE_TRUTH", prompt)
                self.assertNotIn("STALE_SELECTION", prompt)
            for category in ("category_a", "category_b"):
                folder = root / "output" / "tree" / category
                predictions = json.loads((folder / "predictions.json").read_text(encoding="utf-8"))
                self.assertEqual([r["case_idx"] for r in predictions["results"]], list(range(50, 200)))
                for number, start in enumerate((50, 110, 170), start=1):
                    output = folder / f"round_{number:02d}"
                    self.assertTrue((output / "selection.json").is_file())
                    self.assertTrue((output / "selector" / "prompt_0.txt").is_file())
                    source = json.loads((output / "selector" / "source.json").read_text(encoding="utf-8"))
                    self.assertEqual(source["train_indices"], list(range(start)))
                    self.assertEqual(source["test_indices"], list(range(start, min(start + 60, 200))))

    def test_syslog_counts_kpi_aliases_and_alarm_window(self):
        def segment(start, value):
            return {"start_time": start, "end_time": start + 10, "value": [value]}

        alarm = 1760640900
        kpi = {"abnormal_timing_segments": [segment(alarm, 21), segment(alarm + 100, 5)]}
        case = {"alarm_time": alarm * 1000, "semantic_labels": {
            "anomaly_logs": {
                "syslog_brief": {"linkDown_active(l)": 15, "linkDown_clear(l)": 11, "PEER_STATE_CHG(l)": 10},
                "syslog_message": {"peer/down: <*>": 4},
            },
            "anomaly_kpi": {"cpu_usage": kpi, "memory_usage": kpi, "device_temperature": kpi},
        }}
        raw = {"log": ["flash_count", "syslog_brief", "syslog_message"],
               "kpi": {"cpu_usage": ["alarming", "max"]}}
        selection = bind_training_selection(build_selection_from_selector(raw), training_feature_catalog([case]))
        names, _, (features, missing) = extract_features_for_cases([case, {"semantic_labels": {}}], selection)
        self.assertEqual(list(missing), names)
        self.assertEqual(features["anomaly_logs-syslog_brief:linkDown_active(l)"], 15)
        self.assertEqual(features["anomaly_logs-syslog_message:peer/down: <*>"], 4)
        self.assertEqual(features["anomaly_logs-syslog_brief_flash_count"], 11)
        self.assertEqual(features["anomaly_logs-local_flash_count"], 0)
        self.assertEqual(features["anomaly_logs-remote_flash_count"], 0)
        self.assertEqual(missing["anomaly_logs-syslog_brief_flash_count"], 0)
        self.assertAlmostEqual(features["anomaly_kpi-cpu_utilization-max_alarming"], math.log1p(21))
        for key in ("memory_utilization", "temperature"):
            self.assertAlmostEqual(features[f"anomaly_kpi-{key}-mean_mix"], math.log1p(13))

        legacy = {"semantic_labels": {"alarm_time": alarm, "anomaly_kpi": {"cpu_utilization": kpi},
                                      "anomaly_logs": {"local_flash": {"PEER_STATE_CHG(l)": 7}}}}
        _, _, (features,) = extract_features_for_cases([legacy], selection)
        self.assertEqual(features["anomaly_logs-local_flash_count"], 7)
        self.assertAlmostEqual(features["anomaly_kpi-cpu_utilization-max_alarming"], math.log1p(21))


if __name__ == "__main__":
    unittest.main()
