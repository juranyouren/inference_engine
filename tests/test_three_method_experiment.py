import json
import os
import tempfile
import unittest
from unittest.mock import patch

import config
from inference.selection import build_tree_summary_for_refiner
from inference.tree import build_tree_cot
from llm_inference.selector_refiner import LLMEngine
from run_three_method_experiment import (
    _llm_results_for_payload,
    _validate_scenarios,
    apply_test_mode,
    build_arg_parser,
    build_tree_stage_specs,
    calculate_metrics,
    configure_runtime_environment,
    ensure_valid_working_directory,
    evaluate_saved_results,
    extract_ranked_categories,
)


class ThreeMethodExperimentTests(unittest.TestCase):
    def test_recovers_from_deleted_working_directory_before_spawn(self):
        with patch(
            "run_three_method_experiment.os.getcwd",
            side_effect=FileNotFoundError,
        ), patch("run_three_method_experiment.os.chdir") as chdir:
            recovered = ensure_valid_working_directory()
        chdir.assert_called_once()
        self.assertTrue(recovered.is_absolute())

    def test_refiner_prompt_is_limited_with_head_and_tail_preserved(self):
        class FakeTokenizer:
            @staticmethod
            def encode(text, add_special_tokens=False):
                return [ord(char) for char in text]

            @staticmethod
            def decode(token_ids, skip_special_tokens=True):
                return "".join(chr(token_id) for token_id in token_ids)

        class FakeLlm:
            @staticmethod
            def get_tokenizer():
                return FakeTokenizer()

        engine = LLMEngine.__new__(LLMEngine)
        engine.llm = FakeLlm()
        engine.max_model_len = 80
        prompt, stats = engine.fit_prompt(
            "A" * 50 + "B" * 50,
            max_output_tokens=20,
            safety_tokens=10,
        )
        self.assertTrue(stats["truncated"])
        self.assertLessEqual(stats["final_tokens"], 50)
        self.assertTrue(prompt.startswith("A"))
        self.assertTrue(prompt.endswith("B"))

    def test_refiner_summary_compacts_cases_and_features(self):
        results = []
        for idx in range(30):
            results.append({
                "case_idx": idx,
                "is_correct": False,
                "features": {f"feature_{n}": n for n in range(100)},
                "cot": {"feature_99 > 1.00": 1},
            })
        summary = build_tree_summary_for_refiner({"summary": {}, "results": results})
        self.assertEqual(len(summary["wrong_cases"]), config.REFINER_WRONG_CASE_LIMIT)
        self.assertLessEqual(
            len(summary["wrong_cases"][0]["important_nonzero_features"]),
            config.REFINER_FEATURES_PER_CASE,
        )

    def test_configures_ascend_devices_before_model_initialization(self):
        previous = os.environ.pop("ASCEND_RT_VISIBLE_DEVICES", None)
        try:
            configure_runtime_environment()
            self.assertEqual(
                os.environ["ASCEND_RT_VISIBLE_DEVICES"],
                str(config.ASCEND_RT_VISIBLE_DEVICES),
            )
        finally:
            if previous is None:
                os.environ.pop("ASCEND_RT_VISIBLE_DEVICES", None)
            else:
                os.environ["ASCEND_RT_VISIBLE_DEVICES"] = previous

    def test_preserves_explicit_ascend_devices(self):
        previous = os.environ.get("ASCEND_RT_VISIBLE_DEVICES")
        try:
            os.environ["ASCEND_RT_VISIBLE_DEVICES"] = "0,1,2,3"
            configure_runtime_environment()
            self.assertEqual(
                os.environ["ASCEND_RT_VISIBLE_DEVICES"],
                "0,1,2,3",
            )
        finally:
            if previous is None:
                os.environ.pop("ASCEND_RT_VISIBLE_DEVICES", None)
            else:
                os.environ["ASCEND_RT_VISIBLE_DEVICES"] = previous

    def test_experiment_selects_only_indices_51_through_250(self):
        with tempfile.TemporaryDirectory() as root:
            scenario_name = "告警A"
            scenario_dir = os.path.join(root, scenario_name)
            os.makedirs(scenario_dir)
            for idx in [1, *range(51, 251), 300]:
                path = os.path.join(
                    scenario_dir,
                    f"{scenario_name}_{idx}_label_test.json",
                )
                with open(path, "w", encoding="utf-8") as f:
                    f.write("{}")

            scenarios = _validate_scenarios(root, "all", 1, 200, 51, 250)
            self.assertEqual(scenarios[0]["indices"], list(range(51, 251)))

    def test_tree_cot_records_root_to_leaf_conditions(self):
        class FakeTree:
            children_left = [1, -1, -1]
            children_right = [2, -1, -1]
            feature = [0, -2, -2]
            threshold = [1.5, -2.0, -2.0]

        class FakeClassifier:
            tree_ = FakeTree()

        cot = build_tree_cot(FakeClassifier(), ["feature_a"], [3.0], 1)

        self.assertEqual(len(cot), 1)
        path, leaf_class = next(iter(cot.items()))
        self.assertIn("feature_a >", path)
        self.assertEqual(leaf_class, 1)

    def test_tree_experiment_enables_selector_refiner_by_default(self):
        args = build_arg_parser().parse_args([])
        self.assertEqual(args.selection_source, "selector_refiner")
        self.assertGreater(args.refiner_rounds, 0)
        self.assertEqual(args.start_index, 51)
        self.assertEqual(args.end_index, 250)
        self.assertFalse(hasattr(args, "agentdigest_label_root"))

    def test_test_mode_limits_each_category_to_five_cases(self):
        parser = build_arg_parser()
        args = parser.parse_args(["--test"])
        default_output_dir = parser.get_default("output_dir")
        apply_test_mode(args, default_output_dir)

        self.assertEqual(args.cases_per_category, 5)
        self.assertEqual(args.start_index, 51)
        self.assertEqual(args.end_index, 55)
        self.assertEqual(args.output_dir, os.path.join(default_output_dir, "test_5"))

    def test_tree_test_mode_trains_on_previous_fifty_and_infers_five(self):
        with tempfile.TemporaryDirectory() as root:
            scenario_name = "告警A"
            for idx in range(1, 56):
                with open(
                    os.path.join(root, f"{scenario_name}_{idx}_label_test.json"),
                    "w",
                    encoding="utf-8",
                ) as f:
                    f.write("{}")

            parser = build_arg_parser()
            args = parser.parse_args(["--test"])
            apply_test_mode(args, parser.get_default("output_dir"))
            specs = build_tree_stage_specs({
                "name": scenario_name,
                "alarm_type": scenario_name,
                "data_dir": root,
                "indices": list(range(51, 56)),
            }, args)

        self.assertEqual(specs, [(1, list(range(1, 51)), list(range(51, 56)))])

    def test_extracts_last_candidate_json_ranking(self):
        text = '分析中出现 ["无关内容"]，最终答案```json\n["根因B", "根因A"]\n```'
        ranking, strategy = extract_ranked_categories(text, ["根因A", "根因B"])
        self.assertEqual(ranking, ["根因B", "根因A"])
        self.assertEqual(strategy, "structured_array")

    def test_extracts_category_objects(self):
        text = json.dumps([{"category": "根因A"}, {"category": "根因B"}])
        ranking, _strategy = extract_ranked_categories(text, ["根因A", "根因B"])
        self.assertEqual(ranking, ["根因A", "根因B"])

    def test_initial_experiment_uses_last_cooperation_meta(self):
        with tempfile.TemporaryDirectory() as root:
            scenario_name = "告警A"
            data_dir = os.path.join(root, "labels")
            output_dir = os.path.join(root, "outputs")
            os.makedirs(data_dir)

            label_path = os.path.join(
                data_dir,
                f"{scenario_name}_51_label_test.json",
            )
            with open(label_path, "w", encoding="utf-8") as f:
                json.dump({
                    "root_cause": {"category": "根因B"},
                    "root_cause_candidates": [
                        {"category": "根因A"},
                        {"category": "根因B"},
                    ],
                }, f, ensure_ascii=False)

            meta_dir = os.path.join(output_dir, "51", "round2", "meta")
            reasoner_dir = os.path.join(output_dir, "51", "round2", "reasoner")
            os.makedirs(meta_dir)
            os.makedirs(reasoner_dir)
            with open(
                os.path.join(meta_dir, "raw_responses.txt"),
                "w",
                encoding="utf-8",
            ) as f:
                f.write('["根因B", "根因A"]')
            with open(
                os.path.join(reasoner_dir, "raw_responses.txt"),
                "w",
                encoding="utf-8",
            ) as f:
                f.write('["根因A", "根因B"]')

            results = _llm_results_for_payload(
                {
                    "meta": {"output_dir": output_dir},
                    "summary": {"processed_indices": [51]},
                },
                {
                    "name": scenario_name,
                    "alarm_type": scenario_name,
                    "data_dir": data_dir,
                },
                "cooperation",
            )

            self.assertEqual(results[0]["pred_rc"], ["根因B", "根因A"])
            self.assertIn("cooperation_meta:round2", results[0]["parse_strategy"])
            self.assertTrue(results[0]["parse_attempts"])

    def test_metrics_count_missing_prediction_as_miss(self):
        results = [
            {"groundtruth": "A", "rank": 1},
            {"groundtruth": "B", "rank": 3},
            {"groundtruth": "C", "rank": None, "parse_strategy": "unparsed"},
        ]
        metrics = calculate_metrics(results, elapsed_seconds=6.0)
        self.assertAlmostEqual(metrics["top1"], 1 / 3)
        self.assertAlmostEqual(metrics["top3"], 2 / 3)
        self.assertAlmostEqual(metrics["top5"], 2 / 3)
        self.assertAlmostEqual(metrics["mrr"], (1 + 1 / 3) / 3)
        self.assertEqual(metrics["average_seconds_per_case"], 2.0)
        self.assertEqual(metrics["unparsed_count"], 1)

    def test_evaluate_saved_results_writes_json_and_csv(self):
        with tempfile.TemporaryDirectory() as output_dir:
            method_dir = os.path.join(output_dir, "tree")
            os.makedirs(method_dir)
            payload = {
                "meta": {
                    "runs": [{
                        "scenario_name": "告警A",
                        "elapsed_seconds": 2.0,
                    }],
                },
                "results": [{
                    "scenario_name": "告警A",
                    "groundtruth": "根因A",
                    "rank": 1,
                }],
            }
            with open(
                os.path.join(method_dir, "predictions.json"),
                "w",
                encoding="utf-8",
            ) as f:
                json.dump(payload, f, ensure_ascii=False)

            report = evaluate_saved_results(output_dir, ["tree"])
            self.assertEqual(len(report["metrics"]), 2)
            self.assertTrue(os.path.exists(
                os.path.join(output_dir, "evaluation", "metrics.json"),
            ))
            self.assertTrue(os.path.exists(
                os.path.join(output_dir, "evaluation", "metrics.csv"),
            ))


if __name__ == "__main__":
    unittest.main()
