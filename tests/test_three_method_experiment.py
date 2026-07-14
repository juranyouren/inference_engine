import json
import os
import tempfile
import unittest

from inference.tree import build_tree_cot
from run_three_method_experiment import (
    build_arg_parser,
    calculate_metrics,
    evaluate_saved_results,
    extract_ranked_categories,
)


class ThreeMethodExperimentTests(unittest.TestCase):
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

    def test_extracts_last_candidate_json_ranking(self):
        text = '分析中出现 ["无关内容"]，最终答案```json\n["根因B", "根因A"]\n```'
        ranking, strategy = extract_ranked_categories(text, ["根因A", "根因B"])
        self.assertEqual(ranking, ["根因B", "根因A"])
        self.assertEqual(strategy, "json_array")

    def test_extracts_category_objects(self):
        text = json.dumps([{"category": "根因A"}, {"category": "根因B"}])
        ranking, _strategy = extract_ranked_categories(text, ["根因A", "根因B"])
        self.assertEqual(ranking, ["根因A", "根因B"])

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
