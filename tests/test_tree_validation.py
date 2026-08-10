import unittest
from unittest.mock import patch

from rerank_llm_results import (
    _normalize_result,
    build_arg_parser as build_llm_rerank_parser,
)
from run_three_method_experiment import (
    build_arg_parser as build_experiment_parser,
)
from inference.validation import (
    CaseRangeAdapter,
    rerank_results_with_val,
)
from rule_inferencer.data_process_val import (
    calculate_root_cause_score,
    rerank_root_cause_results,
)


class TreeValidationTests(unittest.TestCase):
    def test_experiment_cli_accepts_tree_val_options(self):
        args = build_experiment_parser().parse_args([
            "--methods",
            "tree",
            "--tree-val",
            "--tree-val-rule-root",
            "/rules",
        ])
        self.assertTrue(args.tree_val)
        self.assertEqual(args.tree_val_rule_root, "/rules")

    def test_llm_rerank_cli_loads_existing_result_directory(self):
        args = build_llm_rerank_parser().parse_args([
            "--experiment-dir",
            "/experiment",
            "--methods",
            "competition",
            "--tree-val-target-size",
            "7",
        ])
        self.assertEqual(args.experiment_dir, "/experiment")
        self.assertEqual(args.methods, ["competition"])
        self.assertEqual(args.tree_val_target_size, 7)

    def test_case_loader_adapter_returns_only_loaded_range(self):
        cases = [
            {"case_idx": 2},
            {"case_idx": 5},
            {"case_idx": 9},
        ]
        adapter = CaseRangeAdapter(cases, [2, 5, 9])
        selected, indices = adapter.get_cases_by_id_range(3, 10)
        self.assertEqual(indices, [5, 9])
        self.assertEqual(selected, [cases[1], cases[2]])

    def test_score_uses_weighted_matches_and_coverage(self):
        detail = calculate_root_cause_score(
            features={"feature_a": 1, "feature_b": 0},
            root_cause="rc-a",
            rule_config={
                "root_causes": {
                    "rc-a": [
                        {
                            "feature": "feature_a",
                            "relation": "eq",
                            "threshold": 1,
                            "weight": 2,
                        },
                        {
                            "feature": "feature_b",
                            "relation": "eq",
                            "threshold": 1,
                            "weight": 1,
                        },
                    ],
                },
            },
        )
        self.assertEqual(detail["score"], 2 / 3)
        self.assertEqual(detail["coverage"], 1.0)
        self.assertFalse(detail["skip_case"])

    @patch("rule_inferencer.data_process_val.extract_features_from_case_range")
    @patch("rule_inferencer.data_process_val.complete_pred_rc_before_rerank")
    def test_rerank_keeps_candidates_and_stable_tie_order(
        self,
        complete_pred_rc,
        extract_features,
    ):
        ret = {
            "results": [{
                "case_idx": 5,
                "alarm_type": "scenario",
                "rc": "rc-b",
                "pred_rc": ["rc-a", "rc-b"],
            }],
        }
        complete_pred_rc.side_effect = lambda ret, **_kwargs: ret
        extract_features.return_value = (
            ["feature_a"],
            [[1]],
            [5],
            {5: {"feature_a": 1}},
        )
        rules = {
            "scenario": {
                "root_causes": {
                    "rc-a": [{
                        "feature": "feature_a",
                        "relation": "eq",
                        "threshold": 1,
                    }],
                    "rc-b": [{
                        "feature": "feature_a",
                        "relation": "eq",
                        "threshold": 1,
                    }],
                },
            },
        }

        output = rerank_root_cause_results(
            ret=ret,
            dataloader=object(),
            rule_configs=rules,
            template_dir="/unused",
        )
        self.assertEqual(
            extract_features.call_args.kwargs["end_id"],
            6,
        )
        result = output["results"][0]
        self.assertEqual(result["pred_rc_val"], ["rc-a", "rc-b"])
        self.assertFalse(result["rerank_changed"])
        self.assertFalse(result["pred_top1_changed"])
        self.assertTrue(result["is_correct_val"] is False)

    @patch("rule_inferencer.data_process_val.rerank_root_cause_results")
    @patch("inference.validation.load_rule_config")
    def test_shared_wrapper_preserves_raw_prediction(
        self,
        load_rules,
        rerank,
    ):
        load_rules.return_value = ("scenario", {"root_causes": {}}, "/rules/scenario.json")

        def fake_rerank(ret, **_kwargs):
            item = ret["results"][0]
            item.update({
                "pred_rc_val": ["rc-b", "rc-a"],
                "pred_top1_rc_val": "rc-b",
                "rerank_changed": True,
                "pred_top1_changed": True,
                "is_correct_val": True,
            })
            return ret

        rerank.side_effect = fake_rerank
        raw = [{
            "case_idx": 1,
            "alarm_type": "scenario",
            "groundtruth": "rc-b",
            "pred_rc": ["rc-a", "rc-b"],
            "rank": 2,
        }]
        output, metadata = rerank_results_with_val(
            scenario={"name": "scenario", "alarm_type": "scenario"},
            cases=[{"semantic_labels": {}}],
            indices=[1],
            results=raw,
            template_dir="/templates",
        )
        self.assertEqual(output[0]["pred_rc"], ["rc-a", "rc-b"])
        self.assertEqual(output[0]["rank"], 2)
        self.assertEqual(output[0]["pred_rc_val"], ["rc-b", "rc-a"])
        self.assertEqual(output[0]["rank_val"], 1)
        self.assertTrue(metadata["enabled"])

    def test_llm_result_mapping_creates_rc_field(self):
        item = _normalize_result(
            {
                "case_idx": 3,
                "groundtruth": "rc-b",
                "pred_rc": "['rc-a', 'rc-b']",
            },
            {"name": "scenario", "alarm_type": "scenario"},
        )
        self.assertEqual(item["rc"], "rc-b")
        self.assertEqual(item["pred_rc"], ["rc-a", "rc-b"])
        self.assertEqual(item["rank"], 2)


if __name__ == "__main__":
    unittest.main()
