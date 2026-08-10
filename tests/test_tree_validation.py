import unittest
from unittest.mock import patch

from infer_by_index import build_arg_parser as build_infer_parser
from inference.tree import _TreeValCaseDataLoader
from run_three_method_experiment import (
    build_arg_parser as build_experiment_parser,
)
from rule_inferencer.data_process_val import (
    calculate_root_cause_score,
    rerank_root_cause_results,
)


class TreeValidationTests(unittest.TestCase):
    def test_infer_cli_tree_val_is_optional(self):
        parser = build_infer_parser()
        self.assertFalse(parser.parse_args([]).tree_val)
        args = parser.parse_args([
            "--tree-val",
            "--tree-val-rule-root",
            "/rules",
            "--tree-val-template-dir",
            "/templates",
            "--tree-val-kpi-mode",
            "alarm",
            "--tree-val-neutral-score",
            "0.25",
        ])
        self.assertTrue(args.tree_val)
        self.assertEqual(args.tree_val_rule_root, "/rules")
        self.assertEqual(args.tree_val_template_dir, "/templates")
        self.assertEqual(args.tree_val_kpi_mode, "alarm")
        self.assertEqual(args.tree_val_neutral_score, 0.25)

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

    def test_case_loader_adapter_returns_only_loaded_range(self):
        cases = [
            {"case_idx": 2},
            {"case_idx": 5},
            {"case_idx": 9},
        ]
        adapter = _TreeValCaseDataLoader(cases, [2, 5, 9])
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


if __name__ == "__main__":
    unittest.main()
