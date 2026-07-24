import unittest
from unittest.mock import patch

from infer_by_index import build_arg_parser as build_infer_parser
from inference.tree import (
    normalize_tree_val_depths,
    select_tree_model_by_validation,
)
from run_three_method_experiment import (
    build_arg_parser as build_experiment_parser,
)


class TreeValidationTests(unittest.TestCase):
    def test_infer_cli_tree_val_is_optional(self):
        parser = build_infer_parser()
        self.assertFalse(parser.parse_args([]).tree_val)
        self.assertTrue(parser.parse_args(["--tree-val"]).tree_val)
        self.assertFalse(
            parser.parse_args(["--no-tree-val"]).tree_val
        )

    def test_experiment_cli_accepts_tree_val_depths(self):
        args = build_experiment_parser().parse_args([
            "--methods",
            "tree",
            "--tree-val",
            "--tree-val-depths",
            "2",
            "4",
        ])
        self.assertTrue(args.tree_val)
        self.assertEqual(args.tree_val_depths, [2, 4])

    def test_depths_are_positive_and_deduplicated(self):
        self.assertEqual(normalize_tree_val_depths([3, 2, 3]), [3, 2])
        with self.assertRaises(ValueError):
            normalize_tree_val_depths([0, 3])

    @patch("inference.tree.predict_by_tree")
    @patch("inference.tree.train_tree_model")
    def test_validation_selects_highest_accuracy_and_keeps_first_tie(
        self,
        train_tree_model,
        predict_by_tree,
    ):
        train_tree_model.side_effect = lambda _cases, _selection, max_depth: (
            f"model-{max_depth}",
            ["feature"],
            f"encoder-{max_depth}",
        )

        accuracy_by_depth = {
            "model-2": [True, False],
            "model-3": [True, True],
            "model-4": [True, True],
        }

        def fake_predict(
            _cases,
            _indices,
            clf,
            _feature_names,
            _label_encoder,
            _selection,
        ):
            return [
                {"is_correct": value}
                for value in accuracy_by_depth[clf]
            ]

        predict_by_tree.side_effect = fake_predict

        clf, _names, _encoder, report = (
            select_tree_model_by_validation(
                train_cases=[{"root_cause": "a"}],
                validation_cases=[
                    {"root_cause": "a"},
                    {"root_cause": "b"},
                ],
                validation_indices=[1, 2],
                selection={},
                depth_candidates=[2, 3, 4],
            )
        )

        self.assertEqual(clf, "model-3")
        self.assertEqual(report["selected_max_depth"], 3)
        self.assertEqual(report["selected_accuracy"], 1.0)
        self.assertEqual(len(report["scores"]), 3)


if __name__ == "__main__":
    unittest.main()
