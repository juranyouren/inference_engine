import unittest

from rerun_failed_llm_cases import merge_results, select_failed_results


class RerunFailedLlmCasesTests(unittest.TestCase):
    def test_selects_only_unparsed_by_default(self):
        results = [
            {"case_idx": 1, "pred_rc": [], "parse_strategy": "unparsed"},
            {"case_idx": 2, "pred_rc": ["A"], "parse_strategy": "structured_array"},
            {"case_idx": 3, "pred_rc": ["B"], "parse_strategy": "candidate_text_fallback"},
        ]
        self.assertEqual(
            [item["case_idx"] for item in select_failed_results(results)],
            [1],
        )
        self.assertEqual(
            [item["case_idx"] for item in select_failed_results(results, True)],
            [1, 3],
        )

    def test_merge_replaces_only_matching_scenario_and_index(self):
        baseline = [
            {"scenario_name": "A", "case_idx": 51, "pred_rc": []},
            {"scenario_name": "A", "case_idx": 52, "pred_rc": ["old"]},
        ]
        replacement = [
            {"scenario_name": "A", "case_idx": 51, "pred_rc": ["new"]},
        ]
        merged = merge_results(baseline, replacement)
        self.assertEqual(merged[0]["pred_rc"], ["new"])
        self.assertEqual(merged[1]["pred_rc"], ["old"])


if __name__ == "__main__":
    unittest.main()
