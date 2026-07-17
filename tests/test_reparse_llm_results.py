import json
import os
import tempfile
import unittest
from pathlib import Path

from reparse_and_score_llm_results import (
    extract_structured_rankings,
    parse_competition_case,
    parse_cooperation_case,
)


def write_text(path: Path, text: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


class ReparseLlmResultsTests(unittest.TestCase):
    def test_extracts_json_and_single_quote_rankings(self):
        text = '$$ ["A", "B"] $$\n最终：[\'B\', \'A\']'
        self.assertEqual(
            extract_structured_rankings(text, ["A", "B"]),
            [["A", "B"], ["B", "A"]],
        )

    def test_cooperation_uses_final_round_meta_when_reasoner_has_no_ranking(self):
        with tempfile.TemporaryDirectory() as root:
            output_dir = Path(root)
            write_text(
                output_dir / "51" / "round2" / "reasoner" / "raw_responses.txt",
                "我检查了接口状态，但这里没有根因排序。",
            )
            write_text(
                output_dir / "51" / "round2" / "meta" / "raw_responses.txt",
                '$$ ["继续检查"], ["根因B", "根因A"] $$',
            )
            ranking, strategy, attempts = parse_cooperation_case(
                output_dir,
                51,
                ["根因A", "根因B"],
            )
            self.assertEqual(ranking, ["根因B", "根因A"])
            self.assertIn("cooperation_meta:round2", strategy)
            self.assertTrue(attempts)

    def test_competition_uses_reasoner_consensus_when_verifier_is_invalid(self):
        with tempfile.TemporaryDirectory() as root:
            output_dir = Path(root)
            alarm_type = "告警A"
            write_text(
                output_dir / f"{alarm_type}_analysis_Reasoner_51" / "raw_responses.txt",
                "Verifier没有给出列表",
            )
            write_text(
                output_dir / f"{alarm_type}_analysis_Meta_51" / "raw_responses.txt",
                '["A", "B", "C"]\n["A", "C", "B"]\n["B", "A", "C"]',
            )
            ranking, strategy, _attempts = parse_competition_case(
                output_dir,
                alarm_type,
                51,
                ["A", "B", "C"],
            )
            self.assertEqual(ranking[0], "A")
            self.assertIn("competition_reasoners", strategy)


if __name__ == "__main__":
    unittest.main()
