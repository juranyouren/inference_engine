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

    def test_cooperation_uses_only_last_round_meta(self):
        """Cooperation 只解析最后一轮 Meta，不管 Reasoner。"""
        with tempfile.TemporaryDirectory() as root:
            output_dir = Path(root)
            # round2 Reasoner 有合法排序，但应该被忽略
            write_text(
                output_dir / "51" / "round2" / "reasoner" / "raw_responses.txt",
                '["根因X", "根因Y"]',
            )
            write_text(
                output_dir / "51" / "round2" / "meta" / "raw_responses.txt",
                '["根因B", "根因A"]',
            )
            # round1 也有 Meta，但不会被用到（只取最后一轮）
            write_text(
                output_dir / "51" / "round1" / "meta" / "raw_responses.txt",
                '["根因C", "根因D"]',
            )
            ranking, strategy, attempts = parse_cooperation_case(
                output_dir,
                51,
                ["根因A", "根因B"],
            )
            self.assertEqual(ranking, ["根因B", "根因A"])
            self.assertIn("cooperation_meta:round2", strategy)
            self.assertTrue(attempts)

    def test_competition_only_uses_reasoner_ignores_meta(self):
        """Competition 只解析 Reasoner，忽略 Meta。"""
        with tempfile.TemporaryDirectory() as root:
            output_dir = Path(root)
            alarm_type = "告警A"
            # Reasoner 有合法排序
            write_text(
                output_dir / f"{alarm_type}_analysis_Reasoner_51" / "raw_responses.txt",
                '["A", "B", "C"]',
            )
            # Meta 即使有不同排序也应该被忽略
            write_text(
                output_dir / f"{alarm_type}_analysis_Meta_51" / "raw_responses.txt",
                '["C", "B", "A"]',
            )
            ranking, strategy, _attempts = parse_competition_case(
                output_dir,
                alarm_type,
                51,
                ["A", "B", "C"],
            )
            self.assertEqual(ranking, ["A", "B", "C"])
            self.assertIn("competition_reasoner:structured_array", strategy)

    def test_competition_unparsed_when_reasoner_empty(self):
        """Competition 的 Reasoner 解析失败则不回退到 Meta，直接 unparsed。"""
        with tempfile.TemporaryDirectory() as root:
            output_dir = Path(root)
            alarm_type = "告警A"
            write_text(
                output_dir / f"{alarm_type}_analysis_Reasoner_51" / "raw_responses.txt",
                "没有排序内容",
            )
            # Meta 有合法排序，但应该被忽略
            write_text(
                output_dir / f"{alarm_type}_analysis_Meta_51" / "raw_responses.txt",
                '["A", "B", "C"]',
            )
            ranking, strategy, _attempts = parse_competition_case(
                output_dir,
                alarm_type,
                51,
                ["A", "B", "C"],
            )
            self.assertEqual(ranking, [])
            self.assertEqual(strategy, "unparsed")


if __name__ == "__main__":
    unittest.main()
