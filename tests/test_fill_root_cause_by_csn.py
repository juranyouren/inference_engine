import json
import os
import tempfile
import unittest

from pathlib import Path

from fill_root_cause_by_csn import (
    extract_csn,
    extract_csn_from_filename,
    fill_root_causes,
)


def write_json(path, data):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False)


class FillRootCauseByCsnTests(unittest.TestCase):
    def test_extracts_csn_from_indexed_label_filename(self):
        self.assertEqual(
            extract_csn_from_filename(Path("告警类型_51_label_csn_part_001.json")),
            "csn_part_001",
        )

    def test_extracts_nested_case_insensitive_csn(self):
        self.assertEqual(
            extract_csn({"semantic_labels": {"CSN": "  abc-1 "}}),
            "abc-1",
        )

    def test_dry_run_then_apply_copies_complete_root_cause(self):
        with tempfile.TemporaryDirectory() as work_dir:
            source_root = os.path.join(work_dir, "source")
            target_root = os.path.join(work_dir, "target")
            source_file = os.path.join(
                source_root,
                "告警A",
                "告警A_51_label_csn-1.json",
            )
            target_file = os.path.join(
                target_root,
                "告警A",
                "告警A_99_label_csn-1.json",
            )
            root_cause = {"category": "根因A", "description": "完整对象"}
            write_json(source_file, {"root_cause": root_cause})
            write_json(target_file, {"semantic_labels": {}})

            dry_run = fill_root_causes(target_root, source_root)
            self.assertEqual(dry_run["summary"]["matched_count"], 1)
            self.assertEqual(dry_run["summary"]["updated_count"], 0)
            with open(target_file, "r", encoding="utf-8") as f:
                self.assertNotIn("root_cause", json.load(f))

            applied = fill_root_causes(target_root, source_root, apply_changes=True)
            self.assertEqual(applied["summary"]["updated_count"], 1)
            with open(target_file, "r", encoding="utf-8") as f:
                self.assertEqual(json.load(f)["root_cause"], root_cause)

    def test_conflicting_source_root_causes_are_not_written(self):
        with tempfile.TemporaryDirectory() as work_dir:
            source_root = os.path.join(work_dir, "source")
            target_root = os.path.join(work_dir, "target")
            write_json(
                os.path.join(source_root, "告警A", "first_label.json"),
                {"csn": "same", "root_cause": {"category": "根因A"}},
            )
            write_json(
                os.path.join(source_root, "告警A", "second_label.json"),
                {"csn": "same", "root_cause": {"category": "根因B"}},
            )
            target_file = os.path.join(target_root, "告警A", "target_label.json")
            write_json(target_file, {"csn": "same"})

            report = fill_root_causes(
                target_root,
                source_root,
                apply_changes=True,
            )
            self.assertEqual(report["summary"]["conflict_count"], 1)
            with open(target_file, "r", encoding="utf-8") as f:
                self.assertNotIn("root_cause", json.load(f))


if __name__ == "__main__":
    unittest.main()
