import copy
import json
import unittest
from pathlib import Path

from llm_interfaces.sop_preprocess import compile_sop


TEMPLATES_DIR = Path(__file__).resolve().parents[1] / "llm_interfaces" / "sop_templates"


class SopPreprocessTests(unittest.TestCase):
    def setUp(self):
        self.sop = json.loads(
            (TEMPLATES_DIR / "network_device_offline.sop.json").read_text(encoding="utf-8")
        )

    def test_each_leaf_retains_all_paths_after_reconvergence(self):
        prompts = compile_sop(self.sop)["root_cause_prompts"]
        prefixes = (((1, False),), ((1, True), (2, False)))
        to_step_six = (((4, True), (5, True)), ((4, False),))
        expected = {
            "rc_1": {((1, True), (2, True))},
            "rc_2": {prefix + ((4, True), (5, False)) for prefix in prefixes},
        }
        for cause, tail in {
            "rc_3": ((6, True),),
            "rc_4": ((6, False), (9, True)),
            "rc_5": ((6, False), (9, False)),
        }.items():
            expected[cause] = {
                prefix + middle + tail for prefix in prefixes for middle in to_step_six
            }

        self.assertEqual(prompts.keys(), expected.keys())
        for cause, template in prompts.items():
            with self.subTest(cause=cause):
                paths = [
                    tuple(
                        (int(step["decision_id"].split("_")[1]), step["required_condition_met"])
                        for step in path
                    )
                    for path in template["paths"]
                ]
                self.assertCountEqual(paths, expected[cause])

    def test_invalid_graphs_are_rejected(self):
        for branch, target, message in [
            ("true", "step_01", "循环"),
            ("true", "unknown", "不存在"),
            ("false", "step_05", "不可达"),
        ]:
            with self.subTest(target=target):
                sop = copy.deepcopy(self.sop)
                sop["decisions"]["step_01"]["branches"][branch] = target
                if message == "不可达":
                    sop["decisions"]["step_02"]["branches"]["false"] = "step_05"
                with self.assertRaisesRegex(ValueError, message):
                    compile_sop(sop)

    def test_thresholds_preserve_the_original_sop_conditions(self):
        decisions = compile_sop(self.sop)["decision_templates"]
        self.assertIn("小于360000ms", decisions["step_05"]["condition"])
        self.assertIn("告警时间前后1分钟", decisions["step_09"]["condition"])
        self.assertIn("超过 100%", decisions["step_09"]["condition"])

    def test_result_templates_only_contain_fields_to_fill(self):
        templates = compile_sop(self.sop)
        self.assertEqual(
            templates.keys(), {"alarm_type", "decision_templates", "root_cause_prompts"}
        )
        for decision in templates["decision_templates"].values():
            self.assertEqual(
                decision["result_template"].keys(),
                {"diagnostic_clues", "condition_met", "confidence", "reason"},
            )
        for cause_id, prompt in templates["root_cause_prompts"].items():
            self.assertEqual(prompt["root_cause"], self.sop["root_causes"][cause_id])
            self.assertEqual(prompt["conclusion_template"].keys(), {"supported", "confidence", "reason"})

    def test_committed_templates_match_source(self):
        templates = json.loads(
            (TEMPLATES_DIR / "network_device_offline.templates.json").read_text(encoding="utf-8")
        )
        self.assertEqual(templates, compile_sop(self.sop))
