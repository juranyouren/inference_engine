import argparse
import json
import os
import tempfile
import types
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from llm_interfaces.main import (
    configure_ascend_devices,
    load_json_case,
    run,
    selected_jobs,
)


def make_args(root, **overrides):
    values = {
        "mode": "both",
        "model_path": "fake-model",
        "competition_json": str(Path(root) / "competition.json"),
        "cooperation_json": str(Path(root) / "cooperation.json"),
        "competition_alarm_type": "competition-alarm",
        "cooperation_alarm_type": "cooperation-alarm",
        "output_dir": str(Path(root) / "output"),
        "ascend_devices": None,
        "gpu_memory_utilization": 0.9,
        "max_model_len": 16384,
    }
    values.update(overrides)
    return argparse.Namespace(**values)


class LlmInterfacesMainTests(unittest.TestCase):
    def test_load_json_case_requires_one_object(self):
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / "case.json"
            path.write_text("[]", encoding="utf-8")
            with self.assertRaises(TypeError):
                load_json_case(path)

    def test_selected_jobs_requires_mode_input(self):
        with tempfile.TemporaryDirectory() as root:
            args = make_args(root, mode="competition", competition_json=None)
            with self.assertRaises(ValueError):
                selected_jobs(args)

    def test_selected_jobs_requires_alarm_type(self):
        with tempfile.TemporaryDirectory() as root:
            args = make_args(
                root,
                mode="competition",
                competition_alarm_type=None,
            )
            with self.assertRaises(ValueError):
                selected_jobs(args)

    def test_device_list_sets_environment_and_parallel_size(self):
        with tempfile.TemporaryDirectory() as root:
            args = make_args(root, ascend_devices="2,3")
            with patch.dict(os.environ, {}, clear=False):
                configure_ascend_devices(args)
                self.assertEqual(
                    os.environ["ASCEND_RT_VISIBLE_DEVICES"], "2,3"
                )
    @patch("llm_interfaces.main._infer_one_case")
    def test_two_modes_share_engine_and_are_called_in_order(self, infer_one_case):
        with tempfile.TemporaryDirectory() as root:
            for name in ("competition", "cooperation"):
                (Path(root) / f"{name}.json").write_text(
                    json.dumps({"semantic_labels": {"sop": ""}}),
                    encoding="utf-8",
                )
            args = make_args(root, ascend_devices="0")
            fake_engine = object()
            competition = object()
            cooperation = object()
            get_instance = Mock(return_value=fake_engine)
            fake_llm_engine_module = types.ModuleType(
                "llm_interfaces.llm_engine"
            )
            fake_llm_engine_module.LLMEngine = SimpleNamespace(
                get_instance=get_instance,
            )
            with (
                patch.dict(os.environ, {}, clear=False),
                patch.dict(
                    "sys.modules",
                    {"llm_interfaces.llm_engine": fake_llm_engine_module},
                ),
                patch(
                    "llm_interfaces.competition.CompetitionInference",
                    return_value=competition,
                ) as competition_class,
                patch(
                    "llm_interfaces.cooperation.CooperationInference",
                    return_value=cooperation,
                ) as cooperation_class,
            ):
                run(args)

        fake_llm_engine_module.LLMEngine.get_instance.assert_called_once_with(
            model_path="fake-model",
            gpu_memory_utilization=0.9,
            max_model_len=16384,
        )
        competition_class.assert_called_once_with(fake_engine)
        cooperation_class.assert_called_once_with(fake_engine)
        self.assertEqual(infer_one_case.call_count, 2)
        self.assertEqual(
            [item.kwargs["mode"] for item in infer_one_case.call_args_list],
            ["competition", "cooperation"],
        )
        self.assertEqual(
            infer_one_case.call_args_list[0].kwargs["label"],
            {"semantic_labels": {"sop": ""}},
        )
        self.assertEqual(
            [item.kwargs["inference"] for item in infer_one_case.call_args_list],
            [competition, cooperation],
        )
        self.assertEqual(
            infer_one_case.call_args_list[0].kwargs["output_dir"],
            str((Path(root) / "output").resolve()),
        )


if __name__ == "__main__":
    unittest.main()
