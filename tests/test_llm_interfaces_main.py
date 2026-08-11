import argparse
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

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
        "batch_size": 1,
        "max_rounds": 2,
        "tensor_parallel_size": None,
        "gpu_memory_utilization": 0.9,
        "max_model_len": 16384,
        "max_tokens": 4096,
        "prompt_safety_tokens": 512,
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

    def test_device_list_sets_environment_and_parallel_size(self):
        with tempfile.TemporaryDirectory() as root:
            args = make_args(root, ascend_devices="2,3", tensor_parallel_size=None)
            with patch.dict(os.environ, {}, clear=False):
                configure_ascend_devices(args)
                self.assertEqual(
                    os.environ["ASCEND_RT_VISIBLE_DEVICES"], "2,3"
                )
        self.assertEqual(args.tensor_parallel_size, 2)

    def test_parallel_size_must_match_visible_devices(self):
        with tempfile.TemporaryDirectory() as root:
            args = make_args(root, ascend_devices="0,1", tensor_parallel_size=1)
            with self.assertRaises(ValueError):
                configure_ascend_devices(args)

    @patch("llm_interfaces.main._infer_one_case")
    def test_two_modes_are_called_directly_in_order(self, infer_one_case):
        with tempfile.TemporaryDirectory() as root:
            for name in ("competition", "cooperation"):
                (Path(root) / f"{name}.json").write_text(
                    json.dumps({"semantic_labels": {"sop": ""}}),
                    encoding="utf-8",
                )
            args = make_args(root, ascend_devices="0")
            with patch.dict(os.environ, {}, clear=False):
                run(args)

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
            infer_one_case.call_args_list[0].kwargs["tensor_parallel_size"],
            1,
        )
        self.assertEqual(
            infer_one_case.call_args_list[0].kwargs["output_dir"],
            str((Path(root) / "output").resolve()),
        )


if __name__ == "__main__":
    unittest.main()
