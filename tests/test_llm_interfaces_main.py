import argparse
import json
import tempfile
import unittest
from pathlib import Path

from llm_interfaces.main import load_json_case, run, selected_jobs


def make_args(root, **overrides):
    values = {
        "mode": "both",
        "model_path": "fake-model",
        "competition_json": str(Path(root) / "competition.json"),
        "cooperation_json": str(Path(root) / "cooperation.json"),
        "competition_alarm_type": "competition-alarm",
        "cooperation_alarm_type": "cooperation-alarm",
        "output_dir": str(Path(root) / "output"),
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


class FakeProcess:
    def __init__(self, events, *, name, target, kwargs):
        self.events = events
        self.name = name
        self.target = target
        self.kwargs = kwargs
        self.exitcode = None

    def start(self):
        self.events.append(f"start:{self.name}")
        self.exitcode = 0

    def join(self):
        self.events.append(f"join:{self.name}")


class FakeContext:
    def __init__(self):
        self.events = []
        self.processes = []

    def Process(self, *, name, target, kwargs):
        process = FakeProcess(
            self.events,
            name=name,
            target=target,
            kwargs=kwargs,
        )
        self.processes.append(process)
        return process


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

    def test_two_modes_are_started_and_joined_serially(self):
        with tempfile.TemporaryDirectory() as root:
            for name in ("competition", "cooperation"):
                (Path(root) / f"{name}.json").write_text(
                    json.dumps({"semantic_labels": {"sop": ""}}),
                    encoding="utf-8",
                )
            args = make_args(root)
            context = FakeContext()

            run(args, process_context=context)

        self.assertEqual(context.events, [
            "start:competition_one_case",
            "join:competition_one_case",
            "start:cooperation_one_case",
            "join:cooperation_one_case",
        ])
        self.assertEqual(
            context.processes[0].kwargs["label"],
            {"semantic_labels": {"sop": ""}},
        )
        self.assertEqual(context.processes[0].kwargs["mode"], "competition")
        self.assertEqual(context.processes[1].kwargs["mode"], "cooperation")


if __name__ == "__main__":
    unittest.main()
