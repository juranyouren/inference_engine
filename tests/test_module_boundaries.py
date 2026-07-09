import importlib
import unittest


class ModuleBoundaryTests(unittest.TestCase):
    def test_refactored_modules_are_importable(self):
        for module_name in [
            "inference_common",
            "selection_pipeline",
            "tree_pipeline",
            "llm_pipeline",
            "infer_by_index",
        ]:
            importlib.import_module(module_name)

    def test_cli_entrypoint_keeps_public_pipeline_functions(self):
        infer_by_index = importlib.import_module("infer_by_index")

        self.assertTrue(callable(infer_by_index.build_arg_parser))
        self.assertTrue(callable(infer_by_index.tree_infer))
        self.assertTrue(callable(infer_by_index.llm_infer))
        self.assertTrue(callable(infer_by_index.run_all_in_separate_processes))


if __name__ == "__main__":
    unittest.main()
