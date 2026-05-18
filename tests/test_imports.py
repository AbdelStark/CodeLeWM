from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class PackageImportBoundaryTest(unittest.TestCase):
    def test_lightweight_package_namespaces_import(self) -> None:
        import codelewm
        import codelewm.eval
        import codelewm.harness
        import codelewm.model
        import codelewm.observability
        import codelewm.training

        self.assertEqual(codelewm.__version__, "0.0.0")
        self.assertIn("ActionViewReportPolicy", codelewm.eval.__all__)
        self.assertIn("RetrievalReport", codelewm.eval.__all__)
        self.assertIn("compute_retrieval_metrics", codelewm.eval.__all__)
        self.assertIn("HardNegativeSamplerConfig", codelewm.eval.__all__)
        self.assertIn("sample_hard_negatives", codelewm.eval.__all__)
        self.assertIn("lexical_baseline_ranks", codelewm.eval.__all__)
        self.assertIn("validate_required_headline_baselines", codelewm.eval.__all__)
        self.assertTrue(hasattr(codelewm.harness, "main"))
        self.assertIn("CodeTransitionModel", codelewm.model.__all__)
        self.assertIn("transition_energy", codelewm.model.__all__)
        self.assertIn("ArtifactManifest", codelewm.observability.__all__)
        self.assertIn("TrainConfig", codelewm.training.__all__)
        self.assertIn("load_train_config", codelewm.training.__all__)
        self.assertIn("train", codelewm.training.__all__)
        self.assertIn("train_cpu_smoke", codelewm.training.__all__)

    def test_moved_seed_modules_have_package_specs(self) -> None:
        expected = [
            "codelewm.model.jepa",
            "codelewm.model.modules",
            "codelewm.training.utils",
        ]

        for module_name in expected:
            with self.subTest(module=module_name):
                self.assertIsNotNone(importlib.util.find_spec(module_name))

    def test_root_wrappers_preserve_legacy_module_names(self) -> None:
        wrappers = {
            "jepa.py": "from codelewm.model.jepa import *",
            "module.py": "from codelewm.model.modules import *",
            "utils.py": "from codelewm.training.utils import *",
        }

        for filename, expected_import in wrappers.items():
            with self.subTest(wrapper=filename):
                source = (ROOT / filename).read_text()
                compile(source, filename, "exec")
                self.assertIn(expected_import, source)


if __name__ == "__main__":
    unittest.main()
