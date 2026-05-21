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
        import codelewm.security
        import codelewm.training

        self.assertEqual(codelewm.__version__, "0.0.0")
        self.assertIn("ActionViewReportPolicy", codelewm.eval.__all__)
        self.assertIn("RetrievalReport", codelewm.eval.__all__)
        self.assertIn("DownstreamRerankBenchmark", codelewm.eval.__all__)
        self.assertIn("build_downstream_rerank_claim_gate", codelewm.eval.__all__)
        self.assertIn("build_downstream_benchmark_pack", codelewm.eval.__all__)
        self.assertIn("read_downstream_rerank_benchmark", codelewm.eval.__all__)
        self.assertIn("compute_retrieval_metrics", codelewm.eval.__all__)
        self.assertIn("HardNegativeSamplerConfig", codelewm.eval.__all__)
        self.assertIn("sample_hard_negatives", codelewm.eval.__all__)
        self.assertIn("lexical_baseline_ranks", codelewm.eval.__all__)
        self.assertIn("validate_required_headline_baselines", codelewm.eval.__all__)
        self.assertTrue(hasattr(codelewm.harness, "main"))
        self.assertIn("ScoreResult", codelewm.harness.__all__)
        self.assertIn("RerankResult", codelewm.harness.__all__)
        self.assertIn("ErrorReport", codelewm.harness.__all__)
        self.assertIn("CandidatePackArtifactResult", codelewm.harness.__all__)
        self.assertIn("capture_candidate_pack", codelewm.harness.__all__)
        self.assertIn("llm_candidate_pack_json_schema", codelewm.harness.__all__)
        self.assertIn("run_llm_world_model_demo", codelewm.harness.__all__)
        self.assertIn("read_llm_world_model_demo_report", codelewm.harness.__all__)
        self.assertIn("write_candidate_pack_artifact", codelewm.harness.__all__)
        self.assertIn("score_result_json_schema", codelewm.harness.__all__)
        self.assertIn("validate_rerank_result_payload", codelewm.harness.__all__)
        self.assertIn("load_scorer", codelewm.harness.__all__)
        self.assertIn("CodeTransitionModel", codelewm.model.__all__)
        self.assertIn("CodeStateEncoder", codelewm.model.__all__)
        self.assertIn("TorchCodeTransitionModel", codelewm.model.__all__)
        self.assertIn("transition_energy", codelewm.model.__all__)
        self.assertIn("ArtifactManifest", codelewm.observability.__all__)
        self.assertIn("LogEvent", codelewm.observability.__all__)
        self.assertIn("write_log_event_jsonl", codelewm.observability.__all__)
        self.assertIn("parse_python_source_text", codelewm.security.__all__)
        self.assertIn("reject_code_execution_config", codelewm.security.__all__)
        self.assertIn("PublicLicenseGateReport", codelewm.security.__all__)
        self.assertIn("enforce_public_license_gate", codelewm.security.__all__)
        self.assertIn("TrainConfig", codelewm.training.__all__)
        self.assertIn("load_train_config", codelewm.training.__all__)
        self.assertIn("train", codelewm.training.__all__)
        self.assertIn("train_cpu_smoke", codelewm.training.__all__)
        self.assertIn("train_torch", codelewm.training.__all__)

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
