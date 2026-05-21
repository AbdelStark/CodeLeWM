from __future__ import annotations

import json
import subprocess
import unittest
from pathlib import Path

from codelewm.harness import DEFAULT_DEMO_SCENARIO_ID


ROOT = Path(__file__).resolve().parents[2]


class LLMWorldModelDemoScriptTest(unittest.TestCase):
    def test_help_documents_scenario_selector(self) -> None:
        completed = subprocess.run(
            ["bash", "scripts/llm-world-model-demo", "--help"],
            cwd=ROOT,
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertIn("[--scenario ID]", completed.stdout)
        self.assertIn(DEFAULT_DEMO_SCENARIO_ID, completed.stdout)

    def test_list_scenarios_is_network_free_json(self) -> None:
        completed = subprocess.run(
            ["bash", "scripts/llm-world-model-demo", "--list-scenarios"],
            cwd=ROOT,
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertEqual(completed.stderr, "")
        payload = json.loads(completed.stdout)
        self.assertEqual(payload["default_scenario_id"], DEFAULT_DEMO_SCENARIO_ID)
        self.assertEqual(payload["scenarios"][0]["scenario_id"], DEFAULT_DEMO_SCENARIO_ID)
        self.assertIn("normalize_label", json.dumps(payload, sort_keys=True))


if __name__ == "__main__":
    unittest.main()
