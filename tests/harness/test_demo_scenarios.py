from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from codelewm.harness import (
    DEFAULT_DEMO_SCENARIO_ID,
    DEMO_SCENARIO_SCHEMA_VERSION,
    DemoScenario,
    DemoScenarioError,
    DemoScenarioFile,
    get_demo_scenario,
    materialize_demo_scenario,
    write_demo_scenario_shell_env,
)


class DemoScenarioTest(unittest.TestCase):
    def test_default_scenario_is_meaningful_bug_fix_metadata(self) -> None:
        scenario = get_demo_scenario()
        payload = scenario.to_dict()

        self.assertEqual(scenario.scenario_id, DEFAULT_DEMO_SCENARIO_ID)
        self.assertEqual(payload["schema_version"], DEMO_SCENARIO_SCHEMA_VERSION)
        self.assertEqual(payload["primary_context_path"], "app.py")
        self.assertIn("blank", scenario.instruction)
        self.assertIn("untitled", scenario.instruction)
        self.assertTrue(payload["expected_static_constraints"]["non_comment_change_required"])
        self.assertIn("normalize_label", payload["expected_static_constraints"]["touched_symbols"])
        self.assertNotIn("comment candidate", scenario.instruction.lower())

    def test_materialize_writes_before_file_and_artifact_safe_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            metadata = materialize_demo_scenario(get_demo_scenario(), Path(tmp))
            before = Path(metadata["before_path"])
            scenario_json = Path(metadata["metadata_path"])

            self.assertTrue(before.is_file())
            self.assertTrue(scenario_json.is_file())
            self.assertEqual(before.read_text(encoding="utf-8"), get_demo_scenario().primary_file.content)
            serialized = scenario_json.read_text(encoding="utf-8")
            loaded = json.loads(serialized)

        self.assertEqual(loaded["scenario_id"], DEFAULT_DEMO_SCENARIO_ID)
        self.assertEqual(loaded["primary_context_path"], "app.py")
        self.assertNotIn(str(Path.home()), serialized)
        self.assertNotIn("TOKEN", serialized.upper())

    def test_shell_env_file_uses_scenario_defaults(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            metadata = materialize_demo_scenario(get_demo_scenario(), root / "fixtures")
            env_file = root / "scenario.env"
            write_demo_scenario_shell_env(metadata, env_file)
            contents = env_file.read_text(encoding="utf-8")

        self.assertIn("CODELEWM_LLM_DEMO_SCENARIO_ID=bugfix-edge-case", contents)
        self.assertIn("CODELEWM_LLM_DEMO_SCENARIO_BEFORE=", contents)
        self.assertIn("CODELEWM_LLM_DEMO_SCENARIO_INSTRUCTION=", contents)
        self.assertNotIn("\nexport ", contents)

    def test_unknown_scenario_fails_with_available_choices(self) -> None:
        with self.assertRaises(DemoScenarioError) as raised:
            get_demo_scenario("missing")

        self.assertIn("unknown demo scenario", str(raised.exception))
        self.assertIn(DEFAULT_DEMO_SCENARIO_ID, str(raised.exception))

    def test_scenario_rejects_token_bearing_paths(self) -> None:
        with self.assertRaises(DemoScenarioError):
            DemoScenario(
                scenario_id="bad",
                title="Bad",
                instruction="Bad",
                prompt_template_id="template",
                files=(DemoScenarioFile(path=".env", content="SECRET=x\n", primary=True),),
            )


if __name__ == "__main__":
    unittest.main()
