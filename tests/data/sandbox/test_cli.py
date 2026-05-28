"""CLI integration test for ``codelewm dataset execute``."""

from __future__ import annotations

import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

from codelewm.harness.cli import build_parser


class DatasetExecuteCLITest(unittest.TestCase):
    def test_execute_emits_json_payload(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            code = root / "code.py"
            code.write_text(
                "def add(a, b):\n    return a + b\n", encoding="utf-8"
            )
            args_input = root / "input.json"
            args_input.write_text(json.dumps([2, 3]), encoding="utf-8")
            parser = build_parser()
            namespace = parser.parse_args(
                [
                    "dataset",
                    "execute",
                    "--code-file",
                    str(code),
                    "--input-file",
                    str(args_input),
                    "--function-name",
                    "add",
                    "--timeout-ms",
                    "3000",
                    "--no-determinism-check",
                    "--json",
                ]
            )
            buf = io.StringIO()
            with redirect_stdout(buf):
                exit_code = namespace.func(namespace)
            self.assertEqual(exit_code, 0)
            payload = json.loads(buf.getvalue())
            self.assertEqual(payload["exit_code"], "ok")
            self.assertEqual(payload["output_repr"], "5")
            self.assertEqual(payload["output_type"], "int")
            self.assertEqual(payload["schema_version"], "codelewm.sandbox_result.v1")


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
