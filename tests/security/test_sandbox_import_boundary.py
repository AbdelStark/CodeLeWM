"""Structural check that the sandbox does not leak into the model paths.

The execution-substrate sandbox is a data-prep component. The training,
inference, scoring, indexing, and evaluation paths must not import it.
This test grep-greps the source tree to make sure that boundary holds.
A future runtime guard (e.g. an audit hook in the package __init__) can
supplement this static check, but a grep is cheap, clear, and authoritative
for the working tree.
"""

from __future__ import annotations

import re
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SANDBOX_IMPORT_PATTERNS = (
    re.compile(r"^\s*from\s+codelewm\.data\.sandbox", re.MULTILINE),
    re.compile(r"^\s*import\s+codelewm\.data\.sandbox", re.MULTILINE),
)
FORBIDDEN_TREES = (
    ROOT / "codelewm" / "training",
    ROOT / "codelewm" / "model",
    ROOT / "codelewm" / "eval",
    ROOT / "codelewm" / "observability",
    # The scorer is the part of the harness that must remain non-executing.
    ROOT / "codelewm" / "harness" / "scorer.py",
    ROOT / "codelewm" / "harness" / "index_runner.py",
    ROOT / "codelewm" / "harness" / "transition_index.py",
    ROOT / "codelewm" / "harness" / "quality.py",
)


class SandboxImportBoundaryTest(unittest.TestCase):
    def test_forbidden_trees_do_not_import_sandbox(self) -> None:
        for tree in FORBIDDEN_TREES:
            with self.subTest(tree=str(tree.relative_to(ROOT))):
                if tree.is_file():
                    paths = [tree]
                elif tree.is_dir():
                    paths = list(tree.rglob("*.py"))
                else:
                    continue
                for path in paths:
                    if "__pycache__" in path.parts:
                        continue
                    text = path.read_text(encoding="utf-8")
                    for pattern in SANDBOX_IMPORT_PATTERNS:
                        match = pattern.search(text)
                        self.assertIsNone(
                            match,
                            f"{path.relative_to(ROOT)} imports codelewm.data.sandbox; "
                            "the sandbox is a data-prep-only module. See "
                            "docs/operations/sandbox_policy.md.",
                        )


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
