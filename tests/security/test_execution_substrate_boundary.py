"""Tests for the execution-substrate claim boundary.

The claim boundary text is embedded into manifests, dataset cards, and
model cards verbatim. These tests ensure the boundary loader returns the
exact file contents and produces a stable fingerprint, that the required
boundary file exists, and that the public language paragraph required by
the boundary appears in the operations and security docs.
"""

from __future__ import annotations

import hashlib
import unittest
from pathlib import Path

from codelewm.security.claim_boundaries import (
    ClaimBoundaryError,
    available_claim_boundaries,
    claim_boundary_fingerprint,
    load_claim_boundary,
)


ROOT = Path(__file__).resolve().parents[2]
EXECUTION_BOUNDARY = "execution_substrate.v1"
REQUIRED_PARAGRAPH = (
    "The execution-pack data artifact is the deterministic output of running\n"
    "licensed public Python submissions in an isolated sandbox under a\n"
    "stdlib-only policy at data-build time. The artifact contains no\n"
    "executable payload; it contains tokenized code, tokenized inputs,\n"
    "tokenized outputs, and metadata. Training and inference never execute\n"
    "code. The sandbox is reused only in the dedicated downstream-evaluation\n"
    "scenario (`execution-rerank`) to label completion correctness against\n"
    "hidden tests, and only on inputs the operator has reviewed."
)


class ExecutionSubstrateBoundaryTest(unittest.TestCase):
    def test_execution_substrate_boundary_is_registered(self) -> None:
        self.assertIn(EXECUTION_BOUNDARY, available_claim_boundaries())

    def test_load_returns_verbatim_file_contents(self) -> None:
        boundary_path = (
            ROOT
            / "codelewm"
            / "security"
            / "claim_boundaries"
            / f"{EXECUTION_BOUNDARY}.md"
        )
        on_disk = boundary_path.read_text(encoding="utf-8")
        self.assertEqual(load_claim_boundary(EXECUTION_BOUNDARY), on_disk)

    def test_required_paragraph_is_present(self) -> None:
        text = load_claim_boundary(EXECUTION_BOUNDARY)
        normalized = "\n".join(
            line[2:] if line.startswith("> ") else line
            for line in text.splitlines()
        )
        self.assertIn(REQUIRED_PARAGRAPH, normalized)

    def test_fingerprint_is_sha256_of_text(self) -> None:
        text = load_claim_boundary(EXECUTION_BOUNDARY)
        expected = hashlib.sha256(text.encode("utf-8")).hexdigest()
        self.assertEqual(claim_boundary_fingerprint(EXECUTION_BOUNDARY), expected)

    def test_unknown_boundary_raises_named_error(self) -> None:
        with self.assertRaises(ClaimBoundaryError):
            load_claim_boundary("does-not-exist")

    def test_invalid_boundary_name_is_rejected(self) -> None:
        for bad in ("", "a/b", "..\\evil", "../escape"):
            with self.subTest(name=bad):
                with self.assertRaises(ClaimBoundaryError):
                    load_claim_boundary(bad)

    def test_sandbox_policy_doc_exists(self) -> None:
        doc = ROOT / "docs" / "operations" / "sandbox_policy.md"
        self.assertTrue(doc.is_file(), "sandbox policy doc is missing")
        content = doc.read_text(encoding="utf-8")
        self.assertIn("Sandbox Policy", content)
        self.assertIn("execution_substrate.v1.md", content)

    def test_security_doc_references_sandbox_subsystem(self) -> None:
        doc = ROOT / "SECURITY.md"
        content = doc.read_text(encoding="utf-8")
        self.assertIn("sandbox", content.lower())

    def test_rfc_0014_is_present(self) -> None:
        rfc = (
            ROOT
            / "docs"
            / "rfcs"
            / "RFC-0014-execution-trace-world-model-substrate.md"
        )
        self.assertTrue(rfc.is_file(), "RFC-0014 is missing")

    def test_roadmap_doc_is_present(self) -> None:
        roadmap = (
            ROOT / "docs" / "roadmap" / "EXECUTION_TRACE_WORLD_MODEL.md"
        )
        self.assertTrue(roadmap.is_file(), "execution-trace roadmap is missing")


if __name__ == "__main__":
    unittest.main()
