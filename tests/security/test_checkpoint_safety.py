from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from codelewm.harness import ScoreError, load_scorer
from codelewm.model import (
    LATENT_DIM,
    build_checkpoint_metadata,
    write_checkpoint_manifest,
)
from codelewm.security import (
    CheckpointTrustError,
    default_checkpoint_manifest_path,
    require_trusted_checkpoint,
)


ROOT = Path(__file__).resolve().parents[2]


class CheckpointTrustGateTest(unittest.TestCase):
    def test_require_trusted_checkpoint_accepts_paired_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            checkpoint = root / "checkpoint.bin"
            checkpoint.write_bytes(b"fixture checkpoint")
            manifest_path = default_checkpoint_manifest_path(checkpoint)
            metadata = build_checkpoint_metadata({"seed": 1, "latent_dim": LATENT_DIM})
            write_checkpoint_manifest(
                metadata=metadata,
                checkpoint_path=checkpoint,
                manifest_path=manifest_path,
            )

            manifest = require_trusted_checkpoint(checkpoint)

        self.assertEqual(manifest.checkpoint_path, "checkpoint.bin")

    def test_require_trusted_checkpoint_refuses_unmanifested_load(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            checkpoint = Path(tmp) / "checkpoint.bin"
            checkpoint.write_bytes(b"fixture checkpoint")

            with self.assertRaises(CheckpointTrustError) as ctx:
                require_trusted_checkpoint(checkpoint)

        message = str(ctx.exception)
        self.assertIn("manifest is required", message)

    def test_require_trusted_checkpoint_refuses_tampered_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            checkpoint = root / "checkpoint.bin"
            checkpoint.write_bytes(b"first")
            manifest_path = default_checkpoint_manifest_path(checkpoint)
            metadata = build_checkpoint_metadata({"seed": 2})
            write_checkpoint_manifest(
                metadata=metadata,
                checkpoint_path=checkpoint,
                manifest_path=manifest_path,
            )
            checkpoint.write_bytes(b"tampered")

            with self.assertRaisesRegex(CheckpointTrustError, "checksum mismatch"):
                require_trusted_checkpoint(checkpoint)

    def test_require_trusted_checkpoint_rejects_mismatched_manifest_target(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            checkpoint = root / "checkpoint.bin"
            decoy = root / "other.bin"
            checkpoint.write_bytes(b"actual")
            decoy.write_bytes(b"decoy")
            manifest_path = root / "shared.manifest.json"
            metadata = build_checkpoint_metadata({"seed": 3})
            write_checkpoint_manifest(
                metadata=metadata,
                checkpoint_path=decoy,
                manifest_path=manifest_path,
            )

            with self.assertRaisesRegex(CheckpointTrustError, "does not reference"):
                require_trusted_checkpoint(checkpoint, manifest_path=manifest_path)


class LoadScorerTrustBoundaryTest(unittest.TestCase):
    def test_load_scorer_refuses_unmanifested_checkpoint_by_default(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            checkpoint = Path(tmp) / "checkpoint.bin"
            checkpoint.write_bytes(b"fixture checkpoint")

            with self.assertRaises(ScoreError) as ctx:
                load_scorer(checkpoint)

        self.assertEqual(ctx.exception.error_type, "checkpoint_error")
        self.assertIn("manifest", str(ctx.exception))

    def test_load_scorer_accepts_checkpoint_with_paired_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            checkpoint = root / "checkpoint.bin"
            checkpoint.write_bytes(b"fixture checkpoint")
            manifest_path = default_checkpoint_manifest_path(checkpoint)
            write_checkpoint_manifest(
                metadata=build_checkpoint_metadata({"seed": 4}),
                checkpoint_path=checkpoint,
                manifest_path=manifest_path,
            )

            scorer = load_scorer(checkpoint)

        self.assertEqual(len(scorer.checkpoint_sha256), 64)

    def test_load_scorer_allow_unsafe_skips_manifest_check(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            checkpoint = Path(tmp) / "checkpoint.bin"
            checkpoint.write_bytes(b"fixture checkpoint")

            scorer = load_scorer(checkpoint, allow_unsafe=True)

        self.assertEqual(len(scorer.checkpoint_sha256), 64)


class ScoreCliTrustBoundaryTest(unittest.TestCase):
    def test_score_cli_refuses_unmanifested_checkpoint_by_default(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            before = root / "before.py"
            candidate = root / "after.py"
            checkpoint = root / "checkpoint.bin"
            before.write_text("value = 1\n", encoding="utf-8")
            candidate.write_text("value = 2\n", encoding="utf-8")
            checkpoint.write_bytes(b"fixture checkpoint")

            completed = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "codelewm.harness.cli",
                    "score",
                    "--before",
                    str(before),
                    "--instruction",
                    "increment value",
                    "--candidate",
                    str(candidate),
                    "--checkpoint",
                    str(checkpoint),
                    "--json",
                ],
                cwd=ROOT,
                check=False,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )

        self.assertEqual(completed.returncode, 2, completed.stderr)
        payload = json.loads(completed.stdout)
        self.assertEqual(payload["error_type"], "checkpoint_error")
        self.assertIn("manifest", payload["message"])


if __name__ == "__main__":
    unittest.main()
