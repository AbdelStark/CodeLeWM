"""Container-image smoke tests.

These tests build the v0.6 runtime image and exercise it. They are
gated on Docker being installed and the daemon being reachable. CI's
default ``--group dev`` lane does not run them; operators do.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
DOCKERFILE = REPO_ROOT / "containers" / "v0_6" / "Dockerfile"
BUILD_SCRIPT = REPO_ROOT / "scripts" / "build-codelewm-runtime"


def _docker_available() -> bool:
    if not shutil.which("docker"):
        return False
    try:
        completed = subprocess.run(
            ["docker", "info"],
            capture_output=True,
            timeout=10,
            check=False,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        return False
    return completed.returncode == 0


_DOCKER_AVAILABLE = _docker_available()
_RUN_CONTAINER_TESTS = bool(os.getenv("CODELEWM_TEST_CONTAINER_BUILD"))


class RuntimeImageDockerfileTest(unittest.TestCase):
    """Static checks that don't need Docker."""

    def test_dockerfile_exists(self) -> None:
        self.assertTrue(DOCKERFILE.is_file())
        content = DOCKERFILE.read_text(encoding="utf-8")
        # Required structural elements documented in the operations doc.
        for marker in (
            "FROM ${BASE_IMAGE}",
            "useradd -m -u 1000 codelewm",
            "USER codelewm",
            "uv pip install --system",
            "CODELEWM_HF_RUN_NAME",
            "CODELEWM_EXECUTION_PACK_REPO_ID",
            "CODELEWM_EXECUTION_PACK_REVISION",
            "CODELEWM_TRAIN_SEED",
            "CODELEWM_TRAIN_CONFIG",
        ):
            with self.subTest(marker=marker):
                self.assertIn(marker, content)

    def test_build_script_is_executable_and_self_documents(self) -> None:
        self.assertTrue(BUILD_SCRIPT.is_file())
        self.assertTrue(
            os.access(BUILD_SCRIPT, os.X_OK), "build script must be executable"
        )
        content = BUILD_SCRIPT.read_text(encoding="utf-8")
        for marker in ("--push", "--platform", "--tag", "CODELEWM_GIT_SHA"):
            with self.subTest(marker=marker):
                self.assertIn(marker, content)

    def test_runbook_doc_exists(self) -> None:
        doc = REPO_ROOT / "docs" / "operations" / "V0_6_RUNTIME_CONTAINER.md"
        self.assertTrue(doc.is_file())
        text = doc.read_text(encoding="utf-8")
        for marker in (
            "scripts/build-codelewm-runtime",
            "codelewm-execution-train-smoke",
            "registry.hf.co",
            "linux/amd64",
        ):
            with self.subTest(marker=marker):
                self.assertIn(marker, text)


@unittest.skipUnless(
    _DOCKER_AVAILABLE and _RUN_CONTAINER_TESTS,
    "docker not available or CODELEWM_TEST_CONTAINER_BUILD not set",
)
class RuntimeImageBuildAndRunTest(unittest.TestCase):
    """Opt-in smoke that builds the image and runs `codelewm --version` inside it."""

    image_tag = "codelewm-runtime:test"

    @classmethod
    def setUpClass(cls) -> None:  # pragma: no cover - opt-in
        subprocess.run(
            [
                str(BUILD_SCRIPT),
                "--tag",
                cls.image_tag,
            ],
            check=True,
            cwd=str(REPO_ROOT),
        )

    def test_codelewm_console_script_resolves(self) -> None:  # pragma: no cover
        completed = subprocess.run(
            ["docker", "run", "--rm", self.image_tag, "codelewm", "--version"],
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(completed.returncode, 0, msg=completed.stderr)
        self.assertIn("codelewm", completed.stdout.lower())

    def test_smoke_runner_help_resolves(self) -> None:  # pragma: no cover
        completed = subprocess.run(
            [
                "docker",
                "run",
                "--rm",
                self.image_tag,
                "codelewm-execution-train-smoke",
                "--help",
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(completed.returncode, 0, msg=completed.stderr)
        self.assertIn("--out", completed.stdout)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
