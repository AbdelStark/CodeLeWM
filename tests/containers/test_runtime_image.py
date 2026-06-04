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
ENTRYPOINT = REPO_ROOT / "containers" / "v0_6" / "entrypoint.sh"
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
            "CODELEWM_EXECUTION_PACK_LOCAL_DIR",
            "codelewm-runtime-entrypoint",
        ):
            with self.subTest(marker=marker):
                self.assertIn(marker, text)

    def test_entrypoint_script_exists_and_is_executable(self) -> None:
        self.assertTrue(ENTRYPOINT.is_file(), f"missing {ENTRYPOINT}")
        self.assertTrue(
            os.access(ENTRYPOINT, os.X_OK),
            f"entrypoint {ENTRYPOINT} is not executable",
        )
        body = ENTRYPOINT.read_text(encoding="utf-8")
        for marker in (
            "CODELEWM_EXECUTION_PACK_LOCAL_DIR",
            "CODELEWM_EXECUTION_PACK_REPO_ID",
            "CODELEWM_EXECUTION_PACK_REVISION",
            "HF_TOKEN",
            "hf download",
            # Operator's command is run (not ``exec``ed) so the
            # entrypoint can run the post-training upload.
            '"$@"',
            # Post-run upload knobs the launcher wires.
            "CODELEWM_UPLOAD_REPO_ID",
            "CODELEWM_RUN_OUTPUT_DIR",
            "hf upload",
        ):
            with self.subTest(marker=marker):
                self.assertIn(marker, body)

    def test_dockerfile_wires_entrypoint(self) -> None:
        content = DOCKERFILE.read_text(encoding="utf-8")
        self.assertIn(
            "containers/v0_6/entrypoint.sh /usr/local/bin/codelewm-runtime-entrypoint",
            content,
        )
        self.assertIn(
            'ENTRYPOINT ["/usr/local/bin/codelewm-runtime-entrypoint"]',
            content,
        )
        self.assertIn("CODELEWM_EXECUTION_PACK_LOCAL_DIR=", content)


class RuntimeImageV07DockerfileTest(unittest.TestCase):
    """Static checks for the v0.7 runtime image (no Docker needed).

    v0.7 training requires its own image because the package source is
    COPYed in at build time and v0.7 carries the RFC-0015 WS-C levers
    (transformer state encoder + InfoNCE retrieval) absent from v0.6.
    """

    DOCKERFILE = REPO_ROOT / "containers" / "v0_7" / "Dockerfile"
    ENTRYPOINT = REPO_ROOT / "containers" / "v0_7" / "entrypoint.sh"

    def test_dockerfile_exists_and_is_v0_7(self) -> None:
        self.assertTrue(self.DOCKERFILE.is_file())
        content = self.DOCKERFILE.read_text(encoding="utf-8")
        for marker in (
            "FROM ${BASE_IMAGE}",
            "useradd -m -u 1000 codelewm",
            "USER codelewm",
            "uv pip install --system",
            'org.opencontainers.image.version="v0.7"',
        ):
            with self.subTest(marker=marker):
                self.assertIn(marker, content)

    def test_dockerfile_wires_v0_7_entrypoint(self) -> None:
        content = self.DOCKERFILE.read_text(encoding="utf-8")
        self.assertIn(
            "containers/v0_7/entrypoint.sh /usr/local/bin/codelewm-runtime-entrypoint",
            content,
        )
        self.assertIn(
            'ENTRYPOINT ["/usr/local/bin/codelewm-runtime-entrypoint"]',
            content,
        )

    def test_entrypoint_script_exists_and_is_executable(self) -> None:
        self.assertTrue(self.ENTRYPOINT.is_file(), f"missing {self.ENTRYPOINT}")
        self.assertTrue(
            os.access(self.ENTRYPOINT, os.X_OK),
            f"entrypoint {self.ENTRYPOINT} is not executable",
        )
        body = self.ENTRYPOINT.read_text(encoding="utf-8")
        for marker in ("hf download", '"$@"', "hf upload"):
            with self.subTest(marker=marker):
                self.assertIn(marker, body)

    def test_build_script_supports_dockerfile_override(self) -> None:
        content = BUILD_SCRIPT.read_text(encoding="utf-8")
        self.assertIn("--dockerfile", content)


class RuntimeImageV08DockerfileTest(unittest.TestCase):
    """Static checks for the v0.8 correctness-aware runtime image."""

    DOCKERFILE = REPO_ROOT / "containers" / "v0_8" / "Dockerfile"
    ENTRYPOINT = REPO_ROOT / "containers" / "v0_8" / "entrypoint.sh"

    def test_dockerfile_exists_and_is_v0_8(self) -> None:
        self.assertTrue(self.DOCKERFILE.is_file())
        content = self.DOCKERFILE.read_text(encoding="utf-8")
        for marker in (
            "FROM ${BASE_IMAGE}",
            "useradd -m -u 1000 codelewm",
            "USER codelewm",
            "uv pip install --system",
            'org.opencontainers.image.version="v0.8"',
            "p_pass BCE head",
            "output-value auxiliary CE head",
        ):
            with self.subTest(marker=marker):
                self.assertIn(marker, content)

    def test_dockerfile_wires_v0_8_entrypoint(self) -> None:
        content = self.DOCKERFILE.read_text(encoding="utf-8")
        self.assertIn(
            "containers/v0_8/entrypoint.sh /usr/local/bin/codelewm-runtime-entrypoint",
            content,
        )
        self.assertIn(
            'ENTRYPOINT ["/usr/local/bin/codelewm-runtime-entrypoint"]',
            content,
        )

    def test_entrypoint_script_exists_and_is_executable(self) -> None:
        self.assertTrue(self.ENTRYPOINT.is_file(), f"missing {self.ENTRYPOINT}")
        self.assertTrue(
            os.access(self.ENTRYPOINT, os.X_OK),
            f"entrypoint {self.ENTRYPOINT} is not executable",
        )
        body = self.ENTRYPOINT.read_text(encoding="utf-8")
        for marker in ("hf download", '"$@"', "hf upload", "codelewm v0.8 run"):
            with self.subTest(marker=marker):
                self.assertIn(marker, body)

    def test_build_script_mentions_v0_8_dockerfile(self) -> None:
        content = BUILD_SCRIPT.read_text(encoding="utf-8")
        self.assertIn("containers/v0_8/Dockerfile", content)
        self.assertIn("ghcr.io/abdelstark/codelewm-runtime:v0.8", content)


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
