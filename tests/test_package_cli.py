from __future__ import annotations

import re
import subprocess
import sys
import unittest
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - exercised on Python 3.10.
    import tomli as tomllib


ROOT = Path(__file__).resolve().parents[1]


def _dependency_names(requirements: list[str]) -> set[str]:
    names: set[str] = set()
    for requirement in requirements:
        match = re.match(r"([A-Za-z0-9_.-]+)", requirement)
        if match:
            names.add(match.group(1).lower().replace("_", "-"))
    return names


class PackageMetadataTest(unittest.TestCase):
    def test_console_script_entrypoint_is_registered(self) -> None:
        metadata = tomllib.loads((ROOT / "pyproject.toml").read_text())

        self.assertEqual(
            metadata["project"]["scripts"]["codelewm"],
            "codelewm.harness.cli:main",
        )

    def test_base_dependencies_stay_lightweight(self) -> None:
        metadata = tomllib.loads((ROOT / "pyproject.toml").read_text())
        base_dependencies = _dependency_names(metadata["project"]["dependencies"])

        self.assertIn("numpy", base_dependencies)
        for heavy_dependency in (
            "torch",
            "torchvision",
            "lightning",
            "hydra-core",
            "omegaconf",
            "stable-pretraining",
            "stable-worldmodel",
            "h5py",
            "pyarrow",
            "scikit-learn",
        ):
            with self.subTest(dependency=heavy_dependency):
                self.assertNotIn(heavy_dependency, base_dependencies)

    def test_dependency_groups_cover_optional_runtime_boundaries(self) -> None:
        metadata = tomllib.loads((ROOT / "pyproject.toml").read_text())
        groups = metadata["dependency-groups"]
        extras = metadata["project"]["optional-dependencies"]

        for group_name in ("dev", "data", "train", "eval", "docs", "release"):
            with self.subTest(group=group_name):
                self.assertIn(group_name, groups)

        self.assertIn("pytest", _dependency_names(groups["dev"]))
        self.assertEqual({"h5py", "pyarrow"}, _dependency_names(groups["data"]))
        self.assertIn("torch", _dependency_names(groups["train"]))
        self.assertIn("stable-worldmodel", _dependency_names(groups["train"]))
        self.assertIn("scikit-learn", _dependency_names(groups["eval"]))
        self.assertIn("build", _dependency_names(groups["release"]))
        self.assertIn("pip-audit", _dependency_names(groups["release"]))
        self.assertIn("twine", _dependency_names(groups["release"]))

        for extra_name in ("data", "train", "eval", "docs", "release"):
            with self.subTest(extra=extra_name):
                self.assertEqual(extras[extra_name], groups[extra_name])

    def test_release_metadata_is_publishable(self) -> None:
        metadata = tomllib.loads((ROOT / "pyproject.toml").read_text())
        project = metadata["project"]

        self.assertEqual(project["readme"], "README.md")
        self.assertEqual(project["license"], "MIT")
        self.assertEqual(project["license-files"], ["LICENSE"])
        self.assertEqual(project["requires-python"], ">=3.10")
        self.assertIn("Development Status :: 2 - Pre-Alpha", project["classifiers"])
        self.assertIn("Typing :: Typed", project["classifiers"])
        self.assertEqual(
            project["urls"]["Repository"],
            "https://github.com/AbdelStark/CodeLeWM",
        )
        self.assertTrue((ROOT / "README.md").is_file())
        self.assertTrue((ROOT / "LICENSE").is_file())

    def test_typed_package_marker_is_included(self) -> None:
        metadata = tomllib.loads((ROOT / "pyproject.toml").read_text())

        self.assertTrue((ROOT / "codelewm" / "py.typed").is_file())
        self.assertEqual(
            metadata["tool"]["setuptools"]["package-data"]["codelewm"],
            ["py.typed"],
        )

    def test_cli_module_help_runs(self) -> None:
        completed = subprocess.run(
            [sys.executable, "-m", "codelewm.harness.cli", "--help"],
            cwd=ROOT,
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertIn("CodeLeWM command-line interface", completed.stdout)

    def test_package_exposes_version(self) -> None:
        completed = subprocess.run(
            [
                sys.executable,
                "-c",
                "import codelewm; print(codelewm.__version__)",
            ],
            cwd=ROOT,
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertRegex(completed.stdout.strip(), r"^\d+\.\d+\.\d+$")


if __name__ == "__main__":
    unittest.main()
