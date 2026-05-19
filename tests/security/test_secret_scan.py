from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from codelewm.security import (
    SECRET_SCAN_REPORT_SCHEMA_VERSION,
    SecretScanError,
    SecretScanReport,
    scan_file,
    scan_paths,
    scan_text,
    secret_scan_report_json_schema,
    validate_secret_scan_report_payload,
)


ROOT = Path(__file__).resolve().parents[2]


class SecretScanTest(unittest.TestCase):
    def test_scan_text_flags_known_patterns_and_redacts_match(self) -> None:
        text = (
            "no secret here\n"
            "openai = sk-abcdefghijklmnopqrstuvwxyz123456\n"
            "GH=ghp_abcdefghijklmnopqrst\n"
            "AWS AKIAABCDEFGHIJKLMN\n"
            "password = SuperSecretValue_1234567890\n"
        )

        findings = scan_text(text, path="config.env")

        patterns = sorted({finding.pattern for finding in findings})
        self.assertIn("openai_api_key", patterns)
        self.assertIn("github_token", patterns)
        self.assertIn("aws_access_key_id", patterns)
        self.assertIn("generic_assigned_secret", patterns)
        for finding in findings:
            self.assertEqual(finding.path, "config.env")
            self.assertGreaterEqual(finding.line, 2)
            self.assertTrue(finding.redacted.startswith("[REDACTED_SECRET"))
            self.assertNotIn("sk-abcdefghijklmnopqrstuvwxyz123456", finding.redacted)
            self.assertNotIn("ghp_abcdefghijklmnopqrst", finding.redacted)
            self.assertNotIn("AKIAABCDEFGHIJKLMN", finding.redacted)
            self.assertNotIn("SuperSecretValue", finding.redacted)

    def test_scan_text_returns_no_findings_for_clean_input(self) -> None:
        findings = scan_text("clean configuration line without secrets\n")

        self.assertEqual(findings, ())

    def test_scan_file_skips_binary_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "binary.bin"
            path.write_bytes(bytes(range(256)))

            findings = scan_file(path)

        self.assertEqual(findings, ())

    def test_scan_paths_walks_directory_and_filters_by_suffix(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            safe = root / "safe.txt"
            secret_log = root / "logs" / "events.jsonl"
            ignored = root / "image.png"
            secret_log.parent.mkdir(parents=True)
            safe.write_text("ok\n", encoding="utf-8")
            secret_log.write_text("token sk-abcdefghijklmnopqrstuvwxyz123456\n", encoding="utf-8")
            ignored.write_text("token sk-abcdefghijklmnopqrstuvwxyz123456\n", encoding="utf-8")

            report = scan_paths([root])

        self.assertIsInstance(report, SecretScanReport)
        self.assertEqual(report.schema_version, SECRET_SCAN_REPORT_SCHEMA_VERSION)
        self.assertFalse(report.ok)
        self.assertIn(str(secret_log), report.paths_scanned)
        self.assertNotIn(str(ignored), report.paths_scanned)
        self.assertEqual(len(report.findings), 1)
        self.assertEqual(report.findings[0].pattern, "openai_api_key")
        self.assertEqual(report.findings[0].line, 1)

    def test_scan_paths_can_be_extended_with_extra_suffixes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            custom = root / "credentials.creds"
            custom.write_text("openai sk-abcdefghijklmnopqrstuvwxyz123456\n", encoding="utf-8")

            default_report = scan_paths([root])
            with_extension = scan_paths([root], include_suffixes=(".creds",))

        self.assertTrue(default_report.ok)
        self.assertFalse(with_extension.ok)
        self.assertEqual(with_extension.findings[0].pattern, "openai_api_key")

    def test_report_serialization_round_trips_through_validation(self) -> None:
        report = scan_paths([])
        payload = report.to_dict()

        loaded = validate_secret_scan_report_payload(payload)
        rendered = json.dumps(payload, sort_keys=True, allow_nan=False)

        self.assertEqual(loaded.to_dict(), report.to_dict())
        self.assertIn(SECRET_SCAN_REPORT_SCHEMA_VERSION, rendered)

    def test_report_schema_pins_required_fields(self) -> None:
        schema = secret_scan_report_json_schema()

        self.assertEqual(schema["properties"]["schema_version"]["const"], SECRET_SCAN_REPORT_SCHEMA_VERSION)
        self.assertIn("paths_scanned", schema["required"])
        self.assertIn("findings", schema["required"])
        self.assertIn("ok", schema["required"])
        finding_schema = schema["properties"]["findings"]["items"]
        self.assertEqual(set(finding_schema["required"]), {"path", "line", "pattern", "redacted"})

    def test_validation_rejects_unknown_schema_version(self) -> None:
        payload = {
            "schema_version": "codelewm.secret_scan.v999",
            "ok": True,
            "paths_scanned": [],
            "findings": [],
        }

        with self.assertRaises(SecretScanError):
            validate_secret_scan_report_payload(payload)


class SecretScanCliTest(unittest.TestCase):
    def test_secret_scan_cli_returns_clean_report_for_safe_directory(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            safe = Path(tmp) / "report.json"
            safe.write_text('{"ok": true}\n', encoding="utf-8")

            completed = _run_secret_scan(tmp)

        self.assertEqual(completed.returncode, 0, completed.stderr)
        payload = json.loads(completed.stdout)
        self.assertEqual(payload["schema_version"], SECRET_SCAN_REPORT_SCHEMA_VERSION)
        self.assertTrue(payload["ok"])
        self.assertEqual(payload["findings"], [])

    def test_secret_scan_cli_flags_secret_and_exits_nonzero(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tainted = Path(tmp) / "events.jsonl"
            tainted.write_text("token sk-abcdefghijklmnopqrstuvwxyz123456\n", encoding="utf-8")

            completed = _run_secret_scan(tmp)

        self.assertEqual(completed.returncode, 2, completed.stderr)
        payload = json.loads(completed.stdout)
        self.assertFalse(payload["ok"])
        self.assertEqual(payload["findings"][0]["pattern"], "openai_api_key")
        self.assertNotIn("sk-abcdefghijklmnopqrstuvwxyz123456", json.dumps(payload, sort_keys=True))

    def test_secret_scan_cli_ignores_public_repo_identifier_false_positives(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            report = Path(tmp) / "artifact.jsonl"
            report.write_text(
                "repo=flask-security-fork\n"
                "record_id=commitpackft:Pawamoy/django-zxcvbn-password:29600a6a8e8fa17e1c5b9f53dde57167450cbf4d:setup.py\n",
                encoding="utf-8",
            )

            completed = _run_secret_scan(tmp)

        self.assertEqual(completed.returncode, 0, completed.stdout)
        payload = json.loads(completed.stdout)
        self.assertTrue(payload["ok"])
        self.assertEqual(payload["findings"], [])


def _run_secret_scan(path: str | Path, *extra_args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            sys.executable,
            "-m",
            "codelewm.harness.cli",
            "secret-scan",
            str(path),
            "--json",
            *extra_args,
        ],
        cwd=ROOT,
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )


if __name__ == "__main__":
    unittest.main()
