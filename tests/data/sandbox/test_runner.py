"""Sandbox runner end-to-end tests.

The runner spawns a subprocess for each invocation, so each case here
costs ~50-200 ms of wall time. Keep the test count small and use the
shortest payloads that exercise the policy edge.
"""

from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

from codelewm.data.sandbox import (
    DEFAULT_SANDBOX_POLICY,
    SANDBOX_RESULT_SCHEMA_VERSION,
    SandboxExitCode,
    SandboxPolicy,
    SandboxPolicyError,
    SandboxResult,
    classify_output_type,
    run_one,
)


def _fast_policy(**overrides: object) -> SandboxPolicy:
    """Return a policy tuned for unit tests: short timeout, no determinism."""

    base = dict(
        timeout_ms=3000,
        determinism_check=False,
        memory_mb=128,
        cpu_seconds=2,
    )
    base.update(overrides)
    return SandboxPolicy(**base)  # type: ignore[arg-type]


class SandboxPolicyValidationTest(unittest.TestCase):
    def test_default_policy_is_stdlib_only(self) -> None:
        self.assertEqual(DEFAULT_SANDBOX_POLICY.import_allowlist, "stdlib_only")
        self.assertTrue(DEFAULT_SANDBOX_POLICY.deny_network)
        self.assertTrue(DEFAULT_SANDBOX_POLICY.deny_subprocess)
        self.assertTrue(DEFAULT_SANDBOX_POLICY.deny_filesystem_writes_outside_scratch)

    def test_invalid_timeout_rejected(self) -> None:
        with self.assertRaises(SandboxPolicyError):
            SandboxPolicy(timeout_ms=0)
        with self.assertRaises(SandboxPolicyError):
            SandboxPolicy(timeout_ms=10**9)

    def test_invalid_memory_rejected(self) -> None:
        with self.assertRaises(SandboxPolicyError):
            SandboxPolicy(memory_mb=0)

    def test_unsupported_allowlist_rejected(self) -> None:
        with self.assertRaises(SandboxPolicyError):
            SandboxPolicy(import_allowlist="all")  # type: ignore[arg-type]

    def test_classify_output_type_covers_common_kinds(self) -> None:
        self.assertEqual(classify_output_type(None), "none")
        self.assertEqual(classify_output_type(True), "bool")
        self.assertEqual(classify_output_type(1), "int")
        self.assertEqual(classify_output_type(1.0), "float")
        self.assertEqual(classify_output_type("x"), "str")
        self.assertEqual(classify_output_type([1, 2]), "list")
        self.assertEqual(classify_output_type({"a": 1}), "dict")
        self.assertEqual(classify_output_type((1,)), "tuple")
        self.assertEqual(classify_output_type(set()), "set")


class SandboxRunnerHappyPathTest(unittest.TestCase):
    def test_function_call_returns_repr_of_value(self) -> None:
        code = "def add(a, b):\n    return a + b\n"
        result = run_one(
            code,
            input_repr=json.dumps([2, 3]),
            function_name="add",
            policy=_fast_policy(),
        )
        self.assertEqual(result.schema_version, SANDBOX_RESULT_SCHEMA_VERSION)
        self.assertEqual(
            result.exit_code,
            SandboxExitCode.OK,
            msg=(
                f"exit_code={result.exit_code.value} "
                f"exception_class={result.exception_class} "
                f"exception_message={result.exception_message} "
                f"policy_violations={result.policy_violations}"
            ),
        )
        self.assertEqual(result.output_repr, "5")
        self.assertEqual(result.output_type, "int")
        self.assertEqual(result.exception_class, None)
        self.assertEqual(result.policy_violations, ())

    def test_function_call_with_kwargs(self) -> None:
        code = "def greet(name, greeting='hi'):\n    return f'{greeting}, {name}!'\n"
        result = run_one(
            code,
            input_repr=json.dumps({"name": "Ada", "greeting": "hello"}),
            function_name="greet",
            policy=_fast_policy(),
        )
        self.assertEqual(result.exit_code, SandboxExitCode.OK)
        self.assertEqual(result.output_repr, "'hello, Ada!'")
        self.assertEqual(result.output_type, "str")

    def test_function_call_returning_collection_classified(self) -> None:
        code = "def f(n):\n    return list(range(n))\n"
        result = run_one(
            code,
            input_repr=json.dumps([3]),
            function_name="f",
            policy=_fast_policy(),
        )
        self.assertEqual(result.exit_code, SandboxExitCode.OK)
        self.assertEqual(result.output_repr, "[0, 1, 2]")
        self.assertEqual(result.output_type, "list")

    def test_script_style_captures_stdout(self) -> None:
        code = "print('hello')\n"
        result = run_one(code, policy=_fast_policy())
        self.assertEqual(result.exit_code, SandboxExitCode.OK)
        self.assertEqual(result.output_kind, "stdout")
        self.assertIn("hello", result.stdout)


class SandboxRunnerExceptionTest(unittest.TestCase):
    def test_raised_exception_is_reported(self) -> None:
        code = "def f():\n    raise ValueError('boom')\n"
        result = run_one(
            code,
            input_repr=json.dumps([]),
            function_name="f",
            policy=_fast_policy(),
        )
        self.assertEqual(result.exit_code, SandboxExitCode.RAISED)
        self.assertEqual(result.exception_class, "ValueError")
        self.assertIn("boom", result.exception_message or "")
        self.assertEqual(result.output_type, "exception")

    def test_index_error_classification(self) -> None:
        code = "def f():\n    return [][0]\n"
        result = run_one(
            code,
            input_repr=json.dumps([]),
            function_name="f",
            policy=_fast_policy(),
        )
        self.assertEqual(result.exit_code, SandboxExitCode.RAISED)
        self.assertEqual(result.exception_class, "IndexError")


class SandboxRunnerPolicyTest(unittest.TestCase):
    def test_network_socket_denied(self) -> None:
        code = (
            "import socket\n"
            "def f():\n"
            "    s = socket.socket(); s.connect(('127.0.0.1', 1))\n"
            "    return 0\n"
        )
        result = run_one(
            code,
            input_repr=json.dumps([]),
            function_name="f",
            policy=_fast_policy(),
        )
        self.assertEqual(result.exit_code, SandboxExitCode.POLICY_VIOLATION)
        self.assertTrue(
            any("network_denied" in v for v in result.policy_violations),
            f"expected a network_denied violation, got {result.policy_violations}",
        )

    def test_subprocess_denied(self) -> None:
        code = (
            "import subprocess\n"
            "def f():\n"
            "    subprocess.Popen(['/bin/echo', 'x'])\n"
            "    return 0\n"
        )
        result = run_one(
            code,
            input_repr=json.dumps([]),
            function_name="f",
            policy=_fast_policy(),
        )
        self.assertEqual(result.exit_code, SandboxExitCode.POLICY_VIOLATION)
        self.assertTrue(
            any("subprocess_denied" in v for v in result.policy_violations),
            f"expected a subprocess_denied violation, got {result.policy_violations}",
        )

    def test_filesystem_write_outside_scratch_denied(self) -> None:
        code = (
            "def f():\n"
            "    with open('/tmp/codelewm-sandbox-escape', 'w') as fh:\n"
            "        fh.write('x')\n"
            "    return 0\n"
        )
        result = run_one(
            code,
            input_repr=json.dumps([]),
            function_name="f",
            policy=_fast_policy(),
        )
        self.assertEqual(result.exit_code, SandboxExitCode.POLICY_VIOLATION)
        self.assertTrue(
            any("filesystem_denied" in v for v in result.policy_violations),
            f"expected filesystem_denied, got {result.policy_violations}",
        )

    def test_filesystem_write_inside_scratch_allowed(self) -> None:
        code = (
            "def f():\n"
            "    with open('inside.txt', 'w') as fh:\n"
            "        fh.write('ok')\n"
            "    return 'wrote'\n"
        )
        result = run_one(
            code,
            input_repr=json.dumps([]),
            function_name="f",
            policy=_fast_policy(),
        )
        self.assertEqual(result.exit_code, SandboxExitCode.OK)
        self.assertEqual(result.output_repr, "'wrote'")

    def test_non_stdlib_import_denied(self) -> None:
        code = (
            "import nonexistent_third_party_xyz\n"
            "def f():\n"
            "    return 0\n"
        )
        result = run_one(
            code,
            input_repr=json.dumps([]),
            function_name="f",
            policy=_fast_policy(),
        )
        # Either policy_violation (import hook caught it) or raised (the
        # import failed with ImportError before our hook decides).
        self.assertIn(
            result.exit_code,
            {SandboxExitCode.POLICY_VIOLATION, SandboxExitCode.RAISED},
        )
        if result.exit_code is SandboxExitCode.POLICY_VIOLATION:
            self.assertTrue(
                any("import_denied" in v for v in result.policy_violations),
                f"expected import_denied, got {result.policy_violations}",
            )


class SandboxRunnerTimeoutTest(unittest.TestCase):
    def test_timeout_kills_runaway(self) -> None:
        code = "def f():\n    i = 0\n    while True:\n        i += 1\n"
        result = run_one(
            code,
            input_repr=json.dumps([]),
            function_name="f",
            policy=_fast_policy(timeout_ms=300),
        )
        self.assertEqual(result.exit_code, SandboxExitCode.TIMEOUT)
        self.assertEqual(result.output_kind, "timeout")


class SandboxRunnerDeterminismTest(unittest.TestCase):
    def test_deterministic_function_passes_check(self) -> None:
        code = "def f(n):\n    return sum(range(n))\n"
        result = run_one(
            code,
            input_repr=json.dumps([5]),
            function_name="f",
            policy=SandboxPolicy(
                timeout_ms=3000,
                determinism_check=True,
                memory_mb=128,
                cpu_seconds=2,
            ),
        )
        self.assertEqual(result.exit_code, SandboxExitCode.OK)
        self.assertTrue(result.determinism_check)
        self.assertEqual(result.output_repr, "10")

    def test_nondeterministic_function_is_caught(self) -> None:
        # Use uuid4 (which is not affected by PYTHONHASHSEED) to ensure
        # outputs differ between the two runs.
        code = (
            "import uuid\n"
            "def f():\n"
            "    return str(uuid.uuid4())\n"
        )
        result = run_one(
            code,
            input_repr=json.dumps([]),
            function_name="f",
            policy=SandboxPolicy(
                timeout_ms=3000,
                determinism_check=True,
                memory_mb=128,
                cpu_seconds=2,
            ),
        )
        self.assertEqual(result.exit_code, SandboxExitCode.NONDETERMINISTIC)
        self.assertFalse(result.determinism_check)


class SandboxResultSerializationTest(unittest.TestCase):
    def test_result_is_json_serializable(self) -> None:
        code = "def f(x):\n    return x * 2\n"
        result = run_one(
            code,
            input_repr=json.dumps([7]),
            function_name="f",
            policy=_fast_policy(),
        )
        payload = result.as_dict()
        encoded = json.dumps(payload)
        decoded = json.loads(encoded)
        self.assertEqual(decoded["schema_version"], SANDBOX_RESULT_SCHEMA_VERSION)
        self.assertEqual(decoded["exit_code"], "ok")
        self.assertEqual(decoded["output_repr"], "14")
        self.assertIn("policy", decoded)


class SandboxAPIContractTest(unittest.TestCase):
    def test_function_name_requires_input_repr(self) -> None:
        with self.assertRaises(SandboxPolicyError):
            run_one(
                "def f():\n    return 1\n",
                function_name="f",
                input_repr=None,
                policy=_fast_policy(),
            )


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
