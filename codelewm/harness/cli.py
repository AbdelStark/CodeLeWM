"""Command-line entry point for CodeLeWM."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

from codelewm import __version__
from codelewm.harness.scorer import ScoreError, load_scorer


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="codelewm",
        description=(
            "CodeLeWM command-line interface. Implementation commands are "
            "introduced incrementally by the spec-tracked issues."
        ),
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {__version__}",
    )
    subparsers = parser.add_subparsers(dest="command")
    score = subparsers.add_parser("score", help="score one candidate after-state")
    score.add_argument("--before", type=Path, required=True, help="before-state Python file")
    score.add_argument("--instruction", required=True, help="instruction text or path to a text file")
    score.add_argument("--candidate", type=Path, required=True, help="candidate after-state Python file")
    score.add_argument("--checkpoint", type=Path, required=True, help="checkpoint file")
    score.add_argument("--device", default="auto", choices=("cpu", "cuda", "mps", "auto"))
    score.add_argument("--json", action="store_true", help="emit JSON output")
    score.set_defaults(func=_score_command)
    parser.set_defaults(func=_print_help)
    return parser


def _print_help(args: argparse.Namespace) -> int:
    args.parser.print_help()
    return 0


def _score_command(args: argparse.Namespace) -> int:
    try:
        instruction = _instruction_arg_to_text(args.instruction)
        scorer = load_scorer(args.checkpoint, device=args.device)
        result = scorer.score_files(
            before=args.before,
            instruction=instruction,
            candidate=args.candidate,
        )
    except ScoreError as exc:
        print(str(exc), file=sys.stderr)
        return 2

    if args.json:
        print(json.dumps(result.to_dict(), indent=2, sort_keys=True))
    else:
        print(f"candidate: {result.candidate}")
        print(f"transition_energy: {result.transition_energy:.6g}")
        print(f"final_score: {result.final_score:.6g}")
    return 0


def _instruction_arg_to_text(value: str) -> str:
    path = Path(value)
    if path.exists():
        if not path.is_file():
            raise ScoreError(f"instruction path is not a file: {path}")
        return path.read_text()
    return value


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.parser = parser
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
