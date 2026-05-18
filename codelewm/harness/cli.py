"""Command-line entry point for CodeLeWM."""

from __future__ import annotations

import argparse
from collections.abc import Sequence

from codelewm import __version__


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
    parser.set_defaults(func=_print_help)
    return parser


def _print_help(args: argparse.Namespace) -> int:
    args.parser.print_help()
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.parser = parser
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
