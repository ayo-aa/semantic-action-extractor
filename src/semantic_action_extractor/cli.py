"""Command-line interface."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import tomllib
from typing import Sequence

from . import __version__
from .baseline import BaselineConfig, RuleBasedExtractor


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="semantic-action-extractor",
        description="Extract source-grounded action frames from text.",
    )
    parser.add_argument(
        "text",
        nargs="*",
        help="Text to extract. If omitted, read UTF-8 text from standard input.",
    )
    parser.add_argument(
        "--input-file",
        type=Path,
        help="Read UTF-8 input from a file instead of an argument or standard input.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        help="Optional TOML configuration containing a [baseline] table.",
    )
    parser.add_argument(
        "--pretty",
        action="store_true",
        help="Indent JSON output for people instead of emitting compact JSON.",
    )
    parser.add_argument("--version", action="version", version=__version__)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.input_file is not None and args.text:
        parser.error("provide either text or --input-file, not both")

    try:
        if args.input_file is not None:
            text = args.input_file.read_text(encoding="utf-8")
        elif args.text:
            text = " ".join(args.text)
        else:
            text = sys.stdin.read()
    except OSError as error:
        parser.error(str(error))

    if not text.strip():
        parser.error("input text cannot be empty")

    try:
        config = BaselineConfig.from_toml(args.config) if args.config else BaselineConfig()
    except (OSError, ValueError, tomllib.TOMLDecodeError) as error:
        parser.error(f"could not load configuration: {error}")

    result = RuleBasedExtractor(config).extract(text)
    json.dump(
        result.to_dict(),
        sys.stdout,
        ensure_ascii=False,
        indent=2 if args.pretty else None,
    )
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
