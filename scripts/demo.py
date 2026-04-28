"""Run the trained CrossLing pipeline on a few sample inputs (or stdin).

Usage:
    python -m scripts.demo                       # canned examples
    python -m scripts.demo "yaar this is mast"   # one-shot
    python -m scripts.demo --interactive         # repl
    python -m scripts.demo --json "..."          # machine-readable output

This is a thin wrapper over scripts/analyze.py — kept for backward
compatibility. New work should target `scripts/analyze.py` /
`src.analyzer.HinglishAnalyzer`.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.analyzer import HinglishAnalyzer


_DEFAULT_INPUTS = [
    "yaar this movie was bakwaas but songs were good",
    "ekdum mast performance, I loved it",
    "traffic itna zyada hai, completely irritating",
    "Sachin ne amazing knock kheli",
    "yeh movi bakwaaaas h brooo",
]


def _emit(report, as_json: bool) -> None:
    if as_json:
        print(json.dumps(report.to_dict(), ensure_ascii=False))
    else:
        print(report.pretty())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("text", nargs="*", help="Input sentence(s).")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--interactive", action="store_true")
    parser.add_argument("--json", action="store_true",
                        help="Emit a JSON line per input.")
    args = parser.parse_args()

    analyzer = HinglishAnalyzer.from_config(args.config)

    if args.interactive:
        print(f"CrossLing interactive mode ({analyzer.backend}). "
              f"Ctrl-D / empty line to quit.")
        while True:
            try:
                text = input("> ").strip()
            except EOFError:
                print()
                break
            if not text:
                break
            _emit(analyzer.analyze(text), args.json)
        return

    inputs = [" ".join(args.text)] if args.text else _DEFAULT_INPUTS
    for text in inputs:
        _emit(analyzer.analyze(text), args.json)


if __name__ == "__main__":
    main()
