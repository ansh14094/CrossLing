"""CrossLing single-sentence analyzer CLI.

Usage:
    python -m scripts.analyze "yeh movi bakwaaaas h brooo"
    python -m scripts.analyze --interactive
    python -m scripts.analyze --json "yaar ekdum mast tha"

If trained checkpoints are available the analyzer uses them; otherwise it
falls back to a rule-based mode so the command works on a fresh clone.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.analyzer import HinglishAnalyzer


def _emit(report, as_json: bool) -> None:
    if as_json:
        print(json.dumps(report.to_dict(), ensure_ascii=False, indent=2))
    else:
        print(report.pretty())


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the CrossLing Hinglish analyzer on a sentence.")
    parser.add_argument("text", nargs="*",
                        help="Sentence to analyze (omit for --interactive).")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--json", action="store_true",
                        help="Emit a JSON report instead of the pretty view.")
    parser.add_argument("--interactive", action="store_true",
                        help="Read sentences from stdin until EOF / blank.")
    parser.add_argument("--backend", default="auto",
                        choices=["auto", "neural", "pretrained", "rule-based"],
                        help="auto: trained checkpoints if present, else "
                             "pretrained (HF Hub) if available, else rule-based. "
                             "pretrained: cardiffnlp sentiment + mined LID + "
                             "mT5/dictionary translator (no training required).")
    args = parser.parse_args()

    if args.backend == "pretrained":
        from src.pretrained import build_analyzer
        analyzer = build_analyzer()
    elif args.backend == "rule-based":
        analyzer = HinglishAnalyzer()
    elif args.backend == "neural":
        analyzer = HinglishAnalyzer.from_config(args.config)
    else:
        # auto: prefer trained checkpoints, then pretrained, then rule-based
        analyzer = HinglishAnalyzer.from_config(args.config)
        if analyzer.backend == "rule-based":
            try:
                from src.pretrained import build_analyzer
                analyzer = build_analyzer()
            except Exception as exc:
                print(f"[analyze] pretrained backend failed: {exc}; "
                      f"using rule-based")

    if args.interactive:
        print(f"CrossLing analyzer ({analyzer.backend}). "
              f"Ctrl-D / blank line to quit.")
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

    if args.text:
        _emit(analyzer.analyze(" ".join(args.text)), args.json)
        return

    # No args: run a few canned examples that exercise different code-mixing
    # regimes (low CMI, high CMI, noisy).
    examples = [
        "yeh movi bakwaaaas h brooo",
        "ekdum mast performance, I loved it!!!",
        "Sachin ne amazing knock kheli",
        "the project deadline close hai, jaldi karo",
        "OMGGG kya baat hai yaaaar",
    ]
    for text in examples:
        _emit(analyzer.analyze(text), args.json)


if __name__ == "__main__":
    main()
