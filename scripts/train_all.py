"""Train all three CrossLing components in one go.

Usage:
    python -m scripts.train_all                       # auto: real data if
                                                      # already downloaded,
                                                      # else synthetic
    python -m scripts.train_all --use-real            # force real-data paths
                                                      # (downloads if missing)
    python -m scripts.train_all --no-real             # force synthetic
    python -m scripts.train_all --lid-data ...        # explicit override
                                --sentiment-data ...
                                --normalizer-data ...

Per-component data flags always win over auto-detection. Components whose
flag is omitted use the canonical data/ paths if present, then fall back
to the synthetic corpus.
"""

from __future__ import annotations

import argparse
import os
import sys

# Make `src` importable when running as a top-level script
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import yaml

from src.real_data import (
    NORM_PATHS,
    SENTIMIX_PATHS,
    download_normalizer,
    download_sentimix,
)
from src.train_lid import train_lid
from src.train_normalizer import train_normalizer
from src.train_sentiment import train_sentiment


def _existing(path: str | None) -> str | None:
    return path if path and os.path.exists(path) and os.path.getsize(path) > 0 else None


def _resolve_data_paths(args) -> tuple[str | None, str | None, str | None]:
    """Pick LID / sentiment / normalizer data paths in priority order:

    1. Explicit --*-data flag.
    2. If --use-real or auto-detect: canonical paths under data/.
    3. None (fall back to synthetic corpus inside the loaders).
    """
    auto_real = args.use_real
    if not args.no_real and not args.use_real:
        # Auto: turn on real data if it's already on disk.
        auto_real = os.path.exists(SENTIMIX_PATHS["train_sentiment"])

    lid = args.lid_data
    sent = args.sentiment_data
    norm = args.normalizer_data

    if auto_real:
        if args.use_real:
            # Materialise data if not present (downloader is idempotent).
            download_sentimix()
            download_normalizer()
        lid = lid or _existing(SENTIMIX_PATHS["train_lid"])
        sent = sent or _existing(SENTIMIX_PATHS["train_sentiment"])
        norm = norm or _existing(NORM_PATHS["train"])

    return lid, sent, norm


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--lid-data", default=None)
    parser.add_argument("--sentiment-data", default=None)
    parser.add_argument("--normalizer-data", default=None)
    parser.add_argument("--use-real", action="store_true",
                        help="Use the real Hinglish datasets under data/, "
                             "downloading them via src.real_data if missing.")
    parser.add_argument("--no-real", action="store_true",
                        help="Force synthetic-only (skip auto-detection).")
    parser.add_argument("--skip", nargs="*", default=[],
                        choices=["lid", "sentiment", "normalizer"])
    args = parser.parse_args()

    with open(args.config, encoding="utf-8") as fh:
        config = yaml.safe_load(fh)

    lid_path, sent_path, norm_path = _resolve_data_paths(args)

    print(f"[train_all] data sources -> "
          f"lid={lid_path or 'synthetic'}, "
          f"sentiment={sent_path or 'synthetic'}, "
          f"normalizer={norm_path or 'synthetic'}")

    if "lid" not in args.skip:
        print("=" * 60)
        print("Training LID")
        print("=" * 60)
        train_lid(config, data_path=lid_path)

    if "sentiment" not in args.skip:
        print("=" * 60)
        print("Training sentiment classifier")
        print("=" * 60)
        train_sentiment(config, data_path=sent_path)

    if "normalizer" not in args.skip:
        print("=" * 60)
        print("Training normalizer (seq2seq)")
        print("=" * 60)
        train_normalizer(config, data_path=norm_path)

    print("\nAll components trained. Run scripts/demo.py or scripts/evaluate.py.")


if __name__ == "__main__":
    main()
