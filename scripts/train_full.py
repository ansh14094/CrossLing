"""Full real-data training run for all three CrossLing components.

Designed to run end-to-end on a single CUDA GPU (T4 / 3090 / A100 / etc.)
with the real datasets in `data/`. On a T4 it takes ~20-30 minutes; on an
A100 closer to 10 minutes. The defaults match a 16-24 GB VRAM budget.

Usage:

    # 1. Make sure the real data is downloaded
    python -m src.real_data --norm-cap 0          # 0 = use full 189k pairs

    # 2. Train everything on GPU
    python -m scripts.train_full --device cuda

    # 3. Evaluate on the held-out test split
    python -m scripts.evaluate --backend neural

Tuning:

  --device cuda|mps|cpu        Override device (default: auto-detect)
  --light                      Smaller batch sizes for laptops / MPS
  --epochs N                   Override epochs for all three components
  --skip lid|sentiment|...     Skip components individually
  --norm-batch / --lid-batch / --sent-batch   Per-component batch overrides

Notes for the user (you):

  * On a fresh GPU machine: `pip install -r requirements.txt` first.
  * The sentiment model (XLM-R base) is the lightest; you can run just
    that without `--use-real` to sanity-check the pipeline (it'll use
    the noisy synthetic corpus baked into src/data.py).
  * If you have multiple GPUs, prefix the run with
    `CUDA_VISIBLE_DEVICES=0` to pin to a single device.
  * mT5-small has 300M params — on <12 GB VRAM, drop --norm-batch to 4
    and consider --norm-len 64.

The trained checkpoints land in `checkpoints/{lid,sentiment,normalizer}`,
which is the same location the analyzer + evaluate scripts read from.
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import yaml
import torch

from src.real_data import (NORM_PATHS, SENTIMIX_PATHS,
                            download_normalizer, download_sentimix)
from src.train_lid import train_lid
from src.train_normalizer import train_normalizer
from src.train_sentiment import train_sentiment


def _detect_device(preference: str) -> str:
    if preference != "auto":
        return preference
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--device", default="auto",
                        choices=["auto", "cuda", "mps", "cpu"])
    parser.add_argument("--light", action="store_true",
                        help="Laptop / MPS friendly batch sizes.")
    parser.add_argument("--epochs", type=int, default=None,
                        help="Override epochs for all three components.")
    parser.add_argument("--lid-batch", type=int, default=None)
    parser.add_argument("--sent-batch", type=int, default=None)
    parser.add_argument("--norm-batch", type=int, default=None)
    parser.add_argument("--norm-len", type=int, default=None,
                        help="Max source/target length for the normalizer.")
    parser.add_argument("--skip", nargs="*", default=[],
                        choices=["lid", "sentiment", "normalizer"])
    parser.add_argument("--no-download", action="store_true",
                        help="Don't auto-download missing data.")
    args = parser.parse_args()

    with open(args.config, encoding="utf-8") as fh:
        config = yaml.safe_load(fh)

    device = _detect_device(args.device)
    config["device"] = device
    print(f"[train_full] device = {device}")

    if not args.no_download:
        if not os.path.exists(SENTIMIX_PATHS["train_sentiment"]):
            download_sentimix()
        if not os.path.exists(NORM_PATHS["train"]):
            download_normalizer()

    if args.light:
        # MPS / 8 GB VRAM-friendly defaults.
        config["lid"]["batch_size"] = args.lid_batch or 16
        config["sentiment"]["batch_size"] = args.sent_batch or 16
        config["normalizer"]["batch_size"] = args.norm_batch or 4
        config["normalizer"]["max_source_length"] = args.norm_len or 64
        config["normalizer"]["max_target_length"] = args.norm_len or 64
    else:
        if args.lid_batch:
            config["lid"]["batch_size"] = args.lid_batch
        if args.sent_batch:
            config["sentiment"]["batch_size"] = args.sent_batch
        if args.norm_batch:
            config["normalizer"]["batch_size"] = args.norm_batch
        if args.norm_len:
            config["normalizer"]["max_source_length"] = args.norm_len
            config["normalizer"]["max_target_length"] = args.norm_len

    if args.epochs:
        for k in ("lid", "sentiment", "normalizer"):
            config[k]["epochs"] = args.epochs

    print("[train_full] effective config:")
    for k in ("lid", "sentiment", "normalizer"):
        c = config[k]
        print(f"  {k:10s} backbone={c['backbone']}  bs={c['batch_size']}  "
              f"epochs={c['epochs']}  lr={c['lr']}")

    lid_data = SENTIMIX_PATHS["train_lid"]
    sent_data = SENTIMIX_PATHS["train_sentiment"]
    norm_data = NORM_PATHS["train"]

    if "lid" not in args.skip:
        print("=" * 60)
        print("Training LID on", lid_data)
        print("=" * 60)
        train_lid(config, data_path=lid_data)

    if "sentiment" not in args.skip:
        print("=" * 60)
        print("Training sentiment on", sent_data)
        print("=" * 60)
        train_sentiment(config, data_path=sent_data)

    if "normalizer" not in args.skip:
        print("=" * 60)
        print("Training normalizer on", norm_data)
        print("=" * 60)
        train_normalizer(config, data_path=norm_data)

    print("\n[train_full] done. Checkpoints in checkpoints/.")
    print("Run evaluation:    python -m scripts.evaluate --backend neural")
    print("Run a sentence:    python -m scripts.analyze 'yeh movi bakwaaaas h brooo'")


if __name__ == "__main__":
    main()
