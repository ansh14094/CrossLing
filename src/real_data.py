"""Fetch real Hinglish datasets from the HuggingFace Hub and convert them
into the on-disk formats CrossLing already trains on.

Why this module exists: the synthetic corpus that ships in `src/data.py` is
small (~30 sentences) and the model's vocabulary stays narrow when trained
only on it. To learn real Hinglish patterns the trainer needs ~10k+ real
sentences. This module downloads three public datasets — no registration,
no scraping — and writes them to `data/`:

  data/
    sentimix/
      train.tsv            # text<TAB>LABEL    (sentiment classifier)
      train.conll          # token<TAB>LANG    (LID head)
      validation.tsv / .conll
      test.tsv / .conll
    english_to_hinglish/
      train.jsonl          # {"src": hinglish, "tgt": english}
      validation.jsonl
      test.jsonl

Datasets used:

  * RTT1/SentiMix
        SemEval-2020 Task 9 Hinglish twitter, with sentence-level
        positive/negative/neutral labels AND per-token Eng/Hin/O tags.
        This single source covers both sentiment and LID training.
        Train ~14k sentences, validation ~3k, test ~3k.

  * findnitai/english-to-hinglish
        ~189k Hinglish<->English parallel sentences for the normalizer.
        We split 90/5/5 train/val/test by hash of the source string so
        re-running produces the same split.

After running `python -m src.real_data --download`, the existing training
flow picks up the new data automatically — `scripts/train_all.py` looks at
the canonical paths first and falls back to the synthetic corpus only if
the real data is missing.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import Counter
from typing import Iterable

DATA_ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                         "data")

SENTIMIX_DIR = os.path.join(DATA_ROOT, "sentimix")
NORM_DIR = os.path.join(DATA_ROOT, "english_to_hinglish")


# Canonical paths the rest of the codebase reads. Keep these stable.
SENTIMIX_PATHS = {
    "train_sentiment": os.path.join(SENTIMIX_DIR, "train.tsv"),
    "val_sentiment": os.path.join(SENTIMIX_DIR, "validation.tsv"),
    "test_sentiment": os.path.join(SENTIMIX_DIR, "test.tsv"),
    "train_lid": os.path.join(SENTIMIX_DIR, "train.conll"),
    "val_lid": os.path.join(SENTIMIX_DIR, "validation.conll"),
    "test_lid": os.path.join(SENTIMIX_DIR, "test.conll"),
}

NORM_PATHS = {
    "train": os.path.join(NORM_DIR, "train.jsonl"),
    "val": os.path.join(NORM_DIR, "validation.jsonl"),
    "test": os.path.join(NORM_DIR, "test.jsonl"),
}


# --------------------------------------------------------------------------- #
# SentiMix
# --------------------------------------------------------------------------- #

def _iter_sentimix_sentences(rows: Iterable[dict],
                             id_to_label: dict[str, str] | None = None
                             ) -> Iterable[tuple[str, list[tuple[str, str]]]]:
    label: str | None = None
    tokens: list[tuple[str, str]] = []
    for row in rows:
        line = row.get("text", "")
        stripped = line.strip()
        if stripped.startswith("meta\t"):
            if label is not None and tokens:
                yield label, tokens
            parts = stripped.split("\t")
            sent_id = parts[1].strip() if len(parts) >= 2 else ""
            if len(parts) >= 3 and parts[-1].strip():
                label = parts[-1].strip().upper()
            elif id_to_label and sent_id in id_to_label:
                label = id_to_label[sent_id].upper()
            else:
                label = None
            tokens = []
            continue
        if not stripped:
            if label is not None and tokens:
                yield label, tokens
            label, tokens = None, []
            continue
        parts = line.split("\t")
        if len(parts) >= 2 and parts[0]:
            tokens.append((parts[0], parts[1]))
    if label is not None and tokens:
        yield label, tokens


def _extract_sentimix_test_answer_key(rows: Iterable[dict]) -> dict[str, str]:
    """Pull the ``Uid,Sentiment`` CSV header that prefixes the test split."""
    answers: dict[str, str] = {}
    for row in rows:
        line = row.get("text", "")
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.lower().startswith("uid,"):
            continue
        if "," in stripped and "\t" not in stripped:
            uid, _, label = stripped.partition(",")
            uid = uid.strip()
            label = label.strip()
            if uid and label and uid.isdigit():
                answers[uid] = label
            continue
        break
    return answers



def _write_sentimix_split(rows: Iterable[dict],
                          tsv_path: str, conll_path: str,
                          id_to_label: dict[str, str] | None = None
                          ) -> tuple[int, Counter]:
    os.makedirs(os.path.dirname(tsv_path), exist_ok=True)
    n = 0
    labels: Counter = Counter()
    valid_sentiment = {"POSITIVE", "NEGATIVE", "NEUTRAL"}
    with open(tsv_path, "w", encoding="utf-8") as tsv_fh, \
         open(conll_path, "w", encoding="utf-8") as conll_fh:
        for label, tokens in _iter_sentimix_sentences(rows, id_to_label):
            if label not in valid_sentiment:
                continue
            text = " ".join(tok for tok, _ in tokens).strip()
            if not text:
                continue
            tsv_fh.write(f"{text}\t{label}\n")
            for tok, lang in tokens:
                conll_fh.write(f"{tok}\t{lang}\n")
            conll_fh.write("\n")
            n += 1
            labels[label] += 1
    return n, labels



def download_sentimix(*, force: bool = False) -> dict[str, int]:
    """Pull RTT1/SentiMix and convert each split.
    
    Note: The public SentiMix dataset does not include sentiment labels for
    the test split (it's a held-out competition dataset). The test.tsv and
    test.conll files will be empty. Use train + validation splits instead.
    """
    from datasets import load_dataset

    if (not force
            and all(os.path.exists(p) for p in SENTIMIX_PATHS.values())
            and all(os.path.getsize(p) > 0 for p in SENTIMIX_PATHS.values())):
        print(f"[real_data] sentimix already present at {SENTIMIX_DIR}; "
              f"use --force to redownload")
        return {}

    print("[real_data] downloading RTT1/SentiMix from HuggingFace...")
    ds = load_dataset("RTT1/SentiMix")

    counts: dict[str, int] = {}
    for hf_split, tsv_key, conll_key in [
        ("train", "train_sentiment", "train_lid"),
        ("validation", "val_sentiment", "val_lid"),
        ("test", "test_sentiment", "test_lid"),
    ]:
        if hf_split not in ds:
            continue
        # The test split prefixes the CoNLL stream with a CSV answer key
        # ("Uid,Sentiment"); the train/val splits inline the label on the
        # meta line. _extract_sentimix_test_answer_key returns {} when the
        # CSV prefix isn't present, so the call is safe for all splits.
        # Materialize to list so we can iterate twice (once for answer key, once for data).
        split_data = list(ds[hf_split])
        id_to_label = _extract_sentimix_test_answer_key(split_data)
        n, label_dist = _write_sentimix_split(
            split_data,
            SENTIMIX_PATHS[tsv_key],
            SENTIMIX_PATHS[conll_key],
            id_to_label=id_to_label,
        )
        counts[hf_split] = n
        if n == 0 and hf_split == "test":
            print(f"[real_data]   sentimix {hf_split}: 0 sentences (no labels in "
                  f"public dataset — this is expected for held-out test splits)")
        else:
            print(f"[real_data]   sentimix {hf_split}: {n} sentences  "
                  f"labels={dict(label_dist)}")
    return counts



# --------------------------------------------------------------------------- #
# English <-> Hinglish parallel corpus (normalizer)
# --------------------------------------------------------------------------- #

def _stable_split(key: str) -> str:
    """Hash-based 90/5/5 split. Stable across runs and machines."""
    h = int(hashlib.md5(key.encode("utf-8")).hexdigest(), 16) % 100
    if h < 90:
        return "train"
    if h < 95:
        return "val"
    return "test"


def download_normalizer(*, force: bool = False, max_examples: int | None = 50_000
                        ) -> dict[str, int]:
    """Pull findnitai/english-to-hinglish and split 90/5/5.

    `max_examples` caps the corpus size (default 50k) — full 189k makes mT5
    fine-tuning slow on CPU/MPS. Pass ``None`` for the full set.
    """
    from datasets import load_dataset

    if (not force
            and all(os.path.exists(p) for p in NORM_PATHS.values())
            and all(os.path.getsize(p) > 0 for p in NORM_PATHS.values())):
        print(f"[real_data] english_to_hinglish already present at {NORM_DIR}; "
              f"use --force to redownload")
        return {}

    print("[real_data] downloading findnitai/english-to-hinglish from HF...")
    ds = load_dataset("findnitai/english-to-hinglish", split="train")

    os.makedirs(NORM_DIR, exist_ok=True)
    counts: dict[str, int] = {"train": 0, "val": 0, "test": 0}
    fhs = {split: open(NORM_PATHS[split], "w", encoding="utf-8")
           for split in counts}
    try:
        for i, row in enumerate(ds):
            if max_examples is not None and i >= max_examples:
                break
            tr = row.get("translation") or row
            src = (tr.get("hi_ng") or "").strip()
            tgt = (tr.get("en") or "").strip()
            if not src or not tgt:
                continue
            split = _stable_split(src)
            fhs[split].write(json.dumps(
                {"src": src, "tgt": tgt}, ensure_ascii=False) + "\n")
            counts[split] += 1
    finally:
        for fh in fhs.values():
            fh.close()

    for k, n in counts.items():
        print(f"[real_data]   english_to_hinglish {k}: {n}")
    return counts


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download real Hinglish datasets and convert them into "
                    "the formats CrossLing's loaders consume.")
    parser.add_argument("--force", action="store_true",
                        help="Redownload even if files already exist.")
    parser.add_argument("--skip", nargs="*", default=[],
                        choices=["sentimix", "normalizer"])
    parser.add_argument("--norm-cap", type=int, default=50_000,
                        help="Cap on parallel-corpus size for the normalizer "
                             "(default 50k; 0 = unlimited).")
    args = parser.parse_args()

    if "sentimix" not in args.skip:
        download_sentimix(force=args.force)
    if "normalizer" not in args.skip:
        cap = None if args.norm_cap == 0 else args.norm_cap
        download_normalizer(force=args.force, max_examples=cap)

    print("\n[real_data] done. Train with real data via:")
    print("    python -m scripts.train_all --use-real")
    print("...or pass explicit --lid-data / --sentiment-data / "
          "--normalizer-data flags.")


if __name__ == "__main__":
    main()
