"""Datasets for the three CrossLing tasks: LID, sentiment, normalization.

Each task has:
  * a torch Dataset that produces model-ready tensors
  * a loader that reads either a real file (CoNLL/TSV/JSON) or falls back to
    a small built-in synthetic corpus, so the full pipeline trains end-to-end
    without dataset-registration roadblocks.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any

import torch
from torch.utils.data import Dataset


# --------------------------------------------------------------------------- #
# Synthetic Hinglish corpora — small but real enough to demonstrate every
# stage of the pipeline. Replace with LINCE / SentiMix / PHINC for production.
# --------------------------------------------------------------------------- #

_LID_SAMPLES: list[list[tuple[str, str]]] = [
    [("yaar", "HIN"), ("this", "ENG"), ("movie", "ENG"), ("was", "ENG"),
     ("bakwaas", "HIN"), ("but", "ENG"), ("songs", "ENG"), ("were", "ENG"),
     ("good", "ENG")],
    [("kal", "HIN"), ("I", "ENG"), ("went", "ENG"), ("to", "ENG"),
     ("Delhi", "NE"), ("with", "ENG"), ("Rohan", "NE")],
    [("matlab", "HIN"), ("seriously", "ENG"), ("?", "OTHER"),
     ("kya", "HIN"), ("baat", "HIN"), ("hai", "HIN")],
    [("the", "ENG"), ("food", "ENG"), ("was", "ENG"), ("ekdum", "HIN"),
     ("mast", "HIN"), ("!", "OTHER")],
    [("bhai", "HIN"), ("traffic", "ENG"), ("itna", "HIN"), ("zyada", "HIN"),
     ("hai", "HIN"), ("today", "ENG")],
    [("mujhe", "HIN"), ("python", "ENG"), ("seekhna", "HIN"),
     ("hai", "HIN")],
    [("Mumbai", "NE"), ("ki", "HIN"), ("rains", "ENG"), ("are", "ENG"),
     ("crazy", "ENG")],
    [("acha", "HIN"), ("then", "ENG"), ("kal", "HIN"), ("milte", "HIN"),
     ("hain", "HIN")],
    [("the", "ENG"), ("project", "ENG"), ("deadline", "ENG"), ("close", "ENG"),
     ("hai", "HIN"), (",", "OTHER"), ("jaldi", "HIN"), ("karo", "HIN")],
    [("Sachin", "NE"), ("ne", "HIN"), ("amazing", "ENG"), ("knock", "ENG"),
     ("kheli", "HIN")],
    [("OMG", "ENG"), ("this", "ENG"), ("biryani", "ENG"), ("ekdam", "HIN"),
     ("tasty", "ENG"), ("hai", "HIN")],
    [("aaj", "HIN"), ("the", "ENG"), ("weather", "ENG"), ("kaafi", "HIN"),
     ("acha", "HIN"), ("hai", "HIN")],
    # noisy/elongated/slang variants
    [("yeh", "HIN"), ("movi", "ENG"), ("bakwaaaas", "HIN"), ("h", "HIN"),
     ("brooo", "ENG")],
    [("OMG", "ENG"), ("kya", "HIN"), ("baat", "HIN"), ("hai", "HIN"),
     ("yaaaar", "HIN"), ("!", "OTHER")],
    [("super", "ENG"), ("duper", "ENG"), ("achaa", "HIN"), ("experience", "ENG"),
     ("hua", "HIN")],
    [("delivery", "ENG"), ("guy", "ENG"), ("was", "ENG"), ("rude", "ENG"),
     ("af", "ENG"), (",", "OTHER"), ("bilkul", "HIN"), ("bekaar", "HIN")],
    [("phone", "ENG"), ("heat", "ENG"), ("ho", "HIN"), ("rha", "HIN"),
     ("h", "HIN")],
]

_SENTIMENT_SAMPLES: list[tuple[str, str]] = [
    ("yaar this movie was bakwaas but songs were good", "NEUTRAL"),
    ("ekdum mast performance, I loved it", "POSITIVE"),
    ("traffic itna zyada hai, completely irritating", "NEGATIVE"),
    ("the food was bilkul tasteless", "NEGATIVE"),
    ("aaj ka match was super exciting", "POSITIVE"),
    ("kuch khaas nahi tha, just an okay film", "NEUTRAL"),
    ("bro this dress is too pyaara", "POSITIVE"),
    ("service was bahut slow, never going back", "NEGATIVE"),
    ("matlab seriously kya baat hai, brilliant work", "POSITIVE"),
    ("biryani was tasty but the ambience was mediocre", "NEUTRAL"),
    ("bilkul bekaar experience, paisa waste", "NEGATIVE"),
    ("the team played really achha aaj", "POSITIVE"),
    ("nothing special, average si film thi", "NEUTRAL"),
    ("yeh phone bahut accha hai, I recommend it", "POSITIVE"),
    ("delivery itni late thi, totally disappointed", "NEGATIVE"),
    ("songs are catchy but story thodi weak hai", "NEUTRAL"),
    ("I am really khush with this purchase", "POSITIVE"),
    ("camera quality bilkul ghatiya hai", "NEGATIVE"),
    # ---- noisy real-world style (elongations, slang, casing) ----
    ("yeh movi bakwaaaas h brooo, paisa waste", "NEGATIVE"),
    ("OMGGG ekdum maaaast yaar, loved itttt!!!", "POSITIVE"),
    ("kya bekaaar service thi, never goin back", "NEGATIVE"),
    ("super duper achaa experience hua aaj", "POSITIVE"),
    ("bro this is sooo borrring, kuch nahi hua", "NEGATIVE"),
    ("matlab seriously?? amazinggg vibes only", "POSITIVE"),
    ("phone heat ho rha h baad mein hang bhi krta hai", "NEGATIVE"),
    ("biryani okay thi, kuch khaas spicy nhi tha", "NEUTRAL"),
    ("OMG the views were stunning yaaaar", "POSITIVE"),
    ("delivery guy was rude af, ekdum bekaar", "NEGATIVE"),
    ("songs decent the but acting kaafi flat lagi", "NEUTRAL"),
    ("LOVE THIS!!!!! ekdum dil khush kar diya", "POSITIVE"),
]

_NORM_SAMPLES: list[tuple[str, str]] = [
    ("yaar this movie was bakwaas but songs were good",
     "friend this movie was rubbish but songs were good"),
    ("kal I went to Delhi with Rohan",
     "yesterday I went to Delhi with Rohan"),
    ("the food was ekdum mast",
     "the food was absolutely awesome"),
    ("bhai traffic itna zyada hai today",
     "brother the traffic is so heavy today"),
    ("mujhe python seekhna hai",
     "I want to learn python"),
    ("Mumbai ki rains are crazy",
     "Mumbai 's rains are crazy"),
    ("acha then kal milte hain",
     "okay then we will meet tomorrow"),
    ("the project deadline close hai, jaldi karo",
     "the project deadline is close, hurry up"),
    ("Sachin ne amazing knock kheli",
     "Sachin played an amazing knock"),
    ("aaj the weather kaafi acha hai",
     "today the weather is quite good"),
    ("biryani ekdam tasty hai",
     "the biryani is absolutely tasty"),
    ("matlab seriously kya baat hai",
     "I mean seriously what a thing"),
]


# --------------------------------------------------------------------------- #
# LID: token-level sequence labeling
# --------------------------------------------------------------------------- #

@dataclass
class LIDExample:
    tokens: list[str]
    labels: list[str]


_LINCE_LID_MAP = {
    "lang1": "ENG", "en": "ENG", "eng": "ENG",
    "lang2": "HIN", "hi": "HIN", "hin": "HIN",
    "ne": "NE", "named": "NE",
    "other": "OTHER", "mixed": "OTHER", "ambiguous": "OTHER",
    "fw": "OTHER", "univ": "OTHER", "unk": "OTHER",
    # SentiMix-Hinglish (SemEval-2020) and similar use single-letter "O".
    "o": "OTHER",
}


def _normalize_lid_label(raw: str) -> str:
    raw_low = raw.strip().lower()
    if raw_low in _LINCE_LID_MAP:
        return _LINCE_LID_MAP[raw_low]
    upper = raw.strip().upper()
    if upper in {"ENG", "HIN", "OTHER", "NE"}:
        return upper
    return "OTHER"


def load_lid_examples(path: str | None = None) -> list[LIDExample]:
    """Load CoNLL-style LID data; fall back to the synthetic corpus.

    Recognises both the CrossLing canonical labels (``ENG/HIN/OTHER/NE``) and
    LINCE / SentiMix-style raw labels (``lang1/lang2/ne/other/mixed/...``).
    Sentences may optionally start with a ``meta\\t...`` header line (SentiMix
    convention); those are skipped here so this loader works on the same files
    the sentiment loader consumes.
    """
    if path and os.path.exists(path):
        examples: list[LIDExample] = []
        tokens: list[str] = []
        labels: list[str] = []
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                line = line.rstrip("\n")
                if not line.strip():
                    if tokens:
                        examples.append(LIDExample(tokens, labels))
                        tokens, labels = [], []
                    continue
                parts = line.split("\t")
                if parts and parts[0].lower() == "meta":
                    continue
                if len(parts) < 2:
                    continue
                tokens.append(parts[0])
                labels.append(_normalize_lid_label(parts[1]))
        if tokens:
            examples.append(LIDExample(tokens, labels))
        return examples
    return [LIDExample([t for t, _ in s], [l for _, l in s]) for s in _LID_SAMPLES]


class LIDDataset(Dataset):
    """Token-classification dataset.

    Aligns word-level labels to subword pieces by labelling only the first
    piece of each word; subsequent pieces and special tokens get -100 so the
    cross-entropy loss ignores them.
    """

    def __init__(self, examples, tokenizer, label2id: dict[str, int],
                 max_length: int = 128) -> None:
        self.examples = examples
        self.tokenizer = tokenizer
        self.label2id = label2id
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        ex = self.examples[idx]
        encoding = self.tokenizer(
            ex.tokens,
            is_split_into_words=True,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
        )
        word_ids = encoding.word_ids(batch_index=0)
        aligned: list[int] = []
        last_word: int | None = None
        for word_id in word_ids:
            if word_id is None:
                aligned.append(-100)
            elif word_id != last_word:
                label = ex.labels[word_id]
                aligned.append(self.label2id.get(label, self.label2id.get("OTHER", 0)))
            else:
                aligned.append(-100)
            last_word = word_id

        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": torch.tensor(aligned, dtype=torch.long),
        }


# --------------------------------------------------------------------------- #
# Sentiment: sequence classification
# --------------------------------------------------------------------------- #

def load_sentiment_examples(path: str | None = None) -> list[tuple[str, str]]:
    """Load sentiment data; fall back to the synthetic corpus.

    Supports two file layouts:
      * TSV: `text<TAB>label` per line (CrossLing default)
      * SentiMix CoNLL: blank-line separated sentences whose first line is
        ``meta\\tID\\tLABEL`` followed by ``token<TAB>lang`` lines. The tokens
        are joined back into whitespace text and the meta-label is used.
    """
    if path and os.path.exists(path):
        if path.endswith(".conll") or path.endswith(".txt"):
            sentimix = _load_sentimix_conll(path)
            if sentimix:
                return sentimix
        out: list[tuple[str, str]] = []
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                parts = line.split("\t")
                if len(parts) >= 2:
                    out.append((parts[0], parts[1].upper()))
        return out
    return list(_SENTIMENT_SAMPLES)


def _load_sentimix_conll(path: str) -> list[tuple[str, str]]:
    """SemEval-2020 SentiMix file format -> (text, LABEL) pairs."""
    out: list[tuple[str, str]] = []
    label: str | None = None
    tokens: list[str] = []
    with open(path, encoding="utf-8") as fh:
        for raw in fh:
            line = raw.rstrip("\n")
            if not line.strip():
                if tokens and label:
                    out.append((" ".join(tokens), label.upper()))
                tokens, label = [], None
                continue
            parts = line.split("\t")
            if parts[0].lower() == "meta" and len(parts) >= 3:
                label = parts[-1]
                continue
            if not parts[0]:
                continue
            tokens.append(parts[0])
    if tokens and label:
        out.append((" ".join(tokens), label.upper()))
    return out


class SentimentDataset(Dataset):
    def __init__(self, examples, tokenizer, label2id: dict[str, int],
                 max_length: int = 128) -> None:
        self.examples = examples
        self.tokenizer = tokenizer
        self.label2id = label2id
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        text, label = self.examples[idx]
        enc = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
        )
        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "labels": torch.tensor(self.label2id[label], dtype=torch.long),
        }


# --------------------------------------------------------------------------- #
# Normalization: Hinglish -> English seq2seq
# --------------------------------------------------------------------------- #

def load_normalization_examples(path: str | None = None) -> list[tuple[str, str]]:
    """Load JSONL with {"src":..., "tgt":...} or fall back to synthetic data."""
    if path and os.path.exists(path):
        out: list[tuple[str, str]] = []
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                out.append((obj["src"], obj["tgt"]))
        return out
    return list(_NORM_SAMPLES)


class NormalizationDataset(Dataset):
    def __init__(self, examples, tokenizer, max_source_length: int = 128,
                 max_target_length: int = 128, source_prefix: str = "") -> None:
        self.examples = examples
        self.tokenizer = tokenizer
        self.max_source_length = max_source_length
        self.max_target_length = max_target_length
        self.source_prefix = source_prefix

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        src, tgt = self.examples[idx]
        src_ids = self.tokenizer(
            self.source_prefix + src,
            truncation=True,
            max_length=self.max_source_length,
            padding="max_length",
            return_tensors="pt",
        )
        tgt_ids = self.tokenizer(
            tgt,
            truncation=True,
            max_length=self.max_target_length,
            padding="max_length",
            return_tensors="pt",
        )
        labels = tgt_ids["input_ids"].squeeze(0)
        labels = labels.masked_fill(labels == self.tokenizer.pad_token_id, -100)
        return {
            "input_ids": src_ids["input_ids"].squeeze(0),
            "attention_mask": src_ids["attention_mask"].squeeze(0),
            "labels": labels,
        }


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

def split_examples(examples, val_ratio: float = 0.2, seed: int = 42):
    import random
    rng = random.Random(seed)
    items = list(examples)
    rng.shuffle(items)
    n_val = max(1, int(len(items) * val_ratio))
    return items[n_val:], items[:n_val]
