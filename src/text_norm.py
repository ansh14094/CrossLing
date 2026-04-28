"""Robust text preprocessor for noisy Hinglish input.

Real-world Hinglish (memes, chat, tweets) carries patterns that synthetic data
rarely covers:
  * elongated vowels for emphasis ("bakwaaaas", "noooo")
  * chat-speak abbreviations ("h" -> "hai", "u" -> "you")
  * inconsistent casing and punctuation runs ("BRO!!!!")
  * common Hinglish slang variants ("bro", "yaar", "bhai")

This module normalizes such input *before* it hits the models, and reports
which transformations fired so downstream consumers (analyzer, report) can
explain the cleanup. Keep this conservative — over-normalization erases
affective signal that the sentiment model relies on.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field


_ELONGATED_RE = re.compile(r"(.)\1{2,}", re.UNICODE)
_PUNCT_RUN_RE = re.compile(r"([!?.,])\1{1,}")
_URL_RE = re.compile(r"https?://\S+|www\.\S+")
_MENTION_RE = re.compile(r"@\w+")
_HASHTAG_RE = re.compile(r"#(\w+)")


# Chat-speak / slang -> conventional Hinglish (kept lowercase)
_SLANG_MAP: dict[str, str] = {
    "u": "you",
    "ur": "your",
    "r": "are",
    "pls": "please",
    "plz": "please",
    "thx": "thanks",
    "tnx": "thanks",
    "k": "ok",
    "kk": "ok",
    "h": "hai",
    "hain": "hain",
    "n": "and",
    "&": "and",
    "y": "why",
    "abt": "about",
    "bcz": "because",
    "bcoz": "because",
    "coz": "because",
    "cuz": "because",
    "wd": "with",
    "w/": "with",
    "lyk": "like",
    "luv": "love",
    "gud": "good",
    "gr8": "great",
    "movi": "movie",
    "bro": "bro",
    "bruh": "bro",
    "broo": "bro",
}


@dataclass
class NormalizationTrace:
    original: str
    cleaned: str
    edits: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {"original": self.original,
                "cleaned": self.cleaned,
                "edits": list(self.edits)}


def collapse_elongations(text: str) -> tuple[str, int]:
    """`bakwaaaas` -> `bakwaas`; collapse runs of 3+ identical chars to 2.

    Two is the conservative cap: English words like 'good', 'see', 'tomorrow'
    legitimately repeat a letter, so we keep doubles intact.
    """
    count = 0

    def _sub(match: re.Match) -> str:
        nonlocal count
        count += 1
        return match.group(1) * 2

    return _ELONGATED_RE.sub(_sub, text), count


def collapse_punct_runs(text: str) -> tuple[str, int]:
    count = 0

    def _sub(match: re.Match) -> str:
        nonlocal count
        count += 1
        return match.group(1)

    return _PUNCT_RUN_RE.sub(_sub, text), count


def expand_slang(text: str) -> tuple[str, list[str]]:
    """Expand single-letter / chatspeak tokens. Returns the new text and the
    list of expansions actually applied (for the trace)."""
    fired: list[str] = []
    out_tokens: list[str] = []
    for tok in re.findall(r"\S+", text):
        # Strip surrounding punctuation for the lookup but keep it on output
        m = re.match(r"^(\W*)(.+?)(\W*)$", tok, flags=re.UNICODE)
        if not m:
            out_tokens.append(tok)
            continue
        pre, core, post = m.group(1), m.group(2), m.group(3)
        key = core.lower()
        if key in _SLANG_MAP and _SLANG_MAP[key] != key:
            replacement = _SLANG_MAP[key]
            fired.append(f"{core}->{replacement}")
            out_tokens.append(f"{pre}{replacement}{post}")
        else:
            out_tokens.append(tok)
    return " ".join(out_tokens), fired


def strip_urls_mentions(text: str) -> tuple[str, list[str]]:
    """Remove URLs and @mentions; keep hashtag content (drop the #)."""
    fired: list[str] = []
    new = text
    if _URL_RE.search(new):
        new = _URL_RE.sub("", new)
        fired.append("url-removed")
    if _MENTION_RE.search(new):
        new = _MENTION_RE.sub("", new)
        fired.append("mention-removed")
    if _HASHTAG_RE.search(new):
        new = _HASHTAG_RE.sub(r"\1", new)
        fired.append("hashtag-flattened")
    return new, fired


def normalize_text(text: str, *,
                   collapse_caps: bool = False,
                   apply_slang: bool = True) -> NormalizationTrace:
    """Run the full preprocessing chain and return a trace of what fired.

    `collapse_caps=True` lowercases everything (use for sentiment robustness
    studies). Default keeps casing because it carries emphasis signal.
    """
    edits: list[str] = []
    cleaned = text.strip()
    cleaned = re.sub(r"\s+", " ", cleaned)

    cleaned, social_edits = strip_urls_mentions(cleaned)
    edits.extend(social_edits)

    cleaned, n_elong = collapse_elongations(cleaned)
    if n_elong:
        edits.append(f"elongations-collapsed:{n_elong}")

    cleaned, n_punct = collapse_punct_runs(cleaned)
    if n_punct:
        edits.append(f"punct-runs-collapsed:{n_punct}")

    if apply_slang:
        cleaned, slang_edits = expand_slang(cleaned)
        if slang_edits:
            edits.append("slang:" + ",".join(slang_edits))

    if collapse_caps:
        cleaned = cleaned.lower()
        edits.append("lowercased")

    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return NormalizationTrace(original=text, cleaned=cleaned, edits=edits)
