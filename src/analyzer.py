"""Top-level analyzer: one function call -> one full Hinglish report.

Designed so the user can do:

    from src.analyzer import HinglishAnalyzer
    a = HinglishAnalyzer()                    # auto-loads checkpoints if present,
    print(a.analyze("yeh movi bakwaaaas h brooo").pretty())

If the trained checkpoints are missing (fresh clone, no training run yet)
the analyzer falls back to a rule-based mode: lexicon LID, lexicon-based
sentiment scoring, and an identity normalizer. The output schema is
identical, so downstream consumers don't branch on training state. This
keeps the demo runnable end-to-end without sitting through model training.

The full report contains:

    text, cleaned, preprocessing_edits   - what cleanup fired
    tokens, language_tags                - per-token LID
    code_switch                          - CMI, switches, dominance, burstiness
    normalized                           - English rendering
    sentiment, sentiment_scores          - 3-way label + soft probs
    token_importance                     - which tokens drove the sentiment
    summary                              - one human-readable paragraph
"""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass, field
from typing import Optional

import yaml

from .codeswitch import analyze_codeswitch
from .pipeline import CrossLingPipeline, CrossLingResult, whitespace_tokenize
from .text_norm import normalize_text


# --------------------------------------------------------------------------- #
# Lightweight rule-based fallback (used when checkpoints don't exist).
# These are NOT meant to be production replacements — they exist so the
# analyzer module is usable on a fresh clone before any training runs.
# --------------------------------------------------------------------------- #

_HIN_LEXICON = {
    "yaar", "bhai", "bhaiya", "yeh", "woh", "kya", "hai", "hain", "h",
    "kaisa", "kaisi", "kaise", "kal", "aaj", "abhi", "matlab", "acha",
    "achha", "achaa", "bakwaas", "bakwas", "mast", "ekdum", "ekdam",
    "bilkul", "bahut", "kaafi", "thoda", "thodi", "zyada", "kuch",
    "khaas", "ghatiya", "bekaar", "khush", "pyaara", "pyara", "mera",
    "meri", "tumhara", "hum", "mein", "mujhe", "tujhe", "ne", "ki",
    "ka", "ke", "ko", "se", "par", "phir", "lekin", "magar", "aur",
    "nahi", "nhi", "haan", "ji", "jaldi", "karo", "kar", "raha", "rahi",
    "rha", "rhi", "tha", "thi", "kheli", "khel", "milte",
    "seekh", "seekhna", "biryani", "yaaaar", "baat", "baate",
}

_NE_HEURISTIC = {"delhi", "mumbai", "rohan", "sachin", "india", "bangalore",
                 "kolkata", "chennai"}

_ENG_LEXICON = {
    "the", "a", "an", "is", "was", "were", "are", "be", "been", "i",
    "you", "we", "they", "he", "she", "it", "this", "that", "these",
    "those", "with", "and", "or", "but", "for", "to", "in", "on", "at",
    "of", "from", "by", "good", "bad", "movie", "song", "songs", "food",
    "team", "match", "performance", "love", "loved", "hate", "tasty",
    "tasteless", "rains", "rain", "weather", "amazing", "awesome",
    "brilliant", "okay", "average", "weak", "strong", "today",
    "yesterday", "tomorrow", "knock", "deadline", "project", "service",
    "experience", "delivery", "phone", "camera", "quality", "story",
    "ambience", "really", "completely", "totally", "super", "very",
    "loved", "irritating", "disappointed", "purchase", "recommend",
    "stunning", "borrring", "boring", "rude", "af", "duper", "all",
    "guy", "going", "back", "people", "things", "thing",
}

_POS_LEXICON = {
    "good", "great", "amazing", "awesome", "love", "loved", "tasty",
    "brilliant", "khush", "pyaara", "pyara", "achha", "acha", "achaa",
    "mast", "ekdum", "ekdam", "stunning", "fantastic", "wonderful",
    "exciting", "happy", "enjoy", "enjoyed", "super", "duper", "perfect",
    "best", "favorite", "favourite", "recommend",
}

_NEG_LEXICON = {
    "bad", "bakwaas", "bakwas", "ghatiya", "bekaar", "tasteless", "rude",
    "irritating", "irritated", "disappointed", "slow", "boring",
    "borrring", "weak", "waste", "hate", "hated", "worst", "horrible",
    "terrible", "shit", "trash", "flat", "mediocre", "ugly", "late",
}


def _rule_lid(tokens: list[str]) -> list[str]:
    tags: list[str] = []
    for tok in tokens:
        low = tok.lower()
        if not low.isalnum() and not any(c.isalnum() for c in low):
            tags.append("OTHER")
            continue
        if low in _NE_HEURISTIC or (tok and tok[0].isupper()
                                    and len(tok) > 2 and low not in _ENG_LEXICON
                                    and low not in _HIN_LEXICON):
            tags.append("NE")
        elif low in _HIN_LEXICON:
            tags.append("HIN")
        elif low in _ENG_LEXICON:
            tags.append("ENG")
        else:
            # Romanized Hindi heuristics: trailing -aa/-ee/-oo or 'h' ending
            if low.endswith(("aa", "ee", "oo", "ka", "ki", "ke", "na",
                             "ne", "se", "wala")):
                tags.append("HIN")
            else:
                tags.append("ENG")
    return tags


def _rule_sentiment(tokens: list[str]) -> tuple[str, dict[str, float]]:
    pos = sum(1 for t in tokens if t.lower() in _POS_LEXICON)
    neg = sum(1 for t in tokens if t.lower() in _NEG_LEXICON)
    if pos == 0 and neg == 0:
        return "NEUTRAL", {"POSITIVE": 0.33, "NEUTRAL": 0.34, "NEGATIVE": 0.33}
    total = pos + neg + 1.0
    p = (pos + 0.5) / total
    n = (neg + 0.5) / total
    neu = max(0.0, 1.0 - p - n)
    s = p + n + neu
    scores = {"POSITIVE": p / s, "NEGATIVE": n / s, "NEUTRAL": neu / s}
    label = max(scores, key=scores.get)
    return label, scores


def _rule_token_importance(tokens: list[str], label: str) -> list[dict]:
    lex = _POS_LEXICON if label == "POSITIVE" else (
        _NEG_LEXICON if label == "NEGATIVE" else set())
    raw = [1.0 if t.lower() in lex else 0.05 for t in tokens]
    s = sum(raw) or 1.0
    return [{"token": t, "importance": r / s} for t, r in zip(tokens, raw)]


# --------------------------------------------------------------------------- #
# Public analyzer
# --------------------------------------------------------------------------- #

@dataclass
class HinglishReport:
    text: str
    cleaned: str
    preprocessing_edits: list[str]
    tokens: list[str]
    language_tags: list[str]
    code_switch: dict
    normalized: str
    sentiment: str
    sentiment_score: float
    sentiment_scores: dict[str, float]
    token_importance: list[dict]
    backend: str
    summary: str = ""

    def to_dict(self) -> dict:
        return asdict(self)

    def pretty(self) -> str:
        token_view = "  ".join(f"{tok}/{tag}" for tok, tag
                               in zip(self.tokens, self.language_tags))
        cs = self.code_switch
        cs_line = (f"CMI={cs['cmi']:.1f}  switches={cs['switch_count']} "
                   f"({cs['switch_fraction']:.2f})  "
                   f"dominant={cs['dominant_language']} "
                   f"({cs['dominance_ratio']:.2f})  "
                   f"mixed={cs['is_code_mixed']}")
        edit_line = (", ".join(self.preprocessing_edits)
                     if self.preprocessing_edits else "none")
        ranked = sorted(self.token_importance,
                        key=lambda d: -d["importance"])[:5]
        imp_view = (",  ".join(f"{d['token']} ({d['importance']:.2f})"
                               for d in ranked) or "—")
        scores_view = "  ".join(f"{k}:{v:.2f}"
                                for k, v in self.sentiment_scores.items())

        bar = "─" * 64
        return (
            f"{bar}\n"
            f"  CrossLing Hinglish Analysis  [backend: {self.backend}]\n"
            f"{bar}\n"
            f"  Hinglish     : {self.text}\n"
            f"  ENGLISH      : {self.normalized}\n"
            f"\n"
            f"  Cleaned      : {self.cleaned}\n"
            f"  Edits        : {edit_line}\n"
            f"  Tokens       : {token_view}\n"
            f"  Code-switch  : {cs_line}\n"
            f"  Sentiment    : {self.sentiment}  (conf={self.sentiment_score:.2f})\n"
            f"               : {scores_view}\n"
            f"  Top tokens   : {imp_view}\n"
            f"  Summary      : {self.summary}\n"
            f"{bar}"
        )


def _summary(report_dict: dict) -> str:
    cs = report_dict["code_switch"]
    parts = []
    if cs["is_code_mixed"]:
        parts.append(
            f"Code-mixed Hinglish (CMI {cs['cmi']:.1f}); "
            f"{cs['switch_count']} switch point(s); "
            f"dominant language is {cs['dominant_language']} "
            f"at {cs['dominance_ratio']*100:.0f}%."
        )
    elif cs["dominant_language"]:
        parts.append(f"Largely monolingual {cs['dominant_language']}.")
    else:
        parts.append("No language-bearing tokens detected.")
    parts.append(
        f"Sentiment classified as {report_dict['sentiment']} "
        f"with confidence {report_dict['sentiment_score']:.2f}."
    )
    if report_dict["token_importance"]:
        top = sorted(report_dict["token_importance"],
                     key=lambda d: -d["importance"])[:3]
        parts.append(
            "Top contributing tokens: "
            + ", ".join(f"'{d['token']}'" for d in top) + "."
        )
    return " ".join(parts)


class HinglishAnalyzer:
    """One object, one call: ``analyzer.analyze(text)``.

    Tries to load the CrossLing checkpoints; falls back to rule-based
    analysis when they aren't present so the module is usable immediately.
    """

    def __init__(self, pipeline: Optional[CrossLingPipeline] = None,
                 backend: str = "neural") -> None:
        self.pipeline = pipeline
        self.backend = backend if pipeline else "rule-based"

    @classmethod
    def from_config(cls, config_path: str = "config.yaml") -> "HinglishAnalyzer":
        if not os.path.exists(config_path):
            return cls()
        try:
            with open(config_path, encoding="utf-8") as fh:
                cfg = yaml.safe_load(fh)
            paths = [cfg["lid"]["output_dir"], cfg["sentiment"]["output_dir"],
                     cfg["normalizer"]["output_dir"]]
            if all(os.path.isdir(p) and os.listdir(p) for p in paths):
                pipe = CrossLingPipeline.from_config(config_path)
                return cls(pipe, backend="neural")
        except Exception as exc:
            print(f"[analyzer] checkpoint load failed: {exc}; "
                  f"using rule-based fallback")
        return cls()

    def analyze(self, text: str, *,
                preprocess: bool = True) -> HinglishReport:
        """Analyse a single sentence.

        ``preprocess=False`` skips elongation/slang cleanup so the resulting
        ``tokens`` list aligns 1-to-1 with the input — useful for LID eval
        where we need to compare against pre-tokenised gold labels.
        """
        if self.pipeline is not None:
            res: CrossLingResult = self.pipeline.run(
                text, apply_normalization=preprocess)
            d = res.to_dict()
            d["backend"] = "neural"
        else:
            d = self._rule_based(text, preprocess=preprocess)
            d["backend"] = "rule-based"
        d["summary"] = _summary(d)
        return HinglishReport(
            text=d["text"],
            cleaned=d["cleaned"],
            preprocessing_edits=d["preprocessing_edits"],
            tokens=d["tokens"],
            language_tags=d["language_tags"],
            code_switch=d["code_switch"],
            normalized=d["normalized"],
            sentiment=d["sentiment"],
            sentiment_score=d["sentiment_score"],
            sentiment_scores=d["sentiment_scores"],
            token_importance=d["token_importance"],
            backend=d["backend"],
            summary=d["summary"],
        )

    def _rule_based(self, text: str, *, preprocess: bool = True) -> dict:
        if preprocess:
            trace = normalize_text(text)
        else:
            from .text_norm import NormalizationTrace
            trace = NormalizationTrace(original=text, cleaned=text, edits=[])
        tokens = whitespace_tokenize(trace.cleaned)
        if not tokens:
            return {
                "text": text, "cleaned": trace.cleaned,
                "preprocessing_edits": trace.edits,
                "tokens": [], "language_tags": [],
                "code_switch": analyze_codeswitch([], []).to_dict(),
                "normalized": "",
                "sentiment": "NEUTRAL",
                "sentiment_score": 1.0,
                "sentiment_scores": {"NEUTRAL": 1.0,
                                     "POSITIVE": 0.0, "NEGATIVE": 0.0},
                "token_importance": [],
            }
        tags = _rule_lid(tokens)
        cs = analyze_codeswitch(tokens, tags)
        label, scores = _rule_sentiment(tokens)
        importance = _rule_token_importance(tokens, label)
        return {
            "text": text,
            "cleaned": trace.cleaned,
            "preprocessing_edits": trace.edits,
            "tokens": tokens,
            "language_tags": tags,
            "code_switch": cs.to_dict(),
            "normalized": trace.cleaned,  # identity in rule-based mode
            "sentiment": label,
            "sentiment_score": scores[label],
            "sentiment_scores": scores,
            "token_importance": importance,
        }


def analyze(text: str, *, config_path: str = "config.yaml") -> HinglishReport:
    """Module-level convenience: load once, analyze a single sentence."""
    return HinglishAnalyzer.from_config(config_path).analyze(text)
