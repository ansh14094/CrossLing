"""End-to-end CrossLing inference pipeline.

Glues the four stages together with explicit information flow:

    raw Hinglish text
        |
        |  -> normalize_text     (collapse elongations, expand chatspeak)
        v
    cleaned text
        |
        |  -> LID                (per-token language tags)
        v
    LID tags ──────► code-switch stats (CMI, switches, dominance)
        |
        v
    Normalizer (text + LID tags fed in as prefix features)
        |
        v
    Sentiment (on the cleaned code-mixed text)
        |
        v
    grad×input token importance for the predicted sentiment

Why sentiment runs on the cleaned code-mixed text rather than the English
normalization: the sentiment model is fine-tuned on code-mixed input;
back-translation drops affective markers ("ekdum", "bilkul"). The
normalizer's output is reported separately for human consumption.

Why the normalizer takes LID tags as features: monolingual mT5 prompts
("normalize hinglish: ...") give the model no signal about which tokens
are already English. Inlining `[HIN/ENG/HIN/...] <text>` is a cheap form
of multi-task conditioning — the LID head's output becomes a structured
feature for the seq2seq encoder rather than being thrown away.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field

import yaml

from .codeswitch import CodeSwitchStats, analyze_codeswitch
from .models import (
    LIDModel,
    LIDPrediction,
    NormalizerModel,
    SentimentModel,
    SentimentPrediction,
    resolve_device,
)
from .text_norm import NormalizationTrace, normalize_text


_TOKEN_RE = re.compile(r"\w+|[^\w\s]", re.UNICODE)


def whitespace_tokenize(text: str) -> list[str]:
    """Split text into word + punctuation tokens, preserving order."""
    return _TOKEN_RE.findall(text)


@dataclass
class CrossLingResult:
    text: str
    cleaned: str
    preprocessing_edits: list[str]
    tokens: list[str]
    language_tags: list[str]
    normalized: str
    sentiment: str
    sentiment_score: float
    sentiment_scores: dict[str, float]
    code_switch: dict
    token_importance: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)

    def pretty(self) -> str:
        token_view = "  ".join(f"{tok}/{tag}" for tok, tag
                               in zip(self.tokens, self.language_tags))
        cs = self.code_switch
        cs_line = (f"CMI={cs['cmi']:.1f}  switches={cs['switch_count']} "
                   f"({cs['switch_fraction']:.2f})  "
                   f"dominant={cs['dominant_language']} "
                   f"({cs['dominance_ratio']:.2f})")
        edit_line = (", ".join(self.preprocessing_edits)
                     if self.preprocessing_edits else "none")
        if self.token_importance:
            ranked = sorted(self.token_importance,
                            key=lambda d: -d["importance"])[:5]
            imp_view = "  ".join(f"{d['token']}({d['importance']:.2f})"
                                 for d in ranked)
        else:
            imp_view = "—"
        return (
            f"Input:        {self.text}\n"
            f"Cleaned:      {self.cleaned}\n"
            f"Edits:        {edit_line}\n"
            f"Tokens:       {token_view}\n"
            f"Code-switch:  {cs_line}\n"
            f"Normalized:   {self.normalized}\n"
            f"Sentiment:    {self.sentiment} ({self.sentiment_score:.2f})\n"
            f"Top tokens:   {imp_view}"
        )


class CrossLingPipeline:
    def __init__(self, lid: LIDModel, normalizer: NormalizerModel,
                 sentiment: SentimentModel) -> None:
        self.lid = lid
        self.normalizer = normalizer
        self.sentiment = sentiment

    @classmethod
    def from_checkpoints(cls, lid_path: str, normalizer_path: str,
                         sentiment_path: str,
                         device_preference: str = "auto") -> "CrossLingPipeline":
        device = resolve_device(device_preference)
        return cls(
            lid=LIDModel.load(lid_path, device),
            normalizer=NormalizerModel.load(normalizer_path, device),
            sentiment=SentimentModel.load(sentiment_path, device),
        )

    @classmethod
    def from_config(cls, config_path: str = "config.yaml") -> "CrossLingPipeline":
        with open(config_path, encoding="utf-8") as fh:
            cfg = yaml.safe_load(fh)
        return cls.from_checkpoints(
            lid_path=cfg["lid"]["output_dir"],
            normalizer_path=cfg["normalizer"]["output_dir"],
            sentiment_path=cfg["sentiment"]["output_dir"],
            device_preference=cfg.get("device", "auto"),
        )

    def run(self, text: str, *, with_importance: bool = True,
            apply_normalization: bool = True) -> CrossLingResult:
        if apply_normalization:
            trace: NormalizationTrace = normalize_text(text)
        else:
            trace = NormalizationTrace(original=text, cleaned=text, edits=[])

        cleaned = trace.cleaned
        tokens = whitespace_tokenize(cleaned)
        if not tokens:
            return CrossLingResult(
                text=text, cleaned=cleaned,
                preprocessing_edits=trace.edits,
                tokens=[], language_tags=[], normalized="",
                sentiment="NEUTRAL", sentiment_score=1.0,
                sentiment_scores={"NEUTRAL": 1.0},
                code_switch=analyze_codeswitch([], []).to_dict(),
                token_importance=[],
            )

        lid_pred: LIDPrediction = self.lid.predict(tokens)
        cs_stats: CodeSwitchStats = analyze_codeswitch(
            lid_pred.tokens, lid_pred.labels)

        normalized: str = self.normalizer.predict(
            cleaned, lid_tags=lid_pred.labels)

        sent_pred: SentimentPrediction = self.sentiment.predict(cleaned)

        importance: list[dict] = []
        if with_importance:
            try:
                words, scores, _, _ = self.sentiment.token_importance(cleaned)
                importance = [{"token": w, "importance": float(s)}
                              for w, s in zip(words, scores)]
            except Exception:
                # Importance is best-effort; never let it break the pipeline.
                importance = []

        return CrossLingResult(
            text=text,
            cleaned=cleaned,
            preprocessing_edits=trace.edits,
            tokens=lid_pred.tokens,
            language_tags=lid_pred.labels,
            normalized=normalized,
            sentiment=sent_pred.label,
            sentiment_score=sent_pred.score,
            sentiment_scores=sent_pred.scores,
            code_switch=cs_stats.to_dict(),
            token_importance=importance,
        )

    def __call__(self, text: str) -> CrossLingResult:
        return self.run(text)
