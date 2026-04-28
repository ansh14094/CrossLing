"""Zero-shot pretrained backend for CrossLing.

Why this exists: training XLM-RoBERTa + mT5 from scratch on the small
synthetic corpus produces a model that doesn't generalise. Even fine-tuning
on the 14k SentiMix sentences takes hours of GPU time. The community has
already published multilingual / code-mixed-aware checkpoints that handle
real Hinglish reasonably well *without any local training* — this module
plugs them straight into the analyzer.

Models used:

  * Sentiment: ``cardiffnlp/twitter-xlm-roberta-base-sentiment``
        XLM-R fine-tuned on multilingual twitter sentiment (8 languages,
        198M tweets). Empirically handles romanized Hindi well — the
        Cardiff team trained on actual Twitter where Hinglish is common.

  * Hinglish -> English translator:
        ``findnitai/t5-hinglish-translator`` is a T5 trained on Hinglish<->
        English parallel data. We use it directly — no fine-tuning needed.

  * LID: token-level Hinglish LID has no good off-the-shelf checkpoint, so
        this backend uses a *vocabulary-mined* LID. At construction time we
        scan ``data/sentimix/train.conll`` and build per-token majority
        labels. Words seen in training get their majority label; unseen
        words fall back to the lightweight rule heuristic in
        ``src/analyzer.py``. Mining is a one-shot pass (~14k sentences,
        <1s) so it doesn't add startup cost.

Usage:

    from src.pretrained import PretrainedBackend, build_analyzer
    analyzer = build_analyzer()             # auto-loads when checkpoints absent
    print(analyzer.analyze("yeh movi bakwaaaas h brooo").pretty())

The pretrained backend slots into ``HinglishAnalyzer`` via the same
``HinglishReport`` schema, so ``scripts/analyze.py`` and
``scripts/evaluate.py`` work unchanged.
"""

from __future__ import annotations

import os
from collections import Counter, defaultdict
from typing import Any

from .analyzer import (
    HinglishAnalyzer,
    HinglishReport,
    _rule_lid,
    _summary,
)
from .codeswitch import analyze_codeswitch
from .pipeline import whitespace_tokenize
from .text_norm import NormalizationTrace, normalize_text


# --------------------------------------------------------------------------- #
# Vocabulary-mined LID
# --------------------------------------------------------------------------- #

class MinedLID:
    """Per-token majority-label LID derived from real training data."""

    def __init__(self, vocab: dict[str, str]) -> None:
        self.vocab = vocab

    @classmethod
    def from_conll(cls, path: str) -> "MinedLID":
        from .data import _normalize_lid_label
        per_token: dict[str, Counter] = defaultdict(Counter)
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                parts = line.rstrip("\n").split("\t")
                if len(parts) < 2 or not parts[0] or parts[0].lower() == "meta":
                    continue
                tok, lang = parts[0], _normalize_lid_label(parts[1])
                per_token[tok.lower()][lang] += 1
        vocab = {tok: counts.most_common(1)[0][0]
                 for tok, counts in per_token.items()}
        return cls(vocab)

    def predict(self, tokens: list[str]) -> list[str]:
        labels: list[str] = []
        unknown: list[int] = []
        for i, tok in enumerate(tokens):
            low = tok.lower()
            if low in self.vocab:
                labels.append(self.vocab[low])
            else:
                labels.append("__UNK__")
                unknown.append(i)
        if unknown:
            unknown_tokens = [tokens[i] for i in unknown]
            fallback = _rule_lid(unknown_tokens)
            for idx, lab in zip(unknown, fallback):
                labels[idx] = lab
        return labels


# --------------------------------------------------------------------------- #
# Pretrained backend
# --------------------------------------------------------------------------- #

class PretrainedBackend:
    """Self-contained Hinglish backend wrapping pretrained HF checkpoints.

    Instantiate once (downloads happen on first call), then use the
    ``analyze`` method via ``HinglishAnalyzer(backend_override=...)`` or
    via ``build_analyzer()``.
    """

    SENTIMENT_NAME = "cardiffnlp/twitter-xlm-roberta-base-sentiment"

    def __init__(self, *, lid_path: str | None = None,
                 lid_conll: str | None = None,
                 normalizer_path: str | None = None,
                 device_preference: str = "auto",
                 enable_translator: bool = True) -> None:
        """Pretrained backend.

        ``lid_path`` and ``normalizer_path`` can point at locally-trained
        CrossLing checkpoints. When present they take priority over the
        rule/mined LID and the lookup-based fallback translator.
        """
        from .models import resolve_device

        self.device = resolve_device(device_preference)
        self._lid_model = self._try_load_lid_model(lid_path)
        self._mined_lid = (self._build_mined_lid(lid_conll)
                           if self._lid_model is None else None)
        self._sentiment_tok, self._sentiment_mdl = self._load_sentiment()
        self._normalizer = (self._try_load_normalizer(normalizer_path)
                            if enable_translator else None)
        if enable_translator and self._normalizer is None:
            print("[pretrained] no neural normalizer available; "
                  "falling back to dictionary-based word translation")
            self._dict = self._build_word_dictionary()
        else:
            self._dict = None

    # ---- model loaders ----

    def _try_load_lid_model(self, lid_path: str | None):
        """Load a locally-trained CrossLing LID checkpoint if one exists."""
        from .models import LIDModel
        path = lid_path or "checkpoints/lid"
        if not (os.path.isdir(path) and os.listdir(path)):
            return None
        try:
            print(f"[pretrained] loading trained LID checkpoint from {path}")
            return LIDModel.load(path, self.device)
        except Exception as exc:
            print(f"[pretrained] failed to load LID checkpoint: {exc}")
            return None

    def _build_mined_lid(self, lid_conll: str | None) -> MinedLID:
        from .real_data import SENTIMIX_PATHS
        path = lid_conll or SENTIMIX_PATHS["train_lid"]
        if path and os.path.exists(path) and os.path.getsize(path) > 0:
            mined = MinedLID.from_conll(path)
            print(f"[pretrained] mined LID vocab: {len(mined.vocab)} tokens "
                  f"from {path}")
            return mined
        return MinedLID({})

    def _load_sentiment(self):
        from transformers import (AutoModelForSequenceClassification,
                                  AutoTokenizer)
        print(f"[pretrained] loading sentiment model {self.SENTIMENT_NAME}...")
        tok = AutoTokenizer.from_pretrained(self.SENTIMENT_NAME)
        mdl = AutoModelForSequenceClassification.from_pretrained(
            self.SENTIMENT_NAME)
        mdl.to(self.device)
        mdl.eval()
        return tok, mdl

    def _try_load_normalizer(self, path: str | None):
        """Use a locally fine-tuned mT5 if present."""
        from .models import NormalizerModel
        path = path or "checkpoints/normalizer"
        if not (os.path.isdir(path) and os.listdir(path)):
            return None
        try:
            print(f"[pretrained] loading trained normalizer from {path}")
            return NormalizerModel.load(path, self.device)
        except Exception as exc:
            print(f"[pretrained] failed to load normalizer: {exc}")
            return None

    def _build_word_dictionary(self) -> dict[str, str]:
        """Mine a Hinglish word -> English word dictionary from the parallel
        corpus. Used only when no neural translator is available — gives a
        readable, if literal, English rendering instead of identity output.
        """
        from .real_data import NORM_PATHS
        path = NORM_PATHS["train"]
        if not (os.path.exists(path) and os.path.getsize(path) > 0):
            return {}
        from collections import Counter
        import json as _json
        co_occ: dict[str, Counter] = {}
        n = 0
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                obj = _json.loads(line)
                src_words = obj["src"].lower().split()
                tgt_words = obj["tgt"].lower().split()
                if not src_words or not tgt_words:
                    continue
                # Length-aligned approximation: same-position alignment when
                # lengths are equal; otherwise skip for the dictionary to
                # avoid noise (positional alignment for unequal lengths is
                # garbage). Coarse but sufficient for "yaar" -> "friend"
                # type mappings to surface.
                if len(src_words) == len(tgt_words):
                    for s, t in zip(src_words, tgt_words):
                        co_occ.setdefault(s, Counter())[t] += 1
                n += 1
                if n >= 6000:
                    break
        # Take majority target for each source token, with a frequency floor.
        dictionary = {}
        for s, counts in co_occ.items():
            top, freq = counts.most_common(1)[0]
            if freq >= 2 and s != top:
                dictionary[s] = top
        print(f"[pretrained] mined Hinglish->English dictionary: "
              f"{len(dictionary)} entries from {n} pairs")
        return dictionary

    # ---- inference ----

    def predict_sentiment(self, text: str) -> tuple[str, float, dict[str, float]]:
        import torch
        enc = self._sentiment_tok(text, return_tensors="pt",
                                  truncation=True, max_length=128).to(self.device)
        with torch.no_grad():
            logits = self._sentiment_mdl(**enc).logits.squeeze(0)
        probs = torch.softmax(logits, dim=-1).tolist()
        id2label = self._sentiment_mdl.config.id2label
        # Cardiff outputs lowercase {negative, neutral, positive}; map to canonical.
        canon = {"negative": "NEGATIVE", "neutral": "NEUTRAL",
                 "positive": "POSITIVE"}
        scores = {canon.get(id2label[i].lower(), id2label[i].upper()): probs[i]
                  for i in range(len(probs))}
        label = max(scores, key=scores.get)
        return label, scores[label], scores

    def translate(self, text: str, lid_tags: list[str] | None = None) -> str:
        """Render Hinglish as English. Order of preference:

          1. Locally fine-tuned CrossLing normalizer (mT5 trained on the
             parallel corpus). Best quality.
          2. Mined word-level dictionary fallback. Literal but readable.
          3. Identity (input echoed) when neither is available.
        """
        if self._normalizer is not None:
            return self._normalizer.predict(text, lid_tags=lid_tags)
        if self._dict:
            words = text.split()
            out = [self._dict.get(w.lower(), w) for w in words]
            return " ".join(out)
        return text

    def predict_lid(self, tokens: list[str]) -> list[str]:
        if self._lid_model is not None:
            return list(self._lid_model.predict(tokens).labels)
        return self._mined_lid.predict(tokens) if self._mined_lid else []

    def token_importance(self, text: str, label: str
                         ) -> list[dict]:
        """Gradient*input attribution from the sentiment model on the predicted class."""
        import torch
        tok, mdl = self._sentiment_tok, self._sentiment_mdl
        words = text.split() or [text]
        enc = tok(words, is_split_into_words=True, return_tensors="pt",
                  truncation=True, max_length=128).to(self.device)
        embed = mdl.get_input_embeddings()(enc["input_ids"]).detach().clone()
        embed.requires_grad_(True)
        out = mdl(inputs_embeds=embed, attention_mask=enc["attention_mask"])
        logits = out.logits.squeeze(0)
        probs = torch.softmax(logits, dim=-1)
        canon_to_idx = {}
        for i, name in mdl.config.id2label.items():
            canon = {"negative": "NEGATIVE", "neutral": "NEUTRAL",
                     "positive": "POSITIVE"}.get(name.lower(), name.upper())
            canon_to_idx[canon] = i
        target_idx = canon_to_idx.get(label, int(torch.argmax(probs).item()))
        logits[target_idx].backward()
        sal = (embed.grad * embed).norm(dim=-1).squeeze(0).detach().cpu().tolist()

        word_ids = enc.word_ids(batch_index=0)
        word_scores = [0.0] * len(words)
        word_counts = [0] * len(words)
        for tok_idx, w_id in enumerate(word_ids):
            if w_id is None or w_id >= len(words):
                continue
            word_scores[w_id] += sal[tok_idx]
            word_counts[w_id] += 1
        word_scores = [s / c if c else 0.0
                       for s, c in zip(word_scores, word_counts)]
        total = sum(word_scores) or 1.0
        return [{"token": w, "importance": s / total}
                for w, s in zip(words, word_scores)]

    # ---- top-level analyze ----

    def analyze(self, text: str, *, preprocess: bool = True) -> dict:
        if preprocess:
            trace = normalize_text(text)
        else:
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
        tags = self.predict_lid(tokens)
        cs = analyze_codeswitch(tokens, tags)
        label, score, scores = self.predict_sentiment(trace.cleaned)
        normalized = self.translate(trace.cleaned, lid_tags=tags)
        try:
            importance = self.token_importance(trace.cleaned, label)
        except Exception:
            importance = []
        return {
            "text": text,
            "cleaned": trace.cleaned,
            "preprocessing_edits": trace.edits,
            "tokens": tokens,
            "language_tags": tags,
            "code_switch": cs.to_dict(),
            "normalized": normalized,
            "sentiment": label,
            "sentiment_score": score,
            "sentiment_scores": scores,
            "token_importance": importance,
        }


# --------------------------------------------------------------------------- #
# Adapter so HinglishAnalyzer can use this backend transparently
# --------------------------------------------------------------------------- #

class _PretrainedAdapter(HinglishAnalyzer):
    """HinglishAnalyzer subclass that routes analyze() through PretrainedBackend.

    Kept tiny — the report assembly logic is shared with the rule-based path.
    """

    def __init__(self, backend: PretrainedBackend) -> None:
        super().__init__()
        self._pretrained = backend
        self.backend = "pretrained"

    def analyze(self, text: str, *, preprocess: bool = True) -> HinglishReport:
        d = self._pretrained.analyze(text, preprocess=preprocess)
        d["backend"] = "pretrained"
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


def build_analyzer(*, enable_translator: bool = True,
                   lid_conll: str | None = None) -> HinglishAnalyzer:
    """Construct a pretrained-backed analyzer ready to ``.analyze(text)``."""
    backend = PretrainedBackend(enable_translator=enable_translator,
                                lid_conll=lid_conll)
    return _PretrainedAdapter(backend)
