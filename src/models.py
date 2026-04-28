"""Model wrappers for the three CrossLing components.

Each wrapper bundles together:
  * a HuggingFace tokenizer
  * a HuggingFace model
  * a label vocabulary (where applicable)
  * `save` / `load` / `predict` methods with a uniform shape

Backbones (configurable):
  * LID:        XLM-RoBERTa + token-classification head
  * Sentiment:  XLM-RoBERTa + sequence-classification head
  * Normalizer: mT5 (encoder-decoder) for Hinglish -> English

Why XLM-R for LID/sentiment: it is pretrained on 100+ languages including
Romanized Hindi, has subword vocabulary that gracefully degrades on OOV
Hinglish spellings (e.g. "bakwaas"), and has a single tokenizer for both
script families. Why mT5 for normalization: T5's text-to-text framing fits
"normalize hinglish: ..." prompting cleanly and the multilingual variant
already knows English vocabulary at the target side.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass

import torch
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoModelForSequenceClassification,
    AutoModelForTokenClassification,
    AutoTokenizer,
)


def resolve_device(preference: str = "auto") -> torch.device:
    if preference == "cpu":
        return torch.device("cpu")
    if preference == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    if preference == "mps" and torch.backends.mps.is_available():
        return torch.device("mps")
    if preference == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
    return torch.device("cpu")


# --------------------------------------------------------------------------- #
# LID
# --------------------------------------------------------------------------- #

@dataclass
class LIDPrediction:
    tokens: list[str]
    labels: list[str]


class LIDModel:
    def __init__(self, model, tokenizer, label2id: dict[str, int],
                 device: torch.device) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.label2id = label2id
        self.id2label = {v: k for k, v in label2id.items()}
        self.device = device
        self.model.to(device)

    @classmethod
    def from_pretrained(cls, name_or_path: str, labels: list[str],
                        device: torch.device) -> "LIDModel":
        label2id = {l: i for i, l in enumerate(labels)}
        id2label = {i: l for l, i in label2id.items()}
        tokenizer = AutoTokenizer.from_pretrained(name_or_path,
                                                  add_prefix_space=True)
        model = AutoModelForTokenClassification.from_pretrained(
            name_or_path,
            num_labels=len(labels),
            id2label=id2label,
            label2id=label2id,
        )
        return cls(model, tokenizer, label2id, device)

    @classmethod
    def load(cls, path: str, device: torch.device) -> "LIDModel":
        with open(os.path.join(path, "labels.json"), encoding="utf-8") as fh:
            label2id = json.load(fh)
        tokenizer = AutoTokenizer.from_pretrained(path, add_prefix_space=True)
        model = AutoModelForTokenClassification.from_pretrained(path)
        return cls(model, tokenizer, label2id, device)

    def save(self, path: str) -> None:
        os.makedirs(path, exist_ok=True)
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)
        with open(os.path.join(path, "labels.json"), "w", encoding="utf-8") as fh:
            json.dump(self.label2id, fh, ensure_ascii=False, indent=2)

    @torch.no_grad()
    def predict(self, words: list[str], max_length: int = 128) -> LIDPrediction:
        """Predict a label per input word. Subword pieces vote via their first piece."""
        self.model.eval()
        enc = self.tokenizer(
            words,
            is_split_into_words=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        ).to(self.device)
        logits = self.model(**enc).logits.squeeze(0)
        pred_ids = logits.argmax(dim=-1).tolist()
        word_ids = enc.word_ids(batch_index=0)

        labels: list[str] = []
        last_word: int | None = None
        for token_idx, word_id in enumerate(word_ids):
            if word_id is None or word_id == last_word:
                continue
            labels.append(self.id2label[pred_ids[token_idx]])
            last_word = word_id

        # If truncation cut off some words, pad with OTHER
        while len(labels) < len(words):
            labels.append("OTHER")
        return LIDPrediction(tokens=list(words), labels=labels[: len(words)])


# --------------------------------------------------------------------------- #
# Sentiment
# --------------------------------------------------------------------------- #

@dataclass
class SentimentPrediction:
    label: str
    score: float
    scores: dict[str, float]


class SentimentModel:
    def __init__(self, model, tokenizer, label2id: dict[str, int],
                 device: torch.device) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.label2id = label2id
        self.id2label = {v: k for k, v in label2id.items()}
        self.device = device
        self.model.to(device)

    @classmethod
    def from_pretrained(cls, name_or_path: str, labels: list[str],
                        device: torch.device) -> "SentimentModel":
        label2id = {l: i for i, l in enumerate(labels)}
        id2label = {i: l for l, i in label2id.items()}
        tokenizer = AutoTokenizer.from_pretrained(name_or_path)
        model = AutoModelForSequenceClassification.from_pretrained(
            name_or_path,
            num_labels=len(labels),
            id2label=id2label,
            label2id=label2id,
        )
        return cls(model, tokenizer, label2id, device)

    @classmethod
    def load(cls, path: str, device: torch.device) -> "SentimentModel":
        with open(os.path.join(path, "labels.json"), encoding="utf-8") as fh:
            label2id = json.load(fh)
        tokenizer = AutoTokenizer.from_pretrained(path)
        model = AutoModelForSequenceClassification.from_pretrained(path)
        return cls(model, tokenizer, label2id, device)

    def save(self, path: str) -> None:
        os.makedirs(path, exist_ok=True)
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)
        with open(os.path.join(path, "labels.json"), "w", encoding="utf-8") as fh:
            json.dump(self.label2id, fh, ensure_ascii=False, indent=2)

    @torch.no_grad()
    def predict(self, text: str, max_length: int = 128) -> SentimentPrediction:
        self.model.eval()
        enc = self.tokenizer(
            text,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        ).to(self.device)
        logits = self.model(**enc).logits.squeeze(0)
        probs = torch.softmax(logits, dim=-1).tolist()
        scores = {self.id2label[i]: probs[i] for i in range(len(probs))}
        best_id = int(max(range(len(probs)), key=lambda i: probs[i]))
        return SentimentPrediction(
            label=self.id2label[best_id],
            score=probs[best_id],
            scores=scores,
        )

    def token_importance(self, text: str, max_length: int = 128
                         ) -> tuple[list[str], list[float], str, float]:
        """Per-(sub)token importance via gradient * input on the embeddings.

        Returns: ``(token_strings, importance_scores, predicted_label,
        confidence)``. Importance is the L2 norm of the gradient times
        embedding for each subword, normalized to sum to 1, and aggregated
        from subwords back to whitespace words by mean-pooling.

        Why gradient*input rather than attention rollout: attention-based
        explanations on classification CLS heads are noisy and depend on
        the head you pick; grad*input directly answers "which input pieces
        moved the predicted-class logit." Cheap (one forward + one backward).
        """
        self.model.eval()
        enc = self.tokenizer(
            text,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
            return_offsets_mapping=False,
        )
        input_ids = enc["input_ids"].to(self.device)
        attn = enc["attention_mask"].to(self.device)

        embed_layer = self.model.get_input_embeddings()
        embeds = embed_layer(input_ids)
        embeds = embeds.detach().clone().requires_grad_(True)

        outputs = self.model(inputs_embeds=embeds, attention_mask=attn)
        logits = outputs.logits.squeeze(0)
        probs = torch.softmax(logits, dim=-1)
        best_id = int(torch.argmax(probs).item())
        confidence = float(probs[best_id].item())

        target = logits[best_id]
        target.backward()
        grad = embeds.grad
        if grad is None:
            saliency = torch.zeros(input_ids.shape[1])
        else:
            saliency = (grad * embeds).norm(dim=-1).squeeze(0).detach().cpu()

        # Aggregate subword saliency back to whitespace words.
        words = text.split()
        encoded_words = self.tokenizer(
            words,
            is_split_into_words=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        word_ids = encoded_words.word_ids(batch_index=0)
        # Re-run saliency over the word-aligned encoding so subword groupings line up.
        embeds_w = embed_layer(encoded_words["input_ids"].to(self.device)
                               ).detach().clone().requires_grad_(True)
        out_w = self.model(inputs_embeds=embeds_w,
                           attention_mask=encoded_words["attention_mask"].to(self.device))
        logits_w = out_w.logits.squeeze(0)
        logits_w[best_id].backward()
        grad_w = embeds_w.grad
        sal_w = (grad_w * embeds_w).norm(dim=-1).squeeze(0).detach().cpu().tolist()

        word_scores = [0.0] * len(words)
        word_counts = [0] * len(words)
        for tok_idx, w_id in enumerate(word_ids):
            if w_id is None or w_id >= len(words):
                continue
            word_scores[w_id] += sal_w[tok_idx]
            word_counts[w_id] += 1
        word_scores = [s / c if c else 0.0
                       for s, c in zip(word_scores, word_counts)]
        total = sum(word_scores)
        if total > 0:
            word_scores = [s / total for s in word_scores]

        return words, word_scores, self.id2label[best_id], confidence


# --------------------------------------------------------------------------- #
# Normalizer (seq2seq)
# --------------------------------------------------------------------------- #

class NormalizerModel:
    def __init__(self, model, tokenizer, device: torch.device,
                 source_prefix: str = "") -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.source_prefix = source_prefix
        self.model.to(device)

    @classmethod
    def from_pretrained(cls, name_or_path: str, device: torch.device,
                        source_prefix: str = "") -> "NormalizerModel":
        tokenizer = AutoTokenizer.from_pretrained(name_or_path)
        model = AutoModelForSeq2SeqLM.from_pretrained(name_or_path)
        return cls(model, tokenizer, device, source_prefix)

    @classmethod
    def load(cls, path: str, device: torch.device) -> "NormalizerModel":
        meta_path = os.path.join(path, "normalizer_meta.json")
        prefix = ""
        if os.path.exists(meta_path):
            with open(meta_path, encoding="utf-8") as fh:
                prefix = json.load(fh).get("source_prefix", "")
        tokenizer = AutoTokenizer.from_pretrained(path)
        model = AutoModelForSeq2SeqLM.from_pretrained(path)
        return cls(model, tokenizer, device, prefix)

    def save(self, path: str) -> None:
        os.makedirs(path, exist_ok=True)
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)
        with open(os.path.join(path, "normalizer_meta.json"), "w",
                  encoding="utf-8") as fh:
            json.dump({"source_prefix": self.source_prefix}, fh)

    @torch.no_grad()
    def predict(self, text: str, max_source_length: int = 128,
                max_target_length: int = 128, num_beams: int = 4,
                lid_tags: list[str] | None = None) -> str:
        """Translate Hinglish -> English.

        If `lid_tags` is supplied, prepend an inline tag sequence
        ("[ENG/HIN/...] : <text>") so the encoder sees per-token language
        information. mT5 tolerates the extra prefix; on tiny synthetic data
        the model learns to ignore the brackets while gaining a language
        prior at inference time.
        """
        self.model.eval()
        prompt = self.source_prefix + self._build_tagged_input(text, lid_tags)
        enc = self.tokenizer(
            prompt,
            truncation=True,
            max_length=max_source_length,
            return_tensors="pt",
        ).to(self.device)
        out = self.model.generate(
            **enc,
            max_length=max_target_length,
            num_beams=num_beams,
            early_stopping=True,
        )
        return self.tokenizer.decode(out[0], skip_special_tokens=True)

    @staticmethod
    def _build_tagged_input(text: str, lid_tags: list[str] | None) -> str:
        if not lid_tags:
            return text
        tag_seq = "/".join(lid_tags)
        return f"[{tag_seq}] {text}"
