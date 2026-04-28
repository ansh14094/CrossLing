"""Evaluation metrics for the three CrossLing tasks.

Choices:
  * LID: token-level macro-F1 + accuracy via seqeval-style flattening
    (we treat each label as its own class — no BIO scheme since LID labels
    are not span-based).
  * Sentiment: macro-F1 + accuracy + per-class report (sklearn).
  * Normalization: corpus-level sacreBLEU + per-sentence chrF as a
    character-level sanity signal for transliteration quality.
"""

from __future__ import annotations

from typing import Iterable

import numpy as np
import sacrebleu
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
)


# --------------------------------------------------------------------------- #
# LID
# --------------------------------------------------------------------------- #

def flatten_token_predictions(
    pred_ids: Iterable[Iterable[int]],
    gold_ids: Iterable[Iterable[int]],
    id2label: dict[int, str],
) -> tuple[list[str], list[str]]:
    """Flatten batched token predictions/labels, dropping ignored (-100) positions."""
    flat_pred: list[str] = []
    flat_gold: list[str] = []
    for p_seq, g_seq in zip(pred_ids, gold_ids):
        for p, g in zip(p_seq, g_seq):
            if g == -100:
                continue
            flat_pred.append(id2label[int(p)])
            flat_gold.append(id2label[int(g)])
    return flat_pred, flat_gold


def lid_metrics(pred_ids, gold_ids, id2label) -> dict[str, float]:
    flat_pred, flat_gold = flatten_token_predictions(pred_ids, gold_ids, id2label)
    if not flat_gold:
        return {"accuracy": 0.0, "macro_f1": 0.0}
    return {
        "accuracy": accuracy_score(flat_gold, flat_pred),
        "macro_f1": f1_score(flat_gold, flat_pred, average="macro", zero_division=0),
        "weighted_f1": f1_score(flat_gold, flat_pred, average="weighted",
                                zero_division=0),
    }


def lid_classification_report(pred_ids, gold_ids, id2label) -> str:
    flat_pred, flat_gold = flatten_token_predictions(pred_ids, gold_ids, id2label)
    return classification_report(flat_gold, flat_pred, zero_division=0)


# --------------------------------------------------------------------------- #
# Sentiment
# --------------------------------------------------------------------------- #

def sentiment_metrics(pred_ids, gold_ids, id2label) -> dict[str, float]:
    pred_labels = [id2label[int(i)] for i in pred_ids]
    gold_labels = [id2label[int(i)] for i in gold_ids]
    return {
        "accuracy": accuracy_score(gold_labels, pred_labels),
        "macro_f1": f1_score(gold_labels, pred_labels, average="macro",
                             zero_division=0),
        "weighted_f1": f1_score(gold_labels, pred_labels, average="weighted",
                                zero_division=0),
    }


def sentiment_classification_report(pred_ids, gold_ids, id2label) -> str:
    pred_labels = [id2label[int(i)] for i in pred_ids]
    gold_labels = [id2label[int(i)] for i in gold_ids]
    return classification_report(gold_labels, pred_labels, zero_division=0)


# --------------------------------------------------------------------------- #
# Normalization (BLEU + chrF)
# --------------------------------------------------------------------------- #

def normalization_metrics(predictions: list[str],
                          references: list[str]) -> dict[str, float]:
    if not predictions:
        return {"bleu": 0.0, "chrf": 0.0}
    bleu = sacrebleu.corpus_bleu(predictions, [references])
    chrf = sacrebleu.corpus_chrf(predictions, [references])
    return {"bleu": float(bleu.score), "chrf": float(chrf.score)}


# --------------------------------------------------------------------------- #
# Generic batched-eval helper used by the training scripts
# --------------------------------------------------------------------------- #

def stack_batch_predictions(per_batch: list[np.ndarray]) -> np.ndarray:
    """Concatenate per-batch arrays even when batches have different padding."""
    if not per_batch:
        return np.zeros((0,), dtype=np.int64)
    if per_batch[0].ndim == 1:
        return np.concatenate(per_batch, axis=0)
    max_len = max(arr.shape[1] for arr in per_batch)
    padded = []
    for arr in per_batch:
        if arr.shape[1] == max_len:
            padded.append(arr)
        else:
            pad_width = ((0, 0), (0, max_len - arr.shape[1]))
            padded.append(np.pad(arr, pad_width, constant_values=-100))
    return np.concatenate(padded, axis=0)
