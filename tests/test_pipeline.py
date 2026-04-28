"""Lightweight tests that don't require network/model downloads.

These exercise the dataset loaders, label alignment, tokenizer-free helpers,
and metric implementations. The full transformer-backed wrappers are
exercised by `scripts/demo.py` after training.
"""

from __future__ import annotations

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data import (
    load_lid_examples,
    load_normalization_examples,
    load_sentiment_examples,
    split_examples,
)
from src.metrics import (
    flatten_token_predictions,
    lid_metrics,
    normalization_metrics,
    sentiment_metrics,
)
from src.pipeline import whitespace_tokenize


class TestData(unittest.TestCase):
    def test_lid_synthetic_examples_load(self):
        examples = load_lid_examples()
        self.assertGreater(len(examples), 0)
        for ex in examples:
            self.assertEqual(len(ex.tokens), len(ex.labels))
            for label in ex.labels:
                self.assertIn(label, {"ENG", "HIN", "OTHER", "NE"})

    def test_sentiment_synthetic_examples_load(self):
        examples = load_sentiment_examples()
        self.assertGreater(len(examples), 0)
        for text, label in examples:
            self.assertTrue(text)
            self.assertIn(label, {"POSITIVE", "NEGATIVE", "NEUTRAL"})

    def test_normalization_synthetic_examples_load(self):
        examples = load_normalization_examples()
        self.assertGreater(len(examples), 0)
        for src, tgt in examples:
            self.assertTrue(src)
            self.assertTrue(tgt)

    def test_split_examples_disjoint(self):
        items = list(range(100))
        train, val = split_examples(items, val_ratio=0.2, seed=0)
        self.assertEqual(len(train) + len(val), 100)
        self.assertEqual(set(train).isdisjoint(set(val)), True)


class TestTokenization(unittest.TestCase):
    def test_whitespace_tokenize_keeps_punct(self):
        toks = whitespace_tokenize("yaar, this is mast!")
        self.assertEqual(toks, ["yaar", ",", "this", "is", "mast", "!"])

    def test_whitespace_tokenize_empty(self):
        self.assertEqual(whitespace_tokenize(""), [])


class TestMetrics(unittest.TestCase):
    def test_flatten_skips_ignore_index(self):
        id2label = {0: "A", 1: "B"}
        preds = [[0, 1, 0]]
        gold = [[0, -100, 1]]
        flat_p, flat_g = flatten_token_predictions(preds, gold, id2label)
        self.assertEqual(flat_p, ["A", "A"])
        self.assertEqual(flat_g, ["A", "B"])

    def test_lid_metrics_perfect(self):
        id2label = {0: "ENG", 1: "HIN"}
        preds = [[0, 1, 1]]
        gold = [[0, 1, 1]]
        m = lid_metrics(preds, gold, id2label)
        self.assertAlmostEqual(m["accuracy"], 1.0)
        self.assertAlmostEqual(m["macro_f1"], 1.0)

    def test_sentiment_metrics(self):
        id2label = {0: "NEG", 1: "POS"}
        m = sentiment_metrics([0, 1, 1], [0, 1, 0], id2label)
        self.assertAlmostEqual(m["accuracy"], 2 / 3)

    def test_normalization_bleu_runs(self):
        m = normalization_metrics(
            ["the food is good", "i went to delhi"],
            ["the food is good", "i went to delhi"],
        )
        self.assertGreater(m["bleu"], 99.0)
        self.assertGreater(m["chrf"], 99.0)


if __name__ == "__main__":
    unittest.main()
