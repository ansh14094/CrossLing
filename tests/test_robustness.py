"""Robustness tests for noisy real-world Hinglish.

These tests run against the rule-based analyzer fallback so they execute
without downloading models. They lock in two properties:

  1. The text-normalization preprocessor handles the noise patterns
     real Hinglish actually has (elongations, slang, casing, runs of
     punctuation, URLs/mentions).
  2. The analyzer produces a stable, well-formed report on noisy input
     and tracks sentiment polarity correctly through the cleanup.
"""

from __future__ import annotations

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.analyzer import HinglishAnalyzer
from src.codeswitch import analyze_codeswitch
from src.text_norm import (
    collapse_elongations,
    collapse_punct_runs,
    expand_slang,
    normalize_text,
    strip_urls_mentions,
)


class TestTextNormalization(unittest.TestCase):
    def test_elongation_collapse(self):
        cleaned, n = collapse_elongations("bakwaaaas")
        self.assertEqual(cleaned, "bakwaas")
        self.assertEqual(n, 1)

    def test_elongation_preserves_legitimate_doubles(self):
        # 'good', 'see', 'really' must survive the collapse
        cleaned, _ = collapse_elongations("good see really")
        self.assertEqual(cleaned, "good see really")

    def test_punct_run_collapse(self):
        cleaned, n = collapse_punct_runs("amazing!!!! wow??")
        self.assertEqual(cleaned, "amazing! wow?")
        self.assertEqual(n, 2)

    def test_slang_expansion(self):
        cleaned, fired = expand_slang("u r the best, plz come")
        self.assertIn("you", cleaned)
        self.assertIn("are", cleaned)
        self.assertIn("please", cleaned)
        self.assertTrue(any(f.startswith("u->") for f in fired))

    def test_urls_and_mentions_removed(self):
        cleaned, fired = strip_urls_mentions(
            "check https://x.com/foo @ansh #hinglish rocks")
        self.assertNotIn("http", cleaned)
        self.assertNotIn("@ansh", cleaned)
        self.assertIn("hinglish", cleaned)
        self.assertIn("url-removed", fired)
        self.assertIn("mention-removed", fired)
        self.assertIn("hashtag-flattened", fired)

    def test_full_pipeline_trace(self):
        trace = normalize_text("yeh movi bakwaaaas h brooo!!!")
        self.assertNotIn("aaaa", trace.cleaned)
        self.assertNotIn("!!!", trace.cleaned)
        self.assertTrue(any("elongations" in e for e in trace.edits))
        self.assertTrue(any("punct-runs" in e for e in trace.edits))


class TestCodeSwitch(unittest.TestCase):
    def test_monolingual_cmi_zero(self):
        stats = analyze_codeswitch(
            ["the", "movie", "was", "good"],
            ["ENG", "ENG", "ENG", "ENG"],
        )
        self.assertEqual(stats.cmi, 0.0)
        self.assertFalse(stats.is_code_mixed)
        self.assertEqual(stats.dominant_language, "ENG")

    def test_evenly_mixed_high_cmi(self):
        stats = analyze_codeswitch(
            ["the", "yaar", "movie", "bakwaas"],
            ["ENG", "HIN", "ENG", "HIN"],
        )
        self.assertGreaterEqual(stats.cmi, 49.0)
        self.assertTrue(stats.is_code_mixed)
        self.assertEqual(len(stats.switch_points), 3)

    def test_named_entities_excluded_from_cmi(self):
        stats = analyze_codeswitch(
            ["Sachin", "ne", "amazing", "knock", "kheli"],
            ["NE", "HIN", "ENG", "ENG", "HIN"],
        )
        self.assertEqual(stats.n_content_tokens, 4)
        self.assertGreater(stats.cmi, 0)

    def test_dominance_ratio(self):
        stats = analyze_codeswitch(
            ["the", "movie", "was", "very", "yaar"],
            ["ENG", "ENG", "ENG", "ENG", "HIN"],
        )
        self.assertEqual(stats.dominant_language, "ENG")
        self.assertAlmostEqual(stats.dominance_ratio, 0.8, places=2)

    def test_token_label_length_mismatch_raises(self):
        with self.assertRaises(ValueError):
            analyze_codeswitch(["a", "b"], ["ENG"])


class TestRuleBasedAnalyzerOnNoisyInput(unittest.TestCase):
    def setUp(self):
        # Force rule-based mode (no checkpoints loaded) — fast and offline.
        self.analyzer = HinglishAnalyzer()

    def test_noisy_negative_classified_negative(self):
        report = self.analyzer.analyze("yeh movi bakwaaaas h brooo!!!")
        self.assertEqual(report.sentiment, "NEGATIVE")
        self.assertNotIn("aaaa", report.cleaned)
        self.assertGreater(len(report.preprocessing_edits), 0)
        self.assertEqual(len(report.tokens), len(report.language_tags))

    def test_noisy_positive_classified_positive(self):
        report = self.analyzer.analyze("OMGGG ekdum maaaast yaaaar loved itttt!!!")
        self.assertEqual(report.sentiment, "POSITIVE")
        # bakwaas-style elongation cleanup should fire
        self.assertTrue(any("elongation" in e
                            for e in report.preprocessing_edits))

    def test_report_has_required_fields(self):
        report = self.analyzer.analyze("kya baat hai yaar")
        d = report.to_dict()
        for key in ("text", "cleaned", "tokens", "language_tags",
                    "code_switch", "sentiment", "sentiment_scores",
                    "token_importance", "summary", "backend"):
            self.assertIn(key, d, f"missing key: {key}")
        self.assertEqual(d["backend"], "rule-based")
        self.assertTrue(d["summary"])

    def test_code_mixed_input_flagged(self):
        report = self.analyzer.analyze(
            "the food was ekdum mast but service slow thi")
        self.assertTrue(report.code_switch["is_code_mixed"])
        self.assertGreater(report.code_switch["cmi"], 0)
        self.assertGreater(report.code_switch["switch_count"], 0)

    def test_empty_input_safe(self):
        report = self.analyzer.analyze("")
        self.assertEqual(report.tokens, [])
        self.assertEqual(report.sentiment, "NEUTRAL")


if __name__ == "__main__":
    unittest.main()
