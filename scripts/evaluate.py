"""Run the CrossLing pipeline against a held-out test split and write a
comprehensive Markdown report.

Usage:
    python -m scripts.evaluate                     # evaluate on real
                                                   # SentiMix test split if
                                                   # downloaded, else synthetic
    python -m scripts.evaluate --sentiment-data data/sentimix/test.tsv \
                               --lid-data       data/sentimix/test.conll \
                               --normalizer-data data/english_to_hinglish/test.jsonl
    python -m scripts.evaluate --max 1000          # cap rows for speed
    python -m scripts.evaluate --out reports/eval.md

The report covers:
  * Per-component metrics (LID macro-F1 + per-class report, sentiment
    macro-F1 + per-class report, normalizer BLEU + chrF).
  * Code-switching distribution over the test set (CMI histogram, average
    switch fraction, dominance breakdown).
  * Robustness slice: metrics restricted to noisy inputs (any
    elongation, slang, or punctuation-run edit fired).
  * Worked examples — 5 illustrative reports with token importance.

Backends:
  * `neural`: loads checkpoints from config.yaml and uses the trained
    XLM-R + mT5 pipeline.
  * `rule-based`: lexicon LID / lexicon sentiment / identity normalizer.
    Used automatically when checkpoints are missing so the evaluation
    harness is exercisable end-to-end without training first.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from collections import Counter
from typing import Iterable

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.analyzer import HinglishAnalyzer
from src.codeswitch import analyze_codeswitch
from src.data import (
    load_lid_examples,
    load_normalization_examples,
    load_sentiment_examples,
)
from src.metrics import (
    lid_classification_report,
    lid_metrics,
    normalization_metrics,
    sentiment_classification_report,
    sentiment_metrics,
)
from src.real_data import NORM_PATHS, SENTIMIX_PATHS
from src.text_norm import normalize_text


# --------------------------------------------------------------------------- #
# Per-component evaluation
# --------------------------------------------------------------------------- #

def _resolve_test_paths(args):
    sent = args.sentiment_data or (SENTIMIX_PATHS["test_sentiment"]
                                   if os.path.exists(SENTIMIX_PATHS["test_sentiment"])
                                   else None)
    lid = args.lid_data or (SENTIMIX_PATHS["test_lid"]
                            if os.path.exists(SENTIMIX_PATHS["test_lid"])
                            else None)
    norm = args.normalizer_data or (NORM_PATHS["test"]
                                    if os.path.exists(NORM_PATHS["test"])
                                    else None)
    return sent, lid, norm


def _take(seq: Iterable, n: int | None):
    return list(seq) if n is None else list(seq)[:n]


def evaluate_sentiment(analyzer: HinglishAnalyzer, examples,
                       label_set=("POSITIVE", "NEGATIVE", "NEUTRAL")):
    """Run sentiment over each example, return metrics + per-row records."""
    label2id = {l: i for i, l in enumerate(label_set)}
    id2label = {i: l for l, i in label2id.items()}
    pred_ids: list[int] = []
    gold_ids: list[int] = []
    records: list[dict] = []

    for text, gold in examples:
        gold = gold.upper().strip()
        if gold not in label2id:
            continue
        report = analyzer.analyze(text)
        pred = report.sentiment
        if pred not in label2id:
            continue
        pred_ids.append(label2id[pred])
        gold_ids.append(label2id[gold])
        records.append({
            "text": text,
            "gold": gold,
            "pred": pred,
            "score": report.sentiment_score,
            "cleaned": report.cleaned,
            "edits": report.preprocessing_edits,
            "code_switch": report.code_switch,
            "top_tokens": sorted(report.token_importance,
                                 key=lambda d: -d["importance"])[:3],
        })

    if not pred_ids:
        return None, []
    metrics = sentiment_metrics(pred_ids, gold_ids, id2label)
    metrics["report"] = sentiment_classification_report(
        pred_ids, gold_ids, id2label)
    return metrics, records


def evaluate_lid(analyzer: HinglishAnalyzer, examples,
                 label_set=("ENG", "HIN", "OTHER", "NE")):
    """LID eval bypasses analyzer.analyze() — gold tokens go straight to the
    underlying LID layer (neural / mined / rule-based) so token boundaries are
    preserved 1-to-1 for fair comparison."""
    from src.analyzer import _rule_lid

    label2id = {l: i for i, l in enumerate(label_set)}
    id2label = {i: l for l, i in label2id.items()}
    pred_seqs: list[list[int]] = []
    gold_seqs: list[list[int]] = []

    # The pretrained backend exposes a predict_lid() shortcut.
    pretrained_be = getattr(analyzer, "_pretrained", None)

    for ex in examples:
        if analyzer.pipeline is not None:
            pred = analyzer.pipeline.lid.predict(ex.tokens).labels
        elif pretrained_be is not None:
            pred = pretrained_be.predict_lid(ex.tokens)
        else:
            pred = _rule_lid(ex.tokens)
        if len(pred) != len(ex.tokens):
            # Truncation by max_length; pad with OTHER so the row still counts.
            pred = list(pred) + ["OTHER"] * (len(ex.tokens) - len(pred))
            pred = pred[:len(ex.tokens)]
        pred_seq = [label2id.get(t, label2id["OTHER"]) for t in pred]
        gold_seq = [label2id.get(l, label2id["OTHER"]) for l in ex.labels]
        pred_seqs.append(pred_seq)
        gold_seqs.append(gold_seq)

    if not pred_seqs:
        return None, 0
    metrics = lid_metrics(pred_seqs, gold_seqs, id2label)
    metrics["report"] = lid_classification_report(
        pred_seqs, gold_seqs, id2label)
    metrics["alignment_drops"] = 0
    return metrics, 0


def evaluate_normalizer(analyzer: HinglishAnalyzer, examples):
    pretrained_be = getattr(analyzer, "_pretrained", None)
    has_translator = (analyzer.pipeline is not None
                      or (pretrained_be is not None
                          and (pretrained_be._normalizer is not None
                               or pretrained_be._dict)))
    if not has_translator:
        return None
    preds: list[str] = []
    refs: list[str] = []
    for src, tgt in examples:
        report = analyzer.analyze(src)
        preds.append(report.normalized)
        refs.append(tgt)
    return normalization_metrics(preds, refs)


# --------------------------------------------------------------------------- #
# Aggregate analytics
# --------------------------------------------------------------------------- #

def _cmi_bucket(cmi: float) -> str:
    if cmi == 0.0:
        return "monolingual (CMI=0)"
    if cmi < 15:
        return "low (0<CMI<15)"
    if cmi < 30:
        return "medium (15-30)"
    if cmi < 45:
        return "high (30-45)"
    return "very high (>=45)"


def codeswitch_summary(records: list[dict]) -> dict:
    """Aggregate code-switching stats across all sentiment records."""
    if not records:
        return {}
    cmis = [r["code_switch"]["cmi"] for r in records]
    sw_frac = [r["code_switch"]["switch_fraction"] for r in records]
    dom = Counter(r["code_switch"]["dominant_language"] for r in records)
    buckets = Counter(_cmi_bucket(c) for c in cmis)
    return {
        "n": len(records),
        "cmi_mean": statistics.mean(cmis),
        "cmi_median": statistics.median(cmis),
        "switch_fraction_mean": statistics.mean(sw_frac),
        "dominance": dict(dom),
        "buckets": dict(buckets),
        "code_mixed_share": sum(1 for r in records
                                if r["code_switch"]["is_code_mixed"]) / len(records),
    }


def robustness_slice_metrics(records: list[dict]) -> dict | None:
    """Restrict metrics to rows where the preprocessor fired."""
    noisy = [r for r in records if r["edits"]]
    clean = [r for r in records if not r["edits"]]
    if not noisy:
        return None

    def _f1(rs):
        labels = sorted({r["gold"] for r in rs} | {r["pred"] for r in rs})
        l2i = {l: i for i, l in enumerate(labels)}
        i2l = {i: l for l, i in l2i.items()}
        if not rs:
            return None
        m = sentiment_metrics(
            [l2i[r["pred"]] for r in rs],
            [l2i[r["gold"]] for r in rs],
            i2l,
        )
        return m

    return {
        "n_noisy": len(noisy),
        "n_clean": len(clean),
        "noisy": _f1(noisy),
        "clean": _f1(clean),
    }


# --------------------------------------------------------------------------- #
# Markdown report
# --------------------------------------------------------------------------- #

def _md_table(rows: list[list[str]], header: list[str]) -> str:
    out = ["| " + " | ".join(header) + " |"]
    out.append("| " + " | ".join("---" for _ in header) + " |")
    for row in rows:
        out.append("| " + " | ".join(str(c) for c in row) + " |")
    return "\n".join(out)


def render_report(*, backend: str, sources: dict, sent_metrics, lid_m,
                  norm_metrics, cs_summary, robustness, examples) -> str:
    lines: list[str] = []
    lines.append("# CrossLing evaluation report")
    lines.append("")
    lines.append(f"- **Backend**: `{backend}`")
    lines.append(f"- **Generated**: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"- **Sentiment data**: `{sources['sentiment'] or 'synthetic'}`")
    lines.append(f"- **LID data**: `{sources['lid'] or 'synthetic'}`")
    lines.append(f"- **Normalizer data**: `{sources['normalizer'] or 'synthetic'}`")
    lines.append("")

    # ---- Sentiment ----
    lines.append("## Sentiment classifier")
    if sent_metrics is None:
        lines.append("_no sentiment evaluation rows_")
    else:
        lines.append("")
        lines.append(_md_table(
            [[f"{sent_metrics['accuracy']:.4f}",
              f"{sent_metrics['macro_f1']:.4f}",
              f"{sent_metrics['weighted_f1']:.4f}"]],
            ["accuracy", "macro F1", "weighted F1"],
        ))
        lines.append("")
        lines.append("```")
        lines.append(sent_metrics["report"].rstrip())
        lines.append("```")
    lines.append("")

    # ---- LID ----
    lines.append("## Language identification")
    if lid_m is None:
        lines.append("_no LID evaluation rows_")
    else:
        lines.append("")
        lines.append(_md_table(
            [[f"{lid_m['accuracy']:.4f}",
              f"{lid_m['macro_f1']:.4f}",
              f"{lid_m['weighted_f1']:.4f}",
              str(lid_m.get('alignment_drops', 0))]],
            ["accuracy", "macro F1", "weighted F1", "alignment drops"],
        ))
        lines.append("")
        lines.append("```")
        lines.append(lid_m["report"].rstrip())
        lines.append("```")
    lines.append("")

    # ---- Normalizer ----
    lines.append("## Hinglish -> English normalizer")
    if norm_metrics is None:
        lines.append("_skipped (no neural normalizer in this backend)_")
    else:
        lines.append("")
        lines.append(_md_table(
            [[f"{norm_metrics['bleu']:.2f}", f"{norm_metrics['chrf']:.2f}"]],
            ["BLEU", "chrF"],
        ))
    lines.append("")

    # ---- Code-switch distribution ----
    lines.append("## Code-switching profile of the test set")
    if not cs_summary:
        lines.append("_n/a_")
    else:
        lines.append("")
        lines.append(_md_table(
            [[cs_summary["n"],
              f"{cs_summary['cmi_mean']:.2f}",
              f"{cs_summary['cmi_median']:.2f}",
              f"{cs_summary['switch_fraction_mean']:.3f}",
              f"{cs_summary['code_mixed_share']*100:.1f}%"]],
            ["sentences", "CMI mean", "CMI median",
             "avg switch frac", "code-mixed share"],
        ))
        lines.append("")
        lines.append("**CMI buckets**")
        lines.append("")
        bucket_rows = [[k, v] for k, v in sorted(cs_summary["buckets"].items())]
        lines.append(_md_table(bucket_rows, ["bucket", "count"]))
        lines.append("")
        lines.append("**Dominant language**")
        lines.append("")
        lines.append(_md_table(
            [[k or "n/a", v] for k, v in cs_summary["dominance"].items()],
            ["language", "count"]))
    lines.append("")

    # ---- Robustness ----
    lines.append("## Robustness slice (noisy vs clean inputs)")
    if not robustness:
        lines.append("_no noisy inputs in this test set_")
    else:
        lines.append("")
        rows = []
        for slice_name in ("noisy", "clean"):
            m = robustness.get(slice_name)
            n = robustness[f"n_{slice_name}"]
            if not m:
                rows.append([slice_name, n, "—", "—", "—"])
                continue
            rows.append([
                slice_name, n,
                f"{m['accuracy']:.4f}",
                f"{m['macro_f1']:.4f}",
                f"{m['weighted_f1']:.4f}",
            ])
        lines.append(_md_table(rows,
                               ["slice", "n", "accuracy",
                                "macro F1", "weighted F1"]))
    lines.append("")

    # ---- Worked examples ----
    lines.append("## Worked examples")
    lines.append("")
    for i, e in enumerate(examples[:5], 1):
        cs = e["code_switch"]
        edits = ", ".join(e["edits"]) if e["edits"] else "none"
        top = "; ".join(f"{d['token']} ({d['importance']:.2f})"
                        for d in e["top_tokens"])
        lines.append(f"**{i}.** `{e['text']}`  ")
        lines.append(f"- gold = `{e['gold']}`, pred = `{e['pred']}` "
                     f"(conf {e['score']:.2f})")
        lines.append(f"- cleaned: `{e['cleaned']}`  edits: {edits}")
        lines.append(f"- CMI = {cs['cmi']:.1f}, switches = "
                     f"{cs['switch_count']}, dominant = "
                     f"{cs['dominant_language']} "
                     f"({cs['dominance_ratio']*100:.0f}%)")
        lines.append(f"- top tokens: {top or '—'}")
        lines.append("")
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--sentiment-data", default=None)
    parser.add_argument("--lid-data", default=None)
    parser.add_argument("--normalizer-data", default=None)
    parser.add_argument("--max", type=int, default=2000,
                        help="Cap rows per component (default 2000; "
                             "0 = unlimited).")
    parser.add_argument("--out", default="reports/eval.md")
    parser.add_argument("--no-normalizer", action="store_true",
                        help="Skip BLEU/chrF eval even when the neural "
                             "normalizer is available (fast path).")
    parser.add_argument("--backend", default="auto",
                        choices=["auto", "neural", "pretrained", "rule-based"])
    args = parser.parse_args()

    cap = None if args.max == 0 else args.max

    sent_path, lid_path, norm_path = _resolve_test_paths(args)
    print(f"[evaluate] sentiment={sent_path or 'synthetic'} "
          f"lid={lid_path or 'synthetic'} "
          f"normalizer={norm_path or 'synthetic'}")

    if args.backend == "pretrained":
        from src.pretrained import build_analyzer
        analyzer = build_analyzer()
    elif args.backend == "rule-based":
        analyzer = HinglishAnalyzer()
    elif args.backend == "neural":
        analyzer = HinglishAnalyzer.from_config(args.config)
    else:
        analyzer = HinglishAnalyzer.from_config(args.config)
        if analyzer.backend == "rule-based":
            try:
                from src.pretrained import build_analyzer
                analyzer = build_analyzer()
            except Exception as exc:
                print(f"[evaluate] pretrained backend failed: {exc}")
    print(f"[evaluate] backend = {analyzer.backend}")

    sentiment_examples = _take(load_sentiment_examples(sent_path), cap)
    lid_examples = _take(load_lid_examples(lid_path), cap)
    norm_examples = _take(load_normalization_examples(norm_path), cap)

    print(f"[evaluate] sentiment rows = {len(sentiment_examples)}, "
          f"lid sentences = {len(lid_examples)}, "
          f"norm pairs = {len(norm_examples)}")

    sent_metrics, sent_records = evaluate_sentiment(analyzer, sentiment_examples)
    lid_m, _drops = evaluate_lid(analyzer, lid_examples)
    norm_m = (None if args.no_normalizer
              else evaluate_normalizer(analyzer, norm_examples))

    cs = codeswitch_summary(sent_records)
    rob = robustness_slice_metrics(sent_records)

    md = render_report(
        backend=analyzer.backend,
        sources={"sentiment": sent_path, "lid": lid_path,
                 "normalizer": norm_path},
        sent_metrics=sent_metrics,
        lid_m=lid_m,
        norm_metrics=norm_m,
        cs_summary=cs,
        robustness=rob,
        examples=sent_records,
    )

    out_path = args.out
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as fh:
        fh.write(md)

    # Also print a short console summary so the user gets immediate feedback.
    print()
    print(md.split("## Worked examples")[0])
    print(f"[evaluate] full report written to {out_path}")


if __name__ == "__main__":
    main()
