# CrossLing — Hinglish NLP Pipeline

End-to-end NLP pipeline for code-mixed Hindi-English (Hinglish) text. The five
stages interact (each one feeds the next), not run in isolation:

1. **Robust text preprocessing** — collapse elongations (`bakwaaaas` → `bakwaas`), expand chatspeak (`u r the best plz` → `you are the best please`), normalise punctuation runs, strip URLs/mentions. Reports every edit it made so the cleanup is auditable.
2. **Token-level Language Identification (LID)** — XLM-RoBERTa + token-classification head, labels: `ENG / HIN / OTHER / NE`.
3. **Code-switch analysis** — Code-Mixing Index (Das & Gambäck 2014), switch points, language dominance, burstiness. Turns LID tags into the quantitative measures that matter for code-mixing research.
4. **Hinglish → English Normalizer** — mT5 fine-tuned as a text-to-text seq2seq model. The normalizer **consumes LID tags as features** (`[HIN/ENG/HIN/...] <text>`) instead of treating them as throw-away output.
5. **Sentiment Analysis** — XLM-RoBERTa sequence classifier, labels: `NEGATIVE / NEUTRAL / POSITIVE`. Fine-tuned **directly on code-mixed input** (no translation pipeline) so the affective signal carried by code-switching is preserved. A gradient×input attribution pass produces **per-token importance scores** so you can see which words drove the prediction.

A **synthetic Hinglish corpus with noisy real-world variants** (elongated vowels, slang, casing chaos) ships with the repo so the pipeline trains end-to-end without dataset registration. Real-world datasets (LINCE LID, SemEval-2020 SentiMix, PHINC) plug in through three CLI flags; the loaders auto-detect LINCE label conventions (`lang1/lang2/ne/...`) and SentiMix's CoNLL+meta layout.

```
raw text
  └► preprocess (cleanup trace)
       └► LID  ─────────────► code-switch stats
            └► Normalizer (uses LID tags as prefix features)
                 └► Sentiment + token importance
                      └► structured report (JSON or pretty)
```

## Architecture

| Component   | Backbone           | Head                  | Loss                                   |
|-------------|--------------------|------------------------|----------------------------------------|
| LID         | `xlm-roberta-base` | Linear, per subword    | CE; subword pieces beyond the first masked with `-100` |
| Sentiment   | `xlm-roberta-base` | Linear over `[CLS]`    | CE (3-way)                             |
| Normalizer  | `google/mt5-small` | Encoder–decoder        | Teacher-forced CE; beam search at inference |

**Why these choices.** XLM-R's multilingual subword vocabulary degrades gracefully on Romanized Hindi spellings (`bakwaas`, `ekdum`); a single tokenizer covers Devanagari and Latin scripts. mT5's text-to-text framing makes prompting (`"normalize hinglish: ..."`) clean and lets the model reorder tokens (`"Mumbai ki rains"` → `"Mumbai's rains"`) instead of word-by-word substitution.

**Why three models, not one multi-task XLM-R.** Joint heads share parameters but couple training schedules, learning-rate sensitivity, and bug surfaces. Three separate models swap independently — production-friendly.

## Project layout

```
CrossLing/
├── config.yaml                  # all hyperparameters
├── requirements.txt
├── src/
│   ├── data.py                  # datasets + noisy synthetic corpus + LINCE/SentiMix loaders
│   ├── real_data.py             # HF Hub downloader (SentiMix + english-to-hinglish)
│   ├── text_norm.py             # elongation / slang / punctuation cleanup with audit trace
│   ├── codeswitch.py            # CMI, switch points, dominance, burstiness
│   ├── models.py                # LID / Sentiment / Normalizer wrappers (+ token importance)
│   ├── metrics.py               # F1, BLEU, chrF
│   ├── pipeline.py              # end-to-end CrossLingPipeline (LID-aware normalizer)
│   ├── analyzer.py              # HinglishAnalyzer.analyze(text) -> rich report
│   ├── pretrained.py            # zero-shot pretrained backend (cardiffnlp/xlm-r + mined LID)
│   ├── train_lid.py             # python -m src.train_lid
│   ├── train_sentiment.py       # python -m src.train_sentiment
│   └── train_normalizer.py      # python -m src.train_normalizer
├── scripts/
│   ├── train_full.py            # GPU full real-data training (canonical)
│   ├── train_all.py             # train all three components (synthetic / --use-real)
│   ├── evaluate.py              # full evaluation report -> reports/eval.md
│   ├── analyze.py               # single-sentence analyzer CLI (rich report)
│   └── demo.py                  # batched / interactive demo
├── data/                        # populated by `python -m src.real_data`
│   ├── sentimix/{train,validation,test}.{tsv,conll}
│   └── english_to_hinglish/{train,validation,test}.jsonl
└── tests/
    ├── test_pipeline.py         # data + metric smoke tests
    └── test_robustness.py       # noisy-input + code-switch tests
```

## Quick start

There are three usage tiers — pick the one that matches the hardware you have.

### Tier 1 — laptop, no training (uses pretrained models from HuggingFace)

```bash
pip install -r requirements.txt
python -m src.real_data                    # download real datasets (~30 MB)
python -m scripts.analyze --backend pretrained "yeh movi bakwaaaas h brooo!!!"
python -m scripts.evaluate --backend pretrained --max 500
```

This uses `cardiffnlp/twitter-xlm-roberta-base-sentiment` (zero-shot, multilingual,
handles real Hinglish well out of the box) for sentiment, a vocabulary mined
from the SentiMix CoNLL training data for LID, and a Hinglish→English
dictionary mined from the parallel corpus for translation. **No training
required**, runs on CPU/MPS/CUDA.

### Tier 2 — GPU, full real-data training (best quality)

```bash
pip install -r requirements.txt
python -m src.real_data --norm-cap 0       # full 189k parallel pairs
python -m scripts.train_full --device cuda # ~10-30 min on A100/3090/T4
python -m scripts.evaluate --backend neural
python -m scripts.analyze "yeh movi bakwaaaas h brooo!!!"
```

This fine-tunes XLM-RoBERTa for sentiment+LID and mT5-small for translation
on the real datasets. Produces the highest-quality models — the analyzer
auto-detects the trained checkpoints in `checkpoints/` and uses them.

### Tier 3 — laptop, synthetic training (fastest sanity check)

```bash
python -m scripts.train_all                # synthetic corpus, ~10 min on MPS
python -m scripts.analyze "yeh movi bakwaaaas h brooo!!!"
```

The bundled synthetic corpus is small — fine for verifying the training
loop end-to-end, but won't generalise to real-world Hinglish. Use Tier 1
or Tier 2 for actual results.

### Real datasets used

| Component   | Source                                | Size                | License | Format on disk |
|-------------|---------------------------------------|---------------------|---------|----------------|
| Sentiment   | `RTT1/SentiMix` (SemEval-2020 Task 9) | 14k/3k/3k splits    | research | `data/sentimix/{train,validation,test}.tsv` |
| LID         | same                                  | shares the splits   | research | `data/sentimix/{train,validation,test}.conll` |
| Normalizer  | `findnitai/english-to-hinglish`       | 27k/1.5k/1.5k splits (cap 30k) | open  | `data/english_to_hinglish/*.jsonl` |

`src/real_data.py` handles both the train/val format (label inlined in the
`meta` line) and the test format (label shipped in a `Uid,Sentiment` CSV
prefix), so all three SentiMix splits become usable. The label maps
(`lang1/lang2/Eng/Hin/O/...`) are normalised to the CrossLing canonical
`ENG/HIN/OTHER/NE` set inside `src/data.py`.

### Per-component training

```bash
python -m src.train_lid        --data data/sentimix/train.conll
python -m src.train_sentiment  --data data/sentimix/train.tsv
python -m src.train_normalizer --data data/english_to_hinglish/train.jsonl
```

Or pass `--use-real` to `scripts/train_all` and let it pick the canonical
paths automatically.

### Full real-data training on a CUDA GPU

`scripts/train_full.py` is the canonical entry point for training all three
components on real data. Tuned for a single mid-range GPU (T4 / 3090 / A100,
~16-24 GB VRAM). On a T4 it finishes in ~25 min; on an A100 in ~10 min.

```bash
# 1. Install deps (one time)
pip install -r requirements.txt

# 2. Download the full real datasets (~21k SentiMix sentences,
#    full 189k Hinglish<->English parallel pairs)
python -m src.real_data --norm-cap 0          # 0 = no cap

# 3. Train all three components on the real data, on the GPU
python -m scripts.train_full --device cuda

# 4. Evaluate on the held-out test split
python -m scripts.evaluate --backend neural --out reports/eval_full.md

# 5. Single-sentence analyser
python -m scripts.analyze "yeh movi bakwaaaas h brooo!!!"
```

Useful tuning flags for `train_full`:

| Flag                     | Effect |
|--------------------------|--------|
| `--device cuda\|mps\|cpu` | Override device auto-detection. |
| `--light`                | Laptop / 8 GB-VRAM defaults: `lid_batch=16`, `sent_batch=16`, `norm_batch=4`, `norm_len=64`. Useful on MPS / smaller cards. |
| `--epochs N`             | Override epochs for all three. |
| `--lid-batch / --sent-batch / --norm-batch N` | Per-component batch overrides. |
| `--norm-len N`           | Max source/target length for the normalizer (default 96). Halve to fit on tighter VRAM. |
| `--skip lid sentiment`   | Only train the normalizer (etc.). |

Mixed precision is auto-selected: bf16 on Ampere+ (RTX 30/40, A100, H100),
fp16 on older CUDA (T4, V100), fp32 otherwise. The normalizer is forced to
bf16-or-fp32 because mT5 is unstable in fp16.

After training, the analyzer and `evaluate` scripts auto-detect the
checkpoints in `checkpoints/{lid,sentiment,normalizer}` and switch to the
neural backend.

### Three-tier inference backend

If you can't / don't want to fine-tune, the analyzer has two fallback
backends so the system always produces useful output:

| Tier | LID | Sentiment | Normalizer | When |
|------|-----|-----------|-----------|------|
| **neural** (best) | XLM-R fine-tuned on SentiMix | XLM-R fine-tuned on SentiMix | mT5 fine-tuned on parallel corpus | After running `scripts/train_full` |
| **pretrained** (mid) | majority-vote vocab mined from SentiMix train CoNLL (50k tokens) + rule fallback | `cardiffnlp/twitter-xlm-roberta-base-sentiment` (zero-shot, multilingual) | dictionary mined from parallel corpus (or fine-tuned mT5 if present) | `python -m scripts.analyze --backend pretrained` |
| **rule-based** (fallback) | hand-coded lexicon + heuristics | lexicon polarity counts | identity | No models / offline |

The `auto` default picks the highest tier available — trained checkpoints
first, then pretrained, then rule-based.

## Evaluation

```bash
python -m scripts.evaluate                              # auto-detect backend
python -m scripts.evaluate --backend pretrained         # zero-shot HF models
python -m scripts.evaluate --backend neural             # trained checkpoints
python -m scripts.evaluate --max 0                      # use entire test split
python -m scripts.evaluate --out reports/run1.md        # custom output path
```

### Backend comparison on the SentiMix test split (n=500)

| Metric                        | rule-based | pretrained | neural (GPU-trained) |
|-------------------------------|-----------:|-----------:|---------------------:|
| Sentiment accuracy            |       0.47 |   **0.63** |  *fill in after run* |
| Sentiment macro-F1            |       0.39 |   **0.63** |  *fill in after run* |
| LID accuracy                  |       0.54 |   **0.89** |  *fill in after run* |
| LID macro-F1                  |       0.46 |   **0.90** |  *fill in after run* |
| Code-mixed share recovered    |    82.2% [biased] | **99.6%** | — |
| Dominant-lang. distribution   | 99.5% ENG [biased] | 67% HIN / 33% ENG | — |

The rule-based backend over-predicts ENG because its English lexicon is
denser than its Hindi one — that's why CMI numbers and dominance breakdowns
look skewed. The pretrained backend (cardiffnlp sentiment + 50K-token
SentiMix-mined LID) closes most of the gap zero-shot.

The report covers:

* **Sentiment**: accuracy / macro-F1 / weighted-F1 + per-class report.
* **LID**: token-level accuracy / macro-F1 / weighted-F1 + per-class report. Token alignment with gold is exact (no whitespace re-tokenisation drift).
* **Normalizer**: corpus BLEU + chrF on the held-out parallel test pairs.
* **Code-switching profile of the test set**: CMI mean/median, switch fraction, code-mixed share, CMI buckets (monolingual / low / medium / high / very high), dominant-language histogram. Tells you what kind of code-mixing the test set actually contains, so you can read sentiment/LID numbers in context.
* **Robustness slice**: sentiment metrics restricted to noisy inputs (any `text_norm` edit fired) vs. clean inputs — quantifies how much elongations / slang / punctuation runs hurt the model.
* **5 worked examples**: full reports with token importance for spot-checking.

### Expected file formats (when bringing your own data)

| Component   | File                          | Format |
|-------------|-------------------------------|--------|
| LID         | `*.conll`                     | CoNLL: `token<TAB>label` per line, blank line between sentences. Both CrossLing canonical labels (`ENG/HIN/OTHER/NE`) and LINCE / SentiMix conventions (`lang1/lang2/Eng/Hin/O/...`) are accepted. |
| Sentiment   | `*.tsv` or `*.conll`          | TSV: `text<TAB>LABEL` per line (`POSITIVE/NEGATIVE/NEUTRAL`). SentiMix CoNLL files (with `meta\tID\tLABEL` headers) are auto-detected. |
| Normalizer  | `*.jsonl`                     | One JSON per line: `{"src": "...", "tgt": "..."}`               |

## Sample code

Single-sentence analyzer — one call, one report:

```python
from src.analyzer import HinglishAnalyzer

analyzer = HinglishAnalyzer.from_config("config.yaml")
report = analyzer.analyze("yeh movi bakwaaaas h brooo!!!")

print(report.pretty())            # human-readable
report.to_dict()                   # JSON-ready dict
```

Pretty output:

```
────────────────────────────────────────────────────────────────
  CrossLing Hinglish Analysis  [backend: neural]
────────────────────────────────────────────────────────────────
  Input        : yeh movi bakwaaaas h brooo!!!
  Cleaned      : yeh movie bakwaas hai bro!
  Edits        : elongations-collapsed:3, punct-runs-collapsed:1, slang:movi->movie,h->hai,broo->bro
  Tokens       : yeh/HIN  movie/ENG  bakwaas/HIN  hai/HIN  bro/ENG  !/OTHER
  Code-switch  : CMI=20.0  switches=2 (0.50)  dominant=HIN (0.80)  mixed=True
  Normalized   : this movie is rubbish brother
  Sentiment    : NEGATIVE  (conf=0.91)
               : POSITIVE:0.04  NEGATIVE:0.91  NEUTRAL:0.05
  Top tokens   : bakwaas (0.62),  movie (0.11),  hai (0.09),  yeh (0.08),  bro (0.06)
  Summary      : Code-mixed Hinglish (CMI 20.0); 2 switch point(s); dominant language is HIN at 80%. Sentiment classified as NEGATIVE with confidence 0.91. Top contributing tokens: 'bakwaas', 'movie', 'hai'.
────────────────────────────────────────────────────────────────
```

Or use the lower-level pipeline directly:

```python
from src.pipeline import CrossLingPipeline

pipe = CrossLingPipeline.from_config("config.yaml")
result = pipe.run("yaar this movie was bakwaas but songs were good")

print(result.tokens)            # ['yaar', 'this', 'movie', ...]
print(result.language_tags)     # ['HIN', 'ENG', 'ENG', ...]
print(result.code_switch)       # {'cmi': 33.3, 'switch_count': 2, ...}
print(result.normalized)        # 'friend this movie was rubbish but songs were good'
print(result.sentiment)         # 'NEUTRAL'
print(result.token_importance)  # [{'token': 'bakwaas', 'importance': 0.42}, ...]
```

CLI:

```bash
python -m scripts.analyze "yaar traffic itna zyada hai"          # pretty
python -m scripts.analyze --json "yaar traffic itna zyada hai"   # JSON
python -m scripts.analyze --interactive                          # REPL
```

Without trained checkpoints, the analyzer transparently falls back to a
rule-based mode (lexicon LID + lexicon sentiment + identity normalizer)
so the command works on a fresh clone before any training run.

## Evaluation

| Component   | Metric                          | Where computed             |
|-------------|----------------------------------|----------------------------|
| LID         | macro-F1, weighted-F1, accuracy + per-class report | end of `train_lid.py` |
| Sentiment   | macro-F1, weighted-F1, accuracy + per-class report | end of `train_sentiment.py` |
| Normalizer  | corpus BLEU + chrF (sacreBLEU)   | end of `train_normalizer.py` |

`compute_metrics` runs after every epoch via the HuggingFace `Trainer`, and the best checkpoint (by `macro_f1` for classifiers, `bleu` for the normalizer) is loaded at the end. Test the metric implementations directly with:

```bash
python -m unittest tests.test_pipeline
```

## Configuration

All hyperparameters live in [config.yaml](config.yaml). Key knobs:

```yaml
lid:        backbone, batch_size, lr, epochs, max_length
sentiment:  same fields
normalizer: backbone (mt5-small / mbart-large-50 / etc.), source_prefix,
            max_source_length, max_target_length, num_beams
```

Switch the LID/sentiment backbone to a smaller model (e.g. `bert-base-multilingual-cased`) on tight hardware; switch the normalizer to `Helsinki-NLP/opus-mt-hi-en` if mT5-small downloads are too heavy.

## Notes on real datasets

* **`python -m src.real_data`** pulls everything you need from HuggingFace Hub:
  * `RTT1/SentiMix` (SemEval-2020 Task 9, 393K-row CoNLL stream → 20k labelled sentences) — used for both **sentiment** and **LID** training, since each token already carries `Eng/Hin/O` tags and each sentence carries a positive/negative/neutral label.
  * `findnitai/english-to-hinglish` (189K parallel pairs, capped to 30K by default for CPU/MPS) — used for the **normalizer**.
* **LINCE** — register at `https://ritual.uh.edu/lince/`. Drop CoNLL files anywhere; the loaders accept the raw LINCE labels (`lang1/lang2/ne/other/mixed/ambiguous/fw/...`) and remap them automatically.
* **PHINC / MUTE** — convert to JSONL with `{"src": hinglish, "tgt": english}` and pass via `--normalizer-data`.

## Code-switching analysis

Beyond per-token language tags, every report carries the quantitative
measures used in the code-mixing literature:

| Field             | Meaning |
|-------------------|---------|
| `cmi`             | Code-Mixing Index (Das & Gambäck 2014). 0 = monolingual, 100 = perfectly mixed. |
| `switch_points`   | Token indices where the content language changes. |
| `switch_fraction` | switches / max(1, content_tokens − 1). |
| `dominant_language`, `dominance_ratio` | Which content language wins, and by how much. |
| `burstiness`      | Goh & Barabási 2008 statistic over inter-switch gaps; `>0` means bursty (long monolingual runs interrupted by rapid switches). Reported when ≥ 3 switches. |
| `runs`            | The (tag, length) RLE of the language sequence. |
| `is_code_mixed`   | Convenience flag (`≥ 2` content languages and `cmi > 0`). |

## Robustness

`src/text_norm.py` collapses noisy patterns *before* model inference and
records what it changed:

```python
from src.text_norm import normalize_text
trace = normalize_text("OMGGG yeh movi bakwaaaas h brooo!!!")
trace.cleaned   # 'OMGG yeh movie bakwaas hai bro!'
trace.edits     # ['elongations-collapsed:3', 'punct-runs-collapsed:1',
                #  'slang:movi->movie,h->hai,broo->bro']
```

The robustness test suite (`tests/test_robustness.py`) locks in the
behaviour on:

* elongated vowels (`bakwaaaas`)
* legitimate doubles preserved (`good`, `see`)
* punctuation runs (`!!!!`, `??`)
* chatspeak expansion (`u r plz` → `you are please`)
* URL / mention / hashtag handling
* SentiMix / LINCE label remapping
* end-to-end report shape on noisy inputs
