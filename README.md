# CrossLing — Hinglish NLP Pipeline

End-to-end NLP for code-mixed Hindi-English (Hinglish) text. Given one
sentence — `yeh movi bakwaaaas h brooo!!!` — the system produces:

- **Cleaned input** (collapses elongations, expands chatspeak)
- **Per-token language tags** (`yeh/HIN movie/ENG bakwaas/HIN ...`)
- **Code-switching analytics** (CMI, switch points, language dominance, burstiness)
- **English translation** of the Hinglish input
- **Sentiment** (positive / negative / neutral) with confidence
- **Per-token importance** showing which words drove the sentiment prediction

All from a single function call.

## What the pipeline does

```
raw text
  └► preprocess (cleanup trace)
       └► LID  ─────────────► code-switching stats (CMI, switches, dominance)
            └► Normalizer (uses LID tags as prefix features)
                 └► Sentiment + token importance
                      └► structured report (JSON or pretty)
```

Each stage feeds the next — the normalizer takes LID tags as features rather
than throwing them away, sentiment runs directly on the code-mixed input
(translation drops affective signal), and gradient×input attribution
explains every prediction.

## Architecture

| Component   | Backbone                  | Head                          | Trained on                          |
|-------------|---------------------------|-------------------------------|-------------------------------------|
| LID         | `xlm-roberta-base`        | Linear, per subword           | SentiMix train (~14K sentences)     |
| Sentiment   | `xlm-roberta-base`        | Linear over `[CLS]`           | SentiMix train (~14K sentences)     |
| Normalizer  | `google/mt5-small`        | Encoder-decoder               | english-to-hinglish (~170K pairs)   |

XLM-R for LID/sentiment because its multilingual subword vocabulary degrades
gracefully on Romanized Hindi (`bakwaas`, `ekdum`); a single tokenizer covers
Devanagari and Latin scripts. mT5 for normalization because text-to-text
prompting fits cleanly and lets the model reorder tokens
(`Mumbai ki rains` → `Mumbai's rains`) instead of word-by-word substitution.

## Three usage tiers

The same `analyze()` API works at three quality tiers. Pick whichever fits
your hardware and patience.

| Tier            | LID                                      | Sentiment                                                       | Normalizer                                  | When to use                       |
|-----------------|------------------------------------------|-----------------------------------------------------------------|---------------------------------------------|-----------------------------------|
| **rule-based**  | hand-coded lexicon                       | lexicon polarity counts                                         | identity                                    | offline, zero install             |
| **pretrained**  | 50K-token vocab mined from SentiMix      | `cardiffnlp/twitter-xlm-roberta-base-sentiment` (zero-shot)     | dictionary mined from parallel corpus       | laptop, no training (~30 sec)     |
| **neural**      | XLM-R fine-tuned                         | XLM-R fine-tuned                                                | mT5 fine-tuned                              | best quality (1 GPU, ~30 min)     |

The analyzer auto-picks the highest tier available. Trained checkpoints
in `checkpoints/` win over pretrained, which wins over rule-based.

## Project layout

```
CrossLing/
├── config.yaml                 # all hyperparameters
├── requirements.txt
├── src/
│   ├── data.py                 # dataset loaders + LINCE/SentiMix label remapping
│   ├── real_data.py            # HuggingFace Hub downloader
│   ├── text_norm.py            # noise cleanup with audit trace
│   ├── codeswitch.py           # CMI, switch points, dominance, burstiness
│   ├── models.py               # LID / Sentiment / Normalizer wrappers + token importance
│   ├── pipeline.py             # end-to-end CrossLingPipeline
│   ├── analyzer.py             # HinglishAnalyzer.analyze(text) → rich report
│   ├── pretrained.py           # zero-shot HF backend
│   ├── metrics.py              # F1, BLEU, chrF
│   └── train_*.py              # per-component training entry points
├── scripts/
│   ├── train_full.py           # GPU full real-data training (canonical)
│   ├── train_all.py            # synthetic / --use-real
│   ├── evaluate.py             # writes reports/eval.md
│   ├── analyze.py              # single-sentence CLI
│   └── demo.py                 # batched / interactive demo
├── tests/                      # 26 tests covering data, metrics, code-switch, robustness
├── data/                       # populated by `python -m src.real_data`
└── checkpoints/                # populated by `python -m scripts.train_full`
```

## Quick start

### Tier 1 — laptop, zero training (uses pretrained models)

```bash
pip install -r requirements.txt
python -m src.real_data                                  # ~30 MB download
python -m scripts.analyze --backend pretrained "yeh movi bakwaaaas h brooo!!!"
python -m scripts.evaluate --backend pretrained --max 500
```

Uses `cardiffnlp/twitter-xlm-roberta-base-sentiment` (multilingual, handles
real Hinglish out of the box) and a vocabulary mined from the SentiMix
training data. **No training required.**

### Tier 2 — GPU, full real-data training (best quality)

```bash
pip install -r requirements.txt
python -m src.real_data --norm-cap 0                     # full ~189K parallel pairs
CUDA_VISIBLE_DEVICES=0 python -m scripts.train_full --device cuda --epochs 1
python -m scripts.evaluate --backend neural --max 0 --out reports/eval_neural.md
python -m scripts.analyze "yeh movi bakwaaaas h brooo!!!"
```

Trains XLM-R for LID + sentiment, mT5-small for translation. Mixed precision
auto-selected (bf16 on Ampere+, fp16 on older CUDA, fp32 otherwise; mT5 is
forced to bf16 or fp32 because fp16 is unstable for it).

### Tier 3 — laptop, synthetic training (sanity check only)

```bash
python -m scripts.train_all                              # ~10 min on MPS
python -m scripts.analyze "yeh movi bakwaaaas h brooo!!!"
```

The bundled synthetic corpus is small — useful for verifying the training
loop end-to-end, but **won't generalise to real-world Hinglish**. Prefer
Tier 1 or 2 for actual results.

## Sample output

```
────────────────────────────────────────────────────────────────
  CrossLing Hinglish Analysis  [backend: neural]
────────────────────────────────────────────────────────────────
  Hinglish     : yeh movi bakwaaaas h brooo!!!
  ENGLISH      : this movie is rubbish bro

  Cleaned      : yeh movie bakwaas hai bro!
  Edits        : elongations-collapsed:3, punct-runs-collapsed:1, slang:movi→movie,h→hai,broo→bro
  Tokens       : yeh/HIN  movie/ENG  bakwaas/HIN  hai/HIN  bro/ENG  !/OTHER
  Code-switch  : CMI=20.0  switches=2 (0.50)  dominant=HIN (0.80)  mixed=True
  Sentiment    : NEGATIVE  (conf=0.91)
               : POSITIVE:0.04  NEGATIVE:0.91  NEUTRAL:0.05
  Top tokens   : bakwaas (0.62),  movie (0.11),  hai (0.09),  yeh (0.08),  bro (0.06)
  Summary      : Code-mixed Hinglish (CMI 20.0); 2 switch point(s);
                 dominant language is HIN at 80%. Sentiment classified
                 as NEGATIVE with confidence 0.91.
────────────────────────────────────────────────────────────────
```

## Real datasets

`python -m src.real_data` pulls from HuggingFace Hub (no registration):

| Dataset                            | Size                                    | Used for          | License    |
|------------------------------------|-----------------------------------------|-------------------|------------|
| `RTT1/SentiMix`                    | 14K train + 3K validation (test labels held by the original SemEval competition; not public) | Sentiment + LID    | research   |
| `findnitai/english-to-hinglish`    | ~189K parallel pairs (90/5/5 split)     | Normalizer        | open       |

> **Note on SentiMix test labels.** The publicly distributed RTT1/SentiMix
> dataset only ships labels for the *train* and *validation* splits — the
> test split is the held-out SemEval-2020 competition file with no public
> labels. CrossLing uses the **validation split as its held-out evaluation
> set** (3000 labelled sentences). The downloader still writes
> `test.tsv` / `test.conll`, but they will be empty; evaluate against
> `data/sentimix/validation.{tsv,conll}` instead.

```bash
# Eval on validation split (the labelled held-out set)
python -m scripts.evaluate --backend neural \
    --sentiment-data data/sentimix/validation.tsv \
    --lid-data       data/sentimix/validation.conll
```

## Evaluation

```bash
python -m scripts.evaluate                                        # auto-detect backend
python -m scripts.evaluate --backend pretrained                   # zero-shot HF models
python -m scripts.evaluate --backend neural                       # trained checkpoints
python -m scripts.evaluate --max 0                                # use entire eval split
python -m scripts.evaluate --out reports/run1.md                  # custom output path
```

## Code-switching analytics

Beyond per-token language tags, every report carries the quantitative
measures used in the code-mixing literature:

| Field             | Meaning |
|-------------------|---------|
| `cmi`             | Code-Mixing Index (Das & Gambäck 2014). 0 = monolingual, 100 = perfectly mixed. |
| `switch_points`   | Token indices where the content language changes |
| `switch_fraction` | switches / max(1, content_tokens − 1) |
| `dominant_language`, `dominance_ratio` | Which content language wins, and by how much |
| `burstiness`      | Goh & Barabási 2008; >0 = bursty (long monolingual runs interrupted by rapid switches). Reported when ≥ 3 switches. |
| `runs`            | The (tag, length) RLE of the language sequence |
| `is_code_mixed`   | `≥ 2` content languages and `cmi > 0` |

## Robustness

`src/text_norm.py` collapses noisy patterns *before* model inference and
records what it changed:

```python
from src.text_norm import normalize_text
trace = normalize_text("OMGGG yeh movi bakwaaaas h brooo!!!")
trace.cleaned   # 'OMGG yeh movie bakwaas hai bro!'
trace.edits     # ['elongations-collapsed:3', 'punct-runs-collapsed:1',
                #  'slang:movi→movie,h→hai,broo→bro']
```

Tests at `tests/test_robustness.py` lock in:

- elongated vowels (`bakwaaaas`)
- legitimate doubles preserved (`good`, `see`)
- punctuation runs (`!!!!`, `??`)
- chatspeak expansion (`u r plz` → `you are please`)
- URL / mention / hashtag handling
- SentiMix / LINCE label remapping
- end-to-end report shape on noisy inputs

## Tests

```bash
python -m unittest tests.test_pipeline tests.test_robustness   # 26 tests
```

Covers data loaders, metrics, tokenization, code-switch metrics, and the
text-normalisation preprocessor.
