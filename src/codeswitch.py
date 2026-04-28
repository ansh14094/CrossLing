"""Code-switching analysis on top of LID tags.

The token-level LID head gives `[ENG, HIN, OTHER, NE]` per word. That alone
is not an analysis of code-mixing — it's just labels. This module turns those
tags into the quantitative measures used in the code-switching literature:

  * Code-Mixing Index (CMI), Das & Gambäck 2014:
      CMI = 100 * (1 - max(w_i) / (N - u))
    where w_i are token counts per language, N is total tokens (excluding
    language-independent tokens — punctuation, NEs), and u is the count of
    language-independent tokens. CMI = 0 means monolingual; higher means
    more mixed; capped at 100.

  * Switch points: positions where consecutive content-language tokens
    differ. We report both the count and the indices, plus the switch
    fraction (#switches / max(1, content_tokens - 1)).

  * Language dominance: which content language has the most tokens, and
    its share. Useful for downstream consumers (e.g. choosing a target
    norm or framing the sentiment).

  * Burstiness, Goh & Barabasi 2008, applied to switch gaps:
      B = (sigma - mu) / (sigma + mu)  in [-1, 1]
    Negative B = regular alternation; positive B = bursty (long mono-
    lingual runs separated by rapid switches). Reported when there are
    >= 3 switches.

These are the measures most commonly cited in code-mixing papers (LINCE
benchmarks, Bali et al. 2014, Das & Gambäck 2014). Computing them is what
takes the project from "I detected the languages" to "I analysed the
mixing behaviour."
"""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass, field


# Tags we treat as language-bearing. NE / OTHER are language-independent for
# CMI purposes (Das & Gambäck explicitly exclude them).
_CONTENT_TAGS = {"ENG", "HIN"}


@dataclass
class CodeSwitchStats:
    n_tokens: int
    n_content_tokens: int
    counts: dict[str, int]
    cmi: float
    switch_points: list[int]
    switch_fraction: float
    dominant_language: str | None
    dominance_ratio: float
    burstiness: float | None
    is_code_mixed: bool
    runs: list[tuple[str, int]] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "n_tokens": self.n_tokens,
            "n_content_tokens": self.n_content_tokens,
            "counts": dict(self.counts),
            "cmi": round(self.cmi, 2),
            "switch_points": list(self.switch_points),
            "switch_count": len(self.switch_points),
            "switch_fraction": round(self.switch_fraction, 3),
            "dominant_language": self.dominant_language,
            "dominance_ratio": round(self.dominance_ratio, 3),
            "burstiness": (None if self.burstiness is None
                           else round(self.burstiness, 3)),
            "is_code_mixed": self.is_code_mixed,
            "runs": [list(r) for r in self.runs],
        }


def _runs(tags: list[str]) -> list[tuple[str, int]]:
    """Group consecutive equal tags into (tag, length) pairs."""
    out: list[tuple[str, int]] = []
    for tag in tags:
        if out and out[-1][0] == tag:
            out[-1] = (tag, out[-1][1] + 1)
        else:
            out.append((tag, 1))
    return out


def _burstiness(gaps: list[int]) -> float | None:
    if len(gaps) < 2:
        return None
    mu = sum(gaps) / len(gaps)
    var = sum((g - mu) ** 2 for g in gaps) / len(gaps)
    sigma = math.sqrt(var)
    if sigma + mu == 0:
        return None
    return (sigma - mu) / (sigma + mu)


def analyze_codeswitch(tokens: list[str], tags: list[str]) -> CodeSwitchStats:
    """Compute code-switching statistics for an aligned (tokens, tags) pair."""
    if len(tokens) != len(tags):
        raise ValueError(f"tokens/tags length mismatch: "
                         f"{len(tokens)} vs {len(tags)}")

    n = len(tokens)
    counts = Counter(tags)
    content_counts = {t: c for t, c in counts.items() if t in _CONTENT_TAGS}
    n_content = sum(content_counts.values())
    u = n - n_content  # language-independent tokens (NE / OTHER)

    if n_content == 0:
        cmi = 0.0
        dominant: str | None = None
        dom_ratio = 0.0
    else:
        max_count = max(content_counts.values())
        denom = n - u
        cmi = 100.0 * (1.0 - max_count / denom) if denom > 0 else 0.0
        dominant = max(content_counts, key=content_counts.get)
        dom_ratio = max_count / n_content

    # Switch points: only over content tokens, in original order.
    content_seq = [(i, t) for i, t in enumerate(tags) if t in _CONTENT_TAGS]
    switch_points: list[int] = []
    for (i_prev, t_prev), (i_cur, t_cur) in zip(content_seq, content_seq[1:]):
        if t_prev != t_cur:
            switch_points.append(i_cur)

    if len(content_seq) > 1:
        switch_fraction = len(switch_points) / (len(content_seq) - 1)
    else:
        switch_fraction = 0.0

    # Burstiness over inter-switch gaps (in content-token units).
    gaps: list[int] = []
    last = -1
    for sp in switch_points:
        # convert original index to content-index
        c_idx = next(j for j, (orig, _) in enumerate(content_seq) if orig == sp)
        gaps.append(c_idx - last)
        last = c_idx
    burst = _burstiness(gaps) if len(switch_points) >= 3 else None

    return CodeSwitchStats(
        n_tokens=n,
        n_content_tokens=n_content,
        counts=dict(counts),
        cmi=cmi,
        switch_points=switch_points,
        switch_fraction=switch_fraction,
        dominant_language=dominant,
        dominance_ratio=dom_ratio,
        burstiness=burst,
        is_code_mixed=len(content_counts) >= 2 and cmi > 0,
        runs=_runs(tags),
    )
