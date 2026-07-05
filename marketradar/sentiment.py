"""why is a config winning? mine the reviews for the aspects buyers praise.

we lean on the rag classifier for the aspect (display, portability, ...) and on
a routed sentiment scorer for the polarity. cheap mode scores sentiment with a
tiny lexicon; quality mode uses distilbert if it's installed. the output is a
short "customers praise the display quality and portability" that feeds straight
into the recommendation rationale.

todo: aspect-level sentiment would be sharper with an aspect-based sentiment
model, but the retrieval + lexicon combo is enough to explain the "why".
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple

from .router import ModelRouter

_POS_WORDS = {"excellent", "fantastic", "impressed", "highlight", "exceeded",
              "best", "love", "great"}
_NEG_WORDS = {"disappointing", "mediocre", "weak", "struggles", "underwhelming",
              "poor"}


def lexicon_polarity(text: str) -> str:
    t = text.lower()
    pos = sum(w in t for w in _POS_WORDS)
    neg = sum(w in t for w in _NEG_WORDS)
    return "positive" if pos >= neg else "negative"


class SentimentScorer:
    """positive / negative, routed lexicon (cheap) vs distilbert (quality)."""

    def __init__(self):
        self.model = "distilbert-sst-2"
        self._pipe = None
        # importable check only. the model itself loads lazily on first use.
        try:
            import transformers  # noqa: F401
            self._available = True
        except Exception:
            self._available = False

    @property
    def available(self) -> bool:
        return self._available

    def _ensure_pipe(self):
        if self._pipe is None:
            from transformers import pipeline
            self._pipe = pipeline(
                "sentiment-analysis",
                model="distilbert-base-uncased-finetuned-sst-2-english")
        return self._pipe

    def score(self, text: str, router: ModelRouter) -> str:
        use = router.choose("sentiment", "lexicon", "distilbert", self.available)
        t0 = time.time()
        model = "lexicon"
        if use == "distilbert":
            try:
                label = self._ensure_pipe()(text)[0]["label"]
                res = "positive" if label.upper().startswith("POS") else "negative"
                model = "distilbert-sst-2"
            except Exception:
                use, res = "lexicon", lexicon_polarity(text)
        else:
            res = lexicon_polarity(text)
        router.log("sentiment", use, model, (time.time() - t0) * 1000)
        return res


@dataclass
class WhySummary:
    config_sig: str
    n_reviews: int
    top_aspects: List[Tuple[str, int, int]] = field(default_factory=list)  # aspect, pos, total
    phrase: str = ""


def _phrase(top: List[Tuple[str, int, int]]) -> str:
    if not top:
        return ""
    return "customers praise the " + " and ".join(a for a, _, _ in top)


def summarize(config_sig: str, reviews: List[Dict[str, Any]],
              classifier, scorer: SentimentScorer,
              router: ModelRouter) -> WhySummary:
    """aggregate predicted positive aspects for one config's reviews."""
    revs = [r for r in reviews if r["config_sig"] == config_sig]
    if not revs:
        return WhySummary(config_sig, 0)

    agg: Dict[str, List[int]] = {}   # aspect -> [positives, total]
    for r in revs:
        aspect = classifier.classify(r["text"])["aspect"]
        sentiment = scorer.score(r["text"], router)
        bucket = agg.setdefault(aspect, [0, 0])
        bucket[1] += 1
        if sentiment == "positive":
            bucket[0] += 1

    ranked = sorted(agg.items(),
                    key=lambda kv: (kv[1][0], kv[1][0] / kv[1][1]), reverse=True)
    top = [(a, v[0], v[1]) for a, v in ranked if v[0] > 0][:2]
    return WhySummary(config_sig, len(revs), top, _phrase(top))
