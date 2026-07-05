"""text embeddings, with a graceful fallback.

if sentence-transformers is installed we use all-minilm-l6-v2. if not, we fall
back to a plain tf-idf vectorizer so the rest of the system keeps working
offline. either way you call .fit() once on a corpus, then .encode().

todo: cache minilm (and the encoded vectors) to disk so we don't reload the
model on every run.
"""

from __future__ import annotations

import time
from functools import lru_cache
from typing import List

import numpy as np


@lru_cache(maxsize=1)
def _load_minilm():
    """Load all-MiniLM-L6-v2 once per process and share it everywhere.

    The resolver and the RAG classifier each build an Embedder, so without this
    the model would load twice per run. Caching keeps a single copy in memory
    and makes every rerun after the first instant."""
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer("all-MiniLM-L6-v2")


def _l2_normalize(mat: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return mat / norms


class Embedder:
    """encodes text to vectors. cosine equals dot product because we normalize."""

    def __init__(self, prefer: str = "minilm"):
        self.backend = "tfidf"
        self.model_name = "tfidf-fallback"
        self._model = None
        self._vec = None
        self.load_ms = 0.0
        if prefer == "minilm":
            t0 = time.time()
            try:
                self._model = _load_minilm()
                self.backend = "minilm"
                self.model_name = "all-MiniLM-L6-v2"
            except Exception:
                # no sentence-transformers (or no network for the download).
                # that's fine, tf-idf still gives us a usable similarity signal.
                self._model = None
            self.load_ms = (time.time() - t0) * 1000

    def fit(self, corpus: List[str]) -> "Embedder":
        # minilm needs no fitting; tf-idf has to learn a vocabulary first.
        if self.backend == "tfidf":
            from sklearn.feature_extraction.text import TfidfVectorizer
            self._vec = TfidfVectorizer(ngram_range=(1, 2), min_df=1)
            self._vec.fit(corpus if corpus else [""])
        return self

    def encode(self, texts: List[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, 1))
        if self.backend == "minilm":
            vecs = np.asarray(self._model.encode(texts, show_progress_bar=False))
        else:
            if self._vec is None:
                self.fit(texts)
            vecs = self._vec.transform(texts).toarray()
        return _l2_normalize(vecs.astype(float))

    @staticmethod
    def cosine(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """row-wise cosine of two already-normalized matrices (a @ b.t)."""
        return a @ b.T
