"""Text embeddings, with a graceful fallback.

If sentence-transformers is installed we use all-MiniLM-L6-v2. If not, we fall
back to a plain TF-IDF vectorizer so the rest of the system keeps working
offline. Either way you call .fit() once on a corpus, then .encode().

TODO: cache MiniLM (and the encoded vectors) to disk so we don't reload the
model on every run.
"""

from __future__ import annotations

import time
from typing import List

import numpy as np


def _l2_normalize(mat: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return mat / norms


class Embedder:
    """Encodes text to vectors. cosine == dot product because we normalize."""

    def __init__(self, prefer: str = "minilm"):
        self.backend = "tfidf"
        self.model_name = "tfidf-fallback"
        self._model = None
        self._vec = None
        self.load_ms = 0.0
        if prefer == "minilm":
            t0 = time.time()
            try:
                from sentence_transformers import SentenceTransformer
                self._model = SentenceTransformer("all-MiniLM-L6-v2")
                self.backend = "minilm"
                self.model_name = "all-MiniLM-L6-v2"
            except Exception:
                # No sentence-transformers (or no network for the download).
                # Fine — TF-IDF still gives us a usable similarity signal.
                self._model = None
            self.load_ms = (time.time() - t0) * 1000

    def fit(self, corpus: List[str]) -> "Embedder":
        # MiniLM needs no fitting; TF-IDF has to learn a vocabulary first.
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
        """Row-wise cosine of two already-normalized matrices (a @ b.T)."""
        return a @ b.T
