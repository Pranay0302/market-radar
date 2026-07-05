"""rag few-shot aspect classifier over the review corpus.

reviews are the one genuinely unstructured input here, so retrieval fits. we
embed a bank of labeled snippets, and to classify a new one we pull its
k-nearest labeled neighbors and let them vote on the aspect and sentiment. that
is the "few-shot" part: the prediction is grounded in real retrieved examples,
not a cold guess.

vector store is chromadb when it's installed, otherwise a sklearn
nearest-neighbors index. either way the classify / evaluate interface is the same.

todo: the vote is plain majority. with flan-t5 wired in we could hand the
retrieved examples to the generator for a proper few-shot prompt in quality mode.
"""

from __future__ import annotations

import time
from collections import Counter
from typing import Any, Dict, List

from .embedding import Embedder
from .router import ModelRouter


class RagAspectClassifier:
    def __init__(self, router: ModelRouter, k: int = 5,
                 embedder_pref: str = "tfidf"):
        self.router = router
        self.k = k
        self.embedder = Embedder(prefer=embedder_pref)
        self.backend = "knn"
        self._labels: List[tuple] = []
        self._nn = None
        self._collection = None

    def fit(self, train_reviews: List[Dict[str, Any]]) -> "RagAspectClassifier":
        texts = [r["text"] for r in train_reviews]
        self._labels = [(r["aspect"], r["sentiment"]) for r in train_reviews]
        self.embedder.fit(texts)
        vecs = self.embedder.encode(texts)
        k = min(self.k, len(vecs))

        try:
            import chromadb
            client = chromadb.Client()
            name = f"reviews_{id(self)}"
            self._collection = client.create_collection(name)
            self._collection.add(
                ids=[str(i) for i in range(len(texts))],
                embeddings=[v.tolist() for v in vecs],
                metadatas=[{"aspect": a, "sentiment": s}
                           for a, s in self._labels],
            )
            self.backend = "chroma"
        except Exception:
            # no chroma, or it errored. sklearn nn does the same job here.
            from sklearn.neighbors import NearestNeighbors
            self._nn = NearestNeighbors(n_neighbors=k, metric="cosine").fit(vecs)
            self.backend = "knn"
        self._k = k
        return self

    def _neighbor_labels(self, text: str) -> List[tuple]:
        v = self.embedder.encode([text])
        if self.backend == "chroma":
            try:
                res = self._collection.query(query_embeddings=[v[0].tolist()],
                                             n_results=self._k)
                return [(m["aspect"], m["sentiment"]) for m in res["metadatas"][0]]
            except Exception:
                pass  # fall through to whatever we can
        _dist, idx = self._nn.kneighbors(v)
        return [self._labels[i] for i in idx[0]]

    def classify(self, text: str) -> Dict[str, Any]:
        t0 = time.time()
        nbrs = self._neighbor_labels(text)
        aspects = [a for a, _ in nbrs]
        sentiments = [s for _, s in nbrs]
        aspect = Counter(aspects).most_common(1)[0][0]
        sentiment = Counter(sentiments).most_common(1)[0][0]
        conf = aspects.count(aspect) / len(aspects)
        self.router.log("aspect_classify", self.backend, self.embedder.model_name,
                        (time.time() - t0) * 1000)
        return {"aspect": aspect, "sentiment": sentiment, "confidence": conf}

    def evaluate(self, test_reviews: List[Dict[str, Any]]) -> Dict[str, Any]:
        """held-out macro f1 for aspect, plus sentiment accuracy."""
        from sklearn.metrics import accuracy_score, f1_score
        preds = [self.classify(r["text"]) for r in test_reviews]
        a_true = [r["aspect"] for r in test_reviews]
        a_pred = [p["aspect"] for p in preds]
        s_true = [r["sentiment"] for r in test_reviews]
        s_pred = [p["sentiment"] for p in preds]
        return {
            "aspect_f1": round(f1_score(a_true, a_pred, average="macro",
                                        zero_division=0), 3),
            "aspect_acc": round(accuracy_score(a_true, a_pred), 3),
            "sentiment_acc": round(accuracy_score(s_true, s_pred), 3),
            "n": len(test_reviews),
            "backend": self.backend,
        }
