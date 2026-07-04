"""cost-aware routing across the open-source backends.

every ml task here has a cheap option and a more capable one. the router picks
between them based on a mode flag and whether the capable model is actually
installed, then keeps a log of what ran so we can show the cost / latency story.

"cost" is a synthetic compute-unit proxy (local models are free in dollars but
not in latency), so the routing decision has something to optimize against.

todo: swap the flat cost table for measured p50 latency per backend and let the
router pick by a real latency budget.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

# rough relative cost per call. cheap tasks stay cheap; the generator is dear.
COST = {
    "template": 0, "lexicon": 1, "knn": 1, "tfidf": 1, "chroma": 2,
    "minilm": 5, "distilbert": 8, "flan-t5": 20,
}


@dataclass
class CallRecord:
    task: str
    backend: str
    model: str
    latency_ms: float
    cost_units: int
    note: str = ""


@dataclass
class ModelRouter:
    mode: str = "cheap"                 # "cheap" or "quality"
    calls: List[CallRecord] = field(default_factory=list)

    def choose(self, task: str, cheap: str, quality: str,
               quality_available: bool) -> str:
        """pick a backend. quality mode uses the capable model when it's there."""
        if quality_available and self.mode == "quality":
            return quality
        return cheap

    def log(self, task: str, backend: str, model: str, latency_ms: float,
            note: str = "") -> None:
        self.calls.append(CallRecord(task, backend, model, round(latency_ms, 1),
                                     COST.get(backend, 1), note))

    def summary(self) -> Dict:
        by_backend: Dict[str, int] = {}
        for c in self.calls:
            by_backend[c.backend] = by_backend.get(c.backend, 0) + 1
        return {
            "mode": self.mode,
            "total_calls": len(self.calls),
            "total_cost_units": sum(c.cost_units for c in self.calls),
            "total_latency_ms": round(sum(c.latency_ms for c in self.calls), 1),
            "calls_by_backend": by_backend,
            "models_used": sorted({c.model for c in self.calls}),
        }
