"""Entity resolution: which of our SKUs does a competitor config map to?

This is the hard requirement — the mapping has to survive a competitor renaming
or restructuring its lineup. So we never look at the name. We score each
competitor config against each of our configs three ways and blend them:

  - structured similarity  (weighted attribute distance, schema.py)
  - graph attribute overlap (Jaccard over the knowledge graph)
  - embedding cosine        (spec text, MiniLM or TF-IDF)

Structured similarity does most of the work; the graph and embedding scores act
as tie-breakers and give us something to show the user ("matched on these
attributes").

TODO: blend weights are hand-picked. Once we have labeled match data, learn them
(and add a confidence threshold that flags "no good match" instead of forcing one).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from . import graph as kg
from .embedding import Embedder
from .schema import CatalogStats, compute_stats, config_signature, similarity, spec_text

# Blend weights. Structured match leads; the other two refine.
W_STRUCT, W_JACCARD, W_EMBED = 0.6, 0.2, 0.2


@dataclass
class Match:
    competitor_sku: str
    competitor_brand: Optional[str]
    competitor_model: Optional[str]        # kept for display only, never scored
    own_sku: str
    score: float
    struct_sim: float
    jaccard: float
    embed_sim: float
    shared_attrs: List[str]


@dataclass
class ResolutionResult:
    matches: List[Match]
    graph: Any                              # the networkx graph, for the UI
    by_competitor: Dict[str, Match]

    def own_for(self, competitor_sku: str) -> Optional[str]:
        m = self.by_competitor.get(competitor_sku)
        return m.own_sku if m else None


def _blend(struct: float, jac: float, emb: float) -> float:
    return W_STRUCT * struct + W_JACCARD * jac + W_EMBED * emb


def resolve_one(spec: Dict[str, Any], own_skus: List[Dict[str, Any]],
                stats: CatalogStats, embedder: Embedder) -> Dict[str, Any]:
    """Best own SKU for a single spec. Handy for tests and one-off lookups."""
    comp_vec = embedder.encode([spec_text(spec)])
    best = None
    for sku in own_skus:
        struct = similarity(spec, sku["spec"], stats)
        emb = float(Embedder.cosine(comp_vec, embedder.encode([spec_text(sku["spec"])]))[0, 0])
        # No graph here, so approximate overlap with the structured score.
        score = _blend(struct, struct, emb)
        if best is None or score > best["score"]:
            best = dict(own_sku=sku["sku_id"], score=score, struct_sim=struct,
                        embed_sim=emb)
    return best


def resolve(own_skus: List[Dict[str, Any]],
            competitor_configs: List[Dict[str, Any]]) -> ResolutionResult:
    """Resolve every competitor config to our closest own SKU."""
    all_specs = [s["spec"] for s in own_skus] + [c["spec"] for c in competitor_configs]
    stats = compute_stats(all_specs)

    # Fit the embedder once on every spec text (own + competitor).
    embedder = Embedder().fit([spec_text(s) for s in all_specs])
    own_texts = embedder.encode([spec_text(s["spec"]) for s in own_skus])

    g = kg.build_graph(own_skus, competitor_configs)

    matches: List[Match] = []
    by_comp: Dict[str, Match] = {}
    for cfg in competitor_configs:
        comp_sig = config_signature(cfg["spec"])
        comp_vec = embedder.encode([spec_text(cfg["spec"])])
        embed_sims = Embedder.cosine(comp_vec, own_texts)[0]

        best: Optional[Match] = None
        for i, sku in enumerate(own_skus):
            own_sig = config_signature(sku["spec"])
            struct = similarity(cfg["spec"], sku["spec"], stats)
            jac = kg.attribute_jaccard(g, comp_sig, own_sig)
            emb = float(embed_sims[i])
            score = _blend(struct, jac, emb)
            if best is None or score > best.score:
                best = Match(
                    competitor_sku=cfg["sku_id"], competitor_brand=cfg.get("brand"),
                    competitor_model=cfg.get("model_name"), own_sku=sku["sku_id"],
                    score=score, struct_sim=struct, jaccard=jac, embed_sim=emb,
                    shared_attrs=kg.shared_attribute_labels(g, comp_sig, own_sig),
                )
        if best is not None:
            kg.add_resolution(g, comp_sig,
                              config_signature(
                                  next(s["spec"] for s in own_skus
                                       if s["sku_id"] == best.own_sku)),
                              best.score)
            matches.append(best)
            by_comp[cfg["sku_id"]] = best

    return ResolutionResult(matches=matches, graph=g, by_competitor=by_comp)
