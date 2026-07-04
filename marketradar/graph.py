"""the portfolio as a knowledge graph.

nodes are skus, configs, and attribute values; edges connect a sku to its config
and a config to each of its attributes:

    sku --has_config--> config --has_attribute--> attribute

entity resolution (resolver.py) then adds config --resolves_to--> config edges
linking a competitor config to our closest own config. the whole point: the
graph is built from attributes, never names, so a rename doesn't touch it.

we use networkx because it's zero-infra and the graph is small.

todo: for a real multi-tenant deployment this maps straight onto neo4j. same
node/edge shape, and the resolves_to lookup becomes a cypher query.
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

import networkx as nx

from .schema import ATTR_NAMES, config_signature


def attr_node(name: str, value: Any) -> Tuple:
    return ("attr", name, value)


def config_node(sig: str) -> Tuple:
    return ("config", sig)


def sku_node(sku_id: str) -> Tuple:
    return ("sku", sku_id)


def _add_config(g: nx.DiGraph, spec: Dict[str, Any]) -> Tuple:
    sig = config_signature(spec)
    cnode = config_node(sig)
    if cnode not in g:
        g.add_node(cnode, kind="config", spec=spec, sig=sig)
        for name in ATTR_NAMES:
            val = spec.get(name)
            if val is None:
                continue
            anode = attr_node(name, val)
            if anode not in g:
                g.add_node(anode, kind="attr", attr=name, value=val)
            g.add_edge(cnode, anode, rel="HAS_ATTRIBUTE")
    return cnode


def build_graph(own_skus: List[Dict[str, Any]],
                competitor_configs: List[Dict[str, Any]]) -> nx.DiGraph:
    """build the sku/config/attribute graph for one tenant plus the market."""
    g = nx.DiGraph()
    for sku in own_skus:
        cnode = _add_config(g, sku["spec"])
        snode = sku_node(sku["sku_id"])
        g.add_node(snode, kind="sku", side="own", sku_id=sku["sku_id"],
                   tenant_id=sku.get("tenant_id"), model_name=sku["model_name"])
        g.add_edge(snode, cnode, rel="HAS_CONFIG")
    for cfg in competitor_configs:
        cnode = _add_config(g, cfg["spec"])
        snode = sku_node(cfg["sku_id"])
        g.add_node(snode, kind="sku", side="competitor", sku_id=cfg["sku_id"],
                   brand=cfg.get("brand"), model_name=cfg.get("model_name"))
        g.add_edge(snode, cnode, rel="HAS_CONFIG")
    return g


def config_attrs(g: nx.DiGraph, sig: str) -> set:
    """attribute nodes hanging off a config, used for overlap scoring."""
    cnode = config_node(sig)
    if cnode not in g:
        return set()
    return {n for n in g.successors(cnode) if g.nodes[n]["kind"] == "attr"}


def attribute_jaccard(g: nx.DiGraph, sig_a: str, sig_b: str) -> float:
    """how much two configs' attribute sets overlap, straight off the graph."""
    a, b = config_attrs(g, sig_a), config_attrs(g, sig_b)
    if not a and not b:
        return 0.0
    return len(a & b) / len(a | b)


def add_resolution(g: nx.DiGraph, comp_sig: str, own_sig: str,
                   score: float) -> None:
    g.add_edge(config_node(comp_sig), config_node(own_sig),
               rel="RESOLVES_TO", score=round(float(score), 4))


def shared_attribute_labels(g: nx.DiGraph, sig_a: str, sig_b: str) -> List[str]:
    """human labels for the attributes two configs share, nice for the ui."""
    shared = config_attrs(g, sig_a) & config_attrs(g, sig_b)
    return [f"{n[1]}={n[2]}" for n in sorted(shared)]
