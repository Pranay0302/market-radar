"""the spec model: how we describe a laptop config without using its name.

every sku (ours or a competitor's) gets boiled down to the same set of
attributes. resolution and traction both lean on this. if a competitor renames a
line, the attributes don't move, so nothing downstream cares.

attributes are typed and weighted on purpose. an opaque embedding could match
configs but couldn't tell priya why two configs matched, and she needs that to
trust the mapping.

todo: the taxonomy is hand-written for laptops right now. real deployments should
pull it from the customer's pim / product catalog and version it per category.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np


@dataclass(frozen=True)
class AttrSpec:
    """one node type in the spec hierarchy."""

    name: str
    kind: str  # "numeric" | "ordinal" | "categorical"
    weight: float  # importance in entity resolution
    order: Optional[List[Any]] = None  # low -> high, for ordinal


# the canonical taxonomy. order only matters for deterministic vectors.
ATTRS: List[AttrSpec] = [
    AttrSpec("cpu_tier", "ordinal", 3.0, order=[3, 5, 7, 9]),
    AttrSpec("cpu_family", "categorical", 1.0),
    AttrSpec("ram_gb", "numeric", 2.0),
    AttrSpec("storage_gb", "numeric", 1.0),
    AttrSpec("gpu_class", "ordinal", 2.5,
             order=["integrated", "rtx2000", "rtx3000", "rtx4000"]),
    AttrSpec("display_in", "numeric", 1.0),
    AttrSpec("display_panel", "categorical", 1.5),  # ips | oled
    AttrSpec("display_res", "ordinal", 1.0, order=["FHD", "QHD", "UHD"]),
    AttrSpec("weight_class", "ordinal", 1.0,
             order=["ultralight", "light", "standard", "heavy"]),
    AttrSpec("chassis", "categorical", 0.5),
]
ATTR_BY_NAME: Dict[str, AttrSpec] = {a.name: a for a in ATTRS}
ATTR_NAMES: List[str] = [a.name for a in ATTRS]

# human-readable labels for the aspects sentiment mining maps spec drivers onto.
ASPECT_FOR_ATTR = {
    "display_panel": "display quality",
    "display_res": "display quality",
    "weight_class": "portability",
    "cpu_tier": "performance",
    "gpu_class": "performance",
    "ram_gb": "performance",
}


# --------------------------------------------------------------------------- #
# entity extraction / normalization
# --------------------------------------------------------------------------- #
# competitor listings come in messy: "intel core ultra 7 155h", "32 gb", etc.
# we never trust the name, but we do need clean attribute values, so these
# helpers pull a tidy spec out of a raw listing.
# todo: cpu/ram parsing is regex-based and covers the common cases. add gpu plus
# display parsing and a fuzzy fallback before pointing this at real feeds.

_CPU_TIER_RE = re.compile(r"(?:core\s+)?(?:ultra\s+)?(?:i)?([3579])\b", re.I)
_RAM_RE = re.compile(r"(\d+)\s*gb", re.I)


def normalize_cpu(raw: Any) -> Dict[str, Any]:
    """turn a messy cpu string into cpu_tier plus cpu_family."""
    if isinstance(raw, (int, float)):
        tier = int(raw)
        return {"cpu_tier": tier, "cpu_family": f"tier{tier}"}
    s = str(raw)
    m = _CPU_TIER_RE.search(s)
    tier = int(m.group(1)) if m else 5
    fam = "Ryzen" if re.search(r"ryzen", s, re.I) else "Core Ultra"
    return {"cpu_tier": tier, "cpu_family": f"{fam} {tier}"}


def normalize_ram(raw: Any) -> int:
    """pull the gb number out of '32 gb' / '32gb' / 32."""
    if isinstance(raw, (int, float)):
        return int(raw)
    m = _RAM_RE.search(str(raw))
    return int(m.group(1)) if m else 0


def _canon_num(name: str, val: Any) -> Any:
    """whole-number numerics become int, so 14.0 (from pandas) equals 14."""
    a = ATTR_BY_NAME.get(name)
    if a and a.kind == "numeric" and val is not None:
        f = float(val)
        return int(f) if f.is_integer() else f
    return val


def normalize_listing(raw: Dict[str, Any]) -> Dict[str, Any]:
    """canonicalize a raw competitor listing into a spec dict.

    accepts either already-canonical keys or messy raw fields (cpu_raw, ram_raw).
    anything not recognized passes through.
    """
    spec: Dict[str, Any] = {}
    if "cpu_raw" in raw:
        spec.update(normalize_cpu(raw["cpu_raw"]))
    if "ram_raw" in raw:
        spec["ram_gb"] = normalize_ram(raw["ram_raw"])
    for name in ATTR_NAMES:
        if name in raw and name not in spec:
            val = raw[name]
            if name == "display_panel" and isinstance(val, str):
                val = val.upper()
            spec[name] = _canon_num(name, val)
    return spec


def config_signature(spec: Dict[str, Any]) -> str:
    """a stable, name-independent identity for a configuration."""
    return "|".join(f"{n}={_canon_num(n, spec.get(n))}" for n in ATTR_NAMES)


def spec_text(spec: Dict[str, Any]) -> str:
    """natural-language bag of the spec, for embedding. no brand or model name."""
    parts = []
    for n in ATTR_NAMES:
        v = spec.get(n)
        if v is not None:
            parts.append(f"{n.replace('_', ' ')} {v}")
    return ", ".join(parts)


# --------------------------------------------------------------------------- #
# numeric encoding (for weighted distance plus nn indexing)
# --------------------------------------------------------------------------- #
@dataclass
class CatalogStats:
    """normalization ranges computed once over the union of configs."""

    numeric_range: Dict[str, tuple] = field(default_factory=dict)


def compute_stats(specs: List[Dict[str, Any]]) -> CatalogStats:
    stats = CatalogStats()
    for a in ATTRS:
        if a.kind == "numeric":
            vals = [float(s[a.name]) for s in specs if s.get(a.name) is not None]
            lo, hi = (min(vals), max(vals)) if vals else (0.0, 1.0)
            stats.numeric_range[a.name] = (lo, hi if hi > lo else lo + 1.0)
    return stats


def _attr_component(a: AttrSpec, va: Any, vb: Any, stats: CatalogStats) -> float:
    """per-attribute normalized difference in [0, 1], before weighting."""
    if va is None or vb is None:
        return 1.0
    if a.kind == "numeric":
        lo, hi = stats.numeric_range.get(a.name, (0.0, 1.0))
        return abs(float(va) - float(vb)) / (hi - lo)
    if a.kind == "ordinal":
        order = a.order or []
        try:
            ia, ib = order.index(va), order.index(vb)
        except ValueError:
            return 0.0 if va == vb else 1.0
        span = max(len(order) - 1, 1)
        return abs(ia - ib) / span
    return 0.0 if va == vb else 1.0  # categorical


def weighted_distance(spec_a: Dict[str, Any], spec_b: Dict[str, Any],
                      stats: CatalogStats) -> float:
    """weighted attribute distance. 0 means identical config. names never enter."""
    num = sum(a.weight * _attr_component(a, spec_a.get(a.name),
                                         spec_b.get(a.name), stats)
              for a in ATTRS)
    den = sum(a.weight for a in ATTRS)
    return num / den


def similarity(spec_a: Dict[str, Any], spec_b: Dict[str, Any],
               stats: CatalogStats) -> float:
    """structured similarity in [0, 1]."""
    return 1.0 - weighted_distance(spec_a, spec_b, stats)


def feature_vector(spec: Dict[str, Any], stats: CatalogStats) -> np.ndarray:
    """weighted numeric encoding whose euclidean distance tracks weighted_distance.

    categorical attrs become weighted one-hot-ish columns; numeric and ordinal
    become weighted scalars. used for optional nn indexing.
    """
    cols: List[float] = []
    for a in ATTRS:
        w = np.sqrt(a.weight)
        v = spec.get(a.name)
        if a.kind == "numeric":
            lo, hi = stats.numeric_range.get(a.name, (0.0, 1.0))
            cols.append(w * ((float(v) - lo) / (hi - lo) if v is not None else 0.0))
        elif a.kind == "ordinal":
            order = a.order or []
            idx = order.index(v) / max(len(order) - 1, 1) if v in order else 0.0
            cols.append(w * idx)
        else:  # one categorical scalar via a stable hash bucket, scaled
            cols.append(w * (hash((a.name, v)) % 997) / 997.0)
    return np.asarray(cols, dtype=float)
