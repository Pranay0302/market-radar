"""The spec model: how we describe a laptop config without using its name.

Every SKU — ours or a competitor's — gets boiled down to the same set of
attributes. Resolution and traction both lean on this. If a competitor renames a
line, the attributes don't move, so nothing downstream cares.

Attributes are typed and weighted on purpose. An opaque embedding could match
configs but couldn't tell Priya *why* two configs matched, and she needs that to
trust the mapping.

TODO: the taxonomy is hand-written for laptops right now. Real deployments should
pull it from the customer's PIM / product catalog and version it per category.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np


@dataclass(frozen=True)
class AttrSpec:
    """One node type in the spec hierarchy."""

    name: str
    kind: str  # "numeric" | "ordinal" | "categorical"
    weight: float  # importance in entity resolution
    order: Optional[List[Any]] = None  # low -> high, for ordinal


# The canonical taxonomy. Order matters only for deterministic vectors.
ATTRS: List[AttrSpec] = [
    AttrSpec("cpu_tier", "ordinal", 3.0, order=[3, 5, 7, 9]),
    AttrSpec("cpu_family", "categorical", 1.0),
    AttrSpec("ram_gb", "numeric", 2.0),
    AttrSpec("storage_gb", "numeric", 1.0),
    AttrSpec("gpu_class", "ordinal", 2.5,
             order=["integrated", "rtx2000", "rtx3000", "rtx4000"]),
    AttrSpec("display_in", "numeric", 1.0),
    AttrSpec("display_panel", "categorical", 1.5),  # IPS | OLED
    AttrSpec("display_res", "ordinal", 1.0, order=["FHD", "QHD", "UHD"]),
    AttrSpec("weight_class", "ordinal", 1.0,
             order=["ultralight", "light", "standard", "heavy"]),
    AttrSpec("chassis", "categorical", 0.5),
]
ATTR_BY_NAME: Dict[str, AttrSpec] = {a.name: a for a in ATTRS}
ATTR_NAMES: List[str] = [a.name for a in ATTRS]

# Human-readable labels for the aspects sentiment mining maps spec drivers onto.
ASPECT_FOR_ATTR = {
    "display_panel": "display quality",
    "display_res": "display quality",
    "weight_class": "portability",
    "cpu_tier": "performance",
    "gpu_class": "performance",
    "ram_gb": "performance",
}


# --------------------------------------------------------------------------- #
# Entity extraction / normalization
# --------------------------------------------------------------------------- #
# Competitor listings come in messy: "Intel Core Ultra 7 155H", "32 GB", etc.
# We never trust the name, but we do need clean attribute values, so these
# helpers pull a tidy spec out of a raw listing.
# TODO: CPU/RAM parsing is regex-based and covers the common cases. Add GPU +
# display parsing and a fuzzy fallback before pointing this at real feeds.

_CPU_TIER_RE = re.compile(r"(?:core\s+)?(?:ultra\s+)?(?:i)?([3579])\b", re.I)
_RAM_RE = re.compile(r"(\d+)\s*gb", re.I)


def normalize_cpu(raw: Any) -> Dict[str, Any]:
    """'Intel Core Ultra 7 155H' -> {cpu_tier: 7, cpu_family: 'Core Ultra 7'}."""
    if isinstance(raw, (int, float)):
        tier = int(raw)
        return {"cpu_tier": tier, "cpu_family": f"tier{tier}"}
    s = str(raw)
    m = _CPU_TIER_RE.search(s)
    tier = int(m.group(1)) if m else 5
    fam = "Ryzen" if re.search(r"ryzen", s, re.I) else "Core Ultra"
    return {"cpu_tier": tier, "cpu_family": f"{fam} {tier}"}


def normalize_ram(raw: Any) -> int:
    """'32 GB' / '32GB' / 32 -> 32."""
    if isinstance(raw, (int, float)):
        return int(raw)
    m = _RAM_RE.search(str(raw))
    return int(m.group(1)) if m else 0


def _canon_num(name: str, val: Any) -> Any:
    """Whole-number numerics -> int, so 14.0 (from pandas) == 14 (from us)."""
    a = ATTR_BY_NAME.get(name)
    if a and a.kind == "numeric" and val is not None:
        f = float(val)
        return int(f) if f.is_integer() else f
    return val


def normalize_listing(raw: Dict[str, Any]) -> Dict[str, Any]:
    """Canonicalize a raw competitor listing into a spec dict.

    Accepts either already-canonical keys or messy raw fields
    (``cpu_raw``, ``ram_raw``). Anything not recognized passes through.
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
    """A stable, name-independent identity for a configuration."""
    return "|".join(f"{n}={_canon_num(n, spec.get(n))}" for n in ATTR_NAMES)


def spec_text(spec: Dict[str, Any]) -> str:
    """Natural-language bag of the spec, for embedding. No brand/model name."""
    parts = []
    for n in ATTR_NAMES:
        v = spec.get(n)
        if v is not None:
            parts.append(f"{n.replace('_', ' ')} {v}")
    return ", ".join(parts)


# --------------------------------------------------------------------------- #
# Numeric encoding (for weighted distance + NN indexing)
# --------------------------------------------------------------------------- #
@dataclass
class CatalogStats:
    """Normalization ranges computed once over the union of configs."""

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
    """Per-attribute normalized difference in [0, 1] (before weighting)."""
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
    """Weighted attribute distance. 0 = identical config. Names never enter."""
    num = sum(a.weight * _attr_component(a, spec_a.get(a.name),
                                         spec_b.get(a.name), stats)
              for a in ATTRS)
    den = sum(a.weight for a in ATTRS)
    return num / den


def similarity(spec_a: Dict[str, Any], spec_b: Dict[str, Any],
               stats: CatalogStats) -> float:
    """Structured similarity in [0, 1]."""
    return 1.0 - weighted_distance(spec_a, spec_b, stats)


def feature_vector(spec: Dict[str, Any], stats: CatalogStats) -> np.ndarray:
    """Weighted numeric encoding whose Euclidean distance ~ weighted_distance.

    Categorical attrs become weighted one-hot columns; numeric/ordinal become
    weighted scalars. Used for optional NN indexing.
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
        else:  # one categorical scalar via stable hash bucket, scaled
            cols.append(w * (hash((a.name, v)) % 997) / 997.0)
    return np.asarray(cols, dtype=float)
