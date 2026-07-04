"""Fake data we can reason about — competitor market, our portfolio, reviews.

Two signals are planted on purpose so the pipeline (and the tests) have a known
answer to check against:

1. Restructure — the competitor "Nimbus" renames its whole line at week 7
   (ProBook -> EliteLine, numbering reshuffled). The spec of each SKU doesn't
   change, so name-matching breaks but spec resolution keeps working.
2. Traction — configs that are both 32GB RAM and OLED show rising sell-out at a
   stable price; everything else stays flat.

Seeded, so runs repeat.

TODO: this is a stand-in for real channel/distributor feeds. A real connector
should land data in the same Dataset shape so nothing downstream changes.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List

import numpy as np
import pandas as pd

WEEKS = 12
RESTRUCTURE_WEEK = 7
GPU_RANK = {"integrated": 0, "rtx2000": 1, "rtx3000": 2, "rtx4000": 3}

# --- Archetype configs (the universe of configurations) --------------------- #
# WINNER = 32GB + OLED  (A1, A2, A5, A10).
_ARCHETYPES: Dict[str, Dict[str, Any]] = {
    "A0":  dict(cpu_tier=7, cpu_family="Core Ultra 7", ram_gb=16, storage_gb=512,
                gpu_class="integrated", display_in=14, display_panel="IPS",
                display_res="FHD", weight_class="light", chassis="aluminum"),
    "A1":  dict(cpu_tier=7, cpu_family="Core Ultra 7", ram_gb=32, storage_gb=1024,
                gpu_class="integrated", display_in=14, display_panel="OLED",
                display_res="QHD", weight_class="light", chassis="aluminum"),
    "A2":  dict(cpu_tier=9, cpu_family="Core Ultra 9", ram_gb=32, storage_gb=2048,
                gpu_class="rtx3000", display_in=16, display_panel="OLED",
                display_res="UHD", weight_class="standard", chassis="magnesium"),
    "A3":  dict(cpu_tier=5, cpu_family="Core Ultra 5", ram_gb=16, storage_gb=512,
                gpu_class="integrated", display_in=14, display_panel="IPS",
                display_res="FHD", weight_class="standard", chassis="plastic"),
    "A4":  dict(cpu_tier=7, cpu_family="Core Ultra 7", ram_gb=32, storage_gb=1024,
                gpu_class="integrated", display_in=16, display_panel="IPS",
                display_res="QHD", weight_class="standard", chassis="aluminum"),
    "A5":  dict(cpu_tier=9, cpu_family="Core Ultra 9", ram_gb=32, storage_gb=2048,
                gpu_class="rtx3000", display_in=16, display_panel="OLED",
                display_res="UHD", weight_class="heavy", chassis="magnesium"),
    "A6":  dict(cpu_tier=5, cpu_family="Core Ultra 5", ram_gb=16, storage_gb=256,
                gpu_class="integrated", display_in=13.3, display_panel="IPS",
                display_res="FHD", weight_class="ultralight", chassis="magnesium"),
    "A7":  dict(cpu_tier=7, cpu_family="Core Ultra 7", ram_gb=16, storage_gb=512,
                gpu_class="integrated", display_in=14, display_panel="OLED",
                display_res="QHD", weight_class="light", chassis="aluminum"),
    "A8":  dict(cpu_tier=9, cpu_family="Core Ultra 9", ram_gb=32, storage_gb=2048,
                gpu_class="rtx4000", display_in=16, display_panel="IPS",
                display_res="UHD", weight_class="heavy", chassis="magnesium"),
    "A9":  dict(cpu_tier=5, cpu_family="Ryzen 5", ram_gb=16, storage_gb=512,
                gpu_class="integrated", display_in=16, display_panel="IPS",
                display_res="FHD", weight_class="standard", chassis="plastic"),
    "A10": dict(cpu_tier=7, cpu_family="Core Ultra 7", ram_gb=32, storage_gb=1024,
                gpu_class="rtx3000", display_in=16, display_panel="OLED",
                display_res="QHD", weight_class="standard", chassis="aluminum"),
    "A11": dict(cpu_tier=9, cpu_family="Ryzen 9", ram_gb=16, storage_gb=1024,
                gpu_class="rtx3000", display_in=16, display_panel="IPS",
                display_res="UHD", weight_class="standard", chassis="aluminum"),
}


def _is_winner(spec: Dict[str, Any]) -> bool:
    return spec["ram_gb"] == 32 and spec["display_panel"] == "OLED"


def _base_price(spec: Dict[str, Any]) -> float:
    return round(650 + spec["cpu_tier"] * 90 + GPU_RANK[spec["gpu_class"]] * 260
                 + (150 if spec["display_panel"] == "OLED" else 0)
                 + spec["ram_gb"] * 3 + spec["display_in"] * 18, -1)


# Which competitor brands carry which archetypes.
_COMPETITOR_MAP = {
    "Nimbus": ["A1", "A2", "A4", "A7", "A10"],   # this brand restructures
    "Vertex": ["A0", "A5", "A8", "A11", "A3"],
    "Cirro":  ["A1", "A6", "A9", "A10", "A2"],
}

# Own portfolio per tenant (enterprise isolation demo uses two tenants).
_OWN_MAP = {
    "acme-pc":   ["A0", "A1", "A3", "A4", "A6", "A7", "A9", "A10", "A11"],
    "globex-pc": ["A2", "A5", "A8", "A11"],
}


# --- Review vocabulary (for RAG few-shot aspect classification) -------------- #
_ASPECT_VOCAB = {
    "display quality": ["screen", "OLED display", "colors", "vivid panel",
                        "brightness", "the display"],
    "portability":     ["light chassis", "thin and portable", "easy to carry",
                        "travel weight", "featherlight", "the form factor"],
    "performance":     ["snappy multitasking", "fast CPU", "render speed",
                        "handles heavy workloads", "responsiveness", "the processor"],
    "battery life":    ["battery life", "hours unplugged", "charge cycles",
                        "all-day battery", "the endurance"],
    "keyboard":        ["keyboard feel", "the trackpad", "key travel",
                        "typing experience", "the keys"],
}
_POS = ["is excellent", "is fantastic", "really impressed me", "is a highlight",
        "exceeded expectations", "is best in class"]
_NEG = ["is disappointing", "feels mediocre", "is a weak point",
        "struggles noticeably", "is underwhelming"]


@dataclass
class Dataset:
    own_portfolio: Dict[str, List[Dict[str, Any]]]  # tenant_id -> SKUs
    market: pd.DataFrame                            # competitor weekly timeseries
    reviews: List[Dict[str, Any]]                   # labeled review corpus
    archetypes: Dict[str, Dict[str, Any]]

    def own_skus(self, tenant_id: str) -> List[Dict[str, Any]]:
        return self.own_portfolio[tenant_id]


def _make_own_portfolio(rng: np.random.RandomState) -> Dict[str, List[Dict]]:
    portfolios: Dict[str, List[Dict]] = {}
    for tenant, arche_ids in _OWN_MAP.items():
        skus = []
        for i, aid in enumerate(arche_ids):
            spec = dict(_ARCHETYPES[aid])
            cost = _base_price(spec) * (0.62 + rng.uniform(-0.03, 0.03))
            price = _base_price(spec) * (1.0 + rng.uniform(-0.02, 0.05))
            demand = int(rng.uniform(120, 340))
            inventory = int(demand * rng.uniform(2.5, 9.0))
            floor = 0.18
            skus.append(dict(
                sku_id=f"{tenant.split('-')[0].upper()}-{i:02d}",
                tenant_id=tenant, archetype=aid,
                model_name=f"{tenant.split('-')[0].title()} Vega {i+1}",
                spec=spec,
                unit_cost=round(cost, 2), list_price=round(price, 2),
                margin_floor_pct=floor,
                inventory_units=inventory, weekly_demand=demand,
            ))
        # Plant two revealing commercial states in acme-pc's portfolio:
        if tenant == "acme-pc":
            # A10 winner but supply-constrained (must NOT be pushed).
            for s in skus:
                if s["archetype"] == "A10":
                    s["weekly_demand"] = 300
                    s["inventory_units"] = 240          # < 1 week cover
                # A4 sits right at the margin floor (no room to cut price).
                if s["archetype"] == "A4":
                    s["unit_cost"] = round(s["list_price"] * (1 - 0.18) - 1, 2)
        portfolios[tenant] = skus
    return portfolios


def _make_market(rng: np.random.RandomState) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for brand, arche_ids in _COMPETITOR_MAP.items():
        for j, aid in enumerate(arche_ids):
            spec = _ARCHETYPES[aid]
            sku_id = f"{brand[:4].upper()}-{j:02d}"
            price0 = _base_price(spec)
            base_units = rng.uniform(240, 520)
            slope = base_units * 0.065 if _is_winner(spec) else base_units * 0.002
            for w in range(1, WEEKS + 1):
                units = base_units + slope * (w - 1) + rng.normal(0, base_units * 0.04)
                price = price0 * (1 + rng.normal(0, 0.008))  # stable band
                # Restructure: Nimbus renames its line at RESTRUCTURE_WEEK.
                if brand == "Nimbus" and w < RESTRUCTURE_WEEK:
                    model = f"Nimbus ProBook {14 + j}"
                elif brand == "Nimbus":
                    model = f"Nimbus EliteLine {90 - j*7}"  # reshuffled numbering
                else:
                    model = f"{brand} {aid}"
                row = dict(
                    week=w, brand=brand, sku_id=sku_id, model_name=model,
                    channel=rng.choice(["distributor", "etail", "direct"]),
                    avg_price=round(price, 2), units_sold=int(max(units, 0)),
                    _true_archetype=aid,
                    # messy raw fields exercise the normalization / entity-extraction path
                    spec_cpu_raw=f"Intel {spec['cpu_family']} {spec['cpu_tier']}5H"
                                 if "Core" in spec["cpu_family"]
                                 else f"AMD {spec['cpu_family']} PRO",
                    spec_ram_raw=f"{spec['ram_gb']} GB",
                )
                for k in ("storage_gb", "gpu_class", "display_in",
                          "display_panel", "display_res", "weight_class",
                          "chassis", "cpu_family", "cpu_tier", "ram_gb"):
                    row[f"spec_{k}"] = spec[k]
                rows.append(row)
    return pd.DataFrame(rows)


def _make_reviews(rng: np.random.RandomState) -> List[Dict[str, Any]]:
    from .schema import config_signature
    reviews: List[Dict[str, Any]] = []
    rid = 0
    seen = set()
    for aid, spec in _ARCHETYPES.items():
        sig = config_signature(spec)
        if sig in seen:
            continue
        seen.add(sig)
        winner = _is_winner(spec)
        for _ in range(7):
            aspect = rng.choice(list(_ASPECT_VOCAB))
            # Winners skew positive on display + portability (the "why").
            if winner and aspect in ("display quality", "portability"):
                senti = "positive" if rng.uniform() < 0.9 else "negative"
            elif winner:
                senti = "positive" if rng.uniform() < 0.6 else "negative"
            else:
                senti = "positive" if rng.uniform() < 0.45 else "negative"
            phrase = rng.choice(_ASPECT_VOCAB[aspect])
            tail = rng.choice(_POS if senti == "positive" else _NEG)
            reviews.append(dict(
                review_id=f"R{rid:04d}", archetype=aid, config_sig=sig,
                text=f"{phrase.capitalize()} {tail}.",
                aspect=aspect, sentiment=senti,
            ))
            rid += 1
    rng.shuffle(reviews)
    split = int(len(reviews) * 0.7)
    for i, r in enumerate(reviews):
        r["split"] = "train" if i < split else "test"
    return reviews


_CACHE: Dict[int, Dataset] = {}


def generate(seed: int = 7) -> Dataset:
    """Build the full synthetic dataset (cached per seed)."""
    if seed in _CACHE:
        return _CACHE[seed]
    rng = np.random.RandomState(seed)
    ds = Dataset(
        own_portfolio=_make_own_portfolio(rng),
        market=_make_market(rng),
        reviews=_make_reviews(rng),
        archetypes=_ARCHETYPES,
    )
    _CACHE[seed] = ds
    return ds


def latest_competitor_configs(market: pd.DataFrame) -> List[Dict[str, Any]]:
    """One canonical config per competitor SKU, from the most recent week.

    Runs the raw listing through the normalization / entity-extraction path so
    resolution never depends on ``model_name``.
    """
    from .schema import ATTR_NAMES, normalize_listing
    latest = market[market["week"] == market["week"].max()]
    out: List[Dict[str, Any]] = []
    for _, r in latest.iterrows():
        raw = {"cpu_raw": r["spec_cpu_raw"], "ram_raw": r["spec_ram_raw"]}
        for n in ATTR_NAMES:
            if n not in ("cpu_tier", "cpu_family", "ram_gb"):
                raw[n] = r[f"spec_{n}"]
        spec = normalize_listing(raw)
        out.append(dict(sku_id=r["sku_id"], brand=r["brand"],
                        model_name=r["model_name"], spec=spec,
                        _true_archetype=r["_true_archetype"]))
    return out
