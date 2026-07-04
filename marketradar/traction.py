"""traction detection: which competitor configs are winning, and on what.

two parts:

1. per competitor sku, look at its weekly sell-out. a config has "traction" if
   units are trending up while the price stays flat, which is a real demand
   shift and not a price cut. we measure the slope of units over weeks
   (normalized by average volume) and check the price is in a tight band.
2. attribution: across all configs, which attribute values line up with the
   rising ones? that's how we get "32gb + oled is what's pulling" instead of
   just "these skus are up".

todo: a linear slope is fine for 12 weeks of clean synthetic data. real feeds
want seasonality handling and a significance test before we alert.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

from .schema import ATTR_NAMES, config_signature

# thresholds. deliberately explicit so they're easy to tune and defend.
PRICE_COV_MAX = 0.04     # price is "stable" under 4% coefficient of variation
NORM_SLOPE_MIN = 0.02    # at least ~2% weekly volume growth
R2_MIN = 0.30            # trend should be reasonably linear, not noise


@dataclass
class TractionSignal:
    brand: str
    sku_id: str
    spec: Dict[str, Any]
    config_sig: str
    norm_slope: float          # weekly growth as a fraction of mean volume
    price_mean: float
    price_cov: float
    r2: float
    drivers: List[Tuple[str, Any, float]] = field(default_factory=list)


def _spec_from_row(row: pd.Series) -> Dict[str, Any]:
    return {n: row[f"spec_{n}"] for n in ATTR_NAMES}


def _fit_trend(weeks: np.ndarray, units: np.ndarray) -> Tuple[float, float]:
    """return (normalized slope, r^2) of a straight-line fit."""
    if len(weeks) < 3 or units.mean() == 0:
        return 0.0, 0.0
    slope, intercept = np.polyfit(weeks, units, 1)
    pred = slope * weeks + intercept
    ss_res = float(np.sum((units - pred) ** 2))
    ss_tot = float(np.sum((units - units.mean()) ** 2))
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
    return slope / units.mean(), r2


def _per_config(market: pd.DataFrame) -> List[Dict[str, Any]]:
    """roll the weekly market up to one record per competitor sku."""
    out = []
    for (brand, sku_id), g in market.groupby(["brand", "sku_id"]):
        g = g.sort_values("week")
        weeks = g["week"].to_numpy(float)
        units = g["units_sold"].to_numpy(float)
        prices = g["avg_price"].to_numpy(float)
        norm_slope, r2 = _fit_trend(weeks, units)
        price_cov = float(prices.std() / prices.mean()) if prices.mean() else 1.0
        spec = _spec_from_row(g.iloc[-1])
        out.append(dict(brand=brand, sku_id=sku_id, spec=spec,
                        config_sig=config_signature(spec),
                        norm_slope=norm_slope, r2=r2,
                        price_mean=float(prices.mean()), price_cov=price_cov))
    return out


def _attribution(configs: List[Dict[str, Any]]) -> Dict[Tuple[str, Any], float]:
    """average normalized slope per (attribute, value), the lift table."""
    overall = np.mean([c["norm_slope"] for c in configs]) if configs else 0.0
    lift: Dict[Tuple[str, Any], float] = {}
    for name in ATTR_NAMES:
        buckets: Dict[Any, List[float]] = {}
        for c in configs:
            buckets.setdefault(c["spec"].get(name), []).append(c["norm_slope"])
        for val, slopes in buckets.items():
            lift[(name, val)] = float(np.mean(slopes) - overall)
    return lift


def detect(market: pd.DataFrame) -> List[TractionSignal]:
    """find competitor configs with rising volume at a stable price."""
    configs = _per_config(market)
    lift = _attribution(configs)

    signals: List[TractionSignal] = []
    for c in configs:
        if (c["price_cov"] <= PRICE_COV_MAX
                and c["norm_slope"] >= NORM_SLOPE_MIN
                and c["r2"] >= R2_MIN):
            # this config's own attributes, ranked by how much lift they carry.
            drivers = sorted(
                ((name, c["spec"].get(name), lift[(name, c["spec"].get(name))])
                 for name in ATTR_NAMES),
                key=lambda t: t[2], reverse=True,
            )
            drivers = [d for d in drivers if d[2] > 0][:3]
            signals.append(TractionSignal(
                brand=c["brand"], sku_id=c["sku_id"], spec=c["spec"],
                config_sig=c["config_sig"], norm_slope=c["norm_slope"],
                price_mean=c["price_mean"], price_cov=c["price_cov"], r2=c["r2"],
                drivers=drivers,
            ))
    signals.sort(key=lambda s: s.norm_slope, reverse=True)
    return signals


def market_drivers(market: pd.DataFrame, top: int = 4) -> List[Tuple[str, Any, float]]:
    """the strongest attribute drivers market-wide, for the dashboard header."""
    configs = _per_config(market)
    lift = _attribution(configs)
    ranked = sorted(lift.items(), key=lambda kv: kv[1], reverse=True)
    return [(name, val, round(v, 4)) for (name, val), v in ranked[:top]]
