"""proactive monitoring: watch the market and raise alerts, don't wait to be asked.

we slide over the weeks and raise two kinds of alert:

  - traction: the first week a config crosses the traction bar (rising sell-out
    at a stable price).
  - restructure: a brand's model names change a lot from one week to the next
    while the underlying config signatures stay put. that's exactly a rename /
    relineup, and we catch it precisely because we track configs by spec, not
    by name.

todo: this replays a static market. in production it runs on a schedule against
the live feed and pushes alerts (slack / email) instead of returning a list.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import pandas as pd

from . import traction
from .schema import ATTR_NAMES, config_signature

NAME_OVERLAP_MAX = 0.3     # under this the names basically changed
SIG_PERSIST_MIN = 0.7      # over this the configs basically stayed
MIN_HISTORY = 6            # weeks of data before we trust a traction slope


@dataclass
class Alert:
    kind: str                  # "traction" or "restructure"
    week: int
    brand: str
    detail: str
    sku_id: Optional[str] = None
    config_sig: Optional[str] = None
    signal: Optional[traction.TractionSignal] = None


def _row_sig(row: pd.Series) -> str:
    return config_signature({n: row[f"spec_{n}"] for n in ATTR_NAMES})


def _traction_alerts(market: pd.DataFrame) -> List[Alert]:
    alerts: List[Alert] = []
    seen = set()
    for w in sorted(market["week"].unique()):
        if w < MIN_HISTORY:
            continue
        window = market[market["week"] <= w]
        for s in traction.detect(window):
            if s.sku_id in seen:
                continue
            seen.add(s.sku_id)
            alerts.append(Alert("traction", int(w), s.brand,
                                f"{s.norm_slope:.1%}/wk at a stable price",
                                sku_id=s.sku_id, config_sig=s.config_sig, signal=s))
    return alerts


def _restructure_alerts(market: pd.DataFrame) -> List[Alert]:
    alerts: List[Alert] = []
    for brand, g in market.groupby("brand"):
        prev_names = prev_sigs = None
        for w in sorted(g["week"].unique()):
            wk = g[g["week"] == w]
            names = set(wk["model_name"])
            sigs = {_row_sig(r) for _, r in wk.iterrows()}
            if prev_names is not None:
                name_overlap = len(names & prev_names) / max(len(names | prev_names), 1)
                sig_persist = len(sigs & prev_sigs) / max(len(sigs | prev_sigs), 1)
                if name_overlap < NAME_OVERLAP_MAX and sig_persist > SIG_PERSIST_MIN:
                    alerts.append(Alert(
                        "restructure", int(w), brand,
                        f"{len(names)} models renamed, configs unchanged, "
                        "so name matching would have lost the thread"))
            prev_names, prev_sigs = names, sigs
    return alerts


def scan(market: pd.DataFrame) -> List[Alert]:
    """all alerts across the window, newest signals first within each week."""
    alerts = _traction_alerts(market) + _restructure_alerts(market)
    return sorted(alerts, key=lambda a: (a.week, a.kind))
