"""the deterministic solver: what can we actually do about this signal?

this is where cost, margin and inventory bite. we enumerate candidate actions
and mark each feasible or not with a plain-english reason. nothing here calls a
model; correctness is guaranteed by arithmetic, not by an llm. two hard rules:

  - never recommend a price cut that drops margin below the sku's floor
  - never recommend pushing demand for a sku we can't supply (stockout risk)

todo: reprice is a flat 5% cut for now. a real version should solve for the
price that best defends share while holding margin above the floor.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List

MIN_COVER_WEEKS = 3.0        # under this much stock cover we treat it as supply-constrained
REPRICE_CUT = 0.05           # defensive price cut we consider
KEY_DRIVER_ATTRS = ("ram_gb", "display_panel")   # what "the winning config" means


@dataclass
class Action:
    kind: str
    label: str
    params: Dict[str, Any]
    projected_margin: float
    feasible: bool
    reasons: List[str] = field(default_factory=list)
    impact: float = 0.0


def weeks_cover(sku: Dict[str, Any]) -> float:
    d = sku.get("weekly_demand", 0)
    return sku["inventory_units"] / d if d else 999.0


def margin_pct(price: float, cost: float) -> float:
    return (price - cost) / price if price else 0.0


def _carries_drivers(own_spec: Dict[str, Any], signal) -> bool:
    """does our sku actually have the winning attributes (32gb + oled)?"""
    return all(own_spec.get(a) == signal.spec.get(a) for a in KEY_DRIVER_ATTRS)


def _friendly(attr: str, val: Any) -> str:
    if attr == "ram_gb":
        return f"{val}GB RAM"
    if attr == "display_panel":
        return f"{val} display"
    if attr == "gpu_class":
        return "integrated graphics" if val == "integrated" \
            else str(val).upper().replace("RTX", "RTX ").strip() + " GPU"
    if attr == "display_res":
        return f"{val} screen"
    if attr == "cpu_tier":
        return f"tier-{val} CPU"
    if attr == "weight_class":
        return f"{val} weight"
    return f"{attr}={val}"


def _driver_labels(signal) -> List[str]:
    return [_friendly(attr, val) for attr, val, _ in signal.drivers]


def enumerate_actions(sku: Dict[str, Any], signal) -> List[Action]:
    """all candidate responses for this sku, each flagged feasible or not."""
    cover = weeks_cover(sku)
    floor = sku["margin_floor_pct"]
    cost = sku["unit_cost"]
    lp = sku["list_price"]
    strength = signal.norm_slope * 100      # weekly % growth, as an impact proxy
    carries = _carries_drivers(sku["spec"], signal)
    actions: List[Action] = []

    # 1. defensive price cut
    target = round(lp * (1 - REPRICE_CUT), 2)
    new_margin = margin_pct(target, cost)
    reasons: List[str] = []
    feasible = True
    if new_margin < floor:
        feasible = False
        reasons.append(f"cut drops margin to {new_margin:.0%}, under the {floor:.0%} floor")
    if cover < MIN_COVER_WEEKS:
        feasible = False
        reasons.append(f"only {cover:.1f}wk of cover, a cut would deepen the stockout")
    actions.append(Action(
        "reprice_down", f"Reprice {sku['sku_id']} to ${target:,.0f} (-5%)",
        {"new_price": target, "new_margin": new_margin},
        new_margin, feasible, reasons, impact=(0.6 * strength if feasible else 0.0)))

    # 2. promote the matching config we already stock
    reasons = []
    feasible = True
    if not carries:
        feasible = False
        reasons.append("this SKU lacks the winning attributes (needs 32GB + OLED)")
    if cover < MIN_COVER_WEEKS:
        feasible = False
        reasons.append(f"only {cover:.1f}wk of cover, can't push demand we can't supply")
    actions.append(Action(
        "promote", f"Promote {sku['sku_id']} on {', '.join(_driver_labels(signal))}",
        {"weeks_cover": round(cover, 1)},
        margin_pct(lp, cost), feasible, reasons,
        impact=(strength + margin_pct(lp, cost) * 10 if feasible else 0.0)))

    # 3. protect supply when it's a winner we simply can't feed
    if carries and cover < MIN_COVER_WEEKS:
        actions.append(Action(
            "reallocate",
            f"Prioritize supply for {sku['sku_id']} ({cover:.1f}wk cover vs rising demand)",
            {"weeks_cover": round(cover, 1)}, margin_pct(lp, cost), True, [],
            impact=0.8 * strength))

    # 4. do nothing (baseline, always available)
    actions.append(Action(
        "hold", f"Hold {sku['sku_id']} and keep monitoring", {},
        margin_pct(lp, cost), True, [], impact=0.1))
    return actions


def split_feasible(actions: List[Action]):
    return ([a for a in actions if a.feasible],
            [a for a in actions if not a.feasible])
