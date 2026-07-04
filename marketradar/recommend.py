"""turn one traction signal + one mapped sku into a recommendation.

we ask the solver for every candidate action, keep the feasible ones, rank them
by impact, and take the top. then the generator writes the rationale from the
already-decided facts. the recommendation carries the infeasible actions too,
because "why we didn't cut price" is often the useful part for priya.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List

from . import constraints
from .constraints import Action
from .generation import RationaleGenerator
from .router import ModelRouter


@dataclass
class Recommendation:
    competitor_sku: str
    competitor_brand: str
    competitor_model: str
    own_sku: str
    own_model: str
    drivers: List[str]
    why: str
    chosen: Action
    ranked: List[Action]
    infeasible: List[Action] = field(default_factory=list)
    rationale: str = ""


def _numbers(action: Action, sku: Dict[str, Any]) -> str:
    floor = sku["margin_floor_pct"]
    if action.kind == "reprice_down":
        return (f"new price ${action.params['new_price']:,.0f}, margin "
                f"{action.params['new_margin']:.0%} vs {floor:.0%} floor")
    if action.kind == "promote":
        return (f"margin {action.projected_margin:.0%} (floor {floor:.0%}), "
                f"{action.params['weeks_cover']}wk of cover")
    if action.kind == "reallocate":
        return (f"{action.params['weeks_cover']}wk cover vs "
                f"{sku['weekly_demand']}/wk demand")
    return f"margin {action.projected_margin:.0%}, no feasible move clears the floor and stock"


def _constraints_txt(action: Action, sku: Dict[str, Any]) -> str:
    floor = sku["margin_floor_pct"]
    if action.kind == "reprice_down":
        return f"holds margin above the {floor:.0%} floor"
    if action.kind == "promote":
        return f"above the {floor:.0%} floor and enough stock to push"
    if action.kind == "reallocate":
        return "avoids pushing demand we can't supply"
    return "no action taken breaks a constraint"


def recommend(own_sku: Dict[str, Any], signal, competitor: Dict[str, Any],
              why: str, generator: RationaleGenerator,
              router: ModelRouter) -> Recommendation:
    actions = constraints.enumerate_actions(own_sku, signal)
    feasible, infeasible = constraints.split_feasible(actions)
    ranked = sorted(feasible, key=lambda a: a.impact, reverse=True)
    chosen = ranked[0]
    drivers = constraints._driver_labels(signal)

    ctx = {
        "brand": competitor.get("brand", "A competitor"),
        "competitor": competitor.get("model_name", competitor.get("sku_id")),
        "own_sku": own_sku["sku_id"],
        "own_model": own_sku["model_name"],
        "drivers": drivers,
        "why": why,
        "action": chosen.label,
        "numbers": _numbers(chosen, own_sku),
        "constraints": _constraints_txt(chosen, own_sku),
    }
    rationale = generator.generate(ctx, router)

    return Recommendation(
        competitor_sku=competitor.get("sku_id"),
        competitor_brand=competitor.get("brand"),
        competitor_model=competitor.get("model_name"),
        own_sku=own_sku["sku_id"], own_model=own_sku["model_name"],
        drivers=drivers, why=why, chosen=chosen, ranked=ranked,
        infeasible=infeasible, rationale=rationale)
