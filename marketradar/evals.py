"""recommendation-correctness evals.

the headline is a golden set of scenarios where we know the right behavior and
hard-fail if the solver breaks a constraint:

  - healthy winner  -> we should act (promote), and nothing feasible may sit
    below the margin floor.
  - sku at the floor -> a price cut is off the table, so reprice must be
    infeasible and the pick must not be a reprice.
  - stockout winner  -> we must not push demand we can't supply, so promote and
    reprice are both infeasible and the pick is reallocate / hold.

alongside those we report three quality metrics on the synthetic data: mapping
accuracy, traction recall and the rag aspect f1.

run with:  python -m marketradar.evals
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List

from . import data_gen, resolver, traction
from .constraints import enumerate_actions, split_feasible, validate_feasible
from .rag import RagAspectClassifier
from .router import ModelRouter
from .traction import TractionSignal

_WINNER_SPEC = dict(cpu_tier=7, cpu_family="Core Ultra 7", ram_gb=32,
                    storage_gb=1024, gpu_class="integrated", display_in=14,
                    display_panel="OLED", display_res="QHD",
                    weight_class="light", chassis="aluminum")


@dataclass
class Check:
    name: str
    passed: bool
    detail: str = ""


@dataclass
class EvalReport:
    checks: List[Check] = field(default_factory=list)
    metrics: Dict[str, Any] = field(default_factory=dict)

    @property
    def passed(self) -> bool:
        return all(c.passed for c in self.checks)


def _winner_signal() -> TractionSignal:
    return TractionSignal(
        brand="TestCo", sku_id="TC-1", spec=_WINNER_SPEC, config_sig="sig",
        norm_slope=0.05, price_mean=1400.0, price_cov=0.01, r2=0.9,
        drivers=[("ram_gb", 32, 0.02), ("display_panel", "OLED", 0.02)])


def _sku(carries: bool, cover: float, margin: float,
         floor: float = 0.18) -> Dict[str, Any]:
    lp = 1500.0
    cost = round(lp * (1 - margin), 2)
    demand = 100
    spec = dict(_WINNER_SPEC) if carries else {**_WINNER_SPEC, "ram_gb": 16,
                                               "display_panel": "IPS"}
    return dict(sku_id="SKU", model_name="Test", spec=spec, unit_cost=cost,
                list_price=lp, margin_floor_pct=floor,
                inventory_units=int(demand * cover), weekly_demand=demand)


def _pick(sku, sig):
    feasible, infeasible = split_feasible(enumerate_actions(sku, sig))
    ranked = sorted(feasible, key=lambda a: a.impact, reverse=True)
    return ranked[0], feasible, infeasible


def _scenario_checks() -> List[Check]:
    sig = _winner_signal()
    checks: List[Check] = []

    # 1. healthy winner: we act, and nothing feasible breaks a rule.
    sku = _sku(carries=True, cover=6, margin=0.40)
    chosen, feasible, _ = _pick(sku, sig)
    checks.append(Check("healthy: acts (promote)", chosen.kind == "promote",
                        f"chose {chosen.kind}"))
    checks.append(Check("healthy: no constraint violation",
                        validate_feasible(feasible, sku) == []))

    # 2. sku at the margin floor: no price cut is allowed.
    sku = _sku(carries=True, cover=6, margin=0.18)
    chosen, feasible, infeasible = _pick(sku, sig)
    reprice_infeasible = any(a.kind == "reprice_down" for a in infeasible)
    checks.append(Check("at-floor: reprice is infeasible", reprice_infeasible))
    checks.append(Check("at-floor: pick is not a reprice",
                        chosen.kind != "reprice_down", f"chose {chosen.kind}"))
    checks.append(Check("at-floor: no constraint violation",
                        validate_feasible(feasible, sku) == []))

    # 3. stockout winner: never push demand we can't supply.
    sku = _sku(carries=True, cover=0.8, margin=0.40)
    chosen, feasible, _ = _pick(sku, sig)
    no_push = not any(a.kind in ("promote", "reprice_down") for a in feasible)
    checks.append(Check("stockout: no push-demand action is feasible", no_push))
    checks.append(Check("stockout: pick is reallocate/hold",
                        chosen.kind in ("reallocate", "hold"), f"chose {chosen.kind}"))
    checks.append(Check("stockout: no constraint violation",
                        validate_feasible(feasible, sku) == []))
    return checks


def _metrics() -> Dict[str, Any]:
    ds = data_gen.generate()
    own = ds.own_skus("acme-pc")
    own_arch = {s["sku_id"]: s["archetype"] for s in own}
    cfgs = data_gen.latest_competitor_configs(ds.market)
    res = resolver.resolve(own, cfgs)

    scored = [c for c in cfgs if c["_true_archetype"] in set(own_arch.values())]
    correct = sum(own_arch[res.own_for(c["sku_id"])] == c["_true_archetype"]
                  for c in scored)

    signals = traction.detect(ds.market)
    detected = {s.sku_id for s in signals}
    winners = {c["sku_id"] for c in cfgs
               if c["spec"]["ram_gb"] == 32 and c["spec"]["display_panel"] == "OLED"}

    router = ModelRouter("cheap")
    clf = RagAspectClassifier(router).fit([r for r in ds.reviews if r["split"] == "train"])
    rag_eval = clf.evaluate([r for r in ds.reviews if r["split"] == "test"])

    return {
        "mapping_accuracy": round(correct / len(scored), 3),
        "mapping_n": len(scored),
        "traction_recall": round(len(detected & winners) / len(winners), 3),
        "traction_precision": round(len(detected & winners) / max(len(detected), 1), 3),
        "aspect_f1": rag_eval["aspect_f1"],
        "aspect_backend": rag_eval["backend"],
    }


def run_evals() -> EvalReport:
    return EvalReport(checks=_scenario_checks(), metrics=_metrics())


def main() -> None:
    report = run_evals()
    print("=" * 60)
    print("MarketRadar :: recommendation-correctness evals")
    print("=" * 60)
    for c in report.checks:
        mark = "PASS" if c.passed else "FAIL"
        extra = f"  ({c.detail})" if c.detail else ""
        print(f"  [{mark}] {c.name}{extra}")
    print("-" * 60)
    for k, v in report.metrics.items():
        print(f"  {k:20s} {v}")
    print("-" * 60)
    print("OVERALL:", "PASS" if report.passed else "FAIL")


if __name__ == "__main__":
    main()
