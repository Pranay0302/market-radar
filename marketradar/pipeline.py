"""end-to-end: data in, alerts and recommendations out.

this is the whole story wired together for one tenant:

  generate -> scope to tenant -> resolve configs -> monitor for alerts ->
  detect traction -> mine the "why" -> recommend under constraints ->
  validate + audit.

run with:  python -m marketradar.pipeline --tenant acme-pc --mode cheap
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from typing import Any, Dict, List

from . import constraints, data_gen, monitor, recommend, resolver, sentiment, traction
from .audit import AuditLog, TenantScope
from .generation import RationaleGenerator
from .rag import RagAspectClassifier
from .recommend import Recommendation
from .router import ModelRouter
from .sentiment import SentimentScorer


@dataclass
class ScoredRecommendation:
    recommendation: Recommendation
    signal: traction.TractionSignal
    eval_badge: str                 # "PASS" or "FAIL"
    violations: List[str] = field(default_factory=list)


@dataclass
class PipelineResult:
    tenant_id: str
    mode: str
    alerts: List[monitor.Alert]
    market_drivers: List[Any]
    resolution: resolver.ResolutionResult
    recommendations: List[ScoredRecommendation]
    model_summary: Dict[str, Any]
    audit_path: str


def run_pipeline(tenant_id: str = "acme-pc", mode: str = "cheap",
                 seed: int = 7) -> PipelineResult:
    ds = data_gen.generate(seed)
    scope = TenantScope(ds)
    own = scope.own_portfolio(tenant_id)          # isolation gate
    own_ids = {s["sku_id"] for s in own}

    cfgs = data_gen.latest_competitor_configs(ds.market)
    comp_by_sku = {c["sku_id"]: c for c in cfgs}
    res = resolver.resolve(own, cfgs)

    alerts = monitor.scan(ds.market)
    signals = traction.detect(ds.market)

    router = ModelRouter(mode)
    clf = RagAspectClassifier(router).fit([r for r in ds.reviews if r["split"] == "train"])
    scorer = SentimentScorer()
    generator = RationaleGenerator()
    audit = AuditLog(tenant_id=tenant_id)

    # one recommendation per affected own sku, driven by its strongest signal.
    best_by_own: Dict[str, traction.TractionSignal] = {}
    for sig in sorted(signals, key=lambda s: s.norm_slope, reverse=True):
        own_id = res.own_for(sig.sku_id)
        if own_id in own_ids and own_id not in best_by_own:
            best_by_own[own_id] = sig

    scored: List[ScoredRecommendation] = []
    for own_id, sig in best_by_own.items():
        sku = next(s for s in own if s["sku_id"] == own_id)
        why = sentiment.summarize(sig.config_sig, ds.reviews, clf, scorer, router).phrase
        rec = recommend.recommend(sku, sig, comp_by_sku[sig.sku_id], why,
                                  generator, router)
        violations = constraints.validate_feasible(rec.ranked, sku)
        badge = "PASS" if not violations else "FAIL"
        scope.assert_isolated(tenant_id, [rec.own_sku])   # isolation check
        audit.record(tenant_id, rec, sig, badge, router.summary()["models_used"])
        scored.append(ScoredRecommendation(rec, sig, badge, violations))

    scored.sort(key=lambda sr: sr.signal.norm_slope, reverse=True)
    return PipelineResult(
        tenant_id=tenant_id, mode=mode, alerts=alerts,
        market_drivers=traction.market_drivers(ds.market),
        resolution=res, recommendations=scored,
        model_summary=router.summary(), audit_path=audit.path)


def _print_report(r: PipelineResult) -> None:
    print("=" * 70)
    print(f"MarketRadar :: tenant {r.tenant_id} :: mode {r.mode}")
    print("=" * 70)
    restr = [a for a in r.alerts if a.kind == "restructure"]
    tract = [a for a in r.alerts if a.kind == "traction"]
    print(f"alerts: {len(tract)} traction, {len(restr)} restructure")
    for a in restr:
        print(f"  [restructure] wk{a.week} {a.brand}: {a.detail}")
    drivers = ", ".join(f"{n}={v}" for n, v, _ in r.market_drivers[:3])
    print(f"market drivers: {drivers}")
    print("-" * 70)
    for sr in r.recommendations:
        rec = sr.recommendation
        print(f"[{sr.eval_badge}] {rec.competitor_brand} {rec.competitor_model} "
              f"-> {rec.own_sku} ({rec.own_model})")
        print(f"   action: {rec.chosen.label}")
        print(f"   rationale: {rec.rationale}")
        blocked = [f"{a.kind} ({a.reasons[0]})" for a in rec.infeasible if a.reasons]
        if blocked:
            print(f"   ruled out: {'; '.join(blocked)}")
        print()
    m = r.model_summary
    print("-" * 70)
    print(f"model routing [{m['mode']}]: {m['total_calls']} calls, "
          f"{m['total_cost_units']} cost units, {m['total_latency_ms']}ms")
    print(f"  backends: {m['calls_by_backend']}")
    print(f"  models:   {m['models_used']}")
    print(f"audit log: {r.audit_path}")


def main() -> None:
    ap = argparse.ArgumentParser(description="MarketRadar pipeline")
    ap.add_argument("--tenant", default="acme-pc")
    ap.add_argument("--mode", default="cheap", choices=["cheap", "quality"])
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()
    _print_report(run_pipeline(args.tenant, args.mode, args.seed))


if __name__ == "__main__":
    main()
