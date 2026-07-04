"""audit log + tenant isolation.

two enterprise concerns live here:

  - isolation: a tenant only ever sees its own portfolio. tenantscope refuses a
    cross-tenant read, so one customer's cost / margin data can't leak into
    another's session.
  - auditability: every recommendation is written to an append-only log with the
    inputs, the models that ran, the constraints applied and the rationale, so
    any recommendation can be explained after the fact.

the log holds no secrets and no pii (the data is synthetic anyway), which is the
posture we'd keep in production too.

todo: swap the jsonl file for an append-only store per tenant (s3 + object lock,
or a wal), and sign each record so the trail is tamper-evident.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from typing import Any, Dict, List

DEFAULT_DIR = "runs"


class TenantIsolationError(Exception):
    pass


class TenantScope:
    """gate every portfolio read through here so tenants stay separated."""

    def __init__(self, dataset):
        self._ds = dataset

    def own_portfolio(self, tenant_id: str) -> List[Dict[str, Any]]:
        if tenant_id not in self._ds.own_portfolio:
            raise TenantIsolationError(f"unknown tenant {tenant_id!r}")
        return self._ds.own_skus(tenant_id)

    def assert_isolated(self, tenant_id: str, sku_ids: List[str]) -> None:
        """confirm every sku we touched actually belongs to this tenant."""
        mine = {s["sku_id"] for s in self._ds.own_skus(tenant_id)}
        foreign = [s for s in sku_ids if s not in mine]
        if foreign:
            raise TenantIsolationError(
                f"tenant {tenant_id!r} touched foreign skus: {foreign}")


def _action_dict(a) -> Dict[str, Any]:
    return {"kind": a.kind, "label": a.label, "feasible": a.feasible,
            "projected_margin": round(a.projected_margin, 4),
            "reasons": a.reasons}


class AuditLog:
    def __init__(self, path: str = None, tenant_id: str = "default"):
        os.makedirs(DEFAULT_DIR, exist_ok=True)
        self.path = path or os.path.join(DEFAULT_DIR, f"audit-{tenant_id}.jsonl")

    def record(self, tenant_id: str, rec, signal, eval_badge: str,
               models_used: List[str]) -> Dict[str, Any]:
        entry = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "tenant_id": tenant_id,
            "competitor_sku": rec.competitor_sku,
            "competitor_model": rec.competitor_model,
            "traction": {"norm_slope": round(signal.norm_slope, 4),
                         "price_cov": round(signal.price_cov, 4)},
            "own_sku": rec.own_sku,
            "drivers": rec.drivers,
            "why": rec.why,
            "chosen_action": _action_dict(rec.chosen),
            "constraints_applied": [_action_dict(a) for a in rec.ranked + rec.infeasible],
            "rationale": rec.rationale,
            "models_used": models_used,
            "eval_badge": eval_badge,
        }
        with open(self.path, "a") as f:
            f.write(json.dumps(entry) + "\n")
        return entry

    def records(self) -> List[Dict[str, Any]]:
        if not os.path.exists(self.path):
            return []
        with open(self.path) as f:
            return [json.loads(line) for line in f if line.strip()]
