"""constraints: the two hard rules the solver must never break."""

from marketradar.constraints import (enumerate_actions, split_feasible,
                                     validate_feasible)
from marketradar.evals import _pick, _sku, _winner_signal


def test_never_reprices_below_the_margin_floor():
    sig = _winner_signal()
    sku = _sku(carries=True, cover=6, margin=0.18)   # already at the floor
    feasible, infeasible = split_feasible(enumerate_actions(sku, sig))
    assert any(a.kind == "reprice_down" for a in infeasible)
    assert validate_feasible(feasible, sku) == []


def test_never_pushes_demand_on_a_stockout():
    sig = _winner_signal()
    sku = _sku(carries=True, cover=0.8, margin=0.40)  # under one week of cover
    feasible, _ = split_feasible(enumerate_actions(sku, sig))
    assert not any(a.kind in ("promote", "reprice_down") for a in feasible)
    assert validate_feasible(feasible, sku) == []


def test_healthy_winner_gets_a_real_action():
    sig = _winner_signal()
    sku = _sku(carries=True, cover=6, margin=0.40)
    chosen, feasible, _ = _pick(sku, sig)
    assert chosen.kind == "promote"
    assert validate_feasible(feasible, sku) == []