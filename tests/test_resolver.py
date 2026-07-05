"""resolver: mapping accuracy and the key property, surviving a rename."""

from marketradar import data_gen, resolver
from marketradar.embedding import Embedder
from marketradar.schema import (ATTR_NAMES, compute_stats, normalize_listing,
                                spec_text)


def _config_at_week(market, sku_id, week):
    r = market[(market.sku_id == sku_id) & (market.week == week)].iloc[0]
    raw = {"cpu_raw": r["spec_cpu_raw"], "ram_raw": r["spec_ram_raw"]}
    for n in ATTR_NAMES:
        if n not in ("cpu_tier", "cpu_family", "ram_gb"):
            raw[n] = r[f"spec_{n}"]
    return normalize_listing(raw)


def test_mapping_accuracy_is_perfect_on_known_pairs():
    ds = data_gen.generate()
    own = ds.own_skus("acme-pc")
    own_arch = {s["sku_id"]: s["archetype"] for s in own}
    cfgs = data_gen.latest_competitor_configs(ds.market)
    res = resolver.resolve(own, cfgs)
    scored = [c for c in cfgs if c["_true_archetype"] in set(own_arch.values())]
    correct = sum(own_arch[res.own_for(c["sku_id"])] == c["_true_archetype"]
                  for c in scored)
    assert correct == len(scored)


def test_resolution_survives_rename():
    ds = data_gen.generate()
    own = ds.own_skus("acme-pc")
    cfgs = data_gen.latest_competitor_configs(ds.market)
    stats = compute_stats([s["spec"] for s in own] + [c["spec"] for c in cfgs])
    emb = Embedder(prefer="tfidf").fit([spec_text(s["spec"]) for s in own])

    nimbus = ds.market[ds.market.brand == "Nimbus"].sku_id.unique()
    for sku in nimbus:
        early = _config_at_week(ds.market, sku, 3)    # probook era
        late = _config_at_week(ds.market, sku, 12)    # eliteline era
        a = resolver.resolve_one(early, own, stats, emb)["own_sku"]
        b = resolver.resolve_one(late, own, stats, emb)["own_sku"]
        assert a == b, f"{sku} mapping changed across the rename"

    # sanity: the names really did change, so the test above means something.
    n3 = set(ds.market[(ds.market.brand == "Nimbus") & (ds.market.week == 3)].model_name)
    n12 = set(ds.market[(ds.market.brand == "Nimbus") & (ds.market.week == 12)].model_name)
    assert n3.isdisjoint(n12)