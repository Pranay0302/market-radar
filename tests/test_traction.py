"""traction: catches the planted 32GB+OLED signal, leaves flat configs alone."""

from marketradar import data_gen, traction


def test_detects_only_the_planted_winner_signal():
    ds = data_gen.generate()
    sigs = traction.detect(ds.market)
    assert sigs, "no traction detected at all"
    for s in sigs:
        assert s.spec["ram_gb"] == 32 and s.spec["display_panel"] == "OLED"
        assert s.price_cov <= traction.PRICE_COV_MAX


def test_attribution_surfaces_ram_and_panel():
    ds = data_gen.generate()
    drivers = {(a, v) for s in traction.detect(ds.market) for a, v, _ in s.drivers}
    assert ("ram_gb", 32) in drivers
    assert ("display_panel", "OLED") in drivers


def test_flat_configs_are_not_flagged():
    ds = data_gen.generate()
    flagged = {s.sku_id for s in traction.detect(ds.market)}
    per = traction._per_config(ds.market)
    non_winners = [c for c in per
                   if not (c["spec"]["ram_gb"] == 32
                           and c["spec"]["display_panel"] == "OLED")]
    assert non_winners
    assert all(c["sku_id"] not in flagged for c in non_winners)