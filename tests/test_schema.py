"""schema: entity normalization and the spec-distance ordering."""

from marketradar import data_gen
from marketradar.schema import (ATTR_NAMES, compute_stats, config_signature,
                                normalize_listing, weighted_distance)


def test_normalization_extracts_clean_attrs():
    spec = normalize_listing({
        "cpu_raw": "Intel Core Ultra 7 155H",
        "ram_raw": "32 GB",
        "display_panel": "oled",
    })
    assert spec["cpu_tier"] == 7
    assert spec["cpu_family"] == "Core Ultra 7"
    assert spec["ram_gb"] == 32
    assert spec["display_panel"] == "OLED"


def test_signature_treats_14_and_14point0_the_same():
    base = {n: 1 for n in ATTR_NAMES}
    assert config_signature({**base, "display_in": 14}) == \
        config_signature({**base, "display_in": 14.0})


def test_weighted_distance_orders_configs():
    a = data_gen.generate().archetypes
    stats = compute_stats(list(a.values()))
    near = weighted_distance(a["A1"], a["A10"], stats)   # both 32GB OLED
    far = weighted_distance(a["A1"], a["A3"], stats)     # very different
    assert near < far