"""report: a shareable PDF is produced for the seed findings and the edge cases."""

import datetime
import types

from marketradar import data_gen
from marketradar.pipeline import run_pipeline
from marketradar.report import build_report_pdf


def _pdf(result, ds, tenant, prepared_for=""):
    out = build_report_pdf(result, ds, tenant, prepared_for,
                           generated_on=datetime.date(2026, 7, 5))
    assert isinstance(out, (bytes, bytearray))
    assert out[:4] == b"%PDF"          # a real pdf, not an empty/garbage blob
    assert len(out) > 1000             # has actual content, not just a header
    return out


def test_report_builds_for_seed_findings():
    ds = data_gen.generate()
    result = run_pipeline("acme-pc")
    _pdf(result, ds, "acme-pc", prepared_for="Northwind Retail")


def test_report_builds_without_a_client_name():
    ds = data_gen.generate()
    result = run_pipeline("acme-pc")
    _pdf(result, ds, "acme-pc", prepared_for="")


def test_report_handles_zero_recommendations():
    # a tenant snapshot with no signals mapping in: ds is never touched.
    empty = types.SimpleNamespace(
        tenant_id="acme-pc", mode="cheap",
        alerts=[types.SimpleNamespace(kind="traction")],
        market_drivers=[("ram_gb", 32, 0.4)],
        recommendations=[],
    )
    _pdf(empty, None, "acme-pc", prepared_for="Acme")