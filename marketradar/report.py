"""render a shareable client report as a PDF.

this is pure presentation: it takes a finished PipelineResult and lays out the
findings a product line manager would forward to a client. no decisions are made
here and nothing streamlit-specific is imported, so it stays unit-testable and
runs anywhere the pipeline does.

uses fpdf2 (pure python, no native deps) so it works on streamlit cloud. the core
pdf fonts are latin-1 only, so every string is sanitized before it is drawn.
"""

from __future__ import annotations

import datetime as _dt
from typing import Any, Dict, Optional

from fpdf import FPDF
from fpdf.enums import XPos, YPos

from .constraints import margin_pct, weeks_cover

# palette mirrors the dashboard so the report reads as the same product.
_INK = (23, 32, 28)
_MUTED = (107, 114, 128)
_ACCENT = (15, 118, 110)
_GOOD = (21, 128, 61)
_BAD = (185, 28, 28)
_LINE = (224, 224, 222)

# characters the dashboard uses that the core pdf fonts can't encode.
_SUBS = {
    "→": "->", "←": "<-", "—": " - ", "–": "-",
    "‘": "'", "’": "'", "“": '"', "”": '"',
    "•": "-", "…": "...", "×": "x", "✓": "", "✕": "x",
    " ": " ",
}


def _ascii(s: Any) -> str:
    """make any value safe for the latin-1 core fonts."""
    out = str(s)
    for bad, good in _SUBS.items():
        out = out.replace(bad, good)
    return out.encode("latin-1", "replace").decode("latin-1")


class _Report(FPDF):
    def footer(self) -> None:
        self.set_y(-14)
        self.set_font("Helvetica", size=7)
        self.set_text_color(*_MUTED)
        self.cell(0, 5, _ascii(
            "Figures computed by MarketRadar's deterministic constraint solver. "
            "Confidential."), align="L")
        self.cell(0, 5, f"Page {self.page_no()}", align="R")


def _rule(pdf: _Report, gap_before: float = 2.0, gap_after: float = 3.0) -> None:
    pdf.ln(gap_before)
    pdf.set_draw_color(*_LINE)
    pdf.set_line_width(0.2)
    y = pdf.get_y()
    pdf.line(pdf.l_margin, y, pdf.w - pdf.r_margin, y)
    pdf.ln(gap_after)


def _label(pdf: _Report, text: str) -> None:
    pdf.set_font("Helvetica", "B", 7.5)
    pdf.set_text_color(*_MUTED)
    pdf.cell(0, 5, _ascii(text.upper()), new_x=XPos.LMARGIN, new_y=YPos.NEXT)


def _body(pdf: _Report, text: str, size: float = 9.5,
          color: tuple = _INK, style: str = "") -> None:
    pdf.set_font("Helvetica", style, size)
    pdf.set_text_color(*color)
    pdf.multi_cell(0, 5, _ascii(text), new_x=XPos.LMARGIN, new_y=YPos.NEXT)


def _header(pdf: _Report, tenant: str, gen: _dt.date, prepared_for: str) -> None:
    pdf.set_font("Helvetica", "B", 20)
    pdf.set_text_color(*_ACCENT)
    pdf.cell(0, 9, "MarketRadar", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.set_font("Helvetica", "", 10.5)
    pdf.set_text_color(*_MUTED)
    pdf.cell(0, 6, "Competitor action report", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.ln(1)

    client = prepared_for.strip() if prepared_for else "-"
    pdf.set_font("Helvetica", "", 9)
    pdf.set_text_color(*_INK)
    meta = (f"Prepared for: {client}          Tenant: {tenant}          "
            f"Date: {gen.strftime('%d %b %Y')}")
    pdf.cell(0, 5, _ascii(meta), new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    _rule(pdf, gap_before=2.5, gap_after=4)


def _section(pdf: _Report, title: str) -> None:
    pdf.ln(2)
    pdf.set_font("Helvetica", "B", 12)
    pdf.set_text_color(*_INK)
    pdf.cell(0, 7, _ascii(title), new_x=XPos.LMARGIN, new_y=YPos.NEXT)


def _badge(pdf: _Report, badge: str) -> None:
    color = _GOOD if str(badge).upper() == "PASS" else _BAD
    pdf.set_font("Helvetica", "B", 8)
    pdf.set_text_color(*color)
    pdf.cell(0, 5, _ascii(f"[ {badge} ]"), new_x=XPos.LMARGIN, new_y=YPos.NEXT)


def _recommendation(pdf: _Report, idx: int, sr: Any,
                    own_by_id: Dict[str, Dict[str, Any]]) -> None:
    rec = sr.recommendation
    _rule(pdf, gap_before=3, gap_after=3)

    # route: competitor -> own sku
    comp = " ".join(x for x in [rec.competitor_brand, rec.competitor_model] if x)
    pdf.set_font("Helvetica", "B", 10.5)
    pdf.set_text_color(*_INK)
    pdf.multi_cell(0, 6, _ascii(f"{idx}.  {comp}  ->  {rec.own_sku} ({rec.own_model})"),
                   new_x=XPos.LMARGIN, new_y=YPos.NEXT)

    # chosen action + eval badge
    pdf.set_font("Helvetica", "B", 9.5)
    pdf.set_text_color(*_ACCENT)
    pdf.cell(pdf.epw * 0.72, 5, _ascii(f"Action: {rec.chosen.label}"))
    _badge(pdf, sr.eval_badge)

    # rationale
    pdf.ln(1)
    _body(pdf, rec.rationale or "(no rationale generated)", size=9.5, color=(51, 65, 59))

    # your sku right now
    own = own_by_id.get(rec.own_sku)
    if own:
        margin = margin_pct(own["list_price"], own["unit_cost"])
        floor = own["margin_floor_pct"]
        cover = weeks_cover(own)
        pdf.ln(1)
        _label(pdf, "Your SKU right now")
        nums = (f"Unit cost ${own['unit_cost']:,.0f}   |   List ${own['list_price']:,.0f}"
                f"   |   Margin {margin:.0%} (floor {floor:.0%})"
                f"   |   Inventory {own['inventory_units']:,}   |   {cover:.1f}wk cover")
        _body(pdf, nums, size=9)

    # drivers
    if rec.drivers:
        pdf.ln(0.5)
        _label(pdf, "What's driving it")
        _body(pdf, ", ".join(rec.drivers), size=9)

    # options the solver considered
    pdf.ln(0.5)
    _label(pdf, "Options the solver considered")
    for a in rec.ranked:
        tag = "  (recommended)" if a is rec.chosen else ""
        _body(pdf, f"+ {a.label}{tag}", size=9, color=_GOOD)
    for a in rec.infeasible:
        reason = a.reasons[0] if a.reasons else "not applicable"
        _body(pdf, f"x {a.label} - {reason}", size=9, color=_MUTED)


def build_report_pdf(result: Any, ds: Any, tenant: str,
                     prepared_for: str = "",
                     generated_on: Optional[_dt.date] = None) -> bytes:
    """render a full findings snapshot for one tenant as PDF bytes.

    result: a marketradar.pipeline.PipelineResult
    ds:     the dataset (for the tenant's own-SKU numbers)
    """
    gen = generated_on or _dt.date.today()

    pdf = _Report(format="A4", unit="mm")
    pdf.set_auto_page_break(auto=True, margin=18)
    pdf.set_margins(left=18, top=16, right=18)
    pdf.set_title(f"MarketRadar report - {tenant}")
    pdf.add_page()

    _header(pdf, tenant, gen, prepared_for)

    # market pulse
    _section(pdf, "Market pulse")
    drivers = getattr(result, "market_drivers", [])[:4]
    driver_txt = ", ".join(f"{n} = {v}" for n, v, *_ in drivers) or "no dominant drivers"
    _body(pdf, driver_txt)
    tract = sum(1 for a in result.alerts if a.kind == "traction")
    restr = sum(1 for a in result.alerts if a.kind == "restructure")
    _body(pdf, f"Alerts: {tract} traction, {restr} restructure.", color=_MUTED)

    # recommendations
    recs = result.recommendations
    _section(pdf, f"Recommendations ({len(recs)})")
    if not recs:
        _body(pdf, "No competitor traction maps into this tenant's portfolio right now.",
              color=_MUTED)
    else:
        own_by_id = {s["sku_id"]: s for s in ds.own_skus(tenant)}
        for i, sr in enumerate(recs, 1):
            _recommendation(pdf, i, sr, own_by_id)

    return bytes(pdf.output())