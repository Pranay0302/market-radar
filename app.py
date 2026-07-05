"""MarketRadar dashboard: a calm, minimal, but demo-ready read on the market.

Run with:  streamlit run app.py

Layout is master-detail. Pick a signal on the left, and the right side drills in:
the traction chart, the spec match, your own SKU's numbers, the mined "why", the
recommendation, and every option the solver considered. Tabs below cover the
resolution map, model routing, and the audit trail.
"""

from __future__ import annotations

import html

import pandas as pd
import streamlit as st

from marketradar import data_gen
from marketradar.audit import AuditLog
from marketradar.auth import DEFAULT_PASS, DEFAULT_USER, check_credentials
from marketradar.constraints import margin_pct, weeks_cover
from marketradar.pipeline import run_pipeline
from marketradar.report import build_report_pdf

st.set_page_config(page_title="MarketRadar", page_icon="•", layout="wide")

CSS = """
<style>
:root {
  --ink:#17201c; --muted:#6b7280; --line:#ececec; --accent:#0f766e;
  --accent-soft:#e7f3f0; --good:#15803d; --good-soft:#e9f6ec; --chip:#f4f4f3;
}
.block-container { max-width: 1080px; padding-top: 2rem; }
html, body, [class*="css"] { font-family:-apple-system,"Inter",system-ui,sans-serif; }
.mr-title { font-size:1.7rem; font-weight:680; color:var(--ink); letter-spacing:-0.02em; }
.mr-sub { color:var(--muted); font-size:0.95rem; margin:0.1rem 0 1.3rem; }
.mr-label { text-transform:uppercase; letter-spacing:0.08em; font-size:0.7rem;
  color:var(--muted); font-weight:600; margin:1.3rem 0 0.5rem; }
.tiles { display:flex; gap:14px; }
.tile { flex:1; background:#fff; border:1px solid var(--line); border-radius:12px; padding:14px 16px; }
.tile .num { font-size:1.5rem; font-weight:680; color:var(--ink); }
.tile .lab { color:var(--muted); font-size:0.78rem; }
.chip { display:inline-block; background:var(--chip); color:#374151; border-radius:999px;
  padding:3px 11px; font-size:0.8rem; margin:0 6px 6px 0; }
.callout { border:1px solid var(--line); border-left:3px solid var(--accent); background:#fbfbfa;
  border-radius:10px; padding:12px 16px; color:var(--ink); font-size:0.9rem; margin-bottom:0.5rem; }
.callout .rn { color:var(--muted); }
.route { color:var(--ink); font-size:1.02rem; margin-bottom:2px; }
.route .brand { color:var(--muted); }
.route .arrow { color:var(--accent); margin:0 8px; }
.route .own { font-weight:680; }
.route .omodel { color:var(--muted); margin-left:6px; font-size:0.85rem; }
.badge { border-radius:999px; padding:3px 10px; font-size:0.72rem; font-weight:600;
  text-transform:uppercase; letter-spacing:0.04em; }
.badge.accent { background:var(--accent-soft); color:var(--accent); }
.badge.good { background:var(--good-soft); color:var(--good); margin-left:6px; }
.rationale { color:#33413b; font-size:0.94rem; line-height:1.55; background:#fbfbfa;
  border:1px solid var(--line); border-radius:12px; padding:14px 16px; }
.bar-row { display:flex; align-items:center; gap:10px; margin:5px 0; font-size:0.82rem; }
.bar-lab { width:120px; color:var(--muted); }
.bar-track { flex:1; background:#f0f0ef; border-radius:6px; height:8px; overflow:hidden; }
.bar-fill { height:100%; border-radius:6px; }
.bar-val { width:46px; text-align:right; color:var(--ink); }
.act { display:flex; gap:10px; padding:8px 0; border-bottom:1px solid #f2f2f1; font-size:0.9rem; }
.act .mark { width:16px; }
.act .lab { color:var(--ink); }
.act.chosen .lab { font-weight:680; color:var(--accent); }
.act .why { color:var(--muted); font-size:0.82rem; }
.foot { color:var(--muted); font-size:0.78rem; }

/* centered, informative "running the pipeline" state */
@keyframes mr-spin { to { transform: rotate(360deg); } }
@keyframes mr-fade { from { opacity:0; transform: translateY(4px); } to { opacity:1; transform:none; } }
.mr-loading { display:flex; justify-content:center; padding:2.2rem 0 2.6rem; }
.mr-load-card { width:100%; max-width:600px; border:1px solid var(--line);
  border-radius:16px; padding:26px 30px 28px; background:#fbfbfa; text-align:center;
  animation: mr-fade 0.25s ease both; }
.mr-spinner { width:34px; height:34px; margin:0 auto 15px; border-radius:50%;
  border:3px solid var(--accent-soft); border-top-color:var(--accent);
  animation: mr-spin 0.85s linear infinite; }
.mr-load-title { font-size:1.15rem; font-weight:680; color:var(--ink); letter-spacing:-0.01em; }
.mr-load-sub { color:var(--muted); font-size:0.88rem; line-height:1.5; margin:5px auto 20px; max-width:460px; }
.mr-facts { text-align:left; display:flex; flex-direction:column; gap:11px;
  border-top:1px solid var(--line); padding-top:18px; }
.mr-fact { display:flex; gap:12px; align-items:flex-start; }
.mr-fact .n { flex:none; width:24px; height:24px; border-radius:7px; background:var(--accent-soft);
  color:var(--accent); font-size:0.7rem; font-weight:700; display:flex; align-items:center;
  justify-content:center; margin-top:1px; }
.mr-fact .ft { display:flex; flex-direction:column; gap:1px; }
.mr-fact .ft b { color:var(--ink); font-size:0.87rem; font-weight:650; }
.mr-fact .ft span { color:var(--muted); font-size:0.82rem; line-height:1.45; }
</style>
"""


# Login screen: a minimal, neutral sign-in card centered in the viewport on both
# axes. The card IS the Streamlit form (its stable data-testid is what we style),
# because widgets can't live inside injected HTML. Kept monochrome — no accent
# color — so it reads calm and clean.
LOGIN_CSS = """
<style>
[data-testid="stHeader"] { display: none; }
[data-testid="stAppViewContainer"], .stApp {
  background: radial-gradient(1200px 720px at 50% -10%, #ffffff 0%, #f4f5f6 58%, #edeff0 100%);
}
/* center the sign-in card on both axes. stMain is the flex parent whose direct
   child is the single block-container column, so we center that column. */
.stApp [data-testid="stMain"] {
  display: flex; align-items: center; justify-content: center; min-height: 100vh;
}
.stApp [data-testid="stMainBlockContainer"] {
  width: 372px; max-width: calc(100vw - 2rem); padding: 0;
}
[data-testid="stForm"] {
  width: 100%; padding: 34px 32px 30px;
  background: #ffffff; border: 1px solid #ececec; border-radius: 16px;
  box-shadow: 0 12px 44px rgba(15,20,25,0.08);
}
[data-testid="stForm"] input { border-radius: 8px !important; }
/* drop the "Press Enter to submit form" hint under the inputs */
[data-testid="stForm"] [data-testid="InputInstructions"] { display: none !important; }
/* keep the focus ring monochrome (the app theme accent is teal) */
[data-testid="stForm"] div[data-baseweb="input"]:focus-within {
  border-color: #17201c !important; box-shadow: none !important;
}
.mr-login-brand { font-size:1.5rem; font-weight:700; letter-spacing:-0.02em;
  color:#17201c; text-align:center; }
.mr-login-tag { color:#6b7280; font-size:0.85rem; text-align:center; margin:3px 0 20px; }
[data-testid="stForm"] label { font-size:0.78rem !important; color:#4b5563 !important;
  font-weight:600 !important; }
[data-testid="stFormSubmitButton"] button {
  background:#17201c; color:#fff; border:0; border-radius:10px;
  padding:0.5rem 0; font-weight:600; }
[data-testid="stFormSubmitButton"] button:hover { background:#2a3630; color:#fff; }
</style>
"""


def _expected_credentials():
    """let a deployment override the hardcoded pair via st.secrets [auth]."""
    try:
        auth = st.secrets.get("auth", {})
        return (auth.get("user", DEFAULT_USER), auth.get("password", DEFAULT_PASS))
    except Exception:
        return (DEFAULT_USER, DEFAULT_PASS)


def _render_login() -> None:
    st.markdown(LOGIN_CSS, unsafe_allow_html=True)
    with st.form("login_form"):
        st.markdown("<div class='mr-login-brand'>MarketRadar</div>"
                    "<div class='mr-login-tag'>OEM commercial action layer</div>",
                    unsafe_allow_html=True)
        user = st.text_input("Username", placeholder="Username")
        pw = st.text_input("Password", type="password", placeholder="Password")
        submitted = st.form_submit_button("Sign in", use_container_width=True)
    if submitted:
        exp_user, exp_pass = _expected_credentials()
        if check_credentials(user, pw, exp_user, exp_pass):
            st.session_state["authenticated"] = True
            st.rerun()
        else:
            st.error("Incorrect username or password.")


# the pipeline stages, shown as steady "facts" while it runs so the wait
# doubles as a tour of what the system is actually doing.
_PIPELINE_FACTS = [
    ("Resolve", "Each competitor configuration is matched to your nearest SKU by "
     "structured specs, knowledge-graph overlap and embeddings — names are shown, "
     "never matched on."),
    ("Monitor", "Weeks of sell-out are scanned for line restructures and demand traction."),
    ("Traction", "A model is flagged when it gains share while holding a stable price — "
     "sell-out slope over price variation."),
    ("Mine the why", "A retrieval classifier pulls labeled review snippets and distilbert "
     "scores sentiment for each aspect buyers mention."),
    ("Recommend", "A constraint solver ranks reprice / promote / reallocate under your "
     "margin floor and weeks-of-cover."),
    ("Explain & audit", "flan-t5 phrases the call in plain English and every decision is "
     "appended to a tamper-evident log."),
]


def _loading_html(mode: str) -> str:
    if mode == "quality":
        sub = ("Quality mode is loading MiniLM, distilbert and flan-t5 to run locally. "
               "The first run warms the models — after that every rerun is instant.")
    else:
        sub = ("Cheap mode runs entirely on deterministic sklearn, lexicon and template "
               "backends — no model downloads, just the pipeline end to end.")
    facts = "".join(
        f"<div class='mr-fact'><span class='n'>{i:02d}</span>"
        f"<span class='ft'><b>{html.escape(t)}</b><span>{html.escape(d)}</span></span></div>"
        for i, (t, d) in enumerate(_PIPELINE_FACTS, 1))
    return (f"<div class='mr-loading'><div class='mr-load-card'>"
            f"<div class='mr-spinner'></div>"
            f"<div class='mr-load-title'>Running the pipeline</div>"
            f"<div class='mr-load-sub'>{html.escape(sub)}</div>"
            f"<div class='mr-facts'>{facts}</div></div></div>")


@st.cache_data(show_spinner=False)
def _load(tenant: str, mode: str):
    return run_pipeline(tenant, mode)


@st.cache_data(show_spinner=False)
def _dataset():
    return data_gen.generate()


def _chips(items) -> str:
    return "".join(f"<span class='chip'>{html.escape(str(x))}</span>" for x in items)


def _tiles(pairs) -> str:
    cells = "".join(f"<div class='tile'><div class='num'>{n}</div>"
                    f"<div class='lab'>{html.escape(l)}</div></div>" for n, l in pairs)
    return f"<div class='tiles'>{cells}</div>"


def _bar(label: str, frac: float, value: str, color: str = "var(--accent)") -> str:
    pct = max(0, min(100, frac * 100))
    return (f"<div class='bar-row'><div class='bar-lab'>{html.escape(label)}</div>"
            f"<div class='bar-track'><div class='bar-fill' style='width:{pct:.0f}%;"
            f"background:{color}'></div></div>"
            f"<div class='bar-val'>{html.escape(value)}</div></div>")


def _route_html(rec, badge) -> str:
    return (f"<div class='route'><span class='brand'>{html.escape(rec.competitor_brand or '')}</span> "
            f"{html.escape(rec.competitor_model or '')}<span class='arrow'>&rarr;</span>"
            f"<span class='own'>{html.escape(rec.own_sku)}</span>"
            f"<span class='omodel'>{html.escape(rec.own_model)}</span>"
            f"<span style='float:right'><span class='badge accent'>"
            f"{html.escape(rec.chosen.kind.replace('_',' '))}</span>"
            f"<span class='badge good'>&#10003; {html.escape(badge)}</span></span></div>")


def _detail(sr, result, ds, tenant):
    rec = sr.recommendation
    match = result.resolution.by_competitor.get(rec.competitor_sku)
    own = next(s for s in ds.own_skus(tenant) if s["sku_id"] == rec.own_sku)
    cover = weeks_cover(own)
    margin = margin_pct(own["list_price"], own["unit_cost"])
    floor = own["margin_floor_pct"]

    st.markdown(_route_html(rec, sr.eval_badge), unsafe_allow_html=True)

    left, right = st.columns([3, 2], gap="large")
    with left:
        st.markdown("<div class='mr-label'>Traction signal</div>", unsafe_allow_html=True)
        wk = ds.market[ds.market.sku_id == rec.competitor_sku].sort_values("week")
        u0, p0 = wk.units_sold.iloc[0], wk.avg_price.iloc[0]
        chart = pd.DataFrame({
            "week": wk.week,
            "sell-out volume": (wk.units_sold / u0 * 100).round(1),
            "price": (wk.avg_price / p0 * 100).round(1),
        }).set_index("week")
        st.line_chart(chart, height=230, color=["#0f766e", "#c2410c"])
        st.caption(f"{sr.signal.norm_slope:.1%}/wk sell-out growth at a stable price "
                   f"(CoV {sr.signal.price_cov:.1%}). Indexed to week 1 = 100.")
    with right:
        st.markdown("<div class='mr-label'>Spec match &middot; no name matching</div>",
                    unsafe_allow_html=True)
        if match:
            st.markdown(
                _bar("structured", match.struct_sim, f"{match.struct_sim:.2f}")
                + _bar("graph overlap", match.jaccard, f"{match.jaccard:.2f}")
                + _bar("embedding", match.embed_sim, f"{match.embed_sim:.2f}")
                + _bar("blended", match.score, f"{match.score:.2f}", "var(--good)"),
                unsafe_allow_html=True)
            st.caption("matched on " + ", ".join(match.shared_attrs[:5]))

    st.markdown("<div class='mr-label'>What's driving it</div>", unsafe_allow_html=True)
    st.markdown(_chips(rec.drivers), unsafe_allow_html=True)

    st.markdown("<div class='mr-label'>Your SKU right now</div>", unsafe_allow_html=True)
    c = st.columns(5)
    c[0].metric("Unit cost", f"${own['unit_cost']:,.0f}")
    c[1].metric("List price", f"${own['list_price']:,.0f}")
    c[2].metric("Margin", f"{margin:.0%}", f"floor {floor:.0%}", delta_color="off")
    c[3].metric("Inventory", f"{own['inventory_units']:,}")
    c[4].metric("Weeks cover", f"{cover:.1f}")

    ws = sr.why_summary
    if ws and ws.top_aspects:
        st.markdown("<div class='mr-label'>Why buyers pick it &middot; mined from reviews</div>",
                    unsafe_allow_html=True)
        bars = "".join(_bar(a, (pos / total if total else 0), f"{pos}/{total}",
                            "var(--good)") for a, pos, total in ws.top_aspects)
        st.markdown(bars, unsafe_allow_html=True)
        st.caption(f"{ws.n_reviews} reviews classified by the RAG aspect model")

    st.markdown("<div class='mr-label'>Recommendation</div>", unsafe_allow_html=True)
    st.markdown(f"<div class='rationale'><b>{html.escape(rec.chosen.label)}</b><br>"
                f"{html.escape(rec.rationale)}</div>", unsafe_allow_html=True)

    st.markdown("<div class='mr-label'>Options the solver considered</div>",
                unsafe_allow_html=True)
    rows = ""
    for a in rec.ranked:
        cls = "act chosen" if a is rec.chosen else "act"
        tag = " &middot; recommended" if a is rec.chosen else ""
        rows += (f"<div class='{cls}'><span class='mark' style='color:var(--good)'>&#10003;</span>"
                 f"<span class='lab'>{html.escape(a.label)}{tag}</span></div>")
    for a in rec.infeasible:
        reason = a.reasons[0] if a.reasons else "not applicable"
        rows += (f"<div class='act'><span class='mark' style='color:#b91c1c'>&#10005;</span>"
                 f"<span class='lab'>{html.escape(a.label)}</span> "
                 f"<span class='why'>&mdash; {html.escape(reason)}</span></div>")
    st.markdown(rows, unsafe_allow_html=True)


# --- auth gate -------------------------------------------------------------- #
# Everything below is the authenticated app. Until the user signs in we render
# the login card and stop, so the pipeline never runs for an anonymous visitor.
if not st.session_state.get("authenticated", False):
    _render_login()
    st.stop()

# --- sidebar ---------------------------------------------------------------- #
with st.sidebar:
    st.markdown("### MarketRadar")
    st.caption("OEM commercial action layer")
    tenant = st.selectbox("Tenant", ["acme-pc", "globex-pc"])
    mode = st.radio("Model routing", ["cheap", "quality"], horizontal=True,
                    help="quality loads MiniLM + distilbert + flan-t5 locally")

# --- header ----------------------------------------------------------------- #
st.markdown(CSS, unsafe_allow_html=True)
st.markdown("<div class='mr-title'>MarketRadar</div>", unsafe_allow_html=True)
st.markdown("<div class='mr-sub'>Competitor configuration moves, mapped to your "
            "portfolio and turned into constrained actions.</div>", unsafe_allow_html=True)

# Centered, informative loading state while the (possibly model-heavy) pipeline
# runs. Shown once per tenant/mode in a session: the first quality run warms the
# models and takes a moment, so the card gives the user something steady to read;
# repeat runs are cache hits and skip it to avoid a flash.
_seen = st.session_state.setdefault("_computed", set())
_slot = st.empty()
if (tenant, mode) not in _seen:
    _slot.markdown(_loading_html(mode), unsafe_allow_html=True)

result = _load(tenant, mode)
ds = _dataset()

_slot.empty()
_seen.add((tenant, mode))

# --- sidebar: share + sign out (needs the computed result/ds) --------------- #
with st.sidebar:
    st.divider()
    st.markdown("#### Share findings")
    st.caption("Export this tenant's findings as a PDF to send to a client.")
    prepared_for = st.text_input("Prepared for", placeholder="Client name",
                                 help="Printed on the report cover.")
    st.download_button(
        "Download report (PDF)",
        data=build_report_pdf(result, ds, tenant, prepared_for),
        file_name=f"MarketRadar_{tenant}_report.pdf",
        mime="application/pdf",
        use_container_width=True)
    st.divider()
    if st.button("Sign out", use_container_width=True):
        st.session_state["authenticated"] = False
        st.rerun()

tract = [a for a in result.alerts if a.kind == "traction"]
restr = [a for a in result.alerts if a.kind == "restructure"]
passed = sum(1 for sr in result.recommendations if sr.eval_badge == "PASS")

st.markdown(_tiles([
    (len(tract), "traction alerts"),
    (len(restr), "restructure alerts"),
    (len(result.recommendations), "recommendations"),
    (f"{passed}/{len(result.recommendations) or 1}", "passed evals"),
]), unsafe_allow_html=True)

st.markdown("<div class='mr-label'>Market pulse</div>", unsafe_allow_html=True)
st.markdown(_chips(f"{n}={v}" for n, v, _ in result.market_drivers[:4]),
            unsafe_allow_html=True)

for a in restr:
    wk = ds.market[ds.market.brand == a.brand]
    before = sorted(wk[wk.week == a.week - 1].model_name.unique())[:2]
    after = sorted(wk[wk.week == a.week].model_name.unique())[:2]
    st.markdown(
        f"<div class='callout'><b>Restructure &middot; {html.escape(a.brand)} (wk {a.week}).</b> "
        f"<span class='rn'>{html.escape(', '.join(before))} &hellip; &rarr; "
        f"{html.escape(', '.join(after))} &hellip;</span> {html.escape(a.detail)}</div>",
        unsafe_allow_html=True)

# --- tabs ------------------------------------------------------------------- #
t_rec, t_map, t_model, t_audit = st.tabs(
    ["Recommendations", "Resolution map", "Model routing", "Audit trail"])

with t_rec:
    if not result.recommendations:
        st.info("No competitor traction maps into this tenant's portfolio right now.")
    else:
        labels = [f"{sr.recommendation.competitor_brand} {sr.recommendation.competitor_model}"
                  f"  →  {sr.recommendation.own_sku}   ·   {sr.recommendation.chosen.kind}"
                  for sr in result.recommendations]
        picked = st.radio("Signals", range(len(labels)),
                          format_func=lambda i: labels[i], label_visibility="collapsed")
        st.divider()
        _detail(result.recommendations[picked], result, ds, tenant)

with t_map:
    st.caption("Every competitor config resolved to the nearest own SKU by spec. "
               "Model names are shown but never used for matching.")
    rows = [{"competitor": m.competitor_model, "own": m.own_sku,
             "score": round(m.score, 2), "structured": round(m.struct_sim, 2),
             "embedding": round(m.embed_sim, 2),
             "matched on": ", ".join(m.shared_attrs[:4])}
            for m in result.resolution.matches]
    st.dataframe(rows, width="stretch", hide_index=True)

with t_model:
    s = result.model_summary
    st.caption(f"Routing mode: **{s['mode']}**. Cheap keeps everything on offline "
               "fallbacks; quality engages the local open-source models.")
    mc = st.columns(4)
    mc[0].metric("Calls", s["total_calls"])
    mc[1].metric("Cost units", s["total_cost_units"])
    mc[2].metric("Latency", f"{s['total_latency_ms']:.0f} ms")
    mc[3].metric("Models", len(s["models_used"]))
    st.markdown("<div class='mr-label'>Backends used</div>", unsafe_allow_html=True)
    st.dataframe([{"backend": b, "calls": n} for b, n in s["calls_by_backend"].items()],
                 width="stretch", hide_index=True)
    st.markdown(_chips(s["models_used"]), unsafe_allow_html=True)

with t_audit:
    st.caption("Every recommendation is written to an append-only log with its "
               "inputs, models, constraints and rationale.")
    records = AuditLog(tenant_id=tenant).records()[-len(result.recommendations):]
    st.dataframe([{"competitor": r["competitor_sku"], "own": r["own_sku"],
                   "action": r["chosen_action"]["kind"], "eval": r["eval_badge"],
                   "models": ", ".join(r["models_used"])} for r in records],
                 width="stretch", hide_index=True)
    if records:
        with st.expander("Full audit record (latest)"):
            st.json(records[-1])

st.markdown(f"<div class='foot' style='margin-top:1.4rem'>audit log &middot; "
            f"{html.escape(result.audit_path)}</div>", unsafe_allow_html=True)
