"""
NZ Climate Dashboard
Home / landing page — NZ Climate Indicator Map
"""
import streamlit as st

st.set_page_config(
    page_title="NZ Climate Dashboard",
    page_icon="🌏",
    layout="wide",
)

# ── Global CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display&family=DM+Sans:wght@300;400;500;600&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    font-size: 14px !important;
}
h1, h2, h3, h4 {
    font-family: 'DM Serif Display', serif !important;
    line-height: 1.2 !important;
}

/* Make Streamlit's header transparent but DO NOT collapse it —
   collapsing it ('display:none' or 'height:0') destroys the sidebar
   collapse/expand button along with everything else inside the header. */
header[data-testid="stHeader"] {
    background: transparent !important;
}

/* Belt-and-braces: ensure the sidebar collapsed-control button is always
   present, visible, and on top of everything else, no matter what
   Streamlit version is in play. */
[data-testid="collapsedControl"],
[data-testid="stSidebarCollapsedControl"] {
    display: flex !important;
    visibility: visible !important;
    opacity: 1 !important;
    position: fixed !important;
    top: 0.5rem !important;
    left: 0.5rem !important;
    z-index: 999999 !important;
    pointer-events: auto !important;
}

div.block-container { padding-top: 3rem !important; max-width: 1080px; }

/* Hero section */
.hero {
    background: linear-gradient(135deg, #0a2540 0%, #103358 60%, #1a4f7a 100%);
    border-radius: 0 0 2rem 2rem;
    padding: 3.5rem 3rem 2.5rem;
    margin-bottom: 2.5rem;
    position: relative;
    overflow: hidden;
}
.hero::before {
    content: '';
    position: absolute;
    inset: 0;
    background: radial-gradient(ellipse at 80% 50%, rgba(74,144,217,0.18) 0%, transparent 65%);
    pointer-events: none;
}
.hero-eyebrow {
    font-family: 'DM Sans', sans-serif;
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 0.18em;
    text-transform: uppercase;
    color: #7ecfff !important;
    margin-bottom: 0.75rem;
}
.hero-title {
    font-family: 'DM Serif Display', serif !important;
    font-size: 2.5rem !important;
    color: #ffffff !important;
    margin: 0 0 1rem !important;
    line-height: 1.15 !important;
}
.hero-sub {
    font-size: 1.0rem;
    color: #aad4f5 !important;
    max-width: 680px;
    line-height: 1.65;
    font-weight: 300;
}
.method-pills {
    display: flex;
    gap: 0.6rem;
    margin-top: 1.4rem;
    flex-wrap: wrap;
    position: relative;
    z-index: 2;
}
.method-pill {
    background: rgba(255,255,255,0.08);
    border: 1px solid rgba(126,207,255,0.35);
    color: #e8f4ff !important;
    border-radius: 999px;
    padding: 0.35rem 0.95rem;
    font-size: 0.78rem;
    font-weight: 500;
    letter-spacing: 0.02em;
}
.method-pill.sd { border-color: rgba(74,144,217,0.55); }
.method-pill.dd { border-color: rgba(34,168,34,0.55); }

/* Methods explainer section */
.methods-section { margin: 1rem 0 2.5rem; }
.methods-eyebrow {
    font-size: 0.7rem;
    font-weight: 600;
    letter-spacing: 0.18em;
    text-transform: uppercase;
    color: #1a4f7a;
    margin-bottom: 0.5rem;
}
.methods-section h2 {
    font-family: 'DM Serif Display', serif !important;
    font-size: 1.7rem !important;
    color: #0a2540 !important;
    margin: 0 0 0.75rem !important;
}
.methods-intro {
    font-size: 0.95rem;
    color: #445566;
    line-height: 1.7;
    max-width: 820px;
    margin-bottom: 1.5rem;
}
.methods-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 1.2rem;
    margin-bottom: 1.2rem;
}
.method-card {
    background: #fafcff;
    border-radius: 1rem;
    padding: 1.5rem 1.4rem 1.4rem;
    border-top: 4px solid #ccc;
    box-shadow: 0 1px 3px rgba(10,37,64,0.04);
}
.method-card.sd-card {
    border-top-color: #4a90d9;
    background: linear-gradient(180deg, #f3f8fd 0%, #fafcff 100%);
}
.method-card.dd-card {
    border-top-color: #22a822;
    background: linear-gradient(180deg, #f3fbf3 0%, #fafffa 100%);
}
.method-icon { font-size: 1.7rem; margin-bottom: 0.4rem; }
.method-tag {
    font-size: 0.7rem;
    font-weight: 700;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: #1a4f7a;
    margin-bottom: 0.2rem;
}
.method-card.dd-card .method-tag { color: #145214; }
.method-title {
    font-family: 'DM Serif Display', serif;
    font-size: 1.05rem;
    color: #0a2540;
    margin-bottom: 0.6rem;
}
.method-body {
    font-size: 0.85rem;
    color: #445566;
    line-height: 1.65;
}
.methods-takeaway {
    background: #fff8e6;
    border-left: 4px solid #e8a020;
    border-radius: 0.5rem;
    padding: 0.9rem 1.1rem;
    font-size: 0.88rem;
    color: #4a3a1a;
    line-height: 1.6;
}

/* Feature cards */
.features {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 1.2rem;
    margin: 1.5rem 0 2.5rem;
}
.feat {
    background: #f5f8fc;
    border: 1px solid #dce7f3;
    border-radius: 1rem;
    padding: 1.4rem 1.3rem;
}
.feat-icon { font-size: 1.6rem; margin-bottom: 0.55rem; }
.feat-title {
    font-family: 'DM Serif Display', serif;
    font-size: 1.0rem;
    color: #0a2540;
    margin-bottom: 0.4rem;
}
.feat-body { font-size: 0.85rem; color: #445566; line-height: 1.6; }

/* Section heading */
.section-heading {
    font-family: 'DM Serif Display', serif;
    font-size: 1.4rem;
    color: #0a2540;
    margin: 1rem 0 0.8rem;
}

/* Divider */
.divider {
    border: none;
    border-top: 1px solid #dce7f3;
    margin: 2rem 0;
}

/* Footer */
.footer {
    font-size: 0.75rem;
    color: #8899aa;
    padding: 1.5rem 0 2rem;
    text-align: center;
    line-height: 1.7;
}

/* Streamlit button override */
div[data-testid="stButton"] > button {
    background: #1a4f7a !important;
    color: #ffffff !important;
    border: none !important;
    border-radius: 0.6rem !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 0.95rem !important;
    font-weight: 500 !important;
    padding: 0.65rem 2rem !important;
    transition: background 0.2s ease !important;
}
div[data-testid="stButton"] > button:hover {
    background: #0a2540 !important;
}
</style>
<script>
/* Heartbeat that re-applies the sidebar control styles every 500 ms.
   Streamlit re-renders parts of the app frequently and can briefly
   strip the styles or detach the button — this keeps it pinned. */
(function() {
    function fixBtn() {
        var sels = ['[data-testid="collapsedControl"]',
                    '[data-testid="stSidebarCollapsedControl"]'];
        for (var i = 0; i < sels.length; i++) {
            var btn = window.parent.document.querySelector(sels[i])
                   || document.querySelector(sels[i]);
            if (!btn) continue;
            btn.style.setProperty('display',    'flex',    'important');
            btn.style.setProperty('visibility', 'visible', 'important');
            btn.style.setProperty('opacity',    '1',       'important');
            btn.style.setProperty('position',   'fixed',   'important');
            btn.style.setProperty('top',        '0.5rem',  'important');
            btn.style.setProperty('left',       '0.5rem',  'important');
            btn.style.setProperty('z-index',    '999999',  'important');
            btn.style.setProperty('pointer-events', 'auto', 'important');
        }
    }
    fixBtn();
    setInterval(fixBtn, 500);
    try {
        new MutationObserver(fixBtn).observe(document.body, { childList: true, subtree: true });
    } catch (e) {}
})();
</script>
""", unsafe_allow_html=True)

# ── Hero ──────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
<div class="hero-eyebrow">New Zealand · Climate Projections · Historical baseline → end of century</div>
<h1 class="hero-title">NZ Climate Indicator<br>Dashboard</h1>
<p class="hero-sub">
Explore how temperature, rainfall, and wind are projected to change across
New Zealand over the coming decades. Compare four emissions pathways, see
what the full range of climate models tells us, and switch between two
independent downscaling methods to find out where projections agree and
where uncertainty matters most.
</p>
<div class="method-pills">
<span class="method-pill sd">📊 Statistical · AI · 12 km</span>
<span class="method-pill dd">🌀 Dynamical · CCAM · 5 km</span>
<span class="method-pill">15 indicators</span>
<span class="method-pill">4 SSP scenarios</span>
</div>
</div>
""", unsafe_allow_html=True)

# ── Methods explainer ─────────────────────────────────────────────────────────
st.markdown("""
<div class="methods-section">
<div class="methods-eyebrow">Understanding the methods</div>
<h2>Two independent views of New Zealand's climate future</h2>
<p class="methods-intro">
Global climate models simulate the whole planet at coarse resolution —
typical grid cells are 100–200&nbsp;km across, far too blunt to capture
New Zealand's mountains, coastlines, and microclimates.
<strong>Downscaling</strong> bridges that gap by translating coarse global
signals onto a high-resolution NZ grid. This dashboard runs two
complementary downscaling methods in parallel — both are shown side by
side so you can compare them directly with a single click.
</p>
<div class="methods-grid">
<div class="method-card sd-card">
<div class="method-icon">📊</div>
<div class="method-tag">Statistical Downscaling — "SD"</div>
<div class="method-title">AI-based · 12 km NZ grid</div>
<div class="method-body">
Uses machine-learning models trained on decades of New Zealand
observations to translate output from global climate models onto a
fine NZ grid. Fast and efficient — we can downscale many global
models, scenarios, and seasons at moderate cost. Strong for indicators
where statistical relationships with the large-scale climate are
well-established (e.g. temperature, seasonal rainfall).
</div>
</div>
<div class="method-card dd-card">
<div class="method-icon">🌀</div>
<div class="method-tag">Dynamical Downscaling — "DD"</div>
<div class="method-title">Regional climate model · 5 km NZ grid</div>
<div class="method-body">
Runs a high-resolution regional climate model (CCAM) over New Zealand
that <em>physically simulates</em> the atmosphere — winds, clouds,
rainfall, terrain interactions. Computationally expensive, but resolves
the physics of orographic rainfall, sea breezes, and extremes more
directly. Available for a smaller selection of global models and
scenarios.
</div>
</div>
</div>
<p class="methods-takeaway">
💡 <strong>Why two methods?</strong> When SD and DD agree on a projection,
you can plan with high confidence — the result is robust to method choice.
When they disagree, that's a signal of genuine scientific uncertainty in
how that variable will respond, and worth flagging in any risk assessment.
On the map page, a single toggle switches between methods instantly,
without losing your zoom or location.
</p>
</div>
""", unsafe_allow_html=True)

# ── Feature cards ─────────────────────────────────────────────────────────────
st.markdown('<div class="section-heading">What you can do</div>', unsafe_allow_html=True)

st.markdown("""
<div class="features">

<div class="feat">
<div class="feat-icon">🗺️</div>
<div class="feat-title">Side-by-side maps</div>
<div class="feat-body">
Two synced panels show <strong>absolute conditions</strong> (what the
climate looks like) alongside the <strong>change vs. today's baseline</strong>
(how much it differs). Pan and zoom one — the other follows.
</div>
</div>

<div class="feat">
<div class="feat-icon">⏯️</div>
<div class="feat-title">Animated timeline</div>
<div class="feat-body">
Press play to watch climate signals evolve from today through to the end
of the century. Smooth interpolation between data snapshots gives a
continuous view of how changes unfold.
</div>
</div>

<div class="feat">
<div class="feat-icon">🔬</div>
<div class="feat-title">Model uncertainty at a click</div>
<div class="feat-body">
Click any location to open a time-series chart showing the full spread
across the global model ensemble — 50% and 90% intervals, plus the
ensemble mean. Both downscaling methods overlaid for direct comparison.
</div>
</div>

<div class="feat">
<div class="feat-icon">☁️</div>
<div class="feat-title">Four emissions scenarios</div>
<div class="feat-body">
Projections span four futures — from strong mitigation (SSP1-2.6) to
very high emissions (SSP5-8.5) — so risk can be assessed across a
plausible range rather than a single storyline.
</div>
</div>

<div class="feat">
<div class="feat-icon">🌡️</div>
<div class="feat-title">15 climate indicators</div>
<div class="feat-body">
Beyond mean temperature: frost days, hot days above 30 °C, heavy
rainfall events, wind extremes, and more — the indicators most relevant
to infrastructure, agriculture, and natural hazard planning.
</div>
</div>

<div class="feat">
<div class="feat-icon">📍</div>
<div class="feat-title">Likelihood of record-breaking events</div>
<div class="feat-body">
Dedicated indicators show how the probability of breaking heat,
rainfall, and wind records shifts under each emissions scenario —
directly relevant to risk assessments and adaptation planning.
</div>
</div>

</div>
""", unsafe_allow_html=True)

# ── CTA ───────────────────────────────────────────────────────────────────────
col_cta, col_spacer = st.columns([1, 2])
with col_cta:
    if st.button("Open NZ Climate Indicator Map →", use_container_width=True):
        st.switch_page("pages/2_NZ_Map.py")

st.markdown('<hr class="divider">', unsafe_allow_html=True)

# ── Quick reference ───────────────────────────────────────────────────────────
with st.expander("Quick reference: methods, scenarios, indicators"):
    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("**Downscaling methods**")
        st.markdown("""
- **📊 Statistical (SD)** — AI / machine-learning · 12 km grid · all CMIP6 models, all SSPs  
- **🌀 Dynamical (DD)** — CCAM regional climate model · 5 km grid · selected models / SSPs  
        """)
        st.markdown("**Emissions scenarios**")
        st.markdown("""
- **SSP1-2.6** — Low emissions / strong mitigation  
- **SSP2-4.5** — Intermediate / moderate action  
- **SSP3-7.0** — High emissions / delayed action  
- **SSP5-8.5** — Very high emissions / fossil-fuel intensive  
        """)
        st.markdown("**Baseline periods**")
        st.markdown("""
- **1995–2014** — Recent observational baseline (recommended)  
- **1986–2005** — Earlier baseline (aligned with IPCC AR5)  
        """)
    with col_b:
        st.markdown("**Indicator categories**")
        st.markdown("""
- **Temperature** — TX, TN, TXx, TNn, FD, TX25, TX30  
- **Precipitation** — PR, Rx1day, RR1mm, RR25mm, R99p, DD1mm   
- **Record chance** — REC_TXx, REC_Rx1day 
        """)
        st.markdown("**Seasons**")
        st.markdown("""
- ANN (Annual), DJF (Summer), MAM (Autumn), JJA (Winter), SON (Spring)  
        """)
        st.markdown("**Reading the chart**")
        st.markdown("""
- Filled bands = current method's 50% / 90% model spread  
- Dashed bands = the *other* method's spread (for comparison)  
- Solid line = ensemble mean · Dark line = selected single model  
        """)

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="footer">
Climate projections from CMIP6 global model ensembles, downscaled to New Zealand using<br>
Statistical (AI · 12 km) and Dynamical (CCAM · 5 km) methods<br>
Scenarios: SSP1-2.6 · SSP2-4.5 · SSP3-7.0 · SSP5-8.5 &nbsp;·&nbsp; Baselines: 1986–2005 · 1995–2014
</div>
""", unsafe_allow_html=True)