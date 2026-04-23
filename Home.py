"""
NZ Climate Dashboard
Home / landing page — NZ Climate Indicator Map only
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
header[data-testid="stHeader"] { display: none; }
div.block-container { padding-top: 0 !important; max-width: 1080px; }

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
    color: #7ecfff;
    margin-bottom: 0.75rem;
}
.hero-title {
    font-family: 'DM Serif Display', serif;
    font-size: 2.5rem;
    color: #ffffff;
    margin: 0 0 1rem;
    line-height: 1.15;
}
.hero-sub {
    font-size: 1.0rem;
    color: #aad4f5;
    max-width: 640px;
    line-height: 1.65;
    font-weight: 300;
}

/* Feature cards */
.features {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 1.2rem;
    margin: 2rem 0 2.5rem;
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
""", unsafe_allow_html=True)

# ── Hero ──────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
  <div class="hero-eyebrow">New Zealand · Climate Projections · Current reference period - End of century</div>
  <h1 class="hero-title">NZ Climate Indicator<br>Dashboard</h1>
  <p class="hero-sub">
    Explore how temperature, rainfall, and wind are projected to change across
    New Zealand over the coming decades. Compare outcomes under different
    emissions pathways and see what the range of climate models tells us —
    both where they agree, and where uncertainty remains.
  </p>
</div>
""", unsafe_allow_html=True)

# ── What this dashboard does ──────────────────────────────────────────────────
st.markdown("""
<div class="features">

  <div class="feat">
    <div class="feat-icon">🗺️</div>
    <div class="feat-title">Interactive climate maps</div>
    <div class="feat-body">
      See how climate indicators are projected to change across every part of
      New Zealand — from Northland to Southland — at fine spatial detail.
      Animate through future decades to watch trends unfold over time.
    </div>
  </div>

  <div class="feat">
    <div class="feat-icon">📊</div>
    <div class="feat-title">Change vs. current climate</div>
    <div class="feat-body">
      Two side-by-side maps let you compare <strong>what conditions are
      projected to look like</strong> alongside <strong>how much they differ
      from today's baseline</strong> — making it easy to communicate both
      impact and magnitude.
    </div>
  </div>

  <div class="feat">
    <div class="feat-icon">🔬</div>
    <div class="feat-title">Model agreement & uncertainty</div>
    <div class="feat-body">
      Click any location on the map to see how the full range of global
      climate models responds over time. Where models agree, confidence is
      high; where they diverge, the spread tells you where uncertainty matters most.
    </div>
  </div>

  <div class="feat">
    <div class="feat-icon">☁️</div>
    <div class="feat-title">Four emissions scenarios</div>
    <div class="feat-body">
      Projections span four futures — from strong mitigation to very high
      emissions — so decision-makers can assess risk across a plausible range
      rather than committing to a single storyline.
    </div>
  </div>

  <div class="feat">
    <div class="feat-icon">🌡️</div>
    <div class="feat-title">25+ climate indicators</div>
    <div class="feat-body">
      Go beyond average temperature. Explore frost days, hot days above 30°C,
      heavy rainfall events, wind extremes, and more — the indicators most
      relevant to infrastructure, agriculture, and natural hazard planning.
    </div>
  </div>

  <div class="feat">
    <div class="feat-icon">📍</div>
    <div class="feat-title">Likelihood of record-breaking events</div>
    <div class="feat-body">
      Dedicated indicators show how the probability of breaking heat, rainfall,
      and wind records shifts under each emissions scenario — directly relevant
      to risk assessments and adaptation planning.
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
with st.expander("Indicator & scenario quick reference"):
    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("**Scenarios available**")
        st.markdown("""
- **SSP1-2.6** — Low emissions / strong mitigation  
- **SSP2-4.5** — Intermediate / moderate action  
- **SSP3-7.0** — High emissions / delayed action  
- **SSP5-8.5** — Very high emissions / fossil-fuel intensive  
        """)
        st.markdown("**Baseline periods**")
        st.markdown("""
- **1995–2014** — Recent observational baseline  
- **1986–2005** — Earlier baseline (aligned with IPCC AR5)  
        """)
    with col_b:
        st.markdown("**Indicator categories**")
        st.markdown("""
- **Temperature** — TX, TN, TXx, TNn, FD, TX25, TX30  
- **Precipitation** — PR, Rx1day, RR1mm, RR25mm, R99p, DD1mm  
- **Wind** — sfcwind, Wd10, Wd25, Wx1day, Wd99pVAL  
- **Record chance** — REC_TXx, REC_TNn, REC_Rx1day, REC_Wx1day  
        """)
        st.markdown("**Seasons**")
        st.markdown("""
- ANN (Annual), DJF (Summer), MAM (Autumn), JJA (Winter), SON (Spring)  
        """)

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="footer">
  Climate projections based on global model ensembles downscaled to a 12 km NZ grid<br>
  Scenarios: SSP1-2.6 · SSP2-4.5 · SSP3-7.0 · SSP5-8.5 &nbsp;·&nbsp; Baselines: 1986–2005 · 1995–2014
</div>
""", unsafe_allow_html=True)