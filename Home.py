"""
NIWA-REMS Climate Dashboard
Home / landing page
"""
import streamlit as st

st.set_page_config(
    page_title="NIWA-REMS Climate Dashboard",
    page_icon="🌏",
    layout="wide",
)

st.markdown("""
<style>
html, body, [class*="css"]  { font-size: 14px !important; }
h1, h2, h3, h4 { font-size: 1.2rem !important; line-height: 1.2 !important; }
header[data-testid="stHeader"] { display: none; }
div.block-container { padding-top: 2.5rem; }
</style>
""", unsafe_allow_html=True)

st.title("🌏 NIWA-REMS Climate Dashboard")
st.markdown("---")

col1, col2 = st.columns(2)

with col1:
    st.subheader("📈 Climate Graph")
    st.markdown(
        """
        View projected changes in temperature, precipitation and wind speed
        for New Zealand regions under different emissions scenarios (SSP126–SSP585).
        Compare CMIP6 model ensembles against observed records.
        """
    )
    if st.button("Open Climate Graph →", use_container_width=True):
        st.switch_page("pages/1_Climate_Graph.py")

with col2:
    st.subheader("🗺️ NZ Climate Indicator Maps")
    st.markdown(
        """
        Explore gridded climate indicator maps across New Zealand.
        Select a scenario, indicator, season, and time period to visualise
        the multi-model ensemble mean on a 12 km resolution NZ grid.
        """
    )
    if st.button("Open NZ Map →", use_container_width=True):
        st.switch_page("pages/2_NZ_Map.py")

st.markdown("---")
st.caption(
    "Data: NIWA-REMS ML-Downscaled CMIP6 indicators (output_v3). "
    "Scenarios: historical, SSP1-2.6, SSP2-4.5, SSP3-7.0, SSP5-8.5."
)
