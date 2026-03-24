# pages/2_NZ_Map.py
"""
NZ Climate Indicator Map
Reads NetCDF files from index_data/{scenario}/static_maps/{indicator}/
Averages across all matching models and displays as an interactive map.
"""
from pathlib import Path
import re

import numpy as np
import streamlit as st

st.set_page_config(page_title="NZ Climate Indicator Map", layout="wide")

st.markdown("""
<style>
html, body, [class*="css"]  { font-size: 14px !important; }
h1, h2, h3, h4 { font-size: 1.2rem !important; line-height: 1.2 !important; }
header[data-testid="stHeader"] { display: none; }
div.block-container { padding-top: 2.5rem; }
</style>
""", unsafe_allow_html=True)

# ── Try to import optional heavy deps gracefully ──────────────────────────────
try:
    import xarray as xr
    HAS_XARRAY = True
except ImportError:
    HAS_XARRAY = False

try:
    import plotly.graph_objects as go
    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False

# ── Constants ─────────────────────────────────────────────────────────────────
DATA_ROOT = Path("index_data")

SCENARIOS = ["historical", "ssp126", "ssp245", "ssp370", "ssp585"]

SCENARIO_LABELS = {
    "historical": "Historical",
    "ssp126": "SSP1-2.6 (low emissions)",
    "ssp245": "SSP2-4.5 (moderate emissions)",
    "ssp370": "SSP3-7.0 (high emissions)",
    "ssp585": "SSP5-8.5 (very high emissions)",
}

# Human-readable indicator descriptions
INDICATOR_LABELS = {
    "DD1mm":       "Dry days (<1 mm/day)",
    "FD":          "Frost days (Tmin < 0°C)",
    "PR":          "Precipitation",
    "R99p":        "99th percentile precipitation",
    "R99pVAL":     "99th percentile rainfall value",
    "R99pVALWet":  "99th-pct rainfall on wet days",
    "RR1mm":       "Rain days (≥1 mm)",
    "RR25mm":      "Heavy rain days (≥25 mm)",
    "Rx1day":      "Max 1-day precipitation",
    "sfcwind":     "Mean surface wind speed",
    "TN":          "Mean minimum temperature",
    "TNn":         "Coldest night (annual min Tmin)",
    "TX":          "Mean maximum temperature",
    "TX25":        "Days with Tmax > 25°C",
    "TX30":        "Days with Tmax > 30°C",
    "TXx":         "Hottest day (annual max Tmax)",
    "Wd10":        "Wind days ≥ 10 m/s",
    "Wd25":        "Wind days ≥ 25 m/s",
    "Wd99pVAL":    "99th-pct wind speed value",
    "Wx1day":      "Max 1-day wind speed",
}

SEASON_LABELS = {
    "ANN": "Annual",
    "DJF": "Summer (DJF)",
    "MAM": "Autumn (MAM)",
    "JJA": "Winter (JJA)",
    "SON": "Spring (SON)",
}

# ── Helpers ───────────────────────────────────────────────────────────────────
def list_indicators(scenario: str) -> list[str]:
    """Return sorted list of indicator folders available for a scenario."""
    base = DATA_ROOT / scenario / "static_maps"
    if not base.exists():
        return []
    return sorted(d.name for d in base.iterdir() if d.is_dir())


def list_seasons_for(scenario: str, indicator: str) -> list[str]:
    """Return seasons present in filenames for this scenario/indicator."""
    folder = DATA_ROOT / scenario / "static_maps" / indicator
    if not folder.exists():
        return []
    seasons = set()
    for f in folder.glob("*.nc"):
        # filename: {indicator}_{scenario}_{model}_{ensemble}_*_{SEASON}_NZ12km.nc
        parts = f.stem.split("_")
        # Season is the second-to-last part before NZ12km
        for part in parts:
            if part in SEASON_LABELS:
                seasons.add(part)
    return sorted(seasons, key=lambda s: list(SEASON_LABELS.keys()).index(s) if s in SEASON_LABELS else 99)


def list_periods_for(scenario: str, indicator: str) -> list[str]:
    """Return baseline periods present in filenames (e.g. bp1986-2005)."""
    folder = DATA_ROOT / scenario / "static_maps" / indicator
    if not folder.exists():
        return []
    periods = set()
    for f in folder.glob("*.nc"):
        m = re.search(r"(bp\d{4}-\d{4})", f.name)
        if m:
            periods.add(m.group(1))
    return sorted(periods)


def find_nc_files(scenario: str, indicator: str, season: str, period: str) -> list[Path]:
    """
    Find all NetCDF files matching the given selector.
    File pattern: {indicator}_{scenario}_{model}_{ensemble}_*_{period}_{season}_NZ12km.nc
    """
    folder = DATA_ROOT / scenario / "static_maps" / indicator
    if not folder.exists():
        return []
    matched = []
    for f in folder.glob("*.nc"):
        stem = f.name
        if season in stem and period in stem:
            matched.append(f)
    return sorted(matched)


@st.cache_data(show_spinner="Loading NetCDF data…")
def load_ensemble_mean(
    scenario: str, indicator: str, season: str, period: str
) -> tuple[np.ndarray | None, np.ndarray | None, np.ndarray | None, int, str]:
    """
    Load all matching NetCDF files, extract the first data variable,
    and return the ensemble mean as (data_2d, lats, lons, n_models, var_name).
    """
    files = find_nc_files(scenario, indicator, season, period)
    if not files:
        return None, None, None, 0, ""

    arrays, lats, lons, var_name = [], None, None, ""

    for f in files:
        try:
            ds = xr.open_dataset(f, engine="netcdf4")

            # Find lat/lon coordinates
            lat_names = [c for c in ds.coords if c.lower() in ("lat","latitude","y","rlat")]
            lon_names = [c for c in ds.coords if c.lower() in ("lon","longitude","x","rlon")]

            if not lat_names or not lon_names:
                # Try dims
                lat_names = [d for d in ds.dims if d.lower() in ("lat","latitude","y","rlat")]
                lon_names = [d for d in ds.dims if d.lower() in ("lon","longitude","x","rlon")]

            if not lat_names or not lon_names:
                ds.close()
                continue

            lat_name = lat_names[0]
            lon_name = lon_names[0]

            # First non-coord data variable
            data_vars = [v for v in ds.data_vars if v.lower() not in ("lat","lon","time")]
            if not data_vars:
                ds.close()
                continue

            vname = data_vars[0]
            var_name = vname

            da = ds[vname]
            # If there's a time or depth dimension with size 1, squeeze it
            extra_dims = [d for d in da.dims if d not in (lat_name, lon_name)]
            for dim in extra_dims:
                if da.sizes[dim] == 1:
                    da = da.isel({dim: 0})

            if lats is None:
                lats = ds[lat_name].values
                lons = ds[lon_name].values

            arrays.append(da.values.astype(float))
            ds.close()

        except Exception:
            continue

    if not arrays:
        return None, None, None, 0, var_name

    stack = np.stack(arrays, axis=0)
    mean  = np.nanmean(stack, axis=0)
    return mean, lats, lons, len(arrays), var_name


def build_map_figure(
    data: np.ndarray,
    lats: np.ndarray,
    lons: np.ndarray,
    title: str,
    colorscale: str = "RdBu_r",
    units: str = "",
) -> "go.Figure":
    """
    Build a Plotly figure from a 2D data array on a lat/lon grid.
    Handles both 1-D coordinate arrays (regular grid) and 2-D arrays (rotated grid).
    """
    # Flatten for scatter if coordinates are 2D; use imshow for regular grid
    if lats.ndim == 2 and lons.ndim == 2:
        # Rotated / irregular grid → scatter
        lat_flat  = lats.ravel()
        lon_flat  = lons.ravel()
        data_flat = data.ravel()

        # Remove NaN
        valid = np.isfinite(data_flat)
        lat_flat  = lat_flat[valid]
        lon_flat  = lon_flat[valid]
        data_flat = data_flat[valid]

        vmin, vmax = np.nanpercentile(data_flat, 2), np.nanpercentile(data_flat, 98)

        fig = go.Figure(go.Scattermap(
            lat=lat_flat.tolist(),
            lon=lon_flat.tolist(),
            mode="markers",
            marker=dict(
                size=4,
                color=data_flat.tolist(),
                colorscale=colorscale,
                cmin=vmin,
                cmax=vmax,
                colorbar=dict(title=units, thickness=15, len=0.7),
                showscale=True,
            ),
            hovertemplate="Lat: %{lat:.2f}<br>Lon: %{lon:.2f}<br>Value: %{marker.color:.2f}<extra></extra>",
        ))

        fig.update_layout(
            map=dict(
                style="carto-positron",
                center=dict(lat=-41.5, lon=172.5),
                zoom=4.2,
            ),
            title=dict(text=title, x=0.05, font=dict(size=16)),
            height=680,
            margin=dict(l=0, r=0, t=50, b=0),
        )

    else:
        # Regular 1-D lat/lon arrays → imshow / heatmap
        # Ensure lats are sorted ascending for display
        if lats[0] > lats[-1]:
            lats = lats[::-1]
            data = data[::-1, :]

        vmin, vmax = np.nanpercentile(data[np.isfinite(data)], 2), \
                     np.nanpercentile(data[np.isfinite(data)], 98)

        import plotly.express as px
        import pandas as pd

        # Build a flat DataFrame for density_mapbox / scatter_mapbox
        lat_grid, lon_grid = np.meshgrid(lats, lons, indexing="ij")
        df_map = pd.DataFrame({
            "lat":   lat_grid.ravel(),
            "lon":   lon_grid.ravel(),
            "value": data.ravel(),
        }).dropna(subset=["value"])

        fig = go.Figure(go.Scattermap(
            lat=df_map["lat"].tolist(),
            lon=df_map["lon"].tolist(),
            mode="markers",
            marker=dict(
                size=4,
                color=df_map["value"].tolist(),
                colorscale=colorscale,
                cmin=vmin,
                cmax=vmax,
                colorbar=dict(title=units, thickness=15, len=0.7),
                showscale=True,
            ),
            hovertemplate="Lat: %{lat:.2f}<br>Lon: %{lon:.2f}<br>Value: %{marker.color:.2f}<extra></extra>",
        ))

        fig.update_layout(
            map=dict(
                style="carto-positron",
                center=dict(lat=-41.5, lon=172.5),
                zoom=4.2,
            ),
            title=dict(text=title, x=0.05, font=dict(size=16)),
            height=680,
            margin=dict(l=0, r=0, t=50, b=0),
        )

    return fig


# ── Pick a sensible colour scale per indicator ────────────────────────────────
def colorscale_for(indicator: str) -> str:
    cold = {"FD", "TNn", "TN"}
    warm = {"TX", "TXx", "TX25", "TX30"}
    rain = {"PR", "RR1mm", "RR25mm", "Rx1day", "R99p", "R99pVAL", "R99pVALWet"}
    wind = {"sfcwind", "Wd10", "Wd25", "Wd99pVAL", "Wx1day"}
    dry  = {"DD1mm"}

    if indicator in cold:
        return "Blues"
    if indicator in warm:
        return "Reds"
    if indicator in rain:
        return "YlGnBu"
    if indicator in wind:
        return "Purples"
    if indicator in dry:
        return "YlOrBr"
    return "RdBu_r"


# ── UI ────────────────────────────────────────────────────────────────────────
st.title("🗺️ NZ Climate Indicator Map")

if not HAS_XARRAY:
    st.error(
        "**`xarray` is not installed.** "
        "Add `xarray` and `netcdf4` to `requirements.txt` and redeploy."
    )
    st.stop()

if not HAS_PLOTLY:
    st.error("**`plotly` is not installed.**")
    st.stop()

if not DATA_ROOT.exists():
    st.warning(
        f"Data directory **`{DATA_ROOT}/`** not found.  \n"
        "Place the `output_v3` folder contents under `index_data/` in the repo root."
    )
    st.stop()

# Sidebar controls
with st.sidebar:
    logo_path = Path("logos/esnz_logo_horz_new.png")
    if logo_path.exists():
        st.image(str(logo_path))

    st.subheader("Scenario")
    scenario = st.selectbox(
        "",
        SCENARIOS,
        format_func=lambda s: SCENARIO_LABELS.get(s, s),
        label_visibility="collapsed",
    )

    indicators_avail = list_indicators(scenario)
    if not indicators_avail:
        st.warning(f"No indicator data found under `{DATA_ROOT}/{scenario}/static_maps/`")
        st.stop()

    st.subheader("Indicator")
    indicator = st.selectbox(
        "",
        indicators_avail,
        format_func=lambda i: f"{i} — {INDICATOR_LABELS.get(i, '')}",
        label_visibility="collapsed",
    )

    seasons_avail = list_seasons_for(scenario, indicator)
    if not seasons_avail:
        seasons_avail = ["ANN"]

    st.subheader("Season")
    season = st.selectbox(
        "",
        seasons_avail,
        format_func=lambda s: SEASON_LABELS.get(s, s),
        label_visibility="collapsed",
    )

    periods_avail = list_periods_for(scenario, indicator)
    if not periods_avail:
        periods_avail = ["bp1995-2014"]

    st.subheader("Baseline period")
    period = st.selectbox(
        "",
        periods_avail,
        label_visibility="collapsed",
        help="The reference period encoded in the filename (e.g. bp1995-2014).",
    )

    st.markdown("---")
    st.subheader("Colour scale")
    default_cs = colorscale_for(indicator)
    COLOUR_SCALES = [
        "RdBu_r","RdBu","Reds","Blues","Greens","Purples",
        "YlOrRd","YlGnBu","Viridis","Plasma","Cividis","Turbo",
    ]
    colorscale = st.selectbox(
        "",
        COLOUR_SCALES,
        index=COLOUR_SCALES.index(default_cs) if default_cs in COLOUR_SCALES else 0,
        label_visibility="collapsed",
    )

# Main area
col_map, col_info = st.columns([3, 1], vertical_alignment="top")

with col_info:
    st.markdown(f"**Indicator:** {indicator}")
    st.markdown(f"**Description:** {INDICATOR_LABELS.get(indicator, '—')}")
    st.markdown(f"**Scenario:** {SCENARIO_LABELS.get(scenario, scenario)}")
    st.markdown(f"**Season:** {SEASON_LABELS.get(season, season)}")
    st.markdown(f"**Period:** {period}")
    st.markdown("---")

    # List of files that will be loaded
    files = find_nc_files(scenario, indicator, season, period)
    if files:
        st.markdown(f"**{len(files)} model file(s) found:**")
        for f in files:
            # Extract model name from filename
            parts = f.stem.split("_")
            # model is usually 3rd token: {indicator}_{scenario}_{model}...
            model_name = parts[2] if len(parts) > 2 else f.stem
            st.caption(f"• {model_name}")
    else:
        st.warning("No files matched the current selection.")

with col_map:
    if not files:
        st.info("Select a combination above that has data files available.")
    else:
        data, lats, lons, n_models, var_name = load_ensemble_mean(scenario, indicator, season, period)

        if data is None:
            st.error(
                "Could not load any data. "
                "Check that the NetCDF files are valid and that `netcdf4` is installed."
            )
        else:
            season_label    = SEASON_LABELS.get(season, season)
            scenario_label  = SCENARIO_LABELS.get(scenario, scenario)
            indicator_label = INDICATOR_LABELS.get(indicator, indicator)

            title = (
                f"{indicator} — {indicator_label}<br>"
                f"{scenario_label} | {season_label} | {period} "
                f"(ensemble mean, n={n_models})"
            )

            fig = build_map_figure(
                data, lats, lons,
                title=title,
                colorscale=colorscale,
                units=indicator,
            )

            st.plotly_chart(fig, use_container_width=True)

            # Stats
            valid = data[np.isfinite(data)]
            if valid.size > 0:
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Min",  f"{np.nanmin(valid):.2f}")
                c2.metric("Mean", f"{np.nanmean(valid):.2f}")
                c3.metric("Max",  f"{np.nanmax(valid):.2f}")
                c4.metric("Std",  f"{np.nanstd(valid):.2f}")
