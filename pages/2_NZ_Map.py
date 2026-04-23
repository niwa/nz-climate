# pages/2_NZ_Map.py — Part 1/3
"""
NZ Climate Indicator Map — dual-panel (absolute + change) animated timeline.
================================================================================
 
OVERVIEW
--------
A Streamlit page that renders *two* synchronised animated maps of a NZ climate
indicator over time:
 
    LEFT panel  — multi-model ensemble *absolute values* (sequential colour scale).
    RIGHT panel — *change* from a chosen historical baseline (diverging scale).
 
The two panels share a single timeline slider. Between snapshots (discrete data
files on disk), the browser cross-fades precomputed PNG frames so the animation
stays smooth without re-hitting the server.
 
DATA MODEL
----------
NetCDF files follow this name convention:
 
    {indicator}_{scenario}_{model}_{ensemble}_change_{fp}_{bp}_{season}_NZ12km.nc
 
    indicator  e.g. TX, FD, PR, REC_TXx ...
    scenario   historical | ssp126 | ssp245 | ssp370 | ssp585
    model      GCM name (e.g. ACCESS-CM2)
    ensemble   realisation tag (e.g. r1i1p1f1)
    fp         future period  (fpYYYY-YYYY)  OR  'base' for historical
    bp         baseline period (bpYYYY-YYYY)
    season     ANN | DJF | MAM | JJA | SON
 
HIGH-LEVEL FLOW
---------------
    1. Sidebar form: user picks SSP / baseline / indicator / season / model.
    2. Build a timeline of snapshot periods for that combo.
    3. Load a pre-computed uncertainty cache (model spread across GCMs).
    4. Render each snapshot to a PNG on the server (results are cached to disk).
    5. Ship PNGs + per-point data to the browser; a JS player handles playback,
       blending, hover tooltips and click-to-pin time-series charts.
 
PERFORMANCE NOTES
-----------------
    * Expensive work (colour ranges, frames, uncertainty bands) is cached on
      disk under assets/ and in Streamlit's @st.cache_data / @st.cache_resource.
    * All blending happens in the client (Canvas 2D) — the server only ever
      sends one PNG per snapshot per panel.
 
SECURITY / DEPLOYMENT NOTES FOR REVIEWERS
-----------------------------------------
    * DATA_ROOT and REC_ROOT below point at NIWA-internal HPC paths (ESI).
      Both are overridable via environment variables (NZMAP_DATA_ROOT,
      NZMAP_REC_ROOT) so the module can be deployed without code changes.
    * No credentials, tokens, or API keys are embedded here. The paths are the
      only piece of internal info, and they are now fully externalised.
"""
 
# ── Standard library ────────────────────────────────────────────────────────
from pathlib import Path
import re
import io as _io
import os
import base64 as _base64
import json as _json
import hashlib
import pickle
 
# ── Third-party: always required ────────────────────────────────────────────
import numpy as np
import streamlit as st
 
 
# ============================================================================
# Streamlit page setup
# ============================================================================
st.set_page_config(page_title="NZ Climate Indicator Map", layout="wide")
 
# Compact theme — reduces default Streamlit padding/font so both map panels fit
# side-by-side on a 1280-wide screen without horizontal scroll.
st.markdown(
    """
<style>
html, body, [class*="css"]  { font-size: 14px !important; }
h1, h2, h3, h4 { font-size: 1.2rem !important; line-height: 1.2 !important; }
header[data-testid="stHeader"] { display: none; }
div.block-container { padding-top: 2.5rem; }
</style>
""",
    unsafe_allow_html=True,
)
 
 
# ============================================================================
# Optional dependencies
# ----------------------------------------------------------------------------
# Imported lazily with HAS_* flags so the page can fail fast with a clear
# message later (see "Guards" section) rather than a raw ImportError.
# ============================================================================
try:
    import xarray as xr
    HAS_XARRAY = True
except ImportError:
    HAS_XARRAY = False
 
try:
    import plotly.graph_objects as go  # noqa: F401  (used for type compat only)
    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False
 
try:
    import geopandas as gpd
    import shapely
    from shapely.ops import unary_union
    HAS_GEOPANDAS = True
except ImportError:
    HAS_GEOPANDAS = False
 
 
# ============================================================================
# Constants & configuration
# ============================================================================
 
# ── Data roots ──────────────────────────────────────────────────────────────
# NIWA-internal HPC paths. Overridable via env vars so the same code can run
# both on the research cluster and on a public deployment where data lives
# elsewhere (or is bundled/mounted differently).
DATA_ROOT = Path(os.environ.get(
    "NZMAP_DATA_ROOT",
    "/esi/project/niwa03712/ML_Downscaled_CMIP6/NIWA-REMS_indicators/dummy",
))
REC_ROOT = Path(os.environ.get(
    "NZMAP_REC_ROOT",
    "/esi/project/niwa03712/westphall/nz-climate-git/REC",
))
 
# ── Demo mode (no HPC data available) ────────────────────────────────────────
# Activated automatically when the real data paths are absent but
# test/demo_data/ exists.  Run  `python test/generate_demo_data.py`  once
# to populate the synthetic dataset, then commit test/demo_data/.
_DEMO_DATA_ROOT = Path("test/demo_data")
_DEMO_MODE      = not DATA_ROOT.exists() and _DEMO_DATA_ROOT.exists()
if _DEMO_MODE:
    DATA_ROOT = _DEMO_DATA_ROOT
    REC_ROOT  = _DEMO_DATA_ROOT
 
# ── Shapefiles (shipped with the repo) ──────────────────────────────────────
COASTLINE_SHP = Path("assets/coastlines/nz-coastlines-and-islands-polygons-topo-150k.shp")
REGIONS_SHP   = Path("assets/coastlines/nz-regional-council-boundaries-topo-150k.shp")
 
# ── Animation granularity ───────────────────────────────────────────────────
YEAR_STEP  = 3       # years per interpolated frame between real snapshots
SEASON_ANN = "ANN"   # season code for annual aggregation
 
# ── Human-readable labels ───────────────────────────────────────────────────
SEASON_LABELS = {
    "ANN": "Annual",
    "DJF": "Summer (DJF)",
    "MAM": "Autumn (MAM)",
    "JJA": "Winter (JJA)",
    "SON": "Spring (SON)",
}
 
INDICATOR_LABELS = {
    # Precipitation / dry
    "DD1mm":      "Dry days (<1 mm/day)",
    "PR":         "Precipitation",
    "R99p":       "99th-pct precipitation",
    "R99pVAL":    "99th-pct rainfall value",
    "R99pVALWet": "99th-pct rainfall (wet days)",
    "RR1mm":      "Rain days (>=1 mm)",
    "RR25mm":     "Heavy rain days (>=25 mm)",
    "Rx1day":     "Max 1-day precipitation",
    # Temperature
    "FD":   "Frost days (Tmin < 0 degC)",
    "TN":   "Mean minimum temperature",
    "TNn":  "Coldest night (annual min Tmin)",
    "TX":   "Mean maximum temperature",
    "TX25": "Days with Tmax > 25 degC",
    "TX30": "Days with Tmax > 30 degC",
    "TXx":  "Hottest day (annual max Tmax)",
    # Wind
    "sfcwind":  "Mean surface wind speed",
    "Wd10":     "Wind days >= 10 m/s",
    "Wd25":     "Wind days >= 25 m/s",
    "Wd99pVAL": "99th-pct wind speed value",
    "Wx1day":   "Max 1-day wind speed",
    # Record-chance indicators (non-exceedance probabilities)
    "REC_TXx":    "Hot record chance (annual max Tmax)",
    "REC_Rx1day": "Wet record chance (annual max rainfall)",
    "REC_Wx1day": "Wind record chance (annual max wind speed)",
    "REC_TNn":    "Cold record chance (annual min Tmin)",
}
 
INDICATOR_UNITS = {
    "DD1mm": "days/yr", "FD": "days/yr", "PR": "mm/yr",
    "R99p": "mm/yr", "R99pVAL": "mm/day", "R99pVALWet": "mm/day",
    "RR1mm": "days/yr", "RR25mm": "days/yr", "Rx1day": "mm/day",
    "sfcwind": "m/s",
    "TN": "°C", "TNn": "°C", "TX": "°C",
    "TX25": "days/yr", "TX30": "days/yr", "TXx": "°C",
    "Wd10": "days/yr", "Wd25": "days/yr", "Wd99pVAL": "m/s", "Wx1day": "m/s",
    "REC_TXx": "%", "REC_Rx1day": "%", "REC_Wx1day": "%", "REC_TNn": "%",
}
 
# Temperature indicators whose raw NetCDF values are in Kelvin.
# Detected later by a mean-value heuristic; kept here so the logic is centralised.
_KELVIN_INDICATORS = {"TX", "TXx", "TN", "TNn"}
 
# Indicators whose *absolute* values span many orders of magnitude →
# LogNorm gives clearer colour separation than linear.
_LOG_INDICATORS = {
    "PR", "RR1mm", "RR25mm", "Rx1day",
    "R99p", "R99pVAL", "R99pVALWet",
    "Wd10", "Wd25", "Wd99pVAL", "Wx1day",
    "sfcwind",
}
 
# Indicators whose *change* signal is heavy-tailed → SymLogNorm (symmetric log)
# handles both small widespread changes and extreme local changes.
_SYMLOG_CHANGE_INDICATORS = {
    "PR", "RR1mm", "RR25mm", "Rx1day",
    "R99p", "R99pVAL", "R99pVALWet",
}
 
# ── Scenario / baseline option lists ────────────────────────────────────────
SSP_OPTIONS = ["ssp126", "ssp245", "ssp370", "ssp585"]
SSP_LABELS = {
    "ssp126": "SSP1-2.6 — Low emissions",
    "ssp245": "SSP2-4.5 — Moderate emissions",
    "ssp370": "SSP3-7.0 — High emissions",
    "ssp585": "SSP5-8.5 — Very high emissions",
}
 
BP_OPTIONS = ["bp1995-2014", "bp1986-2005"]
BP_LABELS = {
    "bp1995-2014": "bp1995-2014  (recent baseline)",
    "bp1986-2005": "bp1986-2005  (earlier baseline)",
}
 
# REC (record-chance) indicators live in a separate data tree and have their
# own physical units for the absolute panel.
_REC_INDICATORS = {"REC_TXx", "REC_Rx1day", "REC_Wx1day", "REC_TNn"}
_REC_ABS_UNITS = {
    "REC_TXx":    "°C",
    "REC_Rx1day": "mm/day",
    "REC_Wx1day": "m/s",
    "REC_TNn":    "°C",
}
 
 
# ============================================================================
# File discovery helpers
# ----------------------------------------------------------------------------
# All of these walk the NetCDF directory tree and filter filenames by the
# tokenised naming convention documented at the top of the module.
# ============================================================================
 
def _indicator_root(indicator: str) -> Path:
    """Return the data root for an indicator (REC indicators live separately)."""
    return REC_ROOT if indicator in _REC_INDICATORS else DATA_ROOT
 
 
def fp_centre_year(fp_tag: str) -> float:
    """Mid-year of a period tag like 'fp2041-2060' or 'bp1995-2014'."""
    m = re.search(r"(\d{4})-(\d{4})", fp_tag)
    return (int(m.group(1)) + int(m.group(2))) / 2 if m else 0.0
 
 
def list_indicators(scenario: str) -> list[str]:
    """All indicator codes with at least one file under the given scenario."""
    results: set[str] = set()
    for root in (DATA_ROOT, REC_ROOT):
        base = root / scenario / "static_maps"
        if base.exists():
            results.update(d.name for d in base.iterdir() if d.is_dir())
    return sorted(results)
 
 
def discover_seasons(scenario: str, indicator: str) -> list[str]:
    """Seasons present on disk for a given scenario/indicator, ordered Annual→Spring."""
    folder = _indicator_root(indicator) / scenario / "static_maps" / indicator
    if not folder.exists():
        return [SEASON_ANN]
    seasons: set[str] = set()
    for f in folder.glob("*.nc"):
        for s in SEASON_LABELS:
            if f"_{s}_" in f.name:
                seasons.add(s)
    order = list(SEASON_LABELS)
    return sorted(seasons, key=lambda s: order.index(s) if s in order else 99) \
        or [SEASON_ANN]
 
 
def discover_fp_tags(scenario: str, indicator: str,
                     bp_tag: str, season: str) -> list[str]:
    """
    Discover all period tags for the given combination.
 
    Historical files use '_base_<bp>_' (we also use the bp tag as the fp tag
    so the timeline can treat 'historical' as a point in time).
    SSP files use '_change_<fp>_<bp>_'.
    """
    folder = _indicator_root(indicator) / scenario / "static_maps" / indicator
    if not folder.exists():
        return []
    tags: set[str] = set()
 
    if scenario == "historical":
        for f in folder.glob("*.nc"):
            if "_base_" in f.name and f"_{season}_" in f.name:
                m = re.search(r"_base_(bp\d{4}-\d{4})_", f.name)
                if m:
                    tags.add(m.group(1))
        return sorted(tags, key=lambda t: int(re.search(r"(\d{4})", t).group(1)))
 
    for f in folder.glob("*.nc"):
        if f"_{bp_tag}_" in f.name and f"_{season}_" in f.name:
            m = re.search(r"_change_(fp\d{4}-\d{4})_", f.name)
            if m:
                tags.add(m.group(1))
    return sorted(tags, key=lambda t: int(re.search(r"fp(\d{4})", t).group(1)))
 
 
def discover_models(scenario: str, indicator: str,
                    bp_tag: str, season: str) -> list[str]:
    """List GCM model keys available for this combination, parsed out of filenames."""
    folder = _indicator_root(indicator) / scenario / "static_maps" / indicator
    if not folder.exists():
        return []
 
    # Regex captures the model token between the scenario and the ensemble tag.
    # Ensemble tag matches rXiYpZ / rXiYpZfW / rXiYfZ style realisations.
    pat = re.compile(
        rf"^{re.escape(indicator)}_{re.escape(scenario)}_(.+?)_[ri]\d+[ip]\d+[pf]\d+(?:f\d+)?_"
    )
    models: set[str] = set()
    for f in folder.glob("*.nc"):
        if f"_{season}_" not in f.name:
            continue
        if scenario == "historical":
            if "_base_" not in f.name or f"_{bp_tag}_" not in f.name:
                continue
        else:
            if "_change_" not in f.name or f"_{bp_tag}_" not in f.name:
                continue
        m = pat.match(f.name)
        if m:
            models.add(m.group(1))
    return sorted(models)
 
 
def find_nc_files(scenario: str, indicator: str,
                  fp_tag: str, bp_tag: str, season: str,
                  model_key: str | None = None) -> list[Path]:
    """Return all NetCDF files matching the given query, optionally filtered by model."""
    folder = _indicator_root(indicator) / scenario / "static_maps" / indicator
    if not folder.exists():
        return []
 
    model_pat = (
        re.compile(r"_" + re.escape(model_key) + r"_[ri]\d+[ip]\d+[pf]\d+")
        if model_key else None
    )
 
    def _ok(f: Path) -> bool:
        if model_pat and not model_pat.search(f.name):
            return False
        return f"_{season}_" in f.name
 
    if scenario == "historical":
        return sorted(
            f for f in folder.glob("*.nc")
            if f"_base_{fp_tag}_" in f.name and _ok(f)
        )
    return sorted(
        f for f in folder.glob("*.nc")
        if f"_{fp_tag}_" in f.name
        and f"_{bp_tag}_" in f.name
        and _ok(f)
    )
 
 
# ============================================================================
# Data loaders
# ----------------------------------------------------------------------------
# Both loaders read every matching NetCDF, strip fill/scale/offset attrs, and
# stack the 2D arrays. @st.cache_data memoises per-parameter-tuple so that
# switching back to a previous selection is instant.
# ============================================================================
 
@st.cache_data(show_spinner=False)
def load_ensemble_mean(scenario: str, indicator: str,
                       fp_tag: str, bp_tag: str,
                       season: str,
                       model_key: str | None = None,
                       var_name: str | None = None) -> tuple:
    """
    Load all matching files and return (mean_array, lat, lon, n_files).
 
    When `model_key` is None, the mean is over all GCMs (the ensemble mean).
    When set, typically only one file matches — the "mean" is that file.
 
    Returns (None, None, None, 0) on any failure, which callers treat as
    "no data for this combination".
    """
    files = find_nc_files(scenario, indicator, fp_tag, bp_tag, season, model_key)
    if not files:
        return None, None, None, 0
 
    arrays: list[np.ndarray] = []
    lats = lons = None
 
    for f in files:
        try:
            ds = xr.open_dataset(f, engine="netcdf4", mask_and_scale=False)
 
            # NetCDF files from different tools use different coord names —
            # find the first usable lat/lon pair (case-insensitive).
            lat_names = [c for c in list(ds.coords) + list(ds.dims)
                         if c.lower() in ("lat", "latitude", "y", "rlat")]
            lon_names = [c for c in list(ds.coords) + list(ds.dims)
                         if c.lower() in ("lon", "longitude", "x", "rlon")]
            if not lat_names or not lon_names:
                ds.close(); continue
            lat_name, lon_name = lat_names[0], lon_names[0]
 
            data_vars = [v for v in ds.data_vars
                         if v.lower() not in ("lat", "lon", "time")]
            if not data_vars:
                ds.close(); continue
 
            # Use requested variable if specified, else the first real data var.
            chosen_var = var_name if (var_name and var_name in ds.data_vars) else data_vars[0]
            da = ds[chosen_var]
 
            # Drop any leading dimensions (time, bounds, ...) — take slice 0.
            for dim in [d for d in da.dims if d not in (lat_name, lon_name)]:
                da = da.isel({dim: 0})
            arr = da.values.astype(float)
 
            # Manual decoding of fill/scale/offset, since mask_and_scale=False
            # above (that flag is off so we can detect the raw fill value).
            if hasattr(da, "attrs"):
                fv = da.attrs.get("_FillValue", da.attrs.get("missing_value"))
                if fv is not None:
                    arr[arr == float(fv)] = np.nan
                scale  = float(da.attrs.get("scale_factor", 1.0))
                offset = float(da.attrs.get("add_offset",   0.0))
                if scale != 1.0 or offset != 0.0:
                    arr = arr * scale + offset
 
            if lats is None:
                lats = ds[lat_name].values
                lons = ds[lon_name].values
            arrays.append(arr)
            ds.close()
        except Exception:
            # Swallow per-file errors — one bad file shouldn't kill the ensemble.
            continue
 
    if not arrays:
        return None, None, None, 0
 
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # all-NaN slice warnings
        result = np.nanmean(np.stack(arrays, 0), 0)
    return result, lats, lons, len(arrays)
 
 
@st.cache_data(show_spinner=False)
def load_model_range(scenario: str, indicator: str,
                     fp_tag: str, bp_tag: str, season: str) -> tuple:
    """
    Load all models for a scenario/period and return percentile bands.
 
    Returns (min, p5, p25, p75, p95, max, n_models) across the GCM axis, or
    all-None when there are fewer than 2 models (no meaningful spread).
    """
    if scenario == "historical":
        return None, None, None, None, None, None, 0
 
    files = find_nc_files(scenario, indicator, fp_tag, bp_tag, season, None)
    if len(files) < 2:
        return None, None, None, None, None, None, len(files)
 
    arrays: list[np.ndarray] = []
    for f in files:
        try:
            # ── Mirror the read logic in load_ensemble_mean. Kept as its own
            #    function (rather than shared) because the stacking semantics
            #    differ: here we need the raw per-model stack, not a mean.
            ds = xr.open_dataset(f, engine="netcdf4", mask_and_scale=False)
            lat_names = [c for c in list(ds.coords) + list(ds.dims)
                         if c.lower() in ("lat", "latitude", "y", "rlat")]
            lon_names = [c for c in list(ds.coords) + list(ds.dims)
                         if c.lower() in ("lon", "longitude", "x", "rlon")]
            if not lat_names or not lon_names:
                ds.close(); continue
            lat_name, lon_name = lat_names[0], lon_names[0]
            data_vars = [v for v in ds.data_vars
                         if v.lower() not in ("lat", "lon", "time")]
            if not data_vars:
                ds.close(); continue
            da = ds[data_vars[0]]
            for dim in [d for d in da.dims if d not in (lat_name, lon_name)]:
                da = da.isel({dim: 0})
            arr = da.values.astype(float)
            if hasattr(da, "attrs"):
                fv = da.attrs.get("_FillValue", da.attrs.get("missing_value"))
                if fv is not None:
                    arr[arr == float(fv)] = np.nan
                scale  = float(da.attrs.get("scale_factor", 1.0))
                offset = float(da.attrs.get("add_offset",   0.0))
                if scale != 1.0 or offset != 0.0:
                    arr = arr * scale + offset
            arrays.append(arr)
            ds.close()
        except Exception:
            continue
 
    if not arrays:
        return None, None, None, None, None, None, 0
 
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        stk = np.stack(arrays, 0)
        return (
            np.nanmin(stk, 0),
            np.nanpercentile(stk, 5,  axis=0),
            np.nanpercentile(stk, 25, axis=0),
            np.nanpercentile(stk, 75, axis=0),
            np.nanpercentile(stk, 95, axis=0),
            np.nanmax(stk, 0),
            len(arrays),
        )
 
 
def build_timeline(indicator: str, bp_tag: str,
                   ssp: str, season: str) -> list[tuple]:
    """
    Stitch the historical baseline + future periods into a single ordered
    timeline of snapshots. Each entry is:
 
        (label, scenario, fp_tag, bp_tag, centre_year)
 
    The historical entry is filtered to only include fp == bp_tag so we don't
    duplicate baselines across the two bp options.
    """
    snapshots: list[tuple] = []
 
    # Historical slice: use only the period that matches the selected baseline
    all_hist = discover_fp_tags("historical", indicator, bp_tag, season)
    for period in [p for p in all_hist if p == bp_tag]:
        yr = fp_centre_year(period)
        label = f"Historical {period.replace('bp', '').replace('-', '–')}"
        snapshots.append((label, "historical", period, period, yr))
 
    # Future slices under the chosen SSP
    for fp in discover_fp_tags(ssp, indicator, bp_tag, season):
        yr = fp_centre_year(fp)
        label = fp.replace("fp", "").replace("-", "–")
        snapshots.append((label, ssp, fp, bp_tag, yr))
 
    return sorted(snapshots, key=lambda x: x[4])
 
 
# ============================================================================
# Colour ranges (change + absolute)
# ----------------------------------------------------------------------------
# Colour ranges are pre-computed once and stashed in JSON under assets/ so the
# diverging / sequential bars stay stable as the user switches snapshots. If
# the JSON is missing for an indicator, we fall back to a live computation
# (slow, surfaces a toast).
# ============================================================================
 
# ── Change scale (diverging, ±half_range) ───────────────────────────────────
_COLOR_RANGES_PATH = Path(__file__).parent.parent / "assets/color_ranges" / "color_ranges.json"
_PRECOMPUTED_RANGES: dict[str, float] = {}
if _COLOR_RANGES_PATH.exists():
    try:
        with open(_COLOR_RANGES_PATH) as _f:
            _PRECOMPUTED_RANGES = _json.load(_f)
    except Exception as e:
        st.warning(f"Could not load color_ranges.json: {e}")
 
# ── Absolute scale (sequential, [vmin, vmax]) ───────────────────────────────
_ABS_COLOR_RANGES_PATH = Path(__file__).parent.parent / "assets/color_ranges" / "abs_color_ranges.json"
_PRECOMPUTED_ABS_RANGES: dict[str, dict] = {}
if _ABS_COLOR_RANGES_PATH.exists():
    try:
        with open(_ABS_COLOR_RANGES_PATH) as _f:
            _PRECOMPUTED_ABS_RANGES = _json.load(_f)
    except Exception as e:
        st.warning(f"Could not load abs_color_ranges.json: {e}")
 
 
@st.cache_data(show_spinner=False)
def compute_color_range(indicator: str) -> float:
    """
    Half-range for the change (Δ) colour scale. Symmetric: the bar goes
    -value..+value. Uses the pre-computed JSON when available.
    """
    # Re-read JSON each call so edits to the file during dev are picked up;
    # the @st.cache_data decorator caches the *function result*, not the file.
    ranges: dict[str, float] = {}
    if _COLOR_RANGES_PATH.exists():
        with open(_COLOR_RANGES_PATH) as f:
            ranges = _json.load(f)
    if indicator in ranges:
        return ranges[indicator]
    if indicator in _PRECOMPUTED_RANGES:
        return _PRECOMPUTED_RANGES[indicator]
 
    # ── Live fallback — iterate every SSP × bp × season × fp and compute
    #    the percentile of |values| across the pooled distribution.
    st.toast(f"'{indicator}' not in color_ranges.json — computing live.", icon="⚠️")
    all_vals: list[np.ndarray] = []
    for ssp_opt in SSP_OPTIONS:
        for bp in BP_OPTIONS:
            for season in discover_seasons("historical", indicator):
                for fp in discover_fp_tags(ssp_opt, indicator, bp, season):
                    data, _, _, _ = load_ensemble_mean(ssp_opt, indicator, fp, bp, season, None)
                    if data is not None:
                        finite = data[np.isfinite(data)].ravel()
                        if len(finite):
                            all_vals.append(finite)
    if not all_vals:
        return 1.0
    combined = np.concatenate(all_vals)
    # Symlog indicators need a wider pct cutoff — the tail is the whole point.
    pct = 99 if indicator in _SYMLOG_CHANGE_INDICATORS else 98
    return max(float(np.nanpercentile(np.abs(combined), pct)), 1e-6)
 
 
# Count-like indicators must floor at zero — a negative "days per year" is
# nonsensical and would let LogNorm misbehave.
_ZERO_FLOOR_INDICATORS = {
    "WD10", "Wd10", "WD25", "Wd25",
    "TX25", "TX30", "FD", "RR25mm",
    "REC_Rx1day", "REC_TXx", "REC_Wx1day", "REC_TNn",
}
 
 
@st.cache_data(show_spinner=False)
def compute_abs_color_range(indicator: str) -> tuple[float, float]:
    """
    (vmin, vmax) for the absolute-value panel. REC indicators use hand-tuned
    physical ranges; everything else reads from abs_color_ranges.json with a
    live fallback that pools (baseline + change) values across all scenarios.
    """
    # Read JSON at call time (see note in compute_color_range).
    abs_ranges: dict[str, dict] = {}
    if _ABS_COLOR_RANGES_PATH.exists():
        with open(_ABS_COLOR_RANGES_PATH) as f:
            abs_ranges = _json.load(f)
    if indicator in abs_ranges:
        r = abs_ranges[indicator]
        vmin, vmax = float(r["vmin"]), float(r["vmax"])
        # Note: we fall through to the REC/precomputed blocks below if the
        # JSON entry is a REC indicator or needs zero-flooring. The original
        # control flow preserves JSON values for non-REC, non-floored cases
        # implicitly — keep in mind for the review.
 
    # REC indicators: abs panel shows the running record in physical units,
    # not a percentile. These ranges are hand-chosen for NZ.
    if indicator in _REC_INDICATORS:
        if indicator == "REC_TXx":    return 20.0, 45.0    # °C
        if indicator == "REC_TNn":    return -15.0, 10.0   # °C
        if indicator == "REC_Rx1day": return 50.0, 400.0   # mm/day
        if indicator == "REC_Wx1day": return 10.0, 40.0    # m/s
        return 0.0, 1.0
 
    if indicator in _PRECOMPUTED_ABS_RANGES:
        r = _PRECOMPUTED_ABS_RANGES[indicator]
        vmin, vmax = float(r["vmin"]), float(r["vmax"])
        if indicator in _ZERO_FLOOR_INDICATORS:
            vmin = 0.0
        if indicator in _LOG_INDICATORS:
            vmin = max(vmin, 1e-3)  # LogNorm cannot accept zero/negative
        return vmin, vmax
 
    # ── Live fallback — pool baseline + (baseline + change) across every
    #    bp × season × ssp combination, convert K→°C where applicable, and
    #    take percentiles.
    st.toast(f"'{indicator}' not in abs_color_ranges.json — computing live.", icon="⚠️")
    all_vals: list[np.ndarray] = []
    for bp in BP_OPTIONS:
        for season in discover_seasons("historical", indicator):
            baseline = None
            # Historical baseline: there is one period where fp == bp.
            for fp in discover_fp_tags("historical", indicator, bp, season):
                if fp == bp:
                    data, _, _, _ = load_ensemble_mean(
                        "historical", indicator, fp, bp, season, None
                    )
                    if data is not None:
                        baseline = data.copy()
                        arr = baseline[np.isfinite(baseline)].ravel()
                        if indicator in _KELVIN_INDICATORS and len(arr) and np.nanmean(arr) > 200:
                            arr = arr - 273.15
                        all_vals.append(arr)
                    break
            # Future: abs = baseline + change
            for ssp_opt in SSP_OPTIONS:
                for fp in discover_fp_tags(ssp_opt, indicator, bp, season):
                    change, _, _, _ = load_ensemble_mean(
                        ssp_opt, indicator, fp, bp, season, None
                    )
                    if change is None or baseline is None:
                        continue
                    abs_data = baseline + change
                    arr = abs_data[np.isfinite(abs_data)].ravel()
                    if indicator in _KELVIN_INDICATORS and len(arr) and np.nanmean(arr) > 200:
                        arr = arr - 273.15
                    all_vals.append(arr)
 
    if not all_vals:
        return 0.0, 1.0
    combined = np.concatenate(all_vals)
    is_log = indicator in _LOG_INDICATORS
    pct_lo = 1  if is_log else 2
    pct_hi = 99 if is_log else 98
    vmin = float(np.nanpercentile(combined, pct_lo))
    vmax = float(np.nanpercentile(combined, pct_hi))
    if is_log:
        vmin = max(vmin, 1e-3)
    if vmax - vmin < 1e-6:
        vmax = vmin + 1.0
    return vmin, vmax
 
 
# ============================================================================
# Colourmap selection
# ----------------------------------------------------------------------------
# Each indicator maps to both a diverging map (for the change panel) and a
# sequential map (for the absolute panel). Kept as lookup functions rather
# than dict constants because some share the same map logic.
# ============================================================================
 
def colorscale_for(indicator: str) -> str:
    """Diverging colour map for the Δ (change) panel."""
    if indicator in {"TX", "TXx", "TX25", "TX30", "TN", "TNn"}:
        return "RdBu_r"
    if indicator == "FD":
        return "RdBu"
    if indicator in {"PR", "RR1mm", "RR25mm", "Rx1day",
                     "R99p", "R99pVAL", "R99pVALWet"}:
        return "BrBG"
    if indicator == "DD1mm":
        return "BrBG_r"
    if indicator in {"sfcwind", "Wd10", "Wd25", "Wd99pVAL", "Wx1day"}:
        return "PuOr"
    # REC indicators show Δ record probability — more records = red, fewer = green.
    if indicator in _REC_INDICATORS:
        return "RdYlGn_r"
    return "RdBu_r"
 
 
def colorscale_abs_for(indicator: str) -> str:
    """Sequential colour map for the absolute-value panel."""
    if indicator == "FD":                                return "Blues"
    if indicator in {"TN", "TNn"}:                       return "RdYlBu_r"
    if indicator in {"TX", "TXx", "TX25", "TX30"}:       return "YlOrRd"
    if indicator in {"PR", "RR1mm", "RR25mm", "Rx1day",
                     "R99p", "R99pVAL", "R99pVALWet"}:   return "YlGnBu"
    if indicator in {"sfcwind", "Wd10", "Wd25",
                     "Wd99pVAL", "Wx1day"}:              return "Purples"
    if indicator == "DD1mm":                             return "YlOrBr"
    # REC indicators mirror their parent indicator's palette.
    if indicator == "REC_TXx":    return "YlOrRd"
    if indicator == "REC_TNn":    return "RdYlBu_r"
    if indicator == "REC_Rx1day": return "YlGnBu"
    if indicator == "REC_Wx1day": return "Purples"
    return "viridis"
 
 
def _mpl_cmap(plotly_name: str):
    """Resolve a plotly-ish colour-scale name to a matplotlib colormap."""
    import matplotlib
    if not plotly_name:
        return matplotlib.colormaps["RdBu_r"]
    for name in (plotly_name, plotly_name.lower()):
        try:
            return matplotlib.colormaps[name]
        except KeyError:
            continue
    return matplotlib.colormaps["RdBu_r"]
 
 
# Indicators whose absolute values already live in a narrow range where
# LogNorm squashes meaningful variation. Linear norm gives better contrast.
_LINEAR_ABS_INDICATORS = {
    "RR1mm", "RR25mm", "Rx1day",
    "R99p", "R99pVAL", "R99pVALWet",
    "Wd10", "Wd25", "Wd99pVAL", "Wx1day",
    "sfcwind",
    "REC_Rx1day", "REC_TXx", "REC_Wx1day", "REC_TNn_ABS",
}
 
 
def _make_norm(indicator: str, vmin: float, vmax: float, is_change: bool):
    """
    Build the matplotlib Normalize subclass appropriate for this indicator
    and panel. Central helper so colour-bar rendering and snapshot rendering
    stay consistent.
    """
    import matplotlib.colors as mcolors
 
    if is_change and indicator in _SYMLOG_CHANGE_INDICATORS:
        # Symlog: linear near zero (linthresh), log farther out. 10% of vmax
        # as linthresh keeps near-zero detail without swallowing extremes.
        linthresh = max(abs(vmax) * 0.10, 1e-3)
        return mcolors.SymLogNorm(
            linthresh=linthresh, linscale=0.5,
            vmin=vmin, vmax=vmax, base=10,
        )
 
    if (not is_change
            and indicator in _LOG_INDICATORS
            and indicator not in _LINEAR_ABS_INDICATORS):
        safe_vmin = max(vmin, 1e-3)  # LogNorm rejects zero/negative
        return mcolors.LogNorm(vmin=safe_vmin, vmax=vmax)
 
    return mcolors.Normalize(vmin=vmin, vmax=vmax)

# Part 2/3
#
# This part covers:
#   1. Disk-backed caches for rendered frames and uncertainty bands
#   2. Virtual-frame timeline (interpolates between real snapshots)
#   3. Geometry loaders (coastline polygon, NZ SVG outline, borders GeoJSON)
#   4. The main snapshot renderer — turns 2D data into RGBA PNGs
#   5. The colourbar renderer
#
# PART 2 SECURITY NOTES FOR REVIEWERS
# -----------------------------------
# * The on-disk caches (assets/frame_cache, assets/uncertainty_cache) are
#   loaded with `pickle.load`. Pickle can execute arbitrary code on load.
#   This is safe as long as *only this app writes to those folders* on the
#   machine running it. If the app is ever deployed somewhere a third party
#   can drop files into assets/, switch to a safer format (e.g. npz, or
#   json+base64 for the PNGs).
# * No network I/O here. External tile URLs live in Part 3.


# ============================================================================
# Frame cache (rendered PNG snapshots)
# ----------------------------------------------------------------------------
# prerender_snapshots() is the hot path — minutes of CPU per selection. Its
# inputs are deterministic (same params → same PNGs), so we hash the params
# into a filename and pickle the result. Keys include colour range and log
# mode, because changing either invalidates the PNG.
# ============================================================================
FRAME_CACHE_DIR = Path("assets/frame_cache")
FRAME_CACHE_DIR.mkdir(parents=True, exist_ok=True)


def _frame_cache_key(indicator: str, ssp: str, bp_tag: str, season: str,
                     model_key, colorscale: str,
                     vmin: float, vmax: float,
                     log_mode: str = "linear") -> str:
    """MD5 of every input that can change what the rendered PNG looks like."""
    parts = (f"{indicator}|{ssp}|{bp_tag}|{season}|"
             f"{model_key or 'ensemble'}|{colorscale}|"
             f"{vmin:.6f}|{vmax:.6f}|{log_mode}")
    return hashlib.md5(parts.encode()).hexdigest()


def _log_mode(indicator: str, is_change: bool) -> str:
    """
    Descriptor of which Normalize subclass we'll end up using.
    Mirrors the branching in _make_norm() — kept in sync manually.
    """
    if is_change and indicator in _SYMLOG_CHANGE_INDICATORS:
        return "symlog"
    if not is_change and indicator in _LOG_INDICATORS:
        return "log"
    return "linear"


def _load_frame_cache(key: str):
    """Return the cached tuple for `key`, or None if missing/corrupt."""
    path = FRAME_CACHE_DIR / f"{key}.pkl"
    if not path.exists():
        return None
    try:
        with open(path, "rb") as f:
            return pickle.load(f)  # NOTE: see security block at top of Part 2
    except Exception:
        return None


def _save_frame_cache(key: str, payload: tuple) -> None:
    """Write `payload` under `key`. No locking — last writer wins."""
    path = FRAME_CACHE_DIR / f"{key}.pkl"
    with open(path, "wb") as f:
        pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)


# ============================================================================
# Uncertainty cache (model-spread bands)
# ----------------------------------------------------------------------------
# Populated offline by precompute_uncertainty.py, which walks every SSP × bp ×
# season × fp and stores per-snapshot percentiles/min/max across the GCM axis.
# The app is read-only here; if the cache file is missing the page aborts
# (there is also a live-compute fallback further down, but it's slow).
# ============================================================================
UNCERTAINTY_CACHE_DIR = (
    _DEMO_DATA_ROOT / "uncertainty_cache"
    if _DEMO_MODE else
    Path("assets/uncertainty_cache")
)


def _uncertainty_cache_key(indicator: str, ssp: str,
                           bp_tag: str, season: str) -> str:
    """
    Matches precompute_uncertainty.py byte-for-byte — any change to the format
    of this string must be mirrored in the precompute script or caches will
    never hit.
    """
    parts = f"{indicator}|{ssp}|{bp_tag}|{season}"
    return hashlib.md5(parts.encode()).hexdigest()


@st.cache_data(show_spinner=False)
def load_uncertainty_cache(indicator: str, ssp: str,
                           bp_tag: str, season: str) -> dict | None:
    """Load the pre-computed uncertainty bands for this selection, or None."""
    key  = _uncertainty_cache_key(indicator, ssp, bp_tag, season)
    path = UNCERTAINTY_CACHE_DIR / f"{key}.pkl"
    if not path.exists():
        return None
    try:
        with open(path, "rb") as f:
            return pickle.load(f)  # NOTE: see security block at top of Part 2
    except Exception:
        return None


# ============================================================================
# Virtual frame timeline
# ----------------------------------------------------------------------------
# Given a list of real snapshot years (one per NetCDF period), produce a
# denser list of "frame years" at YEAR_STEP spacing. The browser blends
# consecutive snapshots linearly along this dense axis.
# ============================================================================
def compute_frame_timeline(snap_years: list[float],
                           year_step: float) -> tuple[list[float], list[int]]:
    """
    Returns (frame_years, snap_frame_idx).

    frame_years     — every animation frame's centre year (floats)
    snap_frame_idx  — for each real snapshot, the index into frame_years
                      where that snapshot sits exactly (t = 0 in the blend)

    Example with two snapshots at 2005 and 2020 and year_step=3:
        frame_years    = [2005, 2008, 2011, 2014, 2017, 2020]
        snap_frame_idx = [0, 5]
    """
    frame_years:    list[float] = []
    snap_frame_idx: list[int]   = []
    fi = 0
    for i in range(len(snap_years) - 1):
        snap_frame_idx.append(fi)
        ya, yb  = snap_years[i], snap_years[i + 1]
        # Round so exactly-aligned spacings don't produce a phantom extra step.
        n_steps = max(1, round((yb - ya) / year_step))
        for step in range(n_steps):
            t = step / n_steps
            frame_years.append(ya + t * (yb - ya))
            fi += 1
    # Final snapshot doesn't get its own loop — append it explicitly.
    snap_frame_idx.append(fi)
    frame_years.append(snap_years[-1])
    return frame_years, snap_frame_idx


# ============================================================================
# Geometry loaders — coastline, NZ outline (for loading screen), borders
# ----------------------------------------------------------------------------
# All three use @st.cache_resource because the loaded objects hold C-level
# state (GEOS geometries) that @st.cache_data would try to pickle.
# ============================================================================

@st.cache_resource(show_spinner=False)
def load_coastline_polygon():
    """
    Load the NZ coastline as a single (Multi)Polygon in EPSG:4326.
    Returns None when geopandas is unavailable, the file is missing, or
    bounds look wrong — callers then fall back to a KDTree land mask.
    """
    if not HAS_GEOPANDAS:
        return None
    if not COASTLINE_SHP.exists():
        st.warning(
            f"Coastline shapefile not found at `{COASTLINE_SHP}`. "
            "Falling back to KDTree land mask."
        )
        return None
    try:
        gdf = gpd.read_file(COASTLINE_SHP)
        # Normalise to WGS84 — data ships in NZ projection variants sometimes.
        if gdf.crs is not None and gdf.crs.to_epsg() != 4326:
            gdf = gdf.to_crs(epsg=4326)
        elif gdf.crs is None:
            gdf = gdf.set_crs(epsg=4326, allow_override=True)

        # Sanity check: bounds must be roughly earth-shaped. Catches the case
        # where the file was converted with the wrong datum.
        bds = gdf.total_bounds
        if not (-200 < bds[0] < 200 and -90 < bds[1] < 90):
            st.warning(f"Coastline bounds look wrong ({bds}). Using KDTree fallback.")
            return None

        poly_geoms = [
            g for g in gdf.geometry
            if g is not None and g.geom_type in ("Polygon", "MultiPolygon")
        ]
        if not poly_geoms:
            return None
        return unary_union(poly_geoms)
    except Exception as exc:
        st.warning(f"Could not load coastline shapefile: {exc}. Using KDTree fallback.")
        return None


@st.cache_resource(show_spinner=False)
def nz_loader_svg_data() -> dict | None:
    """
    Build a simplified NZ outline as an SVG <path d=...> string plus a
    viewBox. Used for the whimsical filling-up-with-water loading screen
    (see build_loading_screen_html in Part 3).
    """
    from shapely.geometry import MultiPolygon

    poly = load_coastline_polygon()
    if poly is None:
        return None

    # Chatham Islands sit east of the antimeridian and wreck the viewBox.
    # Keep only polygons with a centroid in NZ's main longitude band.
    if poly.geom_type == "MultiPolygon":
        kept = [p for p in poly.geoms
                if 160 < p.centroid.x < 180 and p.area > 0.01]
        if kept:
            poly = MultiPolygon(kept) if len(kept) > 1 else kept[0]

    minx, miny, maxx, maxy = poly.bounds
    y_flip = maxy + miny  # flip y axis so SVG's top-left origin makes sense

    # Simplify for SVG path compactness (bounds are captured pre-simplify
    # so the loader frame doesn't shift when the tolerance changes).
    simplified = poly.simplify(0.01, preserve_topology=True)

    def ring_d(coords, y_flip):
        parts = []
        for i, xy in enumerate(coords):
            x, y = float(xy[0]), float(xy[1])
            cmd = "M" if i == 0 else "L"
            parts.append(f"{cmd}{x:.3f},{(y_flip - y):.3f}")
        parts.append("Z")
        return "".join(parts)

    def poly_d(p, y_flip):
        parts = [ring_d(p.exterior.coords, y_flip)]
        for interior in p.interiors:
            parts.append(ring_d(interior.coords, y_flip))
        return "".join(parts)

    if simplified.geom_type == "MultiPolygon":
        d = "".join(poly_d(p, y_flip) for p in simplified.geoms)
    else:
        d = poly_d(simplified, y_flip)

    pad = 0.2
    w = (maxx - minx) + 2 * pad
    h = (maxy - miny) + 2 * pad
    vb = f"{minx - pad:.3f} {miny - pad:.3f} {w:.3f} {h:.3f}"

    return {"d": d, "viewBox": vb}


@st.cache_resource(show_spinner=False)
def load_borders_geojson() -> dict:
    """
    Return {"country": <str|None>, "regions": <str|None>} where each value
    is a GeoJSON string ready to be inlined into the HTML player.
    Simplified to SIMPLIFY_TOL degrees to keep the payload small.
    """
    out = {"country": None, "regions": None}
    if not HAS_GEOPANDAS:
        return out

    SIMPLIFY_TOL = 0.005  # ~500m at NZ latitudes — invisible at page zoom

    coast = load_coastline_polygon()
    if coast is not None:
        try:
            out["country"] = _json.dumps(
                coast.simplify(SIMPLIFY_TOL, preserve_topology=True)
                     .__geo_interface__
            )
        except Exception:
            pass

    if REGIONS_SHP.exists():
        try:
            gdf = gpd.read_file(REGIONS_SHP)
            if gdf.crs is not None and gdf.crs.to_epsg() != 4326:
                gdf = gdf.to_crs(epsg=4326)
            elif gdf.crs is None:
                gdf = gdf.set_crs(epsg=4326, allow_override=True)
            gdf["geometry"] = gdf.geometry.simplify(SIMPLIFY_TOL, preserve_topology=True)
            out["regions"] = gdf.to_json()
        except Exception:
            pass

    return out


# ============================================================================
# Snapshot renderer (the expensive one)
# ----------------------------------------------------------------------------
# Takes a stack of (n_snaps, n_points) values at irregular (lat, lon) points
# and produces n_snaps base64 PNGs ready for the browser Canvas blend.
#
# Pipeline per snapshot:
#   1. Build a regular Web-Mercator pixel grid covering the map extent.
#   2. Nearest-neighbour interpolate scattered values onto that grid.
#   3. Gaussian-smooth, masking NaNs so they don't bleed.
#   4. Overlay a linear (Delaunay) interpolation where it's well-defined —
#      gives sharper results inside the data convex hull.
#   5. Flood-fill colour out to the coastline (no "shrinkage" at the coast).
#   6. Apply a hard coastline alpha mask.
#
# Heavy imports (matplotlib, PIL, scipy.*) live inside the function so this
# module can be imported even when those are unavailable.
#
# REVIEW FLAGS
# ------------
# * In the original file the fallback branch of the second block referenced
#   an undefined `mask_threshold_rad`. Renamed to `mask_threshold` here —
#   that path was a NameError if ever hit.
# ============================================================================
@st.cache_data(show_spinner=False, max_entries=12)
def prerender_snapshots(
    lat_v: np.ndarray,
    lon_v: np.ndarray,
    snap_data_stacked: np.ndarray,
    vmin: float,
    vmax: float,
    colorscale: str,
    indicator: str = "",
    is_change: bool = False,
) -> tuple:
    """
    Return (snap_b64_list, lat_min, lat_max, lon_min, lon_max, mask_threshold_deg).

    snap_b64_list         — one base64-encoded PNG per input snapshot
    lat_min/.../lon_max   — bounds the PNGs cover (passed to Leaflet)
    mask_threshold_deg    — point spacing (used client-side for hover snap-to)
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.colors as mcolors  # noqa: F401 (used via _make_norm)
    from scipy.spatial import Delaunay, KDTree
    from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator
    from scipy.ndimage import distance_transform_edt
    from scipy.ndimage import gaussian_filter as _gf
    from PIL import Image as _PILImage

    cmap = _mpl_cmap(colorscale)
    norm = _make_norm(indicator, vmin, vmax, is_change)
    n_snaps = snap_data_stacked.shape[0]

    # ── Pixel grid in Web-Mercator ──────────────────────────────────────────
    # Mercator preserves north-up rectangles, which makes the Canvas blend
    # in the browser a simple drawImage() rather than a per-pixel warp.
    def merc(lat_deg):
        return np.log(np.tan(np.pi / 4 + np.deg2rad(lat_deg) / 2))

    lat_min = float(np.nanmin(lat_v)) - 0.3
    lat_max = float(np.nanmax(lat_v)) + 0.3
    lon_min = float(np.nanmin(lon_v)) - 0.3
    lon_max = float(np.nanmax(lon_v)) + 0.3

    lon_v_rad   = np.deg2rad(lon_v)
    merc_v      = merc(lat_v)
    lon_min_rad = np.deg2rad(lon_min);  lon_max_rad = np.deg2rad(lon_max)
    merc_min    = merc(lat_min);        merc_max    = merc(lat_max)

    # Choose an output size that matches the Mercator aspect ratio.
    fig_w = 5.0
    fig_h = fig_w * ((merc_max - merc_min) / (lon_max_rad - lon_min_rad))
    DPI   = 200
    out_w = int(fig_w * DPI)
    out_h = int(fig_h * DPI)

    xi = np.linspace(lon_min_rad, lon_max_rad, out_w)
    yi = np.linspace(merc_min,    merc_max,    out_h)
    xi_grid, yi_grid = np.meshgrid(xi, yi)

    # Equivalent degree grid (for shapely contains_xy and KDTree fallback).
    xi_deg = np.rad2deg(xi_grid)
    yi_deg = np.rad2deg(2.0 * np.arctan(np.exp(yi_grid)) - np.pi / 2)

    points_merc = np.column_stack([lon_v_rad, merc_v])
    tri = Delaunay(points_merc)  # noqa: F841 (kept as a sanity check)

    # Estimate typical data-point spacing so the KDTree-fallback mask can
    # decide what counts as "near a data point".
    kd = KDTree(points_merc)
    sample_n   = min(2000, len(points_merc))
    rng        = np.random.default_rng(0)
    sample_idx = rng.choice(len(points_merc), sample_n, replace=False)
    nn_dists, _ = kd.query(points_merc[sample_idx], k=2)
    grid_spacing_rad = float(np.median(nn_dists[:, 1]))
    mask_threshold   = grid_spacing_rad * 1.1

    # ── Land-mask block ────────────────────
    # Buffers the coastline outward by ~0.2 pixels so boundary pixels (which
    # often fall just outside the raw polygon due to the raster grid) get
    # included. Without the buffer the coast has a one-pixel halo of zero
    # alpha — visible as a ragged white fringe.
    coast_poly = load_coastline_polygon()
    if coast_poly is not None:
        pixel_size_deg = (lon_max - lon_min) / out_w
        coast_buffered = coast_poly.buffer(pixel_size_deg * 0.2)
        land_mask = shapely.contains_xy(
            coast_buffered, xi_deg.ravel(), yi_deg.ravel()
        ).reshape(out_h, out_w)
    else:
        # Fixed in this revision: the original referenced an undefined
        # `mask_threshold_rad` here, which would NameError if ever hit.
        pixel_pts      = np.column_stack([xi_grid.ravel(), yi_grid.ravel()])
        pixel_dists, _ = kd.query(pixel_pts, k=1, workers=-1)
        land_mask      = (pixel_dists <= mask_threshold).reshape(out_h, out_w)

    # Map every pixel to its nearest LAND pixel — used to flood data colour
    # into coastline pixels that didn't get a value from interpolation.
    _, _near_idx = distance_transform_edt(~land_mask, return_indices=True)
    near_row = _near_idx[0]
    near_col = _near_idx[1]

    # Hard coastline alpha (1 = land, 0 = sea). No feathering on purpose —
    # feathered edges looked smeared against the basemap.
    _hard_alpha = land_mask.astype(np.float32)

    # ── NaN-aware Gaussian smoothing ────────────────────────────────────────
    _SIGMA = 5.0

    def _smooth_nan(arr2d: np.ndarray) -> np.ndarray:
        """Gaussian smooth, ignoring NaNs, only within land."""
        valid  = np.isfinite(arr2d) & land_mask
        filled = np.where(valid, arr2d, 0.0)
        # Normalised convolution: divide smoothed-values by smoothed-mask.
        wt_num = _gf(filled.astype(np.float64), sigma=_SIGMA)
        wt_den = _gf(valid.astype(np.float64),  sigma=_SIGMA)
        with np.errstate(invalid="ignore"):
            out = np.where(wt_den > 1e-6, wt_num / wt_den, np.nan)
        return out

    # ── Per-snapshot render ─────────────────────────────────────────────────
    def _render(data_1d: np.ndarray) -> str:
        """Turn one (n_points,) data vector into a base64 PNG."""
        data_f = data_1d.astype(float)

        # Use only valid source points; skip frames with no data at all.
        valid_src = np.isfinite(data_f)
        if not np.any(valid_src):
            blank = np.zeros((out_h, out_w, 4), dtype=np.uint8)
            img   = _PILImage.fromarray(blank[::-1], mode="RGBA")
            buf   = _io.BytesIO()
            img.save(buf, format="PNG", optimize=False)
            buf.seek(0)
            b64 = _base64.b64encode(buf.read()).decode()
            buf.close()
            return b64

        pts_valid  = points_merc[valid_src]
        data_valid = data_f[valid_src]

        # 1) Nearest-neighbour everywhere — guarantees no NaNs on land.
        nearest_interp = NearestNDInterpolator(pts_valid, data_valid)
        zi = nearest_interp(xi_grid, yi_grid)

        # 2) Smooth within land, then patch any residual NaNs with NN values.
        zi_land = np.where(land_mask, zi, np.nan)
        zi = _smooth_nan(zi_land)
        still_nan = ~np.isfinite(zi) & land_mask
        if np.any(still_nan):
            zi[still_nan] = nearest_interp(xi_grid[still_nan], yi_grid[still_nan])

        # 3) Prefer linear interpolation wherever Delaunay covers — that's
        #    the convex hull of the data, so inland pixels get the sharper
        #    version. Outside the hull, we keep the NN+smoothed version.
        tri_valid     = Delaunay(pts_valid)
        linear_interp = LinearNDInterpolator(tri_valid, data_valid)
        zi_lin = linear_interp(xi_grid, yi_grid)
        interior = np.isfinite(zi_lin)
        zi[interior] = zi_lin[interior]

        # 4) Replace NaNs with vmin before the colormap so empty cells show
        #    the bottom of the scale (e.g. white for Purples) rather than
        #    black, which is what matplotlib does by default for NaN+norm.
        zi_mapped = np.where(np.isfinite(zi), zi, norm.vmin)
        rgba_grid = cmap(norm(zi_mapped))

        # 5) Flood each coastal pixel's RGB to its nearest land pixel so
        #    the coastline looks filled rather than showing a halo.
        rgba_grid[:, :, :3] = rgba_grid[near_row, near_col, :3]

        # 6) Clip to the hard coastline alpha, then zero-alpha any land
        #    pixels that still have no data (shouldn't happen after NN fill,
        #    but belt-and-braces).
        rgba_grid[:, :, 3] = _hard_alpha
        nan_land = ~np.isfinite(zi) & land_mask
        if np.any(nan_land):
            rgba_grid[nan_land, 3] = 0.0

        # PNG expects top-down rows; our grid is bottom-up (Mercator y
        # increases upward), so flip before encoding.
        rgba_uint8 = (rgba_grid[::-1] * 255).clip(0, 255).astype(np.uint8)
        img = _PILImage.fromarray(rgba_uint8, mode="RGBA")
        buf = _io.BytesIO()
        img.save(buf, format="PNG", optimize=False)
        buf.seek(0)
        b64 = _base64.b64encode(buf.read()).decode()
        buf.close()
        return b64

    snap_b64 = [_render(snap_data_stacked[i]) for i in range(n_snaps)]
    mask_threshold_deg = float(np.rad2deg(mask_threshold))
    return snap_b64, lat_min, lat_max, lon_min, lon_max, mask_threshold_deg


# ============================================================================
# Colourbar renderer
# ----------------------------------------------------------------------------
# Rendered once per (vmin, vmax, scale, units) combo and inlined as base64.
# Kept separate from prerender_snapshots so tick-formatting logic for Log /
# SymLog axes stays out of the hot renderer path.
# ============================================================================
@st.cache_data(show_spinner=False, max_entries=24)
def render_colorbar_b64(vmin: float, vmax: float,
                        colorscale: str, units: str,
                        indicator: str = "", is_change: bool = False) -> str:
    """Return a base64 PNG of the vertical colour bar, transparent background."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    import matplotlib.ticker as mticker

    cmap = _mpl_cmap(colorscale)
    norm = _make_norm(indicator, vmin, vmax, is_change)

    fig, ax = plt.subplots(figsize=(1.0, 6.5))
    fig.patch.set_alpha(0)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, fraction=1.0, pad=0)

    # Log / SymLog need explicit locators or matplotlib falls back to the
    # default linear locator, which ignores the non-linear spacing entirely.
    if isinstance(norm, mcolors.LogNorm):
        cbar.ax.yaxis.set_major_locator(
            mticker.LogLocator(base=10, subs=[1, 2, 3, 5], numticks=12)
        )
        cbar.ax.yaxis.set_minor_locator(
            mticker.LogLocator(base=10, subs=np.arange(1, 10), numticks=50)
        )
        cbar.ax.yaxis.set_major_formatter(
            mticker.FuncFormatter(lambda x, _: f"{x:g}")
        )
        cbar.ax.yaxis.set_minor_formatter(mticker.NullFormatter())
    elif isinstance(norm, mcolors.SymLogNorm):
        cbar.ax.yaxis.set_major_locator(
            mticker.SymmetricalLogLocator(
                base=10, linthresh=norm.linthresh, subs=[1, 2, 3, 5]
            )
        )
        cbar.ax.yaxis.set_major_formatter(
            mticker.FuncFormatter(lambda x, _: f"{x:g}")
        )

    cbar.set_label(units, fontsize=14)
    cbar.ax.tick_params(labelsize=12)
    ax.remove()  # the dummy axes we attached the ScalarMappable to
    fig.tight_layout(pad=0.2)

    buf = _io.BytesIO()
    fig.savefig(buf, format="png", transparent=True, dpi=150, bbox_inches="tight")
    buf.seek(0)
    b64 = _base64.b64encode(buf.read()).decode()
    buf.close()
    plt.close(fig)
    return b64


# Part 3/3
# This part covers:
#   1. build_html_player  — the large dual-map HTML+JS string sent to the browser
#   2. build_loading_screen_html — the NZ-silhouette loading animation
#   3. Dependency guards (xarray, plotly, data root)
#   4. Sidebar form (scenario / indicator / model selectors + opacity/speed controls)
#   5. Main execution flow: timeline → cache → render → player → summary stats
#
# PART 3 SECURITY / PRIVACY NOTES FOR REVIEWERS
# -----------------------------------------------
# * DATA_ROOT and REC_ROOT contain internal HPC path strings
#   ("/esi/project/niwa03712/..."). These have been moved to environment
#   variables (NZMAP_DATA_ROOT, NZMAP_REC_ROOT) in the annotated Part 1 and
#   are referenced only via the Path() constants defined there.  They should
#   NOT be re-hardcoded here.
# * The sidebar renders a small HTML snippet via st.components.v1.html().
#   It posts messages to sibling iframes via window.parent — this is
#   intentional (opacity/speed controls) and confined to the same origin.
# * No user data, credentials, or personal information is stored or transmitted.
# * The "pills" banner at the top of the page previously contained a typo
#   (`border-radFius`) in an inline style string — noted below, harmless.
# * build_loading_screen_html contains a dead second `return` statement that
#   makes the second HTML template unreachable. Flagged below; safe to delete.
 
 
# ============================================================================
# Dual-panel HTML player
# ----------------------------------------------------------------------------
# build_html_player() returns a single self-contained HTML document string.
# It embeds:
#   - Two Leaflet maps (left = absolute values, right = Δ change)
#   - Canvas overlays that cross-fade between precomputed PNG snapshots
#   - A shared timeline slider with tick marks
#   - Hover tooltips (nearest-point interpolation, 30 ms throttle)
#   - Click-to-pin time-series Chart.js panels (one per map, draggable/resizable)
#   - GeoJSON border overlays (country outline + regional council boundaries)
#
# All data (base64 PNGs, per-point value arrays, uncertainty bands) is inlined
# as JSON literals so the iframe is fully self-contained after the initial
# server render.
#
# REVIEW FLAGS
# ------------
# * The function signature has grown to ~40 parameters. Consider grouping into
#   dataclasses (e.g. MapPanelData, ChartData) in a follow-up refactor.
# * snap_colours is accepted as a parameter but never forwarded to JS — the
#   JS side hard-codes its own colour list. Safe to remove the parameter.
# * The JS `vertLinePlugin` for Chart.js computes the nearest snapshot year
#   on every `afterDraw` call. For large SNAP_YEARS arrays this is O(n);
#   fine for the current dataset sizes (<10 snapshots).
# ============================================================================
def build_html_player(
    # Change map (right panel)
    snap_b64:           list,
    colorbar_b64:       str,
    # Absolute map (left panel)
    snap_b64_abs:       list,
    colorbar_b64_abs:   str,
    # Shared timeline
    snap_years:         list,
    frame_years:        list,
    snap_frame_idx:     list,
    snap_labels:        list,
    snap_colours:       list,   # NOTE: accepted but unused in JS — see REVIEW FLAG above
    # Map bounds
    lat_min: float,     lat_max: float,
    lon_min: float,     lon_max: float,
    # Playback
    frame_ms:           int,
    header_html:        str,
    dot_opacity:        float,
    # Hover data — change map
    hover_lats=None,
    hover_lons=None,
    snap_fp_ranges=None,
    snap_vals_list=None,
    hover_units:        str = "",
    # Hover data — absolute map
    snap_abs_vals_list=None,
    abs_units:          str = "",
    # Threshold
    mask_threshold_deg: float = 0.05,
    # Change map chart
    chart_lo_vals=None,
    chart_p5_vals=None,
    chart_p25_vals=None,
    chart_p75_vals=None,
    chart_p95_vals=None,
    chart_hi_vals=None,
    chart_ens_vals=None,
    chart_ymin=None,
    chart_ymax=None,
    # Absolute map chart
    abs_chart_lo_vals=None,
    abs_chart_p5_vals=None,
    abs_chart_p25_vals=None,
    abs_chart_p75_vals=None,
    abs_chart_p95_vals=None,
    abs_chart_hi_vals=None,
    abs_chart_ens_vals=None,
    abs_chart_ymin=None,
    abs_chart_ymax=None,
    # Border overlays
    country_geojson=None,
    regions_geojson=None,
) -> str:
    import json as _json
    n_frames   = len(frame_years)
    center_lat = (lat_min + lat_max) / 2
    center_lon = (lon_min + lon_max) / 2
 
    hover_units_js = _json.dumps(hover_units)
    abs_units_js   = _json.dumps(abs_units)
 
    snap_frames_change_json = _json.dumps(snap_b64)
    snap_frames_abs_json    = _json.dumps(snap_b64_abs)
    snap_years_json         = _json.dumps(snap_years)
    years_json              = _json.dumps(frame_years)
    snap_idx_json           = _json.dumps(snap_frame_idx)
    snap_labels_json        = _json.dumps(snap_labels)
    snap_fp_ranges_json     = _json.dumps(snap_fp_ranges if snap_fp_ranges else snap_labels)
 
    _hover_enabled = (hover_lats is not None and hover_lons is not None
                      and snap_vals_list is not None)
    if _hover_enabled:
        hover_lats_json    = _json.dumps([round(v, 4) for v in hover_lats])
        hover_lons_json    = _json.dumps([round(v, 4) for v in hover_lons])
        snap_vals_json     = _json.dumps([[round(float(x), 3) for x in row]
                                          for row in snap_vals_list])
        snap_abs_vals_json = (_json.dumps([[round(float(x), 3) for x in row]
                                           for row in snap_abs_vals_list])
                              if snap_abs_vals_list is not None else "null")
    else:
        hover_lats_json = hover_lons_json = snap_vals_json = snap_abs_vals_json = "null"
 
    def _enc(arr):
        """JSON-encode a (n_snaps × n_pts) array, or return 'null'."""
        if arr is None:
            return "null"
        return _json.dumps([[round(float(x), 3) for x in row] for row in arr])
 
    # GeoJSON is already a JSON string (or None) — inline as-is.
    country_geojson_js = country_geojson if country_geojson else "null"
    regions_geojson_js = regions_geojson if regions_geojson else "null"
 
    # Uncertainty bands — change panel
    chart_lo_json  = _enc(chart_lo_vals)
    chart_p5_json  = _enc(chart_p5_vals)
    chart_p25_json = _enc(chart_p25_vals)
    chart_p75_json = _enc(chart_p75_vals)
    chart_p95_json = _enc(chart_p95_vals)
    chart_hi_json  = _enc(chart_hi_vals)
    chart_ens_json = _enc(chart_ens_vals)
    chart_ymin_js  = _json.dumps(chart_ymin)
    chart_ymax_js  = _json.dumps(chart_ymax)
 
    # Uncertainty bands — absolute panel
    abs_chart_lo_json  = _enc(abs_chart_lo_vals)
    abs_chart_p5_json  = _enc(abs_chart_p5_vals)
    abs_chart_p25_json = _enc(abs_chart_p25_vals)
    abs_chart_p75_json = _enc(abs_chart_p75_vals)
    abs_chart_p95_json = _enc(abs_chart_p95_vals)
    abs_chart_hi_json  = _enc(abs_chart_hi_vals)
    abs_chart_ens_json = _enc(abs_chart_ens_vals)
    abs_chart_ymin_js  = _json.dumps(abs_chart_ymin)
    abs_chart_ymax_js  = _json.dumps(abs_chart_ymax)
 
    # ------------------------------------------------------------------
    # The returned string is a complete HTML document.  It uses Python
    # f-string interpolation to inline all data.  Double-braces {{ }}
    # are literal JS braces; single braces {var} are Python substitutions.
    # ------------------------------------------------------------------
    return f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<link rel="stylesheet"
  href="https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.9.4/leaflet.min.css">
<script src="https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.9.4/leaflet.min.js"></script>
<script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.1/chart.umd.min.js"></script>
<style>
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{ font-family: Arial, sans-serif; background: white; overflow-x: hidden; }}
  #header {{ padding: 5px 12px 2px; font-size: 11px; color: #444; border-bottom: 1px solid #eee; }}
 
  #maps-row {{ display: flex; gap: 5px; width: 100%; padding: 0 4px; }}
  .map-panel {{ flex: 1; min-width: 0; display: flex; flex-direction: column; }}
  .panel-title {{
    text-align: center; font-size: 14px; font-weight: 700; letter-spacing: 0.04em;
    padding: 6px 0 5px; border-radius: 4px 4px 0 0; margin-bottom: 2px;
  }}
  .panel-title.change-title {{ background: #e8f0fb; color: #1a4a9a; }}
  .panel-title.abs-title    {{ background: #fdf3e8; color: #8a4000; }}
  .map-wrap {{ position: relative; height: 510px; }}
  .map-div  {{ width: 100%; height: 100%; }}
  .colorbar-box {{
    position: absolute; right: 8px; top: 8px; z-index: 999;
    background: rgba(255,255,255,0.88); border-radius: 5px;
    padding: 4px; box-shadow: 0 1px 4px rgba(0,0,0,.15);
  }}
 
  #controls {{ padding: 10px 48px 0 12px; overflow: visible; }}
  #btn-row  {{ display: flex; align-items: flex-start; gap: 8px; overflow: visible; }}
  #slider-wrap {{ flex: 1; display: flex; flex-direction: column; padding-top: 0; margin-left: 12px; overflow: visible; }}
  .ctrl-btn {{
    padding: 4px 13px; font-size: 12px; cursor: pointer;
    border: 1px solid #bbb; border-radius: 4px; background: #f4f4f4;
    margin-top: 7px; flex-shrink: 0; width: 80px; text-align: center;
  }}
  .ctrl-btn:hover {{ background: #e2e2e2; }}
  #timeline-slider {{ width: 100%; cursor: pointer; accent-color: #4a90d9; margin: 4px 0 0; }}
  #tick-row {{ position: relative; width: 100%; height: 70px; overflow: visible; margin-top: 0; }}
  .tick {{ position: absolute; display: flex; flex-direction: column; align-items: center;
           pointer-events: none; user-select: none; transform: translateX(-50%); }}
  .tick.snap-tick {{ top: -28px; }}
  .tick-text-snap {{ font-size: 10px; font-weight: 700; color: #222; white-space: nowrap;
                     margin-bottom: 3px; user-select: none; }}
  .snap-line  {{ width: 1px; height: 14px; background: #888; }}
  .tick.year-tick {{ top: 0; }}
  .tick-line  {{ width: 1px; height: 5px; background: #bbb; }}
  .tick-text-year {{ font-size: 9px; color: #666; white-space: nowrap;
                     transform: rotate(-45deg) translateX(-4px);
                     transform-origin: top left; margin-top: 12px; user-select: none; }}
 
  #hover-tip {{
    position: fixed; z-index: 9999; pointer-events: none;
    background: rgba(20,20,20,0.85); color: #fff;
    border-radius: 6px; padding: 6px 10px;
    font-size: 12px; line-height: 1.55; white-space: nowrap;
    box-shadow: 0 2px 6px rgba(0,0,0,.35); display: none;
  }}
  #hover-tip .tip-val   {{ font-size: 14px; font-weight: 700; }}
  #hover-tip .tip-change {{ color: #7ecfff; }}
  #hover-tip .tip-abs   {{ color: #ffd480; }}
 
  .chart-panel {{
    position: absolute; top: 10px; left: 10px; z-index: 998;
    width: 360px; height: 240px;
    background: rgba(255,255,255,0.96);
    border-radius: 8px; box-shadow: 0 3px 14px rgba(0,0,0,.25);
    padding: 10px 12px 8px;
    cursor: grab; user-select: none;
    resize: both; overflow: hidden;
    min-width: 220px; min-height: 150px;
    display: none; flex-direction: column;
  }}
  .chart-panel:active  {{ cursor: grabbing; }}
  .chart-header {{ display: flex; justify-content: space-between; align-items: center; margin-bottom: 4px; }}
  .chart-title  {{ font-size: 11px; font-weight: 700; color: #333; }}
  .chart-close  {{ background: none; border: none; cursor: pointer; font-size: 14px;
                   color: #888; padding: 0 2px; line-height: 1; }}
  .chart-close:hover {{ color: #333; }}
  .chart-canvas-wrap {{ position: relative; flex: 1; min-height: 120px; }}
</style>
</head>
<body>
<div id="header">{header_html}</div>
 
<div id="maps-row">
 
  <!-- Left panel: Absolute values -->
  <div class="map-panel">
    <div class="panel-title abs-title" id="abs-panel-title">—</div>
    <div class="map-wrap" id="map-wrap-b">
      <div id="map-b" class="map-div"></div>
      <div class="colorbar-box">
        <img src="data:image/png;base64,{colorbar_b64_abs}" style="height:290px">
      </div>
      <div class="chart-panel" id="chart-panel-abs">
        <div class="chart-header">
          <span class="chart-title" id="abs-chart-title"></span>
          <button class="chart-close" id="abs-chart-close">✕</button>
        </div>
        <div class="chart-canvas-wrap"><canvas id="abs-chart-canvas"></canvas></div>
      </div>
    </div>
  </div>
 
  <!-- Right panel: Change from baseline -->
  <div class="map-panel">
    <div class="panel-title change-title">Δ Climate Change Signal</div>
    <div class="map-wrap" id="map-wrap-a">
      <div id="map-a" class="map-div"></div>
      <div class="colorbar-box">
        <img src="data:image/png;base64,{colorbar_b64}" style="height:290px">
      </div>
      <div class="chart-panel" id="chart-panel">
        <div class="chart-header">
          <span class="chart-title" id="chart-title"></span>
          <button class="chart-close" id="chart-close">✕</button>
        </div>
        <div class="chart-canvas-wrap"><canvas id="chart-canvas"></canvas></div>
      </div>
    </div>
  </div>
 
</div>
 
<div id="hover-tip">
  <div id="tip-coord"></div>
  <div class="tip-val tip-change" id="tip-change"></div>
  <div class="tip-val tip-abs"    id="tip-abs"></div>
  <div id="tip-period"></div>
</div>
 
<div id="controls">
  <div id="btn-row">
    <button class="ctrl-btn" id="play-btn">▶ Play</button>
    <button class="ctrl-btn" id="reset-btn">⏮ Reset</button>
    <div id="slider-wrap">
      <input type="range" id="timeline-slider" min="0" max="{n_frames - 1}" value="0" step="1">
      <div id="tick-row"></div>
    </div>
  </div>
</div>
 
<script>
(function() {{
 
  // ── Data (inlined by Python) ───────────────────────────────────────────────
  var SNAP_FRAMES_CHANGE = {snap_frames_change_json};
  var SNAP_FRAMES_ABS    = {snap_frames_abs_json};
  var SNAP_YEARS   = {snap_years_json};
  var YEARS        = {years_json};
  var SNAP_IDX     = {snap_idx_json};
  var SNAP_LABELS  = {snap_labels_json};
  var SNAP_FP_RANGES = {snap_fp_ranges_json};
  var N            = YEARS.length;
  var MS           = {frame_ms};
  var INIT_OPACITY = {dot_opacity:.2f};
 
  var LAT_MIN = {lat_min:.4f}, LAT_MAX = {lat_max:.4f};
  var LON_MIN = {lon_min:.4f}, LON_MAX = {lon_max:.4f};
  // Internal canvas resolution — independent of CSS display size.
  var CANVAS_W = 1200, CANVAS_H = 1800;
 
  // ── Leaflet maps ───────────────────────────────────────────────────────────
  // Both maps share identical options; zoom/pan sync is handled via moveend events.
  var mapOpts = {{
    center: [{center_lat:.3f}, {center_lon:.3f}],
    zoom: 5, minZoom: 5, maxZoom: 9,
    maxBounds: [[LAT_MIN-4, LON_MIN-8], [LAT_MAX+4, LON_MAX+8]],
    zoomControl: false,
  }};
  var mapA = L.map('map-a', mapOpts);   // change (right)
  var mapB = L.map('map-b', mapOpts);   // absolute (left)
 
  function addBaseTiles(m) {{
    // Labels pane sits above the data overlay but intercepts no pointer events.
    m.createPane('labelsPane');
    m.getPane('labelsPane').style.zIndex = 650;
    m.getPane('labelsPane').style.pointerEvents = 'none';
    L.tileLayer(
      'https://server.arcgisonline.com/ArcGIS/rest/services/Elevation/World_Hillshade/MapServer/tile/{{z}}/{{y}}/{{x}}',
      {{ attribution: 'ESRI Hillshade', opacity: 0.45 }}
    ).addTo(m);
    L.tileLayer(
      'https://{{s}}.basemaps.cartocdn.com/rastertiles/voyager_only_labels/{{z}}/{{x}}/{{y}}{{r}}.png',
      {{ attribution: '© CartoDB', subdomains: 'abcd', pane: 'labelsPane', opacity: 1.0 }}
    ).addTo(m);
  }}
  addBaseTiles(mapA);
  addBaseTiles(mapB);
  L.control.zoom({{ position: 'bottomright' }}).addTo(mapA);
  L.control.zoom({{ position: 'bottomright' }}).addTo(mapB);
 
  // ── Border overlays ────────────────────────────────────────────────────────
  // Country and regional council GeoJSON are pre-simplified server-side
  // (SIMPLIFY_TOL = 0.005 °) to keep payload small.
  var _countryGeoJSON = {country_geojson_js};
  var _regionsGeoJSON = {regions_geojson_js};
 
  function addBorders(m) {{
    m.createPane('bordersPane');
    m.getPane('bordersPane').style.zIndex = 450;
    m.getPane('bordersPane').style.pointerEvents = 'none';
    var regionLayer = null, countryLayer = null;
    function weightsForZoom(z) {{
      var t = Math.max(0, Math.min(1, (z-4)/6));
      return {{ region: 0.1+t*1.4, country: 0.3+t*2.8 }};
    }}
    if (_regionsGeoJSON) {{
      regionLayer = L.geoJSON(_regionsGeoJSON, {{
        pane: 'bordersPane',
        style: {{ color: '#555', weight: 0.6, opacity: 0.55, fill: false }},
      }}).addTo(m);
    }}
    if (_countryGeoJSON) {{
      countryLayer = L.geoJSON(_countryGeoJSON, {{
        pane: 'bordersPane',
        style: {{ color: '#333', weight: 1.2, opacity: 0.7, fill: false }},
      }}).addTo(m);
    }}
    // Scale border weight with zoom so lines stay proportionate.
    m.on('zoomend', function() {{
      var w = weightsForZoom(m.getZoom());
      if (regionLayer)  regionLayer.setStyle({{ weight: w.region }});
      if (countryLayer) countryLayer.setStyle({{ weight: w.country }});
    }});
  }}
  addBorders(mapA);
  addBorders(mapB);
 
  // ── Canvas overlays ────────────────────────────────────────────────────────
  // Each map gets its own <canvas> in a custom Leaflet pane (zIndex 410,
  // below borders at 450). The canvas is positioned in layer-point space
  // and re-snapped on every move/zoom event.
  function makeOverlay(m) {{
    m.createPane('dataOverlay');
    var pane = m.getPane('dataOverlay');
    pane.style.zIndex = 410;
    pane.style.pointerEvents = 'none';
    pane.classList.remove('leaflet-zoom-animated');
    var oc   = document.createElement('canvas');
    oc.style.position        = 'absolute';
    oc.style.opacity         = INIT_OPACITY;
    oc.style.imageRendering  = 'auto';
    oc.style.background      = 'transparent';
    var octx = oc.getContext('2d', {{ alpha: true }});
    octx.imageSmoothingEnabled = true;
    octx.imageSmoothingQuality = 'high';
    pane.appendChild(oc);
    oc.width = CANVAS_W; oc.height = CANVAS_H;
    return {{ oc: oc, octx: octx }};
  }}
  var ovA = makeOverlay(mapA);
  var ovB = makeOverlay(mapB);
 
  // Cross-fade between two snapshot images at blend factor t ∈ [0,1].
  function drawCanvas(ov, imgA, imgB, t) {{
    if (!imgA) return;
    ov.octx.clearRect(0, 0, CANVAS_W, CANVAS_H);
    ov.octx.globalCompositeOperation = 'source-over';
    ov.octx.fillStyle = 'rgba(255,255,255,0)';
    ov.octx.fillRect(0, 0, CANVAS_W, CANVAS_H);
    ov.octx.globalAlpha = 1.0;
    ov.octx.drawImage(imgA, 0, 0, CANVAS_W, CANVAS_H);
    if (imgB && t > 0.001) {{
      ov.octx.globalAlpha = t;
      ov.octx.drawImage(imgB, 0, 0, CANVAS_W, CANVAS_H);
    }}
    ov.octx.globalAlpha = 1.0;
  }}
 
  // Reposition canvas CSS rect so it aligns with the map's current viewport.
  function repositionCanvas(m, ov) {{
    var nw = m.latLngToLayerPoint(L.latLng(LAT_MAX, LON_MIN));
    var se = m.latLngToLayerPoint(L.latLng(LAT_MIN, LON_MAX));
    ov.oc.style.left      = nw.x + 'px';
    ov.oc.style.top       = nw.y + 'px';
    ov.oc.style.width     = Math.max(1, Math.round(se.x - nw.x)) + 'px';
    ov.oc.style.height    = Math.max(1, Math.round(se.y - nw.y)) + 'px';
    ov.oc.style.transform = '';
  }}
 
  // RAF-gated reposition so rapid pan events don't stack.
  var _rafA = false, _rafB = false;
  function scheduleReposA() {{
    if (_rafA) return; _rafA = true;
    requestAnimationFrame(function() {{ _rafA = false; repositionCanvas(mapA, ovA); _redrawBoth(); }});
  }}
  function scheduleReposB() {{
    if (_rafB) return; _rafB = true;
    requestAnimationFrame(function() {{ _rafB = false; repositionCanvas(mapB, ovB); _redrawBoth(); }});
  }}
 
  var _lastBlend = {{ a: 0, b: 0, t: 0 }};
  function _redrawBoth() {{
    var b = _lastBlend;
    drawCanvas(ovA, snapImgsChange[b.a], snapImgsChange[b.b], b.t);
    drawCanvas(ovB, snapImgsAbs[b.a],    snapImgsAbs[b.b],    b.t);
  }}
 
  // ── Map synchronisation ────────────────────────────────────────────────────
  // Pan/zoom on either map is mirrored to the other via moveend.
  // _syncing flag prevents the echo.
  var _syncing = false;
  mapA.on('move', scheduleReposA);
  mapB.on('move', scheduleReposB);
  mapA.on('moveend', function() {{
    if (_syncing) return; _syncing = true;
    mapB.setView(mapA.getCenter(), mapA.getZoom(), {{ animate: false }});
    repositionCanvas(mapB, ovB); _redrawBoth(); _syncing = false;
  }});
  mapB.on('moveend', function() {{
    if (_syncing) return; _syncing = true;
    mapA.setView(mapB.getCenter(), mapB.getZoom(), {{ animate: false }});
    repositionCanvas(mapA, ovA); _redrawBoth(); _syncing = false;
  }});
 
  // ── Zoom animation ─────────────────────────────────────────────────────────
  // During Leaflet's CSS zoom animation we apply a matching CSS transform
  // to the canvas so it tracks the tiles.  On zoomend the transform is cleared
  // and repositionCanvas() snaps to the correct pixel position.
  var ZOOM_DUR = 250, ZOOM_EASE = 'cubic-bezier(0,0,0.25,1)';
  function animZoom(m, ov, e) {{
    var curW = parseFloat(ov.oc.style.width) || CANVAS_W;
    var curL = parseFloat(ov.oc.style.left)  || 0;
    var curT = parseFloat(ov.oc.style.top)   || 0;
    if (curW < 1) return;
    var nwF = m._latLngToNewLayerPoint(L.latLng(LAT_MAX, LON_MIN), e.zoom, e.center);
    var seF = m._latLngToNewLayerPoint(L.latLng(LAT_MIN, LON_MAX), e.zoom, e.center);
    var sc  = Math.max(1, seF.x - nwF.x) / curW;
    ov.oc.style.transformOrigin = '0 0';
    ov.oc.style.transition      = 'transform ' + ZOOM_DUR + 'ms ' + ZOOM_EASE;
    ov.oc.style.transform       = 'translate(' + (nwF.x - curL) + 'px,' + (nwF.y - curT) + 'px) scale(' + sc + ')';
  }}
  mapA.on('zoomanim', function(e) {{ animZoom(mapA, ovA, e); }});
  mapB.on('zoomanim', function(e) {{ animZoom(mapB, ovB, e); }});
  mapA.on('zoomend', function() {{ ovA.oc.style.transition = ''; ovA.oc.style.transform = ''; scheduleReposA(); scheduleReposB(); }});
  mapB.on('zoomend', function() {{ ovB.oc.style.transition = ''; ovB.oc.style.transform = ''; scheduleReposA(); scheduleReposB(); }});
 
  repositionCanvas(mapA, ovA);
  repositionCanvas(mapB, ovB);
 
  // ── Image preloading ───────────────────────────────────────────────────────
  // All snapshots are loaded in parallel; showFrame(0) fires only once both
  // arrays are fully loaded.
  var snapImgsChange = new Array(SNAP_FRAMES_CHANGE.length);
  var snapImgsAbs    = new Array(SNAP_FRAMES_ABS.length);
  var _loadedChange  = 0, _loadedAbs = 0;
  function checkAllLoaded() {{
    if (_loadedChange === SNAP_FRAMES_CHANGE.length &&
        _loadedAbs    === SNAP_FRAMES_ABS.length) {{ showFrame(0); }}
  }}
  SNAP_FRAMES_CHANGE.forEach(function(b64, i) {{
    var img = new Image();
    img.onload = function() {{ snapImgsChange[i] = img; _loadedChange++; checkAllLoaded(); }};
    img.src = 'data:image/png;base64,' + b64;
  }});
  SNAP_FRAMES_ABS.forEach(function(b64, i) {{
    var img = new Image();
    img.onload = function() {{ snapImgsAbs[i] = img; _loadedAbs++; checkAllLoaded(); }};
    img.src = 'data:image/png;base64,' + b64;
  }});
 
  // ── Frame logic ────────────────────────────────────────────────────────────
  var slider   = document.getElementById('timeline-slider');
  var periodEl = document.getElementById('abs-panel-title');
  var playBtn  = document.getElementById('play-btn');
  var tickRow  = document.getElementById('tick-row');
 
  function getLabel(fi) {{
    var si = SNAP_IDX.indexOf(fi);
    return si >= 0 ? SNAP_LABELS[si] : String(Math.round(YEARS[fi]));
  }}
 
  // Map a virtual frame index to (snapA, snapB, blend_t).
  function frameToBlend(fi) {{
    for (var s = 0; s < SNAP_IDX.length - 1; s++) {{
      if (fi >= SNAP_IDX[s] && fi <= SNAP_IDX[s+1]) {{
        var span = SNAP_IDX[s+1] - SNAP_IDX[s];
        var t    = span > 0 ? (fi - SNAP_IDX[s]) / span : 0;
        return {{ a: s, b: Math.min(s+1, SNAP_FRAMES_CHANGE.length-1), t: t }};
      }}
    }}
    return {{ a: SNAP_FRAMES_CHANGE.length-1, b: SNAP_FRAMES_CHANGE.length-1, t: 0 }};
  }}
 
  var current = 0;
  function showFrame(fi) {{
    fi = Math.max(0, Math.min(N-1, fi));
    current = fi;
    var b = frameToBlend(fi);
    _lastBlend = b;
    drawCanvas(ovA, snapImgsChange[b.a]||null, snapImgsChange[b.b]||null, b.t);
    drawCanvas(ovB, snapImgsAbs[b.a]   ||null, snapImgsAbs[b.b]   ||null, b.t);
    slider.value = fi;
    periodEl.textContent = getLabel(fi);
    // Redraw the timeline cursor on both charts if they are open.
    if (myChartChange) myChartChange.update('none');
    if (myChartAbs)    myChartAbs.update('none');
  }}
 
  // ── Playback controls ──────────────────────────────────────────────────────
  var playing = false, timer = null;
  function tick() {{
    if (!playing) return;
    if (current >= N-1) {{ pause(); return; }}
    showFrame(current + 1);
    timer = setTimeout(tick, MS);
  }}
  function play()  {{ if (current >= N-1) showFrame(0); playing = true;  playBtn.textContent = '⏸ Pause'; timer = setTimeout(tick, MS); }}
  function pause() {{ playing = false; playBtn.textContent = '▶ Play'; clearTimeout(timer); }}
  playBtn.addEventListener('click', function() {{ if (playing) pause(); else play(); }});
  document.getElementById('reset-btn').addEventListener('click', function() {{ pause(); showFrame(0); }});
  slider.addEventListener('input', function() {{ pause(); showFrame(parseInt(this.value)); }});
 
  // postMessage listener — receives opacity and frameMs from the sidebar
  // HTML widget (see _sidebar_controls_html in the Python sidebar block).
  window.addEventListener('message', function(e) {{
    if (!e.data || e.data.type !== 'nzmap') return;
    if (typeof e.data.opacity === 'number') {{
      ovA.oc.style.opacity = e.data.opacity;
      ovB.oc.style.opacity = e.data.opacity;
    }}
    if (typeof e.data.frameMs === 'number') MS = e.data.frameMs;
  }});
 
  // ── Tick marks ─────────────────────────────────────────────────────────────
  // Rebuilt on resize so labels stay aligned with the slider thumb track.
  function buildTicks() {{
    tickRow.innerHTML = '';
    var thumbR     = 8;
    var sliderRect = slider.getBoundingClientRect();
    var rowRect    = tickRow.getBoundingClientRect();
    var leftInset  = (sliderRect.left - rowRect.left)  + thumbR;
    var rightInset = (rowRect.right   - sliderRect.right) + thumbR;
    var trackW     = rowRect.width - leftInset - rightInset;
    for (var i = 0; i < N; i++) {{
      var si = SNAP_IDX.indexOf(i), isSnap = si >= 0;
      var px = leftInset + (i / (N-1)) * trackW;
      if (isSnap) {{
        var div = document.createElement('div');
        div.className = 'tick snap-tick'; div.style.left = px + 'px';
        var lbl  = document.createElement('div'); lbl.className  = 'tick-text-snap'; lbl.textContent  = SNAP_LABELS[si];
        var line = document.createElement('div'); line.className = 'snap-line';
        div.appendChild(lbl); div.appendChild(line); tickRow.appendChild(div);
      }} else {{
        if (i % 2 !== 0) continue;
        var div  = document.createElement('div'); div.className  = 'tick year-tick'; div.style.left = px + 'px';
        var line = document.createElement('div'); line.className = 'tick-line';
        var txt  = document.createElement('div'); txt.className  = 'tick-text-year'; txt.textContent  = String(Math.round(YEARS[i]));
        div.appendChild(line); div.appendChild(txt); tickRow.appendChild(div);
      }}
    }}
  }}
  setTimeout(buildTicks, 200);
  window.addEventListener('resize', buildTicks);
  window.addEventListener('resize', updateChartPanelLimits);
 
  // ── Hover tooltip ──────────────────────────────────────────────────────────
  // Per-frame values are linearly interpolated between the two surrounding
  // snapshots using the same blend factor as the canvas cross-fade.
  // Nearest-point lookup is a brute-force O(n) scan; fine for n ~2000 points.
  var HOVER_LATS    = {hover_lats_json};
  var HOVER_LONS    = {hover_lons_json};
  var SNAP_VALS     = {snap_vals_json};
  var SNAP_ABS_VALS = {snap_abs_vals_json};
  var HOVER_UNITS   = {hover_units_js};
  var ABS_UNITS     = {abs_units_js};
  var HOVER_THRESH  = {mask_threshold_deg:.5f};  // degrees; points beyond this are suppressed
 
  var tip       = document.getElementById('hover-tip');
  var tipCoord  = document.getElementById('tip-coord');
  var tipChange = document.getElementById('tip-change');
  var tipAbs    = document.getElementById('tip-abs');
  var tipPeriod = document.getElementById('tip-period');
 
  if (HOVER_LATS && SNAP_VALS) {{
    function interpVal(vals, ptIdx, fi) {{
      for (var s = 0; s < SNAP_IDX.length - 1; s++) {{
        if (fi >= SNAP_IDX[s] && fi < SNAP_IDX[s+1]) {{
          var t = (fi - SNAP_IDX[s]) / (SNAP_IDX[s+1] - SNAP_IDX[s]);
          return (1-t) * vals[s][ptIdx] + t * vals[s+1][ptIdx];
        }}
      }}
      return vals[vals.length-1][ptIdx];
    }}
    function nearestPoint(lat, lon) {{
      var best = -1, bestDist = Infinity;
      for (var i = 0; i < HOVER_LATS.length; i++) {{
        var dlat = HOVER_LATS[i]-lat, dlon = HOVER_LONS[i]-lon;
        var d = dlat*dlat + dlon*dlon;
        if (d < bestDist) {{ bestDist = d; best = i; }}
      }}
      return {{ idx: best, dist: Math.sqrt(bestDist) }};
    }}
    function showTip(e, nn) {{
      var changeVal = interpVal(SNAP_VALS, nn.idx, current);
      tipCoord.textContent  = 'Lat ' + HOVER_LATS[nn.idx].toFixed(2) + '°  Lon ' + HOVER_LONS[nn.idx].toFixed(2) + '°';
      tipChange.textContent = 'Δ ' + (changeVal >= 0 ? '+' : '') + changeVal.toFixed(2) + ' ' + HOVER_UNITS;
      if (SNAP_ABS_VALS) {{
        var absVal = interpVal(SNAP_ABS_VALS, nn.idx, current);
        tipAbs.textContent = '◆ ' + absVal.toFixed(2) + ' ' + ABS_UNITS;
      }} else {{ tipAbs.textContent = ''; }}
      tipPeriod.textContent = 'Period: ' + getLabel(current);
      tip.style.display = 'block';
      tip.style.left = (e.originalEvent.clientX + 14) + 'px';
      tip.style.top  = (e.originalEvent.clientY - 10) + 'px';
    }}
    var _hThrottle = null;
    function onMapMousemove(e) {{
      if (_hThrottle) return;
      _hThrottle = setTimeout(function() {{ _hThrottle = null; }}, 30);
      var nn = nearestPoint(e.latlng.lat, e.latlng.lng);
      if (nn.dist > HOVER_THRESH) {{ tip.style.display = 'none'; return; }}
      showTip(e, nn);
    }}
    mapA.on('mousemove', onMapMousemove);
    mapB.on('mousemove', onMapMousemove);
    mapA.on('mouseout', function() {{ tip.style.display = 'none'; }});
    mapB.on('mouseout', function() {{ tip.style.display = 'none'; }});
  }}
 
  // ── Chart helpers ──────────────────────────────────────────────────────────
  // vertLinePlugin draws a vertical dashed line at the current frame year.
  // It is registered per-chart (not globally) to avoid polluting other
  // Chart.js instances on the page.
  var vertLinePlugin = {{
    id: 'vertLine',
    afterDraw: function(chart) {{
      var ctx = chart.ctx, xs = chart.scales.x, ys = chart.scales.y;
      var x = xs.getPixelForValue(YEARS[current]);
      if (x < xs.left || x > xs.right) return;
      ctx.save();
      ctx.beginPath(); ctx.strokeStyle = 'rgba(50,50,50,0.6)'; ctx.lineWidth = 1.5;
      ctx.setLineDash([4,3]); ctx.moveTo(x, ys.top); ctx.lineTo(x, ys.bottom); ctx.stroke();
      ctx.setLineDash([]);
      ctx.fillStyle = '#333'; ctx.font = 'bold 9px Arial'; ctx.textAlign = 'center';
      var _si = -1, _sd = Infinity;
      for (var i = 0; i < SNAP_YEARS.length; i++) {{
        var d = Math.abs(YEARS[current] - SNAP_YEARS[i]);
        if (d < _sd) {{ _sd = d; _si = i; }}
      }}
      ctx.fillText((_si >= 0 && _sd < 3) ? SNAP_FP_RANGES[_si] : String(Math.round(YEARS[current])),
                   x, ys.top - 3);
      ctx.restore();
    }}
  }};
 
  // Build Chart.js dataset list for a pinned point.
  // _pinIdx is set when the user clicks the map and shared by both panels.
  function buildDatasets(lo, p5, p25, p75, p95, hi, ens, perModel, snapVals, units) {{
    var snapYrs = SNAP_IDX.map(function(i) {{ return YEARS[i]; }});
    var ensArr  = ens || snapVals;
    var ds = [];
    if (p5 && p95) {{
      ds.push({{ label: '90% interval',
        data: snapYrs.map(function(yr, i) {{ return {{x:yr, y:p95[i][_pinIdx]}}; }}),
        fill: '+1', backgroundColor: 'rgba(74,144,217,0.13)',
        borderColor: 'transparent', borderWidth: 0, pointRadius: 0, tension: 0.35 }});
      ds.push({{ label: '',
        data: snapYrs.map(function(yr, i) {{ return {{x:yr, y:p5[i][_pinIdx]}}; }}),
        fill: false, borderColor: 'transparent', borderWidth: 0, pointRadius: 0, tension: 0.35 }});
    }}
    if (p25 && p75) {{
      ds.push({{ label: '50% interval',
        data: snapYrs.map(function(yr, i) {{ return {{x:yr, y:p75[i][_pinIdx]}}; }}),
        fill: '+1', backgroundColor: 'rgba(74,144,217,0.28)',
        borderColor: 'transparent', borderWidth: 0, pointRadius: 0, tension: 0.35 }});
      ds.push({{ label: '',
        data: snapYrs.map(function(yr, i) {{ return {{x:yr, y:p25[i][_pinIdx]}}; }}),
        fill: false, borderColor: 'transparent', borderWidth: 0, pointRadius: 0, tension: 0.35 }});
    }}
    if (ensArr) {{
      ds.push({{ label: 'Ensemble mean',
        data: snapYrs.map(function(yr, i) {{ return {{x:yr, y:ensArr[i][_pinIdx]}}; }}),
        borderColor: '#4a90d9', backgroundColor: '#4a90d9',
        borderWidth: 2.5, pointRadius: 4, fill: false, tension: 0.35 }});
    }}
    if (ens && snapVals) {{
      ds.push({{ label: 'Selected model',
        data: snapYrs.map(function(yr, i) {{ return {{x:yr, y:snapVals[i][_pinIdx]}}; }}),
        borderColor: '#e05a20', backgroundColor: '#e05a20',
        borderWidth: 2, borderDash: [5,3], pointRadius: 4, fill: false, tension: 0.35 }});
    }}
    return ds;
  }}
 
  function makeChartOptions(units, ymin, ymax) {{
    return {{
      animation: false, responsive: true, maintainAspectRatio: false,
      interaction: {{ mode: 'nearest', intersect: false, axis: 'x' }},
      plugins: {{
        legend: {{ display: true, position: 'top',
          labels: {{ font: {{ size: 9 }}, boxWidth: 12, padding: 5,
            filter: function(item) {{ return item.text !== ''; }} }} }},
        tooltip: {{ callbacks: {{
          title: function(items) {{
            if (!items.length) return '';
            var xVal = items[0].parsed.x;
            var best = -1, bestDist = Infinity;
            for (var i = 0; i < SNAP_YEARS.length; i++) {{
              var d = Math.abs(SNAP_YEARS[i] - xVal);
              if (d < bestDist) {{ bestDist = d; best = i; }}
            }}
            return SNAP_FP_RANGES[best];
          }},
          label: function(ctx) {{
            if (!ctx.dataset.label) return null;
            var v = ctx.parsed.y;
            if (v == null) return null;
            return ctx.dataset.label + ': ' + (v>=0?'+':'') + v.toFixed(2) + ' ' + units;
          }},
        }} }},
      }},
      scales: {{
        x: {{ type: 'linear',
          min: SNAP_YEARS[0], max: SNAP_YEARS[SNAP_YEARS.length-1],
          ticks: {{ font: {{ size: 9 }}, maxTicksLimit: 12,
            callback: function(v) {{
              var best = -1, bestDist = Infinity;
              for (var i = 0; i < SNAP_YEARS.length; i++) {{
                var d = Math.abs(SNAP_YEARS[i] - v);
                if (d < bestDist) {{ bestDist = d; best = i; }}
              }}
              if (best >= 0 && bestDist < 1.0) return SNAP_FP_RANGES[best];
              return '';
            }} }} }},
        y: {{ title: {{ display: true, text: units, font: {{ size: 9 }} }},
          ticks: {{ font: {{ size: 9 }} }},
          min: (ymin !== null) ? ymin : undefined,
          max: (ymax !== null) ? ymax : undefined }},
      }},
    }};
  }}
 
  // Cap chart panel size to 92% of its parent map wrapper.
  function updateChartPanelLimits() {{
    var wrapA = document.getElementById('map-wrap-a').getBoundingClientRect();
    var wrapB = document.getElementById('map-wrap-b').getBoundingClientRect();
    var panelC = document.getElementById('chart-panel');
    var panelA = document.getElementById('chart-panel-abs');
    panelC.style.maxWidth  = Math.floor(wrapA.width  * 0.92) + 'px';
    panelC.style.maxHeight = Math.floor(wrapA.height * 0.92) + 'px';
    panelA.style.maxWidth  = Math.floor(wrapB.width  * 0.92) + 'px';
    panelA.style.maxHeight = Math.floor(wrapB.height * 0.92) + 'px';
  }}
 
  if (typeof ResizeObserver !== 'undefined') {{
    new ResizeObserver(function() {{ if (myChartChange) myChartChange.resize(); }})
      .observe(document.getElementById('chart-panel'));
    new ResizeObserver(function() {{ if (myChartAbs) myChartAbs.resize(); }})
      .observe(document.getElementById('chart-panel-abs'));
  }}
 
  // ── Uncertainty band data (inlined) ───────────────────────────────────────
  var CHART_LO  = {chart_lo_json},  CHART_P5  = {chart_p5_json},
      CHART_P25 = {chart_p25_json}, CHART_P75 = {chart_p75_json},
      CHART_P95 = {chart_p95_json}, CHART_HI  = {chart_hi_json},
      CHART_ENS = {chart_ens_json};
  var CHART_YMIN = {chart_ymin_js};
  var CHART_YMAX = {chart_ymax_js};
 
  var ABS_CHART_LO  = {abs_chart_lo_json},  ABS_CHART_P5  = {abs_chart_p5_json},
      ABS_CHART_P25 = {abs_chart_p25_json}, ABS_CHART_P75 = {abs_chart_p75_json},
      ABS_CHART_P95 = {abs_chart_p95_json}, ABS_CHART_HI  = {abs_chart_hi_json},
      ABS_CHART_ENS = {abs_chart_ens_json};
  var ABS_CHART_YMIN = {abs_chart_ymin_js};
  var ABS_CHART_YMAX = {abs_chart_ymax_js};
 
  var chartPanelChange = document.getElementById('chart-panel');
  var chartPanelAbs    = document.getElementById('chart-panel-abs');
  var myChartChange    = null;
  var myChartAbs       = null;
  var _pinIdx          = 0;   // index into HOVER_LATS/HOVER_LONS for the pinned point
 
  // Show or refresh the change (right-map) chart for a pinned point.
  function showChangeChartPanel(ptIdx, lat, lon) {{
    _pinIdx = ptIdx;
    if (chartPanelChange.style.display === 'none' || !chartPanelChange.style.display) {{
      chartPanelChange.style.width = '360px'; chartPanelChange.style.height = '240px';
    }}
    updateChartPanelLimits();
    document.getElementById('chart-title').textContent =
      lat.toFixed(2) + '° N  ' + lon.toFixed(2) + '° E';
    if (myChartChange) {{ myChartChange.destroy(); myChartChange = null; }}
    myChartChange = new Chart(document.getElementById('chart-canvas'), {{
      type: 'line', plugins: [vertLinePlugin],
      data: {{ datasets: buildDatasets(CHART_LO, CHART_P5, CHART_P25, CHART_P75,
                                       CHART_P95, CHART_HI, CHART_ENS,
                                       CHART_ENS, SNAP_VALS, HOVER_UNITS) }},
      options: makeChartOptions(HOVER_UNITS, CHART_YMIN, CHART_YMAX),
    }});
    chartPanelChange.style.display = 'flex';
  }}
 
  // Show or refresh the absolute (left-map) chart for a pinned point.
  function showAbsChartPanel(ptIdx, lat, lon) {{
    _pinIdx = ptIdx;
    if (chartPanelAbs.style.display === 'none' || !chartPanelAbs.style.display) {{
      chartPanelAbs.style.width = '360px'; chartPanelAbs.style.height = '240px';
    }}
    updateChartPanelLimits();
    document.getElementById('abs-chart-title').textContent =
      lat.toFixed(2) + '° N  ' + lon.toFixed(2) + '° E';
    if (myChartAbs) {{ myChartAbs.destroy(); myChartAbs = null; }}
    myChartAbs = new Chart(document.getElementById('abs-chart-canvas'), {{
      type: 'line', plugins: [vertLinePlugin],
      data: {{ datasets: buildDatasets(ABS_CHART_LO, ABS_CHART_P5, ABS_CHART_P25,
                                       ABS_CHART_P75, ABS_CHART_P95, ABS_CHART_HI,
                                       ABS_CHART_ENS, ABS_CHART_ENS,
                                       SNAP_ABS_VALS, ABS_UNITS) }},
      options: makeChartOptions(ABS_UNITS, ABS_CHART_YMIN, ABS_CHART_YMAX),
    }});
    chartPanelAbs.style.display = 'flex';
  }}
 
  // Map click handlers — pin a point and open the corresponding chart.
  if (HOVER_LATS && SNAP_VALS) {{
    mapA.on('click', function(e) {{
      var nn = nearestPoint(e.latlng.lat, e.latlng.lng);
      if (nn.dist > HOVER_THRESH) return;
      showChangeChartPanel(nn.idx, HOVER_LATS[nn.idx], HOVER_LONS[nn.idx]);
    }});
  }}
  if (HOVER_LATS && SNAP_ABS_VALS) {{
    mapB.on('click', function(e) {{
      var nn = nearestPoint(e.latlng.lat, e.latlng.lng);
      if (nn.dist > HOVER_THRESH) return;
      showAbsChartPanel(nn.idx, HOVER_LATS[nn.idx], HOVER_LONS[nn.idx]);
    }});
  }}
 
  document.getElementById('chart-close').addEventListener('click', function() {{
    chartPanelChange.style.display = 'none';
    if (myChartChange) {{ myChartChange.destroy(); myChartChange = null; }}
  }});
  document.getElementById('abs-chart-close').addEventListener('click', function() {{
    chartPanelAbs.style.display = 'none';
    if (myChartAbs) {{ myChartAbs.destroy(); myChartAbs = null; }}
  }});
 
  // ── Draggable chart panels ─────────────────────────────────────────────────
  // Simple mousedown/mousemove/mouseup drag with boundary clamping to the
  // parent map wrapper. The resize handle (bottom-right corner) is excluded
  // from drag detection so the browser's native resize still works.
  function makeDraggable(panel, wrapId) {{
    var dragX = 0, dragY = 0, startL = 0, startT = 0, dragging = false;
    panel.addEventListener('mousedown', function(e) {{
      var rect = panel.getBoundingClientRect();
      // Exclude the resize handle (bottom-right 16 × 16 px corner).
      if (e.clientX > rect.right - 16 && e.clientY > rect.bottom - 16) return;
      if (e.target.classList.contains('chart-close') ||
          e.target.closest('.chart-canvas-wrap')) return;
      dragging = true;
      dragX = e.clientX; dragY = e.clientY;
      startL = parseInt(panel.style.left) || 0;
      startT = parseInt(panel.style.top)  || 0;
      panel.style.cursor = 'grabbing';
      e.preventDefault();
    }});
    document.addEventListener('mousemove', function(e) {{
      if (!dragging) return;
      var wrap = document.getElementById(wrapId).getBoundingClientRect();
      panel.style.left = Math.max(0, Math.min(startL + e.clientX - dragX,
                                              wrap.width  - panel.offsetWidth))  + 'px';
      panel.style.top  = Math.max(0, Math.min(startT + e.clientY - dragY,
                                              wrap.height - panel.offsetHeight)) + 'px';
    }});
    document.addEventListener('mouseup', function() {{
      if (dragging) {{ dragging = false; panel.style.cursor = 'grab'; }}
    }});
  }}
  makeDraggable(document.getElementById('chart-panel'),     'map-wrap-a');
  makeDraggable(document.getElementById('chart-panel-abs'), 'map-wrap-b');
 
}})();
</script>
</body>
</html>"""
 
 
# ============================================================================
# Loading screen
# ----------------------------------------------------------------------------
# Shown while prerender_snapshots() runs server-side. Uses the NZ silhouette
# SVG path built by nz_loader_svg_data() in Part 2 to animate a "rising water"
# effect — a dark fill sweeps upward through the outline on a loop.
#
# REVIEW FLAG
# -----------
# The function body contains two `return` statements. The second one (below
# the first returned f-string) is unreachable dead code — it references a
# second, slightly different HTML template that was superseded by the first.
# Safe to delete everything from the second `return f"""` onward in a
# follow-up PR.
# ============================================================================
def build_loading_screen_html(svg_data: dict | None, height: int = 630) -> str:
    """
    Return an HTML string for the loading animation iframe.
 
    If svg_data is None (coastline shapefile missing or geopandas unavailable),
    falls back to a plain CSS spinner.
    """
    if svg_data is None:
        body = ('<div style="width:60px;height:60px;border:4px solid #ddd;'
                'border-top-color:#4a90d9;border-radius:50%;'
                'animation:spin 1s linear infinite"></div>'
                '<style>@keyframes spin{to{transform:rotate(360deg)}}</style>')
    else:
        body = f"""
<div class="nz-wrap">
  <svg viewBox="{svg_data['viewBox']}" preserveAspectRatio="xMidYMid meet"
       style="width:100%;height:100%;display:block">
    <path d="{svg_data['d']}"
          style="fill:#d8d8d8;stroke:#999;stroke-width:0.015;
                 vector-effect:non-scaling-stroke"/>
    <path d="{svg_data['d']}" class="nz-fill"
          style="fill:#222;stroke:#000;stroke-width:0.015;
                 vector-effect:non-scaling-stroke"/>
  </svg>
</div>"""
 
    return f"""<!DOCTYPE html>
<html><head><style>
  html, body {{
    margin:0; padding:0; height:{height}px; width:100%;
    position:relative; background:#fafafa; overflow:hidden;
  }}
  .nz-wrap {{
    position:absolute;
    top: 50%;
    left: 50%;
    transform: translate(-50%, -50%);
    width:  280px;
    height: 372px;
    clip-path: inset(0 0 10% 0);
    -webkit-clip-path: inset(0 0 10% 0);
  }}
  .nz-fill {{
    clip-path: inset(100% 0 0 0);
    -webkit-clip-path: inset(100% 0 0 0);
    animation: rise 2.4s cubic-bezier(0.45,0,0.55,1) infinite;
  }}
  @keyframes rise {{
    0%   {{ clip-path: inset(100% 0 0 0); -webkit-clip-path: inset(100% 0 0 0); }}
    50%  {{ clip-path: inset(0 0 0 0);    -webkit-clip-path: inset(0 0 0 0); }}
    100% {{ clip-path: inset(0 0 100% 0); -webkit-clip-path: inset(0 0 100% 0); }}
  }}
</style></head>
<body>{body}</body></html>"""
 
 
# ============================================================================
# Dependency guards
# ----------------------------------------------------------------------------
# These run at module import time (Streamlit reruns the whole file on each
# interaction). st.stop() halts the page with a clear error rather than
# letting a missing library surface as a deep traceback.
# ============================================================================
if not HAS_XARRAY:
    st.error("**`xarray` is not installed.**")
    st.stop()
if not HAS_PLOTLY:
    st.error("**`plotly` is not installed.**")
    st.stop()
if not DATA_ROOT.exists():
    # DATA_ROOT was already redirected to test/demo_data/ if that folder
    # existed (see _DEMO_MODE block in Part 1). If we reach here it means
    # neither the real data nor the demo data was found.
    st.error(
        f"Data directory **`{DATA_ROOT}`** not found "
        "and no demo data at `test/demo_data/`.\n\n"
        "To generate the synthetic demo dataset run:\n"
        "```\npython test/generate_demo_data.py\n```"
    )
    st.stop()
 
_MODEL_ENSEMBLE_MEAN = "Ensemble mean (all models)"
 
 
# ============================================================================
# Sidebar
# ----------------------------------------------------------------------------
# The sidebar uses st.form so that all widget changes are batched — the
# expensive render pipeline only fires when the user clicks "Apply".
#
# Session-state key "applied" holds the last submitted selection. The form
# widgets are initialised from this dict so the UI reflects the current render.
#
# The opacity/speed controls are a separate HTML snippet rendered via
# st.components.v1.html(). It broadcasts a postMessage to sibling iframes
# (the player iframe) so controls take effect without a Python rerun.
# ============================================================================
indicators_avail = list_indicators("historical")
 
if not indicators_avail:
    st.sidebar.warning(f"No data under `{DATA_ROOT}/historical/static_maps/`")
    st.stop()
 
if "applied" not in st.session_state:
    st.session_state["applied"] = dict(
        ssp="ssp370",
        bp_tag="bp1995-2014",
        indicator=indicators_avail[0],
        season=SEASON_ANN,
        model_choice=_MODEL_ENSEMBLE_MEAN,
    )
 
with st.sidebar:
    logo_path = Path("logos/esnz_logo_horz_new.png")
    if logo_path.exists():
        st.image(str(logo_path))
 
    with st.form("map_controls", border=False):
        st.subheader("Future scenario (SSP)")
        _sel_ssp = st.selectbox(
            "SSP scenario", SSP_OPTIONS,
            format_func=lambda s: SSP_LABELS[s],
            label_visibility="collapsed",
            index=SSP_OPTIONS.index(st.session_state["applied"]["ssp"]),
            key="_sel_ssp",
        )
 
        st.subheader("Baseline period")
        _sel_bp_tag = st.selectbox(
            "Baseline period", BP_OPTIONS,
            format_func=lambda b: BP_LABELS[b],
            label_visibility="collapsed",
            index=BP_OPTIONS.index(st.session_state["applied"]["bp_tag"]),
            help="All change values shown are relative to this baseline.",
            key="_sel_bp_tag",
        )
 
        st.markdown("---")
 
        st.subheader("Indicator")
        _ind_default = st.session_state["applied"]["indicator"]
        _ind_idx = indicators_avail.index(_ind_default) if _ind_default in indicators_avail else 0
        _sel_indicator = st.selectbox(
            "Indicator", indicators_avail,
            format_func=lambda i: f"{i} — {INDICATOR_LABELS.get(i, '')}",
            label_visibility="collapsed",
            index=_ind_idx,
            key="_sel_indicator",
        )
 
        # Season list is driven by the *currently applied* indicator, not the
        # in-form selection.  Refreshes on next Apply.
        _applied_ind   = st.session_state["applied"]["indicator"]
        _seasons_avail = discover_seasons("historical", _applied_ind)
        if len(_seasons_avail) > 1:
            st.subheader("Season")
            _seas_default = st.session_state["applied"]["season"]
            _seas_idx = (_seasons_avail.index(_seas_default)
                         if _seas_default in _seasons_avail else 0)
            _sel_season = st.selectbox(
                "Season", _seasons_avail,
                format_func=lambda s: SEASON_LABELS.get(s, s),
                label_visibility="collapsed",
                index=_seas_idx,
                key="_sel_season",
            )
        else:
            _sel_season = SEASON_ANN
 
        st.subheader("Model")
        _avail_models = discover_models(
            st.session_state["applied"]["ssp"],
            _applied_ind,
            st.session_state["applied"]["bp_tag"],
            st.session_state["applied"]["season"],
        )
        _model_options = [_MODEL_ENSEMBLE_MEAN] + _avail_models
        _mod_default   = st.session_state["applied"]["model_choice"]
        _mod_idx       = (_model_options.index(_mod_default)
                          if _mod_default in _model_options else 0)
        _sel_model = st.selectbox(
            "Model", _model_options, label_visibility="collapsed",
            help=f"{len(_avail_models)} models available for this selection.",
            index=_mod_idx,
            key="_sel_model",
        )
 
        st.markdown("---")
        _submitted = st.form_submit_button(
            "▶  Apply", type="primary", use_container_width=True,
        )
 
    if _submitted:
        st.session_state["applied"] = dict(
            ssp=_sel_ssp, bp_tag=_sel_bp_tag, indicator=_sel_indicator,
            season=_sel_season, model_choice=_sel_model,
        )
        st.rerun()
 
    # Unpack the applied config into module-level names used by the rest of
    # the page.
    _cfg = st.session_state["applied"]
    ssp               = _cfg["ssp"]
    bp_tag            = _cfg["bp_tag"]
    indicator         = _cfg["indicator"]
    season            = _cfg["season"]
    model_choice      = _cfg["model_choice"]
    selected_model_key: str | None = (
        None if model_choice == _MODEL_ENSEMBLE_MEAN else model_choice
    )
 
    colorscale     = colorscale_for(indicator)
    colorscale_abs = colorscale_abs_for(indicator)
 
    # ── Opacity / speed controls (postMessage to player iframe) ───────────────
    import streamlit.components.v1 as _sc
    _sidebar_controls_html = """
<style>
  body { margin: 0; padding: 0 4px; font-family: Arial, sans-serif; background: transparent; }
  h4   { font-size: 11px; font-weight: 700; color: #31333f;
         margin: 0 0 8px; text-transform: uppercase; letter-spacing: 0.04em; }
  .lbl { font-size: 13px; color: #31333f; font-weight: 600;
         display: block; margin: 10px 0 4px; }
  .lbl:first-of-type { margin-top: 0; }
  #oSlider { width: 100%; accent-color: #4a90d9; cursor: pointer; display: block; margin-bottom: 1px; }
  #oVal    { font-size: 11px; color: #888; float: right; }
  #sSelect { width: 100%; font-size: 12px; padding: 3px 6px;
             border: 1px solid #ccc; border-radius: 4px;
             background: #fafafa; cursor: pointer; color: #333; margin-top: 2px; }
  #sSelect:focus { outline: none; border-color: #4a90d9; }
</style>
<span class="lbl">Overlay opacity <span id="oVal">0.70</span></span>
<input type="range" id="oSlider" min="0.1" max="1.0" step="0.05" value="0.70">
<span class="lbl">Animation speed</span>
<select id="sSelect">
  <option value="2000">Very Slow</option>
  <option value="900">Slow</option>
  <option value="400" selected>Medium</option>
  <option value="160">Fast</option>
</select>
<script>
(function() {
  function send() {
    var msg = { type: 'nzmap',
                opacity: parseFloat(document.getElementById('oSlider').value),
                frameMs: parseInt(document.getElementById('sSelect').value, 10) };
    // Broadcast to all iframes on the Streamlit page — only the player
    // iframe will respond (it checks e.data.type === 'nzmap').
    var frames = window.parent.document.querySelectorAll('iframe');
    for (var i = 0; i < frames.length; i++) {
      try { frames[i].contentWindow.postMessage(msg, '*'); } catch(e) {}
    }
  }
  document.getElementById('oSlider').addEventListener('input', function() {
    document.getElementById('oVal').textContent = parseFloat(this.value).toFixed(2);
    send();
  });
  document.getElementById('sSelect').addEventListener('change', send);
})();
</script>
"""
    _sc.html(_sidebar_controls_html, height=110)
 
    # These defaults are overridden live by the sidebar HTML widget above;
    # they only apply on initial page load before the first postMessage fires.
    dot_opacity = 0.7
    frame_ms    = 400
 
 
# ============================================================================
# Main execution flow
# ============================================================================
 
# ── 1. Build the snapshot timeline ───────────────────────────────────────────
SNAPSHOTS = build_timeline(indicator, bp_tag, ssp, season)
st.title("🗺️ NZ Climate Indicator Map")
 
if _DEMO_MODE:
    st.info(
        "🧪 **Demo mode** — running on synthetic test data "
        "(TX indicator · SSP3-7.0 · bp1995-2014 · Annual). "
        "The real climate data was not found at the configured paths. "
        "See `test/README.md` for details."
    )
 
season_label = SEASON_LABELS.get(season, season)
 
season_label = SEASON_LABELS.get(season, season)
bp_short     = bp_tag.replace("bp", "").replace("-", "–")
model_label  = selected_model_key if selected_model_key else "Ensemble mean"
 
if not SNAPSHOTS:
    st.error(
        f"No files found for indicator **{indicator}**, baseline **{bp_tag}**, "
        f"season **{season}**. Try a different combination."
    )
    st.stop()
 
SNAP_COLOURS = ["#4a90d9", "#1a5fa8", "#e8a020", "#e05a20",
                "#c0392b", "#8e1a1a", "#5a0f0f", "#2d0808"]
 
# ── Snapshot pill banner ──────────────────────────────────────────────────────
pills = "".join(
    f'<span style="display:inline-block;padding:4px 14px;border-radius:16px;'
    f'background:{SNAP_COLOURS[min(i, len(SNAP_COLOURS)-1)]};color:white;'
    f'font-size:0.78rem;font-weight:600;margin-right:6px;margin-bottom:4px">'
    f'{label}</span>'
    for i, (label, _, _, _, _) in enumerate(SNAPSHOTS)
)
st.markdown(
    f'<div style="margin-bottom:10px">{pills}'
    f'<span style="font-size:0.78rem;color:#888">'
    f'← {len(SNAPSHOTS)} snapshots · browser-blended · left = absolute · right = Δ change'
    f'</span></div>',
    unsafe_allow_html=True,
)
 
# ── 2. Show loading screen while data loads ───────────────────────────────────
import streamlit.components.v1 as _components
 
_map_slot = st.empty()
with _map_slot.container():
    _components.html(
        build_loading_screen_html(nz_loader_svg_data(), height=630),
        height=630, scrolling=False,
    )
 
# ── 3. Load uncertainty cache (abort if missing) ──────────────────────────────
_unc_pre = load_uncertainty_cache(indicator, ssp, bp_tag, season)
if _unc_pre is None:
    if _DEMO_MODE:
        st.error(
            "Demo uncertainty cache not found for the current selection. "
            "The demo covers **TX · SSP3-7.0 · bp1995-2014 · Annual** only. "
            "Re-run `python test/generate_demo_data.py` if the file is missing."
        )
    else:
        st.error(
            "Uncertainty cache not found. "
            "Please run `precompute_uncertainty.py` first."
        )
    st.stop()
 
# Build a KDTree over the cache's spatial points so we can remap them to the
# app's lat/lon grid by nearest-neighbour when needed.
from scipy.spatial import KDTree as _KDTree
_unc_pts  = np.column_stack([_unc_pre["lat_v"], _unc_pre["lon_v"]])
_unc_tree = _KDTree(_unc_pts)
 
_unc_snap_years_pre = _unc_pre["snap_years"]
 
 
def _get_snap_row(band: np.ndarray, yr: float) -> np.ndarray:
    """
    Return the row of `band` whose year is closest to `yr`.
    Returns a NaN row if the nearest year is more than 2 years away
    (indicates the cache was built with a different timeline).
    """
    dists = [abs(yr - uy) for uy in _unc_snap_years_pre]
    best  = int(np.argmin(dists))
    return band[best] if dists[best] < 2.0 else np.full(band.shape[1], np.nan)
 
 
# Coordinates come from the cache — use them for rendering and hover.
_lat_v = _unc_pre["lat_v"]
_lon_v = _unc_pre["lon_v"]
n_cache_pts = len(_lat_v)
 
snap_years    = [yr    for _, _, _, _, yr    in SNAPSHOTS]
snap_labels   = [label for label, _, _, _, _ in SNAPSHOTS]
snap_n_models = _unc_pre["n_models"]
 
# ── 4. Build per-snapshot data stacks (n_snaps × n_pts) ──────────────────────
# When a specific model is selected and the cache contains per-model values,
# use those; otherwise use the ensemble mean from the cache.
if selected_model_key is not None and \
        selected_model_key in _unc_pre.get("model_change_vals", {}):
    _stacked = np.stack(
        [_get_snap_row(_unc_pre["model_change_vals"][selected_model_key], yr)
         for _, _, _, _, yr in SNAPSHOTS],
        axis=0
    )
else:
    _stacked = np.stack(
        [_get_snap_row(_unc_pre["ens_vals"], yr) for _, _, _, _, yr in SNAPSHOTS],
        axis=0
    )
 
if selected_model_key is not None and \
        selected_model_key in _unc_pre.get("model_abs_vals", {}):
    _stacked_abs = np.stack(
        [_get_snap_row(_unc_pre["model_abs_vals"][selected_model_key], yr)
         for _, _, _, _, yr in SNAPSHOTS],
        axis=0
    )
else:
    _stacked_abs = np.stack(
        [_get_snap_row(_unc_pre["abs_ens_vals"], yr) for _, _, _, _, yr in SNAPSHOTS],
        axis=0
    )
 
_valid = np.ones(n_cache_pts, dtype=bool)
 
# ── 5. Compute colour ranges ──────────────────────────────────────────────────
with st.spinner("Computing colour ranges…"):
    shared_half = compute_color_range(indicator)
    abs_vmin, abs_vmax = compute_abs_color_range(indicator)
 
# ── 6. Render (or restore from cache) PNG frames ─────────────────────────────
_lm_change = _log_mode(indicator, is_change=True)
_lm_abs    = _log_mode(indicator, is_change=False)
 
if indicator in _REC_INDICATORS:
    _chg_vmin, _chg_vmax = 0.0, 100.0
else:
    _chg_vmin, _chg_vmax = -shared_half, shared_half
 
# Check disk cache before calling the expensive prerender_snapshots().
_key_change    = _frame_cache_key(
    indicator, ssp, bp_tag, season, selected_model_key,
    colorscale, _chg_vmin, _chg_vmax, _lm_change
)
_cached_change = _load_frame_cache(_key_change)
 
if _cached_change is not None:
    (snap_b64, lat_min, lat_max,
     lon_min, lon_max, _mask_thresh_deg) = _cached_change
else:
    with st.spinner(f"Rendering {len(SNAPSHOTS)} change frames…"):
        (snap_b64, lat_min, lat_max,
         lon_min, lon_max, _mask_thresh_deg) = prerender_snapshots(
            lat_v=_lat_v,
            lon_v=_lon_v,
            snap_data_stacked=_stacked,
            vmin=_chg_vmin,
            vmax=_chg_vmax,
            colorscale=colorscale,
            indicator=indicator,
            is_change=(indicator not in _REC_INDICATORS),
        )
    _save_frame_cache(_key_change, (snap_b64, lat_min, lat_max,
                                    lon_min, lon_max, _mask_thresh_deg))
 
_key_abs    = _frame_cache_key(
    indicator, ssp, bp_tag, season, selected_model_key,
    colorscale_abs, abs_vmin, abs_vmax, _lm_abs
)
_cached_abs = _load_frame_cache(_key_abs)
 
if _cached_abs is not None:
    snap_b64_abs = _cached_abs[0]
else:
    with st.spinner(f"Rendering {len(SNAPSHOTS)} absolute value frames…"):
        _result_abs = prerender_snapshots(
            lat_v=_lat_v,
            lon_v=_lon_v,
            snap_data_stacked=_stacked_abs,
            vmin=abs_vmin,
            vmax=abs_vmax,
            colorscale=colorscale_abs,
            indicator=indicator,
            is_change=False,
        )
    snap_b64_abs = _result_abs[0]
    _save_frame_cache(_key_abs, _result_abs)
 
# ── 7. Virtual frame timeline ─────────────────────────────────────────────────
frame_years_all, snap_frame_idx = compute_frame_timeline(snap_years, YEAR_STEP)
 
# ── 8. Load and re-index uncertainty bands for chart panels ──────────────────
_borders = load_borders_geojson()
 
with st.spinner("Loading model uncertainty range…"):
    _unc = _unc_pre  # already loaded above — same object, no second disk read
 
    if _unc is not None:
        _unc_snap_years  = _unc["snap_years"]
        _unc_snap_labels = _unc["snap_labels"]
 
        # Spatial re-indexing: map the cache's point set onto _lat_v/_lon_v.
        _app_pts = np.column_stack([_lat_v, _lon_v])
        _, _unc_nn_idx = _unc_tree.query(_app_pts, k=1)
 
        def _reindex_unc_band(band: np.ndarray) -> np.ndarray:
            """
            Re-index rows by year AND columns by spatial nearest-neighbour.
            Returns shape (n_snaps, n_app_pts).
            """
            n_our = len(SNAPSHOTS)
            n_pts = len(_lat_v)
            out   = np.full((n_our, n_pts), np.nan)
            for i, (_, _, _, _, yr) in enumerate(SNAPSHOTS):
                dists = [abs(yr - uy) for uy in _unc_snap_years]
                best  = int(np.argmin(dists))
                if dists[best] < 2.0:
                    out[i] = band[best][_unc_nn_idx]
            return out
 
        _lo_stacked  = _reindex_unc_band(_unc["lo_vals"])
        _p5_stacked  = _reindex_unc_band(_unc["p5_vals"])
        _p25_stacked = _reindex_unc_band(_unc["p25_vals"])
        _p75_stacked = _reindex_unc_band(_unc["p75_vals"])
        _p95_stacked = _reindex_unc_band(_unc["p95_vals"])
        _hi_stacked  = _reindex_unc_band(_unc["hi_vals"])
 
        # When a specific model is selected, the ensemble mean is shown as a
        # comparison line alongside the per-model line.
        _ens_stacked     = (None if selected_model_key is None
                            else _reindex_unc_band(_unc["ens_vals"]))
        _abs_ens_stacked = (None if selected_model_key is None
                            else _reindex_unc_band(_unc["abs_ens_vals"]))
 
        # Absolute uncertainty bands — use NaN arrays if not present in cache
        # (older precompute runs may not have included them).
        def _abs_band(key):
            return (_reindex_unc_band(_unc[key])
                    if key in _unc else np.full_like(_lo_stacked, np.nan))
 
        _abs_lo_stacked  = _abs_band("abs_lo_vals")
        _abs_p5_stacked  = _abs_band("abs_p5_vals")
        _abs_p25_stacked = _abs_band("abs_p25_vals")
        _abs_p75_stacked = _abs_band("abs_p75_vals")
        _abs_p95_stacked = _abs_band("abs_p95_vals")
        _abs_hi_stacked  = _abs_band("abs_hi_vals")
 
        _chart_ymin     = _unc.get("chart_ymin",     None)
        _chart_ymax     = _unc.get("chart_ymax",     None)
        _abs_chart_ymin = _unc.get("abs_chart_ymin", None)
        _abs_chart_ymax = _unc.get("abs_chart_ymax", None)
 
        # Build per-snapshot "fp range" labels (e.g. "2041–2060") for chart
        # tooltip titles.
        _snap_fp_ranges = []
        for (_, _, _, _, yr) in SNAPSHOTS:
            dists = [abs(yr - uy) for uy in _unc_snap_years]
            best  = int(np.argmin(dists))
            _snap_fp_ranges.append(
                _unc["snap_fp_ranges"][best] if dists[best] < 2.0
                else str(int(round(yr)))
            )
 
    else:
        # ── Live fallback (slow) ───────────────────────────────────────────
        # Triggered only when the uncertainty cache is missing entirely.
        # precompute_uncertainty.py should be run to populate the cache.
        st.toast("Uncertainty cache not found — computing live (may be slow).", icon="⚠️")
        _lo_data_list, _p5_data_list   = [], []
        _p25_data_list, _p75_data_list = [], []
        _p95_data_list, _hi_data_list  = [], []
        _zeros = np.zeros(n_cache_pts)
        _nans  = np.full(n_cache_pts, np.nan)
        for _label, _scen, _fp, _bp, _yr in SNAPSHOTS:
            if _scen == "historical":
                for lst in (_lo_data_list, _p5_data_list, _p25_data_list,
                            _p75_data_list, _p95_data_list, _hi_data_list):
                    lst.append(_zeros)
            else:
                _lo2d, _p52d, _p252d, _p752d, _p952d, _hi2d, _ = \
                    load_model_range(_scen, indicator, _fp, _bp, season)
                ok = _lo2d is not None
                _lo_data_list.append(_lo2d.ravel()  if ok else _nans)
                _p5_data_list.append(_p52d.ravel()  if ok else _nans)
                _p25_data_list.append(_p252d.ravel() if ok else _nans)
                _p75_data_list.append(_p752d.ravel() if ok else _nans)
                _p95_data_list.append(_p952d.ravel() if ok else _nans)
                _hi_data_list.append(_hi2d.ravel()   if ok else _nans)
 
        _lo_stacked  = np.stack(_lo_data_list,  axis=0)
        _p5_stacked  = np.stack(_p5_data_list,  axis=0)
        _p25_stacked = np.stack(_p25_data_list, axis=0)
        _p75_stacked = np.stack(_p75_data_list, axis=0)
        _p95_stacked = np.stack(_p95_data_list, axis=0)
        _hi_stacked  = np.stack(_hi_data_list,  axis=0)
        _ens_stacked = _abs_ens_stacked = None
        _abs_lo_stacked = _abs_p5_stacked = _abs_p25_stacked = \
            _abs_p75_stacked = _abs_p95_stacked = _abs_hi_stacked = \
            np.full_like(_lo_stacked, np.nan)
        _chart_ymin = _chart_ymax = _abs_chart_ymin = _abs_chart_ymax = None
 
        _snap_fp_ranges = []
        for _label, _scen, _fp, _bp, _yr in SNAPSHOTS:
            m = re.search(r"(\d{4})-(\d{4})", _fp)
            _snap_fp_ranges.append(
                f"{m.group(1)}–{m.group(2)}" if m else str(int(round(_yr)))
            )
 
# ── 9. Units and colour bars ──────────────────────────────────────────────────
if indicator in _REC_INDICATORS:
    units_note     = "%"
    abs_units_note = _REC_ABS_UNITS.get(indicator, "")
else:
    units_note     = f"Δ {INDICATOR_UNITS.get(indicator, '')}"
    abs_units_note = INDICATOR_UNITS.get(indicator, "")
 
with st.spinner("Rendering colour bars…"):
    colorbar_b64 = render_colorbar_b64(
        _chg_vmin, _chg_vmax, colorscale, units_note,
        indicator=indicator, is_change=(indicator not in _REC_INDICATORS),
    )
    colorbar_b64_abs = render_colorbar_b64(
        abs_vmin, abs_vmax, colorscale_abs, abs_units_note,
        indicator=indicator, is_change=False,
    )
 
# ── 10. Assemble and render the player ────────────────────────────────────────
header_html = (
    f"<b>{indicator}</b> — {INDICATOR_LABELS.get(indicator, '')} &nbsp;|&nbsp; "
    f"<b>Season:</b> {season_label} &nbsp;|&nbsp; "
    f"<b>Baseline:</b> {bp_short} &nbsp;|&nbsp; "
    f"<b>Future:</b> {SSP_LABELS[ssp]} &nbsp;|&nbsp; "
    f"<b>Model:</b> {model_label} &nbsp;|&nbsp; "
    f"<span style='color:#8a4000'>Left = absolute</span> &nbsp;·&nbsp; "
    f"<span style='color:#1a4a9a'>Right = Δ change</span>"
)
 
player_html = build_html_player(
    snap_b64=snap_b64,
    colorbar_b64=colorbar_b64,
    snap_b64_abs=snap_b64_abs,
    colorbar_b64_abs=colorbar_b64_abs,
    snap_years=snap_years,
    frame_years=frame_years_all,
    snap_frame_idx=snap_frame_idx,
    snap_labels=snap_labels,
    snap_colours=SNAP_COLOURS,
    lat_min=lat_min, lat_max=lat_max,
    lon_min=lon_min, lon_max=lon_max,
    frame_ms=frame_ms,
    header_html=header_html,
    dot_opacity=dot_opacity,
    hover_lats=_lat_v.tolist(),
    hover_lons=_lon_v.tolist(),
    snap_vals_list=[row.tolist() for row in _stacked],
    hover_units=units_note,
    snap_abs_vals_list=[row.tolist() for row in _stacked_abs],
    abs_units=abs_units_note,
    snap_fp_ranges=_snap_fp_ranges,
    mask_threshold_deg=_mask_thresh_deg,
    chart_lo_vals=[row.tolist()  for row in _lo_stacked],
    chart_p5_vals=[row.tolist()  for row in _p5_stacked],
    chart_p25_vals=[row.tolist() for row in _p25_stacked],
    chart_p75_vals=[row.tolist() for row in _p75_stacked],
    chart_p95_vals=[row.tolist() for row in _p95_stacked],
    chart_hi_vals=[row.tolist()  for row in _hi_stacked],
    chart_ens_vals=([row.tolist() for row in _ens_stacked]
                    if _ens_stacked is not None else None),
    country_geojson=_borders["country"],
    regions_geojson=_borders["regions"],
    chart_ymin=_chart_ymin,
    chart_ymax=_chart_ymax,
    abs_chart_lo_vals=[row.tolist()  for row in _abs_lo_stacked],
    abs_chart_p5_vals=[row.tolist()  for row in _abs_p5_stacked],
    abs_chart_p25_vals=[row.tolist() for row in _abs_p25_stacked],
    abs_chart_p75_vals=[row.tolist() for row in _abs_p75_stacked],
    abs_chart_p95_vals=[row.tolist() for row in _abs_p95_stacked],
    abs_chart_hi_vals=[row.tolist()  for row in _abs_hi_stacked],
    abs_chart_ens_vals=([row.tolist() for row in _abs_ens_stacked]
                        if _abs_ens_stacked is not None else None),
    abs_chart_ymin=_abs_chart_ymin,
    abs_chart_ymax=_abs_chart_ymax,
)
 
with _map_slot.container():
    _components.html(player_html, height=630, scrolling=False)
 
 
# ── 11. Summary statistics table ──────────────────────────────────────────────
# One column per snapshot; values read from the pre-computed summary_stats
# block in the uncertainty cache. Uses st.metric for compact display.
st.markdown("""
<style>
[data-testid="stMetricValue"] { font-size: 1.05rem !important; font-weight: 600; }
[data-testid="stMetricLabel"] { font-size: 0.75rem !important; }
</style>
""", unsafe_allow_html=True)
 
st.markdown("#### Statistics per snapshot")
cols      = st.columns(len(SNAPSHOTS))
abs_units = INDICATOR_UNITS.get(indicator, "")
 
for i, (label, scen, fp, bp, yr) in enumerate(SNAPSHOTS):
    is_hist = scen == "historical"
    # Match this snapshot to the nearest year in the pre-computed summary list.
    _best_si = int(np.argmin([abs(yr - sy) for sy in _unc_snap_years_pre]))
    s        = _unc_pre["summary_stats"][_best_si]
    clr      = SNAP_COLOURS[min(i, len(SNAP_COLOURS) - 1)]
    units_ch = ("%" if indicator in _REC_INDICATORS
                 else f"Δ {INDICATOR_UNITS.get(indicator, '')}")
    units_ab = (_REC_ABS_UNITS.get(indicator, "")
                if indicator in _REC_INDICATORS
                else INDICATOR_UNITS.get(indicator, ""))
    with cols[i]:
        st.markdown(
            f'<div style="border-left:4px solid {clr};padding-left:8px">'
            f'<strong style="font-size:0.82rem">{label}</strong><br>'
            f'<span style="font-size:0.72rem;color:#888">'
            + (
                "baseline record chance"
                if is_hist and indicator in _REC_INDICATORS
                else "baseline (absolute)"
                if is_hist
                else "record chance %"
                if indicator in _REC_INDICATORS
                else "change from baseline"
            )
            + f' · {s["n_models"]} models'
            + '</span></div>',
            unsafe_allow_html=True,
        )
        if is_hist:
            if s["mean_abs"] is not None:
                st.metric("Mean (abs)",  f'{s["mean_abs"]:.2f} {units_ab}')
                st.metric("Range (abs)", f'{s["min_abs"]:.2f} – {s["max_abs"]:.2f} {units_ab}')
        else:
            if s["mean_change"] is not None:
                st.metric("Mean Δ",  f'{s["mean_change"]:+.2f} {units_ch}')
                st.metric("Range Δ", f'{s["min_change"]:+.2f} – {s["max_change"]:+.2f}')
            if s["mean_abs"] is not None:
                st.metric("Mean (abs)", f'{s["mean_abs"]:.2f} {units_ab}')
 