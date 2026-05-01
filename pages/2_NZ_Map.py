# pages/2_NZ_Map.py — dual-method instant toggle edition
"""
NZ Climate Indicator Map — dual-panel (absolute + change) animated timeline.
Model availability is detected from precomputed uncertainty caches only.
"""

from pathlib import Path
import re, io as _io, os, base64 as _base64, json as _json, hashlib, pickle

import numpy as np
import streamlit as st

st.set_page_config(page_title="NZ Climate Indicator Map", layout="wide")
st.markdown("""
<style>
html, body, [class*="css"]  { font-size: 14px !important; }
h1, h2, h3, h4 { font-size: 1.2rem !important; line-height: 1.2 !important; }
header[data-testid="stHeader"] {
    visibility: hidden;
    height: 0px !important;
    min-height: 0px !important;
    overflow: hidden !important;
}
</style>
<script>
(function() {
    function fixBtn() {
        var btn = document.querySelector('[data-testid="collapsedControl"]');
        if (!btn) return;
        btn.style.setProperty('visibility', 'visible', 'important');
        btn.style.setProperty('display',    'flex',    'important');
        btn.style.setProperty('position',   'fixed',   'important');
        btn.style.setProperty('top',        '0.5rem',  'important');
        btn.style.setProperty('left',       '0.5rem',  'important');
        btn.style.setProperty('z-index',    '999999',  'important');
    }
    fixBtn();
    var obs = new MutationObserver(fixBtn);
    obs.observe(document.body, { childList: true, subtree: true });
})();
</script>
""", unsafe_allow_html=True)

try:
    import xarray as xr;   HAS_XARRAY   = True
except ImportError:        HAS_XARRAY   = False
try:
    import plotly.graph_objects as go; HAS_PLOTLY = True  # noqa
except ImportError:        HAS_PLOTLY   = False
try:
    import geopandas as gpd, shapely
    from shapely.ops import unary_union; HAS_GEOPANDAS = True
except ImportError:        HAS_GEOPANDAS = False

# ── Data roots ────────────────────────────────────────────────────────────────
_ON_CLOUD = not Path("assets/frame_cache").exists()
if _ON_CLOUD:
    from blob_storage import load_pkl_blob, load_json_blob

DATA_ROOT    = Path(os.environ.get("NZMAP_DATA_ROOT",    "/nonexistent/data"))
REC_ROOT     = Path(os.environ.get("NZMAP_REC_ROOT",     "/nonexistent/rec"))
DATA_ROOT_DD = Path(os.environ.get("NZMAP_DATA_ROOT_DD", "/nonexistent/data_dd"))
REC_ROOT_DD  = Path(os.environ.get("NZMAP_REC_ROOT_DD",  "/nonexistent/rec_dd"))

_DEMO_DATA_ROOT = Path("test/demo_data")
_DEMO_MODE = not DATA_ROOT.exists() and _DEMO_DATA_ROOT.exists() and not _ON_CLOUD
if _DEMO_MODE:
    DATA_ROOT = DATA_ROOT_DD = REC_ROOT = REC_ROOT_DD = _DEMO_DATA_ROOT

METHOD_OPTIONS = ["sd", "dd"]
METHOD_LABELS  = {"sd": "Statistical (AI, 12 km)", "dd": "Dynamical (CCAM, 5 km)"}

COASTLINE_SHP = Path("assets/coastlines/nz-coastlines-and-islands-polygons-topo-150k.shp")
REGIONS_SHP   = Path("assets/coastlines/nz-regional-council-boundaries-topo-150k.shp")

YEAR_STEP  = 3
SEASON_ANN = "ANN"

SEASON_LABELS = {"ANN":"Annual","DJF":"Summer (DJF)","MAM":"Autumn (MAM)",
                 "JJA":"Winter (JJA)","SON":"Spring (SON)"}
INDICATOR_LABELS = {
    "DD1mm":"Dry days (<1 mm/day)","PR":"Precipitation","R99p":"99th-pct precipitation",
    "R99pVAL":"99th-pct rainfall value","R99pVALWet":"99th-pct rainfall (wet days)",
    "RR1mm":"Rain days (>=1 mm)","RR25mm":"Heavy rain days (>=25 mm)",
    "Rx1day":"Max 1-day precipitation","FD":"Frost days (Tmin < 0 degC)",
    "TN":"Mean minimum temperature","TNn":"Coldest night (annual min Tmin)",
    "TX":"Mean maximum temperature","TX25":"Days with Tmax > 25 degC",
    "TX30":"Days with Tmax > 30 degC","TXx":"Hottest day (annual max Tmax)",
    "sfcwind":"Mean surface wind speed","Wd10":"Wind days >= 10 m/s",
    "Wd25":"Wind days >= 25 m/s","Wd99pVAL":"99th-pct wind speed value",
    "Wx1day":"Max 1-day wind speed","REC_TXx":"Hot record chance (annual max Tmax)",
    "REC_Rx1day":"Wet record chance (annual max rainfall)",
    "REC_Wx1day":"Wind record chance (annual max wind speed)",
    "REC_TNn":"Cold record chance (annual min Tmin)",
}
INDICATOR_UNITS = {
    "DD1mm":"days/yr","FD":"days/yr","PR":"mm/yr","R99p":"mm/yr","R99pVAL":"mm/day",
    "R99pVALWet":"mm/day","RR1mm":"days/yr","RR25mm":"days/yr","Rx1day":"mm/day",
    "sfcwind":"m/s","TN":"°C","TNn":"°C","TX":"°C","TX25":"days/yr","TX30":"days/yr",
    "TXx":"°C","Wd10":"days/yr","Wd25":"days/yr","Wd99pVAL":"m/s","Wx1day":"m/s",
    "REC_TXx":"%","REC_Rx1day":"%","REC_Wx1day":"%","REC_TNn":"%",
}
_KELVIN_INDICATORS   = {"TX","TXx","TN","TNn"}
_LOG_INDICATORS      = {"PR","RR1mm","RR25mm","Rx1day","R99p","R99pVAL","R99pVALWet",
                        "Wd10","Wd25","Wd99pVAL","Wx1day","sfcwind"}
_SYMLOG_CHANGE_INDICATORS = {"PR","RR1mm","RR25mm","Rx1day","R99p","R99pVAL","R99pVALWet"}
_REC_INDICATORS      = {"REC_TXx","REC_Rx1day","REC_Wx1day","REC_TNn"}
_REC_ABS_UNITS       = {"REC_TXx":"°C","REC_Rx1day":"mm/day","REC_Wx1day":"m/s","REC_TNn":"°C"}
_ZERO_FLOOR_INDICATORS = {"WD10","Wd10","WD25","Wd25","TX25","TX30","FD","RR25mm",
                          "REC_Rx1day","REC_TXx","REC_Wx1day","REC_TNn"}
_LINEAR_ABS_INDICATORS = {"RR1mm","RR25mm","Rx1day","R99p","R99pVAL","R99pVALWet",
                          "Wd10","Wd25","Wd99pVAL","Wx1day","sfcwind",
                          "REC_Rx1day","REC_TXx","REC_Wx1day","REC_TNn_ABS"}

SSP_OPTIONS = ["ssp126","ssp245","ssp370","ssp585"]
SSP_LABELS  = {"ssp126":"SSP1-2.6 — Low emissions","ssp245":"SSP2-4.5 — Moderate emissions",
               "ssp370":"SSP3-7.0 — High emissions","ssp585":"SSP5-8.5 — Very high emissions"}
BP_OPTIONS  = ["bp1995-2014","bp1986-2005"]
BP_LABELS   = {"bp1995-2014":"bp1995-2014  (recent baseline)","bp1986-2005":"bp1986-2005  (earlier baseline)"}

_PRECIP_INDICATORS_GROUP = {"DD1mm","PR","R99p","R99pVAL","R99pVALWet","RR1mm","RR25mm","Rx1day"}
_TEMP_INDICATORS_GROUP   = {"FD","TN","TNn","TX","TX25","TX30","TXx"}
_WIND_INDICATORS_GROUP   = {"sfcwind","Wd10","Wd25","Wd99pVAL","Wx1day"}
_SEPARATOR_PREFIX        = "╌╌"

_MODEL_ENSEMBLE_MEAN = "Ensemble mean (all models)"


# ── Cache-driven model discovery ───────────────────────────────────────────────
@st.cache_data(show_spinner=False, ttl=300)
def get_models_from_cache(indicator, ssp, bp_tag, season):
    """
    Return (models_in_sd, models_in_dd) as frozensets, detected purely from
    precomputed uncertainty caches — the single source of truth for what is
    actually renderable.
    """
    unc_sd = _load_uncertainty_cache_raw(indicator, ssp, bp_tag, season, method="sd")
    unc_dd = _load_uncertainty_cache_raw(indicator, ssp, bp_tag, season, method="dd")

    models_sd = frozenset(unc_sd.get("model_change_vals", {}).keys()) if unc_sd else frozenset()
    models_dd = frozenset(unc_dd.get("model_change_vals", {}).keys()) if unc_dd else frozenset()
    return models_sd, models_dd


def _load_uncertainty_cache_raw(indicator, ssp, bp_tag, season, method="sd"):
    """Low-level cache loader without Streamlit caching (used internally)."""
    key  = hashlib.md5(f"{indicator}|{ssp}|{bp_tag}|{season}".encode()).hexdigest()
    cache_dir = (_uncertainty_cache_dir(method))
    path = cache_dir / f"{key}.pkl"
    if _ON_CLOUD:
        try:
            return load_pkl_blob(path)
        except Exception:
            return None
    if not path.exists():
        return None
    try:
        with open(path, "rb") as f:
            return pickle.load(f)
    except Exception:
        return None


# ── Coordinate helper ─────────────────────────────────────────────────────────
def _pick_coord(ds, priority):
    candidates = list(ds.coords) + list(ds.dims) + list(ds.data_vars)
    for name in priority:
        for c in candidates:
            if c.lower() == name:
                try:
                    if ds[c].ndim == 2:
                        return c
                except Exception:
                    pass
    for name in priority:
        for c in candidates:
            if c.lower() == name:
                return c
    return None


# ── File discovery (kept for non-cloud season/fp discovery) ──────────────────
def _indicator_root(indicator, method="sd"):
    if method == "dd": return REC_ROOT_DD if indicator in _REC_INDICATORS else DATA_ROOT_DD
    return REC_ROOT if indicator in _REC_INDICATORS else DATA_ROOT

def fp_centre_year(fp_tag):
    m = re.search(r"(\d{4})-(\d{4})", fp_tag)
    return (int(m.group(1)) + int(m.group(2))) / 2 if m else 0.0

def list_indicators(scenario, method="sd"):
    results = set()
    roots = ((DATA_ROOT_DD, REC_ROOT_DD) if method == "dd" else (DATA_ROOT, REC_ROOT))
    for root in roots:
        base = root / scenario / "static_maps"
        if base.exists():
            results.update(d.name for d in base.iterdir() if d.is_dir())
    return sorted(results)

def discover_seasons(scenario, indicator, method="sd"):
    folder = _indicator_root(indicator, method) / scenario / "static_maps" / indicator
    if not folder.exists(): return [SEASON_ANN]
    seasons = set()
    for f in folder.glob("*.nc"):
        for s in SEASON_LABELS:
            if f"_{s}_" in f.name: seasons.add(s)
    order = list(SEASON_LABELS)
    return sorted(seasons, key=lambda s: order.index(s) if s in order else 99) or [SEASON_ANN]

def discover_fp_tags(scenario, indicator, bp_tag, season, method="sd"):
    folder = _indicator_root(indicator, method) / scenario / "static_maps" / indicator
    if not folder.exists(): return []
    tags = set()
    if scenario == "historical":
        for f in folder.glob("*.nc"):
            if "_base_" in f.name and f"_{season}_" in f.name:
                m = re.search(r"_base_(bp\d{4}-\d{4})_", f.name)
                if m: tags.add(m.group(1))
        return sorted(tags, key=lambda t: int(re.search(r"(\d{4})", t).group(1)))
    for f in folder.glob("*.nc"):
        if f"_{bp_tag}_" in f.name and f"_{season}_" in f.name:
            m = re.search(r"_change_(fp\d{4}-\d{4})_", f.name)
            if m: tags.add(m.group(1))
    return sorted(tags, key=lambda t: int(re.search(r"fp(\d{4})", t).group(1)))

def find_nc_files(scenario, indicator, fp_tag, bp_tag, season, model_key=None, method="sd"):
    folder = _indicator_root(indicator, method) / scenario / "static_maps" / indicator
    if not folder.exists(): return []
    if model_key:
        if method == "dd": model_pat = re.compile(r"_" + re.escape(model_key) + r"_CCAM_")
        else: model_pat = re.compile(r"_" + re.escape(model_key) + r"_[ri]\d+[ip]\d+[pf]\d+")
    else: model_pat = None
    def _ok(f):
        if model_pat and not model_pat.search(f.name): return False
        if method == "dd" and not f.name.endswith("_bc.nc"): return False
        return f"_{season}_" in f.name
    if scenario == "historical":
        return sorted(f for f in folder.glob("*.nc") if f"_base_{fp_tag}_" in f.name and _ok(f))
    return sorted(f for f in folder.glob("*.nc")
                  if f"_{fp_tag}_" in f.name and f"_{bp_tag}_" in f.name and _ok(f))

def _build_grouped_indicator_options(indicators):
    precip = sorted(i for i in indicators if i in _PRECIP_INDICATORS_GROUP)
    temp   = sorted(i for i in indicators if i in _TEMP_INDICATORS_GROUP)
    wind   = sorted(i for i in indicators if i in _WIND_INDICATORS_GROUP)
    rec    = sorted(i for i in indicators if i in _REC_INDICATORS)
    other  = sorted(i for i in indicators
                    if i not in _PRECIP_INDICATORS_GROUP | _TEMP_INDICATORS_GROUP
                                | _WIND_INDICATORS_GROUP | _REC_INDICATORS)
    opts = []
    if precip: opts += [f"{_SEPARATOR_PREFIX} Precipitation"] + precip
    if temp:   opts += [f"{_SEPARATOR_PREFIX} Temperature"]   + temp
    if wind:   opts += [f"{_SEPARATOR_PREFIX} Wind"]          + wind
    if other:  opts += [f"{_SEPARATOR_PREFIX} Other"]         + other
    if rec:    opts += [f"{_SEPARATOR_PREFIX}━ Record Indicators ━"] + rec
    return opts


# ── Data loaders ──────────────────────────────────────────────────────────────
def load_ensemble_mean(scenario, indicator, fp_tag, bp_tag, season,
                       model_key=None, var_name=None, method="sd"):
    files = find_nc_files(scenario, indicator, fp_tag, bp_tag, season, model_key, method=method)
    if not files: return None, None, None, 0
    from collections import defaultdict
    if method == "dd":
        model_pat = re.compile(r"_(.+?)_CCAM_")
    else:
        model_pat = re.compile(r"_([^_]+(?:_[^_]+)?)_r\d+i\d+p\d+")
    model_groups = defaultdict(list)
    ungrouped = []
    for f in files:
        m = model_pat.search(f.name)
        if m: model_groups[m.group(1)].append(f)
        else: ungrouped.append(f)
    lats = lons = None
    model_means = []

    def _load_array(f):
        nonlocal lats, lons
        try:
            ds = xr.open_dataset(f, engine="netcdf4", mask_and_scale=False)
            lat_name = _pick_coord(ds, ["lat", "latitude", "rlat", "y"])
            lon_name = _pick_coord(ds, ["lon", "longitude", "rlon", "x"])
            if not lat_name or not lon_name:
                ds.close(); return None
            data_vars = [v for v in ds.data_vars if v.lower() not in ("lat","lon","time")]
            if not data_vars: ds.close(); return None
            chosen_var = var_name if (var_name and var_name in ds.data_vars) else data_vars[0]
            da = ds[chosen_var]
            for dim in [d for d in da.dims if d not in (lat_name, lon_name)]:
                da = da.isel({dim: 0})
            arr = da.values.astype(float)
            fv = da.attrs.get("_FillValue", da.attrs.get("missing_value"))
            if fv is not None: arr[arr == float(fv)] = np.nan
            arr = arr * float(da.attrs.get("scale_factor", 1.0)) + float(da.attrs.get("add_offset", 0.0))
            if lats is None:
                raw_lats = ds[lat_name].values
                raw_lons = ds[lon_name].values
                if raw_lats.ndim == 2:
                    lats = raw_lats.ravel()
                    lons = raw_lons.ravel()
                else:
                    lg, ng = np.meshgrid(raw_lats, raw_lons, indexing="ij")
                    lats = lg.ravel()
                    lons = ng.ravel()
            ds.close(); return arr
        except Exception: return None

    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        for model_name, model_files in model_groups.items():
            run_arrays = [a for a in (_load_array(f) for f in model_files) if a is not None]
            if run_arrays:
                model_means.append(np.nanmean(np.stack(run_arrays, 0), 0))
        for f in ungrouped:
            a = _load_array(f)
            if a is not None: model_means.append(a)
        if not model_means: return None, None, None, 0
        result = np.nanmean(np.stack(model_means, 0), 0)
    return result, lats, lons, len(model_means)

def build_timeline(indicator, bp_tag, ssp, season, method="sd"):
    snapshots = []
    for period in [p for p in discover_fp_tags("historical", indicator, bp_tag, season, method=method) if p == bp_tag]:
        yr = fp_centre_year(period)
        snapshots.append((f"Historical {period.replace('bp','').replace('-','–')}",
                         "historical", period, period, yr))
    for fp in discover_fp_tags(ssp, indicator, bp_tag, season, method=method):
        yr = fp_centre_year(fp)
        snapshots.append((fp.replace("fp","").replace("-","–"), ssp, fp, bp_tag, yr))
    return sorted(snapshots, key=lambda x: x[4])


# ── Colour ranges ─────────────────────────────────────────────────────────────
_COLOR_RANGES_PATH     = Path(__file__).parent.parent / "assets/color_ranges/color_ranges.json"
_ABS_COLOR_RANGES_PATH = Path(__file__).parent.parent / "assets/color_ranges/abs_color_ranges.json"
_PRECOMPUTED_RANGES, _PRECOMPUTED_ABS_RANGES = {}, {}
if _ON_CLOUD:
    _PRECOMPUTED_RANGES     = load_json_blob(_COLOR_RANGES_PATH)
    _PRECOMPUTED_ABS_RANGES = load_json_blob(_ABS_COLOR_RANGES_PATH)
else:
    for _path, _store in ((_COLOR_RANGES_PATH, _PRECOMPUTED_RANGES),
                           (_ABS_COLOR_RANGES_PATH, _PRECOMPUTED_ABS_RANGES)):
        if _path.exists():
            try:
                with open(_path) as _f: _store.update(_json.load(_f))
            except Exception as e: st.warning(f"Could not load {_path.name}: {e}")

@st.cache_data(show_spinner=False)
def compute_color_range(indicator):
    ranges = {}
    if _COLOR_RANGES_PATH.exists():
        with open(_COLOR_RANGES_PATH) as f: ranges = _json.load(f)
    if indicator in ranges: return ranges[indicator]
    if indicator in _PRECOMPUTED_RANGES: return _PRECOMPUTED_RANGES[indicator]
    st.toast(f"'{indicator}' not in color_ranges.json — computing live.", icon="⚠️")
    all_vals = []
    for ssp_opt in SSP_OPTIONS:
        for bp in BP_OPTIONS:
            for season in discover_seasons("historical", indicator):
                for fp in discover_fp_tags(ssp_opt, indicator, bp, season):
                    data, _, _, _ = load_ensemble_mean(ssp_opt, indicator, fp, bp, season, None)
                    if data is not None:
                        finite = data[np.isfinite(data)].ravel()
                        if len(finite): all_vals.append(finite)
    if not all_vals: return 1.0
    combined = np.concatenate(all_vals)
    pct = 99 if indicator in _SYMLOG_CHANGE_INDICATORS else 98
    return max(float(np.nanpercentile(np.abs(combined), pct)), 1e-6)

@st.cache_data(show_spinner=False)
def compute_abs_color_range(indicator):
    abs_ranges = {}
    if _ABS_COLOR_RANGES_PATH.exists():
        with open(_ABS_COLOR_RANGES_PATH) as f: abs_ranges = _json.load(f)
    if indicator in abs_ranges:
        r = abs_ranges[indicator]; vmin, vmax = float(r["vmin"]), float(r["vmax"])
    if indicator in _REC_INDICATORS:
        if indicator == "REC_TXx":    return 20.0, 45.0
        if indicator == "REC_TNn":    return -15.0, 10.0
        if indicator == "REC_Rx1day": return 50.0, 400.0
        if indicator == "REC_Wx1day": return 10.0, 40.0
        return 0.0, 1.0
    if indicator in _PRECOMPUTED_ABS_RANGES:
        r = _PRECOMPUTED_ABS_RANGES[indicator]; vmin, vmax = float(r["vmin"]), float(r["vmax"])
        if indicator in _ZERO_FLOOR_INDICATORS: vmin = 0.0
        if indicator in _LOG_INDICATORS: vmin = max(vmin, 1e-3)
        return vmin, vmax
    return 0.0, 1.0

def colorscale_for(indicator):
    if indicator in {"TX","TXx","TX25","TX30","TN","TNn"}: return "RdBu_r"
    if indicator == "FD": return "RdBu"
    if indicator in {"PR","RR1mm","RR25mm","Rx1day","R99p","R99pVAL","R99pVALWet"}: return "BrBG"
    if indicator == "DD1mm": return "BrBG_r"
    if indicator in {"sfcwind","Wd10","Wd25","Wd99pVAL","Wx1day"}: return "PuOr"
    if indicator in _REC_INDICATORS: return "RdYlGn_r"
    return "RdBu_r"

def colorscale_abs_for(indicator):
    if indicator == "FD": return "Blues"
    if indicator in {"TN","TNn"}: return "RdYlBu_r"
    if indicator in {"TX","TXx","TX25","TX30"}: return "YlOrRd"
    if indicator in {"PR","RR1mm","RR25mm","Rx1day","R99p","R99pVAL","R99pVALWet"}: return "YlGnBu"
    if indicator in {"sfcwind","Wd10","Wd25","Wd99pVAL","Wx1day"}: return "Purples"
    if indicator == "DD1mm": return "YlOrBr"
    if indicator == "REC_TXx": return "YlOrRd"
    if indicator == "REC_TNn": return "RdYlBu_r"
    if indicator == "REC_Rx1day": return "YlGnBu"
    if indicator == "REC_Wx1day": return "Purples"
    return "viridis"

def _mpl_cmap(plotly_name):
    import matplotlib
    if not plotly_name: return matplotlib.colormaps["RdBu_r"]
    for name in (plotly_name, plotly_name.lower()):
        try: return matplotlib.colormaps[name]
        except KeyError: continue
    return matplotlib.colormaps["RdBu_r"]

def _make_norm(indicator, vmin, vmax, is_change):
    import matplotlib.colors as mcolors
    if is_change and indicator in _SYMLOG_CHANGE_INDICATORS:
        linthresh = max(abs(vmax) * 0.10, 1e-3)
        return mcolors.SymLogNorm(linthresh=linthresh, linscale=0.5,
                                  vmin=vmin, vmax=vmax, base=10)
    if (not is_change and indicator in _LOG_INDICATORS
            and indicator not in _LINEAR_ABS_INDICATORS):
        return mcolors.LogNorm(vmin=max(vmin, 1e-3), vmax=vmax)
    return mcolors.Normalize(vmin=vmin, vmax=vmax)

def _log_mode(indicator, is_change):
    if is_change and indicator in _SYMLOG_CHANGE_INDICATORS: return "symlog"
    if not is_change and indicator in _LOG_INDICATORS: return "log"
    return "linear"


# ── Frame cache ───────────────────────────────────────────────────────────────
FRAME_CACHE_DIR_SD = Path("assets/frame_cache")
FRAME_CACHE_DIR_DD = Path("assets/frame_cache_dd")
if not _ON_CLOUD:
    FRAME_CACHE_DIR_SD.mkdir(parents=True, exist_ok=True)
    FRAME_CACHE_DIR_DD.mkdir(parents=True, exist_ok=True)
FRAME_CACHE_DIR = FRAME_CACHE_DIR_SD

def _frame_cache_dir(method="sd"):
    return FRAME_CACHE_DIR_DD if method == "dd" else FRAME_CACHE_DIR_SD

def _frame_cache_key(indicator, ssp, bp_tag, season, model_key,
                     colorscale, vmin, vmax, log_mode="linear"):
    parts = (f"{indicator}|{ssp}|{bp_tag}|{season}|"
             f"{model_key or 'ensemble'}|{colorscale}|"
             f"{vmin:.6f}|{vmax:.6f}|{log_mode}")
    return hashlib.md5(parts.encode()).hexdigest()

def _load_frame_cache(key, method="sd"):
    path = _frame_cache_dir(method) / f"{key}.pkl"
    if _ON_CLOUD:
        return load_pkl_blob(path)
    if not path.exists():
        return None
    try:
        with open(path, "rb") as f: return pickle.load(f)
    except Exception: return None

def _save_frame_cache(key, payload, method="sd"):
    if _ON_CLOUD:
        return
    path = _frame_cache_dir(method) / f"{key}.pkl"
    with open(path, "wb") as f: pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)


# ── Uncertainty cache ─────────────────────────────────────────────────────────
def _uncertainty_cache_dir(method="sd"):
    if _DEMO_MODE: return _DEMO_DATA_ROOT / "uncertainty_cache"
    return (Path("assets/uncertainty_cache_dd") if method == "dd"
            else Path("assets/uncertainty_cache"))

UNCERTAINTY_CACHE_DIR = _uncertainty_cache_dir("sd")

def _uncertainty_cache_key(indicator, ssp, bp_tag, season):
    return hashlib.md5(f"{indicator}|{ssp}|{bp_tag}|{season}".encode()).hexdigest()

@st.cache_data(show_spinner=False)
def load_uncertainty_cache(indicator, ssp, bp_tag, season, method="sd"):
    key  = _uncertainty_cache_key(indicator, ssp, bp_tag, season)
    path = _uncertainty_cache_dir(method) / f"{key}.pkl"
    if _ON_CLOUD:
        return load_pkl_blob(path)
    if not path.exists():
        return None
    try:
        with open(path, "rb") as f: return pickle.load(f)
    except Exception: return None


# ── Virtual frame timeline ────────────────────────────────────────────────────
def compute_frame_timeline(snap_years, year_step):
    frame_years, snap_frame_idx = [], []
    fi = 0
    for i in range(len(snap_years) - 1):
        snap_frame_idx.append(fi)
        ya, yb = snap_years[i], snap_years[i + 1]
        n_steps = max(1, round((yb - ya) / year_step))
        for step in range(n_steps):
            frame_years.append(ya + step / n_steps * (yb - ya)); fi += 1
    snap_frame_idx.append(fi); frame_years.append(snap_years[-1])
    return frame_years, snap_frame_idx


# ── Geometry loaders ──────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_coastline_polygon():
    if not HAS_GEOPANDAS: return None
    if not COASTLINE_SHP.exists():
        st.warning(f"Coastline shapefile not found at `{COASTLINE_SHP}`. Falling back to KDTree land mask.")
        return None
    try:
        gdf = gpd.read_file(COASTLINE_SHP)
        if gdf.crs is not None and gdf.crs.to_epsg() != 4326: gdf = gdf.to_crs(epsg=4326)
        elif gdf.crs is None: gdf = gdf.set_crs(epsg=4326, allow_override=True)
        bds = gdf.total_bounds
        if not (-200 < bds[0] < 200 and -90 < bds[1] < 90):
            st.warning(f"Coastline bounds look wrong ({bds}). Using KDTree fallback."); return None
        poly_geoms = [g for g in gdf.geometry if g is not None and g.geom_type in ("Polygon","MultiPolygon")]
        return unary_union(poly_geoms) if poly_geoms else None
    except Exception as exc:
        st.warning(f"Could not load coastline shapefile: {exc}. Using KDTree fallback."); return None

@st.cache_resource(show_spinner=False)
def nz_loader_svg_data():
    from shapely.geometry import MultiPolygon
    poly = load_coastline_polygon()
    if poly is None: return None
    if poly.geom_type == "MultiPolygon":
        kept = [p for p in poly.geoms if 160 < p.centroid.x < 180 and p.area > 0.01]
        if kept: poly = MultiPolygon(kept) if len(kept) > 1 else kept[0]
    minx, miny, maxx, maxy = poly.bounds
    y_flip = maxy + miny
    simplified = poly.simplify(0.01, preserve_topology=True)
    def ring_d(coords, yf):
        parts = []
        for i, xy in enumerate(coords):
            parts.append(f"{'M' if i==0 else 'L'}{float(xy[0]):.3f},{(yf-float(xy[1])):.3f}")
        return "".join(parts) + "Z"
    def poly_d(p, yf):
        return ring_d(p.exterior.coords, yf) + "".join(ring_d(i.coords, yf) for i in p.interiors)
    d = ("".join(poly_d(p, y_flip) for p in simplified.geoms)
         if simplified.geom_type == "MultiPolygon" else poly_d(simplified, y_flip))
    pad = 0.2
    return {"d": d, "viewBox": f"{minx-pad:.3f} {miny-pad:.3f} {(maxx-minx+2*pad):.3f} {(maxy-miny+2*pad):.3f}"}

@st.cache_resource(show_spinner=False)
def load_borders_geojson():
    out = {"country": None, "regions": None}
    if not HAS_GEOPANDAS: return out
    SIMPLIFY_TOL = 0.005
    coast = load_coastline_polygon()
    if coast is not None:
        try: out["country"] = _json.dumps(coast.simplify(SIMPLIFY_TOL, preserve_topology=True).__geo_interface__)
        except Exception: pass
    if REGIONS_SHP.exists():
        try:
            gdf = gpd.read_file(REGIONS_SHP)
            if gdf.crs is not None and gdf.crs.to_epsg() != 4326: gdf = gdf.to_crs(epsg=4326)
            elif gdf.crs is None: gdf = gdf.set_crs(epsg=4326, allow_override=True)
            gdf["geometry"] = gdf.geometry.simplify(SIMPLIFY_TOL, preserve_topology=True)
            out["regions"] = gdf.to_json()
        except Exception: pass
    return out


# ── Snapshot renderer ─────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False, max_entries=24)
def prerender_snapshots(lat_v, lon_v, snap_data_stacked, vmin, vmax,
                        colorscale, indicator="", is_change=False):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.colors as mcolors  # noqa
    from scipy.spatial import Delaunay, KDTree
    from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator
    from scipy.ndimage import distance_transform_edt, gaussian_filter as _gf
    from PIL import Image as _PILImage

    cmap   = _mpl_cmap(colorscale)
    norm   = _make_norm(indicator, vmin, vmax, is_change)
    n_snaps = snap_data_stacked.shape[0]

    def merc(lat_deg): return np.log(np.tan(np.pi/4 + np.deg2rad(lat_deg)/2))
    lat_min = float(np.nanmin(lat_v)) - 0.3; lat_max = float(np.nanmax(lat_v)) + 0.3
    lon_min = float(np.nanmin(lon_v)) - 0.3; lon_max = float(np.nanmax(lon_v)) + 0.3
    lon_v_rad = np.deg2rad(lon_v); merc_v = merc(lat_v)
    lon_min_rad = np.deg2rad(lon_min); lon_max_rad = np.deg2rad(lon_max)
    merc_min = merc(lat_min); merc_max = merc(lat_max)
    fig_w = 5.0; fig_h = fig_w * ((merc_max - merc_min) / (lon_max_rad - lon_min_rad))
    DPI = 200; out_w = int(fig_w * DPI); out_h = int(fig_h * DPI)
    xi = np.linspace(lon_min_rad, lon_max_rad, out_w)
    yi = np.linspace(merc_min, merc_max, out_h)
    xi_grid, yi_grid = np.meshgrid(xi, yi)
    xi_deg = np.rad2deg(xi_grid)
    yi_deg = np.rad2deg(2.0 * np.arctan(np.exp(yi_grid)) - np.pi/2)
    points_merc = np.column_stack([lon_v_rad, merc_v])
    kd = KDTree(points_merc)
    sample_n = min(2000, len(points_merc))
    rng = np.random.default_rng(0)
    sample_idx = rng.choice(len(points_merc), sample_n, replace=False)
    nn_dists, _ = kd.query(points_merc[sample_idx], k=2)
    grid_spacing_rad = float(np.median(nn_dists[:, 1]))
    mask_threshold = grid_spacing_rad * 1.1
    coast_poly = load_coastline_polygon()
    if coast_poly is not None:
        pixel_size_deg = (lon_max - lon_min) / out_w
        coast_buffered = coast_poly.buffer(pixel_size_deg * 0.2)
        land_mask = shapely.contains_xy(coast_buffered, xi_deg.ravel(), yi_deg.ravel()).reshape(out_h, out_w)
    else:
        pixel_pts = np.column_stack([xi_grid.ravel(), yi_grid.ravel()])
        pixel_dists, _ = kd.query(pixel_pts, k=1, workers=-1)
        land_mask = (pixel_dists <= mask_threshold).reshape(out_h, out_w)
    _, _near_idx = distance_transform_edt(~land_mask, return_indices=True)
    near_row, near_col = _near_idx[0], _near_idx[1]
    _hard_alpha = land_mask.astype(np.float32)
    _grid_spacing_px = grid_spacing_rad / (lon_max_rad - lon_min_rad) * out_w
    _SIGMA = max(6.0, _grid_spacing_px * 2.2)

    def _smooth_nan(arr2d):
        valid = np.isfinite(arr2d) & land_mask
        filled = np.where(valid, arr2d, 0.0)
        wt_num = _gf(filled.astype(np.float64), sigma=_SIGMA)
        wt_den = _gf(valid.astype(np.float64), sigma=_SIGMA)
        with np.errstate(invalid="ignore"):
            return np.where(wt_den > 1e-6, wt_num / wt_den, np.nan)

    def _render(data_1d):
        data_f = data_1d.astype(float)
        valid_src = np.isfinite(data_f)
        if not np.any(valid_src):
            blank = np.zeros((out_h, out_w, 4), dtype=np.uint8)
            img = _PILImage.fromarray(blank[::-1], mode="RGBA")
            buf = _io.BytesIO(); img.save(buf, format="PNG", optimize=False); buf.seek(0)
            b64 = _base64.b64encode(buf.read()).decode(); buf.close(); return b64
        pts_valid = points_merc[valid_src]; data_valid = data_f[valid_src]
        nearest_interp = NearestNDInterpolator(pts_valid, data_valid)
        zi = nearest_interp(xi_grid, yi_grid)
        zi_land = np.where(land_mask, zi, np.nan); zi = _smooth_nan(zi_land)
        still_nan = ~np.isfinite(zi) & land_mask
        if np.any(still_nan): zi[still_nan] = nearest_interp(xi_grid[still_nan], yi_grid[still_nan])
        tri_valid = Delaunay(pts_valid)
        linear_interp = LinearNDInterpolator(tri_valid, data_valid)
        zi_lin = linear_interp(xi_grid, yi_grid)
        interior = np.isfinite(zi_lin); zi[interior] = zi_lin[interior]
        zi_mapped = np.where(np.isfinite(zi), zi, norm.vmin)
        rgba_grid = cmap(norm(zi_mapped))
        rgba_grid[:, :, :3] = rgba_grid[near_row, near_col, :3]
        rgba_grid[:, :, 3] = _hard_alpha
        nan_land = ~np.isfinite(zi) & land_mask
        if np.any(nan_land): rgba_grid[nan_land, 3] = 0.0
        rgba_uint8 = (rgba_grid[::-1] * 255).clip(0, 255).astype(np.uint8)
        img = _PILImage.fromarray(rgba_uint8, mode="RGBA")
        buf = _io.BytesIO(); img.save(buf, format="PNG", optimize=False); buf.seek(0)
        b64 = _base64.b64encode(buf.read()).decode(); buf.close(); return b64

    snap_b64 = [_render(snap_data_stacked[i]) for i in range(n_snaps)]
    return snap_b64, lat_min, lat_max, lon_min, lon_max, float(np.rad2deg(mask_threshold))


# ── Colourbar renderer ────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False, max_entries=24)
def render_colorbar_b64(vmin, vmax, colorscale, units, indicator="", is_change=False):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt, matplotlib.colors as mcolors, matplotlib.ticker as mticker
    cmap = _mpl_cmap(colorscale); norm = _make_norm(indicator, vmin, vmax, is_change)
    fig, ax = plt.subplots(figsize=(1.0, 6.5)); fig.patch.set_alpha(0)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm); sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, fraction=1.0, pad=0)
    if isinstance(norm, mcolors.LogNorm):
        cbar.ax.yaxis.set_major_locator(mticker.LogLocator(base=10, subs=[1,2,3,5], numticks=12))
        cbar.ax.yaxis.set_minor_locator(mticker.LogLocator(base=10, subs=np.arange(1,10), numticks=50))
        cbar.ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:g}"))
        cbar.ax.yaxis.set_minor_formatter(mticker.NullFormatter())
    elif isinstance(norm, mcolors.SymLogNorm):
        cbar.ax.yaxis.set_major_locator(mticker.SymmetricalLogLocator(base=10, linthresh=norm.linthresh, subs=[1,2,3,5]))
        cbar.ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:g}"))
    cbar.set_label(units, fontsize=14); cbar.ax.tick_params(labelsize=12)
    ax.remove(); fig.tight_layout(pad=0.2)
    buf = _io.BytesIO(); fig.savefig(buf, format="png", transparent=True, dpi=150, bbox_inches="tight")
    buf.seek(0); b64 = _base64.b64encode(buf.read()).decode(); buf.close(); plt.close(fig)
    return b64


# ============================================================================
# build_html_player
# ============================================================================
def build_html_player(
    sd_snap_b64, sd_snap_b64_abs,
    sd_colorbar_b64, sd_colorbar_b64_abs,
    sd_hover_vals, sd_hover_abs_vals,
    sd_chart_p5, sd_chart_p25, sd_chart_p75, sd_chart_p95, sd_chart_ens,
    sd_abs_chart_p5, sd_abs_chart_p25, sd_abs_chart_p75, sd_abs_chart_p95, sd_abs_chart_ens,
    sd_chart_ymin, sd_chart_ymax, sd_abs_chart_ymin, sd_abs_chart_ymax,
    sd_snap_fp_ranges,
    dd_snap_b64, dd_snap_b64_abs,
    dd_colorbar_b64, dd_colorbar_b64_abs,
    dd_hover_vals, dd_hover_abs_vals,
    dd_chart_p5, dd_chart_p25, dd_chart_p75, dd_chart_p95, dd_chart_ens,
    dd_abs_chart_p5, dd_abs_chart_p25, dd_abs_chart_p75, dd_abs_chart_p95, dd_abs_chart_ens,
    dd_chart_ymin, dd_chart_ymax, dd_abs_chart_ymin, dd_abs_chart_ymax,
    dd_snap_fp_ranges,
    sd_hover_lats, sd_hover_lons,
    dd_hover_lats, dd_hover_lons,
    snap_years, frame_years, snap_frame_idx, snap_labels,
    hover_units, abs_units,
    lat_min, lat_max, lon_min, lon_max,
    mask_threshold_deg,
    dd_lat_min, dd_lat_max, dd_lon_min, dd_lon_max,
    dd_mask_threshold_deg,
    frame_ms, dot_opacity, header_html,
    initial_method,
    locked_method=None,
    country_geojson=None, regions_geojson=None,
) -> str:
    import json as _json

    n_frames   = len(frame_years)
    vp_lat_min = min(lat_min, dd_lat_min)
    vp_lat_max = max(lat_max, dd_lat_max)
    vp_lon_min = min(lon_min, dd_lon_min)
    vp_lon_max = max(lon_max, dd_lon_max)
    center_lat = (vp_lat_min + vp_lat_max) / 2
    center_lon = (vp_lon_min + vp_lon_max) / 2

    def _enc(arr):
        if arr is None: return "null"
        return _json.dumps([[round(float(x), 3) for x in row] for row in arr])

    hover_units_js = _json.dumps(hover_units)
    abs_units_js   = _json.dumps(abs_units)
    snap_years_js  = _json.dumps(snap_years)
    frame_years_js = _json.dumps(frame_years)
    snap_idx_js    = _json.dumps(snap_frame_idx)
    snap_labels_js = _json.dumps(snap_labels)
    sd_hover_lats_js = _json.dumps([round(v, 4) for v in sd_hover_lats])
    sd_hover_lons_js = _json.dumps([round(v, 4) for v in sd_hover_lons])
    dd_hover_lats_js = _json.dumps([round(v, 4) for v in dd_hover_lats])
    dd_hover_lons_js = _json.dumps([round(v, 4) for v in dd_hover_lons])
    sd_snap_b64_js   = _json.dumps(sd_snap_b64)
    sd_snap_abs_js   = _json.dumps(sd_snap_b64_abs)
    dd_snap_b64_js   = _json.dumps(dd_snap_b64)
    dd_snap_abs_js   = _json.dumps(dd_snap_b64_abs)
    country_geojson_js = country_geojson if country_geojson else "null"
    regions_geojson_js = regions_geojson if regions_geojson else "null"
    init_method_js     = _json.dumps(initial_method)
    locked_method_js   = _json.dumps(locked_method)
    sd_vals_js     = _enc(sd_hover_vals)
    sd_abs_vals_js = _enc(sd_hover_abs_vals)
    dd_vals_js     = _enc(dd_hover_vals)
    dd_abs_vals_js = _enc(dd_hover_abs_vals)
    sd_fp_ranges_js = _json.dumps(sd_snap_fp_ranges if sd_snap_fp_ranges else snap_labels)
    dd_fp_ranges_js = _json.dumps(dd_snap_fp_ranges if dd_snap_fp_ranges else snap_labels)
    sd_p5_js  = _enc(sd_chart_p5);  sd_p25_js = _enc(sd_chart_p25)
    sd_p75_js = _enc(sd_chart_p75); sd_p95_js = _enc(sd_chart_p95)
    sd_ens_js = _enc(sd_chart_ens)
    dd_p5_js  = _enc(dd_chart_p5);  dd_p25_js = _enc(dd_chart_p25)
    dd_p75_js = _enc(dd_chart_p75); dd_p95_js = _enc(dd_chart_p95)
    dd_ens_js = _enc(dd_chart_ens)
    sd_ap5_js  = _enc(sd_abs_chart_p5);  sd_ap25_js = _enc(sd_abs_chart_p25)
    sd_ap75_js = _enc(sd_abs_chart_p75); sd_ap95_js = _enc(sd_abs_chart_p95)
    sd_aens_js = _enc(sd_abs_chart_ens)
    dd_ap5_js  = _enc(dd_abs_chart_p5);  dd_ap25_js = _enc(dd_abs_chart_p25)
    dd_ap75_js = _enc(dd_abs_chart_p75); dd_ap95_js = _enc(dd_abs_chart_p95)
    dd_aens_js = _enc(dd_abs_chart_ens)
    sd_ymin_js  = _json.dumps(sd_chart_ymin);  sd_ymax_js  = _json.dumps(sd_chart_ymax)
    dd_ymin_js  = _json.dumps(dd_chart_ymin);  dd_ymax_js  = _json.dumps(dd_chart_ymax)
    sd_aymin_js = _json.dumps(sd_abs_chart_ymin); sd_aymax_js = _json.dumps(sd_abs_chart_ymax)
    dd_aymin_js = _json.dumps(dd_abs_chart_ymin); dd_aymax_js = _json.dumps(dd_abs_chart_ymax)
    sd_cb_chg_js = _json.dumps(f"data:image/png;base64,{sd_colorbar_b64}")
    sd_cb_abs_js = _json.dumps(f"data:image/png;base64,{sd_colorbar_b64_abs}")
    dd_cb_chg_js = _json.dumps(f"data:image/png;base64,{dd_colorbar_b64}")
    dd_cb_abs_js = _json.dumps(f"data:image/png;base64,{dd_colorbar_b64_abs}")

    return f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<link rel="stylesheet"
  href="https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.9.4/leaflet.min.css">
<script src="https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.9.4/leaflet.min.js"></script>
<script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.1/chart.umd.min.js"></script>
<style>
* {{ box-sizing:border-box; margin:0; padding:0; }}
body {{ font-family:Arial,sans-serif; background:white; overflow-x:hidden; }}
#header {{ padding:5px 12px 2px; font-size:11px; color:#444; border-bottom:1px solid #eee; }}
#maps-row {{ display:flex; gap:5px; width:100%; padding:0 4px; }}
.map-panel {{ flex:1; min-width:0; display:flex; flex-direction:column; }}
.panel-title {{ text-align:center; font-size:14px; font-weight:700; letter-spacing:0.04em;
                padding:6px 0 5px; border-radius:4px 4px 0 0; margin-bottom:2px; }}
.panel-title.change-title {{ background:#e8f0fb; color:#1a4a9a; }}
.panel-title.abs-title    {{ background:#fdf3e8; color:#8a4000; }}
.map-wrap {{ position:relative; height:510px; }}
.map-div  {{ width:100%; height:100%; }}
.colorbar-box {{ position:absolute; right:8px; top:8px; z-index:999;
                 background:rgba(255,255,255,0.88); border-radius:5px;
                 padding:4px; box-shadow:0 1px 4px rgba(0,0,0,.15); }}
.colorbar-box img {{ height:290px; display:block; }}
#controls {{ padding:10px 48px 0 12px; overflow:visible; }}
#btn-row  {{ display:flex; align-items:flex-start; gap:8px; overflow:visible; }}
#slider-wrap {{ flex:1; display:flex; flex-direction:column; padding-top:0; margin-left:12px; overflow:visible; }}
.ctrl-btn {{ padding:4px 13px; font-size:12px; cursor:pointer;
             border:1px solid #bbb; border-radius:4px; background:#f4f4f4;
             margin-top:7px; flex-shrink:0; width:80px; text-align:center; }}
.ctrl-btn:hover {{ background:#e2e2e2; }}
#timeline-slider {{ width:100%; cursor:pointer; accent-color:#4a90d9; margin:4px 0 0; }}
#tick-row {{ position:relative; width:100%; height:70px; overflow:visible; margin-top:0; }}
.tick {{ position:absolute; display:flex; flex-direction:column; align-items:center;
         pointer-events:none; user-select:none; transform:translateX(-50%); }}
.tick.snap-tick {{ top:-28px; }}
.tick-text-snap {{ font-size:10px; font-weight:700; color:#222; white-space:nowrap; margin-bottom:3px; }}
.snap-line {{ width:1px; height:14px; background:#888; }}
.tick.year-tick {{ top:0; }}
.tick-line {{ width:1px; height:5px; background:#bbb; }}
.tick-text-year {{ font-size:9px; color:#666; white-space:nowrap;
                   transform:rotate(-45deg) translateX(-4px);
                   transform-origin:top left; margin-top:12px; }}
#hover-tip {{
  position:fixed; z-index:9999; pointer-events:none;
  background:rgba(20,20,20,0.85); color:#fff;
  border-radius:6px; padding:6px 10px; font-size:12px; line-height:1.55;
  white-space:nowrap; box-shadow:0 2px 6px rgba(0,0,0,.35); display:none;
}}
#hover-tip .tip-val {{ font-size:14px; font-weight:700; }}
#hover-tip .tip-change {{ color:#7ecfff; }}
#hover-tip .tip-abs    {{ color:#ffd480; }}
.chart-panel {{
  position:absolute; top:10px; left:10px; z-index:998;
  width:360px; height:240px; background:rgba(255,255,255,0.96);
  border-radius:8px; box-shadow:0 3px 14px rgba(0,0,0,.25);
  padding:10px 12px 8px; cursor:grab; user-select:none;
  resize:both; overflow:hidden; min-width:220px; min-height:150px;
  display:none; flex-direction:column;
}}
.chart-panel:active {{ cursor:grabbing; }}
.chart-header {{ display:flex; justify-content:space-between; align-items:center; margin-bottom:4px; }}
.chart-title  {{ font-size:11px; font-weight:700; color:#333; }}
.chart-close  {{ background:none; border:none; cursor:pointer; font-size:14px;
                 color:#888; padding:0 2px; line-height:1; }}
.chart-close:hover {{ color:#333; }}
.chart-canvas-wrap {{ position:relative; flex:1; min-height:120px; }}
</style>
</head>
<body>
<div id="header">{header_html}</div>

<div id="maps-row">
  <div class="map-panel">
    <div class="panel-title abs-title" id="abs-panel-title">—</div>
    <div class="map-wrap" id="map-wrap-b">
      <div id="map-b" class="map-div"></div>
      <div class="colorbar-box"><img id="cbar-abs" src=""></div>
      <div class="chart-panel" id="chart-panel-abs">
        <div class="chart-header">
          <span class="chart-title" id="abs-chart-title"></span>
          <button class="chart-close" id="abs-chart-close">✕</button>
        </div>
        <div class="chart-canvas-wrap"><canvas id="abs-chart-canvas"></canvas></div>
      </div>
    </div>
  </div>
  <div class="map-panel">
    <div class="panel-title change-title">Δ Climate Change Signal</div>
    <div class="map-wrap" id="map-wrap-a">
      <div id="map-a" class="map-div"></div>
      <div class="colorbar-box"><img id="cbar-chg" src=""></div>
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
      <input type="range" id="timeline-slider" min="0" max="{n_frames-1}" value="0" step="1">
      <div id="tick-row"></div>
    </div>
  </div>
</div>

<script>
(function() {{

var SNAP_YEARS   = {snap_years_js};
var YEARS        = {frame_years_js};
var SNAP_IDX     = {snap_idx_js};
var SNAP_LABELS  = {snap_labels_js};
var N            = YEARS.length;
var MS           = {frame_ms};
var INIT_OPACITY = {dot_opacity:.2f};

var LOCKED_METHOD = {locked_method_js};

var BOUNDS = {{
  sd: {{ latMin:{lat_min:.4f}, latMax:{lat_max:.4f}, lonMin:{lon_min:.4f}, lonMax:{lon_max:.4f}, thresh:{mask_threshold_deg:.5f} }},
  dd: {{ latMin:{dd_lat_min:.4f}, latMax:{dd_lat_max:.4f}, lonMin:{dd_lon_min:.4f}, lonMax:{dd_lon_max:.4f}, thresh:{dd_mask_threshold_deg:.5f} }},
}};
var LAT_MIN = Math.min(BOUNDS.sd.latMin, BOUNDS.dd.latMin);
var LAT_MAX = Math.max(BOUNDS.sd.latMax, BOUNDS.dd.latMax);
var LON_MIN = Math.min(BOUNDS.sd.lonMin, BOUNDS.dd.lonMin);
var LON_MAX = Math.max(BOUNDS.sd.lonMax, BOUNDS.dd.lonMax);
var CANVAS_W = 1200, CANVAS_H = 1800;

var HOVER_LATS = {{ sd: {sd_hover_lats_js}, dd: {dd_hover_lats_js} }};
var HOVER_LONS = {{ sd: {sd_hover_lons_js}, dd: {dd_hover_lons_js} }};
var HOVER_UNITS  = {hover_units_js};
var ABS_UNITS    = {abs_units_js};

var FRAMES = {{
  sd: {{ chg: {sd_snap_b64_js}, abs: {sd_snap_abs_js} }},
  dd: {{ chg: {dd_snap_b64_js}, abs: {dd_snap_abs_js} }},
}};
var COLORBARS = {{
  sd: {{ chg: {sd_cb_chg_js}, abs: {sd_cb_abs_js} }},
  dd: {{ chg: {dd_cb_chg_js}, abs: {dd_cb_abs_js} }},
}};
var HOVER_VALS = {{
  sd: {{ chg: {sd_vals_js}, abs: {sd_abs_vals_js} }},
  dd: {{ chg: {dd_vals_js}, abs: {dd_abs_vals_js} }},
}};
var FP_RANGES = {{
  sd: {sd_fp_ranges_js},
  dd: {dd_fp_ranges_js},
}};
var CHART_DATA = {{
  sd: {{
    chg: {{ p5:{sd_p5_js}, p25:{sd_p25_js}, p75:{sd_p75_js}, p95:{sd_p95_js}, ens:{sd_ens_js},
            ymin:{sd_ymin_js}, ymax:{sd_ymax_js} }},
    abs: {{ p5:{sd_ap5_js}, p25:{sd_ap25_js}, p75:{sd_ap75_js}, p95:{sd_ap95_js}, ens:{sd_aens_js},
            ymin:{sd_aymin_js}, ymax:{sd_aymax_js} }},
  }},
  dd: {{
    chg: {{ p5:{dd_p5_js}, p25:{dd_p25_js}, p75:{dd_p75_js}, p95:{dd_p95_js}, ens:{dd_ens_js},
            ymin:{dd_ymin_js}, ymax:{dd_ymax_js} }},
    abs: {{ p5:{dd_ap5_js}, p25:{dd_ap25_js}, p75:{dd_ap75_js}, p95:{dd_ap95_js}, ens:{dd_aens_js},
            ymin:{dd_aymin_js}, ymax:{dd_aymax_js} }},
  }},
}};

var COUNTRY_GEOJSON = {country_geojson_js};
var REGIONS_GEOJSON = {regions_geojson_js};

// ── Active method: respect lock first, then initial_method (from sidebar toggle state)
var activeMethod = LOCKED_METHOD ? LOCKED_METHOD : {init_method_js};

var mapOpts = {{
  center: [{center_lat:.3f}, {center_lon:.3f}],
  zoom: 5, minZoom: 5, maxZoom: 9,
  maxBounds: [[LAT_MIN-4, LON_MIN-8], [LAT_MAX+4, LON_MAX+8]],
  zoomControl: false,
}};
var mapA = L.map('map-a', mapOpts);
var mapB = L.map('map-b', mapOpts);

function addBaseTiles(m) {{
  m.createPane('labelsPane');
  m.getPane('labelsPane').style.zIndex = 650;
  m.getPane('labelsPane').style.pointerEvents = 'none';
  L.tileLayer('https://server.arcgisonline.com/ArcGIS/rest/services/Elevation/World_Hillshade/MapServer/tile/{{z}}/{{y}}/{{x}}',
    {{ attribution:'ESRI Hillshade', opacity:0.45 }}).addTo(m);
  L.tileLayer('https://{{s}}.basemaps.cartocdn.com/rastertiles/voyager_only_labels/{{z}}/{{x}}/{{y}}{{r}}.png',
    {{ attribution:'© CartoDB', subdomains:'abcd', pane:'labelsPane', opacity:1.0 }}).addTo(m);
}}
addBaseTiles(mapA); addBaseTiles(mapB);
L.control.zoom({{ position:'bottomright' }}).addTo(mapA);
L.control.zoom({{ position:'bottomright' }}).addTo(mapB);

function addBorders(m) {{
  m.createPane('bordersPane');
  m.getPane('bordersPane').style.zIndex = 450;
  m.getPane('bordersPane').style.pointerEvents = 'none';
  var regionLayer = null, countryLayer = null;
  function weightsForZoom(z) {{ var t = Math.max(0,Math.min(1,(z-4)/6)); return {{region:0.1+t*1.4, country:0.3+t*2.8}}; }}
  if (REGIONS_GEOJSON) regionLayer = L.geoJSON(REGIONS_GEOJSON, {{ pane:'bordersPane', style:{{color:'#555',weight:0.6,opacity:0.55,fill:false}} }}).addTo(m);
  if (COUNTRY_GEOJSON) countryLayer = L.geoJSON(COUNTRY_GEOJSON, {{ pane:'bordersPane', style:{{color:'#333',weight:1.2,opacity:0.7,fill:false}} }}).addTo(m);
  m.on('zoomend', function() {{
    var w = weightsForZoom(m.getZoom());
    if (regionLayer) regionLayer.setStyle({{weight:w.region}});
    if (countryLayer) countryLayer.setStyle({{weight:w.country}});
  }});
}}
addBorders(mapA); addBorders(mapB);

function makeOverlay(m) {{
  m.createPane('dataOverlay');
  var pane = m.getPane('dataOverlay');
  pane.style.zIndex = 410; pane.style.pointerEvents = 'none';
  pane.classList.remove('leaflet-zoom-animated');
  var oc = document.createElement('canvas');
  oc.style.position = 'absolute'; oc.style.opacity = INIT_OPACITY;
  oc.style.imageRendering = 'auto'; oc.style.background = 'transparent';
  var octx = oc.getContext('2d', {{ alpha:true }});
  octx.imageSmoothingEnabled = true; octx.imageSmoothingQuality = 'high';
  pane.appendChild(oc); oc.width = CANVAS_W; oc.height = CANVAS_H;
  return {{ oc:oc, octx:octx }};
}}
var ovA = makeOverlay(mapA);
var ovB = makeOverlay(mapB);

function drawCanvas(ov, imgA, imgB, t) {{
  if (!imgA) return;
  ov.octx.clearRect(0, 0, CANVAS_W, CANVAS_H);
  ov.octx.globalAlpha = 1.0; ov.octx.drawImage(imgA, 0, 0, CANVAS_W, CANVAS_H);
  if (imgB && t > 0.001) {{ ov.octx.globalAlpha = t; ov.octx.drawImage(imgB, 0, 0, CANVAS_W, CANVAS_H); }}
  ov.octx.globalAlpha = 1.0;
}}

function repositionCanvas(m, ov) {{
  var b = BOUNDS[activeMethod];
  var nw = m.latLngToLayerPoint(L.latLng(b.latMax, b.lonMin));
  var se = m.latLngToLayerPoint(L.latLng(b.latMin, b.lonMax));
  ov.oc.style.left   = nw.x+'px'; ov.oc.style.top    = nw.y+'px';
  ov.oc.style.width  = Math.max(1, Math.round(se.x-nw.x))+'px';
  ov.oc.style.height = Math.max(1, Math.round(se.y-nw.y))+'px';
  ov.oc.style.transform = '';
}}

var _rafA = false, _rafB = false;
function scheduleReposA() {{ if (_rafA) return; _rafA=true; requestAnimationFrame(function(){{ _rafA=false; repositionCanvas(mapA,ovA); _redrawBoth(); }}); }}
function scheduleReposB() {{ if (_rafB) return; _rafB=true; requestAnimationFrame(function(){{ _rafB=false; repositionCanvas(mapB,ovB); _redrawBoth(); }}); }}

var _lastBlend = {{a:0, b:0, t:0}};
var snapImgsChg = {{}}, snapImgsAbs = {{}};
var _loadCounts = {{sd:{{chg:0,abs:0}}, dd:{{chg:0,abs:0}}}};
var _loadTotals = {{
  sd:{{chg:FRAMES.sd.chg.length, abs:FRAMES.sd.abs.length}},
  dd:{{chg:FRAMES.dd.chg.length, abs:FRAMES.dd.abs.length}},
}};
var _initDone = false;

function _checkReady() {{
  if (_initDone) return;
  var m = activeMethod;
  if (_loadCounts[m].chg === _loadTotals[m].chg &&
      _loadCounts[m].abs === _loadTotals[m].abs) {{
    _initDone = true; updateColorbars(); showFrame(0);
  }}
}}

// ── Lazy-load: active method eagerly, other method deferred ──────────────────
function loadMethod(mth) {{
  FRAMES[mth].chg.forEach(function(b64, i) {{
    var img = new Image();
    img.onload = function() {{ snapImgsChg[mth][i] = img; _loadCounts[mth].chg++; _checkReady(); }};
    img.src = 'data:image/png;base64,' + b64;
  }});
  FRAMES[mth].abs.forEach(function(b64, i) {{
    var img = new Image();
    img.onload = function() {{ snapImgsAbs[mth][i] = img; _loadCounts[mth].abs++; _checkReady(); }};
    img.src = 'data:image/png;base64,' + b64;
  }});
}}

['sd','dd'].forEach(function(mth) {{
  snapImgsChg[mth] = new Array(FRAMES[mth].chg.length);
  snapImgsAbs[mth] = new Array(FRAMES[mth].abs.length);
  if (mth !== activeMethod) {{
    setTimeout(function() {{ loadMethod(mth); }}, 2000);
  }} else {{
    loadMethod(mth);
  }}
}});

function _redrawBoth() {{
  var b = _lastBlend, m = activeMethod;
  drawCanvas(ovA, snapImgsChg[m][b.a], snapImgsChg[m][b.b], b.t);
  drawCanvas(ovB, snapImgsAbs[m][b.a], snapImgsAbs[m][b.b], b.t);
}}

function updateColorbars() {{
  document.getElementById('cbar-chg').src = COLORBARS[activeMethod].chg;
  document.getElementById('cbar-abs').src = COLORBARS[activeMethod].abs;
}}

var _syncing = false;
mapA.on('move', scheduleReposA); mapB.on('move', scheduleReposB);
mapA.on('moveend', function() {{ if (_syncing) return; _syncing=true; mapB.setView(mapA.getCenter(),mapA.getZoom(),{{animate:false}}); repositionCanvas(mapB,ovB); _redrawBoth(); _syncing=false; }});
mapB.on('moveend', function() {{ if (_syncing) return; _syncing=true; mapA.setView(mapB.getCenter(),mapB.getZoom(),{{animate:false}}); repositionCanvas(mapA,ovA); _redrawBoth(); _syncing=false; }});

var ZOOM_DUR = 250, ZOOM_EASE = 'cubic-bezier(0,0,0.25,1)';
function animZoom(m, ov, e) {{
  var b = BOUNDS[activeMethod];
  var curW = parseFloat(ov.oc.style.width)||CANVAS_W, curL = parseFloat(ov.oc.style.left)||0, curT = parseFloat(ov.oc.style.top)||0;
  if (curW < 1) return;
  var nwF = m._latLngToNewLayerPoint(L.latLng(b.latMax, b.lonMin), e.zoom, e.center);
  var seF = m._latLngToNewLayerPoint(L.latLng(b.latMin, b.lonMax), e.zoom, e.center);
  var sc = Math.max(1,seF.x-nwF.x)/curW;
  ov.oc.style.transformOrigin='0 0'; ov.oc.style.transition='transform '+ZOOM_DUR+'ms '+ZOOM_EASE;
  ov.oc.style.transform='translate('+(nwF.x-curL)+'px,'+(nwF.y-curT)+'px) scale('+sc+')';
}}
mapA.on('zoomanim', function(e){{ animZoom(mapA,ovA,e); }});
mapB.on('zoomanim', function(e){{ animZoom(mapB,ovB,e); }});
mapA.on('zoomend', function(){{ ovA.oc.style.transition=''; ovA.oc.style.transform=''; scheduleReposA(); scheduleReposB(); }});
mapB.on('zoomend', function(){{ ovB.oc.style.transition=''; ovB.oc.style.transform=''; scheduleReposA(); scheduleReposB(); }});
repositionCanvas(mapA,ovA); repositionCanvas(mapB,ovB);

var slider   = document.getElementById('timeline-slider');
var periodEl = document.getElementById('abs-panel-title');
var playBtn  = document.getElementById('play-btn');
var tickRow  = document.getElementById('tick-row');

function getLabel(fi) {{
  var si = SNAP_IDX.indexOf(fi);
  return si >= 0 ? SNAP_LABELS[si] : String(Math.round(YEARS[fi]));
}}

function frameToBlend(fi) {{
  for (var s = 0; s < SNAP_IDX.length-1; s++) {{
    if (fi >= SNAP_IDX[s] && fi <= SNAP_IDX[s+1]) {{
      var span = SNAP_IDX[s+1]-SNAP_IDX[s];
      return {{a:s, b:Math.min(s+1,snapImgsChg[activeMethod].length-1), t:span>0?(fi-SNAP_IDX[s])/span:0}};
    }}
  }}
  return {{a:snapImgsChg[activeMethod].length-1, b:snapImgsChg[activeMethod].length-1, t:0}};
}}

var current = 0;
function showFrame(fi) {{
  fi = Math.max(0, Math.min(N-1, fi)); current = fi;
  var b = frameToBlend(fi); _lastBlend = b;
  var m = activeMethod;
  drawCanvas(ovA, snapImgsChg[m][b.a]||null, snapImgsChg[m][b.b]||null, b.t);
  drawCanvas(ovB, snapImgsAbs[m][b.a]||null, snapImgsAbs[m][b.b]||null, b.t);
  slider.value = fi; periodEl.textContent = getLabel(fi);
  if (myChartChange) myChartChange.update('none');
  if (myChartAbs)    myChartAbs.update('none');
}}

window.switchMethod = function(mth) {{
  if (LOCKED_METHOD && mth !== LOCKED_METHOD) return;
  if (mth === activeMethod) return;
  if (_loadCounts[mth].chg < _loadTotals[mth].chg ||
      _loadCounts[mth].abs < _loadTotals[mth].abs) {{
    setTimeout(function() {{ window.switchMethod(mth); }}, 500);
    return;
  }}
  activeMethod = mth;
  updateColorbars();
  repositionCanvas(mapA, ovA);
  repositionCanvas(mapB, ovB);
  showFrame(current);
  if (_pinLat !== null && _pinLon !== null) {{
    var nn = nearestPoint(_pinLat, _pinLon);
    _pinIdx = nn.idx;
    if (myChartChange && _pinIdx >= 0) {{
      showChangeChartPanel(_pinIdx, HOVER_LATS[mth][_pinIdx], HOVER_LONS[mth][_pinIdx]);
    }}
    if (myChartAbs && _pinIdx >= 0) {{
      showAbsChartPanel(_pinIdx, HOVER_LATS[mth][_pinIdx], HOVER_LONS[mth][_pinIdx]);
    }}
  }}
  try {{ window.parent.postMessage({{type:'nzmap_method', method:mth}}, '*'); }} catch(e) {{}}
}};

var playing = false, timer = null;
function tick() {{ if (!playing) return; if (current>=N-1) {{ pause(); return; }} showFrame(current+1); timer=setTimeout(tick,MS); }}
function play()  {{ if (current>=N-1) showFrame(0); playing=true; playBtn.textContent='⏸ Pause'; timer=setTimeout(tick,MS); }}
function pause() {{ playing=false; playBtn.textContent='▶ Play'; clearTimeout(timer); }}
playBtn.addEventListener('click', function(){{ if (playing) pause(); else play(); }});
document.getElementById('reset-btn').addEventListener('click', function(){{ pause(); showFrame(0); }});
slider.addEventListener('input', function(){{ pause(); showFrame(parseInt(this.value)); }});

window.addEventListener('message', function(e) {{
  if (!e.data) return;
  if (e.data.type === 'nzmap') {{
    if (typeof e.data.opacity === 'number') {{ ovA.oc.style.opacity=e.data.opacity; ovB.oc.style.opacity=e.data.opacity; }}
    if (typeof e.data.frameMs === 'number') MS = e.data.frameMs;
  }}
  if (e.data.type === 'nzmap_method' && e.data.method) {{ window.switchMethod(e.data.method); }}
}});

function buildTicks() {{
  tickRow.innerHTML = '';
  var thumbR = 8, sliderRect = slider.getBoundingClientRect(), rowRect = tickRow.getBoundingClientRect();
  var leftInset = (sliderRect.left-rowRect.left)+thumbR, rightInset = (rowRect.right-sliderRect.right)+thumbR;
  var trackW = rowRect.width-leftInset-rightInset;
  for (var i = 0; i < N; i++) {{
    var si = SNAP_IDX.indexOf(i), isSnap = si >= 0;
    var px = leftInset + (i/(N-1))*trackW;
    if (isSnap) {{
      var div = document.createElement('div'); div.className='tick snap-tick'; div.style.left=px+'px';
      var lbl = document.createElement('div'); lbl.className='tick-text-snap'; lbl.textContent=SNAP_LABELS[si];
      var line = document.createElement('div'); line.className='snap-line';
      div.appendChild(lbl); div.appendChild(line); tickRow.appendChild(div);
    }} else {{
      if (i%2!==0) continue;
      var div = document.createElement('div'); div.className='tick year-tick'; div.style.left=px+'px';
      var line = document.createElement('div'); line.className='tick-line';
      var txt = document.createElement('div'); txt.className='tick-text-year'; txt.textContent=String(Math.round(YEARS[i]));
      div.appendChild(line); div.appendChild(txt); tickRow.appendChild(div);
    }}
  }}
}}
setTimeout(buildTicks, 200);
window.addEventListener('resize', buildTicks);
window.addEventListener('resize', updateChartPanelLimits);

var tip = document.getElementById('hover-tip');
var tipCoord = document.getElementById('tip-coord');
var tipChange = document.getElementById('tip-change');
var tipAbs    = document.getElementById('tip-abs');
var tipPeriod = document.getElementById('tip-period');

function interpVal(vals, ptIdx, fi) {{
  for (var s=0; s<SNAP_IDX.length-1; s++) {{
    if (fi>=SNAP_IDX[s] && fi<SNAP_IDX[s+1]) {{
      var t=(fi-SNAP_IDX[s])/(SNAP_IDX[s+1]-SNAP_IDX[s]);
      return (1-t)*vals[s][ptIdx]+t*vals[s+1][ptIdx];
    }}
  }}
  return vals[vals.length-1][ptIdx];
}}

function nearestPoint(lat, lon) {{
  var lats = HOVER_LATS[activeMethod];
  var lons = HOVER_LONS[activeMethod];
  var best=-1, bestDist=Infinity;
  for (var i=0; i<lats.length; i++) {{
    var d=(lats[i]-lat)**2+(lons[i]-lon)**2;
    if (d<bestDist){{bestDist=d; best=i;}}
  }}
  return {{idx:best, dist:Math.sqrt(bestDist)}};
}}

function getHoverThresh() {{
  return BOUNDS[activeMethod].thresh;
}}

function showTip(e, nn) {{
  var m = activeMethod;
  var lats = HOVER_LATS[m], lons = HOVER_LONS[m];
  var chgVals = HOVER_VALS[m].chg, absVals = HOVER_VALS[m].abs;
  if (!chgVals) return;
  var changeVal = interpVal(chgVals, nn.idx, current);
  tipCoord.textContent  = 'Lat '+lats[nn.idx].toFixed(2)+'°  Lon '+lons[nn.idx].toFixed(2)+'°';
  tipChange.textContent = 'Δ '+(changeVal>=0?'+':'')+changeVal.toFixed(2)+' '+HOVER_UNITS;
  if (absVals) {{
    tipAbs.textContent = '◆ '+interpVal(absVals,nn.idx,current).toFixed(2)+' '+ABS_UNITS;
  }} else {{ tipAbs.textContent=''; }}
  tipPeriod.textContent = 'Period: '+getLabel(current);
  tip.style.display='block';
  tip.style.left=(e.originalEvent.clientX+14)+'px';
  tip.style.top=(e.originalEvent.clientY-10)+'px';
}}
var _hThrottle = null;
function onMapMousemove(e) {{
  if (_hThrottle) return; _hThrottle=setTimeout(function(){{_hThrottle=null;}},30);
  var nn = nearestPoint(e.latlng.lat, e.latlng.lng);
  if (nn.dist > getHoverThresh()) {{ tip.style.display='none'; return; }}
  var absVals = HOVER_VALS[activeMethod].abs;
  if (absVals) {{
    var absVal = interpVal(absVals, nn.idx, current);
    if (!isFinite(absVal)) {{ tip.style.display='none'; return; }}
  }} else {{
    var chgVals = HOVER_VALS[activeMethod].chg;
    if (chgVals) {{
      var chgVal = interpVal(chgVals, nn.idx, current);
      if (!isFinite(chgVal)) {{ tip.style.display='none'; return; }}
    }}
  }}
  showTip(e, nn);
}}
mapA.on('mousemove', onMapMousemove); mapB.on('mousemove', onMapMousemove);
mapA.on('mouseout', function(){{tip.style.display='none';}});
mapB.on('mouseout', function(){{tip.style.display='none';}});

var vertLinePlugin = {{
  id:'vertLine',
  afterDraw: function(chart) {{
    var ctx=chart.ctx, xs=chart.scales.x, ys=chart.scales.y;
    var x=xs.getPixelForValue(YEARS[current]);
    if (x<xs.left||x>xs.right) return;
    ctx.save(); ctx.beginPath();
    ctx.strokeStyle='rgba(50,50,50,0.6)'; ctx.lineWidth=1.5; ctx.setLineDash([4,3]);
    ctx.moveTo(x,ys.top); ctx.lineTo(x,ys.bottom); ctx.stroke(); ctx.setLineDash([]);
    ctx.fillStyle='#333'; ctx.font='bold 9px Arial'; ctx.textAlign='center';
    var _si=-1, _sd=Infinity;
    var fpRanges = FP_RANGES[activeMethod];
    for (var i=0; i<SNAP_YEARS.length; i++) {{
      var d=Math.abs(YEARS[current]-SNAP_YEARS[i]);
      if (d<_sd){{_sd=d;_si=i;}}
    }}
    ctx.fillText((_si>=0&&_sd<3)?fpRanges[_si]:String(Math.round(YEARS[current])), x, ys.top-3);
    ctx.restore();
  }}
}};

var bandPlugin = {{
  id:'bandPlugin',
  afterLayout: function(chart) {{
    chart._bandPixels = {{}};
    var xAx=chart.scales.x, yAx=chart.scales.y;
    chart.data.datasets.forEach(function(ds) {{
      if (!ds._bandRole) return;
      chart._bandPixels[ds._bandRole] = ds.data.map(function(pt) {{
        return {{x:xAx.getPixelForValue(pt.x), y:yAx.getPixelForValue(pt.y)}};
      }});
    }});
  }},
  beforeDatasetsDraw: function(chart) {{
    var ctx=chart.ctx, pixels=chart._bandPixels;
    if (!pixels) return;
    var xAx=chart.scales.x, yAx=chart.scales.y;
    var sdC90='rgba(74,144,217,0.13)', sdC50='rgba(74,144,217,0.28)';
    var ddC90='rgba(34,168,34,0.13)',  ddC50='rgba(34,168,34,0.28)';
    var sdS90='rgba(74,144,217,0.55)', sdS50='rgba(74,144,217,0.85)';
    var ddS90='rgba(34,168,34,0.55)',  ddS50='rgba(34,168,34,0.85)';
    var primFill90 = activeMethod==='sd' ? sdC90 : ddC90;
    var primFill50 = activeMethod==='sd' ? sdC50 : ddC50;
    var otherStroke90 = activeMethod==='sd' ? ddS90 : sdS90;
    var otherStroke50 = activeMethod==='sd' ? ddS50 : sdS50;
    function clip() {{
      ctx.beginPath();
      ctx.rect(xAx.left, yAx.top, xAx.right-xAx.left, yAx.bottom-yAx.top);
      ctx.clip();
    }}
    function curvePath(pts) {{
      ctx.moveTo(pts[0].x, pts[0].y);
      for (var i=1; i<pts.length; i++) {{
        var cpx=(pts[i-1].x+pts[i].x)/2;
        ctx.bezierCurveTo(cpx,pts[i-1].y, cpx,pts[i].y, pts[i].x,pts[i].y);
      }}
    }}
    function drawFilled(topRole, botRole, color) {{
      var tp=pixels[topRole], bp=pixels[botRole];
      if (!tp||!bp||!tp.length||!bp.length) return;
      ctx.save(); clip();
      ctx.beginPath(); curvePath(tp);
      ctx.lineTo(bp[bp.length-1].x, bp[bp.length-1].y);
      for (var j=bp.length-2;j>=0;j--) {{
        var cpx=(bp[j+1].x+bp[j].x)/2;
        ctx.bezierCurveTo(cpx,bp[j+1].y, cpx,bp[j].y, bp[j].x,bp[j].y);
      }}
      ctx.closePath(); ctx.fillStyle=color; ctx.fill(); ctx.restore();
    }}
    function drawDashedEdge(role, strokeColor) {{
      var pts=pixels[role];
      if (!pts||!pts.length) return;
      ctx.save(); clip();
      ctx.setLineDash([5,4]); ctx.strokeStyle=strokeColor; ctx.lineWidth=1.2;
      ctx.beginPath(); curvePath(pts); ctx.stroke();
      ctx.setLineDash([]); ctx.restore();
    }}
    drawFilled('primary90top','primary90bot', primFill90);
    drawFilled('primary50top','primary50bot', primFill50);
    drawDashedEdge('other90top', otherStroke90);
    drawDashedEdge('other90bot', otherStroke90);
    drawDashedEdge('other50top', otherStroke50);
    drawDashedEdge('other50bot', otherStroke50);
  }}
}};

function buildDatasets(panelType, pinIdx) {{
  var m   = activeMethod;
  var oth = m==='sd' ? 'dd' : 'sd';
  var primCD  = CHART_DATA[m][panelType];
  var otherCD = CHART_DATA[oth][panelType];
  var snapYrs = SNAP_IDX.map(function(i){{ return YEARS[i]; }});

  var otherPinIdx = -1;
  if (otherCD) {{
    var _pLat  = HOVER_LATS[m][pinIdx];
    var _pLon  = HOVER_LONS[m][pinIdx];
    var _oLats = HOVER_LATS[oth];
    var _oLons = HOVER_LONS[oth];
    var _bd = Infinity;
    for (var _oi = 0; _oi < _oLats.length; _oi++) {{
      var _od = (_oLats[_oi]-_pLat)*(_oLats[_oi]-_pLat) +
                (_oLons[_oi]-_pLon)*(_oLons[_oi]-_pLon);
      if (_od < _bd) {{ _bd = _od; otherPinIdx = _oi; }}
    }}
  }}

  var primColor  = m==='sd' ? '#4a90d9' : '#22a822';
  var otherColor = m==='sd' ? '#22a822' : '#4a90d9';
  var units      = panelType==='chg' ? HOVER_UNITS : ABS_UNITS;
  var hoverVals  = HOVER_VALS[m][panelType];
  var otherHoverVals = HOVER_VALS[oth][panelType];

  var ds = [];
  function pushBandRole(role, label, data) {{
    ds.push({{label:label, _bandRole:role, data:data,
             fill:false, borderColor:'transparent', borderWidth:0, pointRadius:0, tension:0.35}});
  }}

  if (primCD.p95) {{
    pushBandRole('primary90top','', snapYrs.map(function(yr,i){{ return {{x:yr,y:primCD.p95[i][pinIdx]}}; }}));
    pushBandRole('primary90bot','90% interval', snapYrs.map(function(yr,i){{ return {{x:yr,y:primCD.p5[i][pinIdx]}}; }}));
  }}
  if (primCD.p75) {{
    pushBandRole('primary50top','', snapYrs.map(function(yr,i){{ return {{x:yr,y:primCD.p75[i][pinIdx]}}; }}));
    pushBandRole('primary50bot','50% interval', snapYrs.map(function(yr,i){{ return {{x:yr,y:primCD.p25[i][pinIdx]}}; }}));
  }}
  if (otherCD && otherCD.p95 && otherPinIdx >= 0) {{
    pushBandRole('other90top','', snapYrs.map(function(yr,i){{ return {{x:yr,y:otherCD.p95[i][otherPinIdx]}}; }}));
    pushBandRole('other90bot',(oth==='sd'?'SD':'DD')+' 90%', snapYrs.map(function(yr,i){{ return {{x:yr,y:otherCD.p5[i][otherPinIdx]}}; }}));
  }}
  if (otherCD && otherCD.p75 && otherPinIdx >= 0) {{
    pushBandRole('other50top','', snapYrs.map(function(yr,i){{ return {{x:yr,y:otherCD.p75[i][otherPinIdx]}}; }}));
    pushBandRole('other50bot',(oth==='sd'?'SD':'DD')+' 50%', snapYrs.map(function(yr,i){{ return {{x:yr,y:otherCD.p25[i][otherPinIdx]}}; }}));
  }}

  var ensData = primCD.ens || hoverVals;
  if (ensData) {{
    ds.push({{label:(m==='sd'?'SD':'DD')+' ensemble mean',
      data:snapYrs.map(function(yr,i){{ return {{x:yr,y:ensData[i][pinIdx]}}; }}),
      borderColor:primColor, backgroundColor:primColor,
      borderWidth:2.5, pointRadius:4, fill:false, tension:0.35}});
  }}
  if (hoverVals && primCD.ens) {{
    ds.push({{label:'Selected model',
      data:snapYrs.map(function(yr,i){{ return {{x:yr,y:hoverVals[i][pinIdx]}}; }}),
      borderColor:m==='sd'?'#1a4a9a':'#145214', backgroundColor:m==='sd'?'#1a4a9a':'#145214',
      borderWidth:2, pointRadius:4, fill:false, tension:0.35}});
  }}
  var otherEnsData = (otherCD && otherCD.ens) || (otherCD && otherHoverVals);
  if (otherEnsData && otherPinIdx >= 0) {{
    ds.push({{label:(oth==='sd'?'SD':'DD')+' ensemble mean',
      data:snapYrs.map(function(yr,i){{ return {{x:yr,y:otherEnsData[i][otherPinIdx]}}; }}),
      borderColor:otherColor, backgroundColor:otherColor,
      borderWidth:2, borderDash:[6,3], pointRadius:3, fill:false, tension:0.35}});
  }}
  return ds;
}}

function makeChartOptions(units, ymin, ymax) {{
  var fpRanges = FP_RANGES[activeMethod];
  return {{
    animation:false, responsive:true, maintainAspectRatio:false,
    interaction:{{mode:'nearest',intersect:false,axis:'x'}},
    plugins:{{
      legend:{{display:true, position:'top',
        labels:{{font:{{size:9}}, boxWidth:12, padding:5,
          filter:function(item){{
            return item.text!=='' &&
                   item.text.indexOf('top')===-1 &&
                   item.text.indexOf('bot')===-1;
          }}}}}},
      tooltip:{{callbacks:{{
        title:function(items){{
          if (!items.length) return '';
          var xVal=items[0].parsed.x, best=-1, bestDist=Infinity;
          for (var i=0;i<SNAP_YEARS.length;i++){{var d=Math.abs(SNAP_YEARS[i]-xVal);if(d<bestDist){{bestDist=d;best=i;}}}}
          return fpRanges[best];
        }},
        label:function(ctx){{
          if (!ctx.dataset.label||ctx.dataset._bandRole) return null;
          if (ctx.dataset.label.indexOf('top')!==-1||ctx.dataset.label.indexOf('bot')!==-1) return null;
          var v=ctx.parsed.y; if(v==null) return null;
          return ctx.dataset.label+': '+(v>=0?'+':'')+v.toFixed(2)+' '+units;
        }},
      }}}},
    }},
    scales:{{
      x:{{type:'linear', min:SNAP_YEARS[0], max:SNAP_YEARS[SNAP_YEARS.length-1],
          ticks:{{font:{{size:9}}, maxTicksLimit:12,
            callback:function(v){{
              var best=-1,bestDist=Infinity;
              for(var i=0;i<SNAP_YEARS.length;i++){{var d=Math.abs(SNAP_YEARS[i]-v);if(d<bestDist){{bestDist=d;best=i;}}}}
              if(best>=0&&bestDist<1.0) return fpRanges[best];
              return '';
            }}}}}},
      y:{{title:{{display:true,text:units,font:{{size:9}}}},ticks:{{font:{{size:9}}}},
          min:(ymin!==null)?ymin:undefined, max:(ymax!==null)?ymax:undefined}},
    }},
  }};
}}

function updateChartPanelLimits() {{
  var wA=document.getElementById('map-wrap-a').getBoundingClientRect();
  var wB=document.getElementById('map-wrap-b').getBoundingClientRect();
  var pC=document.getElementById('chart-panel'), pA=document.getElementById('chart-panel-abs');
  pC.style.maxWidth=Math.floor(wA.width*0.92)+'px'; pC.style.maxHeight=Math.floor(wA.height*0.92)+'px';
  pA.style.maxWidth=Math.floor(wB.width*0.92)+'px'; pA.style.maxHeight=Math.floor(wB.height*0.92)+'px';
}}

if (typeof ResizeObserver !== 'undefined') {{
  new ResizeObserver(function(){{if(myChartChange)myChartChange.resize();}}).observe(document.getElementById('chart-panel'));
  new ResizeObserver(function(){{if(myChartAbs)myChartAbs.resize();}}).observe(document.getElementById('chart-panel-abs'));
}}

var chartPanelChange = document.getElementById('chart-panel');
var chartPanelAbs    = document.getElementById('chart-panel-abs');
var myChartChange = null, myChartAbs = null, _pinIdx = -1;

function combinedYRange(panelType) {{
  var sd = CHART_DATA.sd[panelType];
  var dd = CHART_DATA.dd[panelType];
  var ymin = null, ymax = null;
  if (sd.ymin !== null && sd.ymin !== undefined) ymin = sd.ymin;
  if (dd.ymin !== null && dd.ymin !== undefined) ymin = (ymin === null) ? dd.ymin : Math.min(ymin, dd.ymin);
  if (sd.ymax !== null && sd.ymax !== undefined) ymax = sd.ymax;
  if (dd.ymax !== null && dd.ymax !== undefined) ymax = (ymax === null) ? dd.ymax : Math.max(ymax, dd.ymax);
  return {{ymin: ymin, ymax: ymax}};
}}

var _pinLat = null, _pinLon = null;

function showChangeChartPanel(ptIdx, lat, lon) {{
  _pinIdx = ptIdx; _pinLat = lat; _pinLon = lon;
  if (chartPanelChange.style.display==='none'||!chartPanelChange.style.display) {{
    chartPanelChange.style.width='360px'; chartPanelChange.style.height='240px';
  }}
  updateChartPanelLimits();
  document.getElementById('chart-title').textContent =
    HOVER_LATS[activeMethod][ptIdx].toFixed(2)+'° N  '+HOVER_LONS[activeMethod][ptIdx].toFixed(2)+'° E';
  if (myChartChange) {{ myChartChange.destroy(); myChartChange=null; }}
  var yr = combinedYRange('chg');
  myChartChange = new Chart(document.getElementById('chart-canvas'), {{
    type:'line', plugins:[vertLinePlugin, bandPlugin],
    data:{{datasets:buildDatasets('chg', ptIdx)}},
    options:makeChartOptions(HOVER_UNITS, yr.ymin, yr.ymax),
  }});
  chartPanelChange.style.display='flex';
}}

function showAbsChartPanel(ptIdx, lat, lon) {{
  _pinIdx = ptIdx; _pinLat = lat; _pinLon = lon;
  if (chartPanelAbs.style.display==='none'||!chartPanelAbs.style.display) {{
    chartPanelAbs.style.width='360px'; chartPanelAbs.style.height='240px';
  }}
  updateChartPanelLimits();
  document.getElementById('abs-chart-title').textContent =
    HOVER_LATS[activeMethod][ptIdx].toFixed(2)+'° N  '+HOVER_LONS[activeMethod][ptIdx].toFixed(2)+'° E';
  if (myChartAbs) {{ myChartAbs.destroy(); myChartAbs=null; }}
  var yr = combinedYRange('abs');
  myChartAbs = new Chart(document.getElementById('abs-chart-canvas'), {{
    type:'line', plugins:[vertLinePlugin, bandPlugin],
    data:{{datasets:buildDatasets('abs', ptIdx)}},
    options:makeChartOptions(ABS_UNITS, yr.ymin, yr.ymax),
  }});
  chartPanelAbs.style.display='flex';
}}

mapA.on('click', function(e) {{
  var nn = nearestPoint(e.latlng.lat, e.latlng.lng);
  if (nn.dist > getHoverThresh()) return;
  var chgVals = HOVER_VALS[activeMethod].chg;
  var absVals = HOVER_VALS[activeMethod].abs;
  if (!chgVals) return;
  if (absVals) {{
    var absVal = interpVal(absVals, nn.idx, current);
    if (!isFinite(absVal)) return;
  }} else {{
    var val = interpVal(chgVals, nn.idx, current);
    if (!isFinite(val)) return;
  }}
  showChangeChartPanel(nn.idx, HOVER_LATS[activeMethod][nn.idx], HOVER_LONS[activeMethod][nn.idx]);
}});
mapB.on('click', function(e) {{
  var nn = nearestPoint(e.latlng.lat, e.latlng.lng);
  if (nn.dist > getHoverThresh()) return;
  var absVals = HOVER_VALS[activeMethod].abs;
  if (!absVals) return;
  var val = interpVal(absVals, nn.idx, current);
  if (!isFinite(val)) return;
  showAbsChartPanel(nn.idx, HOVER_LATS[activeMethod][nn.idx], HOVER_LONS[activeMethod][nn.idx]);
}});

document.getElementById('chart-close').addEventListener('click', function() {{
  chartPanelChange.style.display='none'; if(myChartChange){{myChartChange.destroy();myChartChange=null;}}
}});
document.getElementById('abs-chart-close').addEventListener('click', function() {{
  chartPanelAbs.style.display='none'; if(myChartAbs){{myChartAbs.destroy();myChartAbs=null;}}
}});

function makeDraggable(panel, wrapId) {{
  var dragX=0, dragY=0, startL=0, startT=0, dragging=false;
  panel.addEventListener('mousedown', function(e) {{
    var rect=panel.getBoundingClientRect();
    if (e.clientX>rect.right-16&&e.clientY>rect.bottom-16) return;
    if (e.target.classList.contains('chart-close')||e.target.closest('.chart-canvas-wrap')) return;
    dragging=true; dragX=e.clientX; dragY=e.clientY;
    startL=parseInt(panel.style.left)||0; startT=parseInt(panel.style.top)||0;
    panel.style.cursor='grabbing'; e.preventDefault();
  }});
  document.addEventListener('mousemove', function(e) {{
    if (!dragging) return;
    var wrap=document.getElementById(wrapId).getBoundingClientRect();
    panel.style.left=Math.max(0,Math.min(startL+e.clientX-dragX, wrap.width-panel.offsetWidth))+'px';
    panel.style.top=Math.max(0,Math.min(startT+e.clientY-dragY, wrap.height-panel.offsetHeight))+'px';
  }});
  document.addEventListener('mouseup', function(){{ if(dragging){{dragging=false; panel.style.cursor='grab';}} }});
}}
makeDraggable(document.getElementById('chart-panel'),     'map-wrap-a');
makeDraggable(document.getElementById('chart-panel-abs'), 'map-wrap-b');

}})();
</script>
</body>
</html>"""


# ── Loading screen ────────────────────────────────────────────────────────────
def build_loading_screen_html(svg_data, height=630):
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
    <path d="{svg_data['d']}" style="fill:#d8d8d8;stroke:#999;stroke-width:0.015;vector-effect:non-scaling-stroke"/>
    <path d="{svg_data['d']}" class="nz-fill" style="fill:#222;stroke:#000;stroke-width:0.015;vector-effect:non-scaling-stroke"/>
  </svg>
</div>"""
    return f"""<!DOCTYPE html><html><head><style>
html,body{{margin:0;padding:0;height:{height}px;width:100%;position:relative;background:#fafafa;overflow:hidden;}}
.nz-wrap{{position:absolute;top:50%;left:50%;transform:translate(-50%,-50%);width:280px;height:372px;clip-path:inset(0 0 10% 0);-webkit-clip-path:inset(0 0 10% 0);}}
.nz-fill{{clip-path:inset(100% 0 0 0);-webkit-clip-path:inset(100% 0 0 0);animation:rise 2.4s cubic-bezier(0.45,0,0.55,1) infinite;}}
@keyframes rise{{0%{{clip-path:inset(100% 0 0 0);-webkit-clip-path:inset(100% 0 0 0);}}50%{{clip-path:inset(0 0 0 0);-webkit-clip-path:inset(0 0 0 0);}}100%{{clip-path:inset(0 0 100% 0);-webkit-clip-path:inset(0 0 100% 0);}}}}
</style></head><body>{body}</body></html>"""


# ── Dependency guards ─────────────────────────────────────────────────────────
if not HAS_XARRAY:   st.error("**`xarray` is not installed.**"); st.stop()
if not HAS_PLOTLY:   st.error("**`plotly` is not installed.**"); st.stop()
if not _ON_CLOUD and not DATA_ROOT.exists():
    st.error(f"Data directory **`{DATA_ROOT}`** not found.\n\nRun: `python test/generate_demo_data.py`"); st.stop()


# ── Module import guard ───────────────────────────────────────────────────────
_STREAMLIT_ACTIVE = False
try:
    from streamlit.runtime.scriptrunner import get_script_run_ctx as _get_ctx
    _STREAMLIT_ACTIVE = _get_ctx() is not None
except Exception:
    pass

if _STREAMLIT_ACTIVE:

    import streamlit.components.v1 as _sc

    # ── Session state initialisation ──────────────────────────────────────────
    if "applied" not in st.session_state:
        if _ON_CLOUD:
            _seed_ind = "TX"
        else:
            _seed_inds = list_indicators("historical", method="sd")
            _seed_ind  = _seed_inds[0] if _seed_inds else "TX"
        st.session_state["applied"] = dict(
            ssp="ssp370", bp_tag="bp1995-2014",
            indicator=_seed_ind,
            season=SEASON_ANN, model_choice=_MODEL_ENSEMBLE_MEAN,
        )
    if "method" not in st.session_state:
        st.session_state["method"] = "sd"
    # Flag set when the indicator changes and we need to check model validity
    if "_pending_indicator_check" not in st.session_state:
        st.session_state["_pending_indicator_check"] = False

    _current_method = st.session_state["method"]

    APPROVED_INDICATORS = [
        "DD1mm", "FD", "PR", "REC_Rx1day", "REC_TXx",
        "RR25mm", "Rx1day", "TN", "TNn", "TX", "TX25", "TX30", "TXx",
    ]
    indicators_avail = APPROVED_INDICATORS

    if not indicators_avail:
        st.sidebar.warning("No data found for either downscaling method."); st.stop()

    # ── Cache-based model availability for current applied config ─────────────
    _cfg_applied = st.session_state["applied"]
    _models_sd_applied, _models_dd_applied = get_models_from_cache(
        _cfg_applied["indicator"],
        _cfg_applied["ssp"],
        _cfg_applied["bp_tag"],
        _cfg_applied["season"],
    )
    _all_models_applied   = sorted(_models_sd_applied | _models_dd_applied)
    _shared_models_applied = _models_sd_applied & _models_dd_applied

    # ── Check if a previously selected model is still valid after indicator change
    if st.session_state["_pending_indicator_check"]:
        st.session_state["_pending_indicator_check"] = False
        _prev_model = _cfg_applied["model_choice"]
        if _prev_model != _MODEL_ENSEMBLE_MEAN:
            _in_sd = _prev_model in _models_sd_applied
            _in_dd = _prev_model in _models_dd_applied
            if not _in_sd and not _in_dd:
                # Model not available for new indicator in either method — fall back
                st.session_state["applied"]["model_choice"] = _MODEL_ENSEMBLE_MEAN
                st.toast(
                    f"⚠️ Model **{_prev_model}** is not available for "
                    f"**{_cfg_applied['indicator']}** in either method. "
                    f"Switched to Ensemble mean.",
                    icon="⚠️",
                )
                st.rerun()

    with st.sidebar:
        logo_path = Path("logos/esnz_logo_horz_new.png")
        if logo_path.exists(): st.image(str(logo_path))

        _sc.html(f"""
<style>
body {{ margin:0; padding:4px 4px 8px; font-family:Arial,sans-serif; background:transparent; }}
.lbl {{ font-size:13px; color:#31333f; font-weight:600; display:block; margin:0 0 6px; }}
.toggle-wrap {{ display:flex; gap:0; background:#f0f2f6; border-radius:22px; padding:3px; }}
.m-btn {{
  flex:1; padding:6px 0; font-size:12px; font-weight:700; cursor:pointer;
  border:none; border-radius:18px; background:transparent; color:#666;
  transition:background 0.18s, color 0.18s; white-space:nowrap; text-align:center;
}}
.m-btn.active-sd {{ background:#4a90d9; color:#fff; box-shadow:0 1px 4px rgba(74,144,217,0.4); }}
.m-btn.active-dd {{ background:#22a822; color:#fff; box-shadow:0 1px 4px rgba(34,168,34,0.4); }}
.m-btn.disabled  {{ opacity:0.35; cursor:not-allowed; }}
.caption {{ font-size:11px; color:#888; margin-top:6px; }}
#lock-msg {{ font-size:11px; color:#b05000; margin-top:4px; display:none; }}
</style>
<span class="lbl">Downscaling method</span>
<div class="toggle-wrap">
  <button class="m-btn" id="btn-sd" onclick="pick('sd')">📊 Statistical</button>
  <button class="m-btn" id="btn-dd" onclick="pick('dd')">🌀 Dynamical</button>
</div>
<div class="caption">Statistical = AI 12 km · Dynamical = CCAM 5 km</div>
<div id="lock-msg"></div>
<script>
(function() {{
  var active = '{_current_method}';
  var locked = {_json.dumps(
      "sd" if st.session_state.get("_model_locked_to") == "sd"
      else "dd" if st.session_state.get("_model_locked_to") == "dd"
      else None
  )};

  function render() {{
    var sdBtn = document.getElementById('btn-sd');
    var ddBtn = document.getElementById('btn-dd');
    var lockMsg = document.getElementById('lock-msg');
    sdBtn.className = 'm-btn' + (active==='sd' ? ' active-sd' : '');
    ddBtn.className = 'm-btn' + (active==='dd' ? ' active-dd' : '');
    if (locked === 'sd') {{
      ddBtn.classList.add('disabled');
      ddBtn.title = 'Not available for this model';
      lockMsg.style.display = 'block';
      lockMsg.textContent = '⚠️ Locked to Statistical — model not in Dynamical';
    }} else if (locked === 'dd') {{
      sdBtn.classList.add('disabled');
      sdBtn.title = 'Not available for this model';
      lockMsg.style.display = 'block';
      lockMsg.textContent = '⚠️ Locked to Dynamical — model not in Statistical';
    }} else {{
      lockMsg.style.display = 'none';
    }}
  }}

  window.pick = function(mth) {{
    if (locked && mth !== locked) return;
    if (mth === active) return;
    active = mth;
    render();
    var frames = window.parent.document.querySelectorAll('iframe');
    for (var i = 0; i < frames.length; i++) {{
      try {{ frames[i].contentWindow.postMessage({{type:'nzmap_method', method:mth}}, '*'); }} catch(e) {{}}
    }}
  }};
  render();
}})();
</script>
""", height=105)

        st.markdown("---")

        with st.form("map_controls", border=False):
            st.subheader("Future scenario (SSP)")
            _sel_ssp = st.selectbox("SSP", SSP_OPTIONS, format_func=lambda s: SSP_LABELS[s],
                label_visibility="collapsed",
                index=SSP_OPTIONS.index(st.session_state["applied"]["ssp"]), key="_sel_ssp")

            st.subheader("Baseline period")
            _sel_bp_tag = st.selectbox("Baseline", BP_OPTIONS, format_func=lambda b: BP_LABELS[b],
                label_visibility="collapsed",
                index=BP_OPTIONS.index(st.session_state["applied"]["bp_tag"]), key="_sel_bp_tag")

            st.markdown("---")

            st.subheader("Indicator")
            _ind_default = st.session_state["applied"]["indicator"]
            _grouped_options = _build_grouped_indicator_options(indicators_avail)
            _ind_idx = next((i for i, v in enumerate(_grouped_options) if v == _ind_default), 0)
            _sel_indicator = st.selectbox(
                "Indicator", _grouped_options,
                format_func=lambda i: (
                    f"{'━' * 18}" if i.startswith(f"{_SEPARATOR_PREFIX}━")
                    else f"{'╌' * 6}  {i[len(_SEPARATOR_PREFIX)+1:].upper()}  {'╌' * 6}"
                    if i.startswith(_SEPARATOR_PREFIX)
                    else f"{i} — {INDICATOR_LABELS.get(i, '')}"
                ),
                label_visibility="collapsed", index=_ind_idx, key="_sel_indicator")

            # Use the selected indicator for downstream discovery (not yet applied)
            _preview_ind = (_sel_indicator if not _sel_indicator.startswith(_SEPARATOR_PREFIX)
                            else st.session_state["applied"]["indicator"])

            # ── Season discovery ──────────────────────────────────────────────
            _applied_ind = st.session_state["applied"]["indicator"]
            if _ON_CLOUD:
                _seasons_avail = []
                for _s in SEASON_LABELS:
                    _probe = (load_uncertainty_cache(_applied_ind, st.session_state["applied"]["ssp"],
                                                     st.session_state["applied"]["bp_tag"], _s, method="sd")
                              or load_uncertainty_cache(_applied_ind, st.session_state["applied"]["ssp"],
                                                        st.session_state["applied"]["bp_tag"], _s, method="dd"))
                    if _probe is not None:
                        _seasons_avail.append(_s)
                if not _seasons_avail:
                    _seasons_avail = [SEASON_ANN]
            else:
                _seas_sd = set(discover_seasons("historical", _applied_ind, method="sd"))
                _seas_dd = set(discover_seasons("historical", _applied_ind, method="dd"))
                _seasons_avail = sorted(_seas_sd | _seas_dd,
                                        key=lambda s: list(SEASON_LABELS).index(s) if s in SEASON_LABELS else 99)

            if len(_seasons_avail) > 1:
                st.subheader("Season")
                _seas_default = st.session_state["applied"]["season"]
                _seas_idx = (_seasons_avail.index(_seas_default) if _seas_default in _seasons_avail else 0)
                _sel_season = st.selectbox("Season", _seasons_avail,
                    format_func=lambda s: SEASON_LABELS.get(s, s),
                    label_visibility="collapsed", index=_seas_idx, key="_sel_season")
            else:
                _sel_season = SEASON_ANN

            st.subheader("Model")

            # ── Model discovery: ALWAYS from uncertainty cache ─────────────────
            # Use the *preview* indicator/ssp/bp/season so the dropdown updates
            # immediately when the user changes the indicator before hitting Apply.
            _preview_models_sd, _preview_models_dd = get_models_from_cache(
                _preview_ind, _sel_ssp, _sel_bp_tag, _sel_season
            )
            _avail_models    = sorted(_preview_models_sd | _preview_models_dd)
            _shared_models   = _preview_models_sd & _preview_models_dd

            def _model_label(m):
                if m == _MODEL_ENSEMBLE_MEAN: return m
                tags = []
                if m in _preview_models_sd: tags.append("SD")
                if m in _preview_models_dd: tags.append("DD")
                star = "  ★" if m in _shared_models else ""
                return f"{m} [{'/'.join(tags)}]{star}"

            _model_options = [_MODEL_ENSEMBLE_MEAN] + _avail_models
            _mod_default   = st.session_state["applied"]["model_choice"]
            # If current model not in the new preview list, default to ensemble
            _mod_idx = (_model_options.index(_mod_default)
                        if _mod_default in _model_options else 0)
            _sel_model = st.selectbox("Model", _model_options,
                format_func=_model_label, label_visibility="collapsed",
                help=(f"★ = available in both methods ({len(_shared_models)} shared)  |  "
                      f"Labels show which methods each model appears in"),
                index=_mod_idx, key="_sel_model")

            st.markdown("---")
            _submitted = st.form_submit_button("▶  Apply", type="primary", use_container_width=True)

        if _submitted:
            if _sel_indicator.startswith(_SEPARATOR_PREFIX):
                _sel_indicator = st.session_state["applied"]["indicator"]

            # Guard: make sure chosen model is actually in the cache for this config.
            # This is the authoritative check — no raw file scanning.
            _submit_models_sd, _submit_models_dd = get_models_from_cache(
                _sel_indicator, _sel_ssp, _sel_bp_tag, _sel_season
            )
            _valid_model_set = _submit_models_sd | _submit_models_dd | {_MODEL_ENSEMBLE_MEAN}
            if _sel_model not in _valid_model_set:
                _sel_model = _MODEL_ENSEMBLE_MEAN

            _indicator_changed = (_sel_indicator != st.session_state["applied"]["indicator"])
            st.session_state["applied"] = dict(
                ssp=_sel_ssp, bp_tag=_sel_bp_tag,
                indicator=_sel_indicator, season=_sel_season, model_choice=_sel_model)

            # If indicator changed, trigger the post-rerun availability check
            if _indicator_changed:
                st.session_state["_pending_indicator_check"] = True

            st.rerun()

        _cfg            = st.session_state["applied"]
        ssp             = _cfg["ssp"]
        bp_tag          = _cfg["bp_tag"]
        indicator       = _cfg["indicator"]
        season          = _cfg["season"]
        model_choice    = _cfg["model_choice"]
        selected_model_key = None if model_choice == _MODEL_ENSEMBLE_MEAN else model_choice

        # ── Determine per-method model availability from cache (authoritative) ─
        # These are for the *applied* config (what's actually rendered)
        _models_sd_current = _models_sd_applied
        _models_dd_current = _models_dd_applied

        _model_in_sd = (selected_model_key is None) or (selected_model_key in _models_sd_current)
        _model_in_dd = (selected_model_key is None) or (selected_model_key in _models_dd_current)

        # Lock logic: only lock when a specific model is chosen and it's genuinely
        # absent from one method's cache
        if selected_model_key is not None:
            if _model_in_sd and not _model_in_dd:
                _locked_method = "sd"
                st.session_state["_model_locked_to"] = "sd"
                if _current_method != "sd":
                    st.session_state["method"] = "sd"
                    _current_method = "sd"
                    st.rerun()
            elif _model_in_dd and not _model_in_sd:
                _locked_method = "dd"
                st.session_state["_model_locked_to"] = "dd"
                if _current_method != "dd":
                    st.session_state["method"] = "dd"
                    _current_method = "dd"
                    st.rerun()
            else:
                # Present in both (or neither — treat as unlocked, ensemble fallback)
                _locked_method = None
                st.session_state["_model_locked_to"] = None
        else:
            _locked_method = None
            st.session_state["_model_locked_to"] = None

        method = _current_method

        colorscale     = colorscale_for(indicator)
        colorscale_abs = colorscale_abs_for(indicator)

        _sc.html("""
<style>
body{margin:0;padding:0 4px;font-family:Arial,sans-serif;background:transparent;}
.lbl{font-size:13px;color:#31333f;font-weight:600;display:block;margin:10px 0 4px;}
.lbl:first-of-type{margin-top:0;}
#oSlider{width:100%;accent-color:#4a90d9;cursor:pointer;display:block;margin-bottom:1px;}
#oVal{font-size:11px;color:#888;float:right;}
#sSelect{width:100%;font-size:12px;padding:3px 6px;border:1px solid #ccc;border-radius:4px;background:#fafafa;cursor:pointer;color:#333;margin-top:2px;}
</style>
<span class="lbl">Overlay opacity <span id="oVal">0.70</span></span>
<input type="range" id="oSlider" min="0.1" max="1.0" step="0.05" value="0.70">
<span class="lbl">Animation speed</span>
<select id="sSelect">
  <option value="2000">Very Slow</option><option value="900">Slow</option>
  <option value="400" selected>Medium</option><option value="160">Fast</option>
</select>
<script>
(function(){
  function send(){
    var msg={type:'nzmap',opacity:parseFloat(document.getElementById('oSlider').value),
              frameMs:parseInt(document.getElementById('sSelect').value,10)};
    var frames=window.parent.document.querySelectorAll('iframe');
    for(var i=0;i<frames.length;i++)try{frames[i].contentWindow.postMessage(msg,'*');}catch(e){}
  }
  document.getElementById('oSlider').addEventListener('input',function(){
    document.getElementById('oVal').textContent=parseFloat(this.value).toFixed(2); send();
  });
  document.getElementById('sSelect').addEventListener('change',send);
})();
</script>
""", height=110)

        dot_opacity = 0.7
        frame_ms    = 400

    # ========================================================================
    # Main execution
    # ========================================================================
    st.title("🗺️ NZ Climate Indicator Map")

    if _DEMO_MODE:
        st.info("🧪 **Demo mode** — running on synthetic test data. See `test/README.md`.")

    season_label = SEASON_LABELS.get(season, season)
    bp_short     = bp_tag.replace("bp","").replace("-","–")
    model_label  = selected_model_key if selected_model_key else "Ensemble mean"

    # ── 1. Build timelines ────────────────────────────────────────────────────
    SNAPSHOTS_SD = build_timeline(indicator, bp_tag, ssp, season, method="sd")
    SNAPSHOTS_DD = build_timeline(indicator, bp_tag, ssp, season, method="dd")
    SNAPSHOTS = SNAPSHOTS_SD if SNAPSHOTS_SD else SNAPSHOTS_DD

    if not SNAPSHOTS and _ON_CLOUD:
        _unc_tmp = load_uncertainty_cache(indicator, ssp, bp_tag, season, method="sd")
        if _unc_tmp is None:
            _unc_tmp = load_uncertainty_cache(indicator, ssp, bp_tag, season, method="dd")
        if _unc_tmp is not None:
            SNAPSHOTS = []
            for yr in _unc_tmp["snap_years"]:
                if yr < 2015:
                    SNAPSHOTS.append((f"Historical {bp_tag.replace('bp','').replace('-','–')}",
                                      "historical", bp_tag, bp_tag, yr))
                else:
                    fp = f"fp{int(yr)-9}-{int(yr)+10}"
                    SNAPSHOTS.append((f"{int(yr)-9}–{int(yr)+10}", ssp, fp, bp_tag, yr))
            SNAPSHOTS = sorted(SNAPSHOTS, key=lambda x: x[4])

    if not SNAPSHOTS:
        st.error(f"No files found for indicator **{indicator}**, baseline **{bp_tag}**, season **{season}**."); st.stop()

    SNAP_COLOURS = ["#4a90d9","#1a5fa8","#e8a020","#e05a20","#c0392b","#8e1a1a","#5a0f0f","#2d0808"]

    pills = "".join(
        f'<span style="display:inline-block;padding:4px 14px;border-radius:16px;'
        f'background:{SNAP_COLOURS[min(i,len(SNAP_COLOURS)-1)]};color:white;'
        f'font-size:0.78rem;font-weight:600;margin-right:6px;margin-bottom:4px">'
        f'{label}</span>'
        for i,(label,_,_,_,_) in enumerate(SNAPSHOTS))
    st.markdown(
        f'<div style="margin-bottom:10px">{pills}'
        f'<span style="font-size:0.78rem;color:#888">'
        f'← {len(SNAPSHOTS)} snapshots · toggle SD/DD without losing map position'
        f'</span></div>', unsafe_allow_html=True)

    import streamlit.components.v1 as _components

    _map_slot = st.empty()
    with _map_slot.container():
        _components.html(build_loading_screen_html(nz_loader_svg_data(), height=630), height=630, scrolling=False)

    # ── 2. Load uncertainty caches ────────────────────────────────────────────
    _unc_sd = load_uncertainty_cache(indicator, ssp, bp_tag, season, method="sd")
    _unc_dd = load_uncertainty_cache(indicator, ssp, bp_tag, season, method="dd")

    if _unc_sd is None and _unc_dd is None:
        st.error(
            "Uncertainty cache not found for either downscaling method.\n\n"
            "Run:\n```\npython helper_scripts/precompute_uncertainty.py --method sd\n"
            "python helper_scripts/precompute_uncertainty.py --method dd\n```"); st.stop()

    from scipy.spatial import KDTree as _KDTree

    def _build_method_data_native(unc, method_key):
        if unc is None:
            return dict(
                stacked=None, stacked_abs=None,
                p5=None, p25=None, p75=None, p95=None, ens=None,
                abs_p5=None, abs_p25=None, abs_p75=None, abs_p95=None, abs_ens=None,
                ymin=None, ymax=None, aymin=None, aymax=None,
                fp_ranges=None, lat_v=None, lon_v=None,
            )

        own_lat_v = unc["lat_v"]
        own_lon_v = unc["lon_v"]
        n_pts = len(own_lat_v)
        _unc_snap_years = unc["snap_years"]

        def _get_row(band, yr):
            dists = [abs(yr - uy) for uy in _unc_snap_years]
            best  = int(np.argmin(dists))
            return band[best] if dists[best] < 2.0 else np.full(n_pts, np.nan)

        def _reindex(band):
            return np.stack([_get_row(band, yr) for _,_,_,_,yr in SNAPSHOTS], axis=0)

        # ── Model vs ensemble selection ───────────────────────────────────────
        # Check if this specific method has the selected model in its cache
        _this_method_has_model = (
            selected_model_key is not None and
            selected_model_key in unc.get("model_change_vals", {})
        )

        if _this_method_has_model:
            stacked = np.stack([_get_row(unc["model_change_vals"][selected_model_key], yr)
                                for _,_,_,_,yr in SNAPSHOTS], axis=0)
        else:
            stacked = _reindex(unc["ens_vals"])

        if _this_method_has_model and selected_model_key in unc.get("model_abs_vals", {}):
            stacked_abs = np.stack([_get_row(unc["model_abs_vals"][selected_model_key], yr)
                                    for _,_,_,_,yr in SNAPSHOTS], axis=0)
        else:
            stacked_abs = _reindex(unc["abs_ens_vals"])

        ens     = (_reindex(unc["ens_vals"])     if _this_method_has_model else None)
        abs_ens = (_reindex(unc["abs_ens_vals"]) if _this_method_has_model else None)

        def _ab(key):
            return _reindex(unc[key]) if key in unc else np.full((len(SNAPSHOTS), n_pts), np.nan)

        fp_ranges = []
        for (_,_,_,_,yr) in SNAPSHOTS:
            dists = [abs(yr-uy) for uy in _unc_snap_years]
            best  = int(np.argmin(dists))
            fp_ranges.append(unc["snap_fp_ranges"][best] if dists[best]<2.0 else str(int(round(yr))))

        return dict(
            stacked=stacked, stacked_abs=stacked_abs,
            p5 =_reindex(unc["p5_vals"]),  p25=_reindex(unc["p25_vals"]),
            p75=_reindex(unc["p75_vals"]), p95=_reindex(unc["p95_vals"]),
            ens=ens,
            abs_p5 =_ab("abs_p5_vals"),  abs_p25=_ab("abs_p25_vals"),
            abs_p75=_ab("abs_p75_vals"), abs_p95=_ab("abs_p95_vals"),
            abs_ens=abs_ens,
            ymin=unc.get("chart_ymin"),     ymax=unc.get("chart_ymax"),
            aymin=unc.get("abs_chart_ymin"),aymax=unc.get("abs_chart_ymax"),
            fp_ranges=fp_ranges,
            lat_v=own_lat_v, lon_v=own_lon_v,
        )

    with st.spinner("Loading model uncertainty ranges for both methods…"):
        _dat_sd = _build_method_data_native(_unc_sd, "sd")
        _dat_dd = _build_method_data_native(_unc_dd, "dd")

    _fallback_unc = _unc_sd if _unc_sd is not None else _unc_dd
    _fallback_lat_v = _fallback_unc["lat_v"]
    _fallback_lon_v = _fallback_unc["lon_v"]

    _sd_lat_v = _dat_sd["lat_v"] if _dat_sd["lat_v"] is not None else _fallback_lat_v
    _sd_lon_v = _dat_sd["lon_v"] if _dat_sd["lon_v"] is not None else _fallback_lon_v
    _dd_lat_v = _dat_dd["lat_v"] if _dat_dd["lat_v"] is not None else _fallback_lat_v
    _dd_lon_v = _dat_dd["lon_v"] if _dat_dd["lon_v"] is not None else _fallback_lon_v

    # ── 3. Colour ranges ──────────────────────────────────────────────────────
    with st.spinner("Computing colour ranges…"):
        shared_half      = compute_color_range(indicator)
        abs_vmin, abs_vmax = compute_abs_color_range(indicator)

    _chg_vmin = 0.0 if indicator in _REC_INDICATORS else -shared_half
    _chg_vmax = 100.0 if indicator in _REC_INDICATORS else shared_half

    # ── 4. Render frames ──────────────────────────────────────────────────────
    def _compute_hover_thresh(lat_v, lon_v):
        from scipy.spatial import KDTree
        def merc(l): return np.log(np.tan(np.pi/4 + np.deg2rad(l)/2))
        pts = np.column_stack([np.deg2rad(lon_v), merc(lat_v)])
        kd  = KDTree(pts)
        n   = min(2000, len(pts))
        idx = np.random.default_rng(0).choice(len(pts), n, replace=False)
        d, _ = kd.query(pts[idx], k=2)
        gs  = float(np.median(d[:, 1]))
        return float(np.rad2deg(gs * 0.55))

    def _render_method(dat, mth, lat_v, lon_v):
        lm_chg = _log_mode(indicator, is_change=True)
        lm_abs = _log_mode(indicator, is_change=False)
        key_c  = _frame_cache_key(indicator, ssp, bp_tag, season, selected_model_key,
                                  colorscale, _chg_vmin, _chg_vmax, lm_chg)
        cached = _load_frame_cache(key_c, method=mth)
        if cached:
            b64_c, lat_min, lat_max, lon_min, lon_max, thresh = cached
        else:
            stacked = dat["stacked"] if dat["stacked"] is not None else np.full((len(SNAPSHOTS), len(lat_v)), np.nan)
            with st.spinner(f"Rendering {len(SNAPSHOTS)} change frames ({mth.upper()})…"):
                b64_c, lat_min, lat_max, lon_min, lon_max, thresh = prerender_snapshots(
                    lat_v, lon_v, stacked, _chg_vmin, _chg_vmax,
                    colorscale, indicator, is_change=(indicator not in _REC_INDICATORS))
            _save_frame_cache(key_c, (b64_c,lat_min,lat_max,lon_min,lon_max,thresh), method=mth)

        key_a    = _frame_cache_key(indicator, ssp, bp_tag, season, selected_model_key,
                                    colorscale_abs, abs_vmin, abs_vmax, lm_abs)
        cached_a = _load_frame_cache(key_a, method=mth)
        if cached_a:
            b64_a = cached_a[0]
        else:
            stacked_abs = dat["stacked_abs"] if dat["stacked_abs"] is not None else np.full((len(SNAPSHOTS), len(lat_v)), np.nan)
            with st.spinner(f"Rendering {len(SNAPSHOTS)} absolute frames ({mth.upper()})…"):
                res_a = prerender_snapshots(lat_v, lon_v, stacked_abs,
                                            abs_vmin, abs_vmax, colorscale_abs,
                                            indicator, is_change=False)
            b64_a = res_a[0]
            _save_frame_cache(key_a, res_a, method=mth)

        return b64_c, b64_a, lat_min, lat_max, lon_min, lon_max, thresh

    _b64c_sd, _b64a_sd, lat_min_sd, lat_max_sd, lon_min_sd, lon_max_sd, thresh_sd = _render_method(_dat_sd, "sd", _sd_lat_v, _sd_lon_v)
    _b64c_dd, _b64a_dd, lat_min_dd, lat_max_dd, lon_min_dd, lon_max_dd, thresh_dd = _render_method(_dat_dd, "dd", _dd_lat_v, _dd_lon_v)

    lat_min, lat_max, lon_min, lon_max = lat_min_sd, lat_max_sd, lon_min_sd, lon_max_sd
    thresh    = _compute_hover_thresh(_sd_lat_v, _sd_lon_v)
    thresh_dd = _compute_hover_thresh(_dd_lat_v, _dd_lon_v)

    # ── 5. Colour bars ────────────────────────────────────────────────────────
    units_note     = "%" if indicator in _REC_INDICATORS else f"Δ {INDICATOR_UNITS.get(indicator,'')}"
    abs_units_note = _REC_ABS_UNITS.get(indicator,"") if indicator in _REC_INDICATORS else INDICATOR_UNITS.get(indicator,"")

    with st.spinner("Rendering colour bars…"):
        cb_chg = render_colorbar_b64(_chg_vmin, _chg_vmax, colorscale, units_note,
                                      indicator=indicator, is_change=(indicator not in _REC_INDICATORS))
        cb_abs = render_colorbar_b64(abs_vmin, abs_vmax, colorscale_abs, abs_units_note,
                                      indicator=indicator, is_change=False)

    # ── 6. Virtual frame timeline ─────────────────────────────────────────────
    snap_years     = [yr    for _,_,_,_,yr    in SNAPSHOTS]
    snap_labels    = [label for label,_,_,_,_ in SNAPSHOTS]
    frame_years_all, snap_frame_idx = compute_frame_timeline(snap_years, YEAR_STEP)

    # ── 7. Hover value stacks ─────────────────────────────────────────────────
    def _safe_tolist(arr):
        if arr is None: return None
        return [row.tolist() for row in arr]

    _sd_hover     = _safe_tolist(_dat_sd["stacked"])
    _sd_hover_abs = _safe_tolist(_dat_sd["stacked_abs"])
    _dd_hover     = _safe_tolist(_dat_dd["stacked"])
    _dd_hover_abs = _safe_tolist(_dat_dd["stacked_abs"])

    def _tolist_or_none(arr):
        if arr is None: return None
        return [row.tolist() for row in arr]

    # ── 8. Build player ───────────────────────────────────────────────────────
    _borders = load_borders_geojson()

    header_html = (
        f"<b>{indicator}</b> — {INDICATOR_LABELS.get(indicator,'')} &nbsp;|&nbsp; "
        f"<b>Season:</b> {season_label} &nbsp;|&nbsp; "
        f"<b>Baseline:</b> {bp_short} &nbsp;|&nbsp; "
        f"<b>Future:</b> {SSP_LABELS[ssp]} &nbsp;|&nbsp; "
        f"<b>Model:</b> {model_label} &nbsp;|&nbsp; "
        f"<span style='color:#8a4000'>Left = absolute</span> &nbsp;·&nbsp; "
        f"<span style='color:#1a4a9a'>Right = Δ change</span>"
    )

    player_html = build_html_player(
        sd_snap_b64=_b64c_sd, sd_snap_b64_abs=_b64a_sd,
        sd_colorbar_b64=cb_chg, sd_colorbar_b64_abs=cb_abs,
        sd_hover_vals=_sd_hover, sd_hover_abs_vals=_sd_hover_abs,
        sd_chart_p5 =_tolist_or_none(_dat_sd["p5"]),
        sd_chart_p25=_tolist_or_none(_dat_sd["p25"]),
        sd_chart_p75=_tolist_or_none(_dat_sd["p75"]),
        sd_chart_p95=_tolist_or_none(_dat_sd["p95"]),
        sd_chart_ens=_tolist_or_none(_dat_sd["ens"]),
        sd_abs_chart_p5 =_tolist_or_none(_dat_sd["abs_p5"]),
        sd_abs_chart_p25=_tolist_or_none(_dat_sd["abs_p25"]),
        sd_abs_chart_p75=_tolist_or_none(_dat_sd["abs_p75"]),
        sd_abs_chart_p95=_tolist_or_none(_dat_sd["abs_p95"]),
        sd_abs_chart_ens=_tolist_or_none(_dat_sd["abs_ens"]),
        sd_chart_ymin=_dat_sd["ymin"], sd_chart_ymax=_dat_sd["ymax"],
        sd_abs_chart_ymin=_dat_sd["aymin"], sd_abs_chart_ymax=_dat_sd["aymax"],
        sd_snap_fp_ranges=_dat_sd["fp_ranges"],
        dd_snap_b64=_b64c_dd, dd_snap_b64_abs=_b64a_dd,
        dd_colorbar_b64=cb_chg, dd_colorbar_b64_abs=cb_abs,
        dd_hover_vals=_dd_hover, dd_hover_abs_vals=_dd_hover_abs,
        dd_chart_p5 =_tolist_or_none(_dat_dd["p5"]),
        dd_chart_p25=_tolist_or_none(_dat_dd["p25"]),
        dd_chart_p75=_tolist_or_none(_dat_dd["p75"]),
        dd_chart_p95=_tolist_or_none(_dat_dd["p95"]),
        dd_chart_ens=_tolist_or_none(_dat_dd["ens"]),
        dd_abs_chart_p5 =_tolist_or_none(_dat_dd["abs_p5"]),
        dd_abs_chart_p25=_tolist_or_none(_dat_dd["abs_p25"]),
        dd_abs_chart_p75=_tolist_or_none(_dat_dd["abs_p75"]),
        dd_abs_chart_p95=_tolist_or_none(_dat_dd["abs_p95"]),
        dd_abs_chart_ens=_tolist_or_none(_dat_dd["abs_ens"]),
        dd_chart_ymin=_dat_dd["ymin"], dd_chart_ymax=_dat_dd["ymax"],
        dd_abs_chart_ymin=_dat_dd["aymin"], dd_abs_chart_ymax=_dat_dd["aymax"],
        dd_snap_fp_ranges=_dat_dd["fp_ranges"],
        sd_hover_lats=_sd_lat_v.tolist(), sd_hover_lons=_sd_lon_v.tolist(),
        dd_hover_lats=_dd_lat_v.tolist(), dd_hover_lons=_dd_lon_v.tolist(),
        snap_years=snap_years, frame_years=frame_years_all,
        snap_frame_idx=snap_frame_idx, snap_labels=snap_labels,
        hover_units=units_note, abs_units=abs_units_note,
        lat_min=lat_min, lat_max=lat_max, lon_min=lon_min, lon_max=lon_max,
        mask_threshold_deg=thresh,
        dd_lat_min=lat_min_dd, dd_lat_max=lat_max_dd,
        dd_lon_min=lon_min_dd, dd_lon_max=lon_max_dd,
        dd_mask_threshold_deg=thresh_dd,
        frame_ms=frame_ms, dot_opacity=dot_opacity, header_html=header_html,
        initial_method=method,
        locked_method=_locked_method,
        country_geojson=_borders["country"], regions_geojson=_borders["regions"],
    )

    with _map_slot.container():
        _components.html(player_html, height=630, scrolling=False)

    # ── 9. Summary statistics ─────────────────────────────────────────────────
    st.markdown("""
<style>
[data-testid="stMetricValue"] { font-size:1.05rem !important; font-weight:600; }
[data-testid="stMetricLabel"] { font-size:0.75rem !important; }
</style>""", unsafe_allow_html=True)

    st.markdown("#### Statistics per snapshot")
    cols = st.columns(len(SNAPSHOTS))
    _unc_for_stats = _unc_sd if _unc_sd is not None else _unc_dd
    _unc_snap_years_pre = _unc_for_stats["snap_years"]

    for i, (label, scen, fp, bp, yr) in enumerate(SNAPSHOTS):
        is_hist  = scen == "historical"
        _best_si = int(np.argmin([abs(yr-sy) for sy in _unc_snap_years_pre]))
        s        = _unc_for_stats["summary_stats"][_best_si]
        clr      = SNAP_COLOURS[min(i, len(SNAP_COLOURS)-1)]
        units_ch = ("%" if indicator in _REC_INDICATORS else f"Δ {INDICATOR_UNITS.get(indicator,'')}")
        units_ab = (_REC_ABS_UNITS.get(indicator,"") if indicator in _REC_INDICATORS
                    else INDICATOR_UNITS.get(indicator,""))
        with cols[i]:
            st.markdown(
                f'<div style="border-left:4px solid {clr};padding-left:8px">'
                f'<strong style="font-size:0.82rem">{label}</strong><br>'
                f'<span style="font-size:0.72rem;color:#888">'
                + ("baseline record chance" if is_hist and indicator in _REC_INDICATORS
                   else "baseline (absolute)" if is_hist
                   else "record chance %" if indicator in _REC_INDICATORS
                   else "change from baseline")
                + f' · {s["n_models"]} models</span></div>', unsafe_allow_html=True)
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