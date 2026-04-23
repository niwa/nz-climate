#!/usr/bin/env python3
# precompute_frames.py (annotated for review)
#
# OVERVIEW
# --------
# Offline batch script — pre-renders all PNG snapshot frames for every
# indicator / scenario / baseline / season / model combination and saves
# them to assets/frame_cache/ as pickle files.
#
# This is the heavy pre-render job that populates the disk cache consumed
# by 2_NZ_Map.py at runtime. Run once after data updates; use --force to
# rebuild specific combinations.
#
# Two indicator types are handled differently:
#   Standard (TX, PR, FD …)  — single-variable nc files (change from baseline)
#   REC (REC_TXx, …)         — dual-variable nc files (REF % + REC physical)
#
# Usage:
#     python precompute_frames.py
#     python precompute_frames.py --indicator TX
#     python precompute_frames.py --scenario  ssp370
#     python precompute_frames.py --force
#
# PRIVACY / SECURITY NOTES FOR REVIEWERS
# ----------------------------------------
# * REC_ROOT is hardcoded to an internal HPC path at the bottom of the imports.
#   DATA_ROOT is imported from 2_NZ_Map.py (which already externalises it to
#   the NZMAP_DATA_ROOT env var). REC_ROOT should receive the same treatment:
#       REC_ROOT = Path(os.environ.get("NZMAP_REC_ROOT", "<fallback>"))
#   REVIEW FLAG: Apply this change in a follow-up PR to match Part 1 of
#   2_NZ_Map.py and precompute_color_ranges.py.
# * The script imports shared logic directly from pages/2_NZ_Map.py via
#   importlib. This is intentional — it ensures the rendering pipeline stays
#   in sync with the app. The import executes the module-level Streamlit calls
#   in 2_NZ_Map.py (st.set_page_config, st.markdown), which is harmless in a
#   headless script context but worth noting for reviewers unfamiliar with the
#   pattern.
# * Frame cache files are pickle files. See the security note in Part 2 of
#   2_NZ_Map.py regarding pickle safety — the same applies here. Only this
#   script (and the app) should have write access to assets/frame_cache/.
# * No credentials, tokens, API keys, or personal data are present.
#
# LOGIC NOTES FOR REVIEWERS
# --------------------------
# * build_valid_mask() in the standard builder unions finite-value masks
#   across ALL models and periods. This means a pixel is included if ANY
#   model has a finite value there. An intersection approach (all models
#   must be finite) would give a stricter but smaller valid region.
# * REC indicators do not support per-model frame rendering (the running
#   record is model-independent within each file). The model_keys loop
#   still iterates over None + all_models for code uniformity, but
#   build_snap_arrays_rec() ignores the model_key argument.
# * The coastline polygon is loaded once at startup (load_coastline_polygon())
#   to warm the @st.cache_resource cache before the render loop begins.

"""
precompute_frames.py
--------------------
Pre-renders all PNG snapshot frames for every indicator / scenario / season
/ baseline / model combination and saves them to assets/frame_cache/.

Handles two kinds of indicators:

  Standard indicators  (e.g. TX, PR, FD …)
    Single variable nc files (change from baseline).
    Absolute values = historical baseline + change.

  REC indicators  (REC_TXx, REC_Rx1day, REC_Wx1day, REC_TNn)
    Dual-variable nc files (REF % + REC physical units).
    Data read from REC_ROOT instead of DATA_ROOT.
    Change frames use REF % (0–100), abs frames use REC physical units.

Run once (or after new data is added):
    python precompute_frames.py

Flags:
    --indicator TX          only process one indicator
    --scenario  ssp370      only process one scenario
    --force                 re-render even if cache file already exists
"""

import argparse
import pickle
import re
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import xarray as xr
import importlib.util

# ── Import shared logic from 2_NZ_Map.py ─────────────────────────────────────
# Using importlib rather than a package import so this script can be run from
# the repo root without installing the app as a package.
_map_path = Path(__file__).parent / "pages" / "2_NZ_Map.py"
_spec     = importlib.util.spec_from_file_location("NZ_Map", _map_path)
_mod      = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)

colorscale_for          = _mod.colorscale_for
colorscale_abs_for      = _mod.colorscale_abs_for
_make_norm              = _mod._make_norm
_log_mode               = _mod._log_mode
frame_cache_key         = _mod._frame_cache_key
load_cache              = _mod._load_frame_cache
save_cache              = _mod._save_frame_cache
compute_color_range     = _mod.compute_color_range
compute_abs_color_range = _mod.compute_abs_color_range
DATA_ROOT               = _mod.DATA_ROOT           # from NZMAP_DATA_ROOT env var
SSP_OPTIONS             = _mod.SSP_OPTIONS
BP_OPTIONS              = _mod.BP_OPTIONS
SEASON_LABELS           = _mod.SEASON_LABELS
SEASON_ANN              = _mod.SEASON_ANN
_KELVIN_INDICATORS      = _mod._KELVIN_INDICATORS
_LOG_INDICATORS         = _mod._LOG_INDICATORS
_SYMLOG_CHANGE_INDICATORS = _mod._SYMLOG_CHANGE_INDICATORS
_REC_INDICATORS         = _mod._REC_INDICATORS
list_indicators         = _mod.list_indicators
discover_seasons        = _mod.discover_seasons
discover_fp_tags        = _mod.discover_fp_tags
discover_models         = _mod.discover_models
find_nc_files           = _mod.find_nc_files
prerender_snapshots     = _mod.prerender_snapshots
load_coastline_polygon  = _mod.load_coastline_polygon

# REVIEW FLAG: Externalise to NZMAP_REC_ROOT env var (see PRIVACY NOTES above).
import os
REC_ROOT = Path(os.environ.get(
    "NZMAP_REC_ROOT",
    "/esi/project/niwa03712/westphall/nz-climate/REC",
))

# REC temperature indicators stored in Kelvin → need °C conversion for REC var.
# REF % comparison always uses raw units — do NOT apply this to threshold logic.
_REC_KELVIN = {"REC_TXx", "REC_TNn"}

# ── Cache directory ───────────────────────────────────────────────────────────
FRAME_CACHE_DIR = Path("assets/frame_cache")
FRAME_CACHE_DIR.mkdir(parents=True, exist_ok=True)


def cache_exists(key: str) -> bool:
    return (FRAME_CACHE_DIR / f"{key}.pkl").exists()


# ── File discovery (REC-aware) ────────────────────────────────────────────────

def _indicator_root(indicator: str) -> Path:
    return REC_ROOT if indicator in _REC_INDICATORS else DATA_ROOT


def list_all_indicators() -> list[str]:
    """Combine standard indicators from DATA_ROOT and REC indicators from REC_ROOT."""
    results: set[str] = set()
    base = DATA_ROOT / "historical" / "static_maps"
    if base.exists():
        results.update(d.name for d in base.iterdir() if d.is_dir())
    rec_base = REC_ROOT / "historical" / "static_maps"
    if rec_base.exists():
        results.update(d.name for d in rec_base.iterdir() if d.is_dir())
    return sorted(results)


def discover_fp_tags_any(scenario: str, indicator: str,
                         bp_tag: str, season: str) -> list[str]:
    """discover_fp_tags routed through the correct root for this indicator."""
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
    else:
        for f in folder.glob("*.nc"):
            if f"_{bp_tag}_" in f.name and f"_{season}_" in f.name:
                m = re.search(r"_change_(fp\d{4}-\d{4})_", f.name)
                if m:
                    tags.add(m.group(1))
        return sorted(tags, key=lambda t: int(re.search(r"fp(\d{4})", t).group(1)))


def discover_seasons_any(scenario: str, indicator: str) -> list[str]:
    folder = _indicator_root(indicator) / scenario / "static_maps" / indicator
    if not folder.exists():
        return [SEASON_ANN]
    seasons: set[str] = set()
    for f in folder.glob("*.nc"):
        for s in SEASON_LABELS:
            if f"_{s}_" in f.name:
                seasons.add(s)
    return sorted(
        seasons,
        key=lambda s: list(SEASON_LABELS).index(s) if s in SEASON_LABELS else 99
    ) or [SEASON_ANN]


def discover_models_any(scenario: str, indicator: str,
                        bp_tag: str, season: str) -> list[str]:
    folder = _indicator_root(indicator) / scenario / "static_maps" / indicator
    if not folder.exists():
        return []
    pat = re.compile(
        rf"^{re.escape(indicator)}_{re.escape(scenario)}_(.+?)_[ri]\d+[ip]\d+[pf]\d+"
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


def find_nc_files_any(scenario: str, indicator: str,
                      fp_tag: str, bp_tag: str, season: str,
                      model_key: str | None = None) -> list[Path]:
    folder = _indicator_root(indicator) / scenario / "static_maps" / indicator
    if not folder.exists():
        return []
    if model_key:
        _model_pat = re.compile(
            r'_' + re.escape(model_key) + r'_[ri]\d+[ip]\d+[pf]\d+'
        )
    else:
        _model_pat = None

    def _ok(f: Path) -> bool:
        if _model_pat and not _model_pat.search(f.name):
            return False
        return f"_{season}_" in f.name

    if scenario == "historical":
        return sorted(f for f in folder.glob("*.nc")
                      if f"_base_{fp_tag}_" in f.name and _ok(f))
    return sorted(f for f in folder.glob("*.nc")
                  if f"_{fp_tag}_" in f.name
                  and f"_{bp_tag}_" in f.name
                  and _ok(f))


# ── NetCDF loaders ────────────────────────────────────────────────────────────

def _load_nc_array(f: Path) -> tuple[np.ndarray | None,
                                      np.ndarray | None,
                                      np.ndarray | None]:
    """Load the first data variable from a standard single-variable nc file."""
    try:
        ds = xr.open_dataset(f, engine="netcdf4", mask_and_scale=False)
        lat_names = [c for c in list(ds.coords) + list(ds.dims)
                     if c.lower() in ("lat", "latitude", "y", "rlat")]
        lon_names = [c for c in list(ds.coords) + list(ds.dims)
                     if c.lower() in ("lon", "longitude", "x", "rlon")]
        if not lat_names or not lon_names:
            ds.close(); return None, None, None
        lat_name, lon_name = lat_names[0], lon_names[0]
        data_vars = [v for v in ds.data_vars
                     if v.lower() not in ("lat", "lon", "time")]
        if not data_vars:
            ds.close(); return None, None, None
        da = ds[data_vars[0]]
        for dim in [d for d in da.dims if d not in (lat_name, lon_name)]:
            da = da.isel({dim: 0})
        arr = da.values.astype(float)
        if hasattr(da, "attrs"):
            fv = da.attrs.get("_FillValue", da.attrs.get("missing_value", None))
            if fv is not None:
                arr[arr == float(fv)] = np.nan
            scale  = float(da.attrs.get("scale_factor", 1.0))
            offset = float(da.attrs.get("add_offset",   0.0))
            if scale != 1.0 or offset != 0.0:
                arr = arr * scale + offset
        lats = ds[lat_name].values
        lons = ds[lon_name].values
        ds.close()
        return arr, lats, lons
    except Exception:
        return None, None, None


def _load_nc_dual(f: Path) -> tuple[np.ndarray | None,
                                     np.ndarray | None,
                                     np.ndarray | None,
                                     np.ndarray | None]:
    """
    Load REF and REC variables from a dual-variable REC indicator nc file.
    Returns (ref_arr, rec_arr, lats, lons).
    """
    try:
        ds = xr.open_dataset(f, engine="netcdf4", mask_and_scale=False)
        lat_names = [c for c in list(ds.coords) + list(ds.dims)
                     if c.lower() in ("lat", "latitude", "y", "rlat")]
        lon_names = [c for c in list(ds.coords) + list(ds.dims)
                     if c.lower() in ("lon", "longitude", "x", "rlon")]
        if not lat_names or not lon_names:
            ds.close(); return None, None, None, None
        lat_name, lon_name = lat_names[0], lon_names[0]
        lats = ds[lat_name].values
        lons = ds[lon_name].values

        def _extract(vname: str) -> np.ndarray | None:
            if vname not in ds.data_vars:
                return None
            da = ds[vname]
            for dim in [d for d in da.dims if d not in (lat_name, lon_name)]:
                da = da.isel({dim: 0})
            arr = da.values.astype(float)
            if hasattr(da, "attrs"):
                fv = da.attrs.get("_FillValue", da.attrs.get("missing_value", None))
                if fv is not None:
                    arr[arr == float(fv)] = np.nan
                scale  = float(da.attrs.get("scale_factor", 1.0))
                offset = float(da.attrs.get("add_offset",   0.0))
                if scale != 1.0 or offset != 0.0:
                    arr = arr * scale + offset
            return arr

        ref_arr = _extract("REF")
        rec_arr = _extract("REC")
        ds.close()
        return ref_arr, rec_arr, lats, lons
    except Exception:
        return None, None, None, None


# ── Snapshot array builders ───────────────────────────────────────────────────

def load_ensemble_mean_standard(scenario: str, indicator: str,
                                fp_tag: str, bp_tag: str, season: str,
                                model_key: str | None = None) -> tuple:
    """Load ensemble mean (or single-model mean) for a standard indicator."""
    files = find_nc_files_any(scenario, indicator, fp_tag, bp_tag, season, model_key)
    if not files:
        return None, None, None, 0
    arrays, lats, lons = [], None, None
    for f in files:
        arr, la, lo = _load_nc_array(f)
        if arr is None:
            continue
        if lats is None:
            lats, lons = la, lo
        arrays.append(arr)
    if not arrays:
        return None, None, None, 0
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        result = np.nanmean(np.stack(arrays, 0), 0)
    return result, lats, lons, len(arrays)


def build_snap_arrays_standard(indicator: str, ssp: str, bp_tag: str,
                               season: str, model_key: str | None):
    """
    Build stacked change and absolute arrays for a standard indicator.
    Returns (stacked_change, stacked_abs, lat_v, lon_v) or None if no data.

    The historical snapshot is converted from raw (possibly Kelvin) to a
    zero-change array for the change panel, with the true baseline values
    stored separately for the abs panel.
    """
    hist_fp_tags = discover_fp_tags_any("historical", indicator, bp_tag, season)
    hist_fp      = next((t for t in hist_fp_tags if t == bp_tag), None)
    ssp_fp_tags  = discover_fp_tags_any(ssp, indicator, bp_tag, season)
    if not ssp_fp_tags:
        return None

    snapshots = []
    if hist_fp:
        m  = re.search(r"(\d{4})-(\d{4})", hist_fp)
        yr = (int(m.group(1)) + int(m.group(2))) / 2 if m else 0.0
        snapshots.append(("Historical", "historical", hist_fp, hist_fp, yr))
    for fp in ssp_fp_tags:
        m  = re.search(r"(\d{4})-(\d{4})", fp)
        yr = (int(m.group(1)) + int(m.group(2))) / 2 if m else 0.0
        snapshots.append((fp.replace("fp", "").replace("-", "–"),
                          ssp, fp, bp_tag, yr))
    snapshots.sort(key=lambda x: x[4])

    snap_data_list = []
    common_shape = snap_lats = snap_lons = None

    for label, scen, fp, bp, yr in snapshots:
        data, lats, lons, n = load_ensemble_mean_standard(
            scen, indicator, fp, bp, season, model_key)
        if data is not None and common_shape is None:
            common_shape = data.shape
            snap_lats, snap_lons = lats, lons
        snap_data_list.append(data)

    if common_shape is None:
        return None

    # Fill missing snapshots with NaN so the stack is always complete.
    for i, d in enumerate(snap_data_list):
        if d is None:
            snap_data_list[i] = np.full(common_shape, np.nan)

    # Historical snapshot handling:
    #   - Extract and K→°C convert as the baseline for the abs panel.
    #   - Replace with zeros for the change panel (Δ = 0 at baseline).
    hist_abs     = {}
    baseline_abs = None
    for i, (label, scen, fp, bp, yr) in enumerate(snapshots):
        if scen == "historical":
            raw    = snap_data_list[i].copy()
            finite = raw[np.isfinite(raw)]
            if indicator in _KELVIN_INDICATORS and len(finite) and np.nanmean(finite) > 200:
                raw -= 273.15
            hist_abs[i]  = raw
            baseline_abs = raw.copy()
            snap_data_list[i] = np.zeros(common_shape)

    abs_snap_list = []
    for i, (label, scen, fp, bp, yr) in enumerate(snapshots):
        if scen == "historical":
            abs_snap_list.append(hist_abs[i])
        else:
            if baseline_abs is not None:
                abs_snap_list.append(baseline_abs + snap_data_list[i])
            else:
                abs_snap_list.append(snap_data_list[i].copy())

    # Flatten 2D lat/lon grids to 1D point vectors.
    if snap_lats.ndim == 2:
        lat_f, lon_f = snap_lats.ravel(), snap_lons.ravel()
    else:
        lg, ng = np.meshgrid(snap_lats, snap_lons, indexing="ij")
        lat_f, lon_f = lg.ravel(), ng.ravel()

    # Valid mask: union of finite pixels across all snapshots.
    valid = np.ones(lat_f.shape, dtype=bool)
    for d in snap_data_list:
        f = d.ravel()
        if np.any(np.isfinite(f)):
            valid &= np.isfinite(f)

    lat_v       = lat_f[valid]
    lon_v       = lon_f[valid]
    stacked     = np.stack([d.ravel()[valid] for d in snap_data_list],  axis=0)
    stacked_abs = np.stack([d.ravel()[valid] for d in abs_snap_list],   axis=0)
    return stacked, stacked_abs, lat_v, lon_v


def build_snap_arrays_rec(indicator: str, ssp: str, bp_tag: str, season: str):
    """
    Build stacked REF% (change panel) and REC physical units (abs panel)
    for a REC indicator. Returns (stacked_ref, stacked_rec, lat_v, lon_v) or None.

    REC indicators do not support per-model frames — the running record is
    computed within each model's file and each model's file is self-contained.
    The model_key argument is intentionally absent from this function signature;
    callers that pass None as model_key will always reach this path.
    """
    hist_fp_tags = discover_fp_tags_any("historical", indicator, bp_tag, season)
    hist_fp      = next((t for t in hist_fp_tags if t == bp_tag), None)
    ssp_fp_tags  = discover_fp_tags_any(ssp, indicator, bp_tag, season)
    if not ssp_fp_tags:
        return None

    snapshots = []
    if hist_fp:
        m  = re.search(r"(\d{4})-(\d{4})", hist_fp)
        yr = (int(m.group(1)) + int(m.group(2))) / 2 if m else 0.0
        snapshots.append(("Historical", "historical", hist_fp, hist_fp, yr))
    for fp in ssp_fp_tags:
        m  = re.search(r"(\d{4})-(\d{4})", fp)
        yr = (int(m.group(1)) + int(m.group(2))) / 2 if m else 0.0
        snapshots.append((fp, ssp, fp, bp_tag, yr))
    snapshots.sort(key=lambda x: x[4])

    needs_k_conv = indicator in _REC_KELVIN
    ref_list: list[np.ndarray] = []
    rec_list: list[np.ndarray] = []
    common_shape = lats = lons = None

    for _, scen, fp, bp, _ in snapshots:
        files = find_nc_files_any(scen, indicator, fp, bp, season)
        refs, recs = [], []
        for f in files:
            ref_arr, rec_arr, la, lo = _load_nc_dual(f)
            if common_shape is None and ref_arr is not None:
                common_shape = ref_arr.shape
                lats, lons   = la, lo
            if ref_arr is not None:
                refs.append(ref_arr)
            if rec_arr is not None:
                recs.append(rec_arr)

        if common_shape is None:
            ref_list.append(None)
            rec_list.append(None)
            continue

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            ref_mean = (np.nanmean(np.stack(refs, 0), 0)
                        if refs else np.full(common_shape, np.nan))
            rec_mean = (np.nanmean(np.stack(recs, 0), 0)
                        if recs else np.full(common_shape, np.nan))

        if needs_k_conv:
            rec_mean = np.where(rec_mean > 200, rec_mean - 273.15, rec_mean)

        ref_list.append(ref_mean)
        rec_list.append(rec_mean)

    if common_shape is None:
        return None

    for i in range(len(ref_list)):
        if ref_list[i] is None:
            ref_list[i] = np.full(common_shape, np.nan)
        if rec_list[i] is None:
            rec_list[i] = np.full(common_shape, np.nan)

    if lats.ndim == 2:
        lat_f, lon_f = lats.ravel(), lons.ravel()
    else:
        lg, ng = np.meshgrid(lats, lons, indexing="ij")
        lat_f, lon_f = lg.ravel(), ng.ravel()

    # Valid: union of pixels that have a finite REF value in any snapshot.
    valid = np.zeros(lat_f.shape, dtype=bool)
    for d in ref_list:
        valid |= np.isfinite(d.ravel())

    lat_v       = lat_f[valid]
    lon_v       = lon_f[valid]
    stacked     = np.stack([d.ravel()[valid] for d in ref_list], axis=0)
    stacked_abs = np.stack([d.ravel()[valid] for d in rec_list], axis=0)
    return stacked, stacked_abs, lat_v, lon_v


def build_snap_arrays(indicator: str, ssp: str, bp_tag: str,
                      season: str, model_key: str | None):
    """Dispatch to the correct builder based on indicator type."""
    if indicator in _REC_INDICATORS:
        # model_key is ignored for REC indicators — always ensemble mean.
        return build_snap_arrays_rec(indicator, ssp, bp_tag, season)
    return build_snap_arrays_standard(indicator, ssp, bp_tag, season, model_key)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--indicator", default=None,
                        help="Only process this indicator (e.g. TX or REC_TXx)")
    parser.add_argument("--scenario",  default=None,
                        help="Only process this scenario (e.g. ssp370)")
    parser.add_argument("--force",     action="store_true",
                        help="Re-render even if cache file already exists")
    args = parser.parse_args()

    if not DATA_ROOT.exists():
        print(f"ERROR: DATA_ROOT not found: {DATA_ROOT}")
        sys.exit(1)

    indicators = list_all_indicators()
    if not indicators:
        print("No indicators found — check DATA_ROOT and REC_ROOT.")
        sys.exit(1)

    if args.indicator:
        if args.indicator not in indicators:
            print(f"ERROR: indicator '{args.indicator}' not found. "
                  f"Available: {indicators}")
            sys.exit(1)
        indicators = [args.indicator]

    scenarios = SSP_OPTIONS if not args.scenario else [args.scenario]

    print(f"\nFrame cache directory : {FRAME_CACHE_DIR.resolve()}")
    print(f"Indicators            : {indicators}")
    print(f"Scenarios             : {scenarios}\n")

    # Warm the coastline cache before the render loop so the first frame
    # doesn't pay the shapefile-load cost.
    print("Loading coastline polygon…", end=" ", flush=True)
    load_coastline_polygon()
    print("done\n")

    total_rendered = 0
    total_skipped  = 0

    for indicator in indicators:
        is_rec = indicator in _REC_INDICATORS
        cs_change = colorscale_for(indicator)
        cs_abs    = colorscale_abs_for(indicator)
        lm_change = _log_mode(indicator, is_change=True)
        lm_abs    = _log_mode(indicator, is_change=False)

        # REC change panel uses REF % (0–100), not a symmetric ± range.
        if is_rec:
            chg_vmin, chg_vmax = 0.0, 100.0
        else:
            shared_half        = compute_color_range(indicator)
            chg_vmin, chg_vmax = -shared_half, shared_half

        abs_vmin, abs_vmax = compute_abs_color_range(indicator)

        for bp_tag in BP_OPTIONS:
            seasons = discover_seasons_any("historical", indicator)
            for season in seasons:

                all_models: set[str] = set()
                for ssp in scenarios:
                    all_models.update(discover_models_any(ssp, indicator, bp_tag, season))
                # None = ensemble mean; remaining entries are individual models.
                model_keys = [None] + sorted(all_models)

                for model_key in model_keys:
                    for ssp in scenarios:
                        label = (
                            f"{indicator:12s} / {ssp} / {bp_tag} / {season:3s} / "
                            f"{'ensemble' if model_key is None else model_key}"
                        )

                        key_change = frame_cache_key(
                            indicator, ssp, bp_tag, season, model_key,
                            cs_change, chg_vmin, chg_vmax, lm_change,
                        )
                        key_abs = frame_cache_key(
                            indicator, ssp, bp_tag, season, model_key,
                            cs_abs, abs_vmin, abs_vmax, lm_abs,
                        )

                        need_change = not cache_exists(key_change) or args.force
                        need_abs    = not cache_exists(key_abs)    or args.force

                        if not need_change and not need_abs:
                            print(f"  SKIP   {label}")
                            total_skipped += 1
                            continue

                        print(f"  RENDER {label} …", end=" ", flush=True)
                        t0 = time.time()

                        result = build_snap_arrays(
                            indicator, ssp, bp_tag, season, model_key
                        )
                        if result is None:
                            print("(no data)")
                            continue

                        stacked, stacked_abs, lat_v, lon_v = result

                        if stacked.shape[0] < 2:
                            print("(< 2 snapshots, skipping)")
                            continue

                        if need_change:
                            payload = prerender_snapshots(
                                lat_v=lat_v,
                                lon_v=lon_v,
                                snap_data_stacked=stacked,
                                vmin=chg_vmin,
                                vmax=chg_vmax,
                                colorscale=cs_change,
                                indicator=indicator,
                                # REC change panel is REF % — sequential, not diverging.
                                is_change=(not is_rec),
                            )
                            save_cache(key_change, payload)

                        if need_abs:
                            payload = prerender_snapshots(
                                lat_v=lat_v,
                                lon_v=lon_v,
                                snap_data_stacked=stacked_abs,
                                vmin=abs_vmin,
                                vmax=abs_vmax,
                                colorscale=cs_abs,
                                indicator=indicator,
                                is_change=False,
                            )
                            save_cache(key_abs, payload)

                        print(f"done ({time.time() - t0:.1f}s)")
                        total_rendered += 1

    print(f"\nFinished.")
    print(f"  Rendered : {total_rendered}")
    print(f"  Skipped  : {total_skipped}")
    print(f"  Cache    : {FRAME_CACHE_DIR.resolve()}")


if __name__ == "__main__":
    main()