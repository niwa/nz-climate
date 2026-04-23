#!/usr/bin/env python3
# precompute_uncertainty.py (annotated for review)
#
# OVERVIEW
# --------
# Offline batch script — pre-computes model uncertainty bands, ensemble means,
# absolute values, and summary statistics for every indicator / scenario /
# baseline / season combination and stores them as pickle files in
# assets/uncertainty_cache/.
#
# The Streamlit app loads these caches at runtime via load_uncertainty_cache()
# in 2_NZ_Map.py. If a cache is missing, the app falls back to a slow live
# computation and shows a warning toast.
#
# Two indicator types are handled differently:
#   Standard (TX, PR, FD …) — single-variable nc files
#   REC (REC_TXx, …)        — dual-variable nc files (REF % + REC physical)
#
# The cache key (MD5 hash) must exactly match _uncertainty_cache_key() in
# 2_NZ_Map.py — any change to the key format in either file will cause cache
# misses.
#
# Usage:
#     python precompute_uncertainty.py
#     python precompute_uncertainty.py --indicator TX
#     python precompute_uncertainty.py --scenario  ssp370
#     python precompute_uncertainty.py --bp        bp1995-2014
#     python precompute_uncertainty.py --season    ANN
#     python precompute_uncertainty.py --force
#
# PRIVACY / SECURITY NOTES FOR REVIEWERS
# ----------------------------------------
# * DATA_ROOT and REC_ROOT are hardcoded to internal HPC paths below.
#   REVIEW FLAG: These should be externalised to environment variables
#   (NZMAP_DATA_ROOT, NZMAP_REC_ROOT) to match the pattern applied in
#   2_NZ_Map.py (Part 1) and to avoid HPC paths appearing in committed code.
#   Suggested fix:
#       import os
#       DATA_ROOT = Path(os.environ.get("NZMAP_DATA_ROOT", "<fallback>"))
#       REC_ROOT  = Path(os.environ.get("NZMAP_REC_ROOT",  "<fallback>"))
# * Cache files are pickled with protocol HIGHEST_PROTOCOL. As noted in
#   Part 2 of 2_NZ_Map.py, pickle is safe as long as only this script and
#   the app have write access to assets/uncertainty_cache/. If the deployment
#   environment allows untrusted file drops, switch to npz + JSON.
# * The "westphall" path component in REC_ROOT is a personal directory on
#   the HPC. Should be replaced with a project-level path (see above).
# * No credentials, tokens, API keys, or personal data are stored in the
#   cache payloads — only aggregated numeric arrays and metadata strings.
#
# LOGIC NOTES FOR REVIEWERS
# --------------------------
# * build_valid_mask() unions finite pixels across ALL models and periods.
#   A pixel is included if ANY model/period has a finite value there.
#   Consider switching to an intersection if sparser but more consistent
#   spatial coverage is preferred.
# * At the historical snapshot, the abs uncertainty bands are set to the
#   ensemble mean baseline (no spread), since all models share the same
#   historical baseline estimate. This is a design choice — cross-model
#   variation in the historical baseline is available but not shown.
# * The per-model model_change_vals / model_abs_vals dicts are populated
#   for all models, even when running the ensemble-mean view. They are only
#   used when the user selects a specific model in the sidebar, at which
#   point the app reads these dicts from the cache.

"""
precompute_uncertainty.py
--------------------------
Pre-computes model uncertainty bands, ensemble means, absolute values, and
summary statistics for every indicator / scenario / baseline / season combination.

Payload includes BOTH change and absolute uncertainty bands (n_snaps × n_pts):

  Change panel:  ens_vals, lo_vals, p5_vals, p25_vals, p75_vals, p95_vals, hi_vals
  Absolute panel: abs_ens_vals, abs_lo_vals, abs_p5_vals, abs_p25_vals,
                  abs_p75_vals, abs_p95_vals, abs_hi_vals

  Standard indicators: abs_* = baseline + change_*
  REC indicators:      abs_* = ensemble percentiles over REC physical values

Run once (or after new data is added):
    python precompute_uncertainty.py

Flags:
    --indicator REC_TXx   only process one indicator
    --scenario  ssp370    only process one scenario
    --bp        bp1995-2014
    --season    ANN
    --force               re-compute even if cache already exists
"""

import argparse
import hashlib
import pickle
import re
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import xarray as xr

# ── Constants ─────────────────────────────────────────────────────────────────
# REVIEW FLAG: Externalise to environment variables — see PRIVACY NOTES above.
import os
DATA_ROOT = Path(os.environ.get(
    "NZMAP_DATA_ROOT",
    "/esi/project/niwa03712/ML_Downscaled_CMIP6/NIWA-REMS_indicators/output_v3",
))
REC_ROOT = Path(os.environ.get(
    "NZMAP_REC_ROOT",
    "/esi/project/niwa03712/westphall/nz-climate/REC",
))

UNCERTAINTY_CACHE_DIR = Path("assets/uncertainty_cache")
UNCERTAINTY_CACHE_DIR.mkdir(parents=True, exist_ok=True)

SSP_OPTIONS = ["ssp126", "ssp245", "ssp370", "ssp585"]
BP_OPTIONS  = ["bp1995-2014", "bp1986-2005"]

SEASON_LABELS = {
    "ANN": "Annual",
    "DJF": "Summer (DJF)",
    "MAM": "Autumn (MAM)",
    "JJA": "Winter (JJA)",
    "SON": "Spring (SON)",
}
SEASON_ANN = "ANN"

_KELVIN_INDICATORS = {"TX", "TXx", "TN", "TNn"}
_REC_INDICATORS    = {"REC_TXx", "REC_Rx1day", "REC_Wx1day", "REC_TNn"}
_REC_KELVIN        = {"REC_TXx", "REC_TNn"}

_LOG_INDICATORS = {
    "PR", "RR1mm", "RR25mm", "Rx1day",
    "R99p", "R99pVAL", "R99pVALWet",
    "Wd10", "Wd25", "Wd99pVAL", "Wx1day",
    "sfcwind",
}
_SYMLOG_CHANGE_INDICATORS = {
    "PR", "RR1mm", "RR25mm", "Rx1day",
    "R99p", "R99pVAL", "R99pVALWet",
}


# ── Cache helpers ─────────────────────────────────────────────────────────────
# The cache key format MUST match _uncertainty_cache_key() in 2_NZ_Map.py.
# If you change the format here, update that function too or all cache lookups
# will miss.

def uncertainty_cache_key(indicator: str, ssp: str,
                           bp_tag: str, season: str) -> str:
    return hashlib.md5(f"{indicator}|{ssp}|{bp_tag}|{season}".encode()).hexdigest()

def cache_path(key: str) -> Path:
    return UNCERTAINTY_CACHE_DIR / f"{key}.pkl"

def cache_exists(key: str) -> bool:
    return cache_path(key).exists()

def save_cache(key: str, payload: dict) -> None:
    with open(cache_path(key), "wb") as f:
        pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)


# ── File discovery ────────────────────────────────────────────────────────────

def _indicator_root(indicator: str) -> Path:
    return REC_ROOT if indicator in _REC_INDICATORS else DATA_ROOT


def list_indicators(scenario: str) -> list[str]:
    results: set[str] = set()
    base = DATA_ROOT / scenario / "static_maps"
    if base.exists():
        results.update(d.name for d in base.iterdir() if d.is_dir())
    rec_base = REC_ROOT / scenario / "static_maps"
    if rec_base.exists():
        results.update(d.name for d in rec_base.iterdir() if d.is_dir())
    return sorted(results)


def discover_seasons(scenario: str, indicator: str) -> list[str]:
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


def discover_fp_tags(scenario: str, indicator: str,
                     bp_tag: str, season: str) -> list[str]:
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


def discover_models(scenario: str, indicator: str,
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


def find_nc_files(scenario: str, indicator: str,
                  fp_tag: str, bp_tag: str, season: str,
                  model_key: str | None = None) -> list[Path]:
    folder = _indicator_root(indicator) / scenario / "static_maps" / indicator
    if not folder.exists():
        return []
    if model_key:
        _model_pat = re.compile(r'_' + re.escape(model_key) + r'_[ri]\d+[ip]\d+[pf]\d+')
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

def _load_nc_array(f: Path, var_name: str | None = None,
                   ) -> tuple[np.ndarray | None, np.ndarray | None, np.ndarray | None]:
    """Load a single data variable from an indicator nc file."""
    try:
        ds = xr.open_dataset(f, engine="netcdf4", mask_and_scale=False)
        lat_names = [c for c in list(ds.coords) + list(ds.dims)
                     if c.lower() in ("lat", "latitude", "y", "rlat")]
        lon_names = [c for c in list(ds.coords) + list(ds.dims)
                     if c.lower() in ("lon", "longitude", "x", "rlon")]
        if not lat_names or not lon_names:
            ds.close(); return None, None, None
        lat_name, lon_name = lat_names[0], lon_names[0]

        if var_name and var_name in ds.data_vars:
            chosen = var_name
        else:
            data_vars = [v for v in ds.data_vars
                         if v.lower() not in ("lat", "lon", "time")]
            if not data_vars:
                ds.close(); return None, None, None
            chosen = data_vars[0]

        da = ds[chosen]
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
    """Load REF and REC variables from a dual-variable REC indicator nc file."""
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


def load_all_model_arrays(scenario: str, indicator: str,
                          fp_tag: str, bp_tag: str, season: str,
                          model_key: str | None = None,
                          ) -> tuple[list[np.ndarray], np.ndarray | None, np.ndarray | None]:
    """Load per-model arrays for a standard indicator, returning a list of 2D arrays."""
    files = find_nc_files(scenario, indicator, fp_tag, bp_tag, season, model_key)
    arrays, lats, lons = [], None, None
    for f in files:
        arr, la, lo = _load_nc_array(f)
        if arr is None:
            continue
        if lats is None:
            lats, lons = la, lo
        arrays.append(arr)
    return arrays, lats, lons


def load_all_model_dual(scenario: str, indicator: str,
                        fp_tag: str, bp_tag: str, season: str,
                        model_key: str | None = None,
                        ) -> tuple[list[np.ndarray], list[np.ndarray],
                                   np.ndarray | None, np.ndarray | None]:
    """Load per-model REF and REC arrays for a REC indicator."""
    files = find_nc_files(scenario, indicator, fp_tag, bp_tag, season, model_key)
    ref_arrays, rec_arrays = [], []
    lats = lons = None
    for f in files:
        ref_arr, rec_arr, la, lo = _load_nc_dual(f)
        if ref_arr is None and rec_arr is None:
            continue
        if lats is None:
            lats, lons = la, lo
        if ref_arr is not None:
            ref_arrays.append(ref_arr)
        if rec_arr is not None:
            rec_arrays.append(rec_arr)
    return ref_arrays, rec_arrays, lats, lons


# ── Helpers ───────────────────────────────────────────────────────────────────

def fp_centre_year(fp_tag: str) -> float:
    m = re.search(r"(\d{4})-(\d{4})", fp_tag)
    return (int(m.group(1)) + int(m.group(2))) / 2 if m else 0.0


def fp_label(fp_tag: str) -> str:
    m = re.search(r"(\d{4})-(\d{4})", fp_tag)
    return f"{m.group(1)}–{m.group(2)}" if m else fp_tag


def kelvin_to_celsius(arr: np.ndarray) -> np.ndarray:
    """Convert Kelvin to Celsius only if the data looks like it is in Kelvin."""
    finite = arr[np.isfinite(arr)]
    if len(finite) and np.nanmean(finite) > 200:
        return arr - 273.15
    return arr


def _summary(arr_2d: np.ndarray) -> dict:
    """Return mean/min/max of finite values in a 2D array."""
    finite = arr_2d[np.isfinite(arr_2d)]
    if len(finite) == 0:
        return {"mean": None, "min": None, "max": None}
    return {
        "mean": float(np.nanmean(finite)),
        "min":  float(np.nanmin(finite)),
        "max":  float(np.nanmax(finite)),
    }


def build_valid_mask(indicator: str, ssp: str, bp_tag: str,
                     season: str, common_shape: tuple) -> np.ndarray:
    """
    Union of finite-value masks across all models and periods.
    A pixel is included if any model/period has a finite value there.
    See LOGIC NOTES at the top for the union vs intersection trade-off.
    """
    n_pts = int(np.prod(common_shape))
    valid = np.zeros(n_pts, dtype=bool)
    hist_fps = discover_fp_tags("historical", indicator, bp_tag, season)
    hist_fp  = next((t for t in hist_fps if t == bp_tag), None)
    if hist_fp:
        arrays, _, _ = load_all_model_arrays(
            "historical", indicator, hist_fp, hist_fp, season)
        for a in arrays:
            valid |= np.isfinite(a.ravel())
    for fp in discover_fp_tags(ssp, indicator, bp_tag, season):
        arrays, _, _ = load_all_model_arrays(ssp, indicator, fp, bp_tag, season)
        for a in arrays:
            valid |= np.isfinite(a.ravel())
    return valid


# ── Core: standard indicators ─────────────────────────────────────────────────

def compute_standard(indicator: str, ssp: str,
                     bp_tag: str, season: str) -> dict | None:
    """
    Compute the full uncertainty cache payload for a standard indicator.

    Change bands are computed across the GCM axis for each snapshot.
    Absolute bands = historical baseline + change bands.
    Per-model values are stored separately for the click-chart sidebar.
    Returns None if no data is available for this combination.
    """
    ssp_fp_tags = discover_fp_tags(ssp, indicator, bp_tag, season)
    if not ssp_fp_tags:
        return None

    common_shape = ref_lats = ref_lons = None
    for fp in ssp_fp_tags:
        arrays, lats, lons = load_all_model_arrays(ssp, indicator, fp, bp_tag, season)
        if arrays:
            common_shape = arrays[0].shape
            ref_lats, ref_lons = lats, lons
            break
    if common_shape is None:
        return None

    valid_mask = build_valid_mask(indicator, ssp, bp_tag, season, common_shape)
    if not np.any(valid_mask):
        return None

    if ref_lats.ndim == 2:
        lat_f, lon_f = ref_lats.ravel(), ref_lons.ravel()
    else:
        lg, ng = np.meshgrid(ref_lats, ref_lons, indexing="ij")
        lat_f, lon_f = lg.ravel(), ng.ravel()

    lat_v = lat_f[valid_mask]
    lon_v = lon_f[valid_mask]
    n_pts = int(valid_mask.sum())

    hist_fps = discover_fp_tags("historical", indicator, bp_tag, season)
    hist_fp  = next((t for t in hist_fps if t == bp_tag), None)

    snapshots: list[tuple] = []
    if hist_fp:
        yr = fp_centre_year(hist_fp)
        snapshots.append(("Historical", fp_label(hist_fp), yr,
                          "historical", hist_fp, hist_fp))
    for fp in ssp_fp_tags:
        yr = fp_centre_year(fp)
        snapshots.append((fp_label(fp), fp_label(fp), yr, ssp, fp, bp_tag))
    snapshots.sort(key=lambda x: x[2])
    n_snaps = len(snapshots)

    all_models = discover_models(ssp, indicator, bp_tag, season)

    _nan             = np.full((n_snaps, n_pts), np.nan)
    ens_vals         = _nan.copy()
    lo_vals          = _nan.copy()
    p5_vals          = _nan.copy()
    p25_vals         = _nan.copy()
    p75_vals         = _nan.copy()
    p95_vals         = _nan.copy()
    hi_vals          = _nan.copy()
    abs_ens_vals     = _nan.copy()
    abs_lo_vals      = _nan.copy()
    abs_p5_vals      = _nan.copy()
    abs_p25_vals     = _nan.copy()
    abs_p75_vals     = _nan.copy()
    abs_p95_vals     = _nan.copy()
    abs_hi_vals      = _nan.copy()

    model_change_vals: dict[str, np.ndarray] = {
        mk: np.full((n_snaps, n_pts), np.nan) for mk in all_models
    }
    model_abs_vals: dict[str, np.ndarray] = {
        mk: np.full((n_snaps, n_pts), np.nan) for mk in all_models
    }

    n_models_list: list[int] = []
    summary_stats:  list[dict] = []
    zeros_v = np.zeros(n_pts)

    # Load and K→°C convert the historical baseline.
    baseline_2d: np.ndarray | None = None
    baseline_v:  np.ndarray | None = None
    if hist_fp:
        bl_arrays, _, _ = load_all_model_arrays(
            "historical", indicator, hist_fp, hist_fp, season)
        if bl_arrays:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                baseline_2d = np.nanmean(np.stack(bl_arrays, 0), 0)
            if indicator in _KELVIN_INDICATORS:
                baseline_2d = kelvin_to_celsius(baseline_2d)
            baseline_v = baseline_2d.ravel()[valid_mask].astype(float)

    for si, (label, fp_range, yr, scen, fp, bp) in enumerate(snapshots):
        if scen == "historical":
            # Historical = zero change; abs = the baseline itself (no spread).
            ens_vals[si]     = zeros_v
            lo_vals[si]      = zeros_v
            p5_vals[si]      = zeros_v
            p25_vals[si]     = zeros_v
            p75_vals[si]     = zeros_v
            p95_vals[si]     = zeros_v
            hi_vals[si]      = zeros_v
            abs_ens_vals[si] = baseline_v if baseline_v is not None else zeros_v
            abs_lo_vals[si]  = abs_ens_vals[si]
            abs_p5_vals[si]  = abs_ens_vals[si]
            abs_p25_vals[si] = abs_ens_vals[si]
            abs_p75_vals[si] = abs_ens_vals[si]
            abs_p95_vals[si] = abs_ens_vals[si]
            abs_hi_vals[si]  = abs_ens_vals[si]
            for mk in all_models:
                model_change_vals[mk][si] = zeros_v
                model_abs_vals[mk][si]    = abs_ens_vals[si]
            n_models_list.append(0)
            if baseline_2d is not None:
                s = _summary(baseline_2d)
            else:
                s = {"mean": None, "min": None, "max": None}
            summary_stats.append({
                "mean_change": 0.0,  "min_change": 0.0,  "max_change": 0.0,
                "mean_abs":    s["mean"], "min_abs": s["min"], "max_abs": s["max"],
                "n_models":    0,
            })
            continue

        arrays, _, _ = load_all_model_arrays(scen, indicator, fp, bp, season)
        n_m = len(arrays)
        n_models_list.append(n_m)

        if arrays:
            stk = np.stack([a.ravel()[valid_mask].astype(float)
                            for a in arrays], axis=0)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                ens_vals[si] = np.nanmean(stk, axis=0)
                lo_vals[si]  = np.nanmin(stk,  axis=0)
                hi_vals[si]  = np.nanmax(stk,  axis=0)
                if n_m >= 2:
                    p5_vals[si]  = np.nanpercentile(stk,  5, axis=0)
                    p25_vals[si] = np.nanpercentile(stk, 25, axis=0)
                    p75_vals[si] = np.nanpercentile(stk, 75, axis=0)
                    p95_vals[si] = np.nanpercentile(stk, 95, axis=0)
                else:
                    # Single model: all percentile bands collapse to that model's value.
                    p5_vals[si] = p25_vals[si] = p75_vals[si] = p95_vals[si] = stk[0]

            if baseline_v is not None:
                abs_ens_vals[si]  = baseline_v + ens_vals[si]
                abs_lo_vals[si]   = baseline_v + lo_vals[si]
                abs_hi_vals[si]   = baseline_v + hi_vals[si]
                abs_p5_vals[si]   = baseline_v + p5_vals[si]
                abs_p25_vals[si]  = baseline_v + p25_vals[si]
                abs_p75_vals[si]  = baseline_v + p75_vals[si]
                abs_p95_vals[si]  = baseline_v + p95_vals[si]

            ens_2d = np.full(common_shape, np.nan)
            ens_2d.ravel()[valid_mask] = ens_vals[si]
            s_ch = _summary(ens_2d)

            if baseline_2d is not None:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", RuntimeWarning)
                    abs_2d = baseline_2d + np.nanmean(np.stack(arrays, 0), 0)
                s_ab = _summary(abs_2d)
            else:
                s_ab = {"mean": None, "min": None, "max": None}

            summary_stats.append({
                "mean_change": s_ch["mean"], "min_change": s_ch["min"],
                "max_change":  s_ch["max"],
                "mean_abs":    s_ab["mean"], "min_abs":    s_ab["min"],
                "max_abs":     s_ab["max"],
                "n_models":    n_m,
            })
        else:
            summary_stats.append({
                "mean_change": None, "min_change": None, "max_change": None,
                "mean_abs":    None, "min_abs":    None, "max_abs":    None,
                "n_models":    0,
            })

        for mk in all_models:
            mk_arrays, _, _ = load_all_model_arrays(
                scen, indicator, fp, bp, season, model_key=mk)
            if mk_arrays:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", RuntimeWarning)
                    mk_mean = np.nanmean(
                        np.stack([a.ravel()[valid_mask].astype(float)
                                  for a in mk_arrays], axis=0), axis=0)
                model_change_vals[mk][si] = mk_mean
                if baseline_v is not None:
                    model_abs_vals[mk][si] = baseline_v + mk_mean

    # ── Chart y-range: change panel ──────────────────────────────────────────
    all_lo_f = [lo_vals[si][np.isfinite(lo_vals[si])]
                for si in range(n_snaps) if np.any(np.isfinite(lo_vals[si]))]
    all_hi_f = [hi_vals[si][np.isfinite(hi_vals[si])]
                for si in range(n_snaps) if np.any(np.isfinite(hi_vals[si]))]
    if all_lo_f and all_hi_f:
        chart_ymin = float(np.nanpercentile(np.concatenate(all_lo_f), 2))
        chart_ymax = float(np.nanpercentile(np.concatenate(all_hi_f), 98))
        _pad = (chart_ymax - chart_ymin) * 0.10
        chart_ymin -= _pad
        chart_ymax += _pad
    else:
        chart_ymin, chart_ymax = -1.0, 1.0

    # ── Chart y-range: absolute panel ────────────────────────────────────────
    abs_lo_f = [abs_lo_vals[si][np.isfinite(abs_lo_vals[si])]
                for si in range(n_snaps) if np.any(np.isfinite(abs_lo_vals[si]))]
    abs_hi_f = [abs_hi_vals[si][np.isfinite(abs_hi_vals[si])]
                for si in range(n_snaps) if np.any(np.isfinite(abs_hi_vals[si]))]
    if abs_lo_f and abs_hi_f:
        abs_chart_ymin = float(np.nanpercentile(np.concatenate(abs_lo_f), 2))
        abs_chart_ymax = float(np.nanpercentile(np.concatenate(abs_hi_f), 98))
        _apad = (abs_chart_ymax - abs_chart_ymin) * 0.10
        abs_chart_ymin -= _apad
        abs_chart_ymax += _apad
    else:
        abs_chart_ymin, abs_chart_ymax = chart_ymin, chart_ymax

    return dict(
        snap_labels        = [s[0] for s in snapshots],
        snap_years         = [s[2] for s in snapshots],
        snap_fp_ranges     = [s[1] for s in snapshots],
        lat_v              = lat_v,
        lon_v              = lon_v,
        ens_vals           = ens_vals,
        lo_vals            = lo_vals,
        p5_vals            = p5_vals,
        p25_vals           = p25_vals,
        p75_vals           = p75_vals,
        p95_vals           = p95_vals,
        hi_vals            = hi_vals,
        abs_ens_vals       = abs_ens_vals,
        abs_lo_vals        = abs_lo_vals,
        abs_p5_vals        = abs_p5_vals,
        abs_p25_vals       = abs_p25_vals,
        abs_p75_vals       = abs_p75_vals,
        abs_p95_vals       = abs_p95_vals,
        abs_hi_vals        = abs_hi_vals,
        abs_baseline_v     = baseline_v,
        model_change_vals  = model_change_vals,
        model_abs_vals     = model_abs_vals,
        n_models           = n_models_list,
        summary_stats      = summary_stats,
        chart_ymin         = chart_ymin,
        chart_ymax         = chart_ymax,
        abs_chart_ymin     = abs_chart_ymin,
        abs_chart_ymax     = abs_chart_ymax,
    )


# ── Core: REC indicators ──────────────────────────────────────────────────────

def compute_rec(indicator: str, ssp: str,
                bp_tag: str, season: str) -> dict | None:
    """
    Compute the uncertainty cache payload for a REC indicator.

    Change bands → percentiles over REF % values (model spread).
    Absolute bands → percentiles over REC physical values (model spread).
    Returns None if no data is available for this combination.
    """
    ssp_fp_tags = discover_fp_tags(ssp, indicator, bp_tag, season)
    if not ssp_fp_tags:
        return None

    common_shape = ref_lats = ref_lons = None
    for fp in ssp_fp_tags:
        ref_arrays, _, lats, lons = load_all_model_dual(
            ssp, indicator, fp, bp_tag, season)
        if ref_arrays:
            common_shape = ref_arrays[0].shape
            ref_lats, ref_lons = lats, lons
            break
    if common_shape is None:
        return None

    valid_mask = build_valid_mask(indicator, ssp, bp_tag, season, common_shape)
    if not np.any(valid_mask):
        return None

    if ref_lats.ndim == 2:
        lat_f, lon_f = ref_lats.ravel(), ref_lons.ravel()
    else:
        lg, ng = np.meshgrid(ref_lats, ref_lons, indexing="ij")
        lat_f, lon_f = lg.ravel(), ng.ravel()

    lat_v = lat_f[valid_mask]
    lon_v = lon_f[valid_mask]
    n_pts = int(valid_mask.sum())

    hist_fps = discover_fp_tags("historical", indicator, bp_tag, season)
    hist_fp  = next((t for t in hist_fps if t == bp_tag), None)

    snapshots: list[tuple] = []
    if hist_fp:
        yr = fp_centre_year(hist_fp)
        snapshots.append(("Historical", fp_label(hist_fp), yr,
                          "historical", hist_fp, hist_fp))
    for fp in ssp_fp_tags:
        yr = fp_centre_year(fp)
        snapshots.append((fp_label(fp), fp_label(fp), yr, ssp, fp, bp_tag))
    snapshots.sort(key=lambda x: x[2])
    n_snaps = len(snapshots)

    all_models = discover_models(ssp, indicator, bp_tag, season)

    _nan         = np.full((n_snaps, n_pts), np.nan)
    ens_vals     = _nan.copy()
    lo_vals      = _nan.copy()
    p5_vals      = _nan.copy()
    p25_vals     = _nan.copy()
    p75_vals     = _nan.copy()
    p95_vals     = _nan.copy()
    hi_vals      = _nan.copy()
    abs_ens_vals = _nan.copy()
    abs_lo_vals  = _nan.copy()
    abs_p5_vals  = _nan.copy()
    abs_p25_vals = _nan.copy()
    abs_p75_vals = _nan.copy()
    abs_p95_vals = _nan.copy()
    abs_hi_vals  = _nan.copy()

    model_change_vals: dict[str, np.ndarray] = {
        mk: np.full((n_snaps, n_pts), np.nan) for mk in all_models
    }
    model_abs_vals: dict[str, np.ndarray] = {
        mk: np.full((n_snaps, n_pts), np.nan) for mk in all_models
    }

    n_models_list: list[int] = []
    summary_stats: list[dict] = []
    needs_k_conv = indicator in _REC_KELVIN

    for si, (label, fp_range, yr, scen, fp, bp) in enumerate(snapshots):
        ref_arrays, rec_arrays, _, _ = load_all_model_dual(
            scen, indicator, fp, bp, season)

        n_m = len(ref_arrays)
        n_models_list.append(n_m)

        if not ref_arrays and not rec_arrays:
            summary_stats.append({
                "mean_change": None, "min_change": None, "max_change": None,
                "mean_abs":    None, "min_abs":    None, "max_abs":    None,
                "n_models":    0,
            })
            continue

        # ── REF % — change panel uncertainty bands ────────────────────────
        if ref_arrays:
            ref_stk = np.stack([a.ravel()[valid_mask].astype(float)
                                for a in ref_arrays], axis=0)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                ens_vals[si] = np.nanmean(ref_stk, axis=0)
                lo_vals[si]  = np.nanmin(ref_stk,  axis=0)
                hi_vals[si]  = np.nanmax(ref_stk,  axis=0)
                if n_m >= 2:
                    p5_vals[si]  = np.nanpercentile(ref_stk,  5, axis=0)
                    p25_vals[si] = np.nanpercentile(ref_stk, 25, axis=0)
                    p75_vals[si] = np.nanpercentile(ref_stk, 75, axis=0)
                    p95_vals[si] = np.nanpercentile(ref_stk, 95, axis=0)
                else:
                    p5_vals[si] = p25_vals[si] = p75_vals[si] = p95_vals[si] = ref_stk[0]

        # ── REC physical — absolute panel uncertainty bands ───────────────
        if rec_arrays:
            rec_stk = np.stack([a.ravel()[valid_mask].astype(float)
                                for a in rec_arrays], axis=0)
            if needs_k_conv:
                rec_stk = np.where(rec_stk > 200, rec_stk - 273.15, rec_stk)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                abs_ens_vals[si] = np.nanmean(rec_stk, axis=0)
                abs_lo_vals[si]  = np.nanmin(rec_stk,  axis=0)
                abs_hi_vals[si]  = np.nanmax(rec_stk,  axis=0)
                n_rec = rec_stk.shape[0]
                if n_rec >= 2:
                    abs_p5_vals[si]  = np.nanpercentile(rec_stk,  5, axis=0)
                    abs_p25_vals[si] = np.nanpercentile(rec_stk, 25, axis=0)
                    abs_p75_vals[si] = np.nanpercentile(rec_stk, 75, axis=0)
                    abs_p95_vals[si] = np.nanpercentile(rec_stk, 95, axis=0)
                else:
                    abs_p5_vals[si] = abs_p25_vals[si] = \
                    abs_p75_vals[si] = abs_p95_vals[si] = rec_stk[0]

        # ── Summary statistics ────────────────────────────────────────────
        ref_2d = np.full(common_shape, np.nan)
        if ref_arrays:
            ref_2d.ravel()[valid_mask] = ens_vals[si]
        s_ref = _summary(ref_2d)

        rec_2d = np.full(common_shape, np.nan)
        if rec_arrays:
            rec_mean = np.nanmean(np.stack(rec_arrays, 0), 0)
            if needs_k_conv:
                rec_mean = np.where(rec_mean > 200, rec_mean - 273.15, rec_mean)
            rec_2d = rec_mean
        s_rec = _summary(rec_2d)

        summary_stats.append({
            "mean_change": s_ref["mean"], "min_change": s_ref["min"],
            "max_change":  s_ref["max"],
            "mean_abs":    s_rec["mean"], "min_abs":    s_rec["min"],
            "max_abs":     s_rec["max"],
            "n_models":    n_m,
        })

        # ── Per-model values for click-chart ──────────────────────────────
        for mk in all_models:
            mk_refs, mk_recs, _, _ = load_all_model_dual(
                scen, indicator, fp, bp, season, model_key=mk)
            if mk_refs:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", RuntimeWarning)
                    mk_ref_mean = np.nanmean(
                        np.stack([a.ravel()[valid_mask].astype(float)
                                  for a in mk_refs], axis=0), axis=0)
                model_change_vals[mk][si] = mk_ref_mean
            if mk_recs:
                mk_rec_stk = np.stack([a.ravel()[valid_mask].astype(float)
                                       for a in mk_recs], axis=0)
                if needs_k_conv:
                    mk_rec_stk = np.where(mk_rec_stk > 200,
                                          mk_rec_stk - 273.15, mk_rec_stk)
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", RuntimeWarning)
                    model_abs_vals[mk][si] = np.nanmean(mk_rec_stk, axis=0)

    # ── Chart y-ranges ────────────────────────────────────────────────────────
    all_lo_f = [lo_vals[si][np.isfinite(lo_vals[si])]
                for si in range(n_snaps) if np.any(np.isfinite(lo_vals[si]))]
    all_hi_f = [hi_vals[si][np.isfinite(hi_vals[si])]
                for si in range(n_snaps) if np.any(np.isfinite(hi_vals[si]))]
    if all_lo_f and all_hi_f:
        chart_ymin = float(np.nanpercentile(np.concatenate(all_lo_f), 2))
        chart_ymax = float(np.nanpercentile(np.concatenate(all_hi_f), 98))
        _pad = (chart_ymax - chart_ymin) * 0.10
        chart_ymin -= _pad
        chart_ymax += _pad
    else:
        chart_ymin, chart_ymax = 0.0, 100.0

    abs_lo_f = [abs_lo_vals[si][np.isfinite(abs_lo_vals[si])]
                for si in range(n_snaps) if np.any(np.isfinite(abs_lo_vals[si]))]
    abs_hi_f = [abs_hi_vals[si][np.isfinite(abs_hi_vals[si])]
                for si in range(n_snaps) if np.any(np.isfinite(abs_hi_vals[si]))]
    if abs_lo_f and abs_hi_f:
        abs_chart_ymin = float(np.nanpercentile(np.concatenate(abs_lo_f), 2))
        abs_chart_ymax = float(np.nanpercentile(np.concatenate(abs_hi_f), 98))
        _apad = (abs_chart_ymax - abs_chart_ymin) * 0.10
        abs_chart_ymin -= _apad
        abs_chart_ymax += _apad
    else:
        abs_chart_ymin, abs_chart_ymax = 0.0, 100.0

    return dict(
        snap_labels        = [s[0] for s in snapshots],
        snap_years         = [s[2] for s in snapshots],
        snap_fp_ranges     = [s[1] for s in snapshots],
        lat_v              = lat_v,
        lon_v              = lon_v,
        ens_vals           = ens_vals,
        lo_vals            = lo_vals,
        p5_vals            = p5_vals,
        p25_vals           = p25_vals,
        p75_vals           = p75_vals,
        p95_vals           = p95_vals,
        hi_vals            = hi_vals,
        abs_ens_vals       = abs_ens_vals,
        abs_lo_vals        = abs_lo_vals,
        abs_p5_vals        = abs_p5_vals,
        abs_p25_vals       = abs_p25_vals,
        abs_p75_vals       = abs_p75_vals,
        abs_p95_vals       = abs_p95_vals,
        abs_hi_vals        = abs_hi_vals,
        abs_baseline_v     = None,
        model_change_vals  = model_change_vals,
        model_abs_vals     = model_abs_vals,
        n_models           = n_models_list,
        summary_stats      = summary_stats,
        chart_ymin         = chart_ymin,
        chart_ymax         = chart_ymax,
        abs_chart_ymin     = abs_chart_ymin,
        abs_chart_ymax     = abs_chart_ymax,
    )


# ── Dispatcher ────────────────────────────────────────────────────────────────

def compute_uncertainty_for(indicator: str, ssp: str,
                             bp_tag: str, season: str) -> dict | None:
    """Route to the correct compute function based on indicator type."""
    if indicator in _REC_INDICATORS:
        return compute_rec(indicator, ssp, bp_tag, season)
    return compute_standard(indicator, ssp, bp_tag, season)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--indicator", default=None)
    parser.add_argument("--scenario",  default=None)
    parser.add_argument("--bp",        default=None)
    parser.add_argument("--season",    default=None)
    parser.add_argument("--force",     action="store_true")
    args = parser.parse_args()

    if not DATA_ROOT.exists():
        print(f"ERROR: DATA_ROOT not found: {DATA_ROOT}")
        sys.exit(1)

    indicators = list_indicators("historical")
    if args.indicator:
        if args.indicator not in indicators:
            print(f"ERROR: indicator '{args.indicator}' not found")
            sys.exit(1)
        indicators = [args.indicator]

    scenarios = SSP_OPTIONS if not args.scenario else [args.scenario]
    bp_list   = BP_OPTIONS  if not args.bp       else [args.bp]

    print(f"\nCache dir  : {UNCERTAINTY_CACHE_DIR.resolve()}")
    print(f"Indicators : {indicators}")
    print(f"Scenarios  : {scenarios}")
    print(f"Baselines  : {bp_list}\n")

    total_computed = total_skipped = total_nodata = 0

    for indicator in indicators:
        is_rec = indicator in _REC_INDICATORS
        for bp_tag in bp_list:
            seasons = discover_seasons("historical", indicator)
            if args.season:
                seasons = [args.season] if args.season in seasons else []
                if not seasons:
                    print(f"  WARNING: season '{args.season}' not available for {indicator}")
                    continue

            for season in seasons:
                for ssp in scenarios:
                    label = f"{indicator} / {ssp} / {bp_tag} / {season}"
                    key   = uncertainty_cache_key(indicator, ssp, bp_tag, season)

                    if cache_exists(key) and not args.force:
                        print(f"  SKIP    {label}")
                        total_skipped += 1
                        continue

                    kind = "REC" if is_rec else "standard"
                    print(f"  COMPUTE [{kind}] {label} ...", end="", flush=True)
                    t0 = time.time()

                    payload = compute_uncertainty_for(indicator, ssp, bp_tag, season)

                    if payload is None:
                        print(" (no data)")
                        total_nodata += 1
                        continue

                    save_cache(key, payload)
                    elapsed = time.time() - t0
                    print(f" done ({elapsed:.1f}s) — "
                          f"{len(payload['snap_labels'])} snaps, "
                          f"{payload['lat_v'].shape[0]} pts, "
                          f"{len(payload['model_change_vals'])} models")
                    total_computed += 1

    print(f"\nFinished.  Computed={total_computed}  "
          f"Skipped={total_skipped}  No-data={total_nodata}")
    print(f"Cache: {UNCERTAINTY_CACHE_DIR.resolve()}")


if __name__ == "__main__":
    main()