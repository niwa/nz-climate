#!/usr/bin/env python3
# precompute_records.py (annotated for review)
#
# OVERVIEW
# --------
# Offline computation script — generates REC indicator nc files from raw
# daily NIWA-REMS CCAM downscaled data. This is the only script in the
# project that reads raw daily data; all other scripts and the app itself
# consume the processed indicator nc files it produces.
#
# Four output indicators:
#   REC_TXx    — annual max Tmax record chance (warm record)
#   REC_Rx1day — annual max precipitation record chance (wet record)
#   REC_Wx1day — annual max wind speed record chance (wind record)
#   REC_TNn    — annual min Tmin record chance (cold record)
#
# Each output nc file contains two variables:
#   REF (%) — chance per year of breaking the historical record
#   REC      — running extreme record in physical units
#
# PRIVACY / SECURITY NOTES FOR REVIEWERS
# ----------------------------------------
# * DAILY_ROOT and OUT_ROOT are hardcoded to internal HPC paths below.
#   REVIEW FLAG: These should be externalised to environment variables
#   (NZMAP_DAILY_ROOT, NZMAP_REC_ROOT) for the same reasons as the other
#   scripts — portability, security, and ease of deployment configuration.
#   Suggested fix:
#       import os
#       DAILY_ROOT = Path(os.environ.get("NZMAP_DAILY_ROOT", "<fallback>"))
#       OUT_ROOT   = Path(os.environ.get("NZMAP_REC_ROOT",   "<fallback>"))
# * No credentials, tokens, API keys, or personal data are present.
# * Output nc files contain only gridded climate statistics — no personal
#   or organisational metadata beyond what xarray writes as standard CF
#   conventions (long_name, units, _FillValue).
# * The "westphall" path component in OUT_ROOT is a username/directory on
#   the HPC system. This should be replaced with a project-level path once
#   the output location is finalised (see env var note above).
#
# ALGORITHM NOTES FOR REVIEWERS
# --------------------------------
# * THRESHOLD_RANK = 1: the threshold used is the historical maximum (or
#   minimum for cold records). A future enhancement could use rank 3 as
#   noted in the comment referencing "Sigid et al. 2026" — the current
#   code uses rank 1 (the absolute historical extreme).
# * _REC_UNIT_SCALE is defined but empty {}. The PR conversion (×86400
#   kg m⁻² s⁻¹ → mm/day) is described in the module docstring but the
#   scale is never applied because _REC_UNIT_SCALE["REC_Rx1day"] is never
#   set. REVIEW FLAG: Either populate _REC_UNIT_SCALE or apply the conversion
#   directly in write_nc_dual(). Currently REC_Rx1day REC values are written
#   in raw kg m⁻² s⁻¹ units despite the docstring claiming mm/day.
# * Running records are maintained in raw units throughout the future loop
#   and converted only at write time via rec_scale. The comment
#   "CRITICAL: iterate in chronological order" is important — FUTURE_PERIODS
#   is a dict whose iteration order is insertion order (Python 3.7+), which
#   matches the chronological order defined. Safe, but worth documenting
#   explicitly.
# * The stationary_pct() function uses (THRESHOLD_RANK - 1) / n_hist, which
#   evaluates to 0.0 when THRESHOLD_RANK = 1. That means the historical base
#   file records 0% record chance (no year has previously beaten the max).
#   This is statistically correct for rank-1 (the max itself cannot exceed
#   itself) but differs from the module docstring which describes "rank-3
#   hist threshold". The docstring and constant are inconsistent.

"""
precompute_records.py
---------------------
Computes Record Exceedance Frequency (REF) indicators from daily NIWA-REMS
CCAM data and writes them as standard indicator nc files into the existing
static_maps output tree.

Four output indicators (each writes a single nc file with TWO variables):

  REC_TXx      — annual max tasmax   exceeds rank-3 hist threshold  (warm record)
  REC_Rx1day   — annual max pr       exceeds rank-3 hist threshold  (wet record)
  REC_Wx1day   — annual max sfcwindmax exceeds rank-3 hist threshold (wind record)
  REC_TNn      — annual min tasmin   falls below rank-3 hist threshold (cold record)

Each nc file contains:
  REF  (%)          — chance per year of breaking the historical record
                      stored directly as percent (0–100), not a departure
  REC  (phys units) — running extreme record in physical units
                      TXx/TNn: K (converted to °C by app)
                      Rx1day:  mm/day  (converted from kg m-2 s-1 × 86400)
                      Wx1day:  m s-1
                      warm/wet/wind: running maximum (non-decreasing)
                      cold:          running minimum (non-increasing)

IMPORTANT: The REF threshold comparison is always done in raw units.
           Unit conversion (×86400 for pr) is applied only when writing
           the REC variable to the nc file. This keeps the REF % correct.

Naming convention matches existing indicator files exactly:
  change: {ind}_{ssp}_{model}_{ensemble}_change_{fp}_{bp}_ANN_NZ12km.nc
  base:   {ind}_historical_{model}_{ensemble}_base_{bp}_{bp}_ANN_NZ12km.nc

Usage:
  python precompute_records.py                        # all
  python precompute_records.py --indicator REC_TXx    # one indicator
  python precompute_records.py --scenario  ssp370     # one scenario
  python precompute_records.py --force                # overwrite existing
"""

import argparse
import re
import warnings
from pathlib import Path

import numpy as np
import xarray as xr

# ── Paths ─────────────────────────────────────────────────────────────────────
# REVIEW FLAG: Externalise to environment variables — see PRIVACY NOTES above.
# The "westphall" component in OUT_ROOT is a personal directory on the HPC.
import os
DAILY_ROOT = Path(os.environ.get(
    "NZMAP_DAILY_ROOT",
    "/esi/project/niwa03712/ML_Downscaled_CMIP6/NIWA-REMS_CCAM_public",
))
OUT_ROOT = Path(os.environ.get(
    "NZMAP_REC_ROOT",
    "/esi/project/niwa03712/westphall/nz-climate/REC",
))

# ── Indicator configuration ───────────────────────────────────────────────────
# Maps indicator name → (var_dir, var_hint, phys_units, annual_stat, direction)
#   annual_stat : "max" or "min"
#   direction   : "above" (warm/wet/wind records) or "below" (cold records)
#
# NOTE: REC_Rx1day raw data is in kg m-2 s-1. The REC variable is described
#       as mm/day in the module docstring but the actual conversion is NOT
#       applied — see ALGORITHM NOTES above.
RECORD_INDICATORS = {
    "REC_TXx":    ("tasmax",     "tasmax",     "K",      "max", "above"),
    "REC_Rx1day": ("pr",         "pr",         "mm/day", "max", "above"),
    "REC_Wx1day": ("sfcwindmax", "sfcwindmax", "m s-1",  "max", "above"),
    "REC_TNn":    ("tasmin",     "tasmin",     "K",      "min", "below"),
}

# Unit conversion applied to REC variable before writing. Currently empty —
# see ALGORITHM NOTES for the REC_Rx1day issue.
_PR_SCALE      = 86400.0   # kg m-2 s-1 → mm/day (defined but not wired up)
_REC_UNIT_SCALE = {}        # REVIEW FLAG: populate or remove _PR_SCALE

SCENARIOS = {
    "historical": "CMIP",
    "ssp126":     "ScenarioMIP",
    "ssp245":     "ScenarioMIP",
    "ssp370":     "ScenarioMIP",
    "ssp585":     "ScenarioMIP",
}

BASELINES = {
    "bp1995-2014": (1995, 2014),
    "bp1986-2005": (1986, 2005),
}

# CRITICAL: Must remain in chronological order — the running record accumulates
# across periods. Python 3.7+ dict preserves insertion order, so this is safe
# as long as entries are not reordered.
FUTURE_PERIODS = {
    "fp2021-2040": (2021, 2040),
    "fp2041-2060": (2041, 2060),
    "fp2061-2080": (2061, 2080),
    "fp2081-2100": (2081, 2100),
}

# REVIEW FLAG: Module docstring says "rank-3" but THRESHOLD_RANK = 1.
# rank-1 = the absolute historical maximum (no year can exceed itself),
# giving stationary_pct = 0.0%. Decide which is intended and update the
# docstring, constant, and stationary_pct() formula consistently.
THRESHOLD_RANK = 1


# ── File discovery ────────────────────────────────────────────────────────────

def find_daily_files(scenario: str, var_dir: str) -> list[Path]:
    """Return all nc files for a given scenario + variable directory."""
    base  = DAILY_ROOT / scenario / "daily" / var_dir
    files = []
    if base.exists():
        files.extend(base.glob("*.nc"))
    # Also check versioned subdirectory (e.g. pr_v280125).
    for sub in (DAILY_ROOT / scenario / "daily").glob(f"{var_dir}_v*"):
        if sub.is_dir():
            files.extend(sub.glob("*.nc"))
    return sorted(set(files))


def parse_daily_filename(path: Path) -> dict | None:
    """
    Parse the standard daily filename:
      {var}_{MIP}_{institution}_{model}_{scenario}_{ensemble}_day_...nc
    Returns dict with model/scenario/ensemble keys, or None if parsing fails.
    """
    parts = path.stem.split("_")
    if len(parts) < 7 or parts[6] != "day":
        return None
    if not re.match(r"r\d+i\d+p\d+f\d+", parts[5]):
        return None
    return {"model": parts[3], "scenario": parts[4], "ensemble": parts[5]}


def group_daily_files(scenario: str, var_dir: str) -> dict[tuple, Path]:
    """Return mapping of (model, ensemble) → Path for a given scenario."""
    result = {}
    for f in find_daily_files(scenario, var_dir):
        info = parse_daily_filename(f)
        if info is None or info["scenario"] != scenario:
            continue
        key = (info["model"], info["ensemble"])
        if key not in result:
            result[key] = f
    return result


# ── Data loading ──────────────────────────────────────────────────────────────

def load_annual_stat(nc_path: Path, var_hint: str,
                     year_start: int, year_end: int,
                     stat: str) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """
    Open a daily nc file, select years [year_start, year_end], and compute
    the annual maximum (stat="max") or minimum (stat="min") for each year.

    Returns (annual_stat_stack, lats, lons) with shape (n_years, lat, lon).
    Returns None if the data is unavailable or time selection fails.
    Values are in raw units — no scaling is applied here.
    """
    try:
        ds = xr.open_dataset(nc_path, engine="netcdf4", mask_and_scale=True)
    except Exception as e:
        print(f"\n    Cannot open {nc_path.name}: {e}")
        return None

    data_vars = [v for v in ds.data_vars
                 if var_hint.lower() in v.lower()
                 and v.lower() not in ("lat", "lon", "time")]
    if not data_vars:
        data_vars = [v for v in ds.data_vars
                     if v.lower() not in ("lat", "lon", "time", "lev")]
    if not data_vars:
        ds.close()
        return None

    lat_names = [c for c in list(ds.coords) + list(ds.dims)
                 if c.lower() in ("lat", "latitude", "y", "rlat")]
    lon_names = [c for c in list(ds.coords) + list(ds.dims)
                 if c.lower() in ("lon", "longitude", "x", "rlon")]
    if not lat_names or not lon_names:
        ds.close()
        return None

    lats = ds[lat_names[0]].values
    lons = ds[lon_names[0]].values
    da   = ds[data_vars[0]]

    # Year selection — try dt accessor first, fall back to manual indexing.
    try:
        da_sel = da.sel(time=da.time.dt.year.isin(range(year_start, year_end + 1)))
    except AttributeError:
        try:
            years_in = np.array([t.year for t in da.time.values])
            mask     = (years_in >= year_start) & (years_in <= year_end)
            da_sel   = da.isel(time=mask)
        except Exception as e:
            ds.close()
            print(f"\n    Time selection failed: {e}")
            return None

    if da_sel.sizes.get("time", 0) == 0:
        ds.close()
        return None

    # Load into memory before groupby to avoid slow repeated file seeks.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            da_sel = da_sel.load()
            if stat == "max":
                ann_da = da_sel.groupby("time.year").max(dim="time", skipna=True)
            else:
                ann_da = da_sel.groupby("time.year").min(dim="time", skipna=True)
        except Exception as e:
            ds.close()
            print(f"\n    groupby failed: {e}")
            return None

    ann = ann_da.values.astype(np.float32)
    # Mask obvious fill values.
    ann[ann >  1e15] = np.nan
    ann[ann < -1e10] = np.nan
    ds.close()

    if ann.shape[0] == 0:
        return None

    return ann, lats, lons


# ── Core computation ──────────────────────────────────────────────────────────

def compute_threshold(ann_stack: np.ndarray, rank: int,
                      direction: str) -> np.ndarray:
    """
    Return the rank-th most extreme value at each grid cell.
      direction="above" → rank-th highest  (warm/wet/wind records)
      direction="below" → rank-th lowest   (cold records)
    Always returns values in raw units.
    """
    n    = ann_stack.shape[0]
    rank = min(rank, n)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if direction == "above":
            filled      = np.where(np.isfinite(ann_stack), ann_stack, -np.inf)
            sorted_desc = np.sort(filled, axis=0)[::-1]
            thresh      = sorted_desc[rank - 1]
        else:
            filled     = np.where(np.isfinite(ann_stack), ann_stack, np.inf)
            sorted_asc = np.sort(filled, axis=0)
            thresh     = sorted_asc[rank - 1]
        thresh = np.where(np.isfinite(thresh), thresh, np.nan)
    return thresh.astype(np.float32)


def compute_ref_pct(ann_stack: np.ndarray, threshold: np.ndarray,
                    direction: str) -> np.ndarray:
    """
    Fraction of years where the annual stat breaks the threshold, as a
    percent in [0, 100].
    Both ann_stack and threshold must be in the same (raw) units.
    """
    n_years = ann_stack.shape[0]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if direction == "above":
            exceedances = np.sum(ann_stack > threshold[np.newaxis], axis=0)
        else:
            exceedances = np.sum(ann_stack < threshold[np.newaxis], axis=0)
    return (exceedances.astype(np.float32) / n_years * 100.0)


def stationary_pct(n_hist: int) -> float:
    """
    Expected REF % under a stationary climate.
    With THRESHOLD_RANK = 1: (1 - 1) / n_hist = 0.0%.
    See ALGORITHM NOTES at the top of this file for the rank/docstring
    inconsistency.
    """
    return (THRESHOLD_RANK - 1) / n_hist * 100.0


# ── nc output ─────────────────────────────────────────────────────────────────

def write_nc_dual(ref_data: np.ndarray, rec_data: np.ndarray,
                  lats: np.ndarray, lons: np.ndarray,
                  out_path: Path,
                  ref_long_name: str,
                  rec_long_name: str,
                  rec_units: str) -> None:
    """
    Write REF (%) and REC (rec_units) to a single compressed nc file.
    rec_data must already be in the desired display units before calling.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    FILL = np.float32(1e20)

    def _fill(arr: np.ndarray) -> np.ndarray:
        return np.where(np.isfinite(arr), arr, FILL).astype(np.float32)

    if lats.ndim == 2:
        dims   = ["y", "x"]
        coords = {"lat": (["y", "x"], lats.astype(np.float32)),
                  "lon": (["y", "x"], lons.astype(np.float32))}
    else:
        dims   = ["lat", "lon"]
        coords = {"lat": (["lat"], lats.astype(np.float32)),
                  "lon": (["lon"], lons.astype(np.float32))}

    ds = xr.Dataset(
        {
            "REF": (dims, _fill(ref_data),
                    {"long_name": ref_long_name, "units": "%"}),
            "REC": (dims, _fill(rec_data),
                    {"long_name": rec_long_name, "units": rec_units}),
        },
        coords=coords,
    )
    enc = {v: {"zlib": True, "complevel": 4, "dtype": "float32",
               "_FillValue": FILL}
           for v in ("REF", "REC")}
    ds.to_netcdf(out_path, encoding=enc)
    ds.close()


# ── Path helpers ──────────────────────────────────────────────────────────────

def p_hist(indicator: str, model: str, ensemble: str, bp_tag: str) -> Path:
    """Construct output path for a historical base file."""
    fname = (f"{indicator}_historical_{model}_{ensemble}"
             f"_base_{bp_tag}_{bp_tag}_ANN_NZ12km.nc")
    return OUT_ROOT / "historical" / "static_maps" / indicator / fname


def p_fut(indicator: str, ssp: str, model: str, ensemble: str,
          fp_tag: str, bp_tag: str) -> Path:
    """Construct output path for a future change file."""
    fname = (f"{indicator}_{ssp}_{model}_{ensemble}"
             f"_change_{fp_tag}_{bp_tag}_ANN_NZ12km.nc")
    return OUT_ROOT / ssp / "static_maps" / indicator / fname


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--indicator", default=None, help="e.g. REC_TXx")
    parser.add_argument("--scenario",  default=None, help="e.g. ssp370")
    parser.add_argument("--force",     action="store_true",
                        help="Overwrite existing output files")
    args = parser.parse_args()

    indicators = (
        {args.indicator: RECORD_INDICATORS[args.indicator]}
        if args.indicator and args.indicator in RECORD_INDICATORS
        else RECORD_INDICATORS
    )
    future_scenarios = (
        [args.scenario] if args.scenario and args.scenario != "historical"
        else [s for s in SCENARIOS if s != "historical"]
    )

    for indicator, (var_dir, var_hint, phys_units, stat, direction) in indicators.items():
        rec_scale = _REC_UNIT_SCALE.get(indicator, 1.0)

        print(f"\n{'='*65}")
        print(f"Indicator : {indicator}  (variable: {var_dir})")
        print(f"  Stat={stat}  Direction={direction}  "
              f"REC units={phys_units}  scale={rec_scale}")
        print(f"  REF variable : record chance %  |  "
              f"REC variable : running {stat} in {phys_units}")

        # ── Step 1: historical ────────────────────────────────────────────
        print(f"\n  [1/2] Historical daily data ...")
        hist_files = group_daily_files("historical", var_dir)
        print(f"        {len(hist_files)} models found")
        if not hist_files:
            print(f"  SKIP: no historical data")
            continue

        thresh_store:   dict[str, dict] = {bp: {} for bp in BASELINES}
        running_store:  dict[str, dict] = {bp: {} for bp in BASELINES}
        stat_pct_store: dict[str, dict] = {bp: {} for bp in BASELINES}
        coords_store:   dict[str, dict] = {bp: {} for bp in BASELINES}

        for (model, ensemble), nc_path in sorted(hist_files.items()):
            print(f"    {model} ({ensemble})", end="  ", flush=True)

            for bp_tag, (y0, y1) in BASELINES.items():
                result = load_annual_stat(nc_path, var_hint, y0, y1, stat)
                if result is None:
                    print(f"[{bp_tag}: no data]", end="  ")
                    continue
                ann_stack, lats, lons = result
                n_yrs = ann_stack.shape[0]
                if n_yrs < THRESHOLD_RANK:
                    print(f"[{bp_tag}: only {n_yrs} yrs]", end="  ")
                    continue

                # Threshold computed in raw units.
                thresh   = compute_threshold(ann_stack, THRESHOLD_RANK, direction)
                stat_pct = stationary_pct(n_yrs)
                key      = (model, ensemble)

                thresh_store[bp_tag][key]   = thresh
                running_store[bp_tag][key]  = thresh.copy()  # seed for future periods
                stat_pct_store[bp_tag][key] = stat_pct
                coords_store[bp_tag][key]   = (lats, lons)

                p = p_hist(indicator, model, ensemble, bp_tag)
                if not p.exists() or args.force:
                    write_nc_dual(
                        ref_data      = np.full_like(thresh, stat_pct),
                        rec_data      = thresh * rec_scale,  # convert at write
                        lats=lats, lons=lons,
                        out_path      = p,
                        ref_long_name = (f"Stationary record chance % "
                                         f"(= {stat_pct:.1f}%, {bp_tag})"),
                        rec_long_name = (f"Historical rank-{THRESHOLD_RANK} "
                                         f"annual {stat} — record threshold ({bp_tag})"),
                        rec_units     = phys_units,
                    )
                print(f"[{bp_tag}: n={n_yrs}, base={stat_pct:.1f}%]", end="  ")
            print()

        # ── Step 2: future ────────────────────────────────────────────────
        print(f"\n  [2/2] Future files ...")

        for ssp in future_scenarios:
            print(f"\n    Scenario: {ssp}")
            fut_files = group_daily_files(ssp, var_dir)
            print(f"    {len(fut_files)} models found")

            for bp_tag in BASELINES:
                print(f"\n      Baseline: {bp_tag}")

                # Running record seeded from historical threshold, maintained
                # in raw units. Conversion applied only at write time.
                running: dict[tuple, np.ndarray] = {
                    key: arr.copy()
                    for key, arr in running_store[bp_tag].items()
                }

                # CRITICAL: iterate in chronological order (insertion order).
                for fp_tag, (fy0, fy1) in FUTURE_PERIODS.items():
                    n_written = n_skipped = n_missing = 0

                    for (model, ensemble), nc_path in sorted(fut_files.items()):
                        key = (model, ensemble)
                        if key not in thresh_store[bp_tag]:
                            n_missing += 1
                            continue

                        lats, lons = coords_store[bp_tag][key]
                        thresh     = thresh_store[bp_tag][key]
                        stat_pct   = stat_pct_store[bp_tag][key]

                        p    = p_fut(indicator, ssp, model, ensemble, fp_tag, bp_tag)
                        need = not p.exists() or args.force

                        # Always load to advance the running record, even if
                        # the output file already exists.
                        result = load_annual_stat(nc_path, var_hint, fy0, fy1, stat)
                        if result is None:
                            n_missing += 1
                            continue
                        ann_stack, _, _ = result

                        if ann_stack.shape[1:] != thresh.shape:
                            print(f"\n        Shape mismatch {model}: "
                                  f"{ann_stack.shape[1:]} vs {thresh.shape}")
                            n_missing += 1
                            continue

                        # REF % uses raw units throughout.
                        ref_pct = compute_ref_pct(ann_stack, thresh, direction)

                        # Advance running record (raw units).
                        if direction == "above":
                            period_extreme = np.nanmax(ann_stack, axis=0)
                            running[key]   = np.fmax(
                                running[key], period_extreme
                            ).astype(np.float32)
                        else:
                            period_extreme = np.nanmin(ann_stack, axis=0)
                            running[key]   = np.fmin(
                                running[key], period_extreme
                            ).astype(np.float32)

                        if need:
                            write_nc_dual(
                                ref_data      = ref_pct,
                                rec_data      = running[key] * rec_scale,  # convert at write
                                lats=lats, lons=lons,
                                out_path      = p,
                                ref_long_name = (f"Record chance % per year "
                                                 f"({fp_tag}, rank={THRESHOLD_RANK}, "
                                                 f"{direction} {bp_tag} threshold)"),
                                rec_long_name = (f"Running {direction} record "
                                                 f"(annual {stat}) through {fp_tag}"),
                                rec_units     = phys_units,
                            )
                            n_written += 1
                        else:
                            n_skipped += 1

                    print(f"        {fp_tag}: wrote={n_written}  "
                          f"skipped={n_skipped}  missing={n_missing}")

    print(f"\n{'='*65}")
    print("Done.")


if __name__ == "__main__":
    main()