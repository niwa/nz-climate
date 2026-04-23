#!/usr/bin/env python3
# precompute_color_ranges.py (annotated for review)
#
# OVERVIEW
# --------
# Offline script — run once (or after new indicators/scenarios are added) to
# generate two JSON files consumed by the Streamlit app at runtime:
#
#   assets/color_ranges/color_ranges.json
#       Symmetric half-range for the change (Δ) map panel.
#       Format: { "TX": 4.12, "PR": 320.5, ... }
#
#   assets/color_ranges/abs_color_ranges.json
#       vmin/vmax for the absolute-value map panel.
#       Format: { "TX": {"vmin": 8.2, "vmax": 28.7}, ... }
#
# Usage (from the repo root, with the project conda env active):
#     python precompute_color_ranges.py
#     python precompute_color_ranges.py --abs-only
#     python precompute_color_ranges.py --change-only
#     python precompute_color_ranges.py --force       # recompute all
#
# PRIVACY / SECURITY NOTES FOR REVIEWERS
# ----------------------------------------
# * DATA_ROOT and REC_ROOT are hardcoded to internal HPC paths below.
#   REVIEW FLAG: These should be externalised to environment variables
#   (NZMAP_DATA_ROOT, NZMAP_REC_ROOT) to match the pattern already
#   applied in 2_NZ_Map.py (annotated Part 1). This makes the script
#   safe to commit and portable across deployments without code changes.
#   Suggested fix:
#       import os
#       DATA_ROOT = Path(os.environ.get("NZMAP_DATA_ROOT", "<fallback>"))
#       REC_ROOT  = Path(os.environ.get("NZMAP_REC_ROOT",  "<fallback>"))
# * No credentials, tokens, API keys, or personal data are present.
# * Output JSON files contain only numeric colour-range values — no paths
#   or internal identifiers are written to disk.
#
# LOGIC NOTES FOR REVIEWERS
# --------------------------
# * compute_change_range_for() has a misplaced docstring — it appears AFTER
#   the early-return for REC indicators, making it unreachable as a docstring.
#   It is still displayed by help() because Python picks it up as a string
#   literal, but linters and IDEs will flag it. Move it to just below the
#   `def` line in a follow-up PR.
# * The same issue exists in compute_abs_range_for() — the docstring follows
#   the REC early-return block. Same fix applies.
# * Results are written to disk after every indicator so a crash mid-run
#   does not lose all prior work. The --force flag bypasses the resume logic.

"""
precompute_color_ranges.py
--------------------------
Run this script once (or whenever new indicators / scenarios are added) to
generate TWO JSON files in assets/color_ranges/:

  color_ranges.json       — symmetric half-range for the change (Δ) map
                            { "TX": 4.12, "PR": 320.5, ... }

  abs_color_ranges.json   — vmin/vmax for the absolute-value map
                            { "TX": {"vmin": 8.2, "vmax": 28.7}, ... }

Usage (from the repo root, with the nz-climate conda env active):
    python precompute_color_ranges.py

Flags:
    --abs-only      only recompute abs_color_ranges.json
    --change-only   only recompute color_ranges.json
    --force         recompute all indicators even if already cached
"""

import argparse
import json
import re
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import xarray as xr

(Path(__file__).parent / "assets/color_ranges").mkdir(parents=True, exist_ok=True)

# ── Constants ─────────────────────────────────────────────────────────────────
# REVIEW FLAG: Externalise these to env vars — see PRIVACY NOTES above.
# They are the only internal-infrastructure-specific values in this file.
import os
DATA_ROOT = Path(os.environ.get(
    "NZMAP_DATA_ROOT",
    "/esi/project/niwa03712/ML_Downscaled_CMIP6/NIWA-REMS_indicators/output_v3",
))
REC_ROOT = Path(os.environ.get(
    "NZMAP_REC_ROOT",
    "/esi/project/niwa03712/westphall/nz-climate/REC",
))

SSP_OPTIONS = ["ssp126", "ssp245", "ssp370", "ssp585"]
BP_OPTIONS  = ["bp1995-2014", "bp1986-2005"]
SEASON_ANN  = "ANN"
SEASON_LABELS = {
    "ANN": "Annual",
    "DJF": "Summer (DJF)",
    "MAM": "Autumn (MAM)",
    "JJA": "Winter (JJA)",
    "SON": "Spring (SON)",
}

_REC_INDICATORS = {"REC_TXx", "REC_Rx1day", "REC_Wx1day", "REC_TNn"}
_KELVIN_INDICATORS = {"TX", "TXx", "TN", "TNn"}

# Indicators whose absolute-value map uses LogNorm (vmin must be > 0).
_LOG_INDICATORS = {
    "PR", "RR1mm", "RR25mm", "Rx1day",
    "R99p", "R99pVAL", "R99pVALWet",
    "Wd10", "Wd25", "Wd99pVAL", "Wx1day",
    "sfcwind",
}

# Indicators whose change map uses SymLogNorm (wider percentile cutoff needed).
_SYMLOG_CHANGE_INDICATORS = {
    "PR", "RR1mm", "RR25mm", "Rx1day",
    "R99p", "R99pVAL", "R99pVALWet",
}


# ── Data helpers ──────────────────────────────────────────────────────────────

def _indicator_root(indicator: str) -> Path:
    return REC_ROOT if indicator in _REC_INDICATORS else DATA_ROOT


def list_indicators(scenario: str) -> list[str]:
    """All indicator codes with at least one file under the given scenario."""
    results: set[str] = set()
    base = DATA_ROOT / scenario / "static_maps"
    if base.exists():
        results.update(d.name for d in base.iterdir() if d.is_dir())
    rec_base = REC_ROOT / scenario / "static_maps"
    if rec_base.exists():
        results.update(d.name for d in rec_base.iterdir() if d.is_dir())
    return sorted(results)


def discover_seasons(scenario: str, indicator: str) -> list[str]:
    """Seasons present on disk for a given scenario/indicator."""
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
    """Future period tags available on disk for this combination."""
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


def find_nc_files(scenario: str, indicator: str,
                  fp_tag: str, bp_tag: str, season: str) -> list[Path]:
    folder = _indicator_root(indicator) / scenario / "static_maps" / indicator
    if not folder.exists():
        return []
    if scenario == "historical":
        return sorted(
            f for f in folder.glob("*.nc")
            if f"_base_{fp_tag}_" in f.name and f"_{season}_" in f.name
        )
    return sorted(
        f for f in folder.glob("*.nc")
        if f"_{fp_tag}_" in f.name
        and f"_{bp_tag}_" in f.name
        and f"_{season}_" in f.name
    )


def load_ensemble_mean(scenario: str, indicator: str,
                       fp_tag: str, bp_tag: str,
                       season: str) -> "np.ndarray | None":
    """Load all matching files and return their nanmean across models."""
    files = find_nc_files(scenario, indicator, fp_tag, bp_tag, season)
    if not files:
        return None
    arrays = []
    for f in files:
        try:
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
                fv = da.attrs.get("_FillValue", da.attrs.get("missing_value", None))
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
        return None
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.nanmean(np.stack(arrays, 0), 0)


# ── Change colour range ───────────────────────────────────────────────────────

def compute_change_range_for(indicator: str) -> float:
    """
    Symmetric half-range for the Δ change map.

    Uses 99th percentile for symlog indicators (log compression handles tails),
    98th for all others. REC indicators use a fixed 50.0 (± 50 pp from baseline).

    REVIEW FLAG: The docstring and the REC early-return are swapped in the
    original source — the docstring appeared after the return, making it
    unreachable. Corrected here by moving the docstring to the top.
    """
    if indicator in _REC_INDICATORS:
        return 50.0

    all_vals = []
    for ssp in SSP_OPTIONS:
        for bp in BP_OPTIONS:
            for season in discover_seasons("historical", indicator):
                for fp in discover_fp_tags(ssp, indicator, bp, season):
                    data = load_ensemble_mean(ssp, indicator, fp, bp, season)
                    if data is not None:
                        finite = data[np.isfinite(data)].ravel()
                        if len(finite):
                            all_vals.append(finite)
    if not all_vals:
        return 1.0
    combined = np.concatenate(all_vals)
    pct = 99 if indicator in _SYMLOG_CHANGE_INDICATORS else 98
    return max(float(np.nanpercentile(np.abs(combined), pct)), 1e-6)


# ── Absolute colour range ─────────────────────────────────────────────────────

def _kelvin_to_celsius(arr: np.ndarray) -> np.ndarray:
    """Convert Kelvin to Celsius only if the mean looks like Kelvin (> 200)."""
    finite = arr[np.isfinite(arr)]
    if len(finite) and float(np.nanmean(finite)) > 200:
        return arr - 273.15
    return arr


def load_baseline_abs(indicator: str, bp: str, season: str) -> "np.ndarray | None":
    """Load the historical baseline array for the given bp/season."""
    for fp in discover_fp_tags("historical", indicator, bp, season):
        if fp == bp:
            data = load_ensemble_mean("historical", indicator, fp, bp, season)
            if data is not None:
                if indicator in _KELVIN_INDICATORS:
                    data = _kelvin_to_celsius(data)
                return data
    return None


def compute_abs_range_for(indicator: str) -> dict:
    """
    Return {"vmin": ..., "vmax": ...} for the absolute-value map.

    For log-scaled indicators:
      - Percentile range widened to 5th–95th (log compression handles tails).
      - vmin hard-floored at 1e-3 (LogNorm cannot accept zero or negative).
    For linear indicators:
      - Standard 2nd–98th percentile.

    REC indicators use hand-tuned fixed physical ranges.

    REVIEW FLAG: In the original source the docstring appeared after the REC
    early-return, making it unreachable. Moved to the top here. Same fix
    needed as compute_change_range_for().

    REVIEW FLAG: The REC ranges in precompute_color_ranges.py differ slightly
    from those in 2_NZ_Map.py (e.g. REC_TNn: -10/+20 here vs -15/+10 there).
    Recommend consolidating into a single shared constants file to avoid
    drift between the script and the app.
    """
    if indicator in _REC_INDICATORS:
        if indicator == "REC_TXx":    return {"vmin": 20.0,  "vmax": 45.0}
        if indicator == "REC_TNn":    return {"vmin": -10.0, "vmax": 20.0}
        if indicator == "REC_Rx1day": return {"vmin": 0.0,   "vmax": 5e-4}
        if indicator == "REC_Wx1day": return {"vmin": 10.0,  "vmax": 50.0}

    all_vals = []
    for bp in BP_OPTIONS:
        for season in discover_seasons("historical", indicator):
            baseline = load_baseline_abs(indicator, bp, season)
            if baseline is not None:
                arr = baseline[np.isfinite(baseline)].ravel()
                if len(arr):
                    all_vals.append(arr)

            for ssp in SSP_OPTIONS:
                for fp in discover_fp_tags(ssp, indicator, bp, season):
                    change = load_ensemble_mean(ssp, indicator, fp, bp, season)
                    if change is None or baseline is None:
                        continue
                    abs_data = baseline + change
                    arr = abs_data[np.isfinite(abs_data)].ravel()
                    if len(arr):
                        all_vals.append(arr)

    if not all_vals:
        return {"vmin": 0.0, "vmax": 1.0}

    combined = np.concatenate(all_vals)
    is_log   = indicator in _LOG_INDICATORS
    pct_lo   = 5  if is_log else 2
    pct_hi   = 95 if is_log else 98

    vmin = float(np.nanpercentile(combined, pct_lo))
    vmax = float(np.nanpercentile(combined, pct_hi))

    if is_log:
        vmin = max(vmin, 1e-3)

    if vmax - vmin < 1e-6:
        vmax = vmin + 1.0

    return {"vmin": round(vmin, 4), "vmax": round(vmax, 4)}


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--abs-only",    action="store_true")
    parser.add_argument("--change-only", action="store_true")
    parser.add_argument("--force",       action="store_true",
                        help="Recompute even if already cached")
    args = parser.parse_args()

    if not DATA_ROOT.exists():
        print(f"ERROR: DATA_ROOT not found:\n  {DATA_ROOT}")
        sys.exit(1)

    indicators = list_indicators("historical")
    if not indicators:
        print("No indicators found — check DATA_ROOT.")
        sys.exit(1)

    do_change = not args.abs_only
    do_abs    = not args.change_only

    change_out = Path(__file__).parent / "assets/color_ranges" / "color_ranges.json"
    abs_out    = Path(__file__).parent / "assets/color_ranges" / "abs_color_ranges.json"

    # ── Change ranges ──────────────────────────────────────────────────────
    if do_change:
        print("=" * 60)
        print("CHANGE COLOUR RANGES  →  color_ranges.json")
        print("=" * 60)

        existing: dict[str, float] = {}
        if change_out.exists() and not args.force:
            with open(change_out) as f:
                existing = json.load(f)
            print(f"Resuming — {len(existing)} indicator(s) already cached\n")

        results = dict(existing)
        for i, ind in enumerate(indicators, 1):
            prefix    = f"[{i}/{len(indicators)}]  {ind:<14}"
            norm_note = " [symlog]" if ind in _SYMLOG_CHANGE_INDICATORS else ""
            if ind in results:
                print(f"{prefix}  ±{results[ind]:.4f}  (skipped){norm_note}")
                continue
            print(f"{prefix}  computing{norm_note}...", end="", flush=True)
            t0   = time.time()
            half = compute_change_range_for(ind)
            results[ind] = half
            print(f"  ±{half:.4f}  ({time.time()-t0:.1f}s)")
            # Write after each indicator so a crash doesn't lose prior work.
            with open(change_out, "w") as f:
                json.dump(results, f, indent=2, sort_keys=True)

        print(f"\nDone — {len(results)} indicator(s) → {change_out}\n")

    # ── Absolute ranges ────────────────────────────────────────────────────
    if do_abs:
        print("=" * 60)
        print("ABSOLUTE VALUE RANGES  →  abs_color_ranges.json")
        print("=" * 60)

        existing: dict = {}
        if abs_out.exists() and not args.force:
            with open(abs_out) as f:
                existing = json.load(f)
            print(f"Resuming — {len(existing)} indicator(s) already cached\n")

        results = dict(existing)
        for i, ind in enumerate(indicators, 1):
            prefix    = f"[{i}/{len(indicators)}]  {ind:<14}"
            norm_note = " [log]" if ind in _LOG_INDICATORS else ""
            if ind in results:
                r = results[ind]
                print(f"{prefix}  [{r['vmin']:.4f}, {r['vmax']:.4f}]  (skipped){norm_note}")
                continue
            print(f"{prefix}  computing{norm_note}...", end="", flush=True)
            t0  = time.time()
            rng = compute_abs_range_for(ind)
            results[ind] = rng
            print(f"  [{rng['vmin']:.4f}, {rng['vmax']:.4f}]  ({time.time()-t0:.1f}s)")
            with open(abs_out, "w") as f:
                json.dump(results, f, indent=2, sort_keys=True)

        print(f"\nDone — {len(results)} indicator(s) → {abs_out}\n")


if __name__ == "__main__":
    main()