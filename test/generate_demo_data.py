#!/usr/bin/env python3
"""
Generate synthetic demo data for the NZ Climate Dashboard.

Run this once from the repo root before launching the app without
access to NIWA HPC storage:

    python test/generate_demo_data.py

Creates
-------
test/demo_data/
    historical/static_maps/TX/   — 3 model NC files (base, bp1995-2014, ANN)
    ssp370/static_maps/TX/       — 6 model NC files (2 periods × 3 models)
    uncertainty_cache/           — pre-computed model-spread pickle

assets/color_ranges/
    color_ranges.json            — TX entry added if absent
    abs_color_ranges.json        — TX entry added if absent

Commit the entire test/demo_data/ tree to the repository so reviewers
can run the app without any external data access.
"""
from __future__ import annotations

import hashlib
import json
import pickle
import sys
from pathlib import Path

import numpy as np

try:
    import xarray as xr
    from scipy.ndimage import gaussian_filter
except ImportError as exc:
    sys.exit(f"Missing dependency: {exc}\n  pip install xarray scipy netcdf4")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT       = Path(__file__).resolve().parent.parent   # repo root
DEMO_ROOT  = Path(__file__).resolve().parent / "demo_data"
CACHE_DIR  = DEMO_ROOT / "uncertainty_cache"
COLOR_DIR  = ROOT / "assets" / "color_ranges"

# ---------------------------------------------------------------------------
# Demo configuration  (only TX / ssp370 / bp1995-2014 / ANN generated)
# ---------------------------------------------------------------------------
INDICATOR = "TX"
SCENARIO  = "ssp370"
BP_TAG    = "bp1995-2014"
SEASON    = "ANN"
MODELS    = ["DEMO-GCM-A", "DEMO-GCM-B", "DEMO-GCM-C"]
ENSEMBLE  = "r1i1p1f1"
FP_TAGS   = ["fp2041-2060", "fp2081-2100"]

# Centre years must match fp_centre_year() in the app.
# bp1995-2014 → 2004.5,  fp2041-2060 → 2050.5,  fp2081-2100 → 2090.5
SNAP_YEARS      = [2004.5, 2050.5, 2090.5]
SNAP_LABELS     = ["Historical 1995–2014", "2041–2060", "2081–2100"]
SNAP_FP_RANGES  = ["1995–2014", "2041–2060", "2081–2100"]

# Per-model warming scale (near / far future) and baseline bias
_M_BIAS         = [-0.4,  0.0, +0.4]   # °C baseline offset per model
_NEAR_SCALES    = [ 1.3,  1.5,  1.7]   # °C mean near-future warming per model
_FAR_SCALES     = [ 2.7,  3.2,  3.7]   # °C mean far-future warming per model

# ---------------------------------------------------------------------------
# Synthetic NZ grid  (0.5° resolution, ~675 points)
# ---------------------------------------------------------------------------
LAT_1D = np.arange(-47.0, -33.5, 0.5, dtype="float32")   # 27 pts, S→N
LON_1D = np.arange(166.0, 178.5, 0.5, dtype="float32")   # 25 pts, W→E
LAT_2D, LON_2D = np.meshgrid(LAT_1D, LON_1D, indexing="ij")   # (27, 25)
NLAT, NLON = LAT_2D.shape
N_PTS = NLAT * NLON


# ---------------------------------------------------------------------------
# Synthetic field generators
# ---------------------------------------------------------------------------
def _lat_norm() -> np.ndarray:
    """0 at southern tip, 1 at northern tip."""
    return (LAT_2D - LAT_1D.min()) / (LAT_1D.max() - LAT_1D.min())


def _smooth(arr: np.ndarray, sigma: float = 3.5) -> np.ndarray:
    return gaussian_filter(arr.astype("float64"), sigma=sigma).astype("float32")


def _tx_base(seed: int = 0) -> np.ndarray:
    """
    Synthetic TX baseline (mean daily max temperature, °C).
    North–south gradient (8 °C south → 24 °C north) plus smooth random noise.
    """
    rng = np.random.default_rng(seed)
    field = 8.0 + _lat_norm() * 16.0
    noise = _smooth(rng.standard_normal((NLAT, NLON)).astype("float32"), sigma=3.5) * 1.5
    return (field + noise).astype("float32")


def _tx_change(scale: float, seed: int = 0) -> np.ndarray:
    """
    Synthetic TX change (°C).
    Roughly uniform warming (slightly larger in the north) plus smooth noise.
    """
    rng = np.random.default_rng(seed + 200)
    field = scale * (0.85 + _lat_norm() * 0.30)
    noise = _smooth(rng.standard_normal((NLAT, NLON)).astype("float32"), sigma=4.0) * scale * 0.12
    return (field + noise).astype("float32")


# ---------------------------------------------------------------------------
# NetCDF writing
# ---------------------------------------------------------------------------
def _write_nc(path: Path, data: np.ndarray, var: str = INDICATOR) -> None:
    """Write a 2-D (lat × lon) float32 array to a NetCDF4 file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    ds = xr.Dataset(
        {var: xr.DataArray(
            data.astype("float32"),
            dims=["lat", "lon"],
            attrs={"units": "degC", "long_name": f"Demo {var}"},
        )},
        coords={
            "lat": xr.DataArray(LAT_1D, dims=["lat"], attrs={"units": "degrees_north"}),
            "lon": xr.DataArray(LON_1D, dims=["lon"], attrs={"units": "degrees_east"}),
        },
    )
    # Disable xarray's automatic _FillValue so the app loader reads raw values.
    ds.to_netcdf(path, engine="netcdf4",
                 encoding={var: {"_FillValue": None, "dtype": "float32"}})
    print(f"  wrote  {path.relative_to(ROOT)}")


def generate_nc_files() -> dict[str, np.ndarray]:
    """
    Write 9 NetCDF files (3 models × (1 historical + 2 SSP futures)).
    Returns per-model arrays needed by the cache generator.
    """
    print("\n── NetCDF files ─────────────────────────────────────────────────")
    per_model: dict[str, dict] = {}

    for mi, model in enumerate(MODELS):
        bias  = _M_BIAS[mi]
        base  = _tx_base(seed=mi) + bias
        near  = _tx_change(_NEAR_SCALES[mi], seed=mi * 10)
        far   = _tx_change(_FAR_SCALES[mi],  seed=mi * 10 + 1)
        per_model[model] = {"base": base, "near": near, "far": far}

        # Historical baseline
        hist_dir = DEMO_ROOT / "historical" / "static_maps" / INDICATOR
        _write_nc(
            hist_dir / f"{INDICATOR}_historical_{model}_{ENSEMBLE}"
                       f"_base_{BP_TAG}_{SEASON}_NZ12km.nc",
            base,
        )

        # Future snapshots
        for fp_tag, arr in zip(FP_TAGS, [near, far]):
            ssp_dir = DEMO_ROOT / SCENARIO / "static_maps" / INDICATOR
            _write_nc(
                ssp_dir / f"{INDICATOR}_{SCENARIO}_{model}_{ENSEMBLE}"
                          f"_change_{fp_tag}_{BP_TAG}_{SEASON}_NZ12km.nc",
                arr,
            )

    return per_model


# ---------------------------------------------------------------------------
# Uncertainty cache
# ---------------------------------------------------------------------------
def _cache_key(indicator: str, ssp: str, bp_tag: str, season: str) -> str:
    """Must match _uncertainty_cache_key() in pages/2_NZ_Map.py exactly."""
    return hashlib.md5(f"{indicator}|{ssp}|{bp_tag}|{season}".encode()).hexdigest()


def generate_uncertainty_cache(per_model: dict[str, dict]) -> None:
    """
    Build and pickle the uncertainty-band cache consumed by the app.
    Shape convention:  (n_snaps, n_pts)  where n_pts = NLAT × NLON.
    """
    print("\n── Uncertainty cache ────────────────────────────────────────────")
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    flat_lat = LAT_2D.ravel().astype("float32")
    flat_lon = LON_2D.ravel().astype("float32")

    # ── Per-model data matrices  (n_models, n_pts) ──────────────────────────
    bases = np.stack([per_model[m]["base"].ravel() for m in MODELS])   # (3, n_pts)
    nears = np.stack([per_model[m]["near"].ravel() for m in MODELS])   # (3, n_pts)
    fars  = np.stack([per_model[m]["far"].ravel()  for m in MODELS])   # (3, n_pts)

    # ── Change values  (n_snaps, n_models, n_pts) ───────────────────────────
    # Historical snapshot: zero change by definition.
    all_change = np.stack([
        np.zeros_like(bases),   # snap 0: historical
        nears,                  # snap 1: near future
        fars,                   # snap 2: far  future
    ])  # (3, 3, n_pts)

    # ── Absolute values  (n_snaps, n_models, n_pts) ─────────────────────────
    all_abs = np.stack([
        bases,
        bases + nears,
        bases + fars,
    ])  # (3, 3, n_pts)

    def _band(arr3d: np.ndarray, fn) -> np.ndarray:
        """Reduce over the model axis (axis=1) → (n_snaps, n_pts)."""
        return np.array([fn(arr3d[s], axis=0) for s in range(arr3d.shape[0])],
                        dtype="float32")

    # ── Ensemble means ───────────────────────────────────────────────────────
    ens_vals     = _band(all_change, np.mean)
    abs_ens_vals = _band(all_abs,    np.mean)

    # ── Percentile bands ─────────────────────────────────────────────────────
    lo_vals  = _band(all_change, np.min)
    p5_vals  = _band(all_change, lambda a, axis: np.percentile(a,  5, axis=axis))
    p25_vals = _band(all_change, lambda a, axis: np.percentile(a, 25, axis=axis))
    p75_vals = _band(all_change, lambda a, axis: np.percentile(a, 75, axis=axis))
    p95_vals = _band(all_change, lambda a, axis: np.percentile(a, 95, axis=axis))
    hi_vals  = _band(all_change, np.max)

    abs_lo_vals  = _band(all_abs, np.min)
    abs_p5_vals  = _band(all_abs, lambda a, axis: np.percentile(a,  5, axis=axis))
    abs_p25_vals = _band(all_abs, lambda a, axis: np.percentile(a, 25, axis=axis))
    abs_p75_vals = _band(all_abs, lambda a, axis: np.percentile(a, 75, axis=axis))
    abs_p95_vals = _band(all_abs, lambda a, axis: np.percentile(a, 95, axis=axis))
    abs_hi_vals  = _band(all_abs, np.max)

    # ── Summary stats (one dict per snapshot) ────────────────────────────────
    summary_stats = []
    for si in range(len(SNAP_YEARS)):
        chg = all_change[si]   # (n_models, n_pts)
        abv = all_abs[si]      # (n_models, n_pts)
        is_hist = (si == 0)
        summary_stats.append({
            "mean_change": None  if is_hist else float(np.nanmean(chg)),
            "min_change":  None  if is_hist else float(np.nanmin(chg)),
            "max_change":  None  if is_hist else float(np.nanmax(chg)),
            "mean_abs":    float(np.nanmean(abv)),
            "min_abs":     float(np.nanmin(abv)),
            "max_abs":     float(np.nanmax(abv)),
            "n_models":    len(MODELS),
        })

    # ── Assemble and pickle ──────────────────────────────────────────────────
    cache = {
        "lat_v":         flat_lat,
        "lon_v":         flat_lon,
        "snap_years":    SNAP_YEARS,
        "snap_labels":   SNAP_LABELS,
        "snap_fp_ranges": SNAP_FP_RANGES,
        "n_models":      len(MODELS),
        # Change-panel bands
        "ens_vals":   ens_vals,
        "lo_vals":    lo_vals,
        "p5_vals":    p5_vals,
        "p25_vals":   p25_vals,
        "p75_vals":   p75_vals,
        "p95_vals":   p95_vals,
        "hi_vals":    hi_vals,
        # Absolute-panel bands
        "abs_ens_vals":  abs_ens_vals,
        "abs_lo_vals":   abs_lo_vals,
        "abs_p5_vals":   abs_p5_vals,
        "abs_p25_vals":  abs_p25_vals,
        "abs_p75_vals":  abs_p75_vals,
        "abs_p95_vals":  abs_p95_vals,
        "abs_hi_vals":   abs_hi_vals,
        # Summary table
        "summary_stats": summary_stats,
        # Chart y-axis limits (°C change / absolute °C)
        "chart_ymin":     -0.5,
        "chart_ymax":      5.0,
        "abs_chart_ymin":  5.0,
        "abs_chart_ymax": 28.0,
    }

    key  = _cache_key(INDICATOR, SCENARIO, BP_TAG, SEASON)
    path = CACHE_DIR / f"{key}.pkl"
    with open(path, "wb") as fh:
        pickle.dump(cache, fh, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"  wrote  {path.relative_to(ROOT)}")
    print(f"  cache key: {key}  ({N_PTS} spatial pts × {len(SNAP_YEARS)} snaps)")


# ---------------------------------------------------------------------------
# Color ranges
# ---------------------------------------------------------------------------
def update_color_ranges() -> None:
    """
    Ensure TX has entries in both color-range JSON files.
    Skips any key that already exists so hand-tuned values are not clobbered.
    """
    print("\n── Color ranges ─────────────────────────────────────────────────")
    COLOR_DIR.mkdir(parents=True, exist_ok=True)

    # Change (Δ) half-range  — symmetric ±value
    cr_path = COLOR_DIR / "color_ranges.json"
    cr: dict = json.loads(cr_path.read_text()) if cr_path.exists() else {}
    if INDICATOR not in cr:
        cr[INDICATOR] = 3.5          # ±3.5 °C covers SSP3-7.0 end-of-century
        cr_path.write_text(json.dumps(cr, indent=2))
        print(f"  updated {cr_path.relative_to(ROOT)}")
    else:
        print(f"  skipped {cr_path.name}  ({INDICATOR} already present)")

    # Absolute value range  [vmin, vmax]
    ar_path = COLOR_DIR / "abs_color_ranges.json"
    ar: dict = json.loads(ar_path.read_text()) if ar_path.exists() else {}
    if INDICATOR not in ar:
        ar[INDICATOR] = {"vmin": 6.0, "vmax": 28.0}
        ar_path.write_text(json.dumps(ar, indent=2))
        print(f"  updated {ar_path.relative_to(ROOT)}")
    else:
        print(f"  skipped {ar_path.name}  ({INDICATOR} already present)")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main() -> None:
    print(f"Repository root : {ROOT}")
    print(f"Demo data root  : {DEMO_ROOT}")

    per_model = generate_nc_files()
    generate_uncertainty_cache(per_model)
    update_color_ranges()

    print("\n✅  Done.")
    print()
    print("Next steps:")
    print("  1. git add test/demo_data/ assets/color_ranges/")
    print("  2. git commit -m 'Add synthetic demo dataset for CI/review'")
    print("  3. streamlit run Home.py   # demo mode activates automatically")
    print()
    print("Demo covers: TX indicator · SSP3-7.0 · bp1995-2014 · Annual")
    print("Other SSP / indicator combos show an informative 'no data' message.")


if __name__ == "__main__":
    main()
