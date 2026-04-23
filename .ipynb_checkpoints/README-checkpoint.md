# NZ Climate Indicator Dashboard

An interactive web application for exploring projected climate change across
New Zealand under multiple emissions scenarios. Built with Streamlit.

---

## What the app does

The dashboard visualises **25+ climate indicators** (temperature, rainfall, wind,
and record-chance metrics) downscaled to a 12 km NZ grid from global CMIP6 model
ensembles. The core map page (`pages/2_NZ_Map.py`) renders two side-by-side
animated Leaflet maps:

| Panel | Shows |
|-------|-------|
| **Left** | Absolute projected values (sequential colour scale) |
| **Right** | Change from a chosen historical baseline (diverging colour scale) |

Key features:
- Four SSP emissions scenarios (SSP1-2.6 through SSP5-8.5)
- Two historical baselines (1995–2014, 1986–2005)
- Multi-model ensemble mean + per-model selection
- Animated timeline blending between snapshot periods
- Click-to-pin time-series chart with model uncertainty bands (50 % / 90 % intervals)
- Hover tooltips with interpolated values at any grid point
- Regional council and country border overlays

---

## Repository structure

```
.
├── Home.py                        # Landing page
├── pages/
│   └── 2_NZ_Map.py                # Main map application
├── assets/
│   ├── coastlines/                # NZ coastline + regional council shapefiles
│   ├── color_ranges/              # Pre-computed colour scale JSON files
│   └── frame_cache/               # Runtime PNG cache (gitignored in production)
├── helper_scripts/
│   ├── precompute_color_ranges.py # Compute vmin/vmax across all indicators
│   ├── precompute_frames.py       # Pre-render PNG snapshots (PBS job)
│   ├── precompute_record.py       # Pre-compute record-chance indicators
│   └── precompute_uncertainty.py  # Build model-spread cache (required before running)
├── test/
│   ├── generate_demo_data.py      # Creates synthetic demo dataset
│   ├── test_README.md             # Demo data documentation
│   └── demo_data/                 # Synthetic NetCDF + uncertainty cache (committed)
├── requirements.txt
└── README.md
```

---

## Quick start — demo mode (no NIWA data required)

The app ships with a small synthetic dataset covering the **TX (mean daily max
temperature)** indicator under **SSP3-7.0**, so the full UI can be explored
without access to any external storage.

### 1. Clone the repository

```bash
git clone <repo-url>
cd nz-climate-git
```

### 2. Create a Python environment

**With conda (recommended):**

```bash
conda create -n nz-climate python=3.11
conda activate nz-climate
```

**With venv:**

```bash
python3.11 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
```

> **Python version:** 3.11 is recommended. The app has been tested on 3.11
> and 3.9. Avoid 3.12+ until `cartopy` and `fiona` wheels are widely available
> for it.

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

Some packages (`geopandas`, `fiona`, `cartopy`, `rasterio`) have C extensions.
If `pip` fails on any of them, try conda first:

```bash
conda install -c conda-forge geopandas fiona rasterio cartopy shapely
pip install -r requirements.txt   # installs remaining pure-Python deps
```

### 4. Generate the demo data (first time only)

```bash
python test/generate_demo_data.py
```

This creates the synthetic NetCDF files and uncertainty cache under
`test/demo_data/`. They are already committed to the repository, so you only
need to re-run this if you delete them or want to regenerate from scratch.

### 5. Launch the app

```bash
streamlit run Home.py
```

Open [http://localhost:8501](http://localhost:8501) in your browser. Navigate to
**NZ Climate Indicator Map**. A blue banner confirms demo mode is active.

> Demo mode activates automatically when the real NIWA data paths are absent
> and `test/demo_data/` exists — no configuration needed.

---

## Running on NIWA HPC (ESI)

On the NIWA cluster the Conda environment is initialised via the NIWA module
system before activating the project environment:

```bash
. /opt/niwa/profile/conda_24.11.3_2025.05.1.sh
conda activate nz-climate

/path/to/conda/envs/nz-climate/bin/streamlit run Home.py \
    --server.port 8502 \
    --server.headless true
```

The real data is read from the paths configured in `pages/2_NZ_Map.py`:

```python
DATA_ROOT = Path(os.environ.get(
    "NZMAP_DATA_ROOT",
    "/esi/project/niwa03712/ML_Downscaled_CMIP6/...",
))
REC_ROOT = Path(os.environ.get(
    "NZMAP_REC_ROOT",
    "/esi/project/niwa03712/westphall/nz-climate/REC",
))
```

These can be overridden without code changes via environment variables:

```bash
export NZMAP_DATA_ROOT=/your/data/path
export NZMAP_REC_ROOT=/your/rec/path
streamlit run Home.py
```

Before running with real data, the uncertainty cache must be pre-computed:

```bash
python helper_scripts/precompute_uncertainty.py
```

---

## Cloud / Azure storage

> **🚧 Placeholder — to be completed once Azure access is confirmed.**

The current hardcoded HPC paths will be replaced with Azure Blob Storage
references once the storage account is provisioned. The intended approach is:

- Data will be mounted or accessed via the `NZMAP_DATA_ROOT` / `NZMAP_REC_ROOT`
  environment variables — no code changes required.
- Pre-computed caches (frame PNGs, uncertainty bands) will be stored in a
  dedicated container and synced to the app's `assets/` directory on startup.
- Credentials will be handled via Azure Managed Identity / environment
  variables — no secrets in code.

**The demo mode in this repository proves the full application pipeline works
end-to-end from a clean checkout, independent of any storage backend.**

---

## Notes for code reviewers

### Data paths
The two internal HPC paths in `pages/2_NZ_Map.py` (`DATA_ROOT`, `REC_ROOT`)
are the only pieces of NIWA-internal information in the codebase. Both are
already externalised to environment variables. They will be replaced with
Azure storage references once access is granted.

### Demo mode
The `_DEMO_MODE` flag (Part 1 of `2_NZ_Map.py`) redirects both data roots and
the uncertainty cache directory to `test/demo_data/` when the real paths are
absent. It adds a visible banner but otherwise exercises the identical code path
as production — the same renderer, the same JS player, the same chart logic.

### Pre-computed caches
The app is designed around two levels of caching:

| Cache | Location | Purpose |
|-------|----------|---------|
| `assets/frame_cache/*.pkl` | Disk | Rendered PNG snapshots (expensive, minutes per selection) |
| `assets/uncertainty_cache/*.pkl` | Disk | Model-spread percentile bands (generated by `precompute_uncertainty.py`) |

The frame cache entries committed to this repo were generated from the demo
data and are safe to delete — they will be regenerated on first load.

### Security notes
- No credentials, tokens, or API keys anywhere in the codebase.
- On-disk pickle files (`frame_cache`, `uncertainty_cache`) are loaded with
  `pickle.load`. This is safe as long as only the app writes to those
  directories. The Azure deployment will restrict write access accordingly.
- The sidebar posts `postMessage` to sibling iframes for opacity/speed
  controls — intentional, same-origin only.

### Known minor issues (non-blocking)
- `border-radFius` typo in an inline style string in the pills banner
  (cosmetically harmless, browser ignores unknown properties).
- A dead second `return` statement in `build_loading_screen_html` —
  unreachable code, safe to delete in a follow-up PR.
- Redundant first land-mask block in `prerender_snapshots` — values
  computed are immediately overwritten. Flagged in inline comments.

---

## Environment troubleshooting

| Problem | Fix |
|---------|-----|
| `geopandas` / `fiona` install fails | `conda install -c conda-forge geopandas fiona` |
| `cartopy` install fails | `conda install -c conda-forge cartopy` |
| `No such file or directory: Home.py` | You are not in the repo root — `cd nz-climate-git` |
| Port already in use | Add `--server.port 8503` (or any free port) |
| Demo banner not appearing | Check `test/demo_data/` exists; re-run `python test/generate_demo_data.py` |
| Uncertainty cache error in demo | Re-run `python test/generate_demo_data.py` |

---

## Requirements summary

| Package | Min version | Notes |
|---------|-------------|-------|
| Python | 3.11 | 3.9 also tested |
| streamlit | 1.36 | |
| xarray | 2024.1 | NetCDF reading |
| netcdf4 | 1.6 | xarray engine |
| numpy | 1.26 | |
| scipy | 1.13 | Interpolation, image smoothing |
| matplotlib | latest | PNG rendering |
| Pillow | latest | Image encoding |
| plotly | 5.22 | Chart.js via Streamlit component |
| geopandas | 0.14 | Coastline / region shapefiles |
| shapely | latest | Geometry operations |
| fiona | latest | Shapefile I/O |
| rasterio | latest | Raster support |
| cartopy | latest | Optional — CRS utilities |