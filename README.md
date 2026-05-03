# NZ Climate Indicator Dashboard

An interactive web application for exploring projected climate change across
New Zealand under multiple emissions scenarios. Built with Streamlit.

---

## What the app does

The dashboard visualises **20+ climate indicators** spanning temperature,
precipitation, wind, agroclimatic, and record-chance metrics, downscaled to
high-resolution NZ grids from global CMIP6 model ensembles using two
independent downscaling approaches:

| Method | Description | Resolution |
|--------|-------------|------------|
| **Statistical (SD)** | AI-based statistical downscaling | 12 km |
| **Dynamical (DD)** | Physics-based regional climate model (CCAM) | 5 km |

The core map page (`pages/2_NZ_Map.py`) renders two side-by-side animated
Leaflet maps:

| Panel | Shows |
|-------|-------|
| **Left (orange)** | Absolute projected values (sequential colour scale) |
| **Right (blue)** | Change from a chosen historical baseline (diverging colour scale) |

Key features:
- Four SSP emissions scenarios (SSP1-2.6 through SSP5-8.5)
- Two historical baselines (1995–2014, 1986–2005)
- Toggle between Statistical and Dynamical downscaling methods
- Multi-model ensemble mean + per-model selection
- Animated timeline blending between snapshot periods
- Click-to-pin time-series chart with model uncertainty bands (50% / 90% intervals)
- Both downscaling methods shown simultaneously in uncertainty charts for comparison
- Hover tooltips with interpolated values at any grid point
- Regional council and country border overlays
- Indicator descriptions and per-snapshot summary statistics

---

## How the app works

The app is a **strict cache consumer** — it never performs live computation.
All rendering, ensemble statistics, and uncertainty bands are pre-computed
offline and stored as pickle files in two cache layers:

| Cache | Purpose |
|-------|---------|
| **Frame cache** (`assets/frame_cache/`, `assets/frame_cache_dd/`) | Pre-rendered PNG map snapshots |
| **Uncertainty cache** (`assets/uncertainty_cache/`, `assets/uncertainty_cache_dd/`) | Per-location model spread (percentile bands, ensemble mean) |

In production these caches are served from **Azure Blob Storage** and synced
to the app's `assets/` directory at startup. In demo mode they are read from
the committed `test/demo_data/` directory.

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
│   ├── frame_cache/               # SD frame cache (gitignored; served from Azure)
│   ├── frame_cache_dd/            # DD frame cache (gitignored; served from Azure)
│   ├── uncertainty_cache/         # SD uncertainty cache (gitignored; served from Azure)
│   └── uncertainty_cache_dd/      # DD uncertainty cache (gitignored; served from Azure)
├── test/
│   ├── generate_demo_data.py      # Creates synthetic demo dataset
│   ├── test_README.md             # Demo data documentation
│   └── demo_data/                 # Synthetic NetCDF + uncertainty cache (committed)
├── requirements.txt
└── README.md
```

> **Note:** Pre-compute helper scripts are maintained separately and not
> included in this repository. The app itself only reads from pre-built caches.

---

## Quick start — demo mode (no external data required)

The repository ships with a small synthetic dataset covering the **TX (mean
daily max temperature)** indicator under **SSP3-7.0**, so the full UI can be
explored without access to any external storage or pre-computed caches.

### 1. Clone the repository

```bash
git clone <repo-url>
cd nz-climate
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
> and 3.9. Avoid 3.12+ until `fiona` and `rasterio` wheels are widely
> available for it.

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

Some packages (`geopandas`, `fiona`, `rasterio`) have C extensions. If `pip`
fails on any of them, install via conda first:

```bash
conda install -c conda-forge geopandas fiona rasterio shapely
pip install -r requirements.txt
```

### 4. Generate the demo data (first time only)

```bash
python test/generate_demo_data.py
```

This creates synthetic NetCDF files and an uncertainty cache under
`test/demo_data/`. The outputs are already committed to the repository, so
you only need to re-run this if you delete them or want to regenerate from
scratch.

### 5. Launch the app

```bash
streamlit run Home.py
```

Open [http://localhost:8501](http://localhost:8501) in your browser and
navigate to **NZ Climate Indicator Map**. A blue banner confirms demo mode
is active.

> Demo mode activates automatically when the production data paths are absent
> and `test/demo_data/` exists — no configuration needed.

---

## Production deployment

### Data paths

The app reads data from two configurable root paths, set via environment
variables:

```bash
export NZMAP_DATA_ROOT=/path/to/indicator/data
export NZMAP_REC_ROOT=/path/to/record/indicator/data
```

### Azure Blob Storage

In production, pre-computed caches are stored in Azure Blob Storage and
accessed via the `blob_storage` module. The app detects whether it is running
in cloud mode by checking for the presence of `assets/frame_cache/` on disk:

```python
_ON_CLOUD = not Path("assets/frame_cache").exists()
```

When `_ON_CLOUD` is `True`, all cache reads are routed through Azure Blob
rather than local disk. Credentials are handled via environment variables —
no secrets appear in the codebase.

### Cache structure on Azure

```
<container>/
├── assets/frame_cache/<hash>.pkl          # SD PNG snapshots
├── assets/frame_cache_dd/<hash>.pkl       # DD PNG snapshots
├── assets/uncertainty_cache/<hash>.pkl    # SD model spread
├── assets/uncertainty_cache_dd/<hash>.pkl # DD model spread
└── assets/color_ranges/
    ├── color_ranges.json                  # Change panel colour ranges
    └── abs_color_ranges.json              # Absolute panel colour ranges
```

Cache filenames are MD5 hashes of the full selection key
`(indicator, ssp, baseline, season, model, colorscale, vmin, vmax, log_mode)`,
ensuring exact cache hits with no ambiguity.

---

## Indicator groups

| Group | Indicators |
|-------|-----------|
| **Precipitation** | DD1mm, PR, R99p, R99pVAL, R99pVALWet, RR1mm, RR25mm, Rx1day |
| **Temperature** | FD, TN, TNn, T, TX, TX25, TX30, TXx, DTR |
| **Wind** | sfcwind, Wd10, Wd25, Wd99pVAL, Wx1day |
| **Agroclimatic** | CD18, HD18, GDD5, GDD10, MD15pd, MD15pf, PEDsrad |
| **Record chance** | REC_TXx, REC_TNn, REC_Rx1day, REC_Wx1day |

---

## Environment troubleshooting

| Problem | Fix |
|---------|-----|
| `geopandas` / `fiona` install fails | `conda install -c conda-forge geopandas fiona` |
| `No such file or directory: Home.py` | You are not in the repo root |
| Port already in use | Add `--server.port 8503` (or any free port) |
| Demo banner not appearing | Check `test/demo_data/` exists; re-run `python test/generate_demo_data.py` |
| Uncertainty cache error in demo | Re-run `python test/generate_demo_data.py` |
| Indicator shows "frame cache missing" | Frame cache for that selection has not been pre-computed yet |

---

## Requirements summary

| Package | Min version | Notes |
|---------|-------------|-------|
| Python | 3.11 | 3.9 also tested |
| streamlit | 1.36 | |
| xarray | 2024.1 | NetCDF reading |
| netcdf4 | 1.6 | xarray engine |
| numpy | 1.26 | |
| scipy | 1.13 | Interpolation, spatial smoothing |
| matplotlib | latest | PNG frame rendering, colorbars |
| Pillow | latest | Image encoding |
| geopandas | 0.14 | Coastline / region shapefiles |
| shapely | latest | Geometry operations |
| fiona | latest | Shapefile I/O |

---

## Security notes

- No credentials, tokens, or API keys appear anywhere in the codebase.
- Azure credentials are supplied via environment variables at runtime.
- On-disk pickle files are written only by trusted pre-compute processes;
  the app itself is read-only with respect to the cache directories.
- The sidebar communicates with the map iframe via `postMessage` —
  intentional and same-origin only.