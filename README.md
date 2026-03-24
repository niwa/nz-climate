# NIWA-REMS Climate Dashboard

A two-page Streamlit app for exploring NIWA-REMS ML-Downscaled CMIP6 climate indicators over New Zealand.

---

## Pages

| Page | Description |
|------|-------------|
| **Climate Graph** | Time-series projections of temperature, precipitation, and wind for NZ regions under SSP1-2.6 → SSP5-8.5 |
| **NZ Map** | Interactive gridded map of climate indicators (TX, PR, FD, …) from the NIWA-REMS NetCDF output |

---

## Repository structure

```
climate-dash/
├── Home.py                        # Landing page
├── pages/
│   ├── 1_Climate_Graph.py         # CMIP6 time-series chart
│   └── 2_NZ_Map.py                # Gridded NZ indicator map
├── index_data/                    # ← Place output_v3 contents here
│   ├── historical/
│   │   └── static_maps/
│   │       ├── TX/
│   │       │   └── TX_historical_*.nc
│   │       └── …
│   ├── ssp126/
│   ├── ssp245/
│   ├── ssp370/
│   └── ssp585/
├── climate_data/                  # CMIP6 region-average CSVs (optional)
├── VCSN_data/                     # VCSN observed precipitation CSVs
├── seven_station/                 # 7-station observed temperature CSVs
├── logos/
│   └── esnz_logo_horz_new.png     # (optional) sidebar logo
├── .streamlit/
│   └── config.toml
├── requirements.txt
└── README.md
```

---

## Data setup

### NetCDF indicator maps (NZ Map page)

Copy the entire `output_v3/` directory and rename it `index_data/`:

```bash
cp -r /path/to/output_v3  index_data
```

Expected file name pattern inside each indicator folder:

```
{indicator}_{scenario}_{model}_{ensemble}_base_bp{YYYY-YYYY}_{SEASON}_NZ12km.nc
# e.g.
TX_historical_ACCESS-CM2_r4i1p1f1_base_bp1995-2014_ANN_NZ12km.nc
TX_ssp370_EC-Earth3_r1i1p1f1_base_bp2040-2059_ANN_NZ12km.nc
```

### Climate Graph page

Optional — place CMIP6 region-average CSVs in `climate_data/` with the naming:

```
cmip6_{variable}_{season/month/Annual}_{Region}_{resolution}.csv
```

Columns required: `model`, `experiment`, `year`, `{variable}`.

---

## Running locally

```bash
pip install -r requirements.txt
streamlit run Home.py
```

---

## Deploying to Streamlit Community Cloud

1. Push this repo to GitHub (including `index_data/` — see note below on file size).
2. Connect your repo at [share.streamlit.io](https://share.streamlit.io).
3. Set **Main file path** to `Home.py`.

> **Large data note:** GitHub has a 100 MB per-file limit and a 5 GB repo limit.  
> If your `index_data/` is too large, use [Git LFS](https://git-lfs.com/) or host the
> NetCDF files on an S3/Azure bucket and add a download step in the app.

---

## Dependencies

| Package | Purpose |
|---------|---------|
| `streamlit` | Web app framework |
| `plotly` | Interactive charts and maps |
| `xarray` | NetCDF file reading |
| `netcdf4` | NetCDF backend for xarray |
| `pandas` | Tabular data |
| `numpy` | Numerical arrays |
