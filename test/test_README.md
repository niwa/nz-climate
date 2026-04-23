# Demo data — NZ Climate Dashboard

This folder holds a small synthetic dataset so the app can be reviewed
and demoed without access to NIWA HPC storage.

## Quick start

```bash
# From the repo root — run once, then commit the output
python test/generate_demo_data.py

git add test/demo_data/ assets/color_ranges/
git commit -m "Add synthetic demo dataset"

streamlit run Home.py
```

The app auto-detects demo mode and shows a blue info banner.

---

## What is generated

| Path | Contents |
|------|----------|
| `test/demo_data/historical/static_maps/TX/` | 3 model NC files — TX baseline, bp1995-2014, Annual |
| `test/demo_data/ssp370/static_maps/TX/` | 6 model NC files — 2 future periods × 3 models |
| `test/demo_data/uncertainty_cache/` | Pre-computed model-spread pickle (consumed by the app) |
| `assets/color_ranges/color_ranges.json` | TX half-range entry added if absent |
| `assets/color_ranges/abs_color_ranges.json` | TX vmin/vmax entry added if absent |

The synthetic data uses a smooth north–south temperature gradient with
realistic spatial noise. It is **not** real climate data — it exists solely
to demonstrate the app's UI and animation pipeline.

---

## How demo mode is triggered

`pages/2_NZ_Map.py` checks at startup:

```python
_DEMO_MODE = not DATA_ROOT.exists() and Path("test/demo_data").exists()
```

- `DATA_ROOT` defaults to the NIWA ESI HPC path (overridable via
  `NZMAP_DATA_ROOT` env var).
- If that path is absent **and** `test/demo_data/` exists, demo mode
  activates automatically.  No env var or flag is needed.

In demo mode:
- `DATA_ROOT` and `REC_ROOT` are redirected to `test/demo_data/`.
- The uncertainty cache is read from `test/demo_data/uncertainty_cache/`
  instead of `assets/uncertainty_cache/`.
- A blue info banner is displayed at the top of the map page.

---

## Limitations of the demo dataset

| Feature | Demo behaviour |
|---------|---------------|
| Indicators available | TX only |
| Scenarios available | SSP3-7.0 only (selecting others shows a "no data" message) |
| Baseline period | bp1995-2014 only |
| Season | Annual only |
| Models | 3 synthetic models (DEMO-GCM-A/B/C) |
| Spatial resolution | 0.5° (~50 km), 675 grid points |
| Data values | Synthetic — not real projections |

---

## Regenerating from scratch

If you need to wipe and recreate:

```bash
rm -rf test/demo_data/
python test/generate_demo_data.py
```

The generator is deterministic (fixed random seeds), so the output is
identical across runs on the same Python/NumPy version.
