# pages/1_Climate_Graph.py
import io
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="Climate Graph", layout="wide")

st.markdown("""
<style>
html, body, [class*="css"]  { font-size: 14px !important; }
h1, h2, h3, h4 { font-size: 1.2rem !important; line-height: 1.2 !important; }
header[data-testid="stHeader"] { display: none; }
div.block-container { padding-top: 2.5rem; }
</style>
""", unsafe_allow_html=True)

# ----------------------------
# Config
# ----------------------------
SCEN_COLORS = {
    "ssp126": "#1f77b4",
    "ssp245": "#2ca02c",
    "ssp370": "#ff7f0e",
    "ssp585": "#d62728",
}

OBS_COLORS = {
    "VCSN": "#cc35b8",
    "7-station series": "#2bdada",
}

CMIP6_MODELS = ["ACCESS-CM2", "AWI-CM-1-1-MR", "CNRM-CM6-1", "EC-Earth3", "GFDL-ESM4", "NorESM2-MM"]
HIST_YEARS   = np.arange(1961, 2015)
FUT_YEARS    = np.arange(2015, 2100)
SCENARIOS    = ["ssp126", "ssp245", "ssp370", "ssp585"]
resolution   = "5km_bc"

SEASONS  = ["DJF", "MAM", "JJA", "SON"]
MONTHS   = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
REGIONS  = ["New Zealand", "Auckland", "Otago", "West Coast", "Canterbury", "Tasman", "Wellington"]

OBS_FILES = {
    "Auckland":    "seven_station/Full_TMean-Auckland.csv",
    "Otago":       "seven_station/Full_TMean-Dunedin.csv",
    "West Coast":  "seven_station/Full_TMean-Hokitika.csv",
    "Canterbury":  "seven_station/Full_TMean-Lincoln.csv",
    "Tasman":      "seven_station/Full_TMean-Nelson.csv",
    "Wellington":  "seven_station/Full_TMean-Wellington.csv",
    "New Zealand": "seven_station/Full_TMean-NZT7_TMean_rounded.csv",
}

VAR_META = {
    "tas":     {"label": "Air temperature change (°C)", "units": "°C",     "display": "Air temperature"},
    "pr":      {"label": "Precipitation change (mm)",   "units": "mm",     "display": "Precipitation"},
    "sfcWind": {"label": "Wind speed change (m s⁻¹)",   "units": "m s⁻¹", "display": "Wind speed"},
}

# ----------------------------
# Helpers
# ----------------------------
def read_user_csv(file, variable, freq, month_choice, season_choice):
    try:
        if hasattr(file, "read"):
            text = file.read()
            if isinstance(text, bytes):
                text = text.decode("utf-8", errors="ignore")
            buf = io.StringIO(text)
            df = pd.read_csv(buf)
        else:
            df = pd.read_csv(file)

        lower_cols = {c.lower(): c for c in df.columns}
        needed_base = {"model", "experiment", "year"}
        if not needed_base.issubset(set(map(str.lower, df.columns))):
            st.error("CSV needs columns: model, experiment, year, and a variable column.")
            return None

        var_col = lower_cols.get(variable.lower())
        if var_col is None:
            st.error(f"CSV is missing the '{variable}' column.")
            return None

        df = df.rename(columns={var_col: "tas"})
        for core in ["model", "experiment", "year"]:
            if core not in df.columns:
                df = df.rename(columns={lower_cols[core]: core})

        if "ensemble" not in df.columns:
            df["ensemble"] = "CMIP6"

        df["year"] = pd.to_numeric(df["year"], errors="coerce").astype("Int64")
        df["tas"]  = pd.to_numeric(df["tas"],  errors="coerce")
        df = df.dropna(subset=["year", "tas", "model", "experiment"]).copy()
        df["year"] = df["year"].astype(int)

        if variable == "tas":
            df["tas"] = df["tas"] - 273.15

        if freq == "Monthly":
            df["month"] = month_choice
        if freq == "Seasonal":
            df["season"] = season_choice

        return df.sort_values(["experiment", "model", "year"]).reset_index(drop=True)
    except Exception as e:
        st.error(f"Failed to parse CSV: {e}")
        return None


def load_obs_monthly(path):
    df = pd.read_csv(path, index_col=0)
    df.index.name = "year"
    df = df.apply(pd.to_numeric, errors="coerce")
    m = df.reset_index().melt(id_vars="year", var_name="month_name", value_name="tas")
    name_to_num = {mo: i+1 for i, mo in enumerate(MONTHS)}
    m["month"] = m["month_name"].map(name_to_num)
    m = m.dropna(subset=["tas", "month"])
    m["model"] = "7-station series"
    m["experiment"] = "observed"
    m["ensemble"]   = "obs"
    return m[["ensemble","model","experiment","year","month","tas"]].sort_values(["year","month"])


def obs_select_annual(m_monthly):
    ann = m_monthly.groupby("year")["tas"].mean().reset_index()
    ann[["model","experiment","ensemble"]] = ["7-station series","observed","obs"]
    return ann[["ensemble","model","experiment","year","tas"]]


def obs_select_month(m_monthly, month_num):
    sub = m_monthly[m_monthly["month"] == month_num].copy()
    return sub[["ensemble","model","experiment","year","month","tas"]].sort_values("year")


def obs_select_season(m_monthly, season):
    month_to_season = {
        12:"DJF",1:"DJF",2:"DJF",
        3:"MAM",4:"MAM",5:"MAM",
        6:"JJA",7:"JJA",8:"JJA",
        9:"SON",10:"SON",11:"SON",
    }
    year_adjust = {12:1,1:0,2:0,3:0,4:0,5:0,6:0,7:0,8:0,9:0,10:0,11:0}
    df = m_monthly.copy()
    df["season"]      = df["month"].map(month_to_season)
    df["season_year"] = df["year"] + df["month"].map(year_adjust)
    df = df[df["season"] == season]
    seas = (df.groupby(["season_year","season"])["tas"].mean().reset_index()
              .rename(columns={"season_year":"year"}))
    seas[["model","experiment","ensemble"]] = ["7-station series","observed","obs"]
    return seas[["ensemble","model","experiment","year","season","tas"]].sort_values("year")


def build_vcsn_filename(variable, freq, region, season_choice, month_choice):
    region_tag = region.replace(" ","")
    if freq == "Annual":
        tag = "Annual"
    elif freq == "Seasonal" and season_choice:
        tag = season_choice
    elif freq == "Monthly" and month_choice:
        tag = month_choice
    else:
        tag = "Annual"
    return Path("VCSN_data") / "summary_csv" / f"vcsn_{variable}_{tag}_{region_tag}_5kbc.csv"


def load_vcsn_obs(variable, freq, region, season_choice, month_choice):
    path = build_vcsn_filename(variable, freq, region, season_choice, month_choice)
    if not path.exists():
        return pd.DataFrame()

    df = pd.read_csv(path)
    cols_lower = {c.lower(): c for c in df.columns}
    year_col = cols_lower.get("year")
    if year_col is None:
        return pd.DataFrame()

    if variable in df.columns:
        val_col = variable
    elif "value" in df.columns:
        val_col = "value"
    else:
        non_year = [c for c in df.columns if c != year_col]
        if not non_year:
            return pd.DataFrame()
        val_col = non_year[0]

    out = (df.rename(columns={year_col:"year", val_col:"tas"})
             .loc[:,["year","tas"]])
    out["year"] = pd.to_numeric(out["year"], errors="coerce").astype("Int64")
    out["tas"]  = pd.to_numeric(out["tas"],  errors="coerce")
    out = out.dropna(subset=["year","tas"]).assign(year=lambda x: x["year"].astype(int))

    if freq == "Monthly":
        out["month"]  = MONTHS.index(month_choice) + 1 if month_choice else None
        cols_out = ["ensemble","model","experiment","year","month","tas"]
    elif freq == "Seasonal":
        out["season"] = season_choice
        cols_out = ["ensemble","model","experiment","year","season","tas"]
    else:
        cols_out = ["ensemble","model","experiment","year","tas"]

    out["model"]      = "VCSN"
    out["experiment"] = "observed"
    out["ensemble"]   = "obs"
    return out[cols_out].sort_values("year")


def compute_anomalies(df, base_start, base_end):
    mask = df["year"].between(base_start, base_end) & df["experiment"].isin(["historical","observed"])
    clim = df[mask].groupby("model")["tas"].mean().rename("clim_mean")
    out = df.merge(clim, on="model", how="left")
    out["tas_anom"] = out["tas"] - out["clim_mean"]
    return out


def build_filename(variable, freq, region, resolution, season_choice=None, month_choice=None):
    if freq == "Annual":
        tag = "Annual"
    elif freq == "Seasonal" and season_choice:
        tag = season_choice
    elif freq == "Monthly" and month_choice:
        tag = month_choice
    else:
        tag = "Annual"
    if variable == "sfcWind":
        resolution = "12km_raw"
    return f"climate_data/cmip6_{variable}_{tag}_{region.replace(' ','')}_{resolution}.csv"


def add_label(df, freq):
    if freq == "Seasonal" and "season" in df.columns:
        df["label"] = df["year"].astype(str) + " " + df["season"]
    elif freq == "Monthly" and "month" in df.columns:
        df = df.copy()
        df.loc[df["month"].notna(),"month"] = df.loc[df["month"].notna(),"month"].astype(int)
        df["label"] = df.apply(
            lambda r: f"{r['year']} {MONTHS[int(r['month'])-1]}" if pd.notna(r.get("month")) else str(r["year"]),
            axis=1,
        )
    else:
        df["label"] = df["year"].astype(str)
    return df


def plot_anoms(df, variable, scenarios, show_models=True, value_mode="Relative change", region_choice=""):
    fig = go.Figure()
    units = VAR_META[variable]["units"]
    ycol = "tas_anom" if value_mode.startswith("Relative") else "tas"

    if ycol == "tas_anom":
        fig.add_hline(y=0, line_dash="dot", line_color="black", line_width=1)

    # Historical
    hist = df[df["experiment"] == "historical"]
    if show_models:
        for model, g in hist.groupby("model"):
            fig.add_scatter(
                x=(g["year"]+0.5).tolist(), y=g[ycol].tolist(), mode="lines",
                line=dict(color="silver", width=1.2),
                name=f"{model} (hist)", legendgroup="historical",
                showlegend=False, hoverinfo="skip",
            )

    hist_mean = hist.groupby("year")[ycol].mean().dropna()
    hist_label_map = hist.groupby("year")["label"].first()
    hist_labels = [hist_label_map.get(y, str(y)) for y in hist_mean.index]
    fig.add_scatter(
        x=(hist_mean.index.values+0.5).tolist(), y=hist_mean.values.tolist(), mode="lines",
        line=dict(color="black", width=3),
        name="historical mean", legendgroup="historical",
        customdata=hist_labels,
        hovertemplate=f"Modelled historical mean %{{customdata}}: %{{y:.2f}} {units}<extra></extra>",
    )

    # Scenarios
    for scen in scenarios:
        sub = df[df["experiment"] == scen]
        if sub.empty:
            continue
        if show_models:
            for model, g in sub.groupby("model"):
                fig.add_scatter(
                    x=(g["year"]-0.5).tolist(), y=g[ycol].tolist(), mode="lines",
                    line=dict(color=SCEN_COLORS[scen], width=1.1),
                    opacity=0.3, name=f"{model} ({scen})",
                    legendgroup=scen, showlegend=False, hoverinfo="skip",
                )
        mean_future = sub.groupby("year")[ycol].mean().dropna()
        if not mean_future.empty:
            label_map = sub.groupby("year")["label"].first()
            fut_labels = [label_map.get(y, str(y)) for y in mean_future.index]
            fig.add_scatter(
                x=(mean_future.index.values-0.5).tolist(), y=mean_future.values.tolist(), mode="lines",
                line=dict(color=SCEN_COLORS[scen], width=3),
                name=f"{scen} mean", legendgroup=scen,
                customdata=fut_labels,
                hovertemplate=f"{scen} %{{customdata}}: %{{y:.2f}} {units}<extra></extra>",
            )

    # Observations
    obs = df[(df["experiment"] == "observed") & (df["year"] >= 1961)]
    if not obs.empty:
        for obs_model, g in obs.groupby("model"):
            fig.add_scatter(
                x=(g["year"]+0.5).tolist(), y=g[ycol].tolist(), mode="lines",
                line=dict(width=3, color=OBS_COLORS.get(obs_model, "purple")),
                name=obs_model,
                customdata=g["label"].tolist(),
                hovertemplate=f"{obs_model} %{{customdata}}: %{{y:.2f}} {units}<extra></extra>",
            )

    if month_choice:
        disp = month_choice
    elif season_choice:
        disp = season_choice
    else:
        disp = "Annual"

    if ycol == "tas_anom":
        title_text = f"{VAR_META[variable]['display']} change<br>{disp}<br>{region_choice}"
        y_label    = VAR_META[variable]["label"]
    else:
        title_text = f"{VAR_META[variable]['display']} absolute values<br>{disp}<br>{region_choice}"
        y_label    = f"{VAR_META[variable]['display']} ({units})"

    fig.update_layout(
        title=dict(text=title_text, x=0.05, xanchor="left", y=0.95, yanchor="top", font=dict(size=18)),
        xaxis_title="Year",
        yaxis_title=y_label,
        height=700,
        font=dict(size=14),
        hoverlabel=dict(font_size=12, font_family="Arial"),
        xaxis=dict(showspikes=True, spikecolor="black", spikethickness=1),
        yaxis=dict(showspikes=True, spikecolor="black", spikethickness=1),
        hovermode="x unified", hoverdistance=1, spikedistance=-1,
        legend=dict(
            orientation="h", yanchor="top", y=-0.2, xanchor="center", x=0.5,
            font=dict(size=16), traceorder="normal", valign="top",
            itemsizing="constant", title=None,
        ),
    )
    fig.update_xaxes(
        tickfont=dict(size=16), title_font=dict(size=16),
        range=[1961, 2099], dtick=10,
        showline=True, linewidth=0.5, linecolor="black",
        hoverformat=".0f",
    )
    fig.update_yaxes(
        tickfont=dict(size=16), title_font=dict(size=16),
        showline=True, linewidth=0.5, linecolor="black", mirror=True,
    )
    return fig


# ----------------------------
# Sidebar
# ----------------------------
with st.sidebar:
    logo_path = Path("logos/esnz_logo_horz_new.png")
    if logo_path.exists():
        st.image(str(logo_path))

    st.subheader("Values to display:")
    value_mode = st.radio(
        "", ["Relative change", "Absolute values"], index=0,
        label_visibility="collapsed",
        help="Choose relative change (anomaly) or absolute values.",
    )

    st.subheader("Variable:")
    variable_display = st.selectbox(
        "", [VAR_META[v]["display"] for v in VAR_META], index=0, label_visibility="collapsed"
    )
    variable = [k for k, v in VAR_META.items() if v["display"] == variable_display][0]

    st.subheader("Time resolution:")
    freq = st.radio("", ["Annual", "Seasonal", "Monthly"], index=0, label_visibility="collapsed")
    season_choice, month_choice = None, None
    if freq == "Seasonal":
        season_choice = st.selectbox("Season:", SEASONS, index=0)
    elif freq == "Monthly":
        month_choice = st.selectbox("Month:", MONTHS, index=0)

    st.subheader("Baseline period:")
    base_start, base_end = st.slider(
        "", 1961, 2014, (1995, 2014), step=1,
        label_visibility="collapsed",
        help="Historical baseline for relative change. At least 20 years recommended.",
    )

    st.subheader("Region:")
    region_choice = st.selectbox("", REGIONS, index=0, label_visibility="collapsed")

    st.subheader("Scenarios:")
    SCEN_ORDER = st.multiselect(
        "", SCENARIOS, default=SCENARIOS, label_visibility="collapsed"
    )

    st.subheader("Individual models:")
    if "selected_models" not in st.session_state:
        st.session_state.selected_models = CMIP6_MODELS

    col1, col2 = st.columns(2)
    with col1:
        if st.button("Select all"):
            st.session_state.selected_models = CMIP6_MODELS
    with col2:
        if st.button("Clear all"):
            st.session_state.selected_models = []

    MODEL_ORDER = st.multiselect(
        "", CMIP6_MODELS, default=st.session_state.selected_models,
        key="selected_models", label_visibility="collapsed",
    )

# ----------------------------
# Load data
# ----------------------------
fname = build_filename(variable, freq, region_choice, resolution, season_choice, month_choice)

if Path(fname).exists():
    mnum = MONTHS.index(month_choice) + 1 if month_choice else 1
    df_models = read_user_csv(fname, variable, freq, mnum, season_choice)
else:
    st.info(
        f"Model data file not found: `{fname}`.  "
        "Place your CMIP6 summary CSVs in the `climate_data/` folder to enable model projections."
    )
    df_models = pd.DataFrame(columns=["ensemble","model","experiment","year","tas"])

if df_models is not None and not df_models.empty:
    df_models = df_models[df_models["model"].isin(MODEL_ORDER)]

# Observations
obs_frames = []

if variable == "tas" and region_choice in OBS_FILES and Path(OBS_FILES[region_choice]).exists():
    obs_monthly = load_obs_monthly(OBS_FILES[region_choice])
    if freq == "Annual":
        df_obs7 = obs_select_annual(obs_monthly)
    elif freq == "Seasonal":
        df_obs7 = obs_select_season(obs_monthly, season_choice or "DJF")
    else:
        mnum = MONTHS.index(month_choice) + 1 if month_choice else 1
        df_obs7 = obs_select_month(obs_monthly, mnum)
    obs_frames.append(df_obs7[df_obs7["year"] >= 1961])

df_vcsn = load_vcsn_obs(variable, freq, region_choice, season_choice, month_choice)
if not df_vcsn.empty:
    obs_frames.append(df_vcsn[df_vcsn["year"] >= 1961])

df_obs_all = pd.concat(obs_frames, ignore_index=True) if obs_frames else pd.DataFrame(
    columns=["ensemble","model","experiment","year","tas"]
)

df_raw  = pd.concat([df_models, df_obs_all], ignore_index=True) if df_models is not None else df_obs_all
df_anom = compute_anomalies(df_raw, base_start, base_end)
df_anom = add_label(df_anom, freq)
df_anom["label"] = df_anom["label"] + f" ({region_choice})"

# ----------------------------
# Layout
# ----------------------------
col_plot, col_space, col_box = st.columns([8, 0.1, 4.2], vertical_alignment="top")

show_models = True
with col_box:
    with st.expander("What does this graph show?"):
        if value_mode.startswith("Relative"):
            st.markdown(
                f"This graph shows **{VAR_META[variable]['display'].lower()}** change "
                f"in **{region_choice}** relative to the **{base_start}–{base_end}** baseline."
            )
        else:
            st.markdown(
                f"Absolute **{VAR_META[variable]['display'].lower()}** values for **{region_choice}**."
            )

    with st.expander("Why so many lines?"):
        st.markdown(
            "Projections include **6 CMIP6 models** shown as light lines. "
            "The thick line is the **multi-model mean** per scenario."
        )
        show_models = st.checkbox("Show individual model lines", value=True)

    with st.expander("What are emission scenarios (SSPs)?"):
        st.markdown(
            "- **SSP126**: Low emissions – strong mitigation  \n"
            "- **SSP245**: Moderate emissions  \n"
            "- **SSP370**: High emissions  \n"
            "- **SSP585**: Very high – fossil-fuel intensive"
        )

    if variable == "tas":
        with st.expander("What is the 7-station series?"):
            st.markdown(
                "An **observed** NZ air temperature dataset. "
                "Region-specific station data is shown when available."
            )

with col_plot:
    fig = plot_anoms(
        df_anom, variable, SCEN_ORDER,
        show_models=show_models,
        value_mode="Relative change" if value_mode.startswith("Relative") else "Absolute values",
        region_choice=region_choice,
    )
    st.plotly_chart(fig, use_container_width=True)
