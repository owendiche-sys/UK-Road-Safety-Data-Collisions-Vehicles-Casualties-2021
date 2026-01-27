from __future__ import annotations

import io
import os
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st


# =========================
# Page config
# =========================
st.set_page_config(
    page_title="UK Road Accidents (2021) Dashboard",
    layout="wide",
    initial_sidebar_state="expanded",
)

# =========================
# Styling 
# =========================
APP_CSS = """
<style>
:root{
  --bg:#ffffff;
  --panel:#ffffff;
  --muted:#6b7280;
  --text:#111827;
  --border:rgba(17,24,39,.12);
  --shadow:0 1px 2px rgba(16,24,40,.06), 0 1px 3px rgba(16,24,40,.10);
  --radius:16px;
}

html, body, [class*="css"]  { background: var(--bg); color: var(--text); }
section[data-testid="stSidebar"] { background: #fbfbfc; border-right: 1px solid var(--border); }
.block-container { padding-top: 1.2rem; padding-bottom: 2rem; }

/* Style bordered containers (used as cards) */
div[data-testid="stContainer"]{
  border-radius: var(--radius) !important;
}

/* Make the inner bordered container look like a card */
div[data-testid="stContainer"] > div{
  background: var(--panel);
  border: 1px solid var(--border);
  box-shadow: var(--shadow);
  border-radius: var(--radius);
  padding: 14px 14px 12px 14px;
}

/* headings */
h1, h2, h3 { letter-spacing: -0.02em; }

.small-muted{
  font-size: 0.85rem;
  color: var(--muted);
}

.hr {
  height: 1px;
  background: var(--border);
  margin: 12px 0 12px 0;
}

.kpi-label{
  font-size: 0.85rem;
  font-weight: 800;
  color: #374151;
  margin-bottom: 2px;
}
.kpi-value{
  font-size: 1.8rem;
  font-weight: 900;
  color: #111827;
  line-height: 1.1;
}
.kpi-sub{
  font-size: 0.82rem;
  color: var(--muted);
  margin-top: 6px;
}

.stTabs [data-baseweb="tab-list"] { gap: 8px; }
.stTabs [data-baseweb="tab"]{
  height: 38px;
  background: #f9fafb;
  border: 1px solid var(--border);
  border-radius: 999px;
  padding: 0 14px;
  font-weight: 800;
}
.stTabs [aria-selected="true"]{
  background: #eef2ff;
  border-color: rgba(99,102,241,.35);
}
</style>
"""
st.markdown(APP_CSS, unsafe_allow_html=True)


# =========================
# Helpers
# =========================
@contextmanager
def card() -> None:
    """Bordered container styled via CSS (no HTML open/close tags, so no stray </div>)."""
    try:
        c = st.container(border=True)
    except TypeError:
        # Older Streamlit fallback (shouldn't happen with >=1.31)
        c = st.container()
    with c:
        yield


def hr() -> None:
    st.markdown("<div class='hr'></div>", unsafe_allow_html=True)


def fmt_int(x: float | int | None) -> str:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "NA"
    return f"{int(x):,}"


def fmt_float(x: float | int | None, d: int = 2) -> str:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "NA"
    return f"{float(x):,.{d}f}"


def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    return df


def find_col(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    cols_lower = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in cols_lower:
            return cols_lower[cand.lower()]
    return None


def coerce_datetime(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, errors="coerce", dayfirst=True, infer_datetime_format=True)


def as_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def top_n_value_counts(series: pd.Series, n: int = 10) -> pd.DataFrame:
    vc = series.astype("string").fillna("Missing").value_counts().head(n)
    out = vc.reset_index()
    out.columns = ["Category", "Count"]
    out["Share (%)"] = (out["Count"] / max(out["Count"].sum(), 1) * 100).round(2)
    return out


def _try_read_csv_bytes(file_bytes: bytes) -> pd.DataFrame:
    """
    Read CSV bytes with encoding fallbacks.
    - Default (C engine) supports low_memory=False
    - Python engine fallback does not
    """
    encodings = ["utf-8", "utf-8-sig", "cp1252", "latin1"]
    last_err: Optional[Exception] = None

    for enc in encodings:
        try:
            return pd.read_csv(io.BytesIO(file_bytes), encoding=enc, low_memory=False)
        except Exception as e:
            last_err = e

        try:
            return pd.read_csv(io.BytesIO(file_bytes), encoding=enc, engine="python")
        except Exception as e:
            last_err = e

    raise RuntimeError(f"Unable to read CSV with common encodings. Last error: {last_err}")


def safe_read_csv_path(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")
    with open(path, "rb") as f:
        b = f.read()
    return _try_read_csv_bytes(b)


@dataclass
class DataBundle:
    collisions: pd.DataFrame
    vehicles: Optional[pd.DataFrame]
    casualties: Optional[pd.DataFrame]
    merged: pd.DataFrame
    notes: List[str]


def _prepare_collisions(collisions: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    notes: List[str] = []
    df = standardize_columns(collisions)

    key = find_col(df, ["Accident_Index", "accident_index", "collision_index"])
    if key is None:
        notes.append("Key column not found. Using row index as Accident_Index.")
        df["Accident_Index"] = np.arange(len(df)).astype(str)
    else:
        if key != "Accident_Index":
            df = df.rename(columns={key: "Accident_Index"})

    date_col = find_col(df, ["Date", "Accident_Date", "accident_date"])
    if date_col is not None:
        df[date_col] = coerce_datetime(df[date_col])
        bad = int(df[date_col].isna().sum())
        if bad > 0:
            notes.append(f"Date parsing produced {bad:,} missing values.")
        df = df.rename(columns={date_col: "Date"})
    else:
        notes.append("No date column found. Time-based charts may be limited.")
        df["Date"] = pd.NaT

    time_col = find_col(df, ["Time", "time"])
    if time_col and time_col != "Time":
        df = df.rename(columns={time_col: "Time"})

    lat_col = find_col(df, ["Latitude", "latitude"])
    lon_col = find_col(df, ["Longitude", "longitude", "long"])
    if lat_col and lat_col != "Latitude":
        df = df.rename(columns={lat_col: "Latitude"})
    if lon_col and lon_col != "Longitude":
        df = df.rename(columns={lon_col: "Longitude"})

    if "Date" in df.columns and pd.api.types.is_datetime64_any_dtype(df["Date"]):
        df["Year"] = df["Date"].dt.year
        df["Month"] = df["Date"].dt.month
        df["MonthName"] = df["Date"].dt.month_name()
        df["DayOfWeek"] = df["Date"].dt.day_name()

    return df, notes


def _aggregate_by_accident(df: pd.DataFrame, prefix: str) -> pd.DataFrame:
    df = standardize_columns(df)
    key = find_col(df, ["Accident_Index", "accident_index", "collision_index"])
    if key is None:
        raise ValueError(f"{prefix}: key column not found (Accident_Index / collision_index).")
    if key != "Accident_Index":
        df = df.rename(columns={key: "Accident_Index"})

    agg = df.groupby("Accident_Index").size().reset_index(name=f"{prefix}_count")
    return agg


@st.cache_data(show_spinner=False)
def load_default_bundle() -> DataBundle:
    notes: List[str] = []

    candidates = [
        ("data/collisions.csv", "data/vehicles.csv", "data/casualties.csv"),
        ("collisions.csv", "vehicles.csv", "casualties.csv"),
    ]

    collisions = vehicles = casualties = None

    for c_path, v_path, ca_path in candidates:
        if os.path.exists(c_path) and os.path.exists(v_path) and os.path.exists(ca_path):
            collisions = safe_read_csv_path(c_path)
            vehicles = safe_read_csv_path(v_path)
            casualties = safe_read_csv_path(ca_path)
            notes.append(f"Loaded default files: {c_path}, {v_path}, {ca_path}")
            break

    if collisions is None:
        raise FileNotFoundError(
            "Default files not found. Expected: data/collisions.csv, data/vehicles.csv, data/casualties.csv "
            "(or the same three files next to app.py)."
        )

    collisions_p, notes_c = _prepare_collisions(collisions)
    notes.extend(notes_c)

    merged = collisions_p.copy()

    # Attach aggregation counts (safe if keys exist)
    try:
        v_agg = _aggregate_by_accident(vehicles, prefix="vehicles")
        merged = merged.merge(v_agg, on="Accident_Index", how="left")
    except Exception as e:
        notes.append(f"Vehicles aggregation skipped: {e}")

    try:
        ca_agg = _aggregate_by_accident(casualties, prefix="casualties")
        merged = merged.merge(ca_agg, on="Accident_Index", how="left")
    except Exception as e:
        notes.append(f"Casualties aggregation skipped: {e}")

    return DataBundle(
        collisions=collisions_p,
        vehicles=standardize_columns(vehicles) if vehicles is not None else None,
        casualties=standardize_columns(casualties) if casualties is not None else None,
        merged=merged,
        notes=notes,
    )


@st.cache_data(show_spinner=False)
def load_uploaded_csv(uploaded_file) -> DataBundle:
    b = uploaded_file.read()
    df = standardize_columns(_try_read_csv_bytes(b))

    notes: List[str] = [f"Loaded uploaded dataset: {uploaded_file.name}"]

    key = find_col(df, ["Accident_Index", "accident_index", "collision_index"])
    if key is None:
        df["Accident_Index"] = np.arange(len(df)).astype(str)
        notes.append("Key column not found. Using row index as Accident_Index.")
    else:
        if key != "Accident_Index":
            df = df.rename(columns={key: "Accident_Index"})

    date_col = find_col(df, ["Date", "Accident_Date", "accident_date"])
    if date_col is not None:
        df[date_col] = coerce_datetime(df[date_col])
        df = df.rename(columns={date_col: "Date"})
        df["Year"] = df["Date"].dt.year
        df["Month"] = df["Date"].dt.month
        df["MonthName"] = df["Date"].dt.month_name()
        df["DayOfWeek"] = df["Date"].dt.day_name()

    return DataBundle(
        collisions=pd.DataFrame(),
        vehicles=None,
        casualties=None,
        merged=df,
        notes=notes,
    )


def safe_filter_frame(df: pd.DataFrame, filters: Dict) -> pd.DataFrame:
    out = df.copy()

    # Date range
    if "Date" in out.columns and pd.api.types.is_datetime64_any_dtype(out["Date"]):
        dmin, dmax = filters.get("date_range", (None, None))
        if dmin is not None and dmax is not None:
            out = out[(out["Date"] >= pd.to_datetime(dmin)) & (out["Date"] <= pd.to_datetime(dmax))]

    # District
    district_col = find_col(out, ["Local_Authority_District", "local_authority_district", "District"])
    if district_col and filters.get("district") and filters["district"] != "All":
        out = out[out[district_col].astype("string") == filters["district"]]

    # Road type
    road_col = find_col(out, ["Road_Type", "road_type"])
    if road_col and filters.get("road_type") and filters["road_type"] != "All":
        out = out[out[road_col].astype("string") == filters["road_type"]]

    # Speed limit
    speed_col = find_col(out, ["Speed_limit", "Speed_Limit", "speed_limit"])
    if speed_col and filters.get("speed_limit") and filters["speed_limit"] != "All":
        out = out[out[speed_col].astype("string") == str(filters["speed_limit"])]

    # Optional extra filters
    for key in ["weather", "light", "surface"]:
        colname = filters.get(f"{key}_col")
        choice = filters.get(key)
        if colname and choice and choice != "All" and colname in out.columns:
            out = out[out[colname].astype("string") == choice]

    return out


# =========================
# Sidebar (order: navigation -> data source -> filters -> data panel)
# =========================
with st.sidebar:
    st.markdown("### Navigation")
    page = st.radio(
        "Go to",
        ["Summary", "Exploration", "Trends", "Locations", "Insights", "Data"],
        label_visibility="collapsed",
    )

    st.markdown("### Data source")
    use_upload = st.toggle("Upload a CSV instead of the default files", value=False)

    bundle: Optional[DataBundle] = None
    load_error: Optional[str] = None

    try:
        if use_upload:
            up = st.file_uploader("Upload CSV", type=["csv"])
            if up is None:
                st.info("Upload a CSV to continue.")
            else:
                bundle = load_uploaded_csv(up)
        else:
            bundle = load_default_bundle()
    except Exception as e:
        load_error = str(e)

    st.markdown("### Filters")
    filters: Dict = {}

    if bundle is not None:
        df_base = standardize_columns(bundle.merged)

        # Date range
        if "Date" in df_base.columns and pd.api.types.is_datetime64_any_dtype(df_base["Date"]):
            dmin = df_base["Date"].min()
            dmax = df_base["Date"].max()
            if pd.notna(dmin) and pd.notna(dmax):
                dr = st.date_input(
                    "Date range",
                    value=(dmin.date(), dmax.date()),
                    min_value=dmin.date(),
                    max_value=dmax.date(),
                )
                if isinstance(dr, tuple) and len(dr) == 2:
                    filters["date_range"] = dr

        # District
        district_col = find_col(df_base, ["Local_Authority_District", "local_authority_district", "District"])
        if district_col:
            vals = df_base[district_col].dropna().astype("string")
            top = vals.value_counts().head(50).index.tolist()
            filters["district"] = st.selectbox("District (top 50)", ["All"] + top, index=0)

        # Road type
        road_col = find_col(df_base, ["Road_Type", "road_type"])
        if road_col:
            opts = ["All"] + sorted(df_base[road_col].dropna().astype("string").unique().tolist())
            filters["road_type"] = st.selectbox("Road type", opts, index=0)

        # Speed limit
        speed_col = find_col(df_base, ["Speed_limit", "Speed_Limit", "speed_limit"])
        if speed_col:
            svals = pd.to_numeric(df_base[speed_col], errors="coerce").dropna().astype(int)
            opts = ["All"] + sorted(svals.unique().tolist())
            filters["speed_limit"] = st.selectbox("Speed limit", opts, index=0)

        # Optional extra filters (only if columns exist)
        weather_col = find_col(df_base, ["Weather_Conditions", "weather_conditions", "Weather"])
        light_col = find_col(df_base, ["Light_Conditions", "light_conditions", "Light"])
        surface_col = find_col(df_base, ["Road_Surface_Conditions", "road_surface_conditions", "Road_Surface"])

        if weather_col:
            wopts = ["All"] + sorted(df_base[weather_col].dropna().astype("string").unique().tolist())
            filters["weather_col"] = weather_col
            filters["weather"] = st.selectbox("Weather", wopts, index=0)

        if light_col:
            lopts = ["All"] + sorted(df_base[light_col].dropna().astype("string").unique().tolist())
            filters["light_col"] = light_col
            filters["light"] = st.selectbox("Light conditions", lopts, index=0)

        if surface_col:
            sopts = ["All"] + sorted(df_base[surface_col].dropna().astype("string").unique().tolist())
            filters["surface_col"] = surface_col
            filters["surface"] = st.selectbox("Road surface", sopts, index=0)

    st.markdown("### Data panel")
    show_data_sidebar = st.checkbox("Show filtered preview", value=False)


# =========================
# Main: load guard + filtered data
# =========================
if load_error:
    st.error(load_error)
    st.stop()

if bundle is None:
    st.stop()

df = standardize_columns(bundle.merged)
df_f = safe_filter_frame(df, filters)

if show_data_sidebar:
    st.sidebar.dataframe(df_f.head(20), use_container_width=True)


# =========================
# Header
# =========================
st.title("UK Road Accidents (2021) Dashboard")
st.caption("Accident-level analytics with optional vehicle and casualty aggregation when available.")

if bundle.notes:
    with st.expander("Data loading notes", expanded=False):
        st.write("\n".join([f"- {n}" for n in bundle.notes]))


# =========================
# Pages
# =========================
def render_kpis(df_all: pd.DataFrame, df_filtered: pd.DataFrame) -> None:
    total_acc = len(df_all)
    total_acc_f = len(df_filtered)

    date_range = "NA"
    if "Date" in df_all.columns and pd.api.types.is_datetime64_any_dtype(df_all["Date"]):
        dmin = df_all["Date"].min()
        dmax = df_all["Date"].max()
        if pd.notna(dmin) and pd.notna(dmax):
            date_range = f"{dmin.date()} to {dmax.date()}"

    veh_count_col = find_col(df_all, ["vehicles_count"])
    cas_count_col = find_col(df_all, ["casualties_count"])

    veh_total = int(df_all[veh_count_col].fillna(0).sum()) if veh_count_col else None
    cas_total = int(df_all[cas_count_col].fillna(0).sum()) if cas_count_col else None
    avg_veh = float(df_all[veh_count_col].mean()) if veh_count_col else None
    avg_cas = float(df_all[cas_count_col].mean()) if cas_count_col else None

    cols = st.columns(4, gap="large")
    with cols[0]:
        with card():
            st.markdown("<div class='kpi-label'>Accidents (all data)</div>", unsafe_allow_html=True)
            st.markdown(f"<div class='kpi-value'>{fmt_int(total_acc)}</div>", unsafe_allow_html=True)
            st.markdown(f"<div class='kpi-sub'>Date coverage: {date_range}</div>", unsafe_allow_html=True)
    with cols[1]:
        with card():
            st.markdown("<div class='kpi-label'>Accidents (filtered)</div>", unsafe_allow_html=True)
            st.markdown(f"<div class='kpi-value'>{fmt_int(total_acc_f)}</div>", unsafe_allow_html=True)
            st.markdown("<div class='kpi-sub'>Based on current sidebar filters</div>", unsafe_allow_html=True)
    with cols[2]:
        with card():
            st.markdown("<div class='kpi-label'>Vehicles (aggregated)</div>", unsafe_allow_html=True)
            st.markdown(f"<div class='kpi-value'>{fmt_int(veh_total)}</div>", unsafe_allow_html=True)
            st.markdown(
                f"<div class='kpi-sub'>Average per accident: {fmt_float(avg_veh, 2)}</div>",
                unsafe_allow_html=True,
            )
    with cols[3]:
        with card():
            st.markdown("<div class='kpi-label'>Casualties (aggregated)</div>", unsafe_allow_html=True)
            st.markdown(f"<div class='kpi-value'>{fmt_int(cas_total)}</div>", unsafe_allow_html=True)
            st.markdown(
                f"<div class='kpi-sub'>Average per accident: {fmt_float(avg_cas, 2)}</div>",
                unsafe_allow_html=True,
            )


def render_summary_page(df_all: pd.DataFrame, df_filtered: pd.DataFrame) -> None:
    # KPI cards ONLY on Summary page
    render_kpis(df_all, df_filtered)
    hr()

    c1, c2, c3 = st.columns([1.25, 1.0, 1.0], gap="large")

    with c1:
        with card():
            st.subheader("Overview")
            st.markdown(
                "<div class='small-muted'>Use the sidebar to filter by date, district, road type, speed limit and conditions "
                "(when available). All charts and tables update instantly.</div>",
                unsafe_allow_html=True,
            )
            hr()

            bullets: List[str] = []
            if "Date" in df_filtered.columns and pd.api.types.is_datetime64_any_dtype(df_filtered["Date"]):
                dfx = df_filtered.dropna(subset=["Date"]).copy()
                if not dfx.empty:
                    month_counts = dfx["Date"].dt.to_period("M").value_counts().sort_index()
                    if len(month_counts) > 0:
                        bullets.append(
                            f"Peak month (filtered): {str(month_counts.idxmax())} with {fmt_int(month_counts.max())} accidents."
                        )

                    dow_counts = dfx["Date"].dt.day_name().value_counts()
                    if len(dow_counts) > 0:
                        bullets.append(
                            f"Highest day of week (filtered): {dow_counts.idxmax()} with {fmt_int(dow_counts.max())} accidents."
                        )

                    weekend_share = float((dfx["Date"].dt.dayofweek >= 5).mean()) * 100
                    bullets.append(f"Weekend share (filtered): {weekend_share:.1f}% of accidents.")

            district_col = find_col(df_filtered, ["Local_Authority_District", "local_authority_district", "District"])
            if district_col:
                vc = df_filtered[district_col].astype("string").value_counts()
                if len(vc) > 0:
                    bullets.append(f"Top district (filtered): {vc.idxmax()} with {fmt_int(vc.max())} accidents.")

            speed_col = find_col(df_filtered, ["Speed_limit", "Speed_Limit", "speed_limit"])
            if speed_col:
                svals = pd.to_numeric(df_filtered[speed_col], errors="coerce").dropna().astype(int)
                if len(svals) > 0:
                    bullets.append(f"Most common speed limit (filtered): {int(svals.value_counts().idxmax())}.")

            cas_col = find_col(df_filtered, ["casualties_count"])
            if cas_col:
                cas = pd.to_numeric(df_filtered[cas_col], errors="coerce").fillna(0)
                bullets.append(f"Total casualties (filtered): {fmt_int(int(cas.sum()))}.")

            for b in bullets[:7]:
                st.markdown(f"- {b}")

    with c2:
        with card():
            st.subheader("Road type mix (filtered)")
            road_col = find_col(df_filtered, ["Road_Type", "road_type"])
            if road_col:
                st.dataframe(top_n_value_counts(df_filtered[road_col], n=12), use_container_width=True, height=320)
            else:
                st.info("Road type column not found in the dataset.")

    with c3:
        with card():
            st.subheader("Speed limit mix (filtered)")
            speed_col = find_col(df_filtered, ["Speed_limit", "Speed_Limit", "speed_limit"])
            if speed_col:
                s = pd.to_numeric(df_filtered[speed_col], errors="coerce")
                tbl = top_n_value_counts(s.astype("Int64").astype("string"), n=12)
                st.dataframe(tbl, use_container_width=True, height=320)
            else:
                st.info("Speed limit column not found in the dataset.")

    # Extra: "high impact" table (simple ranking; no ML)
    cas_col = find_col(df_filtered, ["casualties_count"])
    veh_col = find_col(df_filtered, ["vehicles_count"])
    if cas_col or veh_col:
        hr()
        with card():
            st.subheader("Highest-impact accidents (filtered)")
            st.markdown(
                "<div class='small-muted'>Ranked by casualties first (when available), then vehicles as a secondary signal.</div>",
                unsafe_allow_html=True,
            )

            cols_keep: List[str] = []
            for c in ["Accident_Index", "Date"]:
                if c in df_filtered.columns:
                    cols_keep.append(c)

            for cand in [
                ["Local_Authority_District", "local_authority_district", "District"],
                ["Road_Type", "road_type"],
                ["Speed_limit", "Speed_Limit", "speed_limit"],
            ]:
                col = find_col(df_filtered, cand)
                if col and col not in cols_keep:
                    cols_keep.append(col)

            if veh_col and veh_col not in cols_keep:
                cols_keep.append(veh_col)
            if cas_col and cas_col not in cols_keep:
                cols_keep.append(cas_col)

            view = df_filtered[cols_keep].copy() if cols_keep else df_filtered.copy()
            view["_cas"] = pd.to_numeric(df_filtered[cas_col], errors="coerce").fillna(0).astype(float) if cas_col else 0.0
            view["_veh"] = pd.to_numeric(df_filtered[veh_col], errors="coerce").fillna(0).astype(float) if veh_col else 0.0
            view = view.sort_values(["_cas", "_veh"], ascending=False).drop(columns=["_cas", "_veh"])

            st.dataframe(view.head(25), use_container_width=True, height=420)


def render_exploration_page(df_filtered: pd.DataFrame) -> None:
    with card():
        st.subheader("Dataset overview (filtered)")
        c1, c2, c3 = st.columns(3, gap="large")
        with c1:
            st.metric("Rows", fmt_int(len(df_filtered)))
        with c2:
            st.metric("Columns", fmt_int(df_filtered.shape[1]))
        with c3:
            st.metric("Missing cells", fmt_int(int(df_filtered.isna().sum().sum())))

    tab1, tab2, tab3 = st.tabs(["Preview", "Missingness", "Column profiler"])

    with tab1:
        with card():
            st.subheader("Preview")
            st.dataframe(df_filtered.head(80), use_container_width=True, height=560)

    with tab2:
        miss_tbl = (
            df_filtered.isna().mean()
            .sort_values(ascending=False)
            .reset_index()
            .rename(columns={"index": "Column", 0: "MissingRate"})
        )
        miss_tbl["MissingRate"] = (miss_tbl["MissingRate"] * 100).round(2)
        miss_tbl = miss_tbl[miss_tbl["MissingRate"] > 0]

        with card():
            st.subheader("Missingness (% of rows)")
            if len(miss_tbl) == 0:
                st.success("No missing values detected in the filtered dataset.")
            else:
                st.dataframe(miss_tbl, use_container_width=True, height=560)

    with tab3:
        col = st.selectbox("Select a column", df_filtered.columns.tolist())
        s = df_filtered[col]

        with card():
            st.subheader(f"Column profile: {col}")
            st.write(
                {
                    "dtype": str(s.dtype),
                    "non_null_rows": int(s.notna().sum()),
                    "unique_values": int(s.nunique(dropna=True)),
                }
            )
            if pd.api.types.is_numeric_dtype(s):
                sn = pd.to_numeric(s, errors="coerce")
                st.write(
                    {
                        "min": float(np.nanmin(sn.values)) if sn.notna().any() else None,
                        "median": float(np.nanmedian(sn.values)) if sn.notna().any() else None,
                        "mean": float(np.nanmean(sn.values)) if sn.notna().any() else None,
                        "max": float(np.nanmax(sn.values)) if sn.notna().any() else None,
                    }
                )
                st.line_chart(sn.dropna().reset_index(drop=True))
            else:
                st.dataframe(top_n_value_counts(s, n=15), use_container_width=True, height=560)


def render_trends_page(df_filtered: pd.DataFrame) -> None:
    with card():
        st.subheader("Time trends (filtered)")
        st.markdown(
            "<div class='small-muted'>Monthly, weekly and hourly patterns based on available date/time fields.</div>",
            unsafe_allow_html=True,
        )

    if "Date" not in df_filtered.columns or not pd.api.types.is_datetime64_any_dtype(df_filtered["Date"]):
        st.info("A valid date column is not available.")
        return

    df_t = df_filtered.dropna(subset=["Date"]).copy()
    if df_t.empty:
        st.info("No rows remain after removing invalid dates.")
        return

    month = df_t.groupby(df_t["Date"].dt.to_period("M")).size().reset_index(name="Accidents")
    month["Month"] = month["Date"].astype(str)
    month = month.sort_values("Month")

    order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    dow_counts = df_t["Date"].dt.day_name().value_counts().reindex(order).fillna(0).astype(int)
    dow_tbl = dow_counts.reset_index()
    dow_tbl.columns = ["DayOfWeek", "Accidents"]

    c1, c2 = st.columns([1.2, 1.0], gap="large")
    with c1:
        with card():
            st.subheader("Monthly accidents")
            st.line_chart(month.set_index("Month")["Accidents"])
    with c2:
        with card():
            st.subheader("Accidents by day of week")
            st.bar_chart(dow_tbl.set_index("DayOfWeek")["Accidents"])

    time_col = find_col(df_t, ["Time", "time"])
    if time_col:
        tt = df_t[time_col].astype("string").str.strip()
        hh = pd.to_numeric(tt.str.split(":").str[0], errors="coerce")
        if hh.notna().any():
            hour_tbl = hh.dropna().astype(int).clip(0, 23).value_counts().sort_index().reset_index()
            hour_tbl.columns = ["Hour", "Accidents"]
            with card():
                st.subheader("Accidents by hour")
                st.line_chart(hour_tbl.set_index("Hour")["Accidents"])


def render_locations_page(df_filtered: pd.DataFrame) -> None:
    with card():
        st.subheader("Location view (filtered)")
        st.markdown("<div class='small-muted'>Map requires Latitude and Longitude columns.</div>", unsafe_allow_html=True)

    lat_col = find_col(df_filtered, ["Latitude", "latitude"])
    lon_col = find_col(df_filtered, ["Longitude", "longitude", "long"])

    if not lat_col or not lon_col:
        st.info("Latitude/Longitude not found.")
        return

    loc = df_filtered.copy()
    loc[lat_col] = as_numeric(loc[lat_col])
    loc[lon_col] = as_numeric(loc[lon_col])
    loc = loc.dropna(subset=[lat_col, lon_col])

    if loc.empty:
        st.info("No valid coordinates available after cleaning.")
        return

    max_points = 12000
    if len(loc) > max_points:
        loc = loc.sample(max_points, random_state=42)

    with card():
        st.subheader("Accident map (sampled when large)")
        st.map(loc.rename(columns={lat_col: "lat", lon_col: "lon"})[["lat", "lon"]])

    district_col = find_col(df_filtered, ["Local_Authority_District", "local_authority_district", "District"])
    if district_col:
        with card():
            st.subheader("Top districts by accidents (filtered)")
            st.dataframe(top_n_value_counts(df_filtered[district_col], n=15), use_container_width=True, height=520)


def render_insights_page(df_all: pd.DataFrame, df_filtered: pd.DataFrame) -> None:
    with card():
        st.subheader("Insights")
        st.markdown(
            "<div class='small-muted'>Observed patterns and descriptive statistics for the filtered view.</div>",
            unsafe_allow_html=True,
        )

    with card():
        st.markdown("### Data-driven insights")

        bullets: List[str] = []
        bullets.append(f"Filtered accidents: {fmt_int(len(df_filtered))} (out of {fmt_int(len(df_all))}).")

        if "Date" in df_filtered.columns and pd.api.types.is_datetime64_any_dtype(df_filtered["Date"]):
            dfx = df_filtered.dropna(subset=["Date"]).copy()
            if not dfx.empty:
                month_counts = dfx["Date"].dt.to_period("M").value_counts().sort_index()
                if len(month_counts) > 0:
                    bullets.append(
                        f"Peak month (filtered): {str(month_counts.idxmax())} with {fmt_int(month_counts.max())} accidents."
                    )

                dow_counts = dfx["Date"].dt.day_name().value_counts()
                if len(dow_counts) > 0:
                    bullets.append(
                        f"Highest day of week (filtered): {dow_counts.idxmax()} with {fmt_int(dow_counts.max())} accidents."
                    )

                weekend_share = float((dfx["Date"].dt.dayofweek >= 5).mean()) * 100
                bullets.append(f"Weekend share (filtered): {weekend_share:.1f}% of accidents.")

        speed_col = find_col(df_filtered, ["Speed_limit", "Speed_Limit", "speed_limit"])
        if speed_col:
            svals = pd.to_numeric(df_filtered[speed_col], errors="coerce").dropna().astype(int)
            if len(svals) > 0:
                bullets.append(f"Most common speed limit (filtered): {int(svals.value_counts().idxmax())}.")

        for b in bullets[:10]:
            st.markdown(f"- {b}")

        veh_col = find_col(df_filtered, ["vehicles_count"])
        cas_col = find_col(df_filtered, ["casualties_count"])
        rows = []
        if veh_col:
            v = pd.to_numeric(df_filtered[veh_col], errors="coerce").fillna(0)
            rows.append(["Vehicles per accident", float(v.mean()), float(v.median()), float(v.quantile(0.9))])
        if cas_col:
            c = pd.to_numeric(df_filtered[cas_col], errors="coerce").fillna(0)
            rows.append(["Casualties per accident", float(c.mean()), float(c.median()), float(c.quantile(0.9))])

        if rows:
            hr()
            stats = pd.DataFrame(rows, columns=["Metric", "Mean", "Median", "P90"])
            stats["Mean"] = stats["Mean"].round(3)
            stats["Median"] = stats["Median"].round(3)
            stats["P90"] = stats["P90"].round(3)
            st.dataframe(stats, use_container_width=True)


def render_data_page(df_filtered: pd.DataFrame) -> None:
    with card():
        st.subheader("Export")
        st.markdown("<div class='small-muted'>Download the currently filtered accident-level dataset.</div>", unsafe_allow_html=True)
        csv = df_filtered.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="Download filtered CSV",
            data=csv,
            file_name="filtered_accidents.csv",
            mime="text/csv",
        )

    with card():
        st.subheader("Filtered dataset")
        st.dataframe(df_filtered, use_container_width=True, height=560)


# =========================
# Router
# =========================
if page == "Summary":
    render_summary_page(df, df_f)
elif page == "Exploration":
    render_exploration_page(df_f)
elif page == "Trends":
    render_trends_page(df_f)
elif page == "Locations":
    render_locations_page(df_f)
elif page == "Insights":
    render_insights_page(df, df_f)
elif page == "Data":
    render_data_page(df_f)
