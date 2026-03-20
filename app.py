from __future__ import annotations

import io
import os
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
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
# Theme
# =========================
BG = "#F6F8FC"
CARD = "#FFFFFF"
TEXT = "#0F172A"
MUTED = "rgba(15,23,42,0.68)"
BORDER = "rgba(15,23,42,0.08)"
PRIMARY = "#2563EB"
PRIMARY_SOFT = "rgba(37,99,235,0.12)"
SUCCESS = "#059669"
WARNING = "#D97706"
DANGER = "#DC2626"

st.markdown(
    f"""
    <style>
    html, body, [data-testid="stAppViewContainer"] {{
        background: {BG};
        color: {TEXT};
    }}
    .block-container {{
        padding-top: 1.9rem;
        padding-bottom: 1.5rem;
        max-width: 1460px;
    }}
    #MainMenu, footer {{
        visibility: hidden;
    }}
    section[data-testid="stSidebar"] > div {{
        border-right: 1px solid {BORDER};
    }}
    .hero {{
        background: linear-gradient(135deg, rgba(37,99,235,0.12), rgba(37,99,235,0.04));
        border: 1px solid rgba(37,99,235,0.14);
        border-radius: 24px;
        padding: 22px 22px 20px 22px;
        box-shadow: 0 14px 35px rgba(15,23,42,0.05);
    }}
    .badge {{
        display: inline-block;
        background: {PRIMARY_SOFT};
        color: {PRIMARY};
        border: 1px solid rgba(37,99,235,0.14);
        border-radius: 999px;
        padding: 6px 12px;
        font-weight: 800;
        font-size: 12px;
        letter-spacing: 0.02em;
        margin-bottom: 10px;
    }}
    .metric-card {{
        background: {CARD};
        border: 1px solid {BORDER};
        border-radius: 18px;
        padding: 14px 16px;
        box-shadow: 0 10px 30px rgba(15,23,42,0.05);
        min-height: 122px;
    }}
    .metric-title {{
        color: {MUTED};
        font-size: 13px;
        font-weight: 700;
    }}
    .metric-value {{
        color: {TEXT};
        font-size: 29px;
        font-weight: 800;
        margin-top: 6px;
        line-height: 1.1;
    }}
    .metric-sub {{
        color: {MUTED};
        font-size: 12px;
        margin-top: 8px;
    }}
    .section-title {{
        color: {TEXT};
        font-size: 21px;
        font-weight: 850;
    }}
    .section-sub {{
        color: {MUTED};
        font-size: 13px;
        margin-top: 4px;
    }}
    .small {{
        color: {MUTED};
        font-size: 12px;
    }}
    .insight-item {{
        padding: 9px 0;
        border-bottom: 1px solid rgba(15,23,42,0.06);
    }}
    .insight-item:last-child {{
        border-bottom: none;
    }}
    hr {{
        border: none;
        border-top: 1px solid {BORDER};
        margin: 10px 0 14px 0;
    }}
    .stTabs [data-baseweb="tab-list"] {{ gap: 8px; }}
    .stTabs [data-baseweb="tab"] {{
        height: 38px;
        background: #f9fafb;
        border: 1px solid {BORDER};
        border-radius: 999px;
        padding: 0 14px;
        font-weight: 800;
    }}
    .stTabs [aria-selected="true"] {{
        background: #eef2ff;
        border-color: rgba(99,102,241,.35);
    }}
    </style>
    """,
    unsafe_allow_html=True,
)


# =========================
# UI helpers
# =========================
@contextmanager
def section_card(title: str, subtitle: str = ""):
    container = st.container(border=True)
    with container:
        st.markdown(f'<div class="section-title">{title}</div>', unsafe_allow_html=True)
        if subtitle:
            st.markdown(f'<div class="section-sub">{subtitle}</div>', unsafe_allow_html=True)
        st.markdown("<hr>", unsafe_allow_html=True)
        yield


def kpi_card(title: str, value: str, subtitle: str = "") -> None:
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-title">{title}</div>
            <div class="metric-value">{value}</div>
            <div class="metric-sub">{subtitle}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def display_insights(items: List[str], limit: Optional[int] = None) -> None:
    shown = items if limit is None else items[:limit]
    if not shown:
        st.write("No insights are available for the current selection.")
        return
    for item in shown:
        st.markdown(f'<div class="insight-item">{item}</div>', unsafe_allow_html=True)


def fmt_int(x: float | int | None) -> str:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "—"
    return f"{int(x):,}"


def fmt_float(x: float | int | None, digits: int = 2) -> str:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "—"
    return f"{float(x):,.{digits}f}"


# =========================
# Data helpers
# =========================
def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.columns = [str(c).strip() for c in out.columns]
    return out


def find_col(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    lowered = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in lowered:
            return lowered[cand.lower()]
    return None


def coerce_datetime(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, errors="coerce", dayfirst=True)


def as_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def top_n_value_counts(series: pd.Series, n: int = 10) -> pd.DataFrame:
    vc = series.astype("string").fillna("Missing").value_counts().head(n)
    out = vc.reset_index()
    out.columns = ["Category", "Count"]
    out["Share (%)"] = (out["Count"] / max(vc.sum(), 1) * 100).round(2)
    return out


def _try_read_csv_bytes(file_bytes: bytes) -> pd.DataFrame:
    encodings = ["utf-8", "utf-8-sig", "cp1252", "latin1"]
    last_error: Optional[Exception] = None

    for enc in encodings:
        try:
            return pd.read_csv(io.BytesIO(file_bytes), encoding=enc, low_memory=False)
        except Exception as exc:
            last_error = exc
        try:
            return pd.read_csv(io.BytesIO(file_bytes), encoding=enc, engine="python")
        except Exception as exc:
            last_error = exc

    raise RuntimeError(f"Unable to read CSV with common encodings. Last error: {last_error}")


def safe_read_csv_path(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")
    with open(path, "rb") as handle:
        payload = handle.read()
    return _try_read_csv_bytes(payload)


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
        df["Accident_Index"] = np.arange(len(df)).astype(str)
        notes.append("Key column was not found, so a synthetic Accident_Index was created from the row index.")
    elif key != "Accident_Index":
        df = df.rename(columns={key: "Accident_Index"})

    date_col = find_col(df, ["Date", "Accident_Date", "accident_date"])
    if date_col is not None:
        df[date_col] = coerce_datetime(df[date_col])
        bad_dates = int(df[date_col].isna().sum())
        if bad_dates > 0:
            notes.append(f"Date parsing produced {bad_dates:,} missing values.")
        if date_col != "Date":
            df = df.rename(columns={date_col: "Date"})
    else:
        df["Date"] = pd.NaT
        notes.append("No date column was found, so time-based views may be limited.")

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
        df["IsWeekend"] = df["Date"].dt.dayofweek >= 5

    if "Time" in df.columns:
        tt = df["Time"].astype("string").str.strip()
        df["Hour"] = pd.to_numeric(tt.str.split(":").str[0], errors="coerce").clip(0, 23)

    return df, notes


def _aggregate_by_accident(df: pd.DataFrame, prefix: str) -> pd.DataFrame:
    x = standardize_columns(df)
    key = find_col(x, ["Accident_Index", "accident_index", "collision_index"])
    if key is None:
        raise ValueError(f"{prefix}: key column not found (Accident_Index / collision_index).")
    if key != "Accident_Index":
        x = x.rename(columns={key: "Accident_Index"})
    return x.groupby("Accident_Index").size().reset_index(name=f"{prefix}_count")


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
            notes.append(f"Loaded default files: {c_path}, {v_path}, {ca_path}.")
            break

    if collisions is None:
        raise FileNotFoundError(
            "Default files were not found. Expected data/collisions.csv, data/vehicles.csv and data/casualties.csv, or the same three files next to app.py."
        )

    collisions_p, prep_notes = _prepare_collisions(collisions)
    notes.extend(prep_notes)

    merged = collisions_p.copy()

    try:
        v_agg = _aggregate_by_accident(vehicles, prefix="vehicles")
        merged = merged.merge(v_agg, on="Accident_Index", how="left")
    except Exception as exc:
        notes.append(f"Vehicle aggregation was skipped: {exc}")

    try:
        c_agg = _aggregate_by_accident(casualties, prefix="casualties")
        merged = merged.merge(c_agg, on="Accident_Index", how="left")
    except Exception as exc:
        notes.append(f"Casualty aggregation was skipped: {exc}")

    return DataBundle(
        collisions=collisions_p,
        vehicles=standardize_columns(vehicles) if vehicles is not None else None,
        casualties=standardize_columns(casualties) if casualties is not None else None,
        merged=merged,
        notes=notes,
    )


@st.cache_data(show_spinner=False)
def load_uploaded_csv(uploaded_file) -> DataBundle:
    payload = uploaded_file.read()
    df = standardize_columns(_try_read_csv_bytes(payload))
    notes = [f"Loaded uploaded dataset: {uploaded_file.name}."]

    key = find_col(df, ["Accident_Index", "accident_index", "collision_index"])
    if key is None:
        df["Accident_Index"] = np.arange(len(df)).astype(str)
        notes.append("Key column was not found, so a synthetic Accident_Index was created from the row index.")
    elif key != "Accident_Index":
        df = df.rename(columns={key: "Accident_Index"})

    date_col = find_col(df, ["Date", "Accident_Date", "accident_date"])
    if date_col is not None:
        df[date_col] = coerce_datetime(df[date_col])
        if date_col != "Date":
            df = df.rename(columns={date_col: "Date"})
        df["Year"] = df["Date"].dt.year
        df["Month"] = df["Date"].dt.month
        df["MonthName"] = df["Date"].dt.month_name()
        df["DayOfWeek"] = df["Date"].dt.day_name()
        df["IsWeekend"] = df["Date"].dt.dayofweek >= 5

    time_col = find_col(df, ["Time", "time"])
    if time_col is not None:
        if time_col != "Time":
            df = df.rename(columns={time_col: "Time"})
        tt = df["Time"].astype("string").str.strip()
        df["Hour"] = pd.to_numeric(tt.str.split(":").str[0], errors="coerce").clip(0, 23)

    return DataBundle(
        collisions=pd.DataFrame(),
        vehicles=None,
        casualties=None,
        merged=df,
        notes=notes,
    )


def safe_filter_frame(df: pd.DataFrame, filters: Dict) -> pd.DataFrame:
    out = df.copy()

    if "Date" in out.columns and pd.api.types.is_datetime64_any_dtype(out["Date"]):
        start_date, end_date = filters.get("date_range", (None, None))
        if start_date is not None and end_date is not None:
            out = out[(out["Date"] >= pd.to_datetime(start_date)) & (out["Date"] <= pd.to_datetime(end_date))]

    district_col = find_col(out, ["Local_Authority_District", "local_authority_district", "District"])
    if district_col and filters.get("district") and filters["district"] != "All":
        out = out[out[district_col].astype("string") == filters["district"]]

    road_col = find_col(out, ["Road_Type", "road_type"])
    if road_col and filters.get("road_type") and filters["road_type"] != "All":
        out = out[out[road_col].astype("string") == filters["road_type"]]

    speed_col = find_col(out, ["Speed_limit", "Speed_Limit", "speed_limit"])
    if speed_col and filters.get("speed_limit") and filters["speed_limit"] != "All":
        out = out[out[speed_col].astype("string") == str(filters["speed_limit"])]

    for key in ["weather", "light", "surface"]:
        colname = filters.get(f"{key}_col")
        choice = filters.get(key)
        if colname and choice and choice != "All" and colname in out.columns:
            out = out[out[colname].astype("string") == choice]

    return out


# =========================
# Analysis helpers
# =========================
def get_summary_metrics(df: pd.DataFrame) -> dict:
    metrics = {"accidents": len(df)}

    if "Date" in df.columns and pd.api.types.is_datetime64_any_dtype(df["Date"]):
        valid = df["Date"].dropna()
        metrics["date_start"] = valid.min() if not valid.empty else pd.NaT
        metrics["date_end"] = valid.max() if not valid.empty else pd.NaT
        metrics["weekend_share"] = float((df["Date"].dt.dayofweek >= 5).mean() * 100) if not valid.empty else np.nan
    else:
        metrics["date_start"] = pd.NaT
        metrics["date_end"] = pd.NaT
        metrics["weekend_share"] = np.nan

    veh_col = find_col(df, ["vehicles_count"])
    cas_col = find_col(df, ["casualties_count"])

    if veh_col:
        veh = pd.to_numeric(df[veh_col], errors="coerce").fillna(0)
        metrics["vehicles_total"] = int(veh.sum())
        metrics["vehicles_avg"] = float(veh.mean())
    else:
        metrics["vehicles_total"] = None
        metrics["vehicles_avg"] = np.nan

    if cas_col:
        cas = pd.to_numeric(df[cas_col], errors="coerce").fillna(0)
        metrics["casualties_total"] = int(cas.sum())
        metrics["casualties_avg"] = float(cas.mean())
        metrics["high_impact_accidents"] = int((cas >= max(cas.quantile(0.9), 1)).sum()) if len(cas) else 0
    else:
        metrics["casualties_total"] = None
        metrics["casualties_avg"] = np.nan
        metrics["high_impact_accidents"] = None

    return metrics


def monthly_table(df: pd.DataFrame) -> pd.DataFrame:
    if "Date" not in df.columns or not pd.api.types.is_datetime64_any_dtype(df["Date"]):
        return pd.DataFrame(columns=["Month", "Accidents", "Casualties"])

    d = df.dropna(subset=["Date"]).copy()
    if d.empty:
        return pd.DataFrame(columns=["Month", "Accidents", "Casualties"])

    out = d.groupby(d["Date"].dt.to_period("M")).size().reset_index(name="Accidents")
    out["Month"] = out["Date"].astype(str)
    out = out.drop(columns="Date")

    cas_col = find_col(d, ["casualties_count"])
    if cas_col:
        cas = d.groupby(d["Date"].dt.to_period("M"))[cas_col].sum().reset_index(name="Casualties")
        cas["Month"] = cas["Date"].astype(str)
        cas = cas.drop(columns="Date")
        out = out.merge(cas, on="Month", how="left")
    else:
        out["Casualties"] = np.nan

    return out.sort_values("Month").reset_index(drop=True)


def dayofweek_table(df: pd.DataFrame) -> pd.DataFrame:
    if "Date" not in df.columns or not pd.api.types.is_datetime64_any_dtype(df["Date"]):
        return pd.DataFrame(columns=["DayOfWeek", "Accidents"])

    order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    counts = df["Date"].dt.day_name().value_counts().reindex(order).fillna(0).astype(int)
    out = counts.reset_index()
    out.columns = ["DayOfWeek", "Accidents"]
    return out


def hour_table(df: pd.DataFrame) -> pd.DataFrame:
    hour_col = find_col(df, ["Hour"])
    if hour_col is None:
        time_col = find_col(df, ["Time", "time"])
        if time_col is None:
            return pd.DataFrame(columns=["Hour", "Accidents"])
        hh = pd.to_numeric(df[time_col].astype("string").str.split(":").str[0], errors="coerce").clip(0, 23)
    else:
        hh = pd.to_numeric(df[hour_col], errors="coerce").clip(0, 23)

    hh = hh.dropna().astype(int)
    if hh.empty:
        return pd.DataFrame(columns=["Hour", "Accidents"])

    out = hh.value_counts().sort_index().reset_index()
    out.columns = ["Hour", "Accidents"]
    return out


def month_day_heatmap(df: pd.DataFrame) -> pd.DataFrame:
    if "Date" not in df.columns or not pd.api.types.is_datetime64_any_dtype(df["Date"]):
        return pd.DataFrame()

    d = df.dropna(subset=["Date"]).copy()
    if d.empty:
        return pd.DataFrame()

    d["MonthName"] = d["Date"].dt.month_name()
    d["DayOfWeek"] = d["Date"].dt.day_name()
    month_order = [
        "January", "February", "March", "April", "May", "June",
        "July", "August", "September", "October", "November", "December",
    ]
    day_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    pivot = pd.pivot_table(d, index="MonthName", columns="DayOfWeek", values="Accident_Index", aggfunc="count", fill_value=0)
    pivot = pivot.reindex(index=[m for m in month_order if m in pivot.index], columns=day_order, fill_value=0)
    return pivot


def categorical_impact_table(df: pd.DataFrame, column: str, min_records: int = 20) -> pd.DataFrame:
    if column not in df.columns:
        return pd.DataFrame(columns=["Category", "Accidents", "Casualties", "Casualties per accident", "Vehicles per accident"])

    d = df.copy()
    d[column] = d[column].astype("string").fillna("Missing")

    grouped = d.groupby(column, dropna=False).size().reset_index(name="Accidents")
    grouped = grouped.rename(columns={column: "Category"})

    cas_col = find_col(d, ["casualties_count"])
    veh_col = find_col(d, ["vehicles_count"])

    if cas_col:
        cas = d.groupby(column, dropna=False)[cas_col].sum().reset_index(name="Casualties")
        cas = cas.rename(columns={column: "Category"})
        grouped = grouped.merge(cas, on="Category", how="left")
    else:
        grouped["Casualties"] = np.nan

    if veh_col:
        veh = d.groupby(column, dropna=False)[veh_col].mean().reset_index(name="Vehicles per accident")
        veh = veh.rename(columns={column: "Category"})
        grouped = grouped.merge(veh, on="Category", how="left")
    else:
        grouped["Vehicles per accident"] = np.nan

    if "Casualties" in grouped.columns:
        grouped["Casualties per accident"] = grouped["Casualties"] / grouped["Accidents"].replace(0, np.nan)
    else:
        grouped["Casualties per accident"] = np.nan

    grouped = grouped[grouped["Accidents"] >= min_records].copy()
    return grouped.sort_values(["Accidents", "Casualties"], ascending=[False, False]).reset_index(drop=True)


def district_benchmark_table(df: pd.DataFrame, top_n: int = 15) -> pd.DataFrame:
    district_col = find_col(df, ["Local_Authority_District", "local_authority_district", "District"])
    if district_col is None:
        return pd.DataFrame(columns=["District", "Accidents", "Casualties", "Casualties per accident"])

    d = df.copy()
    grouped = d.groupby(district_col, dropna=False).size().reset_index(name="Accidents")
    grouped = grouped.rename(columns={district_col: "District"})

    cas_col = find_col(d, ["casualties_count"])
    if cas_col:
        cas = d.groupby(district_col, dropna=False)[cas_col].sum().reset_index(name="Casualties")
        cas = cas.rename(columns={district_col: "District"})
        grouped = grouped.merge(cas, on="District", how="left")
        grouped["Casualties per accident"] = grouped["Casualties"] / grouped["Accidents"].replace(0, np.nan)
    else:
        grouped["Casualties"] = np.nan
        grouped["Casualties per accident"] = np.nan

    return grouped.sort_values(["Accidents", "Casualties"], ascending=[False, False]).head(top_n).reset_index(drop=True)


def high_impact_table(df: pd.DataFrame) -> pd.DataFrame:
    cas_col = find_col(df, ["casualties_count"])
    veh_col = find_col(df, ["vehicles_count"])

    keep_cols: List[str] = []
    for col in ["Accident_Index", "Date"]:
        if col in df.columns:
            keep_cols.append(col)

    for candidates in [
        ["Local_Authority_District", "local_authority_district", "District"],
        ["Road_Type", "road_type"],
        ["Speed_limit", "Speed_Limit", "speed_limit"],
        ["Weather_Conditions", "weather_conditions", "Weather"],
        ["Light_Conditions", "light_conditions", "Light"],
    ]:
        col = find_col(df, candidates)
        if col and col not in keep_cols:
            keep_cols.append(col)

    if veh_col and veh_col not in keep_cols:
        keep_cols.append(veh_col)
    if cas_col and cas_col not in keep_cols:
        keep_cols.append(cas_col)

    out = df[keep_cols].copy() if keep_cols else df.copy()
    out["_cas"] = pd.to_numeric(df[cas_col], errors="coerce").fillna(0) if cas_col else 0
    out["_veh"] = pd.to_numeric(df[veh_col], errors="coerce").fillna(0) if veh_col else 0
    out = out.sort_values(["_cas", "_veh"], ascending=False).drop(columns=["_cas", "_veh"])
    return out.reset_index(drop=True)


def generate_data_insights(df_all: pd.DataFrame, df_filtered: pd.DataFrame) -> List[str]:
    insights: List[str] = []
    metrics_all = get_summary_metrics(df_all)
    metrics_filtered = get_summary_metrics(df_filtered)

    insights.append(
        f"The current filtered view contains {fmt_int(metrics_filtered['accidents'])} accidents out of {fmt_int(metrics_all['accidents'])} in the loaded dataset."
    )

    month_tbl = monthly_table(df_filtered)
    if not month_tbl.empty:
        peak_month = month_tbl.sort_values("Accidents", ascending=False).iloc[0]
        insights.append(
            f"The peak month in the current view is {peak_month['Month']} with {fmt_int(peak_month['Accidents'])} recorded accidents."
        )

    dow_tbl = dayofweek_table(df_filtered)
    if not dow_tbl.empty:
        peak_day = dow_tbl.sort_values("Accidents", ascending=False).iloc[0]
        insights.append(
            f"{peak_day['DayOfWeek']} records the highest accident count in the filtered view at {fmt_int(peak_day['Accidents'])}."
        )

    hour_tbl = hour_table(df_filtered)
    if not hour_tbl.empty:
        peak_hour = hour_tbl.sort_values("Accidents", ascending=False).iloc[0]
        insights.append(
            f"The busiest accident hour is {int(peak_hour['Hour']):02d}:00 with {fmt_int(peak_hour['Accidents'])} accidents."
        )

    if metrics_filtered["casualties_total"] is not None:
        insights.append(
            f"The filtered accidents are associated with {fmt_int(metrics_filtered['casualties_total'])} casualties, averaging {fmt_float(metrics_filtered['casualties_avg'], 2)} casualties per accident."
        )

    road_col = find_col(df_filtered, ["Road_Type", "road_type"])
    if road_col:
        road_tbl = categorical_impact_table(df_filtered, road_col)
        if not road_tbl.empty:
            top_road = road_tbl.iloc[0]
            insights.append(
                f"{top_road['Category']} is the most common road type in the filtered view, accounting for {fmt_int(top_road['Accidents'])} accidents."
            )

    speed_col = find_col(df_filtered, ["Speed_limit", "Speed_Limit", "speed_limit"])
    if speed_col:
        speed_tbl = categorical_impact_table(df_filtered, speed_col)
        if not speed_tbl.empty:
            top_speed = speed_tbl.iloc[0]
            insights.append(
                f"The busiest speed limit band in the current view is {top_speed['Category']}, with {fmt_int(top_speed['Accidents'])} accidents."
            )

    district_tbl = district_benchmark_table(df_filtered, top_n=15)
    if not district_tbl.empty:
        top_district = district_tbl.iloc[0]
        insights.append(
            f"{top_district['District']} has the highest accident concentration in the filtered view at {fmt_int(top_district['Accidents'])} accidents."
        )

    if np.isfinite(metrics_filtered["weekend_share"]):
        insights.append(
            f"Weekend accidents account for {metrics_filtered['weekend_share']:.1f}% of the filtered view, helping separate commuter patterns from leisure-time risk."
        )

    return insights


# =========================
# Sidebar
# =========================
with st.sidebar:
    st.title("Configuration")
    page = st.radio(
        "View",
        [
            "Executive Overview",
            "Trend & Timing",
            "Road Context",
            "Location Intelligence",
            "Insights",
            "Data Export",
        ],
        index=0,
    )

    st.divider()
    st.subheader("Data source")
    use_upload = st.toggle("Upload a CSV instead of default files", value=False)

    bundle: Optional[DataBundle] = None
    load_error: Optional[str] = None

    try:
        if use_upload:
            uploaded = st.file_uploader("Upload CSV", type=["csv"])
            if uploaded is None:
                st.info("Upload a CSV file to continue.")
            else:
                bundle = load_uploaded_csv(uploaded)
        else:
            bundle = load_default_bundle()
    except Exception as exc:
        load_error = str(exc)

    filters: Dict = {}

    st.divider()
    st.subheader("Filters")
    if bundle is not None:
        base = standardize_columns(bundle.merged)

        if "Date" in base.columns and pd.api.types.is_datetime64_any_dtype(base["Date"]):
            valid_dates = base["Date"].dropna()
            if not valid_dates.empty:
                start = valid_dates.min().date()
                end = valid_dates.max().date()
                date_range = st.date_input(
                    "Date range",
                    value=(start, end),
                    min_value=start,
                    max_value=end,
                )
                if isinstance(date_range, tuple) and len(date_range) == 2:
                    filters["date_range"] = date_range

        district_col = find_col(base, ["Local_Authority_District", "local_authority_district", "District"])
        if district_col:
            districts = base[district_col].dropna().astype("string")
            top_districts = districts.value_counts().head(50).index.tolist()
            filters["district"] = st.selectbox("District", ["All"] + top_districts, index=0)

        road_col = find_col(base, ["Road_Type", "road_type"])
        if road_col:
            road_options = ["All"] + sorted(base[road_col].dropna().astype("string").unique().tolist())
            filters["road_type"] = st.selectbox("Road type", road_options, index=0)

        speed_col = find_col(base, ["Speed_limit", "Speed_Limit", "speed_limit"])
        if speed_col:
            speeds = pd.to_numeric(base[speed_col], errors="coerce").dropna().astype(int)
            speed_options = ["All"] + sorted(speeds.unique().tolist())
            filters["speed_limit"] = st.selectbox("Speed limit", speed_options, index=0)

        weather_col = find_col(base, ["Weather_Conditions", "weather_conditions", "Weather"])
        if weather_col:
            weather_options = ["All"] + sorted(base[weather_col].dropna().astype("string").unique().tolist())
            filters["weather_col"] = weather_col
            filters["weather"] = st.selectbox("Weather", weather_options, index=0)

        light_col = find_col(base, ["Light_Conditions", "light_conditions", "Light"])
        if light_col:
            light_options = ["All"] + sorted(base[light_col].dropna().astype("string").unique().tolist())
            filters["light_col"] = light_col
            filters["light"] = st.selectbox("Light conditions", light_options, index=0)

        surface_col = find_col(base, ["Road_Surface_Conditions", "road_surface_conditions", "Road_Surface"])
        if surface_col:
            surface_options = ["All"] + sorted(base[surface_col].dropna().astype("string").unique().tolist())
            filters["surface_col"] = surface_col
            filters["surface"] = st.selectbox("Road surface", surface_options, index=0)

        show_sidebar_preview = st.checkbox("Show filtered preview in sidebar", value=False)
    else:
        show_sidebar_preview = False


# =========================
# Load guard and filtered data
# =========================
if load_error:
    st.error(load_error)
    st.stop()

if bundle is None:
    st.stop()

df = standardize_columns(bundle.merged)
df_f = safe_filter_frame(df, filters)

if df_f.empty:
    st.warning("No records match the current filter selection. Expand the filters to continue.")
    st.stop()

if show_sidebar_preview:
    st.sidebar.dataframe(df_f.head(15), use_container_width=True)

metrics_all = get_summary_metrics(df)
metrics_filtered = get_summary_metrics(df_f)
insights = generate_data_insights(df, df_f)
month_tbl = monthly_table(df_f)
dow_tbl = dayofweek_table(df_f)
hour_tbl = hour_table(df_f)
heatmap_tbl = month_day_heatmap(df_f)
district_tbl = district_benchmark_table(df_f, top_n=15)
high_impact = high_impact_table(df_f)

road_col = find_col(df_f, ["Road_Type", "road_type"])
speed_col = find_col(df_f, ["Speed_limit", "Speed_Limit", "speed_limit"])
weather_col = find_col(df_f, ["Weather_Conditions", "weather_conditions", "Weather"])
light_col = find_col(df_f, ["Light_Conditions", "light_conditions", "Light"])
surface_col = find_col(df_f, ["Road_Surface_Conditions", "road_surface_conditions", "Road_Surface"])


# =========================
# Header
# =========================
st.markdown(
    """
    <div class="hero">
      <div class="badge">FEATURED PROJECT</div>
      <div style="font-size:32px; font-weight:900; color:#0F172A; line-height:1.15;">
        UK Road Accidents (2021) Dashboard
      </div>
      <div style="margin-top:10px; color:rgba(15,23,42,0.72); font-size:15px; max-width:980px;">
        An insight-led dashboard for understanding when accidents cluster, where impact concentrates,
        and how road context, speed limits, and conditions change the accident picture.
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)
st.write("")

if bundle.notes:
    with st.expander("Data loading notes", expanded=False):
        for note in bundle.notes:
            st.write(f"- {note}")


# =========================
# Pages
# =========================
if page == "Executive Overview":
    peak_month_value = month_tbl.sort_values("Accidents", ascending=False).iloc[0]["Month"] if not month_tbl.empty else "Not available"
    casualty_value = fmt_int(metrics_filtered["casualties_total"]) if metrics_filtered["casualties_total"] is not None else "Not available"
    high_impact_value = fmt_int(metrics_filtered["high_impact_accidents"]) if metrics_filtered["high_impact_accidents"] is not None else "Not available"

    c1, c2, c3, c4 = st.columns(4, gap="large")
    with c1:
        kpi_card("Accidents in current view", fmt_int(metrics_filtered["accidents"]), "Records remaining after the active filters")
    with c2:
        kpi_card("Casualties in current view", casualty_value, "Aggregated casualty count when available")
    with c3:
        kpi_card("Peak accident month", peak_month_value, "Highest monthly concentration in the filtered view")
    with c4:
        kpi_card("High-impact accidents", high_impact_value, "Accidents at or above the upper casualty range")

    st.write("")

    left, right = st.columns([1.15, 0.85], gap="large")
    with left:
        with section_card("Executive Summary", "The strongest accident signals in the current filtered view"):
            display_insights(insights, limit=6)

    with right:
        with section_card("Impact Snapshot", "A compact read on accident concentration and severity"):
            snap1, snap2 = st.columns(2)
            with snap1:
                kpi_card("Weekend share", f"{metrics_filtered['weekend_share']:.1f}%" if np.isfinite(metrics_filtered["weekend_share"]) else "Not available", "Share of filtered accidents occurring on Saturday or Sunday")
            with snap2:
                avg_cas = metrics_filtered["casualties_avg"]
                kpi_card("Casualties per accident", fmt_float(avg_cas, 2) if np.isfinite(avg_cas) else "Not available", "Average casualty load in the current filtered view")

            if road_col:
                mix = top_n_value_counts(df_f[road_col], n=6)
                fig_mix = px.pie(mix, names="Category", values="Count", hole=0.55)
                fig_mix.update_layout(height=260, margin=dict(l=0, r=0, t=10, b=0))
                st.plotly_chart(fig_mix, use_container_width=True)

    st.write("")

    a, b = st.columns([1.05, 0.95], gap="large")
    with a:
        with section_card("Monthly accident pattern", "This shows when accident counts concentrate across the selected time window"):
            if month_tbl.empty:
                st.info("Monthly trend data is not available for the current selection.")
            else:
                fig_month = px.line(month_tbl, x="Month", y="Accidents", markers=True, labels={"Month": "Month", "Accidents": "Accidents"})
                fig_month.update_layout(height=360, margin=dict(l=0, r=0, t=10, b=0))
                st.plotly_chart(fig_month, use_container_width=True)

    with b:
        with section_card("Day-of-week profile", "This highlights whether accident concentration is stronger during workdays or weekends"):
            if dow_tbl.empty:
                st.info("Day-of-week analysis is not available for the current selection.")
            else:
                fig_dow = px.bar(dow_tbl, x="DayOfWeek", y="Accidents", labels={"DayOfWeek": "Day of week", "Accidents": "Accidents"})
                fig_dow.update_layout(height=360, margin=dict(l=0, r=0, t=10, b=0))
                st.plotly_chart(fig_dow, use_container_width=True)

    st.write("")

    x, y = st.columns([1.0, 1.0], gap="large")
    with x:
        with section_card("Road context leaderboard", "Road types ranked by accident volume and casualty burden"):
            if road_col:
                road_tbl = categorical_impact_table(df_f, road_col)
                st.dataframe(road_tbl.head(12), use_container_width=True, hide_index=True)
            else:
                st.info("Road type data is not available in the loaded dataset.")

    with y:
        with section_card("Top districts", "Districts with the highest accident concentration in the current filtered view"):
            if district_tbl.empty:
                st.info("District-level benchmarking is not available in the loaded dataset.")
            else:
                fig_dist = px.bar(
                    district_tbl.iloc[::-1],
                    x="Accidents",
                    y="District",
                    orientation="h",
                    labels={"Accidents": "Accidents", "District": "District"},
                )
                fig_dist.update_layout(height=380, margin=dict(l=0, r=0, t=10, b=0))
                st.plotly_chart(fig_dist, use_container_width=True)

elif page == "Trend & Timing":
    c1, c2, c3, c4 = st.columns(4, gap="large")
    with c1:
        peak_day = dow_tbl.sort_values("Accidents", ascending=False).iloc[0]["DayOfWeek"] if not dow_tbl.empty else "Not available"
        kpi_card("Highest accident day", peak_day, "Day of week with the largest accident count")
    with c2:
        peak_hour = f"{int(hour_tbl.sort_values('Accidents', ascending=False).iloc[0]['Hour']):02d}:00" if not hour_tbl.empty else "Not available"
        kpi_card("Peak accident hour", peak_hour, "Hour with the highest observed accident concentration")
    with c3:
        kpi_card("Weekend share", f"{metrics_filtered['weekend_share']:.1f}%" if np.isfinite(metrics_filtered["weekend_share"]) else "Not available", "Share of accidents occurring during weekends")
    with c4:
        date_span = "Not available"
        if pd.notna(metrics_filtered["date_start"]) and pd.notna(metrics_filtered["date_end"]):
            date_span = f"{metrics_filtered['date_start'].date()} to {metrics_filtered['date_end'].date()}"
        kpi_card("Filtered date coverage", date_span, "Time window currently visible in the dashboard")

    st.write("")

    left, right = st.columns([1.0, 1.0], gap="large")
    with left:
        with section_card("Monthly accident trend", "Use this to identify seasonal concentration or changes across the year"):
            if month_tbl.empty:
                st.info("Monthly trend data is not available.")
            else:
                fig_month = px.line(month_tbl, x="Month", y="Accidents", markers=True)
                fig_month.update_layout(height=360, margin=dict(l=0, r=0, t=10, b=0))
                st.plotly_chart(fig_month, use_container_width=True)
                if month_tbl["Casualties"].notna().any():
                    fig_cas = px.line(month_tbl, x="Month", y="Casualties", markers=True)
                    fig_cas.update_layout(height=300, margin=dict(l=0, r=0, t=10, b=0))
                    st.plotly_chart(fig_cas, use_container_width=True)

    with right:
        with section_card("Day-of-week and hour patterns", "This reveals when operational risk is most concentrated"):
            if dow_tbl.empty:
                st.info("Day-of-week analysis is not available.")
            else:
                fig_dow = px.bar(dow_tbl, x="DayOfWeek", y="Accidents")
                fig_dow.update_layout(height=260, margin=dict(l=0, r=0, t=10, b=0))
                st.plotly_chart(fig_dow, use_container_width=True)
            if not hour_tbl.empty:
                fig_hour = px.line(hour_tbl, x="Hour", y="Accidents", markers=True)
                fig_hour.update_layout(height=260, margin=dict(l=0, r=0, t=10, b=0))
                st.plotly_chart(fig_hour, use_container_width=True)

    st.write("")
    with section_card("Month by day-of-week heatmap", "This view helps spot whether weekday patterns shift across the year"):
        if heatmap_tbl.empty:
            st.info("The heatmap could not be created because valid date fields are not available.")
        else:
            fig_heat = px.imshow(
                heatmap_tbl,
                aspect="auto",
                labels={"x": "Day of week", "y": "Month", "color": "Accidents"},
            )
            fig_heat.update_layout(height=500, margin=dict(l=0, r=0, t=10, b=0))
            st.plotly_chart(fig_heat, use_container_width=True)

elif page == "Road Context":
    feature_options = {}
    if road_col:
        feature_options["Road type"] = road_col
    if speed_col:
        feature_options["Speed limit"] = speed_col
    if weather_col:
        feature_options["Weather"] = weather_col
    if light_col:
        feature_options["Light conditions"] = light_col
    if surface_col:
        feature_options["Road surface"] = surface_col

    c1, c2, c3, c4 = st.columns(4, gap="large")
    with c1:
        dominant = top_n_value_counts(df_f[road_col], 1).iloc[0]["Category"] if road_col else "Not available"
        kpi_card("Most common road type", str(dominant), "Largest accident share in the current filtered view")
    with c2:
        dominant_speed = top_n_value_counts(df_f[speed_col].astype("string"), 1).iloc[0]["Category"] if speed_col else "Not available"
        kpi_card("Most common speed limit", str(dominant_speed), "Speed limit with the largest accident concentration")
    with c3:
        kpi_card("Average vehicles per accident", fmt_float(metrics_filtered["vehicles_avg"], 2) if np.isfinite(metrics_filtered["vehicles_avg"]) else "Not available", "Aggregated only when vehicle data is available")
    with c4:
        kpi_card("Average casualties per accident", fmt_float(metrics_filtered["casualties_avg"], 2) if np.isfinite(metrics_filtered["casualties_avg"]) else "Not available", "Aggregated only when casualty data is available")

    st.write("")

    if not feature_options:
        with section_card("Road context analysis", "No road context columns were found in the loaded dataset"):
            st.info("Road context views require at least one of the road type, speed limit, weather, light, or road surface columns.")
    else:
        left, right = st.columns([1.05, 0.95], gap="large")
        selected_label = left.selectbox("Context feature", list(feature_options.keys()), index=0)
        selected_metric = right.selectbox(
            "Ranking metric",
            ["Accidents", "Casualties", "Casualties per accident", "Vehicles per accident"],
            index=0,
        )
        selected_col = feature_options[selected_label]
        impact_tbl = categorical_impact_table(df_f, selected_col)
        if selected_metric not in impact_tbl.columns:
            selected_metric = "Accidents"

        a, b = st.columns([1.0, 1.0], gap="large")
        with a:
            with section_card("Context ranking", "This ranks categories by the selected metric so the strongest signals surface first"):
                if impact_tbl.empty:
                    st.info("There are not enough records to benchmark the selected context feature.")
                else:
                    show_tbl = impact_tbl.sort_values(selected_metric, ascending=False).head(12).iloc[::-1]
                    fig_context = px.bar(
                        show_tbl,
                        x=selected_metric,
                        y="Category",
                        orientation="h",
                        labels={selected_metric: selected_metric, "Category": selected_label},
                    )
                    fig_context.update_layout(height=420, margin=dict(l=0, r=0, t=10, b=0))
                    st.plotly_chart(fig_context, use_container_width=True)

        with b:
            with section_card("Context benchmark table", "This supports direct comparison across categories using both volume and impact"):
                if impact_tbl.empty:
                    st.info("No benchmark table is available for the selected context feature.")
                else:
                    st.dataframe(impact_tbl.head(15), use_container_width=True, hide_index=True)

        if not impact_tbl.empty:
            st.write("")
            with section_card("Context interpretation", "Concise reading of what the current road context breakdown suggests"):
                top_row = impact_tbl.sort_values(selected_metric, ascending=False).iloc[0]
                bullets = [
                    f"The leading {selected_label.lower()} category by {selected_metric.lower()} is {top_row['Category']}.",
                    f"It accounts for {fmt_int(top_row['Accidents'])} accidents in the current filtered view.",
                ]
                if pd.notna(top_row.get("Casualties", np.nan)):
                    bullets.append(f"Its total recorded casualties are {fmt_int(top_row['Casualties'])}.")
                if pd.notna(top_row.get("Casualties per accident", np.nan)):
                    bullets.append(f"Its average casualty burden is {fmt_float(top_row['Casualties per accident'], 2)} casualties per accident.")
                display_insights(bullets)

elif page == "Location Intelligence":
    lat_col = find_col(df_f, ["Latitude", "latitude"])
    lon_col = find_col(df_f, ["Longitude", "longitude", "long"])

    c1, c2, c3 = st.columns(3, gap="large")
    with c1:
        lead_district = district_tbl.iloc[0]["District"] if not district_tbl.empty else "Not available"
        kpi_card("Top accident district", str(lead_district), "Highest accident concentration in the filtered view")
    with c2:
        top_district_acc = district_tbl.iloc[0]["Accidents"] if not district_tbl.empty else None
        kpi_card("Accidents in top district", fmt_int(top_district_acc), "Volume in the leading district")
    with c3:
        mapped_points = fmt_int(min(len(df_f), 12000)) if lat_col and lon_col else "Not available"
        kpi_card("Mapped accident points", mapped_points, "Sampled when the filtered dataset is very large")

    st.write("")

    left, right = st.columns([1.15, 0.85], gap="large")
    with left:
        with section_card("Accident map", "This view highlights where accident points are concentrated geographically"):
            if not lat_col or not lon_col:
                st.info("Latitude and longitude columns were not found in the loaded dataset.")
            else:
                loc = df_f.copy()
                loc[lat_col] = as_numeric(loc[lat_col])
                loc[lon_col] = as_numeric(loc[lon_col])
                loc = loc.dropna(subset=[lat_col, lon_col])
                if loc.empty:
                    st.info("No valid coordinates are available after cleaning.")
                else:
                    if len(loc) > 12000:
                        loc = loc.sample(12000, random_state=42)
                    st.map(loc.rename(columns={lat_col: "lat", lon_col: "lon"})[["lat", "lon"]])

    with right:
        with section_card("District leaderboard", "Districts ranked by accident count, with casualty burden when available"):
            if district_tbl.empty:
                st.info("District-level benchmarking is not available for the loaded dataset.")
            else:
                st.dataframe(district_tbl, use_container_width=True, hide_index=True)

    if not district_tbl.empty and district_tbl["Casualties per accident"].notna().any():
        st.write("")
        with section_card("District benchmark scatter", "This compares accident volume with casualty burden to surface higher-impact districts"):
            fig_scatter = px.scatter(
                district_tbl,
                x="Accidents",
                y="Casualties per accident",
                size="Accidents",
                hover_name="District",
                labels={"Accidents": "Accidents", "Casualties per accident": "Casualties per accident"},
            )
            fig_scatter.update_layout(height=420, margin=dict(l=0, r=0, t=10, b=0))
            st.plotly_chart(fig_scatter, use_container_width=True)

elif page == "Insights":
    c1, c2, c3, c4 = st.columns(4, gap="large")
    with c1:
        kpi_card("Filtered accidents", fmt_int(metrics_filtered["accidents"]), "Current scope of the analysis")
    with c2:
        kpi_card("Weekend share", f"{metrics_filtered['weekend_share']:.1f}%" if np.isfinite(metrics_filtered["weekend_share"]) else "Not available", "Observed weekend contribution to accident volume")
    with c3:
        peak_month = month_tbl.sort_values("Accidents", ascending=False).iloc[0]["Month"] if not month_tbl.empty else "Not available"
        kpi_card("Peak month", peak_month, "Month with the highest accident count")
    with c4:
        peak_district = district_tbl.iloc[0]["District"] if not district_tbl.empty else "Not available"
        kpi_card("Top district", str(peak_district), "District with the highest accident concentration")

    st.write("")
    with section_card("Insights", "This dashboard is descriptive rather than predictive, so the insight layer focuses on observed accident patterns"):
        st.markdown("**Data-driven insights**")
        display_insights(insights, limit=10)

        st.write("")
        st.markdown("**Analytical interpretation**")
        interpretation = [
            "This project is strongest when used as an accident-pattern dashboard rather than as a raw data browser, because the core value comes from when, where, and under what conditions accidents cluster.",
            "The filtered views help narrow the analysis from a national picture to a more operational one, such as a district, road type, or speed-limit segment.",
            "High-impact accidents deserve separate attention because volume alone does not capture where casualty burden is most concentrated.",
            "The combination of time trends, road context, and district comparison makes this dashboard useful for safety monitoring, operational reporting, and portfolio presentation.",
        ]
        display_insights(interpretation)

    st.write("")
    with section_card("Highest-impact accidents", "Ranked by casualties first and vehicles second when aggregated counts are available"):
        st.dataframe(high_impact.head(20), use_container_width=True, hide_index=True)

elif page == "Data Export":
    with section_card("Download filtered dataset", "Use this page when you need the filtered accident-level data outside the dashboard"):
        csv_bytes = df_f.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="Download filtered CSV",
            data=csv_bytes,
            file_name="filtered_accidents.csv",
            mime="text/csv",
        )
        st.markdown(f'<div class="small">Rows in export: {fmt_int(len(df_f))}</div>', unsafe_allow_html=True)

    with section_card("Filtered dataset preview", "Preview is kept here so the main dashboard remains insight-led"):
        st.dataframe(df_f.head(200), use_container_width=True, height=560)