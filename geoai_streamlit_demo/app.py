"""
GeoAI Capstone Demo App (Streamlit)
-----------------------------------
This app is a Streamlit conversion of your `geoai_report_template.ipynb`.

What it does
- Loads 2025 predictions from S3 (parquet/csv; headerless SageMaker CSV supported)
- Optionally joins with curated actual yields (if available for the same year)
- Produces "capstone-standard" visuals:
  * Cutoff comparison (distribution + county ranking)
  * Actual vs Predicted (if actuals exist)
  * Error metrics table (RMSE, MAE, MAPE, R²) (if actuals exist)
  * Interactive county table + top/bottom counties per cutoff
  * Driver-signal exploration (NDVI/ERA5/Storm feature drivers) if frozen features are available
- Generates a one-click HTML report you can download and share with professors.

Run locally:
  pip install -r requirements.txt
  streamlit run app.py

AWS Auth:
- Use your usual AWS credentials (env vars, ~/.aws/credentials, or IAM role)
- Region defaults to ap-south-1
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import date
from typing import Dict, List, Tuple, Optional
from urllib.parse import urlparse

import numpy as np
import pandas as pd
import streamlit as st

# Plotly is optional but recommended for a better demo
try:
    import plotly.express as px
    import plotly.graph_objects as go
    _HAS_PLOTLY = True
except Exception:
    _HAS_PLOTLY = False

# S3 support
try:
    import boto3
    import s3fs
    _HAS_S3 = True
except Exception:
    _HAS_S3 = False


# ----------------------------
# Utilities
# ----------------------------

def normalize_county_name(x: str) -> str:
    if pd.isna(x):
        return x
    s = str(x).strip().lower()
    return s.replace(" county", "")

def ensure_columns(df: pd.DataFrame, rename_map: dict) -> pd.DataFrame:
    for k, v in rename_map.items():
        if k in df.columns and v not in df.columns:
            df = df.rename(columns={k: v})
    return df

def metric_rmse(y_true, y_pred) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))

def metric_mae(y_true, y_pred) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.mean(np.abs(y_true - y_pred)))

def metric_mape(y_true, y_pred) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    denom = np.where(np.abs(y_true) < 1e-9, np.nan, np.abs(y_true))
    return float(np.nanmean(np.abs((y_true - y_pred) / denom)) * 100.0)

def metric_r2(y_true, y_pred) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return float(1.0 - ss_res / ss_tot) if ss_tot != 0 else np.nan


def _require_s3():
    if not _HAS_S3:
        st.error("Missing S3 dependencies. Install: pip install boto3 s3fs")
        st.stop()

@st.cache_resource
def _s3_client(region_name: str):
    _require_s3()
    return boto3.client("s3", region_name=region_name)

def s3_list_files_under_prefix(region: str, prefix: str, exts=(".parquet", ".csv")) -> List[str]:
    """
    Returns full s3:// URIs for objects under the given s3:// prefix.
    """
    _require_s3()
    u = urlparse(prefix)
    if u.scheme != "s3":
        raise ValueError(f"Expected s3:// prefix, got: {prefix}")

    s3 = _s3_client(region)
    bucket = u.netloc
    key_prefix = u.path.lstrip("/")

    out: List[str] = []
    token = None
    while True:
        kwargs = {"Bucket": bucket, "Prefix": key_prefix, "MaxKeys": 1000}
        if token:
            kwargs["ContinuationToken"] = token
        resp = s3.list_objects_v2(**kwargs)
        for obj in resp.get("Contents", []):
            k = obj["Key"]
            if k.lower().endswith(tuple(e.lower() for e in exts)):
                out.append(f"s3://{bucket}/{k}")
        if resp.get("IsTruncated"):
            token = resp.get("NextContinuationToken")
        else:
            break
    return sorted(out)

@st.cache_data(show_spinner=False)
def s3_read_parquet(uri: str) -> pd.DataFrame:
    _require_s3()
    return pd.read_parquet(uri, engine="pyarrow")

@st.cache_data(show_spinner=False)
def s3_read_csv(uri: str, header: Optional[int] = "infer") -> pd.DataFrame:
    _require_s3()
    return pd.read_csv(uri, header=header)

def _read_any_prediction_file(path_or_s3uri: str) -> pd.DataFrame:
    p = path_or_s3uri.lower()
    if p.endswith(".parquet"):
        return s3_read_parquet(path_or_s3uri)
    if p.endswith(".csv"):
        # Try header first, then fallback to no-header
        try:
            df = s3_read_csv(path_or_s3uri, header="infer")
        except Exception:
            df = s3_read_csv(path_or_s3uri, header=None)
        if df.shape[1] == 1:
            df.columns = ["prediction"]
        return df
    raise ValueError(f"Unsupported file type: {path_or_s3uri}")

@st.cache_data(show_spinner=False)
def load_predictions_for_season(
    region: str,
    bucket: str,
    state_fips: str,
    county_fips: str,
    feature_season: str,
    run_date: str,
    model_name: str,
) -> Tuple[pd.DataFrame, Dict]:
    """
    Loads predictions for a single cutoff season. Supports parquet or csv (headerless SageMaker CSV).
    """
    prefix = (
        f"s3://{bucket}/predictions/"
        f"state_fips={state_fips}/"
        f"county_fips={county_fips}/"
        f"feature_season={feature_season}/"
        f"run_date={run_date}/"
        f"model={model_name}/"
    )

    dbg = {"season": feature_season, "model": model_name, "prefix": prefix}

    files = s3_list_files_under_prefix(region, prefix, exts=(".parquet", ".csv"))
    if not files:
        alt_prefix = prefix + "output/"
        dbg["alt_prefix"] = alt_prefix
        files = s3_list_files_under_prefix(region, alt_prefix, exts=(".parquet", ".csv"))
        dbg["files_found_alt"] = len(files)

    dbg["files_found"] = len(files)
    dbg["files_preview"] = files[:10]

    if not files:
        raise FileNotFoundError(
            f"No parquet/csv found under:\n  {prefix}\n(or)\n  {prefix}output/\n"
            f"Verify your S3 prediction path."
        )

    dfs = []
    for f in files:
        dfs.append(_read_any_prediction_file(f))

    df = pd.concat(dfs, ignore_index=True)
    if df.shape[1] == 1 and "prediction" not in df.columns:
        df.columns = ["prediction"]

    df["feature_season"] = feature_season
    df["model_name"] = model_name
    return df, dbg

@st.cache_data(show_spinner=False)
def load_actuals(region: str, actuals_uri: str) -> pd.DataFrame:
    df = s3_read_parquet(actuals_uri)
    df = ensure_columns(df, {
        "county_name": "county",
        "Yield": "yield_bu_acre",
        "yield": "yield_bu_acre",
        "y": "yield_bu_acre",
        "YEAR": "year",
        "Year": "year",
    })
    if "county" in df.columns:
        df["county_norm"] = df["county"].map(normalize_county_name)
    if "year" in df.columns:
        df["year"] = df["year"].astype(int)
    return df

@st.cache_data(show_spinner=False)
def load_features_frozen(region: str, features_prefix_uri: str) -> pd.DataFrame:
    """
    Loads frozen features (parquet) under a prefix. Assumes features are stored as parquet files.
    """
    files = s3_list_files_under_prefix(region, features_prefix_uri, exts=(".parquet",))
    if not files:
        raise FileNotFoundError(f"No parquet features found under {features_prefix_uri}")
    dfs = [s3_read_parquet(f) for f in files]
    df = pd.concat(dfs, ignore_index=True)
    df = ensure_columns(df, {"county_name": "county", "County": "county", "YEAR": "year", "Year": "year"})
    if "county" in df.columns:
        df["county_norm"] = df["county"].map(normalize_county_name)
    if "year" in df.columns:
        df["year"] = df["year"].astype(int)
    return df


# ----------------------------
# Report helpers
# ----------------------------

def compute_metrics(merged: pd.DataFrame) -> pd.DataFrame:
    """
    Expects columns: feature_season, yield_bu_acre, prediction
    """
    rows = []
    for season, g in merged.groupby("feature_season"):
        y = g["yield_bu_acre"].astype(float).values
        p = g["prediction"].astype(float).values
        rows.append({
            "feature_season": season,
            "n": len(g),
            "RMSE": metric_rmse(y, p),
            "MAE": metric_mae(y, p),
            "MAPE_%": metric_mape(y, p),
            "R2": metric_r2(y, p),
        })
    out = pd.DataFrame(rows).sort_values("feature_season")
    return out

def top_bottom(pred_all: pd.DataFrame, season: str, year: int, n: int = 10) -> pd.DataFrame:
    d = pred_all[(pred_all["year"] == year) & (pred_all["feature_season"] == season)][["county_norm", "prediction"]].dropna()
    top = d.sort_values("prediction", ascending=False).head(n).assign(rank="top")
    bot = d.sort_values("prediction", ascending=True).head(n).assign(rank="bottom")
    out = pd.concat([top, bot], ignore_index=True)
    out["feature_season"] = season
    return out

def _plotly_or_fallback_warning():
    if not _HAS_PLOTLY:
        st.warning("Plotly is not installed. Install it for interactive charts: pip install plotly")


# ----------------------------
# Streamlit UI
# ----------------------------

st.set_page_config(page_title="GeoAI Yield Prediction Demo", layout="wide")

st.title("GeoAI Crop Yield Prediction — Capstone Demo")
st.caption("Interactive report for model predictions (S3) + capstone-standard visuals (cutoff comparison, metrics, rankings, drivers).")

with st.sidebar:
    st.header("Data & Run Settings")

    region = st.text_input("AWS Region", value=os.getenv("AWS_REGION", "ap-south-1"))
    bucket = st.text_input("S3 Bucket", value=os.getenv("GEOAI_BUCKET", "geoai-demo-data"))

    state_fips = st.text_input("State FIPS", value=os.getenv("STATE_FIPS", "19"))
    county_fips = st.text_input("County FIPS", value=os.getenv("COUNTY_FIPS", "ALL"))

    run_date = st.text_input("Run date (run_date=...)", value=os.getenv("RUN_DATE", str(date.today())))

    default_seasons = ["jun01", "jul01", "jul15", "aug01", "aug15"]
    seasons = st.multiselect("Cutoff seasons", options=default_seasons, default=default_seasons)

    st.markdown("---")
    st.subheader("Model name per season")
    st.caption("Must match: predictions/.../model=<model_name>/")

    # Provide sane defaults matching your notebook; you can adjust in UI
    model_defaults = {
        "jun01": "Jun01_LightGBM-limited_withstorm",
        "jul01": "Jul01_LightGBM-tuned_withoutstorm",
        "jul15": "Jul15_LightGBM-limited_withstorm",
        "aug01": "Aug01_LightGBM-limited_withstorm",
        "aug15": "Aug15_LightGBM-limited_withstorm",
    }
    model_names: Dict[str, str] = {}
    for s in seasons:
        model_names[s] = st.text_input(f"model[{s}]", value=model_defaults.get(s, ""))

    st.markdown("---")
    st.subheader("Actuals + Frozen Features (optional)")

    actuals_uri_default = f"s3://{bucket}/curated/yield/state_fips={state_fips}/county_fips={county_fips}/actuals.parquet"
    actuals_uri = st.text_input("Actuals parquet URI", value=actuals_uri_default)

    use_features = st.checkbox("Load frozen features (drivers)", value=True)
    feature_year = st.number_input("Focus year", min_value=2000, max_value=2100, value=2025, step=1)

    st.markdown("---")
    load_btn = st.button("Load / Refresh Data", type="primary")


def _load_all():
    # Predictions
    pred_dfs = []
    debug_info = []
    for s in seasons:
        m = model_names.get(s, "")
        if not m:
            continue
        dfp, dbg = load_predictions_for_season(region, bucket, state_fips, county_fips, s, run_date, m)
        dfp = ensure_columns(dfp, {"pred": "prediction", "Predicted": "prediction", "yhat": "prediction"})
        dfp = ensure_columns(dfp, {"county_name": "county", "County": "county"})
        if "county" in dfp.columns:
            dfp["county_norm"] = dfp["county"].map(normalize_county_name)
        if "year" in dfp.columns:
            dfp["year"] = dfp["year"].astype(int)
        pred_dfs.append(dfp)
        debug_info.append(dbg)

    if not pred_dfs:
        raise RuntimeError("No predictions loaded. Check seasons/model names and S3 paths.")

    pred_all = pd.concat(pred_dfs, ignore_index=True)

    # Some SageMaker outputs may not include county/year columns (single-column prediction).
    # In that case, you must join back with a 'scoring manifest' / input order.
    has_id_cols = ("county_norm" in pred_all.columns) and ("year" in pred_all.columns)
    if not has_id_cols:
        st.warning(
            "Predictions loaded, but missing identifier columns (county/year). "
            "Your prediction outputs look like raw SageMaker CSV (single-column). "
            "For a full demo, ensure your transform output includes county + year columns "
            "(or you keep the input manifest order and merge back)."
        )

    # Actuals (optional)
    actuals_df = None
    try:
        actuals_df = load_actuals(region, actuals_uri)
    except Exception as e:
        st.info(f"Actuals not loaded (optional): {e}")

    # Frozen features (optional)
    features_by_season = {}
    if use_features:
        for s in seasons:
            try:
                fprefix = (
                    f"s3://{bucket}/features_frozen/"
                    f"state_fips={state_fips}/"
                    f"county_fips={county_fips}/"
                    f"feature_season={s}/"
                    f"run_date={run_date}/"
                )
                features_by_season[s] = load_features_frozen(region, fprefix)
            except Exception as e:
                features_by_season[s] = None

    return pred_all, actuals_df, features_by_season, debug_info


if load_btn:
    st.cache_data.clear()

# Load on first render as well (best-effort)
if "data_loaded" not in st.session_state or load_btn:
    with st.spinner("Loading data from S3..."):
        try:
            pred_all, actuals_df, features_by_season, debug_info = _load_all()
            st.session_state["pred_all"] = pred_all
            st.session_state["actuals_df"] = actuals_df
            st.session_state["features_by_season"] = features_by_season
            st.session_state["debug_info"] = debug_info
            st.session_state["data_loaded"] = True
        except Exception as e:
            st.session_state["data_loaded"] = False
            st.error(f"Failed to load data: {e}")

if not st.session_state.get("data_loaded", False):
    st.stop()

pred_all: pd.DataFrame = st.session_state["pred_all"]
actuals_df: Optional[pd.DataFrame] = st.session_state["actuals_df"]
features_by_season: Dict[str, Optional[pd.DataFrame]] = st.session_state["features_by_season"]
debug_info: List[Dict] = st.session_state["debug_info"]


# ----------------------------
# Debug panel
# ----------------------------
with st.expander("Debug: S3 paths / files found"):
    st.json(debug_info)

# ----------------------------
# Quick integrity checks
# ----------------------------
st.subheader("Data sanity checks")

c1, c2, c3 = st.columns(3)
with c1:
    st.metric("Rows (predictions)", f"{len(pred_all):,}")
with c2:
    st.metric("Seasons", ", ".join(sorted(pred_all["feature_season"].unique().tolist())))
with c3:
    yr = pred_all["year"].dropna().unique().tolist() if "year" in pred_all.columns else []
    st.metric("Years present", ", ".join(map(str, sorted(yr))) if yr else "N/A")

if "year" in pred_all.columns and "county_norm" in pred_all.columns:
    chk = (
        pred_all[pred_all["year"] == feature_year]
        .groupby(["feature_season"])["county_norm"]
        .nunique()
        .sort_index()
        .reset_index()
        .rename(columns={"county_norm": "unique_counties_predicted"})
    )
    st.dataframe(chk, use_container_width=True)

    dups = pred_all[pred_all["year"] == feature_year].duplicated(
        subset=["feature_season", "county_norm", "year"], keep=False
    )
    if dups.any():
        st.warning(f"Found {int(dups.sum())} duplicate county-season-year rows in predictions. Consider deduping.")
else:
    st.info("County/year columns not found in prediction outputs; some visualizations will be limited.")


# ----------------------------
# Tabs
# ----------------------------
tab_overview, tab_cutoffs, tab_metrics, tab_drivers, tab_report = st.tabs(
    ["Overview", "Cutoff comparison", "Metrics & Fit", "Driver signals", "Downloadable report"]
)

with tab_overview:
    st.markdown("### What to show in your final demo (best-case scenario)")
    st.markdown(
        """
**Recommended narrative (5–7 minutes):**
1) **Problem**: Forecast Iowa county-level corn yield using remote sensing + weather + storm risk.  
2) **Data**: NDVI (vegetation), ERA5-Land (weather), Storm events, USDA yield labels.  
3) **Models**: Multiple **cutoff-date** models (Jun01 → Aug15) with expanding information set.  
4) **Results**: Show cutoff comparison + top/bottom counties + (if available) actual-vs-predicted + error metrics.  
5) **Interpretability**: Driver signals (NDVI peak/slope, heat days, moisture stress, storm wind days).  
6) **Deployment**: S3 + containers + Step Functions + SageMaker batch transform (already done).  
        """
    )

    st.markdown("### Preview of predictions")
    st.dataframe(pred_all.head(50), use_container_width=True)

with tab_cutoffs:
    _plotly_or_fallback_warning()

    st.markdown("### Distribution of predicted yield (2025) by cutoff")
    if _HAS_PLOTLY and "year" in pred_all.columns and "prediction" in pred_all.columns:
        p2025 = pred_all[pred_all["year"] == feature_year].copy()
        fig = px.histogram(
            p2025,
            x="prediction",
            color="feature_season",
            nbins=30,
            barmode="overlay",
            opacity=0.45,
            title=f"Predicted yield distribution ({feature_year}) by cutoff",
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Need Plotly + year/prediction columns for this chart.")

    st.markdown("### County rankings (top/bottom) by cutoff")
    if "year" in pred_all.columns and "county_norm" in pred_all.columns and "prediction" in pred_all.columns:
        n = st.slider("Top/bottom N", min_value=5, max_value=25, value=10, step=1)
        tbl = pd.concat([top_bottom(pred_all, s, feature_year, n=n) for s in seasons], ignore_index=True)
        st.dataframe(tbl.sort_values(["feature_season", "rank", "prediction"], ascending=[True, True, False]), use_container_width=True)
    else:
        st.info("Need county_norm + year + prediction columns for rankings.")

    st.markdown("### Cutoff agreement (how stable are predictions across cutoffs?)")
    if "year" in pred_all.columns and "county_norm" in pred_all.columns and "prediction" in pred_all.columns:
        p = pred_all[pred_all["year"] == feature_year].copy()
        piv = p.pivot_table(index="county_norm", columns="feature_season", values="prediction", aggfunc="mean")
        st.dataframe(piv.head(25), use_container_width=True)
        if _HAS_PLOTLY:
            corr = piv.corr()
            fig2 = px.imshow(corr, text_auto=True, title=f"Correlation between cutoffs (predictions) — {feature_year}")
            st.plotly_chart(fig2, use_container_width=True)

with tab_metrics:
    st.markdown("### Actual vs Predicted + error metrics (if actuals are available)")
    if actuals_df is None:
        st.info("Actuals not loaded (optional). Point actuals_uri to a valid curated actuals parquet.")
    elif not {"county_norm","year","yield_bu_acre"}.issubset(actuals_df.columns):
        st.warning("Actuals loaded but missing required columns (county_norm, year, yield_bu_acre).")
    elif not {"county_norm","year","prediction","feature_season"}.issubset(pred_all.columns):
        st.warning("Predictions missing required columns to merge with actuals.")
    else:
        merged = pred_all.merge(
            actuals_df[["county_norm", "year", "yield_bu_acre"]],
            on=["county_norm", "year"],
            how="inner"
        )
        merged = merged.dropna(subset=["yield_bu_acre", "prediction"])
        if merged.empty:
            st.warning("No overlap between predictions and actuals (check year/state/county).")
        else:
            met = compute_metrics(merged)
            st.dataframe(met, use_container_width=True)

            if _HAS_PLOTLY:
                season_sel = st.selectbox("Pick cutoff to visualize", options=sorted(merged["feature_season"].unique().tolist()))
                g = merged[merged["feature_season"] == season_sel].copy()

                fig = px.scatter(
                    g,
                    x="yield_bu_acre",
                    y="prediction",
                    hover_name="county_norm",
                    trendline="ols",
                    title=f"Actual vs Predicted — {season_sel} — {feature_year}",
                    labels={"yield_bu_acre": "Actual yield (bu/acre)", "prediction": "Predicted yield (bu/acre)"},
                )
                st.plotly_chart(fig, use_container_width=True)

                g["residual"] = g["prediction"] - g["yield_bu_acre"]
                fig2 = px.histogram(g, x="residual", nbins=35, title=f"Residual distribution — {season_sel}")
                st.plotly_chart(fig2, use_container_width=True)

with tab_drivers:
    st.markdown("### Driver signals (optional): NDVI / Weather / Storm features used for inference")
    st.caption("Loads from features_frozen/... and correlates drivers with predictions for quick interpretability.")

    if not use_features:
        st.info("Enable 'Load frozen features (drivers)' in the sidebar to use this tab.")
    else:
        season_sel = st.selectbox("Cutoff season", options=seasons, index=0)
        fdf = features_by_season.get(season_sel)

        if fdf is None:
            st.warning(f"No frozen features found/loaded for season={season_sel}. Check S3 path or disable this tab.")
        else:
            st.write(f"Frozen features rows: {len(fdf):,}")
            st.dataframe(fdf.head(25), use_container_width=True)

            if "year" in pred_all.columns and "county_norm" in pred_all.columns:
                p = pred_all[(pred_all["feature_season"] == season_sel) & (pred_all["year"] == feature_year)].copy()
                join_cols = ["county_norm", "year"]
                drivers = fdf.copy()

                # pick a small set of common driver columns if present
                candidate_drivers = [
                    "ndvi_peak", "ndvi_slope", "ndvi_auc", "ndvi_at_cutoff",
                    "heat_days_gt32", "rain_sum_mm", "temp_anomaly", "net_moisture_stress",
                    "wind_severe_days_58_cutoff", "hail_days", "tornado_days"
                ]
                driver_cols = [c for c in candidate_drivers if c in drivers.columns]

                if not driver_cols:
                    st.info("No known driver columns found. Select any numeric columns below.")
                    numeric_cols = drivers.select_dtypes(include=[np.number]).columns.tolist()
                    driver_cols = st.multiselect("Pick numeric driver columns", options=numeric_cols, default=numeric_cols[:6])

                dsmall = drivers[join_cols + driver_cols].copy()
                j = p.merge(dsmall, on=join_cols, how="inner").dropna(subset=["prediction"])
                if j.empty:
                    st.warning("No overlap between predictions and frozen features (check keys).")
                else:
                    st.markdown("#### Driver correlation with prediction")
                    corr_rows = []
                    for c in driver_cols:
                        if c in j.columns and pd.api.types.is_numeric_dtype(j[c]):
                            corr_rows.append({"driver": c, "corr_with_pred": float(j[c].corr(j["prediction"]))})
                    corr_df = pd.DataFrame(corr_rows).sort_values("corr_with_pred", ascending=False)
                    st.dataframe(corr_df, use_container_width=True)

                    if _HAS_PLOTLY and len(driver_cols) > 0:
                        driver_pick = st.selectbox("Scatter plot driver vs prediction", options=driver_cols, index=0)
                        fig = px.scatter(
                            j, x=driver_pick, y="prediction",
                            hover_name="county_norm",
                            title=f"{driver_pick} vs prediction ({season_sel}, {feature_year})"
                        )
                        st.plotly_chart(fig, use_container_width=True)

with tab_report:
    st.markdown("### Downloadable HTML report")
    st.caption("Generates a single self-contained HTML file with key charts + tables for your professor demo.")

    def build_report_html() -> str:
        # core summary
        html_parts = []
        html_parts.append("<html><head><meta charset='utf-8'><title>GeoAI Capstone Report</title></head><body>")
        html_parts.append(f"<h1>GeoAI Crop Yield Prediction — Report</h1>")
        html_parts.append(f"<p><b>Bucket:</b> {bucket} &nbsp; <b>State FIPS:</b> {state_fips} &nbsp; <b>County FIPS:</b> {county_fips} &nbsp; <b>Run date:</b> {run_date}</p>")
        html_parts.append(f"<p><b>Seasons:</b> {', '.join(seasons)}</p>")

        # metrics
        if actuals_df is not None and {"county_norm","year","yield_bu_acre"}.issubset(actuals_df.columns) and {"county_norm","year","prediction","feature_season"}.issubset(pred_all.columns):
            merged = pred_all.merge(actuals_df[["county_norm","year","yield_bu_acre"]], on=["county_norm","year"], how="inner")
            merged = merged.dropna(subset=["yield_bu_acre", "prediction"])
            if not merged.empty:
                met = compute_metrics(merged)
                html_parts.append("<h2>Metrics (Actual vs Predicted)</h2>")
                html_parts.append(met.to_html(index=False))

        # top/bottom
        if {"county_norm","year","prediction","feature_season"}.issubset(pred_all.columns):
            tbl = pd.concat([top_bottom(pred_all, s, feature_year, n=10) for s in seasons], ignore_index=True)
            html_parts.append("<h2>Top/Bottom Counties (2025)</h2>")
            html_parts.append(tbl.to_html(index=False))

        # include key charts if plotly exists
        if _HAS_PLOTLY and "year" in pred_all.columns and "prediction" in pred_all.columns:
            p2025 = pred_all[pred_all["year"] == feature_year].copy()
            fig = px.histogram(p2025, x="prediction", color="feature_season", nbins=30, barmode="overlay", opacity=0.45,
                               title=f"Predicted yield distribution ({feature_year}) by cutoff")
            html_parts.append("<h2>Predicted Yield Distribution</h2>")
            html_parts.append(fig.to_html(full_html=False, include_plotlyjs="cdn"))

        html_parts.append("</body></html>")
        return "\n".join(html_parts)

    html = build_report_html().encode("utf-8")
    st.download_button(
        label="Download HTML report",
        data=html,
        file_name=f"geoai_capstone_report_{state_fips}_{county_fips}_{run_date}.html",
        mime="text/html",
        use_container_width=True
    )

