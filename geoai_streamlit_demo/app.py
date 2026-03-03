"""
GeoAI Capstone Demo App (Streamlit) — Updated
--------------------------------------------
Updates requested by Mohan:
1) Load predictions from your predictions outputs in S3, e.g.
   s3://geoai-demo-data/predictions/state_fips=<state_fips>/county_fips=ALL/predict_year=<year>/feature_season=<season>/run_date=<run_date>/model=<model>/
   (supports parquet/csv; searches recursively for a prediction column)
2) UI controls: County + Season + Model + Run date + Year range (2020–2025)
3) "Observed vs Mean Prediction" line chart for 2020–2025 (like your screenshot, but simplified)
4) Optional: Trigger an AWS Step Functions state machine for the selected (run_date, model, season)
5) Download a PDF report (plot + table + (optional) metrics)

Run:
  pip install -r requirements.txt
  streamlit run app.py

AWS Auth:
- Use your usual AWS credentials (env vars, ~/.aws/credentials, or IAM role)
- Region defaults to ap-south-1
"""

from __future__ import annotations

import io
import os
from dataclasses import dataclass
from datetime import date
from typing import Dict, List, Optional, Tuple
from urllib.parse import urlparse

import numpy as np
import pandas as pd
from html import escape as html_escape
import streamlit as st

# Optional (recommended)
try:
    import plotly.express as px
    import plotly.graph_objects as go
    _HAS_PLOTLY = True
except Exception:
    _HAS_PLOTLY = False

# S3 + Step Functions
try:
    import boto3
    import s3fs  # noqa: F401 (pandas uses it for s3://)
    _HAS_AWS = True
except Exception:
    _HAS_AWS = False

# PDF
try:
    from reportlab.lib.pagesizes import letter
    from reportlab.lib.units import inch
    from reportlab.pdfgen import canvas
    from reportlab.lib.utils import ImageReader
    _HAS_PDF = True
except Exception:
    _HAS_PDF = False

# Matplotlib just for exporting the plot image into the PDF
try:
    import matplotlib.pyplot as plt
    _HAS_MPL = True
except Exception:
    _HAS_MPL = False


# ----------------------------
# Secrets helper (optional)
# ----------------------------

def load_aws_secrets_into_env() -> None:
    """
    Supports Streamlit Cloud / local secrets.
    If st.secrets has AWS creds, export them into env vars so boto3/pandas+s3fs can use them.
    """
    try:
        secrets = st.secrets
        _ = list(secrets.keys()) if hasattr(secrets, "keys") else None
    except Exception:
        return

    if "AWS_ACCESS_KEY_ID" in secrets and "AWS_SECRET_ACCESS_KEY" in secrets:
        os.environ["AWS_ACCESS_KEY_ID"] = secrets["AWS_ACCESS_KEY_ID"]
        os.environ["AWS_SECRET_ACCESS_KEY"] = secrets["AWS_SECRET_ACCESS_KEY"]
        os.environ["AWS_DEFAULT_REGION"] = secrets.get("AWS_DEFAULT_REGION", "ap-south-1")
        if "AWS_SESSION_TOKEN" in secrets:
            os.environ["AWS_SESSION_TOKEN"] = secrets["AWS_SESSION_TOKEN"]

load_aws_secrets_into_env()


# ----------------------------
# Utilities
# ----------------------------

def normalize_county_name(x: str) -> str:
    if pd.isna(x):
        return x
    s = str(x).strip().lower()
    s = s.replace(" county", "")
    s = " ".join(s.split())  # collapse repeated whitespace
    return s

def ensure_columns(df: pd.DataFrame, rename_map: dict) -> pd.DataFrame:
    for k, v in rename_map.items():
        if k in df.columns and v not in df.columns:
            df = df.rename(columns={k: v})
    return df

def dedupe_columns(df: pd.DataFrame) -> pd.DataFrame:
    return df.loc[:, ~df.columns.duplicated()].copy()

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



def json_dumps(obj) -> str:
    import json
    return json.dumps(obj, separators=(",", ":"), default=str)

def _require_aws():
    if not _HAS_AWS:
        st.error("Missing AWS deps. Install: pip install boto3 s3fs")
        st.stop()

@st.cache_resource
def _s3_client(region_name: str):
    _require_aws()
    return boto3.client("s3", region_name=region_name)

@st.cache_resource
def _sf_client(region_name: str):
    _require_aws()
    return boto3.client("stepfunctions", region_name=region_name)

def s3_list_files_under_prefix(region: str, prefix: str, exts=(".parquet", ".csv")) -> List[str]:
    """
    Returns full s3:// URIs for objects under the given s3:// prefix.
    """
    _require_aws()
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
    _require_aws()
    return pd.read_parquet(uri, engine="pyarrow")

@st.cache_data(show_spinner=False)
def s3_read_csv(uri: str, header: Optional[int] = "infer") -> pd.DataFrame:
    _require_aws()
    return pd.read_csv(uri, header=header)

def _read_any_file(path_or_s3uri: str) -> pd.DataFrame:
    p = path_or_s3uri.lower()
    if p.endswith(".parquet"):
        return s3_read_parquet(path_or_s3uri)
    if p.endswith(".csv"):
        try:
            df = s3_read_csv(path_or_s3uri, header="infer")
        except Exception:
            df = s3_read_csv(path_or_s3uri, header=None)
        # handle 1-col output
        if df.shape[1] == 1:
            df.columns = ["prediction"]
        return df
    raise ValueError(f"Unsupported file type: {path_or_s3uri}")


# ----------------------------
# Loaders
# ----------------------------

PRED_COL_CANDIDATES = [
    "prediction", "pred", "yhat", "y_pred", "predicted_yield", "mean_prediction", "mean_pred", "Predicted"
]
COUNTY_COL_CANDIDATES = ["county", "county_name", "County"]
YEAR_COL_CANDIDATES = ["year", "YEAR", "Year", "predict_year"]

def _first_present(cols: List[str], candidates: List[str]) -> Optional[str]:
    setcols = set(cols)
    for c in candidates:
        if c in setcols:
            return c
    return None

@st.cache_data(show_spinner=False)
def load_predictions_from_predictions_s3(
    region: str,
    bucket: str,
    state_fips: str,
    predict_year: int,
    # these are selection filters; applied if columns exist or if the path contains them
    feature_season: str,
    run_date: str,
    model_name: str,
    county_fips: str = "ALL",
) -> Tuple[pd.DataFrame, Dict]:
    """
    Reads prediction rows from your *predictions* folder structure, e.g.

      s3://geoai-demo-data/predictions/
        state_fips=19/
        county_fips=ALL/
        predict_year=2025/
        feature_season=jun01/
        run_date=2026-02-27/
        model=Jun01_LightGBM-limited_withstorm/
        predictions.csv

    This function searches recursively for parquet/csv under the *exact* combination prefix.

    Why you saw the error:
    - The earlier app version was reading *features_frozen* (feature rows), not *predictions*.
      Those files contain engineered features, so there is no prediction column.
    """
    # Your S3 layout encodes selections in the path, including `model=<name>`.
    base_prefix = (
        f"s3://{bucket}/predictions/"
        f"state_fips={state_fips}/"
        f"county_fips={county_fips}/"
        f"predict_year={int(predict_year)}/"
        f"feature_season={feature_season}/"
        f"run_date={run_date}/"
        f"model={model_name}/"
    )

    dbg = {
        "base_prefix": base_prefix,
        "feature_season": feature_season,
        "run_date": run_date,
        "model_name": model_name,
    }

    files = s3_list_files_under_prefix(region, base_prefix, exts=(".parquet", ".csv"))
    dbg["files_found"] = len(files)
    dbg["files_preview"] = files[:10]

    if not files:
        raise FileNotFoundError(f"No parquet/csv found under: {base_prefix}")

    dfs = []
    for f in files:
        try:
            dfs.append(_read_any_file(f))
        except Exception:
            # Skip unreadable fragments rather than failing the whole demo
            continue

    if not dfs:
        raise FileNotFoundError(f"Files exist but none were readable under: {base_prefix}")

    df = pd.concat(dfs, ignore_index=True)
    df = dedupe_columns(df)

    # normalize key columns
    pred_col = _first_present(list(df.columns), PRED_COL_CANDIDATES)
    if pred_col and pred_col != "prediction":
        df = df.rename(columns={pred_col: "prediction"})
    if "prediction" not in df.columns:
        # last-resort: if single numeric col, treat it as prediction
        num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
        if len(num_cols) == 1:
            df = df.rename(columns={num_cols[0]: "prediction"})
        else:
            raise ValueError(
                f"Could not find a prediction column. Expected one of {PRED_COL_CANDIDATES}. "
                f"Found columns: {list(df.columns)[:40]}"
            )

    county_col = _first_present(list(df.columns), COUNTY_COL_CANDIDATES)
    if county_col and county_col != "county":
        df = df.rename(columns={county_col: "county"})

    year_col = _first_present(list(df.columns), YEAR_COL_CANDIDATES)
    if year_col and year_col != "year":
        df = df.rename(columns={year_col: "year"})

    # ensure year
    if "year" not in df.columns:
        df["year"] = int(predict_year)
    df["year"] = df["year"].astype(int)

    # ensure county_norm
    if "county" in df.columns:
        df["county_norm"] = df["county"].map(normalize_county_name)

    # attach selection metadata for reporting
    df["feature_season"] = feature_season
    df["run_date"] = run_date
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

    # --- normalize keys used for join ---
    if "county" in df.columns:
        df["county"] = df["county"].astype(str)
        df["county_norm"] = df["county"].map(normalize_county_name)

    if "year" in df.columns:
        df["year"] = pd.to_numeric(df["year"], errors="coerce").astype("Int64")

    # --- IMPORTANT: standardize observed yield column name for the dashboard ---
    # Keep yield_bu_acre, but ALSO provide observed_yield
    if "yield_bu_acre" in df.columns:
        df["observed_yield"] = pd.to_numeric(df["yield_bu_acre"], errors="coerce")

    return df


# ----------------------------
# Plot helpers
# ----------------------------

def build_observed_vs_pred_series(
    pred_df: pd.DataFrame,
    actuals_df: Optional[pd.DataFrame],
    county_norm: str,
    years: List[int],
) -> pd.DataFrame:
    """
    Returns a tidy DataFrame:
      year, observed_yield, mean_prediction

    Special handling:
    - If county_norm is one of {"__statewide__", "iowa", "statewide", "all"}, compute STATEWIDE series
      by aggregating across all counties for each year.
    """
    # Identify special statewide selection
    is_statewide = str(county_norm).strip().lower() in {"__statewide__", "iowa", "statewide", "all"}

    # Base x-axis (ensures all requested years appear)
    base = pd.DataFrame({"year": [int(y) for y in years]})

    # Predictions (county or statewide)
    if "county_norm" not in pred_df.columns and "county" in pred_df.columns:
        pred_df = pred_df.copy()
        pred_df["county_norm"] = pred_df["county"].map(normalize_county_name)

    p = pred_df[pred_df["year"].isin(years)].copy()

    # Drop accidental statewide rows before computing statewide mean
    p = p[~p["county_norm"].isin(["iowa", "statewide", "all"])]

    if is_statewide:
        pred_series = (
            p.groupby("year")["prediction"]
             .mean()
             .reset_index()
             .rename(columns={"prediction": "mean_prediction"})
        )
    else:
        pred_series = (
            p[p["county_norm"] == county_norm]
            .groupby("year")["prediction"]
            .mean()
            .reset_index()
            .rename(columns={"prediction": "mean_prediction"})
        )

    out = base.merge(pred_series, on="year", how="left")

    # Actuals (county or statewide)
    if actuals_df is not None and "year" in actuals_df.columns:
        obs_col = "observed_yield" if "observed_yield" in actuals_df.columns else "yield_bu_acre"
        if obs_col in actuals_df.columns:
            a = actuals_df[actuals_df["year"].isin(years)].copy()

            if "county_norm" not in a.columns and "county" in a.columns:
                a["county_norm"] = a["county"].map(normalize_county_name)

            if is_statewide:
                a_series = (
                    a.groupby("year")[obs_col]
                     .mean()
                     .reset_index()
                     .rename(columns={obs_col: "observed_yield"})
                )
            else:
                a_series = (
                    a[a["county_norm"] == county_norm]
                    .groupby("year")[obs_col]
                    .mean()
                    .reset_index()
                    .rename(columns={obs_col: "observed_yield"})
                )

            out = out.merge(a_series, on="year", how="left")
        else:
            out["observed_yield"] = np.nan
    else:
        out["observed_yield"] = np.nan

    # final tidy
    out["year"] = pd.to_numeric(out["year"], errors="coerce").astype("Int64")
    out["mean_prediction"] = pd.to_numeric(out["mean_prediction"], errors="coerce")
    out["observed_yield"] = pd.to_numeric(out["observed_yield"], errors="coerce")
    return out



def plot_observed_vs_pred_plotly(df: pd.DataFrame, title: str):
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df["year"], y=df["mean_prediction"], mode="lines+markers",
        name="Mean Prediction"
    ))
    if df["observed_yield"].notna().any():
        fig.add_trace(go.Scatter(
            x=df["year"], y=df["observed_yield"], mode="lines+markers",
            name="Observed Yield"
        ))
    fig.update_layout(
        title=title,
        xaxis_title="Year",
        yaxis_title="Yield (bu/acre)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=20, r=20, t=50, b=20),
    )
    return fig


def fig_to_png_bytes_matplotlib(df: pd.DataFrame, title: str) -> bytes:
    """
    For PDF export: render a clean plot with matplotlib and return PNG bytes.
    """
    if not _HAS_MPL:
        raise RuntimeError("matplotlib is required for PDF export plot rendering.")

    fig, ax = plt.subplots(figsize=(8.5, 3.7))
    ax.plot(df["year"], df["mean_prediction"], marker="o", label="Mean Prediction")
    if df["observed_yield"].notna().any():
        ax.plot(df["year"], df["observed_yield"], marker="o", label="Observed Yield")
    ax.set_title(title)
    ax.set_xlabel("Year")
    ax.set_ylabel("Yield (bu/acre)")
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.legend(loc="upper left")
    buf = io.BytesIO()
    fig.tight_layout()
    fig.savefig(buf, format="png", dpi=200)
    plt.close(fig)
    return buf.getvalue()


def build_pdf_report_bytes(
    series_df: pd.DataFrame,
    title: str,
    params: Dict[str, str],
    metrics: Optional[Dict[str, float]] = None,
) -> bytes:
    if not _HAS_PDF:
        raise RuntimeError("reportlab is required for PDF export. Install: pip install reportlab")
    if not _HAS_MPL:
        raise RuntimeError("matplotlib is required for PDF export. Install: pip install matplotlib")

    png = fig_to_png_bytes_matplotlib(series_df, title=title)

    buf = io.BytesIO()
    c = canvas.Canvas(buf, pagesize=letter)
    W, H = letter

    y = H - 0.75 * inch
    c.setFont("Helvetica-Bold", 16)
    c.drawString(0.75 * inch, y, title)
    y -= 0.35 * inch

    c.setFont("Helvetica", 10)
    for k, v in params.items():
        c.drawString(0.75 * inch, y, f"{k}: {v}")
        y -= 0.18 * inch

    if metrics:
        y -= 0.10 * inch
        c.setFont("Helvetica-Bold", 11)
        c.drawString(0.75 * inch, y, "Metrics (Observed vs Mean Prediction, overlapping years only)")
        y -= 0.22 * inch
        c.setFont("Helvetica", 10)
        for k, v in metrics.items():
            c.drawString(0.9 * inch, y, f"{k}: {v:.3f}" if isinstance(v, (float, int)) and not np.isnan(v) else f"{k}: N/A")
            y -= 0.18 * inch

    # plot image
    y -= 0.15 * inch
    img_w = W - 1.5 * inch
    img_h = 3.2 * inch
    img_x = 0.75 * inch
    img_y = y - img_h
    # Use ImageReader to safely embed PNG bytes (avoids PIL format AttributeError)
    try:
        c.drawImage(ImageReader(io.BytesIO(png)), img_x, img_y, width=img_w, height=img_h, preserveAspectRatio=True, mask='auto')
    except Exception:
        # If image embedding fails, skip the chart instead of crashing the app
        c.setFont("Helvetica-Oblique", 10)
        c.drawString(0.75 * inch, y, "[Chart unavailable in PDF export]")

    y = img_y - 0.35 * inch

    # data table
    c.setFont("Helvetica-Bold", 11)
    c.drawString(0.75 * inch, y, "Yearly values")
    y -= 0.25 * inch
    c.setFont("Helvetica", 10)
    headers = ["Year", "Observed", "Mean Prediction"]
    c.drawString(0.75 * inch, y, f"{headers[0]:<6} {headers[1]:>12} {headers[2]:>16}")
    y -= 0.18 * inch
    c.setFont("Helvetica", 10)
    for _, row in series_df.iterrows():
        yy = int(row["year"])
        obs = row["observed_yield"]
        pred = row["mean_prediction"]
        obs_s = f"{obs:.1f}" if pd.notna(obs) else "NA"
        pred_s = f"{pred:.1f}" if pd.notna(pred) else "NA"
        c.drawString(0.75 * inch, y, f"{yy:<6} {obs_s:>12} {pred_s:>16}")
        y -= 0.18 * inch
        if y < 0.9 * inch:
            c.showPage()
            y = H - 0.75 * inch

    c.showPage()
    c.save()
    return buf.getvalue()





def build_html_report_str(
    series_df: pd.DataFrame,
    title: str,
    params: Dict[str, str],
    metrics: Optional[Dict[str, float]] = None,
    fig: Optional[object] = None,
) -> str:
    """Build a single self-contained HTML report string."""
    # Basic CSS for readability
    css = """
    <style>
      body { font-family: -apple-system,BlinkMacSystemFont,Segoe UI,Roboto,Helvetica,Arial,sans-serif; margin: 24px; }
      h1 { margin: 0 0 8px 0; }
      .meta { color: #555; margin-bottom: 18px; }
      .grid { display: grid; grid-template-columns: 1fr 1fr; gap: 12px; margin: 12px 0 18px 0; }
      .card { border: 1px solid #ddd; border-radius: 10px; padding: 12px; }
      .card h3 { margin: 0 0 6px 0; font-size: 14px; color:#333; }
      table { border-collapse: collapse; width: 100%; font-size: 13px; }
      th, td { border: 1px solid #e5e5e5; padding: 6px 8px; text-align: right; }
      th:first-child, td:first-child { text-align: left; }
      th { background: #fafafa; }
      .note { color: #666; font-size: 12px; margin-top: 10px; }
    </style>
    """

    # Params table
    meta_rows = "".join([f"<tr><th>{html_escape(k)}</th><td style='text-align:left'>{html_escape(str(v))}</td></tr>" for k, v in params.items()])
    meta_table = f"<table><tbody>{meta_rows}</tbody></table>"

    # Metrics table
    metrics_html = ""
    if metrics:
        m_rows = "".join([f"<tr><th>{html_escape(k)}</th><td>{'' if v is None else f'{v:.4f}'}</td></tr>" for k, v in metrics.items()])
        metrics_html = f"<table><tbody>{m_rows}</tbody></table>"
    else:
        metrics_html = "<div class='note'>Metrics not available (need observed yields for the selected range).</div>"

    # Figure HTML (Plotly if available)
    fig_html = ""
    if fig is not None and hasattr(fig, "to_html"):
        try:
            fig_html = fig.to_html(include_plotlyjs="cdn", full_html=False)
        except Exception:
            fig_html = ""
    if not fig_html:
        fig_html = "<div class='note'>Chart unavailable (Plotly not installed or figure rendering failed).</div>"

    # Series table
    table_df = series_df.copy()
    # consistent column ordering if present
    cols = [c for c in ["year", "mean_prediction", "prediction", "observed_yield"] if c in table_df.columns]
    if cols:
        table_df = table_df[cols]
    series_table = table_df.to_html(index=False, border=0, classes="dataframe")

    html = f"""
    <!doctype html>
    <html>
    <head>
      <meta charset='utf-8' />
      <title>{html_escape(title)}</title>
      {css}
    </head>
    <body>
      <h1>{html_escape(title)}</h1>
      <div class='meta'>Generated: {html_escape(pd.Timestamp.utcnow().strftime('%Y-%m-%d %H:%M UTC'))}</div>

      <div class='grid'>
        <div class='card'>
          <h3>Run parameters</h3>
          {meta_table}
        </div>
        <div class='card'>
          <h3>Quick metrics</h3>
          {metrics_html}
        </div>
      </div>

      <div class='card'>
        <h3>Observed vs Mean Prediction</h3>
        {fig_html}
      </div>

      <div class='card' style='margin-top:12px'>
        <h3>Yearly values</h3>
        {series_table}
        <div class='note'>Note: Observed yield may be missing for the latest year (e.g., 2025) until USDA publishes it.</div>
      </div>
    </body>
    </html>
    """
    return html



def compute_series_metrics(series_df: pd.DataFrame) -> Optional[Dict[str, float]]:
    d = series_df.dropna(subset=["observed_yield", "mean_prediction"]).copy()
    if len(d) < 2:
        return None
    y_true = d["observed_yield"].astype(float).values
    y_pred = d["mean_prediction"].astype(float).values
    return {
        "RMSE": metric_rmse(y_true, y_pred),
        "MAE": metric_mae(y_true, y_pred),
        "MAPE_%": metric_mape(y_true, y_pred),
        "R2": metric_r2(y_true, y_pred),
    }


# ----------------------------
# Streamlit UI
# ----------------------------

st.set_page_config(page_title="GeoAI Yield Risk Demo", layout="wide")

st.title("GeoAI Early Yield Predection")
st.caption("County-wise observed vs predicted yields for Iowa (2020–2025)")

with st.sidebar:
    st.header("Settings")

    region = st.text_input("AWS Region", value=os.getenv("AWS_REGION", "ap-south-1"))
    bucket = st.text_input("S3 Bucket", value=os.getenv("GEOAI_BUCKET", "geoai-demo-data"))

    state_fips = st.text_input("State FIPS", value=os.getenv("STATE_FIPS", "19"))
    county_fips = st.text_input("County FIPS (data partition)", value=os.getenv("COUNTY_FIPS", "ALL"))

    st.markdown("---")
    st.subheader("Run selection")
    run_date = st.text_input("run_date", value=os.getenv("RUN_DATE", str(date.today())))
    default_seasons = ["jun01", "jul01", "jul15", "aug01", "aug15"]
    feature_season = st.selectbox("Season (cutoff)", options=default_seasons, index=0)

    model_name = st.text_input("Model name", value=os.getenv("MODEL_NAME", "Jun01_LightGBM-limited_withstorm"))

    st.markdown("---")
    st.subheader("Years")
    start_year = st.number_input("Start year", min_value=2010, max_value=2100, value=2020, step=1)
    end_year = st.number_input("End year", min_value=2010, max_value=2100, value=2025, step=1)
    years = list(range(int(start_year), int(end_year) + 1))

    st.markdown("---")
    st.subheader("Actuals (Observed yields)")
    actuals_uri_default = f"s3://{bucket}/curated/yield/state_fips={state_fips}/county_fips={county_fips}/actuals.parquet"
    actuals_uri = st.text_input("Actuals parquet URI", value=actuals_uri_default)

    st.markdown("---")
    load_btn = st.button("Load / Refresh", type="primary")

    st.markdown("---")
    st.subheader("Optional: Trigger Step Functions")
    sf_arn = st.text_input("State machine ARN", value=os.getenv("STATE_MACHINE_ARN", ""))
    do_trigger = st.checkbox("Show trigger controls", value=False)


if load_btn:
    st.cache_data.clear()

# --------------- Load data ---------------

@st.cache_data(show_spinner=False)
def _load_all_years(
    region: str,
    bucket: str,
    state_fips: str,
    county_fips: str,
    years: List[int],
    feature_season: str,
    run_date: str,
    model_name: str,
    actuals_uri: str,
) -> Tuple[pd.DataFrame, Optional[pd.DataFrame], List[Dict]]:
    pred_all = []
    dbg_all = []
    for y in years:
        d, dbg = load_predictions_from_predictions_s3(
            region=region,
            bucket=bucket,
            state_fips=state_fips,
            county_fips=county_fips,
            predict_year=int(y),
            feature_season=feature_season,
            run_date=run_date,
            model_name=model_name,
        )
        pred_all.append(d)
        dbg_all.append(dbg)
    pred_all = pd.concat(pred_all, ignore_index=True)

    # actuals (optional but recommended)
    actuals_df = None
    try:
        actuals_df = load_actuals(region, actuals_uri)
    except Exception:
        actuals_df = None

    return pred_all, actuals_df, dbg_all


with st.spinner("Loading data from S3..."):
    try:
        pred_all, actuals_df, dbg_all = _load_all_years(
            region, bucket, state_fips, county_fips, years, feature_season, run_date, model_name, actuals_uri
        )
    except Exception as e:
        st.error(f"Failed to load predictions: {e}")
        st.stop()

# --------------- County selection ---------------

# We treat "Statewide (Iowa)" as a special option and compute statewide series by aggregating across counties.
STATEWIDE_KEY = "__statewide__"
STATEWIDE_LABEL = "Statewide (Iowa)"

county_options: List[str] = []
if "county_norm" in pred_all.columns:
    county_options = sorted([c for c in pred_all["county_norm"].dropna().unique().tolist() if str(c).strip() != ""])
elif actuals_df is not None and "county_norm" in actuals_df.columns:
    county_options = sorted([c for c in actuals_df["county_norm"].dropna().unique().tolist() if str(c).strip() != ""])

# Remove any accidental statewide-like keys from the list (we add a single clean option)
county_options = [c for c in county_options if str(c).strip().lower() not in {"iowa", "statewide", "all", STATEWIDE_KEY}]

if not county_options:
    st.warning(
        "Could not infer counties from predictions/actuals. "
        "Your prediction outputs might be missing a 'county' column. "
        "If so, update your inference outputs to include county."
    )
    county_selected = None
else:
    options = [STATEWIDE_KEY] + county_options

    def _county_label(x: str) -> str:
        return STATEWIDE_LABEL if x == STATEWIDE_KEY else str(x).title()

    county_selected = st.selectbox("Select county", options=options, index=0, format_func=_county_label)

# ---------------- Tabs ----------------


tab_main, tab_debug, tab_export, tab_valueadd = st.tabs(
    ["County trend (2020–2025)", "Debug (S3 files)", "Download report", "Value-add views"]
)

with tab_main:
    if county_selected is None:
        st.stop()

    series_df = build_observed_vs_pred_series(pred_all, actuals_df, county_selected, years)
    st.dataframe(series_df, use_container_width=True)
    # Diagnostics: show which years are missing observed yields
    missing_obs_years = series_df.loc[series_df["observed_yield"].isna(), "year"].tolist()

    # Only warn for years <= 2024 (since 2025 observed is expected to be missing)
    missing_obs_years = [int(y) for y in missing_obs_years if pd.notna(y) and int(y) <= 2024]

    if missing_obs_years:
        st.warning(
            "Observed yield is missing for these years (likely a join key mismatch or gaps in actuals): "
            + ", ".join(map(str, missing_obs_years))
        )

        # Extra debug: show what actuals has for this county across the selected years
if actuals_df is not None and "year" in actuals_df.columns:
    obs_col = "observed_yield" if "observed_yield" in actuals_df.columns else "yield_bu_acre"

    if county_selected == STATEWIDE_KEY:
        a_dbg = (actuals_df[actuals_df["year"].isin(years)]
                 .groupby("year", as_index=False)[obs_col]
                 .mean()
                 .rename(columns={obs_col: "observed_yield"}))
        st.caption("Actuals (STATEWIDE mean across counties) from curated actuals.parquet (debug):")
        st.dataframe(a_dbg.sort_values("year"), use_container_width=True)
    else:
        a_dbg = actuals_df[
            (actuals_df["county_norm"] == county_selected) &
            (actuals_df["year"].isin(years))
        ][["county", "county_norm", "year", obs_col]].copy()

        a_dbg = a_dbg.rename(columns={obs_col: "observed_yield"})
        st.caption("Actuals rows found for this county in curated actuals.parquet (debug):")
        st.dataframe(a_dbg.sort_values("year"), use_container_width=True)

    title = f"Observed vs Mean Prediction — {(STATEWIDE_LABEL if county_selected==STATEWIDE_KEY else county_selected.title())} — {start_year}-{end_year}"
    if _HAS_PLOTLY:
        fig = plot_observed_vs_pred_plotly(series_df, title=title)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Install plotly for interactive charts: pip install plotly")

    # optional trigger
    if do_trigger and sf_arn.strip():
        st.markdown("### Trigger inference pipeline (Step Functions)")
        c1, c2 = st.columns([1, 1])
        with c1:
            exec_name = st.text_input(
                "Execution name (optional)",
                value=f"{county_selected.replace(' ','_')}-{feature_season}-{run_date}-{start_year}-{end_year}"
            )
        with c2:
            st.caption("Execution input will include state_fips, predict_year(s), season, model, run_date, county_fips=ALL.")

        if st.button("Start execution", type="secondary"):
            try:
                sf = _sf_client(region)
                payload = {
                    "run_date": str(run_date),
                    "state_fips": str(state_fips),
                    "county_fips": "ALL",
                    "feature_season": str(feature_season),
                    "model_name": str(model_name),
                    "predict_years": [int(y) for y in years],
                }
                args = {"stateMachineArn": sf_arn, "input": json_dumps(payload)}
                if exec_name.strip():
                    args["name"] = exec_name.strip()[:80]  # Step Functions name limit
                resp = sf.start_execution(**args)
                st.success("Started execution.")
                st.write("executionArn:", resp.get("executionArn"))
            except Exception as e:
                st.error(f"Failed to start execution: {e}")
    elif do_trigger and not sf_arn.strip():
        st.info("Provide a State machine ARN in the sidebar to enable triggering.")

with tab_debug:
    st.markdown("### Debug info (what was searched under predictions)")
    st.json(dbg_all)
    st.markdown("### Predictions preview")
    st.dataframe(pred_all.head(50), use_container_width=True)
    if actuals_df is not None:
        st.markdown("### Actuals preview")
        st.dataframe(actuals_df.head(50), use_container_width=True)

with tab_export:
    st.markdown("### Download report")

    if county_selected is None:
        st.stop()

    series_df = build_observed_vs_pred_series(pred_all, actuals_df, county_selected, years)
    metrics = compute_series_metrics(series_df)

    params = {
        "bucket": bucket,
        "state_fips": state_fips,
        "county_fips": county_fips,
        "county": (STATEWIDE_LABEL if county_selected==STATEWIDE_KEY else county_selected),
        "season": feature_season,
        "model": model_name,
        "run_date": run_date,
        "years": f"{start_year}-{end_year}",
    }

    # Build the chart figure for embedding in HTML/PDF when possible
    fig = None
    if _HAS_PLOTLY:
        try:
            fig = make_observed_vs_pred_plotly(series_df, county_selected, state_fips)
        except Exception:
            fig = None

    report_title = f"GeoAI Yield Report — {(STATEWIDE_LABEL if county_selected==STATEWIDE_KEY else county_selected.title())} — {start_year}-{end_year}"

    # ---- HTML export (always available) ----
    html_str = build_html_report_str(series_df, title=report_title, params=params, metrics=metrics, fig=fig)
    st.download_button(
        "Download HTML report",
        data=html_str.encode("utf-8"),
        file_name=f"geoai_yield_report_{(STATEWIDE_LABEL if county_selected==STATEWIDE_KEY else county_selected).replace(' ','_')}_{start_year}_{end_year}.html",
        mime="text/html",
        use_container_width=True,
    )

    st.divider()

    # ---- PDF export (optional) ----
    st.markdown("#### PDF export")

    if not _HAS_PDF:
        st.warning(
            "PDF export needs **reportlab**. Install it in the same environment where Streamlit runs: "
            "`pip install reportlab`"
        )
    elif not _HAS_MPL:
        st.warning(
            "PDF export needs **matplotlib**. Install it in the same environment where Streamlit runs: "
            "`pip install matplotlib`"
        )
    else:
        pdf_bytes = build_pdf_report_bytes(series_df, title=report_title, params=params, metrics=metrics)
        st.download_button(
            "Download PDF report",
            data=pdf_bytes,
            file_name=f"geoai_yield_report_{(STATEWIDE_LABEL if county_selected==STATEWIDE_KEY else county_selected).replace(' ','_')}_{start_year}_{end_year}.pdf",
            mime="application/pdf",
            use_container_width=True,
        )

with tab_valueadd:

    st.markdown("### Value-add views (great for final presentation)")
    st.caption("These views help you explain model stability, risk, and outliers quickly.")

    if county_selected is None:
        st.stop()

    # 1) Year-over-year delta (predicted)
    st.markdown("#### 1) Year-over-year change (Mean Prediction)")
    series_df = build_observed_vs_pred_series(pred_all, actuals_df, county_selected, years).copy()
    series_df["pred_yoy_delta"] = series_df["mean_prediction"].diff()
    st.dataframe(series_df[["year", "mean_prediction", "pred_yoy_delta"]], use_container_width=True)

    # 2) County percentile rank in each year (if we have county ids)
    st.markdown("#### 2) County percentile rank (Predicted) within Iowa by year")
    if "county_norm" in pred_all.columns:
        ranks = []
        for y in years:
            g = pred_all[pred_all["year"] == y].copy()
            if g.empty:
                continue
            by = g.groupby("county_norm")["prediction"].mean().reset_index()
            by["percentile"] = by["prediction"].rank(pct=True)
            row = by[by["county_norm"] == county_selected]
            if not row.empty:
                ranks.append({"year": int(y), "predicted_percentile": float(row["percentile"].iloc[0])})
        if ranks:
            rk = pd.DataFrame(ranks).sort_values("year")
            st.dataframe(rk, use_container_width=True)
            if _HAS_PLOTLY:
                fig = px.line(rk, x="year", y="predicted_percentile", markers=True, title="Predicted percentile over time")
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Not enough data to compute percentile ranks.")
    else:
        st.info("County ids are missing in predictions, so percentile ranks can't be computed.")

    # 3) Error over time (if actuals exist)
    st.markdown("#### 3) Error over time (Observed - Mean Prediction)")
    if series_df["observed_yield"].notna().any():
        series_df["error"] = series_df["observed_yield"] - series_df["mean_prediction"]
        st.dataframe(series_df[["year", "observed_yield", "mean_prediction", "error"]], use_container_width=True)
        if _HAS_PLOTLY:
            fig = px.bar(series_df.dropna(subset=["error"]), x="year", y="error", title="Error over time (Observed - Predicted)")
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Actuals not available for these years; error view is disabled.")


