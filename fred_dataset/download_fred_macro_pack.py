#!/usr/bin/env python3
"""
download_fred_macro_pack.py

Downloads the 10-series "Macro Pack" from FRED plus USREC (NBER recession indicator)
as CSV files and (optionally) downloads each series' metadata (data dictionary)
via the official FRED API.

It also builds a model-ready monthly panel (joined on month-end dates) suitable for:
- regression POC on UNRATE (unemployment rate), or
- recession probability modeling (classification backtests) using recession_label.

What it downloads
- Data CSVs (no API key required): https://fred.stlouisfed.org/graph/fredgraph.csv?id=<SERIES_ID>
- Metadata JSON (API key required): https://api.stlouisfed.org/fred/series?series_id=<SERIES_ID>&api_key=...&file_type=json

Outputs
  <out>/
    data/        # one CSV per series (including USREC.csv)
    meta/        # one JSON per series (if api key provided)
    panels/
      macro_panel_monthly.parquet
      macro_panel_monthly.csv
    manifest.json  # run manifest for provenance

Notes on recession_label
- USREC is typically 0/1 monthly recession indicator.
- We align it to month-end and create:
    recession_label = 1 if USREC >= 0.5 else 0
- This gives you a clean binary label for classification backtests.

What you’ll see in the final panel
Feature columns: CPIAUCSL, UNRATE, FEDFUNDS, DGS10, T10Y2Y, INDPRO, PCE, M2SL, CSUSHPISA, UMCSENT
Label column:
USREC (original)
recession_label (0/1 for backtesting)
If you want, I can add an option like --label-horizon 6 to shift the label forward (e.g., predict recession 6 months ahead), which is common for recession prediction backtests.

python -m venv .venv
source .venv/bin/activate
pip install pandas requests pyarrow
export FRED_API_KEY="YOUR_FRED_KEY"   # optional, for metadata JSON
python download_fred_macro_pack.py --out ./fred_macro_pack --build-panel
"""

from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Tuple, List

import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


# -----------------------------
# Configuration
# -----------------------------

@dataclass(frozen=True)
class SeriesSpec:
    series_id: str
    label: str
    expected_frequency_hint: str  # informational only


MACRO_PACK: List[SeriesSpec] = [
    SeriesSpec("CPIAUCSL", "CPI (Inflation)", "Monthly"),
    SeriesSpec("UNRATE", "Unemployment Rate (Target)", "Monthly"),
    SeriesSpec("FEDFUNDS", "Federal Funds Rate", "Monthly"),
    SeriesSpec("DGS10", "10Y Treasury Constant Maturity Rate", "Daily"),
    SeriesSpec("T10Y2Y", "10Y Minus 2Y Yield Spread", "Daily"),
    SeriesSpec("INDPRO", "Industrial Production Index", "Monthly"),
    SeriesSpec("PCE", "Personal Consumption Expenditures", "Monthly"),
    SeriesSpec("M2SL", "M2 Money Stock", "Weekly"),
    SeriesSpec("CSUSHPISA", "Case-Shiller National Home Price Index", "Monthly"),
    SeriesSpec("UMCSENT", "U. of Michigan Consumer Sentiment", "Monthly"),
]

LABEL_SERIES: SeriesSpec = SeriesSpec("USREC", "NBER based Recession Indicators for the United States", "Monthly")

ALL_SERIES: List[SeriesSpec] = MACRO_PACK + [LABEL_SERIES]

FRED_CSV_BASE = "https://fred.stlouisfed.org/graph/fredgraph.csv"
FRED_API_BASE = "https://api.stlouisfed.org/fred/series"


# -----------------------------
# HTTP helpers
# -----------------------------

def make_session() -> requests.Session:
    session = requests.Session()
    retry = Retry(
        total=5,
        backoff_factor=0.8,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET"],
        raise_on_status=False,
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    return session


def http_get_bytes(session: requests.Session, url: str, params: Dict[str, str], timeout: int = 30) -> bytes:
    r = session.get(url, params=params, timeout=timeout)
    if r.status_code >= 400:
        raise RuntimeError(f"GET failed {r.status_code} for {r.url}: {r.text[:200]}")
    return r.content


def http_get_json(session: requests.Session, url: str, params: Dict[str, str], timeout: int = 30) -> dict:
    r = session.get(url, params=params, timeout=timeout)
    if r.status_code >= 400:
        raise RuntimeError(f"GET failed {r.status_code} for {r.url}: {r.text[:200]}")
    return r.json()


# -----------------------------
# Downloaders
# -----------------------------

def download_series_csv(session: requests.Session, series_id: str, out_path: Path) -> None:
    params = {"id": series_id}
    content = http_get_bytes(session, FRED_CSV_BASE, params=params)
    out_path.write_bytes(content)


def download_series_metadata(session: requests.Session, series_id: str, api_key: str, out_path: Path) -> None:
    params = {"series_id": series_id, "api_key": api_key, "file_type": "json"}
    data = http_get_json(session, FRED_API_BASE, params=params)
    out_path.write_text(json.dumps(data, indent=2), encoding="utf-8")


# -----------------------------
# Parsing & panel building
# -----------------------------

def load_fred_csv(path: Path) -> pd.DataFrame:
    """
    Loads FRED CSV into a normalized DataFrame with columns:
      - date: datetime64[ns]
      - value: float (NaN for '.')
    Supports both common formats:
      1) date,value
      2) observation_date,<series_id>
    """
    df = pd.read_csv(path)

    # Normalize header names (lowercase, strip)
    df.columns = [c.strip().lower() for c in df.columns]

    # Case 1: expected format date,value
    if "date" in df.columns and "value" in df.columns:
        date_col = "date"
        value_col = "value"

    # Case 2: common FRED format observation_date,<series_id>
    elif "observation_date" in df.columns and len(df.columns) == 2:
        date_col = "observation_date"
        # second column is the series column name (e.g., cpiaucsl)
        value_col = [c for c in df.columns if c != "observation_date"][0]

    # Case 3: fallback for any 2-col file
    elif len(df.columns) == 2:
        date_col, value_col = df.columns[0], df.columns[1]

    else:
        raise ValueError(f"Unexpected CSV format in {path}: columns={df.columns.tolist()}")

    out = pd.DataFrame()
    out["date"] = pd.to_datetime(df[date_col], errors="coerce")

    # FRED uses '.' for missing values in many exports
    out["value"] = pd.to_numeric(df[value_col].replace(".", pd.NA), errors="coerce")

    out = out.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)
    return out


def to_monthly_series(df: pd.DataFrame, series_id: str, method: str = "mean") -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["date_month_end", series_id])

    s = df.set_index("date")["value"].copy()

    if method == "mean":
        monthly = s.resample("ME").mean()
    elif method == "last":
        monthly = s.resample("ME").last()
    else:
        raise ValueError("method must be 'mean' or 'last'")

    out = monthly.reset_index()
    out.rename(columns={"date": "date_month_end", "value": series_id}, inplace=True)
    out["date_month_end"] = pd.to_datetime(out["date_month_end"], utc=False)
    return out


def build_monthly_panel(data_dir: Path, panel_out_dir: Path) -> Tuple[Path, Path]:
    panel_out_dir.mkdir(parents=True, exist_ok=True)

    monthly_frames: List[pd.DataFrame] = []

    # Build monthly series for macro pack
    for spec in MACRO_PACK:
        csv_path = data_dir / f"{spec.series_id}.csv"
        if not csv_path.exists():
            raise FileNotFoundError(f"Missing CSV: {csv_path}")

        raw = load_fred_csv(csv_path)

        # Use mean resampling for daily/weekly to monthly; OK for POC
        monthly = to_monthly_series(raw, series_id=spec.series_id, method="mean")
        monthly_frames.append(monthly)

    # Merge all macro features
    panel = monthly_frames[0]
    for df in monthly_frames[1:]:
        panel = panel.merge(df, on="date_month_end", how="outer")

    # Add USREC label (monthly) and create recession_label
    usrec_path = data_dir / f"{LABEL_SERIES.series_id}.csv"
    if not usrec_path.exists():
        raise FileNotFoundError(f"Missing CSV: {usrec_path}")

    usrec_raw = load_fred_csv(usrec_path)

    # For USREC, use last-of-month (it’s already monthly, but last makes sense)
    usrec_monthly = to_monthly_series(usrec_raw, series_id=LABEL_SERIES.series_id, method="last")
    panel = panel.merge(usrec_monthly, on="date_month_end", how="left")

    # Create clean binary label for classification backtests
    if LABEL_SERIES.series_id in panel.columns:
        panel["recession_label"] = (panel[LABEL_SERIES.series_id].fillna(0.0) >= 0.5).astype("int8")

    panel = panel.sort_values("date_month_end").reset_index(drop=True)

    # Derived features (optional)
    for col in ["CPIAUCSL", "PCE", "INDPRO", "CSUSHPISA", "UMCSENT"]:
        if col in panel.columns:
            panel[f"{col}_yoy_pct"] = (panel[col] / panel[col].shift(12) - 1.0) * 100.0

    parquet_path = panel_out_dir / "macro_panel_monthly.parquet"
    csv_path = panel_out_dir / "macro_panel_monthly.csv"
    panel.to_parquet(parquet_path, index=False)
    panel.to_csv(csv_path, index=False)

    return parquet_path, csv_path


# -----------------------------
# CLI + manifest
# -----------------------------

def write_manifest(out_dir: Path, api_key_used: bool) -> None:
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "series_pack": [spec.__dict__ for spec in ALL_SERIES],
        "api_key_used_for_metadata": api_key_used,
        "data_endpoint": FRED_CSV_BASE,
        "metadata_endpoint": FRED_API_BASE,
        "label_series": LABEL_SERIES.series_id,
        "label_rule": "recession_label = 1 if USREC >= 0.5 else 0",
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Download FRED macro pack + USREC and optional metadata; build a monthly panel with recession_label."
    )
    parser.add_argument("--out", default="./fred_macro_pack", help="Output directory (default: ./fred_macro_pack)")
    parser.add_argument("--skip-metadata", action="store_true",
                        help="Skip metadata download even if FRED_API_KEY is set.")
    parser.add_argument("--build-panel", action="store_true", help="Build merged monthly panel (parquet + csv).")
    parser.add_argument("--sleep", type=float, default=0.2, help="Sleep seconds between downloads (default: 0.2)")
    args = parser.parse_args()

    out_dir = Path(args.out).resolve()
    data_dir = out_dir / "data"
    meta_dir = out_dir / "meta"
    panel_dir = out_dir / "panels"

    data_dir.mkdir(parents=True, exist_ok=True)
    meta_dir.mkdir(parents=True, exist_ok=True)

    fred_api_key = os.environ.get("FRED_API_KEY", "").strip()
    do_meta = bool(fred_api_key) and (not args.skip_metadata)

    session = make_session()

    print(f"[INFO] Output directory: {out_dir}")
    print(f"[INFO] Downloading {len(ALL_SERIES)} series CSVs from FRED (macro pack + USREC)...")

    for i, spec in enumerate(ALL_SERIES, start=1):
        csv_path = data_dir / f"{spec.series_id}.csv"
        print(f"  [{i:02d}/{len(ALL_SERIES)}] CSV {spec.series_id} -> {csv_path.name}")
        download_series_csv(session, spec.series_id, csv_path)

        if do_meta:
            meta_path = meta_dir / f"{spec.series_id}.meta.json"
            print(f"         META {spec.series_id} -> {meta_path.name}")
            download_series_metadata(session, spec.series_id, fred_api_key, meta_path)

        time.sleep(max(args.sleep, 0.0))

    write_manifest(out_dir, api_key_used=do_meta)

    if args.build_panel:
        print("[INFO] Building model-ready monthly panel with recession_label...")
        parquet_path, csv_path = build_monthly_panel(data_dir, panel_dir)
        print(f"[OK] Wrote: {parquet_path}")
        print(f"[OK] Wrote: {csv_path}")

    print("[DONE] Download complete.")
    if not do_meta:
        print(
            "[NOTE] Metadata was not downloaded. To include metadata, set FRED_API_KEY and rerun without --skip-metadata.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
