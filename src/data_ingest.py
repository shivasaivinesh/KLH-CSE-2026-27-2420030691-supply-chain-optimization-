"""
Module M1 - Data Ingestion, Validation and Cleaning.

Loads the three benchmark datasets, drops personally identifiable information,
repairs dtypes, removes duplicates / impossible values and writes tidied
CSV files to the processed directory for the downstream stages.

Usage:
    python src/data_ingest.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config  # noqa: E402


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------
def _log(msg: str) -> None:
    print(msg, flush=True)


def _quality_report(df: pd.DataFrame, name: str) -> dict:
    """Basic data-quality fingerprint used by the report and the dashboard."""
    rep = {
        "dataset": name,
        "rows": int(len(df)),
        "columns": int(df.shape[1]),
        "duplicate_rows": int(df.duplicated().sum()),
        "missing_cells": int(df.isna().sum().sum()),
        "missing_pct": round(float(df.isna().sum().sum()) / (df.shape[0] * df.shape[1]) * 100, 3),
        "memory_mb": round(float(df.memory_usage(deep=True).sum()) / 1e6, 2),
    }
    return rep


# --------------------------------------------------------------------------
# D1 - DataCo Smart Supply Chain (orders / logistics / delivery risk)
# --------------------------------------------------------------------------
PII_COLUMNS = [
    "Customer Email", "Customer Password", "Customer Fname", "Customer Lname",
    "Customer Street", "Customer Zipcode", "Order Zipcode", "Product Image",
    "Product Description", "Customer Id", "Order Customer Id", "Order Item Id",
]


def load_dataco() -> tuple[pd.DataFrame, dict]:
    _log("\n[D1] DataCo Smart Supply Chain ...")
    raw = pd.read_csv(config.DATACO_CSV, encoding="latin-1", low_memory=False)
    before = _quality_report(raw, "DataCo (raw)")

    df = raw.drop(columns=[c for c in PII_COLUMNS if c in raw.columns])

    # timestamps
    df["order_date"] = pd.to_datetime(df["order date (DateOrders)"], errors="coerce")
    df["ship_date"] = pd.to_datetime(df["shipping date (DateOrders)"], errors="coerce")
    df = df.dropna(subset=["order_date"])
    df = df.drop(columns=["order date (DateOrders)", "shipping date (DateOrders)"])

    # calendar features (drive the seasonal demand signal)
    df["order_month"] = df["order_date"].dt.month
    df["order_dow"] = df["order_date"].dt.dayofweek
    df["order_year"] = df["order_date"].dt.year
    df["order_week"] = df["order_date"].dt.isocalendar().week.astype(int)
    df["order_hour"] = df["order_date"].dt.hour

    # business target + derived operational features
    df["Late_delivery_risk"] = df["Late_delivery_risk"].astype(int)
    df["shipping_delay"] = (df["Days for shipping (real)"] - df["Days for shipment (scheduled)"]).astype(int)
    df["discount_amount"] = (df["Order Item Discount"]).astype(float)
    df["unit_price"] = df["Order Item Product Price"].astype(float).round(2)

    # drop records with no usable signal
    df = df[(df["Order Item Quantity"] > 0) & (df["Sales"] >= 0)]
    df = df.drop_duplicates()
    after = _quality_report(df, "DataCo (clean)")
    _log(f"     raw={before['rows']:,} rows -> clean={after['rows']:,} rows "
         f"({before['rows'] - after['rows']:,} dropped)")

    out = config.PROCESSED_DIR / "dataco_clean.csv"
    df.to_csv(out, index=False)
    _log(f"     saved -> {out}")
    return df, {"before": before, "after": after}


# --------------------------------------------------------------------------
# D2 - Store Item Demand (daily SKU-store demand time series)
# --------------------------------------------------------------------------
def load_store_demand() -> tuple[pd.DataFrame, dict]:
    _log("\n[D2] Store Item Demand Forecasting ...")
    raw = pd.read_csv(config.STORE_DEMAND_CSV)
    before = _quality_report(raw, "StoreDemand (raw)")

    df = raw.copy()
    df["date"] = pd.to_datetime(df["date"])
    df["sales"] = pd.to_numeric(df["sales"], errors="coerce")
    df = df.dropna(subset=["date", "sales"])
    df = df[df["sales"] >= 0]
    df = df.drop_duplicates(subset=["date", "store", "item"])

    df["store"] = df["store"].astype(int)
    df["item"] = df["item"].astype(int)
    df["sku"] = "S" + df["store"].astype(str).str.zfill(2) + "-I" + df["item"].astype(str).str.zfill(2)
    df["dow"] = df["date"].dt.dayofweek
    df["month"] = df["date"].dt.month
    df["year"] = df["date"].dt.year
    df["dayofyear"] = df["date"].dt.dayofyear

    df = df.sort_values(["store", "item", "date"]).reset_index(drop=True)
    after = _quality_report(df, "StoreDemand (clean)")
    _log(f"     {after['rows']:,} rows | {df['store'].nunique()} stores x "
         f"{df['item'].nunique()} items = {df['sku'].nunique()} series")
    _log(f"     period {df['date'].min().date()} -> {df['date'].max().date()}")

    out = config.PROCESSED_DIR / "store_demand_clean.csv.gz"
    df.to_csv(out, index=False, compression="gzip")
    _log(f"     saved -> {out}")
    return df, {"before": before, "after": after}


# --------------------------------------------------------------------------
# D3 - UCI Online Retail II (high-cardinality e-commerce transactions)
# --------------------------------------------------------------------------
def load_online_retail() -> tuple[pd.DataFrame, dict]:
    _log("\n[D3] UCI Online Retail II ...")
    raw = pd.read_csv(config.ONLINE_RETAIL_CSV)
    before = _quality_report(raw, "OnlineRetail (raw)")

    df = raw.copy()
    df["Invoice"] = df["Invoice"].astype(str)
    df["StockCode"] = df["StockCode"].astype(str)
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"], errors="coerce")
    df["Quantity"] = pd.to_numeric(df["Quantity"], errors="coerce")
    df["Price"] = pd.to_numeric(df["Price"], errors="coerce")

    df = df.dropna(subset=["InvoiceDate", "Quantity", "Price", "StockCode"])
    df = df[df["Price"] > 0]
    df["is_cancellation"] = df["Invoice"].str.startswith("C")
    df["revenue"] = df["Quantity"] * df["Price"]

    # demand view: completed sales only (cancellations kept for return-rate KPI)
    sales = df[(~df["is_cancellation"]) & (df["Quantity"] > 0)].copy()
    sales = sales[~sales["StockCode"].str.upper().str.contains("POST|DOT|M|BANK", regex=True, na=False)]
    sales["date"] = sales["InvoiceDate"].dt.normalize()
    sales["dow"] = sales["date"].dt.dayofweek
    sales["month"] = sales["date"].dt.month
    sales = sales.drop_duplicates()

    after = _quality_report(sales, "OnlineRetail (clean sales)")
    _log(f"     raw={before['rows']:,} rows -> clean sales={after['rows']:,} rows")
    _log(f"     {sales['StockCode'].nunique():,} SKUs | {sales['Country'].nunique()} countries | "
         f"period {sales['date'].min().date()} -> {sales['date'].max().date()}")

    out = config.PROCESSED_DIR / "online_retail_clean.csv.gz"
    sales.to_csv(out, index=False, compression="gzip")
    _log(f"     saved -> {out}")
    return sales, {"before": before, "after": after}


# --------------------------------------------------------------------------
def run() -> dict:
    for path, hint in [
        (config.DATACO_CSV, "DataCo"),
        (config.STORE_DEMAND_CSV, "Store Item Demand"),
        (config.ONLINE_RETAIL_CSV, "Online Retail II"),
    ]:
        if not Path(path).exists():
            raise FileNotFoundError(
                f"{hint} dataset missing at {path}.\n"
                f"Run `python src/fetch_data.py` or set SC_DATA_ROOT."
            )
    _, d1 = load_dataco()
    _, d2 = load_store_demand()
    _, d3 = load_online_retail()
    _log("\n[D1..D3] ingestion complete.")
    return {"dataco": d1, "store_demand": d2, "online_retail": d3}


if __name__ == "__main__":
    run()
