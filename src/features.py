"""
Module M2b - Preprocessing and Feature Engineering.

Builds the supervised learning table for multi-horizon demand forecasting:
calendar features, Fourier seasonality terms, lag features, rolling
statistics, external-signal flags (holiday / promotion) and a cold-start
safe SKU descriptor.

The lag/rolling columns are constructed strictly from past values so the
resulting table can be used with a time-based (no-shuffle) split.

Usage:
    python src/features.py            # writes the feature table + report
"""

from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config  # noqa: E402

warnings.filterwarnings("ignore")

LAGS = [1, 2, 3, 7, 14, 21, 28, 56, 364]
ROLL_WINDOWS = [7, 14, 28, 56]
FOURIER_K = 3          # harmonics for weekly + yearly seasonality
SEQ_LAGS = list(range(1, 29))   # consecutive daily lags 1..28 -> deep-learning input sequence


# --------------------------------------------------------------------------
# external signals
# --------------------------------------------------------------------------
def us_holiday_flags(dates: pd.DatetimeIndex) -> pd.DataFrame:
    """Holiday calendar + promotion proxy for the 5-year US retail window."""
    years = sorted(dates.year.unique())
    holidays = []
    for y in years:
        holidays += [
            pd.Timestamp(f"{y}-01-01"),   # New Year
            pd.Timestamp(f"{y}-07-04"),   # Independence Day
            pd.Timestamp(f"{y}-12-25"),   # Christmas
            pd.Timestamp(f"{y}-11-24") + pd.Timedelta(days=0),  # placeholder, fixed below
        ]
        # Thanksgiving = 4th Thursday of November
        nov = pd.date_range(f"{y}-11-01", f"{y}-11-30")
        thu = [d for d in nov if d.dayofweek == 3]
        if len(thu) >= 4:
            holidays.append(thu[3])
        holidays += list(nov[-1:])        # Black-Friday week demand peak
    holidays = pd.DatetimeIndex(sorted(set(holidays)))

    out = pd.DataFrame(index=dates)
    out["is_holiday"] = dates.isin(holidays).astype(int)
    # promotion proxy: retail promo intensity - festive window + weekend uplift
    out["is_promo"] = ((dates.month == 12) & (dates.day >= 10)).astype(int)
    out["is_weekend"] = (dates.dayofweek >= 5).astype(int)
    out["is_month_start"] = dates.is_month_start.astype(int)
    out["is_month_end"] = dates.is_month_end.astype(int)
    return out


def fourier_terms(dates: pd.DatetimeIndex, k: int = FOURIER_K) -> pd.DataFrame:
    """Fourier seasonality encoding for the weekly and yearly cycles."""
    out = pd.DataFrame(index=dates)
    doy = dates.dayofyear.values
    dow = dates.dayofweek.values
    year_len = np.where(np.asarray(dates.is_leap_year), 366, 365)
    for i in range(1, k + 1):
        out[f"year_sin_{i}"] = np.sin(2 * np.pi * i * doy / year_len)
        out[f"year_cos_{i}"] = np.cos(2 * np.pi * i * doy / year_len)
        out[f"week_sin_{i}"] = np.sin(2 * np.pi * i * dow / 7)
        out[f"week_cos_{i}"] = np.cos(2 * np.pi * i * dow / 7)
    return out


# --------------------------------------------------------------------------
# feature table
# --------------------------------------------------------------------------
def build_feature_table(sd: pd.DataFrame, include_lags: bool = True) -> pd.DataFrame:
    """
    Parameters
    ----------
    sd : cleaned store-item demand frame (date, store, item, sku, sales)
    include_lags : False builds only the calendar/external block - used for the
        ablation study that quantifies the value added by lag features.
    """
    sd = sd.sort_values(["store", "item", "date"]).copy()
    idx = pd.DatetimeIndex(sd["date"].unique())
    idx.name = "date"

    cal = pd.concat([us_holiday_flags(idx), fourier_terms(idx)], axis=1)

    sd["dow"] = sd["date"].dt.dayofweek
    sd["month"] = sd["date"].dt.month
    sd["day"] = sd["date"].dt.day
    sd["dayofyear"] = sd["date"].dt.dayofyear
    sd["weekofyear"] = sd["date"].dt.isocalendar().week.astype(int)
    sd["quarter"] = sd["date"].dt.quarter
    sd["is_leap"] = sd["date"].dt.is_leap_year.astype(int)

    grp = sd.groupby(["store", "item"], sort=False)["sales"]
    if include_lags:
        # consecutive daily lags - the sequence block consumed by the LSTM
        for i in SEQ_LAGS:
            sd[f"seq_{i}"] = grp.shift(i)
        for lag in LAGS:
            sd[f"lag_{lag}"] = grp.shift(lag)
        for w in ROLL_WINDOWS:
            sd[f"roll_mean_{w}"] = grp.transform(
                lambda s, w=w: s.shift(1).rolling(w, min_periods=max(2, w // 2)).mean())
            sd[f"roll_std_{w}"] = grp.transform(
                lambda s, w=w: s.shift(1).rolling(w, min_periods=max(2, w // 2)).std())
            sd[f"roll_max_{w}"] = grp.transform(
                lambda s, w=w: s.shift(1).rolling(w, min_periods=max(2, w // 2)).max())
            sd[f"roll_min_{w}"] = grp.transform(
                lambda s, w=w: s.shift(1).rolling(w, min_periods=max(2, w // 2)).min())
        # trend / momentum descriptors
        sd["ewm_28"] = grp.transform(lambda s: s.shift(1).ewm(span=28, min_periods=7).mean())
        sd["lag_ratio_7_28"] = sd["lag_7"] / sd["roll_mean_28"].replace(0, np.nan)
        sd["rolling_trend"] = (sd["roll_mean_7"] - sd["roll_mean_28"]) / sd["roll_mean_28"].replace(0, np.nan)
        sd["cv_28"] = sd["roll_std_28"] / sd["roll_mean_28"].replace(0, np.nan)

    # SKU descriptors (available even for cold-start items -> no history needed)
    item_stats = sd.groupby("item")["sales"].transform("mean")
    sd["item_mean_all"] = item_stats
    sd["store_factor"] = sd.groupby("store")["sales"].transform("mean") / sd["sales"].mean()
    sd["item_store_rank"] = sd["item"].astype(str) + "_" + sd["store"].astype(str)

    out = sd.merge(cal, left_on="date", right_index=True, how="left")
    out = out.replace([np.inf, -np.inf], np.nan)
    return out


FEATURE_COLS_CAL = (
    ["store", "item", "dow", "month", "day", "dayofyear", "weekofyear", "quarter", "is_leap",
     "is_holiday", "is_promo", "is_weekend", "is_month_start", "is_month_end"]
    + [f"year_sin_{i}" for i in range(1, FOURIER_K + 1)]
    + [f"year_cos_{i}" for i in range(1, FOURIER_K + 1)]
    + [f"week_sin_{i}" for i in range(1, FOURIER_K + 1)]
    + [f"week_cos_{i}" for i in range(1, FOURIER_K + 1)]
    + ["item_mean_all", "store_factor"]
)
FEATURE_COLS_LAG = (
    [f"lag_{l}" for l in LAGS]
    + [f"{f}_{w}" for w in ROLL_WINDOWS for f in ("roll_mean", "roll_std", "roll_max", "roll_min")]
    + ["ewm_28", "lag_ratio_7_28", "rolling_trend", "cv_28"]
)
FEATURE_COLS = FEATURE_COLS_CAL + FEATURE_COLS_LAG
# sequence block (chronological order, oldest first) used only by the DL models
SEQ_COLS = [f"seq_{i}" for i in reversed(SEQ_LAGS)]


def run() -> pd.DataFrame:
    print("=" * 78)
    print("MODULE M2b - FEATURE ENGINEERING")
    print("=" * 78)
    sd = pd.read_csv(config.PROCESSED_DIR / "store_demand_clean.csv.gz")
    sd["date"] = pd.to_datetime(sd["date"])

    ft = build_feature_table(sd, include_lags=True)
    ft.to_csv(config.PROCESSED_DIR / "store_demand_features.csv.gz", index=False, compression="gzip")

    corr_target = (ft[FEATURE_COLS + ["sales"]].corr()["sales"]
                   .drop("sales").sort_values(key=abs, ascending=False))
    corr_target.round(4).to_csv(config.TABLES_DIR / "t10_feature_target_correlation.csv")

    report = {
        "rows": int(len(ft)),
        "features": len(FEATURE_COLS),
        "calendar_features": len(FEATURE_COLS_CAL),
        "lag_rolling_features": len(FEATURE_COLS_LAG),
        "nan_rows_after_lags": int(ft[FEATURE_COLS].isna().any(axis=1).sum()),
        "top_15_features_vs_target": {k: round(float(v), 3) for k, v in corr_target.head(15).items()},
    }
    Path(config.METRICS_DIR / "feature_engineering.json").write_text(json.dumps(report, indent=2))

    print(f"    feature table : {len(ft):,} rows x {len(FEATURE_COLS)} features")
    print(f"      calendar/external : {len(FEATURE_COLS_CAL)}")
    print(f"      lag/rolling/trend : {len(FEATURE_COLS_LAG)}")
    print("    top-10 features by |corr| with target:")
    for k, v in list(report["top_15_features_vs_target"].items())[:10]:
        print(f"      {k:<20} {v:+.3f}")
    print("    saved -> processed/store_demand_features.csv.gz")
    return ft


if __name__ == "__main__":
    run()
