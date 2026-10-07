"""
Module M2 - Exploratory Data Analysis.

Implements the 12 analytical techniques agreed in the project roadmap plus the
two supply-chain specific analyses (ABC classification, STL seasonality
decomposition) and writes every figure used in the Review-III report.

Techniques
----------
 1. Dataset overview            8. Univariate analysis
 2. Data-quality assessment     9. Bivariate analysis
 3. Descriptive statistics     10. Multivariate analysis
 4. Missing-value analysis     11. Correlation analysis
 5. Duplicate detection        12. Feature engineering preview
 6. Outlier detection     + ABC analysis + STL decomposition + RFM segments
 7. Distribution / class balance

Usage:
    python src/eda.py
"""

from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from statsmodels.tsa.seasonal import STL

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config  # noqa: E402

warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid", palette="deep")
plt.rcParams.update({"figure.dpi": 130, "savefig.bbox": "tight", "font.size": 9})

FIG = config.FIGURES_DIR
TAB = config.TABLES_DIR


def save(fig, name: str) -> None:
    path = FIG / f"{name}.png"
    fig.savefig(path)
    plt.close(fig)
    print(f"    figure -> {path.name}")


# ==========================================================================
# 1-3  overview / quality / descriptive statistics
# ==========================================================================
def overview_and_quality(dfs: dict[str, pd.DataFrame]) -> dict:
    print("\n[EDA 1-3] overview, data quality, descriptive statistics")
    rows, desc_rows = [], []
    for name, df in dfs.items():
        num = df.select_dtypes(include=[np.number])
        rows.append({
            "dataset": name,
            "rows": len(df),
            "columns": df.shape[1],
            "numeric_cols": num.shape[1],
            "missing_cells": int(df.isna().sum().sum()),
            "missing_pct": round(float(df.isna().sum().sum()) / max(df.shape[0] * df.shape[1], 1) * 100, 3),
            "duplicate_rows": int(df.duplicated().sum()),
            "memory_mb": round(float(df.memory_usage(deep=True).sum()) / 1e6, 2),
        })
        for col in num.columns[:40]:
            s = num[col].dropna()
            if s.empty:
                continue
            desc_rows.append({
                "dataset": name, "column": col, "count": int(s.size),
                "mean": round(float(s.mean()), 3), "std": round(float(s.std()), 3),
                "min": round(float(s.min()), 3), "q25": round(float(s.quantile(.25)), 3),
                "median": round(float(s.median()), 3), "q75": round(float(s.quantile(.75)), 3),
                "max": round(float(s.max()), 3),
                "skew": round(float(s.skew()), 3),
            })
    overview = pd.DataFrame(rows)
    overview.to_csv(TAB / "t01_dataset_overview.csv", index=False)
    pd.DataFrame(desc_rows).to_csv(TAB / "t02_descriptive_statistics.csv", index=False)

    # figure: missing-value heat map of the DataCo source table
    dc = dfs["DataCo"]
    miss = dc.isna().sum()
    miss = miss[miss > 0].sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(7, max(2.4, 0.24 * len(miss))))
    if len(miss):
        miss.plot(kind="barh", ax=ax, color="#c0392b")
        ax.set_xlabel("missing values")
        ax.set_title("Missing-value analysis - DataCo (post-clean)")
        ax.invert_yaxis()
    else:
        ax.text(.5, .5, "No missing values in any retained column",
                ha="center", va="center", fontsize=11)
        ax.axis("off")
    save(fig, "f03_missing_values_dataco")
    return {"overview": overview.to_dict("records")}


# ==========================================================================
# 4-7  duplicates, outliers, distribution, class balance
# ==========================================================================
def outliers_and_target(dfs: dict[str, pd.DataFrame]) -> dict:
    print("\n[EDA 4-7] outliers, distributions, class balance")
    dc = dfs["DataCo"]
    res = {}

    # IQR outlier profile of the main continuous business variables
    cols = ["Sales", "Order Item Quantity", "Order Item Profit Ratio",
            "Days for shipping (real)", "Order Item Discount Rate", "Product Price"]
    out = []
    for c in cols:
        s = pd.to_numeric(dc[c], errors="coerce").dropna()
        q1, q3 = s.quantile(.25), s.quantile(.75)
        iqr = q3 - q1
        lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        out.append({
            "column": c, "q1": round(float(q1), 3), "q3": round(float(q3), 3),
            "iqr": round(float(iqr), 3), "lower_fence": round(float(lo), 3),
            "upper_fence": round(float(hi), 3),
            "outliers": int(((s < lo) | (s > hi)).sum()),
            "outlier_pct": round(float(((s < lo) | (s > hi)).mean() * 100), 2),
        })
    od = pd.DataFrame(out)
    od.to_csv(TAB / "t03_outlier_profile_iqr.csv", index=False)
    res["outliers"] = od.to_dict("records")

    fig, axes = plt.subplots(1, 3, figsize=(11, 3.2))
    for ax, c in zip(axes, ["Sales", "Order Item Quantity", "Product Price"]):
        s = pd.to_numeric(dc[c], errors="coerce").dropna()
        s = s[s <= s.quantile(.99)]
        ax.boxplot(s, vert=True, widths=.5, patch_artist=True,
                   boxprops=dict(facecolor="#aed6f1", color="#2471a3"))
        ax.set_title(f"{c}\n(IQR fences in table)")
        ax.set_ylabel(c)
    fig.suptitle("Outlier detection - IQR boxplots (99th-percentile clipped for display)",
                 y=1.03, fontsize=10)
    save(fig, "f04_outlier_boxplots")

    # class balance of the delay-risk target
    bal = dc["Late_delivery_risk"].value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(4.2, 3.2))
    bars = ax.bar(["On time (0)", "Late (1)"], bal.values, color=["#27ae60", "#c0392b"], width=.55)
    for b, v in zip(bars, bal.values):
        ax.text(b.get_x() + b.get_width() / 2, v, f"{v:,}\n({v / bal.sum() * 100:.1f}%)",
                ha="center", va="bottom", fontsize=8.5)
    ax.set_ylim(0, bal.max() * 1.22)
    ax.set_ylabel("orders")
    ax.set_title("Class balance - Late_delivery_risk")
    save(fig, "f05_class_balance_delay_risk")
    res["class_balance"] = {str(k): int(v) for k, v in bal.items()}

    # univariate: demand distribution + daily demand trend (D2)
    sd = dfs["StoreDemand"]
    daily = sd.groupby("date")["sales"].sum()
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.1))
    axes[0].hist(sd["sales"], bins=45, color="#5dade2", edgecolor="white")
    axes[0].set_title("Univariate - daily sales per store-item")
    axes[0].set_xlabel("units/day"); axes[0].set_ylabel("frequency")
    axes[1].plot(daily.index, daily.values, lw=.7, color="#1a5276")
    axes[1].plot(daily.rolling(30).mean(), lw=1.5, color="#e67e22", label="30-day MA")
    axes[1].set_title("Aggregate daily demand 2013-2017")
    axes[1].set_ylabel("units"); axes[1].legend(fontsize=7)
    dow = sd.groupby("dow")["sales"].mean()
    axes[2].bar(["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"], dow.values, color="#48c9b0")
    axes[2].set_title("Bivariate - mean demand by weekday")
    axes[2].set_ylabel("units")
    save(fig, "f06_univariate_bivariate_demand")
    res["daily_demand_trend"] = {
        "first_90d_mean": round(float(daily.head(90).mean()), 2),
        "last_90d_mean": round(float(daily.tail(90).mean()), 2),
        "peak_date": str(daily.idxmax().date()), "peak_units": int(daily.max()),
    }
    return res


# ==========================================================================
# 8-11  multivariate, correlation, external-signal analysis
# ==========================================================================
def correlation_and_multivariate(dfs: dict[str, pd.DataFrame]) -> dict:
    print("\n[EDA 8-11] correlation, multivariate, external signals")
    dc = dfs["DataCo"]
    res = {}

    corr_cols = ["Late_delivery_risk", "Days for shipping (real)", "Days for shipment (scheduled)",
                 "Order Item Quantity", "Sales", "Order Item Discount Rate",
                 "Product Price", "Order Profit Per Order", "order_month", "order_dow"]
    corr = dc[corr_cols].apply(pd.to_numeric, errors="coerce").corr()
    corr.round(3).to_csv(TAB / "t04_correlation_matrix.csv")
    fig, ax = plt.subplots(figsize=(7.2, 6))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="RdBu_r", center=0,
                annot_kws={"size": 6.5}, cbar_kws={"shrink": .8}, ax=ax)
    ax.set_title("Correlation analysis - key business variables")
    save(fig, "f07_correlation_heatmap")
    res["risk_correlations"] = (corr["Late_delivery_risk"].drop("Late_delivery_risk")
                                .sort_values(key=abs, ascending=False).round(3).to_dict())

    # multivariate: delay-risk rate by shipping mode x market
    piv = (dc.pivot_table(index="Shipping Mode", columns="Market",
                          values="Late_delivery_risk", aggfunc="mean") * 100).round(1)
    piv.to_csv(TAB / "t05_delay_rate_shippingmode_market.csv")
    fig, ax = plt.subplots(figsize=(7, 3.4))
    sns.heatmap(piv, annot=True, fmt=".1f", cmap="YlOrRd", ax=ax, cbar_kws={"label": "% late"})
    ax.set_title("Multivariate - late-delivery rate (%) by shipping mode x market")
    save(fig, "f08_multivariate_delay_heatmap")

    # bivariate: realised vs scheduled shipping days
    ct = pd.crosstab(dc["Days for shipment (scheduled)"], dc["Days for shipping (real)"])
    ct.to_csv(TAB / "t06_scheduled_vs_real_shipping.csv")
    fig, ax = plt.subplots(figsize=(6, 4.2))
    sns.heatmap(np.log1p(ct), cmap="Blues", ax=ax, cbar_kws={"label": "log(1+orders)"})
    ax.set_xlabel("real shipping days"); ax.set_ylabel("scheduled shipping days")
    ax.set_title("Bivariate - scheduled vs realised shipping duration")
    save(fig, "f09_scheduled_vs_real_shipping")

    # external-signal analysis: monthly demand + holiday/promotion effect
    sd = dfs["StoreDemand"]
    monthly = sd.groupby(pd.Grouper(key="date", freq="MS"))["sales"].sum()
    try:
        stl = STL(monthly, period=12, robust=True).fit()
        fig, axes = plt.subplots(4, 1, figsize=(9, 6.4), sharex=True)
        axes[0].plot(monthly.index, monthly.values, color="#1a5276"); axes[0].set_ylabel("observed")
        axes[1].plot(monthly.index, stl.trend, color="#e67e22"); axes[1].set_ylabel("trend")
        axes[2].plot(monthly.index, stl.seasonal, color="#27ae60"); axes[2].set_ylabel("seasonal")
        axes[3].plot(monthly.index, stl.resid, color="#7f8c8d"); axes[3].set_ylabel("residual")
        fig.suptitle("STL seasonality decomposition - monthly aggregate demand (D2)", y=1.0, fontsize=10)
        save(fig, "f10_stl_decomposition")
        seas = stl.seasonal.groupby(stl.seasonal.index.month).mean()
        res["seasonality_index"] = {int(k): round(float(v), 1) for k, v in seas.items()}
        res["trend_growth_pct"] = round(float((stl.trend.iloc[-1] / stl.trend.iloc[0] - 1) * 100), 2)
        res["trend_strength"] = round(float(max(0, 1 - stl.resid.var() / (stl.trend + stl.resid).var())), 3)
        res["seasonal_strength"] = round(float(max(0, 1 - stl.resid.var() / (stl.seasonal + stl.resid).var())), 3)
    except Exception as exc:  # pragma: no cover
        print(f"    STL skipped: {exc}")

    # promotion / holiday proxy effect
    sd = sd.copy()
    sd["is_weekend"] = sd["dow"] >= 5
    hol = pd.to_datetime(["2013-12-25", "2014-12-25", "2015-12-25", "2016-12-25", "2017-12-25",
                          "2013-07-04", "2014-07-04", "2015-07-04", "2016-07-04", "2017-07-04",
                          "2013-11-28", "2014-11-27", "2015-11-26", "2016-11-24", "2017-11-23"])
    sd["is_holiday"] = sd["date"].isin(hol)
    eff = sd.groupby(["is_weekend", "is_holiday"])["sales"].mean().round(2)
    res["calendar_effects"] = {f"weekend={k[0]},holiday={k[1]}": float(v) for k, v in eff.items()}
    fig, ax = plt.subplots(figsize=(5.4, 3.1))
    eff.unstack().plot(kind="bar", ax=ax, color=["#5499c7", "#e59866"])
    ax.set_title("External signals - mean demand by weekend / holiday flag")
    ax.set_ylabel("units/day"); ax.set_xlabel("is_weekend")
    ax.legend(title="is_holiday", fontsize=7)
    save(fig, "f11_calendar_effects")
    return res


# ==========================================================================
# ABC analysis + RFM segmentation (D3)
# ==========================================================================
def abc_and_rfm(online: pd.DataFrame) -> dict:
    print("\n[EDA 12+] ABC classification and RFM segmentation")
    res = {}

    sku = online.groupby("StockCode").agg(units=("Quantity", "sum"), revenue=("revenue", "sum"))
    sku = sku.sort_values("revenue", ascending=False)
    sku["cum_share"] = sku["revenue"].cumsum() / sku["revenue"].sum()
    sku["ABC"] = np.where(sku["cum_share"] <= .80, "A", np.where(sku["cum_share"] <= .95, "B", "C"))
    sku.to_csv(TAB / "t07_abc_classification_skus.csv")
    abc = sku.groupby("ABC").agg(skus=("revenue", "size"), revenue=("revenue", "sum"))
    abc["revenue_share_pct"] = (abc["revenue"] / abc["revenue"].sum() * 100).round(2)
    abc["sku_share_pct"] = (abc["skus"] / abc["skus"].sum() * 100).round(2)
    abc.to_csv(TAB / "t08_abc_summary.csv")
    res["abc"] = abc.reset_index().to_dict("records")

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.4))
    reordered = sku.sort_values("revenue", ascending=False)
    axes[0].plot(np.arange(len(reordered)) / len(reordered) * 100,
                 reordered["cum_share"].values * 100, color="#1a5276")
    axes[0].axhline(80, ls="--", color="#c0392b", lw=1)
    axes[0].axvline(len(reordered[reordered.ABC == "A"]) / len(reordered) * 100,
                    ls=":", color="#c0392b", lw=1)
    axes[0].set_xlabel("% of SKUs (ranked by revenue)"); axes[0].set_ylabel("cumulative revenue %")
    axes[0].set_title("ABC / Pareto curve - Online Retail II")
    axes[1].bar(abc.index, abc["revenue_share_pct"], color=["#c0392b", "#e67e22", "#27ae60"])
    for i, (sk, rs) in enumerate(zip(abc["skus"], abc["revenue_share_pct"])):
        axes[1].text(i, rs, f"{rs:.1f}%\n({sk:,} SKUs)", ha="center", va="bottom", fontsize=7.5)
    axes[1].set_ylim(0, max(abc["revenue_share_pct"]) * 1.3)
    axes[1].set_ylabel("revenue share %"); axes[1].set_title("ABC revenue contribution")
    save(fig, "f12_abc_analysis")

    # RFM
    snap = online["date"].max() + pd.Timedelta(days=1)
    rfm = online.groupby("Customer ID").agg(
        recency=("date", lambda s: (snap - s.max()).days),
        frequency=("Invoice", "nunique"),
        monetary=("revenue", "sum"),
    ).dropna()
    rfm["R_q"] = pd.qcut(rfm["recency"], 4, labels=[4, 3, 2, 1]).astype(int)
    rfm["F_q"] = pd.qcut(rfm["frequency"].rank(method="first"), 4, labels=[1, 2, 3, 4]).astype(int)
    rfm["M_q"] = pd.qcut(rfm["monetary"], 4, labels=[1, 2, 3, 4]).astype(int)
    rfm["segment"] = np.where(rfm[["R_q", "F_q", "M_q"]].mean(axis=1) >= 3.0, "Champions",
                       np.where(rfm[["R_q", "F_q", "M_q"]].mean(axis=1) >= 2.0, "Loyal", "At risk"))
    seg = rfm.groupby("segment").agg(customers=("monetary", "size"),
                                     revenue=("monetary", "sum")).round(2)
    seg["revenue_share_pct"] = (seg["revenue"] / seg["revenue"].sum() * 100).round(2)
    seg.to_csv(TAB / "t09_rfm_segments.csv")
    res["rfm"] = seg.reset_index().to_dict("records")

    fig, ax = plt.subplots(figsize=(4.6, 3.2))
    ax.bar(seg.index, seg["customers"], color="#5499c7")
    for i, (c, r) in enumerate(zip(seg["customers"], seg["revenue_share_pct"])):
        ax.text(i, c, f"{c:,}\n{r:.0f}% rev", ha="center", va="bottom", fontsize=7.5)
    ax.set_ylim(0, seg["customers"].max() * 1.3)
    ax.set_ylabel("customers"); ax.set_title("RFM customer segments (D3)")
    save(fig, "f13_rfm_segments")
    return res


# ==========================================================================
def run() -> dict:
    print("=" * 78)
    print("MODULE M2 - EXPLORATORY DATA ANALYSIS")
    print("=" * 78)
    dc = pd.read_csv(config.PROCESSED_DIR / "dataco_clean.csv", low_memory=False)
    sd = pd.read_csv(config.PROCESSED_DIR / "store_demand_clean.csv.gz")
    orl = pd.read_csv(config.PROCESSED_DIR / "online_retail_clean.csv.gz")
    sd["date"] = pd.to_datetime(sd["date"])
    orl["date"] = pd.to_datetime(orl["date"])

    dfs = {"DataCo": dc, "StoreDemand": sd, "OnlineRetail": orl}
    out: dict = {}
    out.update(overview_and_quality(dfs))
    out.update(outliers_and_target(dfs))
    out.update(correlation_and_multivariate(dfs))
    out.update(abc_and_rfm(orl))

    with open(config.METRICS_DIR / "eda_summary.json", "w") as fh:
        json.dump(out, fh, indent=2, default=str)
    print("\n[M2] EDA complete -> results/figures, results/tables, results/metrics/eda_summary.json")
    return out


if __name__ == "__main__":
    run()
