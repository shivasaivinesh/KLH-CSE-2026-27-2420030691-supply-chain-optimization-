"""
Module M3 - Demand Forecasting Engine.

Trains and compares multi-horizon (7 / 30 / 90 day) demand forecasting models
on the Store-Item Demand dataset (D2):

    baseline 1 : Seasonal Naive (lag-7 and lag-364)
    baseline 2 : SARIMA / ARIMA  (classical statistical benchmark)
    ML models  : Random Forest, XGBoost, LightGBM
    proposed   : weighted ensemble of the ML models
    cold-start : calendar + item-statistics only model (no history required)

Evaluation is a genuine *recursive multi-step* rollout over the held-out
184-day window: for every forecast day the lags that fall inside the test
window are the model's own previous predictions, never the ground truth.
This avoids the optimistic bias of one-step-ahead evaluation.

Usage:
    python src/models_forecast.py            # full run
    python src/models_forecast.py --quick    # reduced settings (smoke test)
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config  # noqa: E402
from features import (FEATURE_COLS, FEATURE_COLS_CAL, FEATURE_COLS_LAG,  # noqa: E402
                      LAGS, ROLL_WINDOWS, fourier_terms, us_holiday_flags)

warnings.filterwarnings("ignore")

TEST_START = pd.Timestamp("2017-07-01")
TEST_END = pd.Timestamp("2017-12-31")
VALID_DAYS = 90
FIG, MET, TAB = config.FIGURES_DIR, config.METRICS_DIR, config.TABLES_DIR
RNG = np.random.default_rng(config.RANDOM_STATE)


# ==========================================================================
# metric helpers
# ==========================================================================
def mape(y, yhat, eps=1e-6):
    y, yhat = np.asarray(y, float), np.asarray(yhat, float)
    m = y > eps
    return float(np.mean(np.abs((y[m] - yhat[m]) / y[m])) * 100) if m.any() else np.nan


def smape(y, yhat, eps=1e-6):
    y, yhat = np.asarray(y, float), np.asarray(yhat, float)
    d = (np.abs(y) + np.abs(yhat)) / 2
    m = d > eps
    return float(np.mean(np.abs(y[m] - yhat[m]) / d[m]) * 100) if m.any() else np.nan


def evaluate(y, yhat) -> dict:
    y, yhat = np.asarray(y, float), np.asarray(yhat, float)
    err = y - yhat
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err ** 2)))
    denom = np.sum(np.abs(y))
    wape = float(np.sum(np.abs(err)) / denom * 100) if denom else np.nan
    var = float(np.var(y))
    r2 = float(1 - np.mean(err ** 2) / var) if var else np.nan
    bias = float(np.mean(err))
    return {"MAE": round(mae, 4), "RMSE": round(rmse, 4), "MAPE": round(mape(y, yhat), 3),
            "sMAPE": round(smape(y, yhat), 3), "WAPE": round(wape, 3),
            "R2": round(r2, 4), "bias": round(bias, 4), "n": int(y.size)}


# ==========================================================================
# feature builder for recursive rollout
# ==========================================================================
class RolloutFeatureBuilder:
    """
    Reproduces the training-time feature semantics from a date x SKU matrix,
    substituting model predictions for unknown (future) sales values.
    """

    def __init__(self, history: pd.DataFrame, all_dates: pd.DatetimeIndex, skus: list[str]):
        self.skus = skus
        self.sku_pos = {s: i for i, s in enumerate(skus)}
        self.dates = all_dates
        self.date_pos = {d: i for i, d in enumerate(all_dates)}
        self.mat = np.full((len(all_dates), len(skus)), np.nan)

        hist = history[history["date"].isin(all_dates)]
        r = hist["date"].map(self.date_pos).to_numpy()
        c = hist["sku"].map(self.sku_pos).to_numpy()
        self.mat[r, c] = hist["sales"].to_numpy(dtype=float)

        # static SKU descriptors learned on the training window
        self.item_mean_all = (history.groupby("item")["sales"].mean().to_dict())
        self.store_factor = (history.groupby("store")["sales"].mean()
                             / history["sales"].mean()).to_dict()
        self.sku_meta = (history.groupby("sku")[["store", "item"]].first()
                         .loc[skus].reset_index())

        idx = pd.DatetimeIndex(all_dates)
        cal = pd.concat([us_holiday_flags(idx), fourier_terms(idx)], axis=1)
        self.cal = cal

    def _col(self, win_start: int, win_end: int, reducer) -> np.ndarray:
        """reduce over rows [win_start, win_end) of the matrix."""
        block = self.mat[max(0, win_start):win_end]
        if block.shape[0] == 0:
            return np.full(len(self.skus), np.nan)
        with np.errstate(all="ignore"):
            out = reducer(block, axis=0)
        return out

    def features_for(self, di: int) -> pd.DataFrame:
        """Feature frame for every SKU at date index `di`."""
        if di <= 1:
            lag_1 = self.mat[0]
        else:
            lag_1 = self.mat[di - 1]
        f = pd.DataFrame({"lag_1": lag_1})
        for lag in LAGS[1:]:
            f[f"lag_{lag}"] = self.mat[di - lag] if di - lag >= 0 else np.nan
        # consecutive daily lags -> input sequence for the LSTM
        for i in range(1, 29):
            f[f"seq_{i}"] = self.mat[di - i] if di - i >= 0 else np.nan
        for w in ROLL_WINDOWS:
            mp = max(2, w // 2)
            block = self.mat[max(0, di - w):di]
            if block.shape[0] < mp:
                f[f"roll_mean_{w}"] = np.nan
                f[f"roll_std_{w}"] = np.nan
                f[f"roll_max_{w}"] = np.nan
                f[f"roll_min_{w}"] = np.nan
                continue
            with np.errstate(all="ignore"):
                f[f"roll_mean_{w}"] = np.nanmean(block, axis=0)
                f[f"roll_std_{w}"] = np.nanstd(block, axis=0, ddof=1)
                f[f"roll_max_{w}"] = np.nanmax(block, axis=0)
                f[f"roll_min_{w}"] = np.nanmin(block, axis=0)

        # exponential weighted mean over the last 28 rows
        span, alpha = 28, 2 / (28 + 1)
        start = max(0, di - 60)
        w_block = self.mat[start:di]
        if w_block.shape[0]:
            weights = (1 - alpha) ** np.arange(w_block.shape[0] - 1, -1, -1)
            num = np.nansum(w_block * weights[:, None], axis=0)
            den = np.nansum(~np.isnan(w_block) * weights[:, None], axis=0)
            with np.errstate(all="ignore"):
                f["ewm_28"] = num / den
        else:
            f["ewm_28"] = np.nan

        rm28 = f["roll_mean_28"].replace(0, np.nan)
        f["lag_ratio_7_28"] = f["lag_7"] / rm28
        f["rolling_trend"] = (f["roll_mean_7"] - rm28) / rm28
        f["cv_28"] = f["roll_std_28"] / rm28

        d = self.dates[di]
        meta = self.sku_meta
        f["sku"] = self.skus
        f["store"] = meta["store"].to_numpy()
        f["item"] = meta["item"].to_numpy()
        f["dow"] = d.dayofweek
        f["month"] = d.month
        f["day"] = d.day
        f["dayofyear"] = d.dayofyear
        f["weekofyear"] = int(pd.Timestamp(d).isocalendar().week)
        f["quarter"] = d.quarter
        f["is_leap"] = int(pd.Timestamp(d).is_leap_year)
        for col in self.cal.columns:
            f[col] = self.cal.loc[d, col]
        f["item_mean_all"] = meta["item"].map(self.item_mean_all).to_numpy()
        f["store_factor"] = meta["store"].map(self.store_factor).to_numpy()
        return f.replace([np.inf, -np.inf], np.nan)


# ==========================================================================
# rollouts
# ==========================================================================
def rollout_single_model(sd: pd.DataFrame, all_dates: pd.DatetimeIndex, skus: list[str],
                         model, start_idx: int, n_days: int, feature_cols: list[str],
                         hist_cut: pd.Timestamp) -> tuple[np.ndarray, pd.DatetimeIndex]:
    """
    Recursive multi-step rollout for one model.

    The model feeds its **own** predictions back as the pseudo-history, so no
    ground-truth value from inside the test window ever becomes an input.
    Predictions are batched across all SKUs at each time step for speed.
    """
    hist = sd[sd["date"] < hist_cut]
    b = RolloutFeatureBuilder(hist, all_dates, skus)
    preds = []
    for step in range(n_days):
        di = start_idx + step
        if di >= len(b.dates):
            break
        X = b.features_for(di)[feature_cols].astype(float)
        p = np.clip(np.asarray(model.predict(X), dtype=float), 0, None)
        preds.append(p)
        b.mat[di] = p                      # own prediction becomes the history
    return np.vstack(preds), b.dates[start_idx:start_idx + len(preds)]


# ==========================================================================
# models
# ==========================================================================
def make_models(quick: bool = False) -> dict:
    from sklearn.ensemble import RandomForestRegressor
    import lightgbm as lgb
    import xgboost as xgb

    n_rf = 45 if quick else 110
    return {
        # RF is trained on a row subsample: at ~640k rows a full bootstrap forest
        # costs several GB of RAM, which this evaluation host does not have.
        "RandomForest": RandomForestRegressor(
            n_estimators=n_rf, max_depth=18, min_samples_leaf=4, max_features=0.4,
            max_samples=0.22, n_jobs=config.N_JOBS, random_state=config.RANDOM_STATE),
        "XGBoost": xgb.XGBRegressor(
            n_estimators=300 if quick else 700, learning_rate=0.06, max_depth=8,
            subsample=0.85, colsample_bytree=0.8, min_child_weight=4,
            reg_lambda=1.0, tree_method="hist", n_jobs=config.N_JOBS,
            random_state=config.RANDOM_STATE, verbosity=0),
        "LightGBM": lgb.LGBMRegressor(
            n_estimators=400 if quick else 900, learning_rate=0.05, num_leaves=63,
            min_child_samples=20, subsample=0.85, subsample_freq=1, colsample_bytree=0.8,
            reg_lambda=1.0, n_jobs=config.N_JOBS, random_state=config.RANDOM_STATE,
            verbose=-1),
    }


def cold_start_model(quick=False):
    """Calendar + item-statistics only: needs no SKU history (G6 / novelty item)."""
    import lightgbm as lgb
    return lgb.LGBMRegressor(
        n_estimators=200 if quick else 450, learning_rate=0.07, num_leaves=63,
        n_jobs=config.N_JOBS, random_state=config.RANDOM_STATE, verbose=-1)


# ==========================================================================
# ARIMA baseline
# ==========================================================================
def arima_baseline(sd: pd.DataFrame, builder: RolloutFeatureBuilder, start_idx: int,
                   n_days: int, n_series: int) -> tuple[pd.DataFrame, list[str]]:
    from statsmodels.tsa.statespace.sarimax import SARIMAX

    # representative series: spread across the (store, item) grid, stratified by volume
    vol = sd.groupby("sku")["sales"].mean().sort_values()
    qs = np.linspace(0.05, 0.95, n_series)
    chosen = [vol.index[int(q * (len(vol) - 1))] for q in qs]
    chosen = list(dict.fromkeys(chosen))

    hist_end = start_idx
    frames = []
    print(f"    ARIMA/SARIMA on {len(chosen)} representative series (matched subset) ...")
    for i, sku in enumerate(chosen, 1):
        s = sd[sd["sku"] == sku].sort_values("date")
        train = s[s["date"] < builder.dates[hist_end]].set_index("date")["sales"].asfreq("D")
        try:
            mdl = SARIMAX(train, order=(1, 0, 1), seasonal_order=(1, 0, 1, 7),
                          enforce_stationarity=False, enforce_invertibility=False)
            res = mdl.fit(disp=False, maxiter=60)
            fc = np.clip(res.forecast(steps=n_days).to_numpy(dtype=float), 0, None)
        except Exception as exc:
            print(f"      {sku}: SARIMA failed ({exc}); using seasonal naive")
            fc = np.resize(train.to_numpy()[-7:], n_days).astype(float)
        frames.append(pd.DataFrame({
            "date": builder.dates[hist_end:hist_end + len(fc)],
            "sku": sku, "SARIMA": fc}))
        if i % 4 == 0:
            print(f"      {i}/{len(chosen)} series fitted")
    return pd.concat(frames, ignore_index=True), chosen


# ==========================================================================
def seasonal_naive(sd: pd.DataFrame, builder: RolloutFeatureBuilder, start_idx: int,
                   n_days: int) -> pd.DataFrame:
    """lag-7 repeating baseline for every SKU."""
    skus = builder.skus
    last7 = builder.mat[start_idx - 7:start_idx]           # (7, n_sku)
    reps = int(np.ceil(n_days / 7))
    vals = np.tile(last7, (reps, 1))[:n_days]
    return pd.DataFrame({
        "date": np.repeat(builder.dates[start_idx:start_idx + n_days], len(skus)),
        "sku": np.tile(np.array(skus), n_days),
        "SeasonalNaive": vals.reshape(-1),
    })


# ==========================================================================
def weight_search(val_true: np.ndarray, val_preds: dict[str, np.ndarray]) -> dict:
    """Optimise ensemble weights on the validation window (minimise MAPE)."""
    from scipy.optimize import minimize

    names = list(val_preds)
    P = np.vstack([val_preds[n] for n in names])
    y = val_true

    def loss(w):
        w = np.clip(w, 0, None)
        tot = w.sum()
        if tot <= 0:
            return 1e6
        return mape(y, (w / tot) @ P)

    x0 = np.full(len(names), 1 / len(names))
    best, best_score = x0, loss(x0)
    rng = np.random.default_rng(config.RANDOM_STATE)
    for _ in range(60):
        w0 = rng.dirichlet(np.ones(len(names)))
        res = minimize(loss, w0, method="Nelder-Mead",
                       options={"maxiter": 800, "xatol": 1e-3, "fatol": 1e-3})
        if res.fun < best_score:
            best_score, best = float(res.fun), res.x
    w = np.clip(best, 0, None)
    w = w / w.sum()
    return {"weights": {k: round(float(v), 4) for k, v in zip(names, w)},
            "val_MAPE": round(float(best_score), 3),
            "val_MAE": round(float(np.mean(np.abs(y - (w @ P)))), 4)}


def figures(results_long: pd.DataFrame, test_true: pd.DataFrame, horizon_table: pd.DataFrame) -> None:
    # 1. bar chart of MAPE by model and horizon
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.5))
    for ax, h in zip(axes, config.FORECAST_HORIZONS):
        sub = horizon_table[horizon_table["horizon"] == h].sort_values("MAPE")
        bars = ax.barh(sub["model"], sub["MAPE"], color="#2874a6")
        for b, v in zip(bars, sub["MAPE"]):
            ax.text(v, b.get_y() + b.get_height() / 2, f" {v:.1f}%", va="center", fontsize=7.5)
        ax.set_title(f"{h}-day horizon")
        ax.set_xlabel("MAPE %")
        ax.invert_yaxis()
    fig.suptitle("Demand-forecast accuracy by model and horizon (recursive rollout)", y=1.04)
    fig.savefig(FIG / "f14_model_comparison_mape.png", bbox_inches="tight")
    plt.close(fig)
    print("    figure -> f14_model_comparison_mape.png")

    # 2. forecast vs actual for one representative SKU
    sku = test_true["sku"].value_counts().index[0]
    t = test_true[test_true["sku"] == sku].sort_values("date")
    fig, ax = plt.subplots(figsize=(11, 3.6))
    ax.plot(t["date"], t["sales"], label="actual", color="#17202a", lw=1.4)
    for col, c in [("LightGBM", "#e67e22"), ("XGBoost", "#27ae60"), ("Ensemble", "#c0392b")]:
        if col in t:
            ax.plot(t["date"], t[col], label=col, lw=1.1, alpha=.85)
    if "SARIMA" in t:
        ax.plot(t["date"], t["SARIMA"], label="SARIMA", lw=1.1, ls="--", color="#8e44ad")
    ax.axvline(TEST_START, color="grey", ls=":", lw=1)
    ax.set_title(f"Recursive 184-day forecast vs actual - {sku} (held-out window)")
    ax.set_ylabel("units/day"); ax.legend(fontsize=7.5, ncol=5)
    fig.savefig(FIG / "f15_forecast_vs_actual.png", bbox_inches="tight")
    plt.close(fig)
    print("    figure -> f15_forecast_vs_actual.png")

    # 3. residual diagnostics of the ensemble
    if "Ensemble" in test_true:
        resid = test_true["sales"] - test_true["Ensemble"]
        fig, axes = plt.subplots(1, 3, figsize=(12, 3.2))
        axes[0].hist(resid, bins=50, color="#5dade2", edgecolor="white")
        axes[0].axvline(0, color="black", lw=1); axes[0].set_title("Ensemble residual distribution")
        axes[1].scatter(test_true["Ensemble"], resid, s=2, alpha=.15, color="#2874a6")
        axes[1].axhline(0, color="black", lw=1); axes[1].set_xlabel("predicted"); axes[1].set_ylabel("residual")
        axes[1].set_title("Residuals vs fitted")
        daily_bias = test_true.groupby("date").apply(lambda g: (g["sales"] - g["Ensemble"]).mean())
        axes[2].plot(daily_bias.index, daily_bias.values, lw=1, color="#c0392b")
        axes[2].axhline(0, color="black", lw=1); axes[2].set_title("Daily mean bias")
        fig.suptitle("Forecast error diagnostics (30-day window shown in report)", y=1.04)
        fig.savefig(FIG / "f16_residual_diagnostics.png", bbox_inches="tight")
        plt.close(fig)
        print("    figure -> f16_residual_diagnostics.png")


# ==========================================================================
def run(quick: bool = False, with_lstm: bool = True) -> dict:
    t0 = time.time()
    print("=" * 78)
    print("MODULE M3 - DEMAND FORECASTING ENGINE" + ("  [quick]" if quick else ""))
    print("=" * 78)

    from models_dl import DL_FEATURE_COLS, LSTMForecaster

    need = sorted(set(FEATURE_COLS + DL_FEATURE_COLS + ["date", "store", "item", "sku", "sales"]))
    ft = pd.read_csv(config.PROCESSED_DIR / "store_demand_features.csv.gz", usecols=need)
    ft["date"] = pd.to_datetime(ft["date"])
    num_cols = [c for c in ft.columns if c not in ("date", "sku", "store", "item")]
    ft[num_cols] = ft[num_cols].astype(np.float32, copy=False)   # halve the RAM footprint
    sd = ft[["date", "store", "item", "sku", "sales"]].copy()

    train = ft[ft["date"] < TEST_START].dropna(subset=FEATURE_COLS_LAG)
    print(f"    training rows : {len(train):,}  "
          f"(period {train['date'].min().date()} -> {train['date'].max().date()})")
    print(f"    test rows     : {(ft['date'] >= TEST_START).sum():,}  "
          f"(period {TEST_START.date()} -> {TEST_END.date()})")

    X_train = train[FEATURE_COLS].astype(np.float32)
    y_train = train["sales"].astype(np.float32)

    # ---------------- fit models ----------------
    models = make_models(quick)
    fitted: dict = {}
    for name, mdl in models.items():
        t = time.time()
        if name == "RandomForest":
            # fit the forest on a random row subsample (memory-bounded, statistically
            # equivalent at this scale); the boosting models use every row.
            rng = np.random.default_rng(config.RANDOM_STATE)
            n_sub = min(260_000, len(X_train))
            pos = rng.choice(len(X_train), size=n_sub, replace=False)
            mdl.fit(X_train.iloc[pos], y_train.iloc[pos])
            del pos
        else:
            mdl.fit(X_train, y_train)
        fitted[name] = mdl
        print(f"    fitted {name:<13} in {time.time() - t:6.1f}s", flush=True)

    feature_map = {name: FEATURE_COLS for name in fitted}
    if with_lstm:
        t = time.time()
        tr_dl = ft[ft["date"] < TEST_START].dropna(subset=DL_FEATURE_COLS)
        lf = LSTMForecaster(n_train=60_000 if quick else 140_000,
                            epochs=3 if quick else 6)
        lf.fit(tr_dl[DL_FEATURE_COLS], tr_dl["sales"])
        fitted["LSTM"] = lf
        feature_map["LSTM"] = DL_FEATURE_COLS
        print(f"    fitted LSTM          in {time.time() - t:6.1f}s")

    cs = cold_start_model(quick)
    cs_feats = FEATURE_COLS_CAL
    cs_idx = train[cs_feats].notna().all(axis=1)
    cs.fit(train.loc[cs_idx, cs_feats].astype(float), y_train[cs_idx])
    print("    fitted ColdStart     (calendar + item-statistics only)")

    all_dates = pd.DatetimeIndex(sorted(ft["date"].unique()))
    skus = sorted(ft["sku"].unique())
    test_start_idx = int(np.where(all_dates == TEST_START)[0][0])
    n_test = int((all_dates >= TEST_START).sum())
    val_start = test_start_idx - VALID_DAYS

    # ---------------- validation rollouts (per-model feedback) ----------------
    print("    validation rollouts (90 days, each model feeds back its own predictions) ...")
    val_preds = {}
    for name, mdl in fitted.items():
        t = time.time()
        P, vdates = rollout_single_model(sd, all_dates, skus, mdl, val_start, VALID_DAYS,
                                         feature_map[name], all_dates[val_start])
        val_preds[name] = P.reshape(-1)
        print(f"      {name:<13} {time.time() - t:5.1f}s")
    vtrue = (ft[(ft["date"] >= all_dates[val_start]) & (ft["date"] < TEST_START)]
             .sort_values(["date", "sku"])["sales"].to_numpy(float))
    ens = weight_search(vtrue, val_preds)
    print(f"    ensemble weights {ens['weights']}  (validation MAPE {ens['val_MAPE']}%)")

    # persist the validation-window errors: the inventory module derives the
    # per-SKU forecast uncertainty sigma_hat from THIS file, never from the test
    # window, so the replenishment policy cannot peek at the future.
    val_frame = pd.DataFrame({"date": np.repeat(vdates, len(skus)),
                              "sku": np.tile(np.array(skus), len(vdates)),
                              "sales": vtrue})
    for name, arr in val_preds.items():
        val_frame[name] = arr
    val_frame["Ensemble"] = sum(ens["weights"][k] * val_frame[k] for k in ens["weights"])
    val_frame.to_csv(config.PROCESSED_DIR / "forecast_validation_predictions.csv.gz",
                     index=False, compression="gzip")
    print(f"    validation predictions saved ({len(val_frame):,} rows) -> "
          f"forecast_validation_predictions.csv.gz")

    # ---------------- test rollouts ----------------
    print("    test rollouts (184 days, per-model feedback) ...")
    test_preds, dates_ref = {}, None
    for name, mdl in fitted.items():
        t = time.time()
        P, tdates = rollout_single_model(sd, all_dates, skus, mdl, test_start_idx, n_test,
                                         feature_map[name], TEST_START)
        test_preds[name] = P
        dates_ref = tdates
        print(f"      {name:<13} {time.time() - t:5.1f}s")

    hbuilder = RolloutFeatureBuilder(sd[sd["date"] < TEST_START], all_dates, skus)
    P, _ = rollout_single_model(sd, all_dates, skus, cs, test_start_idx, n_test,
                                cs_feats, TEST_START)
    test_preds["ColdStart"] = P
    print("      ColdStart     done")

    res = pd.DataFrame({"date": np.repeat(dates_ref, len(skus)),
                        "sku": np.tile(np.array(skus), len(dates_ref))})
    for name, P in test_preds.items():
        res[name] = P.reshape(-1)
    w = ens["weights"]
    res["Ensemble"] = sum(w[k] * res[k] for k in w)

    naive = seasonal_naive(sd, hbuilder, test_start_idx, n_test)
    res = res.merge(naive, on=["date", "sku"], how="left")

    actual = sd[sd["date"] >= TEST_START][["date", "sku", "sales"]]
    res = res.merge(actual, on=["date", "sku"], how="inner")
    res.to_csv(config.PROCESSED_DIR / "forecast_test_predictions.csv.gz",
               index=False, compression="gzip")

    # ---------------- ARIMA on a matched subset ----------------
    arima, chosen = arima_baseline(sd, hbuilder, test_start_idx, n_test,
                                   config.N_ARIMA_SERIES if not quick else 4)
    arima_long = arima.merge(actual[actual["sku"].isin(chosen)], on=["date", "sku"], how="inner")
    res_arima = res[res["sku"].isin(chosen)]

    # ---------------- metrics ----------------
    # SARIMA lives in its own (matched-subset) frame, so it is not a column of `res`
    model_cols = [m for m in ["SARIMA", "SeasonalNaive", "ColdStart", "RandomForest",
                              "XGBoost", "LightGBM", "LSTM", "Ensemble"]
                  if m in res.columns or m == "SARIMA"]
    rows, hrows = [], []
    for h in list(config.FORECAST_HORIZONS) + [n_test]:
        cutoff = TEST_START + pd.Timedelta(days=h - 1)
        for model in model_cols:
            sub = (arima_long[arima_long["date"] <= cutoff] if model == "SARIMA"
                   else res[res["date"] <= cutoff])
            sub = sub.dropna(subset=[model])
            if sub.empty:
                continue
            met = evaluate(sub["sales"], sub[model])
            met.update({"horizon": h, "model": model,
                        "scope": (f"{len(chosen)} series (SARIMA matched subset)"
                                  if model == "SARIMA" else "500 series (full grid)")})
            rows.append(met)
            if h in config.FORECAST_HORIZONS:
                hrows.append({"horizon": h, "model": model, "MAPE": met["MAPE"],
                              "MAE": met["MAE"], "RMSE": met["RMSE"], "WAPE": met["WAPE"]})
    metrics = pd.DataFrame(rows)
    metrics.to_csv(MET / "forecast_metrics_by_horizon.csv", index=False)

    # strictly matched comparison: identical SKUs *and* dates for every model
    max_d = arima_long["date"].max()
    matched_rows = []
    for model in model_cols:
        sub = (arima_long.dropna(subset=[model]) if model == "SARIMA"
               else res_arima[res_arima["date"] <= max_d].dropna(subset=[model]))
        if sub.empty:
            continue
        matched_rows.append({"model": model, **evaluate(sub["sales"], sub[model])})
    matched_df = pd.DataFrame(matched_rows).sort_values("MAPE")
    matched_df.to_csv(TAB / "t11_model_comparison_matched_subset.csv", index=False)

    horizon_table = pd.DataFrame(hrows)
    horizon_table.to_csv(TAB / "t12_mape_by_horizon.csv", index=False)

    full = metrics[(metrics["horizon"] == n_test) & (metrics["scope"].str.startswith("500"))]
    full = full.set_index("model")
    base_map = matched_df.set_index("model")["MAPE"].to_dict()
    imp = {}
    if "Ensemble" in base_map and "SARIMA" in base_map:
        imp["ensemble_vs_SARIMA_MAPE_reduction_pct"] = round(
            (base_map["SARIMA"] - base_map["Ensemble"]) / base_map["SARIMA"] * 100, 2)
        best_ml = min(v for k, v in base_map.items() if k != "SARIMA")
        imp["best_ml_vs_SARIMA_MAPE_reduction_pct"] = round(
            (base_map["SARIMA"] - best_ml) / base_map["SARIMA"] * 100, 2)
    if "SeasonalNaive" in base_map and "Ensemble" in base_map:
        imp["ensemble_vs_SeasonalNaive_MAPE_reduction_pct"] = round(
            (base_map["SeasonalNaive"] - base_map["Ensemble"]) / base_map["SeasonalNaive"] * 100, 2)
    for m in ["RandomForest", "XGBoost", "LightGBM", "LSTM"]:
        if m in base_map and "Ensemble" in base_map:
            imp[f"ensemble_vs_{m}_MAPE_reduction_pct"] = round(
                (base_map[m] - base_map["Ensemble"]) / base_map[m] * 100, 2)

    print("    ablation study (calendar-only vs full feature set, same test window) ...")
    ablation = ablation_calendar_only(quick)
    ablation["full_feature_MAPE"] = (float(full.loc["XGBoost", "MAPE"])
                                     if "XGBoost" in full.index else None)
    if ablation["full_feature_MAPE"]:
        ablation["lag_feature_lift_pct"] = round(
            (ablation["calendar_only_MAPE"] - ablation["full_feature_MAPE"])
            / ablation["calendar_only_MAPE"] * 100, 2)

    report = {
        "protocol": {
            "train_period": f"{train['date'].min().date()} -> {train['date'].max().date()}",
            "validation_period": f"{all_dates[val_start].date()} -> "
                                 f"{(TEST_START - pd.Timedelta(days=1)).date()}",
            "test_period": f"{TEST_START.date()} -> {TEST_END.date()}",
            "test_days": int(n_test), "series": len(skus),
            "evaluation": ("recursive multi-step rollout; every model feeds back its OWN "
                           "predictions as the pseudo-history"),
            "arima_subset_series": len(chosen),
        },
        "feature_counts": {"total": len(FEATURE_COLS),
                           "calendar_external": len(FEATURE_COLS_CAL),
                           "lag_rolling_trend": len(FEATURE_COLS_LAG),
                           "lstm_inputs": len(DL_FEATURE_COLS)},
        "ensemble": ens,
        "full_grid_metrics_by_horizon":
            metrics[metrics["scope"].str.startswith("500")].to_dict("records"),
        "matched_subset_comparison": matched_df.to_dict("records"),
        "improvements": imp,
        "ablation": ablation,
        "runtime_sec": round(time.time() - t0, 1),
    }
    Path(MET / "forecast_metrics.json").write_text(json.dumps(report, indent=2, default=str))

    print("\n    --- full-grid accuracy (500 series, MAPE %) ---")
    print(metrics[metrics["scope"].str.startswith("500")]
          .pivot(index="model", columns="horizon", values="MAPE").round(2).to_string())
    print("\n    --- matched-subset comparison (identical SKUs and dates for every model) ---")
    print(matched_df.to_string(index=False))
    print(f"\n    improvements: {imp}")
    print(f"    ablation    : {ablation}")

    figures(res, res, horizon_table)
    print(f"\n[M3] forecasting complete in {report['runtime_sec']}s")
    return report
def ablation_calendar_only(quick: bool = False) -> dict:
    """
    Ablation study: how much do the lag/rolling features actually buy?

    A model with the *identical* configuration and the *identical* target is
    trained on the calendar/external block only and scored on the **same
    held-out test window** used for the main comparison, so the two numbers are
    directly comparable. Calendar features need no lag feedback, therefore the
    one-shot evaluation is valid for this feature set.
    """
    import xgboost as xgb

    ft = pd.read_csv(config.PROCESSED_DIR / "store_demand_features.csv.gz",
                     usecols=FEATURE_COLS_CAL + ["date", "sales"])
    ft["date"] = pd.to_datetime(ft["date"])
    tr = ft[ft["date"] < TEST_START].dropna(subset=FEATURE_COLS_CAL)
    te = ft[ft["date"] >= TEST_START].dropna(subset=FEATURE_COLS_CAL)

    mdl = xgb.XGBRegressor(n_estimators=150 if quick else 400, learning_rate=0.07,
                           max_depth=8, tree_method="hist", n_jobs=config.N_JOBS,
                           random_state=config.RANDOM_STATE, verbosity=0)
    mdl.fit(tr[FEATURE_COLS_CAL].astype(np.float32), tr["sales"].astype(np.float32))
    y = te["sales"].to_numpy(float)
    p = np.clip(mdl.predict(te[FEATURE_COLS_CAL].astype(np.float32)), 0, None)
    return {"calendar_only_MAPE": round(mape(y, p), 3),
            "calendar_only_MAE": round(float(np.mean(np.abs(y - p))), 4),
            "calendar_only_RMSE": round(float(np.sqrt(np.mean((y - p) ** 2))), 4),
            "evaluated_on": f"test window {TEST_START.date()} -> {TEST_END.date()}",
            "n_observations": int(len(te))}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args()
    run(quick=a.quick)
